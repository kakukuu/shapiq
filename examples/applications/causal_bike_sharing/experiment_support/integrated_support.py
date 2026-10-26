"""Verified data and numbered output helpers for the five integrated notebooks."""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import subprocess
import sys
from contextlib import contextmanager
from datetime import datetime
from itertools import combinations
from pathlib import Path

import data_utils as du
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from IPython.display import HTML, Image, Markdown, display
from local_analysis import plot_case_comparison
from sample_readings import add_sample_readings, build_sample_readings

ROOT = Path(__file__).resolve().parents[1]
RESULTS = "results"
INTEGRATED = "exports"
FEATURES = du.FEATURE_NAMES
PAIRS = list(combinations(range(7), 2))
FAMILIES = {
    "a": ["a"],
    "b1": ["b1"],
    "b2": ["b2", "temp_root", "hum_root", "windspeed_root", "b2_unconfounded"],
    "b3": ["b3", "b3_unconfounded"],
    "lag": ["lag"],
}
STRUCTURAL = [
    "b1",
    "b2",
    "b3",
    "temp_root",
    "hum_root",
    "windspeed_root",
    "b3_unconfounded",
    "b2_unconfounded",
]


def manifest(root=ROOT):
    p = Path(root) / RESULTS / "manifest.json"
    m = json.loads(p.read_text(encoding="utf-8"))
    if m.get("completed") is not True:
        raise RuntimeError(
            "Results are incomplete. Run this notebook's setup cell in RUN_MODE='auto'."
        )
    if m.get("postprocessing_completed") is not True:
        raise RuntimeError("Result publication is incomplete. Rerun this notebook's setup cell.")
    from run_full_holdout import PROTOCOL_VERSION, dependency_versions, source_hash

    config = m.get("config", {})
    if config.get("protocol_version") != PROTOCOL_VERSION or config.get(
        "source_sha256"
    ) != source_hash(root):
        raise RuntimeError(
            "Results were computed with different code or data. Run the notebook's setup cell again."
        )
    if config.get("versions") != dependency_versions():
        raise RuntimeError(
            "Results use different dependency versions. Run the notebook's setup cell again."
        )
    return m


def verified_path(relative, root=ROOT):
    m = manifest(root)
    hashes = {k.replace("\\", "/"): v for k, v in m["artifacts_sha256"].items()}
    relative = str(relative).replace("\\", "/")
    p = Path(root) / RESULTS / relative
    if relative not in hashes or hashlib.sha256(p.read_bytes()).hexdigest() != hashes[relative]:
        raise ValueError(f"Saved artifact checksum mismatch: {relative}")
    return p


def csv(relative, root=ROOT):
    return pd.read_csv(verified_path(relative, root))


def load_experiment(arm, root=ROOT):
    m = manifest(root)
    with np.load(verified_path(f"draws/{arm}.npz", root), allow_pickle=False) as z:
        result = {k: z[k].copy() for k in z.files}
    expected = 146 if arm == "lag" else 147
    assert result["base_phi"].shape == result["variant_phi"].shape == (expected, 7)
    assert result["base_interaction"].shape == result["variant_interaction"].shape == (expected, 21)
    result.update(arm=arm, n=expected, protocol=m["config"])
    for key in (
        "points",
        "item_summary",
        "selected",
        "selection_rules",
        "validation",
        "validation_details",
        "validation_summary",
    ):
        result[key] = csv(f"tables/{arm}_{key}.csv", root)
    np.testing.assert_array_equal(result["row_ids"], result["points"]["row_id"])
    if not result["points"]["date"].is_unique:
        raise ValueError("Duplicate target dates")
    if arm == "lag":
        dates = pd.to_datetime(result["points"]["date"])
        previous = pd.to_datetime(result["points"]["lag_source_date"])
        if not ((dates - previous).dt.days == 1).all():
            raise ValueError("Lag date mismatch")
    else:
        np.testing.assert_array_equal(
            result["points"]["prediction_base"], result["points"]["prediction_variant"]
        )
    selection = Path(root) / RESULTS / "selection_manifest.json"
    if hashlib.sha256(selection.read_bytes()).hexdigest() != m["selection_sha256"]:
        raise ValueError("Frozen selection manifest changed")
    frozen = json.loads(selection.read_text(encoding="utf-8"))["selection"][arm]
    expected_cases = [(int(s["row_id"]), s["case_rule"]) for s in frozen]
    actual_cases = list(
        zip(result["selected"]["row_id"].astype(int), result["selected"]["case_rule"], strict=True)
    )
    if expected_cases != actual_cases:
        raise ValueError("Frozen case selection mismatch")
    result["rank"] = json.loads(
        verified_path("rank_calibration.json", root).read_text(encoding="utf-8")
    )[arm]
    result["models"] = json.loads(verified_path("models.json", root).read_text(encoding="utf-8"))
    return result


def ensure_results(root=ROOT, mode="auto", causal_sample_size=500, marginal_sample_size="all"):
    """Notebook entry point: validate the complete protocol or compute it in a fresh process.

    auto reuses matching results, recompute regenerates summaries using verified
    numeric caches, and load refuses missing/stale data without running anything.
    """
    from run_full_holdout import (
        BUDGET,
        DISCOVERY_SEED,
        REFERENCE_SEEDS,
        VALIDATION_SEEDS,
        VARIANTS,
        sampling_config,
    )

    if mode not in ("auto", "recompute", "load"):
        raise ValueError("RUN_MODE must be 'auto', 'recompute', or 'load'.")
    requested = sampling_config(causal_sample_size, marginal_sample_size)
    root = Path(root).resolve()
    reason = None
    try:
        m = manifest(root)
        config = m["config"]
        expected = {
            "sampling": requested,
            "budget": BUDGET,
            "discovery_seed": DISCOVERY_SEED,
            "reference_seeds": list(REFERENCE_SEEDS),
            "validation_seeds": list(VALIDATION_SEEDS),
            "variants": VARIANTS,
        }
        if any(config.get(k) != v for k, v in expected.items()):
            raise RuntimeError(
                "Saved sampling settings or experiment protocol differ from the notebook configuration."
            )
        hashes = m["artifacts_sha256"]
        required = {
            "models.json",
            "rank_calibration.json",
            "protocol.json",
            "selection_manifest.json",
        }
        if not required.issubset(hashes):
            raise ValueError("Missing provenance checksums")
        for relative, digest in hashes.items():
            path = root / RESULTS / relative.replace("\\", "/")
            if not path.is_file() or hashlib.sha256(path.read_bytes()).hexdigest() != digest:
                raise ValueError(f"Missing or modified result: {relative}")
    except (FileNotFoundError, RuntimeError, ValueError, KeyError) as error:
        reason = str(error)
    if mode == "load" and reason is not None:
        raise RuntimeError(reason + " Set RUN_MODE='auto' to compute the current protocol.")
    if reason is not None or mode == "recompute":
        print(
            "Computing all experiments and independent-seed validation. "
            + (reason or "Recompute requested."),
            flush=True,
        )
        recompute_results(root, causal_sample_size, marginal_sample_size)
    else:
        print(
            "Reusing verified results: matching code, data, sampling settings and artifact checksums."
        )
    return manifest(root)


@contextmanager
def calculation_lock(root):
    """Do not let two notebook kernels publish into the same result directory."""
    import psutil

    path = Path(root) / ".experiment_run.lock"
    for attempt in range(2):
        try:
            fd = os.open(path, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
            break
        except FileExistsError as error:
            try:
                owner = json.loads(path.read_text(encoding="utf-8"))
                live = (
                    psutil.pid_exists(owner["pid"])
                    and abs(psutil.Process(owner["pid"]).create_time() - owner["started"]) < 1
                )
            except (ValueError, KeyError, psutil.Error):
                live = True
            if live or attempt:
                raise RuntimeError(
                    "Another notebook is computing these experiments. Wait for it to finish, then Run All here."
                ) from error
            path.unlink()
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            json.dump({"pid": os.getpid(), "started": psutil.Process().create_time()}, handle)
        yield
    finally:
        path.unlink(missing_ok=True)


def recompute_results(root=ROOT, causal_sample_size=500, marginal_sample_size="all"):
    with calculation_lock(Path(root).resolve()):
        _recompute_results(root, causal_sample_size, marginal_sample_size)


def _recompute_results(root=ROOT, causal_sample_size=500, marginal_sample_size="all"):
    root = Path(root).resolve()
    saved = root / RESULTS / "manifest.json"
    if saved.exists():
        target = root / "archive" / ("results_" + datetime.now().strftime("%Y%m%d_%H%M%S_%f"))
        target.resolve().relative_to(root)
        source = (root / RESULTS).resolve()
        source.relative_to(root)
        target.parent.mkdir(parents=True, exist_ok=True)
        # Also archive incomplete publications: parameters may have changed
        # after a failed run. Numeric caches still allow safe resumption.
        target.mkdir()
        for entry in source.iterdir():
            if entry.name == "cache":
                continue
            entry.resolve().relative_to(root)
            # Case-selection revisions share the same numerical estimator.
            # Preserve their draws even when the parent reports are rebuilt.
            if entry.name == "targeted_cases" and (entry / "cache").is_dir():
                shared = source / "cache"
                shared.mkdir(exist_ok=True)
                for draw in (entry / "cache").glob("*.npz"):
                    draw.resolve().relative_to(root)
                    destination = shared / draw.name
                    if not destination.exists():
                        shutil.move(str(draw), str(destination))
            shutil.move(str(entry), str(target / entry.name))
        exported = (root / INTEGRATED).resolve()
        exported.relative_to(root)
        if exported.is_dir():
            shutil.move(str(exported), str(target / INTEGRATED))
    env = os.environ.copy()
    env.update(PYTHONUTF8="1", PYTHONIOENCODING="utf-8", MPLBACKEND="Agg")
    command = [
        sys.executable,
        "-u",
        str(root / "experiment_support/run_full_holdout.py"),
        "--causal-samples",
        str(causal_sample_size),
        "--marginal-samples",
        str(marginal_sample_size),
    ]
    with subprocess.Popen(
        command,
        cwd=root,
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        encoding="utf-8",
        errors="replace",
    ) as process:
        for line in process.stdout:
            print(line, end="", flush=True)
        if process.wait():
            raise RuntimeError(
                "Experiment calculation failed; the output above shows the failing step. Rerun to resume verified caches."
            )


def table(frame, number, title, root=ROOT, expanded=True):
    directory = Path(root) / INTEGRATED / "tables"
    directory.mkdir(parents=True, exist_ok=True)
    frame.to_csv(directory / f"{number}.csv", index=False)
    display(Markdown(f"**Table {number}. {title}**"))
    if expanded:
        print(
            frame.to_string(
                index=False,
                max_rows=None,
                max_cols=None,
                max_colwidth=None,
                float_format=lambda v: f"{v:.6g}",
            )
        )
    else:
        content = frame.to_html(
            index=False, max_rows=None, max_cols=None, float_format=lambda v: f"{v:.6g}"
        )
        display(
            HTML(
                f'<details><summary>Expand the complete table ({len(frame)} rows)</summary><div style="overflow:auto;max-height:650px">{content}</div></details>'
            )
        )
    return frame


def figure(fig, number, root=ROOT):
    directory = Path(root) / INTEGRATED / "figures"
    directory.mkdir(parents=True, exist_ok=True)
    fig.savefig(directory / f"{number}.png", dpi=160, bbox_inches="tight")
    fig.savefig(directory / f"{number}.pdf", bbox_inches="tight")
    display(fig)
    plt.close(fig)


def saved_figure(relative, number, title, root=ROOT):
    if str(relative).replace("\\", "/") == "figures/common_case_comparison.png":
        from full_holdout_notebooks import _current_common_case

        if not _current_common_case(root, manifest(root)):
            raise ValueError(
                "Regenerate the common frozen-case figure with _complete_full_holdout_figures.py"
            )
    p = verified_path(relative, root)
    directory = Path(root) / INTEGRATED / "figures"
    directory.mkdir(parents=True, exist_ok=True)
    shutil.copy2(p, directory / f"{number}.png")
    pdf = p.with_suffix(".pdf")
    if pdf.exists():
        verified_path(str(Path(relative).with_suffix(".pdf")), root)
        shutil.copy2(pdf, directory / f"{number}.pdf")
    display(Markdown(f"**Figure {number}. {title}**"))
    display(Image(filename=str(p)))


def design_table(data):
    return pd.DataFrame(
        [
            ("Comparison", data["arm"]),
            ("Held-out target dates", data["n"]),
            ("Training target dates", 584),
            ("Prediction features", 7),
            ("Explanation budget", data["protocol"]["budget"]),
            ("Discovery seed", data["protocol"]["discovery_seed"]),
            ("Fixed-model reference seeds", str(data["protocol"]["reference_seeds"])),
            ("Case validation seeds", str(data["protocol"]["validation_seeds"])),
            ("Difference convention", "modified minus baseline"),
            (
                "Causal sampling",
                f"copula; {data['protocol']['sampling']['causal_sample_size']} imputation draws",
            ),
            ("Marginal background", data["protocol"]["sampling"]["marginal_sample_size"]),
            ("Target-date range", f"{data['points'].date.min()} to {data['points'].date.max()}"),
        ],
        columns=["setting", "value"],
    )


def bootstrap_table(values, labels, n_boot=5000, seed=42):
    values = np.asarray(values, dtype=float)
    if (
        values.ndim != 2
        or len(values) < 2
        or values.shape[1] != len(labels)
        or not np.isfinite(values).all()
    ):
        raise ValueError("Expected finite paired sample-by-feature values")
    if n_boot < 2:
        raise ValueError("n_boot must be at least two")
    rng = np.random.default_rng(seed)
    means = np.empty((n_boot, values.shape[1]))
    for start in range(0, n_boot, 200):
        stop = min(start + 200, n_boot)
        idx = rng.integers(0, len(values), size=(stop - start, len(values)))
        means[start:stop] = values[idx].mean(axis=1)
    lo, hi = np.quantile(means, [0.025, 0.975], axis=0)
    return pd.DataFrame(
        {
            "feature": labels,
            "mean": values.mean(0),
            "ci_low": lo,
            "ci_high": hi,
            "ci_excludes_zero": (lo > 0) | (hi < 0),
        }
    )


def plot_intervals(frame, title):
    ordered = frame.iloc[np.argsort(-np.abs(frame["mean"].to_numpy()), kind="stable")]
    fig, ax = plt.subplots(figsize=(10, 5), layout="constrained")
    y = np.arange(len(ordered))
    ax.hlines(y, ordered.ci_low, ordered.ci_high, color="#5b7e91", lw=2)
    ax.scatter(ordered["mean"], y, color="#b36b50", zorder=3)
    ax.axvline(0, color="0.4", ls="--", lw=0.8)
    ax.set_yticks(y, ordered.feature)
    ax.invert_yaxis()
    ax.set_xlabel("Mean paired attribution change")
    ax.set_title(
        title + "\nPointwise 95% row-bootstrap intervals, conditional on fixed explanations"
    )
    return fig


def rank_table(data):
    result = data["rank"]
    rows = []
    for key in du.TAU_VARIANTS:
        item = result[key]
        diff = item.get("paired_difference", {})
        rows.append(
            {
                "metric": key,
                "mean": item["mean"],
                "row_CI_low": item["ci_lo"],
                "row_CI_high": item["ci_hi"],
                "repeat_reference": result.get("null", {}).get(key, {}).get("mean", np.nan),
                "paired_adjusted_low": diff.get("family_ci_lo", np.nan),
                "paired_adjusted_high": diff.get("family_ci_hi", np.nan),
                "conditional_decision": du.tau_verdict(item),
            }
        )
    return pd.DataFrame(rows)


def prediction_checks(data):
    arm = data["arm"]
    bp = data["base_phi"]
    pp = data["variant_phi"]
    delta = (pp - bp).mean(0)
    mag = (np.abs(pp) - np.abs(bp)).mean(0)
    inter = (data["variant_interaction"] - data["base_interaction"]).mean(0)

    def d(name):
        return delta[FEATURES.index(name)]

    def dm(name):
        return mag[FEATURES.index(name)]

    def edge(a, b):
        return inter[PAIRS.index(tuple(sorted([FEATURES.index(a), FEATURES.index(b)])))]

    if arm == "b1":
        weather = [d(f) for f in ("temp", "hum", "windspeed")]
        rows = [
            (
                "All three named weather signed shifts > 0",
                bool(np.all(np.array(weather) > 0)),
                str(weather),
            ),
            ("Mean I(temp,hum) decreases", bool(edge("temp", "hum") < 0), edge("temp", "hum")),
            (
                "Revised weak atemp shift < max Layer-1 shift",
                bool(
                    abs(d("atemp"))
                    < max(abs(d(f)) for f in ("cosyear", "sinyear", "temp", "windspeed", "hum"))
                ),
                abs(d("atemp")),
            ),
        ]
    elif arm == "b2":
        rows = [
            (
                "Season mean per-row magnitudes increase in sum",
                bool(dm("cosyear") + dm("sinyear") > 0),
                dm("cosyear") + dm("sinyear"),
            ),
            (
                "Mean I(cosyear,temp) increases",
                bool(edge("cosyear", "temp") > 0),
                edge("cosyear", "temp"),
            ),
            ("Trend mean magnitude decreases", bool(dm("trend") < 0), dm("trend")),
        ]
    elif arm == "b3":
        rows = [
            ("Atemp mean magnitude increases", bool(dm("atemp") > 0), dm("atemp")),
            (
                "Mean I(atemp,temp) decreases",
                bool(edge("atemp", "temp") < 0),
                edge("atemp", "temp"),
            ),
            (
                "Weather mean magnitudes decrease in sum",
                bool(sum(dm(f) for f in ("temp", "hum", "windspeed")) < 0),
                sum(dm(f) for f in ("temp", "hum", "windspeed")),
            ),
        ]
    else:
        rows = []
    return pd.DataFrame(rows, columns=["operational_check", "direction_matches", "observed_value"])


def case_plots(data, row_id, rule, case_number, validation_number, root=ROOT, show_discovery=True):
    selected = data["selected"]
    if not ((selected.row_id == row_id) & (selected.case_rule == rule)).any():
        raise ValueError("Requested case is not in the frozen selection")
    chosen = selected[(selected.row_id == row_id) & (selected.case_rule == rule)].iloc[0]
    k = list(data["row_ids"]).index(row_id)
    readings = build_sample_readings(data["points"].iloc[k], data["arm"])
    label = f"Figure {case_number}: {rule} | {chosen.date} | original row {row_id} | percentile {chosen.effect_score_percentile:.1f}"
    if show_discovery:
        fig = plot_case_comparison(
            data["base_phi"][k],
            data["variant_phi"][k],
            data["base_interaction"][k],
            data["variant_interaction"][k],
            feature_names=FEATURES,
            case_label=label,
            variant_name=data["arm"],
        )
        add_sample_readings(fig, readings)
        figure(fig, case_number, root)
    details = data["validation_details"]
    details = details[(details.row_id == row_id) & (details.case_rule == rule)]
    seeds = data["protocol"]["validation_seeds"]
    phi = (
        details[details.order == 1]
        .pivot(index="seed", columns="item", values="delta")
        .reindex(index=seeds, columns=FEATURES)
        .to_numpy()
    )
    pair_labels = [" x ".join(FEATURES[i] for i in pair) for pair in PAIRS]
    interaction = (
        details[details.order == 2]
        .pivot(index="seed", columns="item", values="delta")
        .reindex(index=seeds, columns=pair_labels)
        .to_numpy()
    )
    if not np.isfinite(phi).all() or not np.isfinite(interaction).all():
        raise ValueError("Missing validation repeat")
    dp = data["variant_phi"][k] - data["base_phi"][k]
    di = data["variant_interaction"][k] - data["base_interaction"][k]
    top = np.argsort(-np.abs(di), kind="stable")[:5]
    fig, axes = plt.subplots(1, 2, figsize=(15, 5.8), layout="constrained")
    for ax, values, disc, labels in [
        (axes[0], phi, dp, FEATURES),
        (axes[1], interaction[:, top], di[top], [pair_labels[i] for i in top]),
    ]:
        x = np.arange(len(labels))
        for i, values_i in enumerate(values):
            ax.scatter(x + (i - 3.5) * 0.035, values_i, color="#477f9e", alpha=0.45, s=18)
        med = np.median(values, axis=0)
        lo, hi = np.quantile(values, [0.1, 0.9], axis=0)
        ax.errorbar(
            x,
            med,
            yerr=[med - lo, hi - med],
            fmt="o",
            color="#24566d",
            capsize=4,
            label="Fresh seeds: median / 10–90%",
        )
        ax.scatter(x, disc, marker="*", s=120, color="#ba5b43", label="Discovery seed 42", zorder=4)
        ax.axhline(0, color="0.5", lw=0.6)
        ax.set_xticks(x, labels, rotation=40, ha="right")
        ax.set_ylabel("Modified minus baseline")
        ax.legend(fontsize=8)
    axes[0].set_title("All first-order changes")
    axes[1].set_title("Largest discovery interaction changes")
    fig.suptitle(
        f"Figure {validation_number}: {rule} validation | {chosen.date} | row {row_id}\n8 new paired seeds; spread is not a population confidence interval.",
        fontsize=12,
    )
    add_sample_readings(fig, readings)
    figure(fig, validation_number, root)


def paired_lag_variance(base_phi, variant_phi, n_boot=5000, seed=42, feature_names=None):
    base = np.asarray(base_phi, dtype=float)
    variant = np.asarray(variant_phi, dtype=float)
    if (
        base.ndim != 2
        or base.shape != variant.shape
        or len(base) < 2
        or not np.isfinite(base).all()
        or not np.isfinite(variant).all()
    ):
        raise ValueError("Finite, matched sample-by-feature arrays required")
    if n_boot < 2:
        raise ValueError("n_boot must be at least two")
    names = (
        feature_names
        if feature_names is not None
        else [f"feature_{i}" for i in range(base.shape[1])]
    )
    if len(names) != base.shape[1]:
        raise ValueError("Feature label count mismatch")
    rng = np.random.default_rng(seed)
    b = np.empty((n_boot, base.shape[1]))
    v = np.empty_like(b)
    for start in range(0, n_boot, 200):
        stop = min(n_boot, start + 200)
        idx = rng.integers(0, len(base), size=(stop - start, len(base)))
        b[start:stop] = base[idx].mean(1)
        v[start:stop] = variant[idx].mean(1)
    vb = b.var(0, ddof=1)
    vv = v.var(0, ddof=1)
    ratios = np.divide(vv, vb, out=np.full_like(vv, np.nan), where=vb > 0)
    return {
        "table": pd.DataFrame(
            {
                "feature": names,
                "same_day_mean_variance": vb,
                "lag_mean_variance": vv,
                "ratio": ratios,
            }
        ),
        "base_draws": b,
        "variant_draws": v,
        "total_variance_ratio": float(vv.sum() / vb.sum()) if vb.sum() > 0 else np.nan,
        "n_resample_rows": len(base),
        "n_boot": n_boot,
    }


def interaction_thresholds(base_interaction, variant_interaction):
    a = np.asarray(base_interaction).mean(0)
    b = np.asarray(variant_interaction).mean(0)
    rows = []
    for fraction in (0.01, 0.05, 0.1, 0.2, 0.3, 0.5):
        rows.append(
            {
                "fraction_of_own_max": fraction,
                "same_day_pairs": int((np.abs(a) > fraction * np.max(np.abs(a))).sum()),
                "lag_pairs": int((np.abs(b) > fraction * np.max(np.abs(b))).sum()),
            }
        )
    return pd.DataFrame(rows)


def plot_lag_diagnostics(variance, thresholds, title):
    fig, axes = plt.subplots(1, 2, figsize=(14, 5), layout="constrained")
    t = variance["table"]
    x = np.arange(len(t))
    axes[0].bar(x - 0.18, t.same_day_mean_variance, 0.36, label="Same-day")
    axes[0].bar(x + 0.18, t.lag_mean_variance, 0.36, label="Lag")
    axes[0].set_xticks(x, t.feature, rotation=30, ha="right")
    axes[0].set_yscale("log")
    axes[0].set_ylabel("Row-bootstrap variance of mean attribution")
    axes[0].legend()
    axes[0].set_title(
        f"Shared resample dates; total ratio = {variance['total_variance_ratio']:.3f}"
    )
    axes[1].plot(thresholds.fraction_of_own_max, thresholds.same_day_pairs, "o-", label="Same-day")
    axes[1].plot(thresholds.fraction_of_own_max, thresholds.lag_pairs, "s-", label="Lag")
    axes[1].set_xlabel("Threshold as fraction of each arm's max |mean interaction|")
    axes[1].set_ylabel("Number of retained interaction pairs")
    axes[1].legend()
    fig.suptitle(
        title + "\nDescriptive matched-date design comparison; model and feature semantics differ."
    )
    return fig


def group_tables(arms, root=ROOT):
    return {
        a: {
            "summary": csv(f"tables/{a}_item_summary.csv", root),
            "points": csv(f"tables/{a}_points.csv", root),
            "score_scales": csv(f"tables/{a}_score_scales.csv", root),
        }
        for a in arms
    }


def current_summary(data, family=None, variance=None, root=ROOT, include_legacy=False):
    arms = family or [data["arm"]]
    summary = csv("tables/aggregate_summary.csv", root)
    result = summary[summary.variant.isin(arms)].copy()
    print(f"All results in this notebook use {data['n']} held-out target dates.")
    print(
        "Mean signed changes, absolute displacement, and magnitude changes are distinct summaries."
    )
    print(
        "Rank decisions are conditional fixed-model reference diagnostics, not causal truth or a sweep-wide test."
    )
    if include_legacy:
        for _, row in data["selection_rules"].iterrows():
            print(
                f"Legacy whole-explanation case rule {row.case_rule}: eligible {int(row.eligible_count)}/{int(row.cohort_size)}; {row.status}."
            )
    if variance is not None:
        print(
            f"Paired row-bootstrap total variance ratio = {variance['total_variance_ratio']:.4f}; "
            f"{variance['n_boot']} resamples of {variance['n_resample_rows']} paired dates."
        )
        print(
            "This measures row composition at fixed explanation seeds; it is not fresh Monte-Carlo variance."
        )
    print(
        "Case repeat plots are estimator-conditional checks. Strong discovery cases are not population evidence."
    )
    return result


def change_network(data, number, row_id=None, case_rule=None, top_k=3, root=ROOT):
    """Annotate a signed mean or one exact original row, from discovery arrays.

    Independent-seed summaries remain in the separate validation figures.
    Both graph arms here always come from the same saved discovery run.
    """
    import re

    from interaction_change_graph import plot_interaction_change_graph

    if not re.fullmatch(r"[A-Z][A-Z0-9]*-\d+", number):
        raise ValueError("Use an experiment-prefixed figure number.")
    labels = {
        "a": "Marginal",
        "lag": "Previous-day weather",
        "b1": "Confounding mark omitted",
        "b2": "Season codes promoted",
        "b3": "Mediator merged",
        "temp_root": "Temperature promoted",
        "hum_root": "Humidity promoted",
        "windspeed_root": "Windspeed promoted",
        "b2_unconfounded": "Season root, unmarked",
        "b3_unconfounded": "Merged, unmarked",
    }
    arrays = [
        data[k] for k in ("base_phi", "variant_phi", "base_interaction", "variant_interaction")
    ]
    if row_id is None:
        if case_rule is not None:
            raise ValueError("Mean view cannot have a single-case rule.")
        scope = f"{data['arm']} mean | discovery seed 42"
        date = None
    else:
        if isinstance(row_id, bool) or not float(row_id).is_integer():
            raise ValueError("row_id must be an original integer row ID")
        row_id = int(row_id)
        matches = np.flatnonzero(data["row_ids"] == row_id)
        if len(matches) != 1:
            raise ValueError("Requested original row is not in this held-out comparison")
        if case_rule is not None:
            selected = data["selected"]
            if not ((selected.row_id == row_id) & (selected.case_rule == case_rule)).any():
                raise ValueError("Case rule and row do not match the frozen selection")
        index = int(matches[0])
        arrays = [a[index] for a in arrays]
        date = str(data["points"].iloc[index]["date"])
        scope = f"{data['arm']} | {date} | original row {row_id}"
        if case_rule is not None:
            scope += f" | {case_rule}"
        scope += " | discovery seed 42"
    fig = plot_interaction_change_graph(
        *arrays,
        feature_names=FEATURES,
        title_prefix=f"Figure {number}",
        baseline_label={"a": "Causal", "lag": "Same-day weather"}.get(data["arm"], "Baseline"),
        variant_label=labels.get(data["arm"], data["arm"]),
        scope_label=scope,
        top_k=top_k,
    )
    directory = Path(root) / INTEGRATED / "change_graph_values"
    directory.mkdir(parents=True, exist_ok=True)
    if row_id is not None:
        readings = build_sample_readings(data["points"].iloc[index], data["arm"])
        add_sample_readings(fig, readings)
        (directory / f"{number}_readings.json").write_text(
            json.dumps(readings, ensure_ascii=False, indent=2), encoding="utf-8"
        )
    summary = fig.change_graph_summary
    for kind in ("nodes", "edges"):
        full = summary[kind].copy()
        full.insert(0, "variant", data["arm"])
        full.insert(1, "scope", "mean" if row_id is None else "sample")
        full.insert(2, "row_id", row_id)
        full.insert(3, "date", date)
        full.to_csv(directory / f"{number}_{kind}.csv", index=False)
    display(
        Markdown(
            f"**Figure {number}: directly annotated interaction changes.** "
            "Colors encode signed values; thickness encodes absolute values. "
            "Absolute-strength arrows and sign flips are marked separately. "
            f"The same union of top-{top_k} baseline, modified and change edges is used in every panel. "
            f"Complete [node values](exports/change_graph_values/{number}_nodes.csv) and "
            f"[all 21 pair values](exports/change_graph_values/{number}_edges.csv) are retained."
        )
    )
    figure(fig, number, root)
