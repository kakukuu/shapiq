"""Full-heldout, exploratory structural sensitivity and frozen local case validation.

The five experiment notebooks call this module automatically in a fresh process.
Data, fitted models, sampling settings and source fingerprints govern caches.
"""

from __future__ import annotations

import hashlib
import importlib.metadata
import json
import time
import warnings
from concurrent.futures import ProcessPoolExecutor, as_completed
from itertools import combinations
from pathlib import Path

import data_utils as du
import numpy as np
import pandas as pd
from example_paths import library_directory
from experiment_context import build_context, build_paired_lag_contexts
from threadpoolctl import threadpool_limits

from shapiq import InteractionValues
from shapiq.causal.ordering_graph import OrderingGraph

ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "results"
PROTOCOL_VERSION = "2026-10-13-fixed-imputation"
DEFAULT_CAUSAL_SAMPLE_SIZE = 500
DEFAULT_MARGINAL_SAMPLE_SIZE = "all"
EFFICIENCY_ATOL = 1e-4  # Absolute target units; permits floating-point regression residuals.
BUDGET = 5000
DISCOVERY_SEED = 42
REFERENCE_SEEDS = du.ESTIMATOR_REPEAT_SEEDS
VALIDATION_SEEDS = (809, 907, 1009, 1103, 1201, 1301, 1409, 1511)
FEATURES = du.FEATURE_NAMES
PAIRS = list(combinations(range(7), 2))
VARIANTS = {
    "baseline": {
        "label": "Baseline ordering",
        "ordering": du.ORDERING,
        "confounding": du.CONFOUNDING,
    },
    "b1": {
        "label": "Weather confounding mark off",
        "ordering": du.ORDERING_B1,
        "confounding": du.CONFOUNDING_B1,
    },
    "b2": {
        "label": "Season codes to root layer",
        "ordering": du.ORDERING_B2,
        "confounding": du.CONFOUNDING_B2,
    },
    "b3": {
        "label": "Mediator joined to weather layer",
        "ordering": du.ORDERING_B3,
        "confounding": du.CONFOUNDING_B3,
    },
    "temp_root": {
        "label": "Temperature to root layer",
        "ordering": [[0, 3], [1, 2, 5, 6], [4]],
        "confounding": [False, True, False],
    },
    "hum_root": {
        "label": "Humidity to root layer",
        "ordering": [[0, 6], [1, 2, 3, 5], [4]],
        "confounding": [False, True, False],
    },
    "windspeed_root": {
        "label": "Windspeed to root layer",
        "ordering": [[0, 5], [1, 2, 3, 6], [4]],
        "confounding": [False, True, False],
    },
    "b3_unconfounded": {
        "label": "Flattened mediator, confounding off",
        "ordering": du.ORDERING_B3,
        "confounding": [False, False],
    },
    "b2_unconfounded": {
        "label": "Season root, confounding off",
        "ordering": du.ORDERING_B2,
        "confounding": [False, False, False],
    },
}
A_SPEC = {"label": "Marginal versus causal", "imputer": "marginal"}
_WORKER_LIMIT = None


def encode(value):
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, Path):
        return str(value)
    raise TypeError(type(value).__name__)


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, default=encode), encoding="utf-8"
    )
    temporary.replace(path)


def source_hash(root=ROOT):
    """Hash source and data using stable names across checkout locations."""
    root = Path(root).resolve()
    library = library_directory(root)
    paths = [
        ("example/" + p.relative_to(root).as_posix(), p)
        for p in sorted((root / "experiment_support").glob("*.py"))
    ]
    paths += [
        ("example/" + p.relative_to(root).as_posix(), p)
        for p in sorted((root / "data").glob("*.csv"))
    ]
    paths += [
        ("library/" + p.relative_to(library).as_posix(), p) for p in sorted(library.rglob("*.py"))
    ]
    h = hashlib.sha256()
    for name, path in paths:
        h.update(name.encode())
        h.update(path.read_bytes())
    return h.hexdigest()


def sampling_config(
    causal_sample_size=DEFAULT_CAUSAL_SAMPLE_SIZE, marginal_sample_size=DEFAULT_MARGINAL_SAMPLE_SIZE
):
    for name, value in (
        ("causal_sample_size", causal_sample_size),
        ("marginal_sample_size", marginal_sample_size),
    ):
        if name == "marginal_sample_size" and value == "all":
            continue
        if not isinstance(value, (int, np.integer)) or isinstance(value, bool) or value < 1:
            raise ValueError(
                f"{name} must be a positive integer"
                + (" or 'all'" if name.startswith("marginal") else "")
            )
    return {
        "causal_sample_size": int(causal_sample_size),
        "marginal_sample_size": marginal_sample_size,
        "causal_sampling_method": "copula",
        "marginal_sampling_method": "joint",
    }


def dependency_versions():
    return {
        p: importlib.metadata.version(p)
        for p in ("numpy", "scipy", "xgboost", "scikit-learn", "matplotlib")
    }


def check_efficiency(phi, interaction, empty, predictions):
    residual = (
        np.asarray(phi).sum(axis=1) + np.asarray(interaction).sum(axis=1) + empty - predictions
    )
    if not np.isfinite(residual).all() or not np.allclose(
        residual, 0, rtol=0, atol=EFFICIENCY_ATOL
    ):
        raise ValueError(
            f"Attribution decomposition does not reconstruct predictions: max residual {np.max(np.abs(residual))}"
        )
    return residual


def catalog():
    def ancestors(spec):
        return {
            (a, b)
            for i, c in enumerate(spec["ordering"])
            for prev in spec["ordering"][:i]
            for a in prev
            for b in c
        }

    def conditioning(spec):
        g = OrderingGraph(spec["ordering"], spec["confounding"])
        entries = {}
        for mask in range(128):
            observed = np.array([bool(mask & (1 << i)) for i in range(7)])
            for layer, members in enumerate(spec["ordering"]):
                cond = tuple(g.get_conditioning_set(layer, observed).tolist())
                for f in members:
                    if not observed[f]:
                        entries[(mask, f)] = cond
        assert len(entries) == 448
        return entries

    base_edges = ancestors(VARIANTS["baseline"])
    base_cond = conditioning(VARIANTS["baseline"])
    records = []
    for name, spec in VARIANTS.items():
        flat = sum(spec["ordering"], [])
        if sorted(flat) != list(range(7)) or len(spec["confounding"]) != len(spec["ordering"]):
            raise ValueError(f"Invalid partition: {name}")
        edges = ancestors(spec)
        sets = conditioning(spec)
        changed = [k for k in sets if sets[k] != base_cond[k]]
        records.append(
            {
                "variant": name,
                "label": spec["label"],
                "ordering": json.dumps(spec["ordering"]),
                "confounding": json.dumps(spec["confounding"]),
                "added_ancestor_relations": len(edges - base_edges),
                "removed_ancestor_relations": len(base_edges - edges),
                "changed_conditioning_items": len(changed),
                "conditioning_items_total": 448,
                "changed_coalitions": len({k[0] for k in changed}),
                "added_relations": json.dumps(
                    [[FEATURES[a], FEATURES[b]] for a, b in sorted(edges - base_edges)]
                ),
                "removed_relations": json.dumps(
                    [[FEATURES[a], FEATURES[b]] for a, b in sorted(base_edges - edges)]
                ),
            }
        )
    return pd.DataFrame(records)


def _init_worker():
    global _WORKER_LIMIT
    _WORKER_LIMIT = threadpool_limits(limits=1)
    warnings.filterwarnings("ignore", category=UserWarning)


def _draw_worker(job):
    """Each worker receives an identical serialized fitted context; never refits."""
    context, spec, seed, row_ids, engine_hash, versions, cache_dir = job[:7]
    sampling = job[7] if len(job) > 7 else sampling_config()
    rows = context.X_test.loc[row_ids]
    spec = {k: v for k, v in spec.items() if k != "label"}
    is_causal = spec.get("imputer", "causal") == "causal"
    sample_size = sampling["causal_sample_size" if is_causal else "marginal_sample_size"]
    fingerprint = context.fingerprints()
    provenance = {
        "context": fingerprint,
        "graph": spec,
        "seed": seed,
        "initialization_seed": 42,
        "row_ids": [int(i) for i in row_ids],
        "row_values_sha256": hashlib.sha256(rows.to_numpy().tobytes()).hexdigest(),
        "budget": BUDGET,
        "index": "k-SII",
        "max_order": 2,
        "source_sha256": engine_hash,
        "versions": versions,
        "sampling": sampling,
        "sample_size": sample_size,
        "sampling_method": "copula" if is_causal else "joint",
        "worker_thread_limit": 1,
        "protocol_version": PROTOCOL_VERSION,
    }
    serialized = json.dumps(provenance, sort_keys=True, default=encode)
    key = hashlib.sha256(serialized.encode()).hexdigest()
    path = Path(cache_dir) / (key + ".npz")
    if path.exists():
        with np.load(path, allow_pickle=False) as saved:
            if saved["metadata"].item() != serialized:
                raise ValueError("Cache metadata mismatch")
            phi = saved["phi"]
            interaction = saved["interaction"]
            empty = saved["empty"]
            if (
                phi.shape != (len(rows), 7)
                or interaction.shape != (len(rows), 21)
                or empty.shape != (len(rows),)
                or saved["row_ids"].shape != (len(rows),)
                or not np.array_equal(saved["row_ids"], np.asarray(row_ids, dtype=int))
            ):
                raise ValueError("Cache array shape mismatch")
            if not all(np.isfinite(a).all() for a in (phi, interaction, empty)):
                raise ValueError("Nonfinite cache values")
            check_efficiency(phi, interaction, empty, context.predict(rows))
        return str(path), True
    ex = context.explainer(**spec, seed=42, sample_size=sample_size)
    if is_causal:
        if ex.imputer.sampling_method != "copula" or ex.imputer.sample_size != sample_size:
            raise AssertionError("Unexpected sampling backend/defaults")
    phi = []
    interaction = []
    empty = []
    for row_id in row_ids:
        result = du.explain_one(ex, context.X_test.loc[row_id], budget=BUDGET, seed=seed)
        d = result.dict_values
        phi.append([d[(i,)] for i in range(7)])
        interaction.append([d[p] for p in PAIRS])
        empty.append(float(result.baseline_value))
    values = {
        "phi": np.asarray(phi),
        "interaction": np.asarray(interaction),
        "empty": np.asarray(empty),
        "row_ids": np.asarray(row_ids, dtype=int),
        "metadata": np.array(serialized),
    }
    for v in ("phi", "interaction", "empty"):
        if not np.isfinite(values[v]).all():
            raise ValueError("Nonfinite explanation")
    check_efficiency(values["phi"], values["interaction"], values["empty"], context.predict(rows))
    temporary = path.with_suffix(".tmp.npz")
    np.savez_compressed(temporary, **values)
    temporary.replace(path)
    return str(path), False


def load_draw(path):
    with np.load(path, allow_pickle=False) as z:
        return {k: z[k].copy() for k in ("phi", "interaction", "empty", "row_ids")}


def as_sii(draw):
    result = []
    for phi, inter, empty in zip(draw["phi"], draw["interaction"], draw["empty"], strict=True):
        d = {(): float(empty)}
        d.update({(i,): float(v) for i, v in enumerate(phi)})
        d.update({p: float(v) for p, v in zip(PAIRS, inter, strict=True)})
        result.append(
            InteractionValues(
                d,
                index="k-SII",
                max_order=2,
                min_order=0,
                n_players=7,
                baseline_value=float(empty),
                estimation_budget=BUDGET,
            )
        )
    return result


def subset(draw, row_ids):
    index = {int(v): i for i, v in enumerate(draw["row_ids"])}
    positions = [index[int(i)] for i in row_ids]
    return {k: v[positions] for k, v in draw.items()}


def batch(pool, jobs, source, versions, sampling):
    futures = {}
    for key, context, spec, seed, ids in jobs:
        future = pool.submit(
            _draw_worker,
            (context, spec, seed, list(ids), source, versions, OUTPUT / "cache", sampling),
        )
        futures[future] = key
    results = {}
    for future in as_completed(futures):
        try:
            path, cached = future.result()
        except Exception:
            for pending in futures:
                pending.cancel()
            raise
        key = futures[future]
        results[key] = load_draw(path)
        print(
            f"{key}: {'cache' if cached else 'computed'} {len(results[key]['row_ids'])} rows",
            flush=True,
        )
    return results


def context_table(context, other=None):
    result = context.test_metadata.copy()
    for name in FEATURES:
        result[name] = context.X_test[name]
    result["y_true"] = context.y_test
    result["prediction_base"] = context.predict()
    result["prediction_variant"] = context.predict() if other is None else other.predict()
    if other is not None:
        for name in FEATURES:
            result[name + "_lag_design"] = other.X_test[name]
        for name in other.test_metadata:
            if name.startswith("lag_source"):
                result[name] = other.test_metadata[name]
    return result.reset_index(drop=True)


def save_tables(arm, tables):
    for key, suffix in [
        ("points", "points"),
        ("feature_details", "features"),
        ("interaction_details", "interactions"),
    ]:
        tables[key].to_csv(
            OUTPUT / "tables" / f"{arm}_{suffix}.csv", index=False, float_format="%.12g"
        )


def ensure_dirs():
    for folder in ("cache", "tables", "figures", "draws", "legacy_notebooks"):
        (OUTPUT / folder).mkdir(parents=True, exist_ok=True)


def run(
    causal_sample_size=DEFAULT_CAUSAL_SAMPLE_SIZE, marginal_sample_size=DEFAULT_MARGINAL_SAMPLE_SIZE
):
    import matplotlib
    from local_analysis import (
        build_local_tables,
        select_cases,
    )

    matplotlib.use("Agg")
    started = time.time()
    ensure_dirs()
    assert not set(VALIDATION_SEEDS) & ({DISCOVERY_SEED} | set(REFERENCE_SEEDS))
    engine = source_hash()
    versions = dependency_versions()
    sampling = sampling_config(causal_sample_size, marginal_sample_size)
    cat = catalog()
    cat.to_csv(OUTPUT / "tables/structural_catalog.csv", index=False)
    config = {
        "protocol_version": PROTOCOL_VERSION,
        "sampling": sampling,
        "efficiency_atol": EFFICIENCY_ATOL,
        "budget": BUDGET,
        "discovery_seed": DISCOVERY_SEED,
        "reference_seeds": REFERENCE_SEEDS,
        "validation_seeds": VALIDATION_SEEDS,
        "variants": VARIANTS,
        "source_sha256": engine,
        "versions": versions,
        "selection_policy": "typical, strong, high_rank_large_shift (tau>=.9 and effect>=Q75), weak; frozen before validation",
        "scope": "Exploratory; no overall or cross-variant causal significance claim. Random holdout, not forecasting.",
    }
    write_json(OUTPUT / "protocol.json", config)
    write_json(OUTPUT / "manifest.json", {"completed": False, "config": config})
    with threadpool_limits(limits=1):
        context = build_context()
        sd, lag = build_paired_lag_contexts()
    assert context.X_test.shape == (147, 7) and sd.X_test.shape == lag.X_test.shape == (146, 7)
    assert np.array_equal(sd.X_test.index, lag.X_test.index)
    model_provenance = {
        c.label: {"fingerprints": c.fingerprints(), "metrics": c.metrics()}
        for c in (context, sd, lag)
    }
    for c in (context, sd, lag):
        pd.concat(
            [c.train_metadata.assign(split="train"), c.test_metadata.assign(split="test")]
        ).to_csv(OUTPUT / "tables" / f"{c.label}_split.csv", index=False)
    write_json(OUTPUT / "models.json", model_provenance)
    with ProcessPoolExecutor(max_workers=2, initializer=_init_worker) as pool:
        jobs = [(name, context, spec, 42, context.X_test.index) for name, spec in VARIANTS.items()]
        jobs.append(("a", context, A_SPEC, 42, context.X_test.index))
        jobs.extend(
            [
                ("lag_base", sd, VARIANTS["baseline"], 42, sd.X_test.index),
                ("lag_pert", lag, VARIANTS["baseline"], 42, lag.X_test.index),
            ]
        )
        discovery = batch(pool, jobs, engine, versions, sampling)
        all_tables = {}
        selected = {}
        avail = {}
        arm_names = ["a"] + [n for n in VARIANTS if n != "baseline"] + ["lag"]
        for arm in arm_names:
            base = discovery["lag_base"] if arm == "lag" else discovery["baseline"]
            pert = discovery["lag_pert"] if arm == "lag" else discovery[arm]
            meta = context_table(sd, lag) if arm == "lag" else context_table(context)
            tables = build_local_tables(
                base["phi"],
                pert["phi"],
                base["interaction"],
                pert["interaction"],
                feature_names=FEATURES,
                row_ids=base["row_ids"],
                context=meta,
                variant_name=arm,
            )
            all_tables[arm] = tables
            save_tables(arm, tables)
            selected[arm], avail[arm] = select_cases(tables["points"])
            selected[arm].to_csv(OUTPUT / "tables" / f"{arm}_selected.csv", index=False)
            avail[arm].to_csv(OUTPUT / "tables" / f"{arm}_selection_rules.csv", index=False)
            np.savez_compressed(
                OUTPUT / "draws" / f"{arm}.npz",
                row_ids=base["row_ids"],
                base_phi=base["phi"],
                variant_phi=pert["phi"],
                base_interaction=base["interaction"],
                variant_interaction=pert["interaction"],
                base_empty=base["empty"],
                variant_empty=pert["empty"],
            )
        frozen_selection = {arm: sel.to_dict(orient="records") for arm, sel in selected.items()}
        write_json(
            OUTPUT / "selection_manifest.json",
            {
                "discovery_seed": 42,
                "validation_seeds": VALIDATION_SEEDS,
                "selection": frozen_selection,
                "policy": config["selection_policy"],
            },
        )
        selection_hash = hashlib.sha256(
            (OUTPUT / "selection_manifest.json").read_bytes()
        ).hexdigest()
        print(
            "Discovery complete; case selection frozen before independent-seed validation.",
            flush=True,
        )
        jobs = [
            (f"reference_{seed}", context, VARIANTS["baseline"], seed, context.X_test.index)
            for seed in REFERENCE_SEEDS
        ]
        union = sorted(
            {int(i) for arm in arm_names if arm != "lag" for i in selected[arm]["row_id"]}
        )
        lag_ids = sorted({int(i) for i in selected["lag"]["row_id"]})
        for seed in VALIDATION_SEEDS:
            jobs.append((f"validate_base_{seed}", context, VARIANTS["baseline"], seed, union))
            for arm in arm_names:
                if arm == "lag":
                    continue
                ids = sorted({int(i) for i in selected[arm]["row_id"]})
                jobs.append(
                    (
                        f"validate_{arm}_{seed}",
                        context,
                        A_SPEC if arm == "a" else VARIANTS[arm],
                        seed,
                        ids,
                    )
                )
            jobs.extend(
                [
                    (f"validate_lag_base_{seed}", sd, VARIANTS["baseline"], seed, lag_ids),
                    (f"validate_lag_pert_{seed}", lag, VARIANTS["baseline"], seed, lag_ids),
                ]
            )
        repeated = batch(pool, jobs, engine, versions, sampling)
    if (
        hashlib.sha256((OUTPUT / "selection_manifest.json").read_bytes()).hexdigest()
        != selection_hash
    ):
        raise AssertionError("Case selection changed after validation began")
    # Descriptive local validation, aggregate summaries and renderers are implemented below.
    finish(
        context,
        sd,
        lag,
        discovery,
        repeated,
        all_tables,
        selected,
        avail,
        config,
        model_provenance,
        selection_hash,
        started,
    )


def finish(
    context,
    sd,
    lag,
    discovery,
    repeated,
    all_tables,
    selected,
    avail,
    config,
    models,
    selection_hash,
    started,
):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from local_analysis import (
        build_local_tables,
        plot_aggregate_overview,
        plot_case_comparison,
        plot_sample_overview,
    )

    aggregate = []
    rank_results = {}
    validation_records = {}
    base = discovery["baseline"]
    null_runs = [as_sii(repeated[f"reference_{s}"]) for s in REFERENCE_SEEDS]
    ref_phi = np.stack(
        [np.abs(repeated[f"reference_{s}"]["phi"] - base["phi"]).sum(1) for s in REFERENCE_SEEDS]
    )
    ref_inter = np.stack(
        [
            np.abs(repeated[f"reference_{s}"]["interaction"] - base["interaction"]).sum(1)
            for s in REFERENCE_SEEDS
        ]
    )
    pd.DataFrame(
        {
            "row_id": base["row_ids"],
            "reference_first_L1_mean": ref_phi.mean(0),
            "reference_second_L1_mean": ref_inter.mean(0),
            "reference_first_L1_q10": np.quantile(ref_phi, 0.1, axis=0),
            "reference_first_L1_q90": np.quantile(ref_phi, 0.9, axis=0),
        }
    ).to_csv(OUTPUT / "tables/estimator_reference_points.csv", index=False)
    for arm, tables in all_tables.items():
        b = discovery["lag_base"] if arm == "lag" else base
        p = discovery["lag_pert"] if arm == "lag" else discovery[arm]
        rank_results[arm] = du.rank_metrics(
            as_sii(b),
            as_sii(p),
            FEATURES,
            n_boot=5000,
            seed=42,
            sii_null_runs=None if arm == "lag" else null_runs,
        )
        ranks = rank_results[arm]
        # Lag dates are paired, but models and input semantics differ; no null verdict.
        points = tables["points"]
        record = {
            "variant": arm,
            "n_samples": len(points),
            "mean_first_L1_change": points.first_l1_delta.mean(),
            "mean_second_L1_change": points.second_l1_delta.mean(),
            "median_effect_score": points.effect_score.median(),
            "q90_effect_score": points.effect_score.quantile(0.9),
            "mean_tau_first_signed": points.tau_first_signed.mean(),
            "mean_tau_second_signed": points.tau_second_signed.mean(),
            "mean_top5_overlap": points.top_k_interaction_overlap.mean(),
            "first_signed_sum": tables["summary"].query("order == 1").signed_mean_delta.sum(),
            "first_magnitude_change_sum": tables["summary"]
            .query("order == 1")
            .mean_magnitude_change.sum(),
            "reference_first_L1_mean": np.nan if arm == "lag" else ref_phi.mean(),
            "reference_second_L1_mean": np.nan if arm == "lag" else ref_inter.mean(),
            "rank_first_conditional_verdict": du.tau_verdict(ranks["tau1_signed"]),
            "rank_second_conditional_verdict": du.tau_verdict(ranks["tau2_signed"]),
        }
        aggregate.append(record)
        tables["summary"].to_csv(OUTPUT / "tables" / f"{arm}_item_summary.csv", index=False)
        tables["global_summary"].to_csv(OUTPUT / "tables" / f"{arm}_distribution.csv", index=False)
        tables["score_scales"].to_csv(OUTPUT / "tables" / f"{arm}_score_scales.csv", index=False)
        fig = plot_aggregate_overview({arm: tables}, OUTPUT / "figures" / f"{arm}_averages.png")
        if arm == "lag":
            fig.suptitle(
                "Matched-date same-day / lag design comparison (models and inputs differ)",
                fontsize=14,
            )
            fig.savefig(OUTPUT / "figures" / f"{arm}_averages.png", dpi=180, bbox_inches="tight")
            fig.savefig(OUTPUT / "figures" / f"{arm}_averages.pdf", bbox_inches="tight")
        plt.close(fig)
        rep_rows = []
        details = []
        summaries = []
        for _, chosen in selected[arm].iterrows():
            row_id = int(chosen.row_id)
            rule = chosen.case_rule
            bp = subset(b, [row_id])
            pp = subset(p, [row_id])
            dp = pp["phi"][0] - bp["phi"][0]
            di = pp["interaction"][0] - bp["interaction"][0]
            focus_feature = int(np.argmax(np.abs(dp)))
            focus_pair = int(np.argmax(np.abs(di)))
            base_reps = []
            pert_reps = []
            all_dp = []
            all_di = []
            for seed in VALIDATION_SEEDS:
                rb = subset(
                    repeated[
                        f"validate_lag_base_{seed}" if arm == "lag" else f"validate_base_{seed}"
                    ],
                    [row_id],
                )
                rp = subset(
                    repeated[
                        f"validate_lag_pert_{seed}" if arm == "lag" else f"validate_{arm}_{seed}"
                    ],
                    [row_id],
                )
                base_reps.append(rb)
                pert_reps.append(rp)
                vphi = rp["phi"][0] - rb["phi"][0]
                vint = rp["interaction"][0] - rb["interaction"][0]
                all_dp.append(vphi)
                all_di.append(vint)
                metric = build_local_tables(
                    rb["phi"],
                    rp["phi"],
                    rb["interaction"],
                    rp["interaction"],
                    feature_names=FEATURES,
                    row_ids=[row_id],
                    variant_name=arm,
                )["points"].iloc[0]
                rep_rows.append(
                    {
                        "variant": arm,
                        "row_id": row_id,
                        "case_rule": rule,
                        "seed": seed,
                        "date": chosen.get("date", ""),
                        "first_L1": np.abs(vphi).sum(),
                        "second_L1": np.abs(vint).sum(),
                        "tau_first_signed": metric.tau_first_signed,
                        "tau_second_signed": metric.tau_second_signed,
                        "focus_feature": FEATURES[focus_feature],
                        "focus_feature_delta": vphi[focus_feature],
                        "focus_pair": " x ".join(FEATURES[i] for i in PAIRS[focus_pair]),
                        "focus_pair_delta": vint[focus_pair],
                    }
                )
                for order, labels, vs, ds in [
                    (1, FEATURES, vphi, dp),
                    (2, [" x ".join(FEATURES[i] for i in pair) for pair in PAIRS], vint, di),
                ]:
                    for item, value, disc in zip(labels, vs, ds, strict=True):
                        details.append(
                            {
                                "variant": arm,
                                "row_id": row_id,
                                "case_rule": rule,
                                "seed": seed,
                                "order": order,
                                "item": item,
                                "delta": value,
                                "discovery_delta": disc,
                                "direction_agrees": bool(np.sign(value) == np.sign(disc)),
                            }
                        )
            all_dp = np.array(all_dp)
            all_di = np.array(all_di)
            # Independent seed-pair spreads within both fixed arms, for context.
            noise_b_phi = []
            noise_p_phi = []
            noise_b_int = []
            noise_p_int = []
            for i in range(0, len(VALIDATION_SEEDS), 2):
                noise_b_phi.append(np.abs(base_reps[i]["phi"] - base_reps[i + 1]["phi"]).sum())
                noise_p_phi.append(np.abs(pert_reps[i]["phi"] - pert_reps[i + 1]["phi"]).sum())
                noise_b_int.append(
                    np.abs(base_reps[i]["interaction"] - base_reps[i + 1]["interaction"]).sum()
                )
                noise_p_int.append(
                    np.abs(pert_reps[i]["interaction"] - pert_reps[i + 1]["interaction"]).sum()
                )
            summaries.append(
                {
                    "variant": arm,
                    "row_id": row_id,
                    "date": chosen.get("date", ""),
                    "case_rule": rule,
                    "n_validation_seeds": len(VALIDATION_SEEDS),
                    "discovery_percentile": chosen.effect_score_percentile,
                    "eligible_count": int(chosen.eligible_count),
                    "cohort_size": int(chosen.cohort_size),
                    "focus_feature": FEATURES[focus_feature],
                    "feature_direction_agreement": np.mean(
                        np.sign(all_dp[:, focus_feature]) == np.sign(dp[focus_feature])
                    ),
                    "feature_delta_mean": all_dp[:, focus_feature].mean(),
                    "feature_delta_std": all_dp[:, focus_feature].std(ddof=1),
                    "feature_delta_q10": np.quantile(all_dp[:, focus_feature], 0.1),
                    "feature_delta_q90": np.quantile(all_dp[:, focus_feature], 0.9),
                    "focus_pair": " x ".join(FEATURES[i] for i in PAIRS[focus_pair]),
                    "pair_direction_agreement": np.mean(
                        np.sign(all_di[:, focus_pair]) == np.sign(di[focus_pair])
                    ),
                    "pair_delta_mean": all_di[:, focus_pair].mean(),
                    "pair_delta_std": all_di[:, focus_pair].std(ddof=1),
                    "first_L1_discovery": np.abs(dp).sum(),
                    "first_L1_validation_mean": np.abs(all_dp).sum(1).mean(),
                    "second_L1_discovery": np.abs(di).sum(),
                    "second_L1_validation_mean": np.abs(all_di).sum(1).mean(),
                    "base_same_graph_first_L1": np.mean(noise_b_phi),
                    "pert_same_graph_first_L1": np.mean(noise_p_phi),
                    "base_same_graph_second_L1": np.mean(noise_b_int),
                    "pert_same_graph_second_L1": np.mean(noise_p_int),
                }
            )
            label = (
                f"{rule} | source row {row_id} | {chosen.get('date', '')} | "
                f"discovery percentile {chosen.effect_score_percentile:.1f} | eligible {int(chosen.eligible_count)}/{int(chosen.cohort_size)}"
            )
            fig = plot_case_comparison(
                bp["phi"][0],
                pp["phi"][0],
                bp["interaction"][0],
                pp["interaction"][0],
                feature_names=FEATURES,
                case_label=label,
                variant_name=arm,
                output_path=OUTPUT / "figures" / f"{arm}_case_{rule}_{row_id}.png",
            )
            plt.close(fig)
            validation_plot(arm, row_id, rule, dp, di, all_dp, all_di, label)
        pd.DataFrame(rep_rows).to_csv(OUTPUT / "tables" / f"{arm}_validation.csv", index=False)
        pd.DataFrame(details).to_csv(
            OUTPUT / "tables" / f"{arm}_validation_details.csv", index=False
        )
        pd.DataFrame(summaries).to_csv(
            OUTPUT / "tables" / f"{arm}_validation_summary.csv", index=False
        )
        validation_records[arm] = summaries
        print(f"{arm}: tables and {len(selected[arm])} frozen cases rendered", flush=True)
    structural = {arm: all_tables[arm] for arm in VARIANTS if arm != "baseline"}
    for fn, file in [
        (plot_aggregate_overview, "structural_averages.png"),
        (plot_sample_overview, "sample_by_variant.png"),
    ]:
        fig = fn(structural, OUTPUT / "figures" / file)
        plt.close(fig)
    input_structure_plot()
    pd.DataFrame(aggregate).to_csv(OUTPUT / "tables/aggregate_summary.csv", index=False)
    write_json(OUTPUT / "rank_calibration.json", rank_results)
    # Historical 50-row sample is only a subset of the new ordinary heldout context.
    fixed_keys = (
        "model_sha256",
        "background_sha256",
        "train_features_sha256",
        "test_features_sha256",
    )
    assert all(
        context.fingerprints()[k] == models[context.label]["fingerprints"][k] for k in fixed_keys
    )
    efficiency = []
    for name, draw in discovery.items():
        c = sd if name == "lag_base" else lag if name == "lag_pert" else context
        predictions = c.predict(c.X_test.loc[draw["row_ids"]])
        residual = check_efficiency(draw["phi"], draw["interaction"], draw["empty"], predictions)
        efficiency.extend(
            {"variant": name, "row_id": int(r), "prediction": float(p), "residual": float(e)}
            for r, p, e in zip(draw["row_ids"], predictions, residual, strict=True)
        )
    pd.DataFrame(efficiency).to_csv(OUTPUT / "tables/efficiency_checks.csv", index=False)
    artifacts = {
        str(p.relative_to(OUTPUT)): hashlib.sha256(p.read_bytes()).hexdigest()
        for folder in ("tables", "figures", "draws")
        for p in sorted((OUTPUT / folder).glob("*"))
        if p.is_file()
    }
    for name in (
        "rank_calibration.json",
        "models.json",
        "protocol.json",
        "selection_manifest.json",
    ):
        artifacts[name] = hashlib.sha256((OUTPUT / name).read_bytes()).hexdigest()
    manifest = {
        "completed": True,
        "postprocessing_completed": False,
        "config": config,
        "elapsed_seconds": time.time() - started,
        "ordinary_test_rows": 147,
        "paired_lag_test_rows": 146,
        "structural_perturbations": 8,
        "comparisons": list(all_tables),
        "selection_sha256": selection_hash,
        "selected_cases": sum(len(v) for v in selected.values()),
        "artifacts_sha256": artifacts,
        "max_efficiency_residual": max(abs(r["residual"]) for r in efficiency),
    }
    write_json(OUTPUT / "manifest.json", manifest)
    write_report(aggregate, selected, validation_records, models, manifest)
    complete_figures()
    print(
        "COMPLETE",
        json.dumps(
            {
                k: manifest[k]
                for k in (
                    "elapsed_seconds",
                    "ordinary_test_rows",
                    "paired_lag_test_rows",
                    "selected_cases",
                )
            }
        ),
        flush=True,
    )


def validation_plot(arm, row_id, rule, disc_phi, disc_int, phi_repeats, int_repeats, label):
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 2, figsize=(15, 6), layout="constrained")
    top = np.argsort(-np.abs(disc_int), kind="stable")[:5]
    for ax, data, discovery, labels in [
        (axes[0], phi_repeats, disc_phi, FEATURES),
        (
            axes[1],
            int_repeats[:, top],
            disc_int[top],
            [" x ".join(FEATURES[i] for i in PAIRS[k]) for k in top],
        ),
    ]:
        x = np.arange(len(labels))
        for i in range(len(data)):
            ax.scatter(
                x + (i - (len(data) - 1) / 2) * 0.035, data[i], s=20, alpha=0.4, color="#336b87"
            )
        med = np.median(data, axis=0)
        low = np.quantile(data, 0.1, axis=0)
        high = np.quantile(data, 0.9, axis=0)
        ax.errorbar(
            x,
            med,
            yerr=[med - low, high - med],
            fmt="o",
            color="#184e77",
            capsize=4,
            label="Fresh seeds: median and 10–90% spread",
        )
        ax.scatter(
            x, discovery, marker="*", s=140, color="#c05746", label="Discovery seed 42", zorder=4
        )
        ax.axhline(0, color="0.5", lw=0.7)
        ax.set_xticks(x, labels, rotation=40, ha="right")
        ax.set_ylabel("Modified minus baseline attribution")
        ax.legend(fontsize=8)
    axes[0].set_title("All first-order differences")
    axes[1].set_title("Five largest discovery interaction differences")
    fig.suptitle(
        f"{arm} | {label}\n8 new paired explanation seeds; repeat spread is not a population confidence interval.",
        fontsize=11,
    )
    stem = OUTPUT / "figures" / f"{arm}_validation_{rule}_{row_id}"
    fig.savefig(stem.with_suffix(".png"), dpi=170, bbox_inches="tight")
    fig.savefig(stem.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)


def input_structure_plot():
    import matplotlib.pyplot as plt
    from matplotlib.patches import Patch

    fig, axes = plt.subplots(3, 3, figsize=(17, 14), layout="constrained")
    for ax, (name, spec) in zip(axes.flat, VARIANTS.items(), strict=True):
        pos = {
            f: (j - (len(group) - 1) / 2, -level)
            for level, group in enumerate(spec["ordering"])
            for j, f in enumerate(group)
        }
        for level, group in enumerate(spec["ordering"]):
            for earlier in spec["ordering"][:level]:
                for a in earlier:
                    for b in group:
                        ax.annotate(
                            "",
                            xy=pos[b],
                            xytext=pos[a],
                            arrowprops={
                                "arrowstyle": "->",
                                "color": "#b2bac2",
                                "lw": 0.65,
                                "alpha": 0.45,
                                "shrinkA": 14,
                                "shrinkB": 14,
                            },
                        )
            color = "#bce0df" if spec["confounding"][level] else "#ead6ba"
            xs = [pos[f][0] for f in group]
            ys = [pos[f][1] for f in group]
            ax.scatter(xs, ys, s=1050, color=color, edgecolors="#52616b", zorder=3)
            for f in group:
                ax.text(*pos[f], FEATURES[f], ha="center", va="center", fontsize=8, zorder=4)
        ax.set_xlim(-3.2, 3.2)
        ax.set_ylim(-2.65, 0.6)
        ax.axis("off")
        ax.set_title(name + "\n" + spec["label"], fontsize=11)
    fig.legend(
        handles=[
            Patch(color="#bce0df", label="Component confounding flag True"),
            Patch(color="#ead6ba", label="Component confounding flag False"),
        ],
        loc="lower center",
        ncol=2,
    )
    fig.suptitle(
        "Input ordering configurations\nArrows encode all implied ancestor relations, not individually edited causal edges or output SII interactions.",
        fontsize=14,
    )
    for suffix in (".png", ".pdf"):
        fig.savefig(
            OUTPUT / "figures" / ("input_structures" + suffix), dpi=180, bbox_inches="tight"
        )
    plt.close(fig)


def write_report(aggregate, selected, validation, models, manifest):
    """Describe the complete cohort and distinguish supplementary L1 cases."""
    lines = [
        "# Full-holdout results",
        "",
        "A and eight structural variants share a fitted model, training background and 147 test dates. "
        "The Lag comparison uses 146 matched target dates with separately fitted models.",
        "",
        "See the [example guide](../README.md) for execution and target-driven case selection.",
        "",
        "- [Aggregate summaries](tables/aggregate_summary.csv) and [structure definitions](tables/structural_catalog.csv).",
        "- [Complete tables](tables/), [figures](figures/) and [attribution arrays](draws/).",
        "",
        "## Case selection",
        "",
        "The notebooks select target-specific cases and write their records to `targeted_cases/`. "
        "Those selections are computed when each notebook reaches its case-selection section.",
        "",
        "The counts below and `selection_manifest.json` refer only to supplementary whole-explanation L1 cases. "
        "Their effect score averages first- and second-order L1 changes normalized by the corresponding "
        "median baseline L1 magnitude. Typical, largest and smallest scores define these supplementary cases.",
        "",
        "## Predictive performance",
        "",
        "|Design|Training dates|Test dates|RMSE|R²|",
        "|---|---:|---:|---:|---:|",
    ]
    for name, info in models.items():
        m = info["metrics"]
        lines.append(f"|{name}|{m['n_train']}|{m['n_test']}|{m['rmse']:.2f}|{m['r2']:.3f}|")
    lines += [
        "",
        "## Comparison summaries",
        "",
        "|Comparison|N|Mean singleton L1 change|Mean pair L1 change|Mean singleton signed τ|Mean pair signed τ|Supplementary L1 cases|",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for a in aggregate:
        arm = a["variant"]
        lines.append(
            f"|{arm}|{a['n_samples']}|{a['mean_first_L1_change']:.1f}|{a['mean_second_L1_change']:.1f}|{a['mean_tau_first_signed']:.3f}|{a['mean_tau_second_signed']:.3f}|{len(selected[arm])}|"
        )
    lines += [
        "",
        "## Interpretation",
        "",
        "These structural comparisons and case selections are exploratory. Rank calibration adjusts four "
        "metrics within each configuration. Repeat 10–90% ranges describe imputation variability, "
        "rather than population confidence intervals. The random holdout assesses attribution designs, "
        "rather than prospective forecasting performance.",
        "",
        "## Reproducibility",
        "",
        "The manifest records model, background, test rows, structures, seeds, dependencies and source hashes. "
        "Complete result checksums are verified before rendering.",
        f"Sampling configuration: `{manifest['config']['sampling']}`.",
        f"Maximum prediction reconstruction residual: {manifest['max_efficiency_residual']:.3g} rentals.",
        f"Supplementary selection SHA256: `{manifest['selection_sha256']}`.",
        "",
    ]
    (OUTPUT / "README.md").write_text("\n".join(lines), encoding="utf-8")


def complete_figures():
    from common_case_figure import render_common_case

    manifest = json.loads((OUTPUT / "manifest.json").read_text(encoding="utf-8"))
    shared = render_common_case(OUTPUT)
    if shared:
        manifest["common_frozen_case"] = shared["layout"]
        for key in ("png", "pdf", "layout_path"):
            path = Path(shared[key])
            manifest["artifacts_sha256"][str(path.relative_to(OUTPUT))] = hashlib.sha256(
                path.read_bytes()
            ).hexdigest()
    manifest["figure_png_count"] = len(list((OUTPUT / "figures").glob("*.png")))
    manifest["figure_pdf_count"] = len(list((OUTPUT / "figures").glob("*.pdf")))
    manifest["postprocessing_completed"] = True
    write_json(OUTPUT / "manifest.json", manifest)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--causal-samples", type=int, default=DEFAULT_CAUSAL_SAMPLE_SIZE)
    parser.add_argument("--marginal-samples", default=DEFAULT_MARGINAL_SAMPLE_SIZE)
    args = parser.parse_args()
    marginal = (
        args.marginal_samples if args.marginal_samples == "all" else int(args.marginal_samples)
    )
    run(args.causal_samples, marginal)
