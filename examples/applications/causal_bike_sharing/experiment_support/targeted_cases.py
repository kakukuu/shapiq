"""Target-first, frozen descriptive case selection and independent-seed checks.

This layer consumes the verified full-cohort experiment. It does not change the
model, attribution estimator or original discovery arrays. Only missing case
repetitions are computed; numerical caches are shared across selection rules.
"""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import subprocess
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import datetime
from html import escape
from pathlib import Path

import data_utils as du
import numpy as np
import pandas as pd
import run_full_holdout as pipeline
from case_selection import DEFAULT_CONFIG, default_target_specs, select_target_cases
from integrated_support import (
    RESULTS,
    ROOT,
    calculation_lock,
    load_experiment,
    manifest,
    verified_path,
)
from threadpoolctl import threadpool_limits

DIRECTORY = "targeted_cases"
VERSION = "target-first-v1"
TABLES = (
    "selected_unique",
    "reasons",
    "candidates",
    "rules",
    "target_specs",
    "validation_details",
    "validation_summary",
    "reason_validation",
)


def _hash(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _records(frame):
    return json.loads(frame.to_json(orient="records", double_precision=15))


def _json_hash(value):
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, ensure_ascii=False, default=pipeline.encode).encode(
            "utf-8"
        )
    ).hexdigest()


def _folder(root, arm):
    if arm not in (*pipeline.VARIANTS, "a", "lag") or arm == "baseline":
        raise ValueError(f"Unknown comparison: {arm}")
    return Path(root) / RESULTS / DIRECTORY / arm


def _provenance(arm, root, config, targets):
    parent = manifest(root)
    hashes = {k.replace("\\", "/"): v for k, v in parent["artifacts_sha256"].items()}
    inputs = [f"draws/{arm}.npz", f"tables/{arm}_points.csv", "models.json"]
    for name in inputs:
        verified_path(name, root)
    return {
        "version": VERSION,
        "arm": arm,
        "config": config,
        "targets": targets,
        "selection_source_sha256": _hash(Path(__file__).with_name("case_selection.py")),
        "validation_source_sha256": _hash(__file__),
        "validation_summary_source_sha256": _hash(
            Path(__file__).with_name("case_selection_figures.py")
        ),
        "inputs": {name: hashes[name] for name in inputs},
        "numerical_protocol": parent["config"],
        "validation_seeds": list(parent["config"]["validation_seeds"]),
    }


def _read_manifest(arm, root, expected=None):
    folder = _folder(root, arm)
    value = json.loads((folder / "manifest.json").read_text(encoding="utf-8"))
    if not value.get("completed"):
        raise RuntimeError(f"Targeted validation is incomplete for {arm}.")
    if expected is not None and value["provenance"] != expected:
        raise RuntimeError(f"Targeted selection settings or inputs changed for {arm}.")
    required = {f"{name}.csv" for name in TABLES} | {"selection.json"}
    if not required.issubset(value.get("artifacts_sha256", {})):
        raise ValueError(f"Missing targeted case checksums for {arm}")
    for name, digest in value["artifacts_sha256"].items():
        if _hash(folder / name) != digest:
            raise ValueError(f"Targeted case checksum mismatch: {arm}/{name}")
    frozen = json.loads((folder / "selection.json").read_text(encoding="utf-8"))
    if frozen["provenance"] != value["provenance"]:
        raise ValueError("Frozen selection provenance differs from the completed validation")
    return value


def ensure_targeted_cases(arms, root=ROOT, mode="auto", selection_config=None, targets=None):
    """Notebook entry: freeze target-specific cases and verify all selected dates.

    No validation statistic influences the selection, number of cases or their
    presentation order. Failed repeat checks remain in the resulting evidence.
    """
    arms = list(dict.fromkeys(arms))
    if not arms:
        raise ValueError("At least one comparison is required.")
    if mode not in ("auto", "load", "recompute"):
        raise ValueError("mode must be auto, load, or recompute")
    config = {**DEFAULT_CONFIG, **(selection_config or {})}
    extra = set(config) - set(DEFAULT_CONFIG)
    if extra:
        raise ValueError(f"Unknown selection settings: {sorted(extra)}")
    requested = {arm: (targets or {}).get(arm, default_target_specs(arm)) for arm in arms}
    stale = []
    for arm in arms:
        # Also validate custom thresholds and target names before creating files.
        select_target_cases(load_experiment(arm, root), requested[arm], **config)
        expected = _provenance(arm, root, config, requested[arm])
        try:
            _read_manifest(arm, root, expected)
        except (FileNotFoundError, RuntimeError, ValueError, KeyError):
            stale.append(arm)
    if mode == "load" and stale:
        raise RuntimeError(
            f"Missing or stale target-specific validation: {stale}. Use RUN_MODE='auto'."
        )
    if mode == "recompute":
        stale = arms
    if stale:
        request = {
            "arms": stale,
            "root": str(Path(root).resolve()),
            "selection_config": config,
            "targets": {arm: requested[arm] for arm in stale},
        }
        requests = Path(root) / RESULTS / DIRECTORY / "requests"
        requests.mkdir(parents=True, exist_ok=True)
        path = requests / (_json_hash(request) + ".json")
        pipeline.write_json(path, request)
        print(
            f"Freezing target-first cases and validating {', '.join(stale)}; no case-count cap.",
            flush=True,
        )
        environment = os.environ.copy()
        environment.update(PYTHONUTF8="1", PYTHONIOENCODING="utf-8", MPLBACKEND="Agg")
        command = [sys.executable, "-u", str(Path(__file__).resolve()), "--request", str(path)]
        with subprocess.Popen(
            command,
            cwd=root,
            env=environment,
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
                    "Targeted case validation failed. Matching numerical caches can be resumed."
                )
    else:
        print("Reusing verified target-first selections and all independent-seed checks.")
    for arm in arms:
        _read_manifest(arm, root, _provenance(arm, root, config, requested[arm]))
    return {"arms": arms, "config": config, "status": "verified"}


def load_targeted_cases(arm, root=ROOT):
    saved = _read_manifest(arm, root)
    p = saved["provenance"]
    _read_manifest(arm, root, _provenance(arm, root, p["config"], p["targets"]))
    result = {name: pd.read_csv(_folder(root, arm) / f"{name}.csv") for name in TABLES}
    result.update(arm=arm, manifest=saved, data=load_experiment(arm, root))
    return result


def _numeric_identity(context, spec, protocol):
    causal = spec.get("imputer", "causal") == "causal"
    return {
        "context": context.fingerprints(),
        "graph": {k: v for k, v in spec.items() if k != "label"},
        "source_sha256": protocol["source_sha256"],
        "versions": protocol["versions"],
        "sampling": protocol["sampling"],
        "budget": protocol["budget"],
        "initialization_seed": protocol["discovery_seed"],
        "index": "k-SII",
        "max_order": 2,
        "protocol_version": protocol["protocol_version"],
        "worker_thread_limit": 1,
        "sample_size": protocol["sampling"][
            "causal_sample_size" if causal else "marginal_sample_size"
        ],
        "sampling_method": "copula" if causal else "joint",
    }


def _read_cached_rows(paths, context, spec, protocol, requested_ids):
    expected = _numeric_identity(context, spec, protocol)
    expected_json = {key: _json_hash(value) for key, value in expected.items()}
    found = {}
    for path in paths:
        with np.load(path, allow_pickle=False) as saved:
            serialized = saved["metadata"].item()
            metadata = json.loads(serialized)
            if any(_json_hash(metadata.get(k)) != v for k, v in expected_json.items()):
                continue
            seed = int(metadata["seed"])
            if seed not in protocol["validation_seeds"]:
                continue
            if hashlib.sha256(serialized.encode()).hexdigest() != path.stem:
                raise ValueError(f"Invalid numerical cache key: {path.name}")
            raw_ids = saved["row_ids"]
            if raw_ids.ndim != 1 or raw_ids.dtype.kind not in "iu":
                raise ValueError("Cached row IDs must be a one-dimensional integer array")
            ids = raw_ids.astype(int)
            if not set(ids) & set(requested_ids):
                continue
            if ids.tolist() != metadata["row_ids"] or len(set(ids)) != len(ids):
                raise ValueError("Invalid cached row identities")
            rows = context.X_test.loc[ids]
            if (
                hashlib.sha256(rows.to_numpy().tobytes()).hexdigest()
                != metadata["row_values_sha256"]
            ):
                raise ValueError("Cached inputs differ from the current fitted context")
            arrays = {k: saved[k].copy() for k in ("phi", "interaction", "empty")}
            if (
                arrays["phi"].shape != (len(ids), 7)
                or arrays["interaction"].shape != (len(ids), 21)
                or arrays["empty"].shape != (len(ids),)
            ):
                raise ValueError("Invalid numerical cache shape")
            if not all(np.isfinite(v).all() for v in arrays.values()):
                raise ValueError("Nonfinite cached attribution")
            pipeline.check_efficiency(
                arrays["phi"], arrays["interaction"], arrays["empty"], context.predict(rows)
            )
            for i, row_id in enumerate(ids):
                if row_id not in requested_ids:
                    continue
                key = (seed, int(row_id))
                record = {k: v[i].copy() for k, v in arrays.items()}
                if key in found:
                    for name in record:
                        if not np.array_equal(found[key][name], record[name]):
                            raise ValueError(
                                "Conflicting numerical caches for the same point and seed"
                            )
                found[key] = record
    return found


def _reason_validation(reasons, details, points):
    """Recheck the named phenomenon, not merely the direction of one delta."""
    from scipy.stats import kendalltau

    pairs = list(pipeline.PAIRS)
    labels = {
        1: du.FEATURE_NAMES,
        2: [f"{du.FEATURE_NAMES[i]} × {du.FEATURE_NAMES[j]}" for i, j in pairs],
    }
    rows = []
    for reason in reasons.to_dict(orient="records"):
        subset = details[details.row_id.eq(reason["row_id"])]
        held, count = 0, 0
        for _seed, group in subset.groupby("seed", sort=True):
            own = group[group.order.eq(int(reason["order"]))].set_index("item")
            target = own.loc[reason["item"]]
            rule = reason["case_rule"]
            same_direction = np.sign(target.delta) == np.sign(reason["delta"])
            material = abs(target.delta) >= reason["material_threshold"]
            if rule == "material_sign_flip":
                ok = (
                    target.baseline * target.modified < 0
                    and min(abs(target.baseline), abs(target.modified))
                    >= reason["endpoint_threshold"]
                    and np.sign(target.baseline) == np.sign(reason["baseline"])
                    and np.sign(target.modified) == np.sign(reason["modified"])
                )
            elif rule == "rank_preserved_large_change":
                ordered = own.loc[labels[int(reason["order"])]]
                tau = float(kendalltau(ordered.baseline, ordered.modified).statistic)
                ok = (
                    tau >= reason["rank_threshold"]
                    and same_direction
                    and material
                    and abs(target.delta) >= reason["target_abs_delta_quantile"]
                )
            elif rule == "paired_redistribution":
                singles = group[group.order.eq(1)].set_index("item")
                opposite = False
                for suffix in ("a", "b"):
                    name = reason.get(f"related_feature_{suffix}")
                    if isinstance(name, str) and name in singles.index:
                        value = singles.loc[name, "delta"]
                        opposite |= (
                            value * target.delta < 0
                            and abs(value) >= reason[f"related_threshold_{suffix}"]
                        )
                ok = same_direction and material and opposite
            elif rule == "typical_target_change":
                # 'Typical' describes the fixed discovery cohort; repeats do
                # not reselect or claim to establish a new cohort median.
                ok = same_direction
            else:
                ok = same_direction and material
            count += 1
            held += int(bool(ok))
        repeat_labels = {
            "material_sign_flip": "Same material endpoint sign reversal",
            "rank_preserved_large_change": "Own-order rank and material target shift retained",
            "paired_redistribution": "Material opposing pair/singleton changes retained",
            "typical_target_change": "Target delta direction only; typicality belongs to discovery",
            "opposite_direction": "Material direction opposite to discovery-cohort median retained",
            "largest_target_change": "Material target delta direction retained; maximum not re-ranked",
        }
        rows.append(
            {
                **reason,
                "repeat_check": repeat_labels[reason["case_rule"]],
                "n_validation_seeds": count,
                "phenomenon_repeat_count": held,
                "phenomenon_repeat_fraction": held / count if count else np.nan,
                "validation_note": "Discovery extremeness/typicality is not re-ranked under validation seeds.",
            }
        )
    return pd.DataFrame(rows)


def compute_targeted_cases(arms, root=ROOT, selection_config=None, targets=None):
    """CLI implementation; persist every selection before computing repetitions."""
    from case_selection_figures import summarize_target_validation
    from experiment_context import build_context, build_paired_lag_contexts

    root = Path(root).resolve()
    config = {**DEFAULT_CONFIG, **(selection_config or {})}
    parent = manifest(root)
    protocol = parent["config"]
    plans, frozen_hashes = {}, {}
    cache_dir = root / RESULTS / DIRECTORY / "cache"
    cache_dir.mkdir(parents=True, exist_ok=True)
    with calculation_lock(root), threadpool_limits(limits=1):
        models = json.loads(verified_path("models.json", root).read_text(encoding="utf-8"))
        context = build_context() if any(arm != "lag" for arm in arms) else None
        same, lagged = build_paired_lag_contexts() if "lag" in arms else (None, None)
        for c in (context, same, lagged):
            if c is not None and c.fingerprints() != models[c.label]["fingerprints"]:
                raise ValueError(
                    "Reconstructed model/background differ from discovery; rerun discovery."
                )
        # Validate every requested selection before replacing any existing result.
        for arm in arms:
            target_list = (targets or {}).get(arm, default_target_specs(arm))
            data = load_experiment(arm, root)
            tables = select_target_cases(data, target_list, **config)
            provenance = _provenance(arm, root, config, target_list)
            plans[arm] = {"data": data, "tables": tables, "provenance": provenance}
        for arm, plan in plans.items():
            tables, provenance = plan["tables"], plan["provenance"]
            folder = _folder(root, arm)
            if folder.exists():
                archived = (
                    root
                    / "archive"
                    / ("targeted_cases_" + datetime.now().strftime("%Y%m%d_%H%M%S_%f"))
                    / arm
                )
                folder.resolve().relative_to(root)
                archived.resolve().relative_to(root)
                archived.parent.mkdir(parents=True, exist_ok=True)
                shutil.move(str(folder), str(archived))
            folder.mkdir(parents=True)
            frozen = {
                "provenance": provenance,
                "selected_unique": _records(tables["selected_unique"]),
                "reasons": _records(tables["reasons"]),
            }
            pipeline.write_json(folder / "selection.json", frozen)
            frozen_hashes[arm] = _hash(folder / "selection.json")
            for key, frame in tables.items():
                frame.to_csv(folder / f"{key}.csv", index=False)
            pipeline.write_json(
                folder / "manifest.json", {"completed": False, "provenance": provenance}
            )
            print(
                f"{arm}: frozen {len(tables['selected_unique'])} dates / {len(tables['reasons'])} target-specific reasons",
                flush=True,
            )
        contexts, specs, requested_ids = {}, {}, {}
        for arm, plan in plans.items():
            ids = set(plan["tables"]["selected_unique"].row_id.astype(int))
            base_key = "lag_base" if arm == "lag" else "baseline"
            contexts[base_key] = same if arm == "lag" else context
            contexts[arm] = lagged if arm == "lag" else context
            specs[base_key] = pipeline.VARIANTS["baseline"]
            specs[arm] = (
                pipeline.VARIANTS["baseline"]
                if arm == "lag"
                else pipeline.A_SPEC
                if arm == "a"
                else pipeline.VARIANTS[arm]
            )
            requested_ids.setdefault(base_key, set()).update(ids)
            requested_ids.setdefault(arm, set()).update(ids)
        paths = sorted((root / RESULTS / "cache").glob("*.npz")) + sorted(cache_dir.glob("*.npz"))
        caches, jobs = {}, []
        for key, ids in requested_ids.items():
            caches[key] = _read_cached_rows(paths, contexts[key], specs[key], protocol, ids)
            for seed in protocol["validation_seeds"]:
                missing = sorted(i for i in ids if (seed, i) not in caches[key])
                if missing:
                    job = (
                        contexts[key],
                        specs[key],
                        seed,
                        missing,
                        protocol["source_sha256"],
                        protocol["versions"],
                        cache_dir,
                        protocol["sampling"],
                    )
                    jobs.append((key, seed, job))
        print(
            f"Reusing {sum(map(len, caches.values()))} verified point/seed draws; computing {sum(len(j[2][3]) for j in jobs)} missing draws.",
            flush=True,
        )
        if jobs:
            with ProcessPoolExecutor(max_workers=2, initializer=pipeline._init_worker) as pool:
                futures = {
                    pool.submit(pipeline._draw_worker, job): (key, seed, job)
                    for key, seed, job in jobs
                }
                for future in as_completed(futures):
                    try:
                        path, cached = future.result()
                    except Exception:
                        for pending in futures:
                            pending.cancel()
                        raise
                    key, seed, job = futures[future]
                    draw = pipeline.load_draw(path)
                    for i, row_id in enumerate(draw["row_ids"]):
                        caches[key][(seed, int(row_id))] = {
                            k: draw[k][i] for k in ("phi", "interaction", "empty")
                        }
                    print(
                        f"{key} / seed {seed}: {len(draw['row_ids'])} missing rows validated",
                        flush=True,
                    )
        for arm, plan in plans.items():
            folder = _folder(root, arm)
            if _hash(folder / "selection.json") != frozen_hashes[arm]:
                raise ValueError("Selection changed during validation")
            rows = []
            data = plan["data"]
            base_key = "lag_base" if arm == "lag" else "baseline"
            metadata = data["points"].set_index("row_id")
            for row_id in plan["tables"]["selected_unique"].row_id.astype(int):
                for seed in protocol["validation_seeds"]:
                    left, right = caches[base_key][(seed, row_id)], caches[arm][(seed, row_id)]
                    for order, key, labels in (
                        (1, "phi", du.FEATURE_NAMES),
                        (
                            2,
                            "interaction",
                            [
                                f"{du.FEATURE_NAMES[i]} × {du.FEATURE_NAMES[j]}"
                                for i, j in pipeline.PAIRS
                            ],
                        ),
                    ):
                        for item, baseline, modified in zip(
                            labels, left[key], right[key], strict=True
                        ):
                            rows.append(
                                {
                                    "arm": arm,
                                    "row_id": row_id,
                                    "date": metadata.loc[row_id, "date"],
                                    "seed": seed,
                                    "order": order,
                                    "item": item,
                                    "baseline": float(baseline),
                                    "modified": float(modified),
                                    "delta": float(modified - baseline),
                                    "abs_delta": float(abs(modified - baseline)),
                                    "sign_flip": bool(baseline * modified < 0),
                                }
                            )
            details = pd.DataFrame(
                rows,
                columns=[
                    "arm",
                    "row_id",
                    "date",
                    "seed",
                    "order",
                    "item",
                    "baseline",
                    "modified",
                    "delta",
                    "abs_delta",
                    "sign_flip",
                ],
            )
            summaries = []
            reasons = plan["tables"]["reasons"]
            for row_id in plan["tables"]["selected_unique"].row_id.astype(int):
                summary = summarize_target_validation(details, reasons, arm=arm, row_id=row_id)
                summary["arm"], summary["row_id"] = arm, row_id
                summaries.append(summary)
            validation_summary = (
                pd.concat(summaries, ignore_index=True)
                if summaries
                else pd.DataFrame(columns=["arm", "row_id", "order", "item", "n_seeds"])
            )
            reason_validation = (
                _reason_validation(reasons, details, data["points"])
                if len(reasons)
                else reasons.assign(n_validation_seeds=pd.Series(dtype=int))
            )
            for key, frame in (
                ("validation_details", details),
                ("validation_summary", validation_summary),
                ("reason_validation", reason_validation),
            ):
                frame.to_csv(folder / f"{key}.csv", index=False)
            artifacts = {p.name: _hash(p) for p in folder.iterdir() if p.suffix == ".csv"}
            artifacts["selection.json"] = frozen_hashes[arm]
            pipeline.write_json(
                folder / "manifest.json",
                {
                    "completed": True,
                    "provenance": plan["provenance"],
                    "artifacts_sha256": artifacts,
                    "unique_dates": len(plan["tables"]["selected_unique"]),
                    "selection_reasons": len(reasons),
                    "validation_seeds": protocol["validation_seeds"],
                    "selection_frozen_before_validation_stage": True,
                    "selection_uses_validation_values": False,
                    "validation_note": "Seeds are distinct from discovery; matching existing draws may be reused. Exploratory, not preregistered confirmation.",
                },
            )
            print(
                f"{arm}: complete; every selected target retained regardless of repeat results.",
                flush=True,
            )


def _save_case_figure(fig, path):
    import matplotlib.pyplot as plt

    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=135, bbox_inches="tight")
    fig.savefig(path.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)


def _table_html(frame):
    return frame.to_html(index=False, na_rep="", float_format=lambda x: f"{x:.6g}")


def render_targeted_cases(focused, prefix, root=ROOT, expanded_tables=False):
    """Render every selected date; merge reasons and explicitly show target pairs.

    Each case has two visible native PNG outputs, independent of HTML table
    expansion. Long audit tables follow the figures; no top-N limit is applied.
    """
    from case_selection_figures import plot_target_validation
    from interaction_change_graph import plot_interaction_change_graph
    from IPython.display import HTML, Image, Markdown, display
    from sample_readings import add_sample_readings, build_sample_readings

    data, arm = focused["data"], focused["arm"]
    folder = _folder(root, arm)
    heading = f"{arm}: {len(focused['selected_unique'])} dates, {len(focused['reasons'])} selection reasons"
    display(Markdown("### " + heading))
    display(
        Markdown(
            "Every selected date is rendered below as two native notebook images. "
            "Only supporting tables are collapsible; table settings never hide the figures."
        )
    )
    dates = data["points"].set_index("row_id")
    render_path = folder / "render_manifest.json"
    try:
        renders = json.loads(render_path.read_text(encoding="utf-8"))
    except (FileNotFoundError, ValueError):
        renders = {}
    render_sources = {
        name: _hash(Path(__file__).with_name(name))
        for name in (
            "targeted_cases.py",
            "interaction_change_graph.py",
            "case_selection_figures.py",
            "sample_readings.py",
        )
    }
    for selected in focused["selected_unique"].itertuples(index=False):
        row_id = int(selected.row_id)
        reasons = focused["reasons"].loc[focused["reasons"].row_id.eq(row_id)]
        index = list(data["row_ids"]).index(row_id)
        readings = build_sample_readings(data["points"].iloc[index], arm)
        fields = ["base_phi", "variant_phi", "base_interaction", "variant_interaction"]
        arrays = [data[name][index] for name in fields]
        features = sorted(set(reasons.loc[reasons.order.eq(1), "item"]))
        pair_names = sorted(set(reasons.loc[reasons.order.eq(2), "item"]))
        for item in pair_names:
            features.extend(item.split(" × "))
        features = list(dict.fromkeys(features))
        date = str(dates.loc[row_id, "date"])
        title = f"{prefix} | {arm} | {date} | original row {row_id}"
        graph_path = folder / "figures" / f"case_{row_id}_network.png"
        repeat_path = folder / "figures" / f"case_{row_id}_validation.png"
        readings_path = folder / f"case_{row_id}_readings.json"
        render_key = _json_hash(
            {
                "source": render_sources,
                "prefix": prefix,
                "row_id": row_id,
                "inputs": focused["manifest"]["artifacts_sha256"],
                "readings": readings,
            }
        )
        previous = renders.get(str(row_id), {})
        render_files = [
            graph_path,
            graph_path.with_suffix(".pdf"),
            repeat_path,
            repeat_path.with_suffix(".pdf"),
            folder / f"case_{row_id}_nodes.csv",
            folder / f"case_{row_id}_edges.csv",
            readings_path,
        ]
        reusable = previous.get("key") == render_key and all(
            p.is_file()
            and previous.get("files", {}).get(p.relative_to(folder).as_posix()) == _hash(p)
            for p in render_files
        )
        if not reusable:
            network = plot_interaction_change_graph(
                *arrays,
                feature_names=du.FEATURE_NAMES,
                title_prefix=title,
                scope_label="Targets fixed before validation",
                baseline_label={"a": "Causal", "lag": "Same-day weather"}.get(arm, "Baseline"),
                variant_label={"a": "Marginal", "lag": "Previous-day weather"}.get(arm, arm),
                focus_features=features,
                focus_pairs=pair_names,
            )
            for kind in ("nodes", "edges"):
                network.change_graph_summary[kind].to_csv(
                    folder / f"case_{row_id}_{kind}.csv", index=False
                )
            add_sample_readings(network, readings)
            _save_case_figure(network, graph_path)
            repeated = plot_target_validation(
                focused["validation_details"], reasons, arm=arm, row_id=row_id, title_prefix=title
            )
            add_sample_readings(repeated, readings)
            _save_case_figure(repeated, repeat_path)
            pipeline.write_json(readings_path, readings)
            renders[str(row_id)] = {
                "key": render_key,
                "files": {p.relative_to(folder).as_posix(): _hash(p) for p in render_files},
            }
            pipeline.write_json(render_path, renders)
        summary = focused["validation_summary"].loc[focused["validation_summary"].row_id.eq(row_id)]
        checks = focused["reason_validation"].loc[focused["reason_validation"].row_id.eq(row_id)]
        columns = [
            c
            for c in (
                "order",
                "item",
                "case_rule",
                "target_origin",
                "baseline",
                "modified",
                "delta",
                "material_threshold",
                "endpoint_threshold",
                "eligible_count",
                "cohort_size",
                "selection_reason",
            )
            if c in reasons
        ]
        check_columns = [
            c
            for c in (
                "order",
                "item",
                "case_rule",
                "repeat_check",
                "phenomenon_repeat_count",
                "n_validation_seeds",
            )
            if c in checks
        ]
        summary_columns = [
            c
            for c in (
                "order",
                "item",
                "delta_mean",
                "delta_median",
                "delta_q10",
                "delta_q90",
                "delta_direction_count",
                "sign_flip_count",
                "same_flip_direction_count",
                "material_sign_flip_count",
                "n_seeds",
            )
            if c in summary
        ]
        seed_count = (
            focused["validation_details"]
            .loc[focused["validation_details"].row_id.eq(row_id), "seed"]
            .nunique()
        )
        caption = f"{date} · row {row_id} · {len(reasons)} reasons · " + "; ".join(
            reasons["item"].drop_duplicates()
        )
        display(Markdown("#### " + caption))
        display(
            Markdown(
                f"Discovery seed {data['protocol']['discovery_seed']}; {seed_count} paired validation seeds. "
                "Gold outlines identify the selected targets. The top band gives the original sample's inputs and outcome. "
                "Both figures are shown directly, including unstable results."
            )
        )
        # Native image/png avoids HTML sanitization and closed <details> blocks
        # hiding images in VS Code/Jupyter. It also works on a render-cache hit.
        display(Image(filename=str(graph_path), embed=True))
        display(Image(filename=str(repeat_path), embed=True))
        opened = " open" if expanded_tables else ""
        content = (
            f"<details{opened}><summary>Selection reasons and repeat checks for row {row_id}</summary>"
            f'<div style="overflow:auto">{_table_html(reasons[columns])}'
            f"{_table_html(summary[summary_columns])}{_table_html(checks[check_columns])}</div>"
            "<p>Repeat spread is not a population confidence interval. Opposing changes in singleton and interaction terms "
            "describe this k-SII decomposition, not an identified causal pathway.</p></details>"
        )
        display(HTML(content))
    # Keep bulky audit tables after every case image. Default columns retain
    # every candidate row but omit duplicated cohort/input metadata; the CSV
    # and expanded mode retain all columns.
    display(Markdown("#### Selection audit and complete candidate records"))
    for name, title in (
        ("rules", "Eligibility thresholds, candidate counts and omitted categories"),
        ("selected_unique", "Selected dates (merged across target/rule reasons)"),
        ("reasons", "Every target-specific selection reason"),
        ("candidates", "Complete candidate audit, including ineligible dates"),
        (
            "reason_validation",
            "Independent-seed phenomenon checks (no post-validation reselection)",
        ),
    ):
        frame = focused[name]
        if name == "candidates" and not expanded_tables:
            compact = [
                c
                for c in (
                    "row_id",
                    "date",
                    "order",
                    "item",
                    "case_rule",
                    "baseline",
                    "modified",
                    "delta",
                    "material_threshold",
                    "endpoint_threshold",
                    "tau",
                    "eligible",
                    "selected",
                    "exclusion_reason",
                )
                if c in frame
            ]
            frame = frame[compact]
        opened = " open" if expanded_tables else ""
        relative_csv = (folder / f"{name}.csv").relative_to(Path(root)).as_posix()
        display(
            HTML(
                f"<details{opened}><summary>{escape(title)} ({len(frame)} rows)</summary>"
                f'<p><a href="{escape(relative_csv)}">Full CSV with all columns</a></p>'
                f'<div style="overflow:auto;max-height:650px">{_table_html(frame)}</div></details>'
            )
        )
    relative = folder.relative_to(Path(root)).as_posix()
    display(
        Markdown(
            f"All candidates, selection reasons and repeated values: [{arm} case files]({relative}/). No fixed case-count cap was applied."
        )
    )


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--request", required=True)
    request = json.loads(Path(parser.parse_args().request).read_text(encoding="utf-8"))
    compute_targeted_cases(**request)
