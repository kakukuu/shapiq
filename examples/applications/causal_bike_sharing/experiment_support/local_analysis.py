"""Descriptive sample-level analysis for paired attribution experiments.

Interaction columns must follow ``combinations(range(n_features), 2)``.
These are output SII interactions, not edges of the input causal structure.
No sample-selection rule or plot in this module constitutes a significance test.
"""

from __future__ import annotations

from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import kendalltau


def _validate_arrays(
    baseline_phi, variant_phi, baseline_interactions, variant_interactions, feature_names
):
    arrays = [
        np.asarray(a, dtype=float)
        for a in (baseline_phi, variant_phi, baseline_interactions, variant_interactions)
    ]
    d = len(feature_names)
    if d < 2 or len(set(feature_names)) != d:
        raise ValueError("At least two unique feature names are required.")
    if arrays[0].ndim != 2 or arrays[0].shape[1] != d:
        raise ValueError("First-order arrays must have shape (n_samples, n_features).")
    n = len(arrays[0])
    if n == 0:
        raise ValueError("At least one paired sample is required.")
    expected = [(n, d), (n, d), (n, d * (d - 1) // 2), (n, d * (d - 1) // 2)]
    if any(a.shape != shape for a, shape in zip(arrays, expected, strict=True)):
        raise ValueError("Paired arrays have inconsistent sample or interaction dimensions.")
    if any(not np.isfinite(a).all() for a in arrays):
        raise ValueError("Attributions must be finite; resolve missing samples before pairing.")
    return arrays


def _tau_rows(left, right):
    # A constant vector has no ranking. Preserve NaN instead of inventing agreement.
    return np.asarray(
        [
            float(kendalltau(a, b).statistic) if np.ptp(a) > 0 and np.ptp(b) > 0 else np.nan
            for a, b in zip(left, right, strict=True)
        ]
    )


def _top_indices(values, k):
    # Stable ties are resolved by the fixed, documented feature/pair column order.
    return np.argsort(-np.abs(values), kind="stable")[:k]


def _reference_scale(values):
    """One common scale per order, derived exclusively from baseline samples."""
    l1 = np.abs(values).sum(axis=1)
    median = float(np.median(l1))
    # The all-zero baseline has no meaningful relative scale. A one-unit fallback
    # is explicit in the exported summary, and avoids silently dividing by zero.
    if median > 0:
        return median, "median_baseline_L1"
    mean = float(np.mean(l1))
    return (
        (mean, "mean_baseline_L1_zero_median") if mean > 0 else (1.0, "unit_fallback_zero_baseline")
    )


def _detail_table(base, variant, labels, row_ids, order, variant_name):
    n, width = base.shape
    return pd.DataFrame(
        {
            "variant": variant_name,
            "row_id": np.repeat(row_ids, width),
            "row_position": np.repeat(np.arange(n), width),
            "order": order,
            "item": np.tile(labels, n),
            "baseline": base.reshape(-1),
            "modified": variant.reshape(-1),
            "delta": (variant - base).reshape(-1),
            "abs_delta": np.abs(variant - base).reshape(-1),
            "magnitude_change": (np.abs(variant) - np.abs(base)).reshape(-1),
            # Moving to or from exactly zero is reported separately from reversal.
            "sign_flip": ((base * variant) < 0).reshape(-1),
            "zero_transition": ((base == 0) != (variant == 0)).reshape(-1),
        }
    )


def _item_summary(base, variant, labels, order, variant_name):
    delta = variant - base
    absolute = np.abs(delta)
    return pd.DataFrame(
        {
            "variant": variant_name,
            "order": order,
            "item": labels,
            "n_samples": len(base),
            "baseline_mean": base.mean(axis=0),
            "modified_mean": variant.mean(axis=0),
            "signed_mean_delta": delta.mean(axis=0),
            "mean_abs_delta": absolute.mean(axis=0),
            "mean_magnitude_change": (np.abs(variant) - np.abs(base)).mean(axis=0),
            "median_delta": np.median(delta, axis=0),
            "delta_q10": np.quantile(delta, 0.10, axis=0),
            "delta_q90": np.quantile(delta, 0.90, axis=0),
            "median_abs_delta": np.median(absolute, axis=0),
            "abs_delta_q10": np.quantile(absolute, 0.10, axis=0),
            "abs_delta_q90": np.quantile(absolute, 0.90, axis=0),
            "sign_flip_fraction": (base * variant < 0).mean(axis=0),
        }
    )


def build_local_tables(
    baseline_phi,
    variant_phi,
    baseline_interactions,
    variant_interactions,
    *,
    feature_names,
    row_ids,
    context=None,
    variant_name="variant",
    top_k=5,
):
    """Return complete point/detail/summary tables for exactly paired rows.

    ``context`` must already follow the supplied array row order. It is copied
    positionally, not joined on an assumed pandas index. Caller is responsible
    for matching dates before computing paired arrays. IDs must be unique.

    ``effect_score`` is the average of first/second-order L1 shifts divided by
    their respective median baseline L1 magnitudes across this sample cohort.
    It is a descriptive ranking score, not a test statistic or percentage.
    """
    base_phi, pert_phi, base_int, pert_int = _validate_arrays(
        baseline_phi, variant_phi, baseline_interactions, variant_interactions, feature_names
    )
    ids = np.asarray(list(row_ids))
    n = len(base_phi)
    if len(ids) != n or pd.Index(ids).has_duplicates or pd.isna(ids).any():
        raise ValueError("row_ids must contain one unique, nonmissing ID per paired row.")
    pairs = list(combinations(range(len(feature_names)), 2))
    pair_labels = [f"{feature_names[i]} × {feature_names[j]}" for i, j in pairs]
    if not isinstance(top_k, (int, np.integer)) or top_k < 1:
        raise ValueError("top_k must be a positive integer.")
    k = min(int(top_k), len(pairs))
    phi_delta, int_delta = pert_phi - base_phi, pert_int - base_int
    phi_scale, phi_scale_method = _reference_scale(base_phi)
    int_scale, int_scale_method = _reference_scale(base_int)
    points = pd.DataFrame(
        {
            "variant": variant_name,
            "row_id": ids,
            "row_position": np.arange(n),
            "tau_first_signed": _tau_rows(base_phi, pert_phi),
            "tau_first_absolute": _tau_rows(np.abs(base_phi), np.abs(pert_phi)),
            "tau_second_signed": _tau_rows(base_int, pert_int),
            "tau_second_absolute": _tau_rows(np.abs(base_int), np.abs(pert_int)),
            "first_l1_delta": np.abs(phi_delta).sum(axis=1),
            "second_l1_delta": np.abs(int_delta).sum(axis=1),
            "first_mean_abs_delta": np.abs(phi_delta).mean(axis=1),
            "second_mean_abs_delta": np.abs(int_delta).mean(axis=1),
            "first_signed_delta_sum": phi_delta.sum(axis=1),
            "second_signed_delta_sum": int_delta.sum(axis=1),
            "first_magnitude_change_sum": (np.abs(pert_phi) - np.abs(base_phi)).sum(axis=1),
            "second_magnitude_change_sum": (np.abs(pert_int) - np.abs(base_int)).sum(axis=1),
            "first_sign_flips": (base_phi * pert_phi < 0).sum(axis=1),
            "second_sign_flips": (base_int * pert_int < 0).sum(axis=1),
            "first_zero_transitions": ((base_phi == 0) != (pert_phi == 0)).sum(axis=1),
            "second_zero_transitions": ((base_int == 0) != (pert_int == 0)).sum(axis=1),
            "largest_feature_delta": [feature_names[i] for i in np.abs(phi_delta).argmax(axis=1)],
            "largest_feature_delta_value": phi_delta[
                np.arange(n), np.abs(phi_delta).argmax(axis=1)
            ],
            "largest_interaction_delta": [pair_labels[i] for i in np.abs(int_delta).argmax(axis=1)],
            "largest_interaction_delta_value": int_delta[
                np.arange(n), np.abs(int_delta).argmax(axis=1)
            ],
            "top_k_interactions": k,
            "top_k_interaction_overlap": [
                len(set(_top_indices(a, k)) & set(_top_indices(b, k)))
                for a, b in zip(base_int, pert_int, strict=True)
            ],
        }
    )
    points["top_k_interaction_overlap_fraction"] = points["top_k_interaction_overlap"] / k
    points["effect_score"] = 0.5 * (
        points["first_l1_delta"] / phi_scale + points["second_l1_delta"] / int_scale
    )
    for metric in ("effect_score", "first_l1_delta", "second_l1_delta"):
        points[f"{metric}_percentile"] = points[metric].rank(method="average", pct=True) * 100
    if context is not None:
        ctx = pd.DataFrame(context).reset_index(drop=True)
        if len(ctx) != n:
            raise ValueError("Context length must match paired samples.")
        for column in ctx:
            if column == "row_id":
                if not np.array_equal(ctx[column].to_numpy(), ids):
                    raise ValueError("Context row_id order does not match paired arrays.")
            elif column in set(points.columns) | {
                "order",
                "item",
                "baseline",
                "modified",
                "delta",
                "abs_delta",
                "magnitude_change",
                "sign_flip",
                "zero_transition",
            }:
                raise ValueError(f"Context column shadows a metric: {column}")
            else:
                points[column] = ctx[column].to_numpy()
    feature_details = _detail_table(base_phi, pert_phi, list(feature_names), ids, 1, variant_name)
    interaction_details = _detail_table(base_int, pert_int, pair_labels, ids, 2, variant_name)
    # Include identity/context in detail CSVs as well as the compact point table.
    if context is not None:
        context_columns = [c for c in ctx if c != "row_id"]
        for details in (feature_details, interaction_details):
            for column in context_columns:
                details[column] = ctx[column].to_numpy()[details["row_position"].to_numpy()]
    summary = pd.concat(
        [
            _item_summary(base_phi, pert_phi, list(feature_names), 1, variant_name),
            _item_summary(base_int, pert_int, pair_labels, 2, variant_name),
        ],
        ignore_index=True,
    )
    global_rows = []
    for metric in (
        "first_l1_delta",
        "second_l1_delta",
        "effect_score",
        "tau_first_signed",
        "tau_first_absolute",
        "tau_second_signed",
        "tau_second_absolute",
        "top_k_interaction_overlap_fraction",
        "first_sign_flips",
        "second_sign_flips",
    ):
        values = points[metric].dropna()
        global_rows.append(
            {
                "variant": variant_name,
                "metric": metric,
                "n_samples": n,
                "n_defined": len(values),
                "mean": values.mean(),
                "median": values.median(),
                "q10": values.quantile(0.1),
                "q25": values.quantile(0.25),
                "q75": values.quantile(0.75),
                "q90": values.quantile(0.9),
                "min": values.min(),
                "max": values.max(),
            }
        )
    scales = pd.DataFrame(
        [
            {"order": 1, "baseline_reference_L1": phi_scale, "method": phi_scale_method},
            {"order": 2, "baseline_reference_L1": int_scale, "method": int_scale_method},
        ]
    )
    return {
        "points": points,
        "feature_details": feature_details,
        "interaction_details": interaction_details,
        "summary": summary,
        "global_summary": pd.DataFrame(global_rows),
        "score_scales": scales,
    }


def select_cases(points):
    """Select fixed descriptive cases; return (selected, rule_availability).

    High-rank cases require signed Kendall tau >= 0.9 for BOTH orders and a
    score >= the cohort's 75th percentile. They do not prove equivalence.
    Rules may select the same row; duplication is reported, not silently filled
    with a different type of case. Ties are broken by ascending original row ID.
    """
    if len(points) == 0:
        raise ValueError("Cannot select cases from an empty table.")
    cohort = points.copy()
    try:
        cohort = cohort.sort_values("row_id", kind="stable")
    except TypeError:
        cohort = cohort.assign(_id_sort=cohort["row_id"].map(str)).sort_values(
            "_id_sort", kind="stable"
        )
    median, q75 = cohort["effect_score"].quantile([0.5, 0.75])
    high = cohort[
        (cohort["tau_first_signed"] >= 0.9)
        & (cohort["tau_second_signed"] >= 0.9)
        & (cohort["effect_score"] >= q75)
        & (cohort["effect_score"] > 0)
    ]
    rules = [
        (
            "typical",
            cohort,
            (cohort["effect_score"] - median).abs(),
            True,
            "Nearest to cohort median effect_score; ascending row_id resolves ties.",
        ),
        (
            "strong",
            cohort,
            cohort["effect_score"],
            False,
            "Maximum effect_score; ascending row_id resolves ties.",
        ),
        (
            "high_rank_large_shift",
            high,
            high["effect_score"],
            False,
            "Both signed taus >= 0.9, positive effect_score >= cohort q75; choose maximum score.",
        ),
        (
            "weak",
            cohort,
            cohort["effect_score"],
            True,
            "Minimum effect_score; ascending row_id resolves ties.",
        ),
    ]
    chosen, availability, previously_chosen = [], [], set()
    for rule, eligible, criterion, ascending, description in rules:
        availability.append(
            {
                "case_rule": rule,
                "eligible_count": len(eligible),
                "cohort_size": len(cohort),
                "effect_score_median": median,
                "effect_score_q75": q75,
                "rule_description": description,
                "status": "selected" if len(eligible) else "omitted",
                "reason": ""
                if len(eligible)
                else "No row meets both rank and large-shift thresholds.",
            }
        )
        if not len(eligible):
            continue
        ranked = eligible.assign(_criterion=criterion).sort_values(
            "_criterion", ascending=ascending, kind="stable"
        )
        result = ranked.iloc[0].drop(labels=["_criterion", "_id_sort"], errors="ignore").to_dict()
        result.update(
            case_rule=rule,
            eligible_count=len(eligible),
            cohort_size=len(cohort),
            effect_score_q75=q75,
            selection_rule=description,
            duplicates_previous_case=result["row_id"] in previously_chosen,
        )
        previously_chosen.add(result["row_id"])
        chosen.append(result)
    return pd.DataFrame(chosen), pd.DataFrame(availability)


def print_full_tables(tables, *, include_details=True):
    """Print every row/column without notebook display truncation."""
    keys = ["points", "summary", "global_summary"]
    if include_details:
        keys += ["feature_details", "interaction_details"]
    for key in keys:
        print(f"\n{key} ({len(tables[key])} rows)\n")
        print(
            tables[key].to_string(
                index=False, max_rows=None, max_cols=None, float_format=lambda x: f"{x:.6g}"
            )
        )


def export_tables(tables, directory):
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    paths = {}
    for key, value in tables.items():
        path = directory / f"{key}.csv"
        value.to_csv(path, index=False, encoding="utf-8-sig")
        paths[key] = path
    return paths


def _save_figure(fig, output_path):
    if output_path is not None:
        path = Path(output_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        stem = path.with_suffix("")
        fig.savefig(stem.parent / f"{stem.name}.png", dpi=180, bbox_inches="tight")
        fig.savefig(stem.parent / f"{stem.name}.pdf", bbox_inches="tight")
    return fig


def plot_aggregate_overview(tables_by_variant, output_path=None):
    """Six panels distinguish direction, displacement, and magnitude change."""
    import matplotlib.pyplot as plt

    names = list(tables_by_variant)
    if not names:
        raise ValueError("At least one variant is required.")
    fig, axes = plt.subplots(
        3,
        2,
        figsize=(22, max(9, len(names) * 1.1)),
        gridspec_kw={"width_ratios": [1, 2.4]},
        constrained_layout=True,
    )
    metrics = [
        ("signed_mean_delta", "Mean signed change", "RdBu_r"),
        ("mean_abs_delta", "Mean absolute change", "YlOrRd"),
        ("mean_magnitude_change", "Mean change in attribution magnitude", "RdBu_r"),
    ]
    for row, (metric, title, cmap) in enumerate(metrics):
        for col, order in enumerate((1, 2)):
            records = []
            for name in names:
                summary = tables_by_variant[name]["summary"]
                records.append(summary.loc[summary["order"] == order].set_index("item")[metric])
            matrix = pd.DataFrame(records, index=names)
            if matrix.isna().any().any():
                raise ValueError("Variants must share item labels for aggregate comparison.")
            ax = axes[row, col]
            maximum = max(float(np.abs(matrix.to_numpy()).max()), 1e-12)
            im = ax.imshow(
                matrix,
                aspect="auto",
                cmap=cmap,
                vmin=0 if metric == "mean_abs_delta" else -maximum,
                vmax=maximum,
            )
            ax.set_xticks(
                range(len(matrix.columns)), matrix.columns, rotation=55, ha="right", fontsize=8
            )
            ax.set_yticks(range(len(names)), names, fontsize=9)
            ax.set_title(
                f"{title}: {'first-order attribution' if order == 1 else 'SII interaction'}"
            )
            fig.colorbar(im, ax=ax, shrink=0.8, label="Attribution units")
    fig.suptitle(
        "Paired attribution-comparison summaries · output interactions are not input causal edges",
        fontsize=14,
    )
    return _save_figure(fig, output_path)


def plot_sample_overview(tables_by_variant, output_path=None):
    """Complete sample-by-variant score heatmap, sorted by original row ID."""
    import matplotlib.pyplot as plt

    if not tables_by_variant:
        raise ValueError("At least one variant is required.")
    records = {
        name: tables["points"].set_index("row_id")["effect_score"]
        for name, tables in tables_by_variant.items()
    }
    matrix = pd.DataFrame(records).sort_index()
    if matrix.isna().any().any():
        raise ValueError("Sample overview requires the same matched row IDs for every variant.")
    # A common heatmap scale requires common baseline scale definitions.
    reference = next(iter(tables_by_variant.values()))["score_scales"][
        "baseline_reference_L1"
    ].to_numpy()
    if any(
        not np.allclose(t["score_scales"]["baseline_reference_L1"], reference)
        for t in tables_by_variant.values()
    ):
        raise ValueError("Sample overview requires common baseline normalization across variants.")
    fig, ax = plt.subplots(
        figsize=(max(8, len(matrix.columns) * 1.1), max(6, len(matrix) * 0.115)),
        constrained_layout=True,
    )
    im = ax.imshow(matrix, aspect="auto", cmap="YlOrRd", vmin=0)
    ax.set_xticks(range(len(matrix.columns)), matrix.columns, rotation=45, ha="right")
    first_points = next(iter(tables_by_variant.values()))["points"].set_index("row_id")
    labels = [
        f"{row_id} | {str(first_points.loc[row_id, 'date'])[:10]}"
        if "date" in first_points
        else str(row_id)
        for row_id in matrix.index
    ]
    ax.set_yticks(range(len(matrix)), labels, fontsize=6)
    ax.set_ylabel("Held-out sample ID | target date (when available)")
    ax.set_title(
        "Sample sensitivity by structural variant\nMean of baseline-normalized first/second-order L1 shifts"
    )
    fig.colorbar(im, ax=ax, shrink=0.55, label="Descriptive effect score")
    return _save_figure(fig, output_path)


def plot_case_comparison(
    baseline_phi,
    variant_phi,
    baseline_interactions,
    variant_interactions,
    *,
    feature_names,
    case_label,
    variant_name="variant",
    output_path=None,
):
    """One row: baseline, variant, and paired differences on comparable scales."""
    import matplotlib.pyplot as plt

    arrays = _validate_arrays(
        *[
            np.asarray(a)[None, :]
            for a in (baseline_phi, variant_phi, baseline_interactions, variant_interactions)
        ],
        feature_names,
    )
    phi0, phi1, int0, int1 = [a[0] for a in arrays]
    d = len(feature_names)
    pairs = list(combinations(range(d), 2))
    fig, axes = plt.subplots(2, 3, figsize=(17, 10), constrained_layout=True)
    phi_limit = max(float(np.max(np.abs(np.r_[phi0, phi1]))), 1e-12) * 1.08
    int_limit = max(float(np.max(np.abs(np.r_[int0, int1]))), 1e-12)
    for col, (phi, ints, label) in enumerate(
        [
            (phi0, int0, "Baseline"),
            (phi1, int1, variant_name),
            (phi1 - phi0, int1 - int0, "Modified − baseline"),
        ]
    ):
        limit_phi = phi_limit if col < 2 else max(float(np.abs(phi).max()), 1e-12) * 1.08
        ax = axes[0, col]
        ax.barh(range(d), phi, color=np.where(phi >= 0, "#bb3d3d", "#346ea8"))
        ax.set_yticks(range(d), feature_names)
        ax.invert_yaxis()
        ax.set_xlim(-limit_phi, limit_phi)
        ax.axvline(0, color="0.45", linewidth=0.7)
        ax.set_xlabel("First-order attribution")
        ax.set_title(label)
        mat = np.full((d, d), np.nan)
        for value, (i, j) in zip(ints, pairs, strict=True):
            mat[i, j] = mat[j, i] = value
        limit_int = int_limit if col < 2 else max(float(np.abs(ints).max()), 1e-12)
        ax = axes[1, col]
        im = ax.imshow(mat, cmap="RdBu_r", vmin=-limit_int, vmax=limit_int)
        ax.set_xticks(range(d), feature_names, rotation=45, ha="right")
        ax.set_yticks(range(d), feature_names)
        for i in range(d):
            for j in range(d):
                if i != j:
                    ax.text(
                        j,
                        i,
                        f"{mat[i, j]:.1f}",
                        ha="center",
                        va="center",
                        fontsize=7,
                        color="white" if abs(mat[i, j]) > limit_int * 0.65 else "black",
                    )
        ax.set_title("Output pairwise SII interactions")
        fig.colorbar(im, ax=ax, shrink=0.65)
    fig.suptitle(
        f"{case_label}\nBaseline/modified panels share scales; difference panels use symmetric scales.\n"
        "SII interactions do not represent input causal edges.",
        fontsize=12,
    )
    return _save_figure(fig, output_path)
