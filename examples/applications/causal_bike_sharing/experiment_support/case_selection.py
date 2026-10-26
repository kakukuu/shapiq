"""Discovery-only, target-driven selection of explanatory examples.

Select a feature or an interaction *before* selecting a date. Every target/rule
has at most one winner, but there is no global cap on representative dates.
All candidates, eligibility conditions and unsuccessful rules are returned.
These are exploratory display rules, not statistical significance tests.
Independent-seed validation must not be used to replace or discard winners.
"""

from __future__ import annotations

from itertools import combinations

import numpy as np
import pandas as pd
from scipy.stats import kendalltau

FEATURE_NAMES = ("trend", "cosyear", "sinyear", "temp", "atemp", "windspeed", "hum")
SELECTION_VERSION = "target-driven-v1"
DEFAULT_CONFIG = dict(
    min_absolute_change=25.0,
    baseline_scale_fraction=0.1,
    rank_threshold=0.9,
    large_change_quantile=0.75,
)
RULES = (
    "largest_target_change",
    "material_sign_flip",
    "rank_preserved_large_change",
    "typical_target_change",
    "opposite_direction",
    "paired_redistribution",
)


def default_target_specs(arm, *, scan_all=False, feature_names=FEATURE_NAMES):
    """Editable, declared research targets; known data-led targets are labelled.

    These defaults were designed after earlier exploratory analyses. They are
    declared before this selection/validation pass, not preregistered hypotheses.
    ``scan_all`` explicitly expands the scope to all singleton and pair terms.
    """
    names = tuple(feature_names)
    pair_labels = [f"{a} × {b}" for a, b in combinations(names, 2)]
    if scan_all:
        return [
            dict(
                order=order,
                item=item,
                origin="exploratory_scan",
                rationale="Explicit all-target exploratory scan.",
            )
            for order, items in ((1, names), (2, pair_labels))
            for item in items
        ]
    weather = ("temp", "hum", "windspeed")
    plans = {
        "a": (weather + ("atemp",), (("temp", "atemp"), ("temp", "hum"))),
        "b1": (weather, (("temp", "hum"),)),
        "b2": (("cosyear", "sinyear", "temp"), (("cosyear", "temp"), ("sinyear", "temp"))),
        "b2_unconfounded": (
            ("cosyear", "sinyear", "temp", "hum"),
            (("cosyear", "temp"), ("temp", "hum")),
        ),
        "b3": (("temp", "atemp"), (("temp", "atemp"),)),
        "b3_unconfounded": (("temp", "atemp", "hum"), (("temp", "atemp"), ("temp", "hum"))),
        "temp_root": (("temp", "atemp"), (("temp", "atemp"), ("cosyear", "temp"))),
        "hum_root": (("hum", "temp"), (("temp", "hum"), ("cosyear", "hum"))),
        "windspeed_root": (
            ("windspeed", "temp"),
            (("temp", "windspeed"), ("cosyear", "windspeed")),
        ),
        "lag": (weather, (("temp", "atemp"), ("temp", "hum"))),
    }
    if arm not in plans:
        raise ValueError(f"No default targets for arm {arm!r}; supply targets explicitly.")
    singles, pairs = plans[arm]
    result = [
        dict(
            order=1,
            item=name,
            origin="research_focus",
            rationale="Feature directly relevant to this comparison's input/structure change.",
        )
        for name in singles
    ]
    result += [
        dict(
            order=2,
            item=f"{a} × {b}",
            origin="research_focus",
            rationale="Interaction involving the comparison's focal feature(s).",
        )
        for a, b in pairs
    ]
    if arm == "b1":
        result += [
            dict(
                order=order,
                item=item,
                origin="previously_observed_exploratory",
                rationale="Season-related pattern noticed in the earlier exploratory results.",
            )
            for order, item in ((1, "cosyear"), (2, "cosyear × temp"))
        ]
    return result


def _sort_ids(frame):
    try:
        return frame.sort_values("row_id", kind="stable")
    except TypeError:
        return (
            frame.assign(_id_sort=frame.row_id.map(str))
            .sort_values("_id_sort", kind="stable")
            .drop(columns="_id_sort")
        )


def _inputs(data, feature_names):
    names = tuple(feature_names)
    if len(names) < 2 or len(set(names)) != len(names):
        raise ValueError("At least two unique feature names are required.")
    ids = np.asarray(data["row_ids"])
    if ids.ndim != 1 or not len(ids) or pd.isna(ids).any() or pd.Index(ids).has_duplicates:
        raise ValueError("row_ids must be nonempty, unique and nonmissing.")
    n, d = len(ids), len(names)
    keys = ("base_phi", "variant_phi", "base_interaction", "variant_interaction")
    arrays = [np.asarray(data[key], dtype=float) for key in keys]
    shapes = ((n, d), (n, d), (n, d * (d - 1) // 2), (n, d * (d - 1) // 2))
    if any(
        a.shape != shape or not np.isfinite(a).all()
        for a, shape in zip(arrays, shapes, strict=True)
    ):
        raise ValueError("Expected finite, consistently paired attribution arrays.")
    points = pd.DataFrame(data.get("points", {"row_id": ids})).copy()
    if "row_id" not in points or not np.array_equal(points.row_id.to_numpy(), ids):
        raise ValueError("Point metadata must follow the exact attribution row_id order.")
    points = points.reset_index(drop=True)
    # Recompute order-specific ranks from the discovery arrays, never from
    # validation or a rounded/reporting table. Undefined rankings remain NaN.
    for order, left, right in ((1, arrays[0], arrays[1]), (2, arrays[2], arrays[3])):
        points[f"tau_{'first' if order == 1 else 'second'}_signed"] = [
            float(kendalltau(a, b).statistic) if np.ptp(a) and np.ptp(b) else np.nan
            for a, b in zip(left, right, strict=True)
        ]
    points["row_position"] = np.arange(n)
    points["arm"] = str(data["arm"])
    points["variant"] = str(data["arm"])
    return names, ids, arrays, points


def _specs(targets, names):
    labels = {1: list(names), 2: [f"{a} × {b}" for a, b in combinations(names, 2)]}
    raw = targets.to_dict("records") if isinstance(targets, pd.DataFrame) else list(targets)
    if not raw:
        raise ValueError("Declare at least one target.")
    specs, seen = [], set()
    for item in raw:
        spec = dict(item)
        order, label = spec.get("order"), spec.get("item")
        if order not in (1, 2) or label not in labels[order]:
            raise ValueError(f"Unknown target: order={order!r}, item={label!r}.")
        if (order, label) in seen:
            raise ValueError(f"Duplicate target: {(order, label)!r}.")
        seen.add((order, label))
        specs.append(
            dict(
                order=int(order),
                item=label,
                origin=str(spec.get("origin", "user_declared")),
                rationale=str(spec.get("rationale", "Notebook-declared target.")),
                column_index=labels[order].index(label),
            )
        )
    return pd.DataFrame(specs), labels


def select_target_cases(
    data,
    targets=None,
    *,
    feature_names=FEATURE_NAMES,
    min_absolute_change=25.0,
    baseline_scale_fraction=0.1,
    rank_threshold=0.9,
    large_change_quantile=0.75,
):
    """Return target specs, all candidates, rule accounting and merged cases.

    ``data`` is the verified discovery dictionary returned by ``load_experiment``;
    only its arm, row_ids, attribution arrays and point metadata are consumed.
    The materiality threshold of each target is
    ``max(min_absolute_change, baseline_scale_fraction * median(abs(baseline)))``.
    It is an editable practical display threshold, not a significance cutoff.

    Strong, rank-preserved, opposite-direction and redistribution cases must
    exceed this threshold; a sign reversal requires BOTH endpoint magnitudes
    to meet it. Typical cases are closest to the signed target-delta median,
    including small/no-change controls. Every exact ranking tie uses ascending
    original row_id. Related singleton thresholds are computed separately.

    There is no top-N truncation. A date selected for several targets/rules is
    displayed once in ``selected_unique``; every reason survives in ``reasons``.
    """
    params = (min_absolute_change, baseline_scale_fraction, rank_threshold, large_change_quantile)
    if not all(np.isfinite(v) for v in params):
        raise ValueError("Selection thresholds must be finite.")
    if min_absolute_change <= 0 or baseline_scale_fraction < 0:
        raise ValueError(
            "Materiality requires a positive absolute floor and a nonnegative scale fraction."
        )
    if not -1 <= rank_threshold <= 1 or not 0 <= large_change_quantile <= 1:
        raise ValueError("Rank threshold must be in [-1, 1] and quantile in [0, 1].")
    names, ids, arrays, points = _inputs(data, feature_names)
    arm = str(data["arm"])
    specs, labels = _specs(
        default_target_specs(arm, feature_names=names) if targets is None else targets, names
    )
    base_phi, variant_phi, base_int, variant_int = arrays
    phi_delta = variant_phi - base_phi
    singleton_thresholds = np.maximum(
        min_absolute_change, baseline_scale_fraction * np.median(np.abs(base_phi), axis=0)
    )
    pair_indices = list(combinations(range(len(names)), 2))
    candidates, rule_rows, reasons = [], [], []
    for spec in specs.to_dict("records"):
        order, label, column = spec["order"], spec["item"], spec["column_index"]
        left, right = (base_phi, variant_phi) if order == 1 else (base_int, variant_int)
        baseline, modified = left[:, column], right[:, column]
        delta = modified - baseline
        absolute = np.abs(delta)
        median = float(np.median(delta))
        baseline_median = float(np.median(np.abs(baseline)))
        material = max(float(min_absolute_change), baseline_scale_fraction * baseline_median)
        q75 = float(np.quantile(absolute, large_change_quantile))
        rank_metric = "tau_first_signed" if order == 1 else "tau_second_signed"
        frame = points.copy()
        frame["case_id"] = [f"{arm}:{row_id}" for row_id in ids]
        fields = dict(
            order=order,
            item=label,
            target_id=f"{order}:{label}",
            target_origin=spec["origin"],
            target_rationale=spec["rationale"],
            material_threshold=material,
            baseline_median_abs=baseline_median,
            endpoint_threshold=material,
            min_absolute_change=min_absolute_change,
            baseline_scale_fraction=baseline_scale_fraction,
            cohort_median_delta=median,
            target_abs_delta_quantile=q75,
            large_change_quantile=large_change_quantile,
            rank_threshold=rank_threshold,
            rank_metric=rank_metric,
            selection_version=SELECTION_VERSION,
        )
        for key, value in fields.items():
            frame[key] = value
        frame["baseline"], frame["modified"] = baseline, modified
        frame["delta"], frame["abs_delta"] = delta, absolute
        frame["tau"] = frame[rank_metric]
        frame["typical_distance"] = np.abs(delta - median)
        frame["material_change"] = absolute >= material
        frame["sign_flip"] = baseline * modified < 0
        frame["material_sign_flip"] = (
            frame.sign_flip & (np.abs(baseline) >= material) & (np.abs(modified) >= material)
        )
        frame["linked_singletons"] = ""
        frame["linked_opposite_count"] = 0
        frame["linked_opposite_abs_delta_sum"] = 0.0
        if order == 2:
            i, j = pair_indices[column]
            linked = phi_delta[:, [i, j]] * delta[:, None] < 0
            linked &= np.abs(phi_delta[:, [i, j]]) >= singleton_thresholds[[i, j]]
            frame["linked_singletons"] = [
                " | ".join(names[k] for present, k in zip(row, (i, j), strict=True) if present)
                for row in linked
            ]
            frame["linked_opposite_count"] = linked.sum(axis=1)
            frame["linked_opposite_abs_delta_sum"] = (np.abs(phi_delta[:, [i, j]]) * linked).sum(
                axis=1
            )
            for suffix, index in (("a", i), ("b", j)):
                frame[f"related_feature_{suffix}"] = names[index]
                frame[f"related_delta_{suffix}"] = phi_delta[:, index]
                frame[f"related_threshold_{suffix}"] = singleton_thresholds[index]
        frame = _sort_ids(frame)
        descriptions = {
            "largest_target_change": "Material target change; choose the largest absolute target delta.",
            "material_sign_flip": "Opposite endpoint signs, both magnitudes at least the target threshold; choose largest absolute delta.",
            "rank_preserved_large_change": "Same-order signed tau meets the rank threshold; target absolute delta meets both materiality and its cohort quantile; choose largest absolute delta.",
            "typical_target_change": "Choose the target delta nearest its cohort signed median; small or zero changes are valid controls.",
            "opposite_direction": "Material target delta opposite to the nonzero cohort signed median; choose largest absolute delta.",
            "paired_redistribution": "Material interaction change with at least one member's material singleton delta in the opposite direction; choose largest interaction absolute delta. Descriptive co-movement, not causal-path evidence.",
        }
        eligible_masks = {
            "largest_target_change": frame.material_change,
            "material_sign_flip": frame.material_sign_flip,
            "rank_preserved_large_change": (
                frame.material_change & (frame.tau >= rank_threshold) & (frame.abs_delta >= q75)
            ),
            "typical_target_change": pd.Series(True, index=frame.index),
            "opposite_direction": frame.material_change & (frame.delta * median < 0),
            "paired_redistribution": frame.material_change & (frame.linked_opposite_count >= 1),
        }
        for rule in RULES[:5] if order == 1 else RULES:
            current = frame.copy()
            current["case_rule"] = rule
            current["eligible"] = eligible_masks[rule]
            current["rule_description"] = descriptions[rule]
            current["tie_break"] = "ascending original row_id"
            current["cohort_size"] = len(ids)
            current["eligible_count"] = int(current.eligible.sum())
            current["selection_criterion"] = (
                current.typical_distance if rule == "typical_target_change" else current.abs_delta
            )
            current["criterion_direction"] = (
                "minimum" if rule == "typical_target_change" else "maximum"
            )
            current["candidate_rank"] = np.nan
            current["selected"] = False
            eligible = current[current.eligible].sort_values(
                "selection_criterion", ascending=rule == "typical_target_change", kind="stable"
            )
            current.loc[eligible.index, "candidate_rank"] = np.arange(1, len(eligible) + 1)
            # Every rejected row carries the exact failed conditions, allowing
            # readers to audit omissions without inferring them from a plot.
            exclusions = []
            for row in current.itertuples():
                failed = []
                if (
                    rule
                    in (
                        "largest_target_change",
                        "rank_preserved_large_change",
                        "opposite_direction",
                        "paired_redistribution",
                    )
                    and not row.material_change
                ):
                    failed.append("target change below materiality threshold")
                if rule == "material_sign_flip" and not row.material_sign_flip:
                    failed.append("no sign reversal with both endpoints above threshold")
                if rule == "rank_preserved_large_change":
                    if not np.isfinite(row.tau) or row.tau < rank_threshold:
                        failed.append("same-order signed tau undefined or below threshold")
                    if row.abs_delta < q75:
                        failed.append("target absolute delta below cohort quantile")
                if rule == "opposite_direction" and not row.delta * median < 0:
                    failed.append(
                        "cohort median is zero"
                        if median == 0
                        else "delta not opposite to cohort median"
                    )
                if rule == "paired_redistribution" and row.linked_opposite_count < 1:
                    failed.append("no material opposite change in either member singleton")
                exclusions.append("; ".join(failed))
            current["exclusion_reason"] = exclusions
            availability = dict(
                arm=arm,
                variant=arm,
                **fields,
                case_rule=rule,
                eligible_count=len(eligible),
                cohort_size=len(ids),
                rule_description=descriptions[rule],
                tie_break="ascending original row_id",
                status="selected" if len(eligible) else "omitted",
                selected_row_id=None,
                reason="" if len(eligible) else "No discovery row meets all stated conditions.",
            )
            if len(eligible):
                chosen_index = eligible.index[0]
                current.loc[chosen_index, "selected"] = True
                winner = current.loc[chosen_index].to_dict()
                winner["selection_reason"] = (
                    f"{label}: {descriptions[rule]} "
                    f"Selected row {winner['row_id']} from {len(eligible)}/{len(ids)} eligible dates; "
                    f"delta={winner['delta']:.6g}, threshold={material:.6g}."
                )
                reasons.append(winner)
                availability["selected_row_id"] = winner["row_id"]
            rule_rows.append(availability)
            candidates.append(current)
    reason_table = pd.DataFrame(reasons)
    merged = []
    for (_, row_id), rows in reason_table.groupby(["arm", "row_id"], sort=False):
        # Preserve common point metadata while the target-specific quantities
        # stay in the long reason table instead of silently retaining one target.
        point = points.loc[points.row_id == row_id].iloc[0].to_dict()
        point.update(
            case_id=f"{arm}:{row_id}",
            reason_count=len(rows),
            target_count=len(rows.target_id.unique()),
            target_items=" | ".join(dict.fromkeys(rows.item)),
            case_rules=" | ".join(dict.fromkeys(rows.case_rule)),
            focus_features=" | ".join(dict.fromkeys(rows.loc[rows.order == 1, "item"])),
            focus_interactions=" | ".join(dict.fromkeys(rows.loc[rows.order == 2, "item"])),
        )
        merged.append(point)
    specs["arm"] = arm
    specs["selection_version"] = SELECTION_VERSION
    return dict(
        selected_unique=_sort_ids(pd.DataFrame(merged)).reset_index(drop=True),
        reasons=reason_table.reset_index(drop=True),
        candidates=pd.concat(candidates, ignore_index=True),
        rules=pd.DataFrame(rule_rows),
        target_specs=specs,
    )
