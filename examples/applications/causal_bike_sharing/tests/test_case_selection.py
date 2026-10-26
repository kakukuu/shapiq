"""Selection criteria, anti-cherry-picking accounting and data alignment."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from case_selection import default_target_specs, select_target_cases

NAMES = ("a", "b", "c")


def discovery(delta, *, baseline=None, pair_delta=None, ids=None):
    delta = np.asarray(delta, dtype=float)
    n = len(delta)
    base = (
        np.tile([100.0, 1000.0, 10000.0], (n, 1))
        if baseline is None
        else np.asarray(baseline, dtype=float)
    )
    pairs = np.tile([100.0, 200.0, 300.0], (n, 1))
    ids = np.arange(n) if ids is None else np.asarray(ids)
    return dict(
        arm="test",
        row_ids=ids,
        base_phi=base,
        variant_phi=base + delta,
        base_interaction=pairs,
        variant_interaction=pairs + (0 if pair_delta is None else np.asarray(pair_delta)),
        points=pd.DataFrame({"row_id": ids, "date": [f"date-{i}" for i in ids]}),
    )


def select(data, *targets, **kwargs):
    return select_target_cases(
        data,
        [dict(order=order, item=item) for order, item in targets],
        feature_names=NAMES,
        **kwargs,
    )


def rule_row(result, name, item="a"):
    return result["rules"].query("case_rule == @name and item == @item").iloc[0]


def test_same_order_rank_condition_does_not_require_other_order():
    data = discovery(
        [[300, 0, 0], [30, 0, 0], [0, 0, 0]], pair_delta=[[200, 0, -200], [0, 0, 0], [0, 0, 0]]
    )
    result = select(data, (1, "a"))
    row = result["reasons"].query("case_rule == 'rank_preserved_large_change'").iloc[0]
    assert row.row_id == 0
    assert row.tau_first_signed == 1
    assert row.tau_second_signed == -1


def test_sign_reversal_requires_material_endpoints_not_just_large_delta():
    data = discovery([[-110, 0, 0], [-150, 0, 0], [50, 0, 0]])
    result = select(data, (1, "a"))
    rule = rule_row(result, "material_sign_flip")
    assert rule.eligible_count == 1
    assert rule.selected_row_id == 1
    assert rule.endpoint_threshold == 25
    rejected = (
        result["candidates"].query("case_rule == 'material_sign_flip' and row_id == 0").iloc[0]
    )
    assert rejected.sign_flip and not rejected.eligible
    assert rejected.exclusion_reason


def test_signed_median_typical_and_material_opposite_direction():
    data = discovery([[-100, 0, 0], [0, 0, 0], [10, 0, 0], [100, 0, 0], [200, 0, 0]])
    result = select(data, (1, "a"))
    assert rule_row(result, "typical_target_change").selected_row_id == 2
    assert rule_row(result, "opposite_direction").selected_row_id == 0
    typical = result["reasons"].query("case_rule == 'typical_target_change'").iloc[0]
    assert not typical.material_change


def test_zero_median_has_no_fabricated_opposite_direction():
    result = select(discovery([[-100, 0, 0], [0, 0, 0], [100, 0, 0]]), (1, "a"))
    rule = rule_row(result, "opposite_direction")
    assert rule.status == "omitted" and rule.eligible_count == 0
    rows = result["candidates"].query("case_rule == 'opposite_direction'")
    assert rows.exclusion_reason.str.contains("median is zero").all()


def test_no_change_retains_only_typical_control_and_reports_omissions():
    result = select(discovery([[0, 0, 0], [0, 0, 0]], ids=[20, 10]), (1, "a"))
    assert result["reasons"].case_rule.tolist() == ["typical_target_change"]
    assert result["selected_unique"].row_id.tolist() == [10]
    assert (result["rules"].status == "omitted").sum() == 4
    assert len(result["candidates"]) == 10


def test_exact_ties_resolve_by_id_and_shared_winner_keeps_all_reasons():
    data = discovery([[-200, 0, 0], [-200, 0, 0], [-200, 0, 0]], ids=[30, 10, 20])
    result = select(data, (1, "a"))
    assert set(result["reasons"].row_id) == {10}
    assert len(result["selected_unique"]) == 1
    assert result["selected_unique"].iloc[0].reason_count == len(result["reasons"])
    eligible = result["candidates"].query("case_rule == 'largest_target_change'")
    assert eligible.row_id.tolist() == [10, 20, 30]
    assert eligible.candidate_rank.tolist() == [1, 2, 3]


def test_pair_redistribution_requires_member_singleton_and_own_threshold():
    data = discovery(
        [[0, 0, 900], [-10, 0, 0], [-100, 0, 0]], pair_delta=[[500, 0, 0], [400, 0, 0], [300, 0, 0]]
    )
    result = select(data, (2, "a × b"))
    rule = rule_row(result, "paired_redistribution", "a × b")
    assert rule.selected_row_id == 2 and rule.eligible_count == 1
    reason = result["reasons"].query("case_rule == 'paired_redistribution'").iloc[0]
    assert reason.linked_singletons == "a"
    assert reason.related_threshold_a == 25
    assert reason.related_threshold_b == 100
    assert "not causal-path evidence" in reason.rule_description


def test_scale_threshold_and_quantile_are_recorded_per_target():
    data = discovery([[100, 110, 0], [0, 100, 0], [0, 99, 0]])
    result = select(data, (1, "a"), (1, "b"))
    assert rule_row(result, "largest_target_change").material_threshold == 25
    rule = rule_row(result, "largest_target_change", "b")
    assert rule.material_threshold == 100 and rule.eligible_count == 2
    assert rule.target_abs_delta_quantile == 105


def test_candidate_audit_covers_all_target_rules_and_selected_reasons():
    result = select(discovery([[50, 0, 0], [-150, 0, 0]]), (1, "a"), (2, "a × b"))
    assert len(result["candidates"]) == 2 * (5 + 6)
    assert len(result["rules"]) == 11
    assert result["candidates"].selected.sum() == len(result["reasons"])
    assert result["candidates"].query("eligible").exclusion_reason.eq("").all()
    assert result["candidates"].query("not eligible").exclusion_reason.ne("").all()
    assert result["selected_unique"].row_id.is_unique


def test_validation_values_are_ignored_and_input_is_not_mutated():
    data = discovery([[100, 0, 0], [0, 0, 0]])
    before = data["points"].copy(deep=True)
    expected = select(data, (1, "a"))
    data["validation"] = pd.DataFrame({"row_id": [0], "delta": [-99999], "stable": [False]})
    actual = select(data, (1, "a"))
    pd.testing.assert_frame_equal(expected["reasons"], actual["reasons"])
    pd.testing.assert_frame_equal(before, data["points"])


def test_invalid_alignment_duplicate_targets_and_nonfinite_data_rejected():
    data = discovery([[100, 0, 0], [0, 0, 0]])
    with pytest.raises(ValueError, match="Duplicate target"):
        select(data, (1, "a"), (1, "a"))
    with pytest.raises(ValueError, match="Unknown target"):
        select(data, (2, "b × a"))
    data["points"] = data["points"].iloc[::-1]
    with pytest.raises(ValueError, match="row_id order"):
        select(data, (1, "a"))
    data["points"] = data["points"].iloc[::-1]
    data["base_phi"][0, 0] = np.nan
    with pytest.raises(ValueError, match="finite"):
        select(data, (1, "a"))


def test_undefined_rank_never_qualifies_as_preserved():
    data = discovery([[100, 100, 100]], baseline=[[0, 0, 0]])
    result = select(data, (1, "a"))
    assert rule_row(result, "rank_preserved_large_change").status == "omitted"


def test_default_targets_distinguish_prior_exploration_and_full_scan():
    b1 = pd.DataFrame(default_target_specs("b1"))
    assert b1.loc[b1.item == "cosyear", "origin"].item() == "previously_observed_exploratory"
    assert b1.loc[b1.item == "temp", "origin"].item() == "research_focus"
    assert len(default_target_specs("b3", scan_all=True)) == 28
    with pytest.raises(ValueError, match="No default targets"):
        default_target_specs("unknown")
