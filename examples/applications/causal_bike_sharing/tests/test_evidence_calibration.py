"""Regression tests for conditional estimator calibration."""

from __future__ import annotations

import unittest
from unittest.mock import patch

import data_utils as du
import numpy as np


class FakeSII:
    def __init__(self, phi, edges=(1, 2, 3)):
        self.dict_values = {(i,): float(x) for i, x in enumerate(phi)}
        self.interactions = {(0, 1): edges[0], (0, 2): edges[1], (1, 2): edges[2]}

    def get_n_order(self, n):
        return self


class CalibrationTests(unittest.TestCase):
    def test_repeat_changes_only_rng_on_same_object_and_rows(self):
        ex, rows = object(), object()
        with patch.object(du, "explain_multi", return_value=["result"]) as call:
            out = du.estimator_repeats(ex, rows, budget=128, n=3, seeds=(7, 19))
        self.assertEqual(len(out), 2)
        for item, seed in zip(call.call_args_list, (7, 19), strict=True):
            self.assertIs(item.args[0], ex)
            self.assertIs(item.args[1], rows)
            self.assertEqual(item.kwargs["seed"], seed)
            self.assertEqual(item.kwargs["budget"], 128)

    def test_perfect_null_and_reversed_observation(self):
        base = [FakeSII([1, 2, 3]) for _ in range(6)]
        pert = [FakeSII([3, 2, 1], (3, 2, 1)) for _ in range(6)]
        res = du.rank_metrics(base, pert, ["a", "b", "c"], sii_null_runs=[base, base], n_boot=100)
        for k in du.TAU_VARIANTS:
            self.assertAlmostEqual(res[k]["paired_difference"]["mean"], -2)
            self.assertEqual(du.tau_verdict(res[k]), "distortion")

    def test_identical_observation_and_repeat_is_unresolved(self):
        base = [FakeSII([1, 2, 3]) for _ in range(6)]
        same = [FakeSII([2, 1, 3]) for _ in range(6)]
        res = du.rank_metrics(base, same, ["a", "b", "c"], sii_null_runs=[same, same], n_boot=100)
        self.assertEqual(res["tau1_signed"]["paired_difference"]["family_ci_hi"], 0)
        self.assertEqual(du.tau_verdict(res["tau1_signed"]), "indistinct")

    def test_pairing_retains_covariance(self):
        base = [FakeSII([1, 2, 3]) for _ in range(6)]
        varied = [FakeSII(p) for p in ([1, 2, 3], [3, 2, 1], [2, 1, 3]) * 2]
        res = du.rank_metrics(
            base, varied, ["a", "b", "c"], sii_null_runs=[varied, varied], n_boot=100
        )
        self.assertGreater(res["tau1_signed"]["std"], 0)
        self.assertAlmostEqual(res["tau1_signed"]["paired_difference"]["ci_lo"], 0)
        self.assertAlmostEqual(res["tau1_signed"]["paired_difference"]["ci_hi"], 0)

    def test_unmatched_rows_cannot_get_null_verdict(self):
        base = [FakeSII([1, 2, 3]) for _ in range(3)]
        with self.assertRaises(ValueError):
            du.rank_metrics(base, base, ["a", "b", "c"], sii_null_runs=[base], paired=False)
        res = du.rank_metrics(base, base, ["a", "b", "c"], paired=False, n_boot=10)
        self.assertEqual(du.tau_verdict(res["tau1_signed"]), "descriptive")

    def test_invalid_repeats_rejected(self):
        with self.assertRaises(ValueError):
            du.estimator_repeats(None, None, seeds=(42, 7))

    def test_real_explainer_repeatability_and_fixed_background(self):
        X, y = du.load_data()
        ex, rows = du.get_explainer(X, y)
        background = ex.imputer.data.copy()
        model = ex.imputer.model
        first = du.explain_multi(ex, rows, n=2, budget=5000, seed=42)
        changed = du.explain_multi(ex, rows, n=2, budget=5000, seed=7)
        again = du.explain_multi(ex, rows, n=2, budget=5000, seed=42)
        np.testing.assert_array_equal(background, ex.imputer.data)
        self.assertIs(model, ex.imputer.model)
        for a, b in zip(first, again, strict=True):
            np.testing.assert_array_equal(a.values, b.values)
        self.assertTrue(
            any(not np.array_equal(a.values, b.values) for a, b in zip(first, changed, strict=True))
        )
        # Lag optimization: duplicate/reordered rows produce precisely cached values.
        idx = [1, 0, 1]
        repeated = du.explain_multi(ex, rows.iloc[idx], n=3, budget=5000)
        np.testing.assert_array_equal(
            np.array([r.values for r in repeated]), np.array([first[i].values for i in idx])
        )


if __name__ == "__main__":
    unittest.main()
