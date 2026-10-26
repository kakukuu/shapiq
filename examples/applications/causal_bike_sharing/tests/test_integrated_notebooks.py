"""Integration regressions without refitting or recomputing explanations.

The paired bootstrap tests use algebraic relationships that independent-arm
resampling would violate. Artifact checks verify the original row/date pairing
and contrast direction consumed by the integrated notebook source.
"""

from __future__ import annotations

import os
import unittest
from pathlib import Path

import integrated_support as support
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
ARMS = ("a", "b1", "b2", "b3", "lag")
COMPARISONS = ARMS + (
    "temp_root",
    "hum_root",
    "windspeed_root",
    "b2_unconfounded",
    "b3_unconfounded",
)


def _source(cell):
    value = cell.get("source", "")
    return value if isinstance(value, str) else "".join(value)


class PairedBootstrapTests(unittest.TestCase):
    def setUp(self):
        self.base = np.array(
            [
                [1.0, 4.0, -3.0],
                [2.0, 7.0, 6.0],
                [8.0, 3.0, 2.0],
                [-2.0, 9.0, 4.0],
                [3.0, -1.0, 9.0],
                [7.0, 2.0, -6.0],
            ]
        )

    def test_identical_arms_have_identical_replicates(self):
        result = support.paired_lag_variance(self.base, self.base, n_boot=128, seed=83)
        np.testing.assert_array_equal(result["base_draws"], result["variant_draws"])
        self.assertEqual(result["base_draws"].shape, (128, 3))
        self.assertEqual(result["n_resample_rows"], len(self.base))
        self.assertEqual(result["n_boot"], 128)
        self.assertAlmostEqual(result["total_variance_ratio"], 1.0)

    def test_affine_arms_preserve_each_paired_draw(self):
        # The same row index set must be used on both sides of every draw.
        variant = 2 * self.base + np.array([19.0, -8.0, 2.0])
        result = support.paired_lag_variance(self.base, variant, n_boot=257, seed=107)
        np.testing.assert_allclose(
            result["variant_draws"],
            2 * result["base_draws"] + np.array([19.0, -8.0, 2.0]),
            rtol=1e-13,
            atol=1e-13,
        )
        self.assertAlmostEqual(result["total_variance_ratio"], 4.0, places=11)

    def test_bootstrap_rejects_unpaired_or_nonfinite_rows(self):
        invalid = self.base.copy()
        invalid[0, 0] = np.nan
        for left, right in (
            (self.base, self.base[:-1]),
            (self.base, self.base[:, :-1]),
            (invalid, self.base),
            (self.base[:1], self.base[:1]),
        ):
            with self.subTest(left=left.shape, right=right.shape):
                with self.assertRaises(ValueError):
                    support.paired_lag_variance(left, right, n_boot=64, seed=2)

    def test_fixed_per_feature_shift_has_degenerate_row_interval(self):
        values = np.tile([12.0, -8.0, 0.0], (11, 1))
        table = support.bootstrap_table(values, ["positive", "negative", "zero"], n_boot=99, seed=4)
        self.assertEqual(table.feature.tolist(), ["positive", "negative", "zero"])
        for column in ("mean", "ci_low", "ci_high"):
            np.testing.assert_array_equal(table[column].to_numpy(), [12.0, -8.0, 0.0])
        self.assertEqual(table.ci_excludes_zero.tolist(), [True, True, False])


@unittest.skipUnless(
    os.environ.get("SHAPIQ_VERIFY_SAVED_RESULTS") == "1",
    "Set SHAPIQ_VERIFY_SAVED_RESULTS=1 after a completed current run.",
)
class SavedNumericalContractTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.loaded = {arm: support.load_experiment(arm, root=ROOT) for arm in COMPARISONS}

    def test_all_full_holdout_rows_stay_in_original_order(self):
        for arm, result in self.loaded.items():
            n = 146 if arm == "lag" else 147
            with self.subTest(arm=arm):
                self.assertEqual(result["n"], n)
                self.assertEqual(result["base_phi"].shape, (n, 7))
                self.assertEqual(result["variant_phi"].shape, (n, 7))
                self.assertEqual(result["base_interaction"].shape, (n, 21))
                self.assertEqual(result["variant_interaction"].shape, (n, 21))
                self.assertTrue(result["points"].row_id.is_unique)
                np.testing.assert_array_equal(result["row_ids"], result["points"].row_id)
                with np.load(ROOT / "results" / "draws" / f"{arm}.npz") as saved:
                    for name in (
                        "row_ids",
                        "base_phi",
                        "variant_phi",
                        "base_interaction",
                        "variant_interaction",
                    ):
                        np.testing.assert_array_equal(result[name], saved[name])

    def test_signed_and_absolute_changes_follow_modified_minus_baseline(self):
        for arm, result in self.loaded.items():
            summary = result["item_summary"]
            for order, baseline, modified in (
                (1, result["base_phi"], result["variant_phi"]),
                (2, result["base_interaction"], result["variant_interaction"]),
            ):
                delta = modified - baseline
                rows = summary.loc[summary.order.eq(order)]
                with self.subTest(arm=arm, order=order):
                    np.testing.assert_allclose(
                        rows.signed_mean_delta, delta.mean(axis=0), rtol=1e-9, atol=1e-8
                    )
                    np.testing.assert_allclose(
                        rows.mean_abs_delta, np.abs(delta).mean(axis=0), rtol=1e-9, atol=1e-8
                    )
                    np.testing.assert_allclose(
                        rows.mean_magnitude_change,
                        (np.abs(modified) - np.abs(baseline)).mean(axis=0),
                        rtol=1e-9,
                        atol=1e-8,
                    )

    def test_lag_dates_and_semantics_are_paired(self):
        result = self.loaded["lag"]
        points = result["points"]
        difference = pd.to_datetime(points.date) - pd.to_datetime(points.lag_source_date)
        self.assertTrue(difference.eq(pd.Timedelta(days=1)).all())
        for feature in ("trend", "cosyear", "sinyear", "atemp"):
            np.testing.assert_allclose(
                points[feature], points[f"{feature}_lag_design"], rtol=0, atol=1e-10
            )
        for feature in ("temp", "hum", "windspeed"):
            self.assertTrue((points[feature] != points[f"{feature}_lag_design"]).any())
        tables = ROOT / "results" / "tables"
        same_day = pd.read_csv(tables / "paired_same_day_split.csv")
        lag = pd.read_csv(tables / "paired_lag_split.csv")
        pd.testing.assert_frame_equal(
            same_day[["row_id", "date", "split"]], lag[["row_id", "date", "split"]]
        )
        test_ids = same_day.loc[same_day.split.eq("test"), "row_id"].to_numpy()
        np.testing.assert_array_equal(result["row_ids"], test_ids)

    def test_frozen_cases_keep_seed_provenance(self):
        self.assertTrue(all(len(result["selected"]) >= 3 for result in self.loaded.values()))
        for arm, result in self.loaded.items():
            seeds = set(result["protocol"]["validation_seeds"])
            selected = result["selected"]
            with self.subTest(arm=arm):
                self.assertTrue(set(selected.row_id).issubset(result["row_ids"]))
                self.assertNotIn(result["protocol"]["discovery_seed"], seeds)
                self.assertFalse(seeds.intersection(result["protocol"]["reference_seeds"]))
                for case in selected.itertuples(index=False):
                    repeats = result["validation"].loc[
                        result["validation"].row_id.eq(case.row_id)
                        & result["validation"].case_rule.eq(case.case_rule)
                    ]
                    self.assertEqual(set(repeats.seed), seeds)
                    self.assertEqual(len(repeats), len(seeds))
