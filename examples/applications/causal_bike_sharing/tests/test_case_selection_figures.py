"""Selected targets remain visible and validation reports actual reversals."""

from __future__ import annotations

import unittest

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
from case_selection_figures import plot_target_validation, summarize_target_validation


def _fixture():
    rows = []
    for order, item in ((1, "temperature"), (2, "temperature × humidity")):
        for seed, baseline, modified in zip(
            (809, 907, 1009, 1103), (10.0, 10.0, -10.0, 0.0), (-10.0, 10.0, 10.0, 0.0), strict=True
        ):
            rows.append(
                dict(
                    arm="B3",
                    row_id=10,
                    date="2011-01-11",
                    order=order,
                    item=item,
                    seed=seed,
                    baseline=baseline,
                    modified=modified,
                    delta=modified - baseline,
                )
            )
    reasons = pd.DataFrame(
        [
            dict(
                order=1,
                item="temperature",
                case_rule="sign_flip",
                baseline=10.0,
                modified=-10.0,
                delta=-20.0,
                endpoint_threshold=8.0,
            ),
            dict(
                order=1,
                item="temperature",
                case_rule="large_change",
                baseline=10.0,
                modified=-10.0,
                delta=-20.0,
                endpoint_threshold=8.0,
            ),
            dict(
                order=2,
                item="humidity × temperature",
                case_rule="sign_flip",
                baseline=10.0,
                modified=-10.0,
                delta=-20.0,
                endpoint_threshold=12.0,
            ),
        ]
    )
    return pd.DataFrame(rows), reasons


class TargetValidationTests(unittest.TestCase):
    def tearDown(self):
        plt.close("all")

    def test_target_dedup_and_flip_counts_distinguish_direction_and_magnitude(self):
        details, reasons = _fixture()
        summary = summarize_target_validation(details, reasons)
        self.assertEqual(len(summary), 2)
        feature = summary.iloc[0]
        self.assertEqual(feature.case_rules, "sign_flip | large_change")
        self.assertEqual(feature.n_seeds, 4)
        self.assertEqual(feature.delta_direction_count, 1)
        self.assertEqual(feature.sign_flip_count, 2)
        self.assertEqual(feature.same_flip_direction_count, 1)
        self.assertEqual(feature.material_sign_flip_count, 2)
        self.assertEqual(feature.baseline_positive_count, 2)
        self.assertEqual(feature.modified_zero_count, 1)
        self.assertEqual(feature.delta_median, 0)
        self.assertAlmostEqual(feature.delta_q10, -14)
        self.assertAlmostEqual(feature.delta_q90, 14)
        self.assertEqual(summary.iloc[1].material_sign_flip_count, 0)

    def test_delta_threshold_is_not_a_material_flip_threshold(self):
        details, reasons = _fixture()
        reasons = reasons.drop(columns="endpoint_threshold").assign(threshold=12.0)
        summary = summarize_target_validation(details, reasons)
        self.assertTrue(summary.material_sign_flip_count.isna().all())
        self.assertEqual(summary.sign_flip_count.tolist(), [2, 2])

    def test_plot_contains_all_selected_targets_and_actual_seed_count(self):
        details, reasons = _fixture()
        additions = []
        reason_additions = []
        for item in ("wind", "trend", "season"):
            additions.append(details[details.order == 1].assign(item=item))
            reason_additions.append(reasons.iloc[[0]].assign(item=item))
        details = pd.concat([details, *additions], ignore_index=True)
        reasons = pd.concat([reasons, *reason_additions], ignore_index=True)
        fig = plot_target_validation(details, reasons)
        self.assertEqual(len(fig.target_validation_summary), 5)
        self.assertEqual(len(fig.axes[0].get_yticklabels()), 5)
        self.assertTrue(any("sign flip 2/4" in text.get_text() for text in fig.axes[0].texts))
        fig.canvas.draw()

    def test_selected_targets_cannot_be_silently_missing_or_incomplete(self):
        details, reasons = _fixture()
        for incomplete in (
            details[details.order == 1],
            details.iloc[:-1],
            pd.concat([details, details.iloc[[0]]], ignore_index=True),
        ):
            with self.subTest(size=len(incomplete)), self.assertRaises(ValueError):
                summarize_target_validation(incomplete, reasons)

    def test_multiple_cases_need_filter_and_rule_alias_supported(self):
        details, reasons = _fixture()
        details = pd.concat([details, details.assign(row_id=11)], ignore_index=True)
        with self.assertRaises(ValueError):
            summarize_target_validation(details, reasons)
        summary = summarize_target_validation(
            details, reasons.rename(columns={"case_rule": "rule"}), arm="B3", row_id=11
        )
        self.assertEqual(summary.row_id.unique().tolist(), [11])

    def test_inconsistent_discovery_or_delta_fails(self):
        details, reasons = _fixture()
        reasons.loc[1, ["baseline", "delta"]] = [11.0, -21.0]
        with self.assertRaises(ValueError):
            summarize_target_validation(details, reasons)
        details, reasons = _fixture()
        details.loc[0, "delta"] = 100.0
        with self.assertRaises(ValueError):
            summarize_target_validation(details, reasons)


if __name__ == "__main__":
    unittest.main()
