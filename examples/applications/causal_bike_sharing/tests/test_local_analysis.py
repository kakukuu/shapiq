"""Analytical regression cases for paired sample analysis and selection."""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from local_analysis import (
    build_local_tables,
    export_tables,
    plot_aggregate_overview,
    plot_case_comparison,
    plot_sample_overview,
    select_cases,
)


class LocalAnalysisTests(unittest.TestCase):
    names = ["a", "b", "c"]

    def tables(self, phi0, phi1, int0=None, int1=None, **kwargs):
        if int0 is None:
            int0 = np.tile([1.0, 2.0, 3.0], (len(phi0), 1))
        if int1 is None:
            int1 = int0
        return build_local_tables(
            phi0,
            phi1,
            int0,
            int1,
            feature_names=self.names,
            row_ids=kwargs.pop("row_ids", range(len(phi0))),
            **kwargs,
        )

    def test_direction_cancellation_retains_displacement(self):
        tables = self.tables([[10, 2, 3], [10, 2, 3]], [[110, 2, 3], [-90, 2, 3]])
        a = tables["summary"].query("order == 1 and item == 'a'").iloc[0]
        self.assertEqual(a["signed_mean_delta"], 0)
        self.assertEqual(a["mean_abs_delta"], 100)
        self.assertEqual(a["mean_magnitude_change"], 90)
        self.assertEqual(a["sign_flip_fraction"], 0.5)
        self.assertEqual(len(tables["feature_details"]), 6)

    def test_sign_reversal_is_distinct_from_magnitude_change(self):
        tables = self.tables([[10, 2, 3]], [[-10, 2, 3]])
        detail = tables["feature_details"].iloc[0]
        self.assertEqual(detail["delta"], -20)
        self.assertEqual(detail["abs_delta"], 20)
        self.assertEqual(detail["magnitude_change"], 0)
        self.assertTrue(detail["sign_flip"])
        self.assertEqual(tables["points"].iloc[0]["first_sign_flips"], 1)

    def test_constant_rank_undefined_and_zero_transition_not_sign_flip(self):
        tables = self.tables([[0, 0, 0]], [[1, 2, 3]], [[0, 0, 0]], [[0, 0, 0]])
        row = tables["points"].iloc[0]
        self.assertTrue(np.isnan(row["tau_first_signed"]))
        self.assertTrue(np.isnan(row["tau_second_signed"]))
        self.assertEqual(row["first_sign_flips"], 0)
        self.assertEqual(row["first_zero_transitions"], 3)
        self.assertTrue(np.isfinite(row["effect_score"]))
        self.assertEqual(tables["score_scales"].iloc[0]["method"], "unit_fallback_zero_baseline")

    def test_fixed_rules_and_id_ties(self):
        base = np.tile([1.0, 2.0, 3.0], (4, 1))
        pert = base * np.array([1.0, 2.0, 2.0, 5.0])[:, None]
        tables = self.tables(
            base,
            pert,
            base,
            pert,
            row_ids=[40, 30, 20, 10],
            context=pd.DataFrame({"date": ["d0", "d1", "d2", "d3"]}),
        )
        cases, availability = select_cases(tables["points"])
        cases = cases.set_index("case_rule")
        self.assertEqual(cases.loc["typical", "row_id"], 20)
        self.assertEqual(cases.loc["strong", "row_id"], 10)
        self.assertEqual(cases.loc["weak", "row_id"], 40)
        self.assertEqual(cases.loc["high_rank_large_shift", "row_id"], 10)
        self.assertTrue(cases.loc["high_rank_large_shift", "duplicates_previous_case"])
        self.assertEqual(cases.loc["strong", "date"], "d3")
        self.assertEqual(
            availability.set_index("case_rule").loc["high_rank_large_shift", "eligible_count"], 1
        )

    def test_missing_high_rank_case_is_omitted(self):
        base = np.tile([1.0, 2.0, 3.0], (2, 1))
        pert = -base * np.array([1.0, 2.0])[:, None]
        cases, availability = select_cases(self.tables(base, pert, base, pert)["points"])
        self.assertNotIn("high_rank_large_shift", cases["case_rule"].tolist())
        row = availability.set_index("case_rule").loc["high_rank_large_shift"]
        self.assertEqual(row["status"], "omitted")
        self.assertTrue(row["reason"])

    def test_pairing_and_shape_guards(self):
        base = [[1, 2, 3], [3, 4, 5]]
        with self.assertRaisesRegex(ValueError, "unique"):
            self.tables(base, base, row_ids=[1, 1])
        with self.assertRaisesRegex(ValueError, "row_id order"):
            self.tables(base, base, row_ids=[1, 2], context={"row_id": [2, 1]})
        with self.assertRaisesRegex(ValueError, "inconsistent"):
            self.tables(base, base, [[1], [2]], [[1], [2]])
        with self.assertRaisesRegex(ValueError, "shadows"):
            self.tables(base, base, context={"delta": [1, 2]})

    def test_lexical_interaction_order_and_top_k(self):
        tables = self.tables([[1, 2, 3]], [[1, 2, 3]], [[9, 8, 1]], [[1, 8, 9]], top_k=1)
        self.assertEqual(
            tables["interaction_details"]["item"].tolist(), ["a × b", "a × c", "b × c"]
        )
        self.assertEqual(tables["points"].iloc[0]["top_k_interaction_overlap"], 0)

    def test_exports_and_plot_scales(self):
        base = np.tile([1.0, 2.0, 3.0], (3, 1))
        pert = base * np.array([1.0, 2.0, 4.0])[:, None]
        tables = self.tables(base, pert, base, pert)
        with tempfile.TemporaryDirectory(dir=Path(__file__).parent) as tmp:
            root = Path(tmp)
            export_tables(tables, root / "tables")
            reread = pd.read_csv(root / "tables" / "feature_details.csv")
            self.assertEqual(len(reread), 9)
            figs = [
                plot_aggregate_overview({"test": tables}, root / "aggregate.png"),
                plot_sample_overview({"test": tables}, root / "sample.png"),
                plot_case_comparison(
                    base[0],
                    pert[-1],
                    base[0],
                    pert[-1],
                    feature_names=self.names,
                    case_label="row 2",
                    output_path=root / "case.test.png",
                ),
            ]
            case_axes = figs[-1].axes
            self.assertEqual(case_axes[0].get_xlim(), case_axes[1].get_xlim())
            self.assertEqual(case_axes[3].images[0].get_clim(), case_axes[4].images[0].get_clim())
            for name in ("aggregate", "sample", "case.test"):
                for ext in ("png", "pdf"):
                    self.assertGreater((root / f"{name}.{ext}").stat().st_size, 0)
            for fig in figs:
                plt.close(fig)


if __name__ == "__main__":
    unittest.main()
