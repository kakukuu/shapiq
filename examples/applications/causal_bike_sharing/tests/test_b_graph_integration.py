"""Check B-network coverage, exact source rows and targeted publication.

These checks only read the saved discovery arrays. No fitted model or
explanation run is invoked, and notebook writes are confined to temp folders.
"""

from __future__ import annotations

import json
import tempfile
import unittest
from itertools import combinations
from pathlib import Path
from unittest.mock import patch

import matplotlib

matplotlib.use("Agg")
import integrated_support as support
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import sample_readings

ROOT = Path(__file__).resolve().parents[1]
B_ARMS = ("b1", "b2", "b3")


class ChangeGraphSourceValueTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        # Deliberately non-positional row IDs and heterogeneous paired changes
        # test the plotting contract independently of old saved explanations.
        rng = np.random.default_rng(734)
        n = 147
        rows = 1000 + 2 * np.arange(n)
        phi = rng.normal(size=(n, 7))
        interactions = rng.normal(size=(n, 21))
        cls.data = {
            "arm": "b2",
            "n": n,
            "row_ids": rows,
            "base_phi": phi,
            "variant_phi": phi + rng.normal(size=phi.shape),
            "base_interaction": interactions,
            "variant_interaction": interactions + rng.normal(size=interactions.shape),
            "points": pd.DataFrame(
                {
                    "row_id": rows,
                    "date": pd.date_range("2011-01-01", periods=n).strftime("%Y-%m-%d"),
                    "temp": 10.125 + 0.1 * np.arange(n),
                    "atemp": 14.375 + 0.1 * np.arange(n),
                    "hum": 61.5 + 0.1 * np.arange(n),
                    "windspeed": 7.25 + 0.02 * np.arange(n),
                    "trend": np.arange(n),
                    "cosyear": np.cos(np.arange(n) / 366.0),
                    "sinyear": np.sin(np.arange(n) / 366.0),
                    "y_true": 1500 + 7 * np.arange(n),
                    "prediction_base": 1540.25 + 7 * np.arange(n),
                    "prediction_variant": 1540.25 + 7 * np.arange(n),
                }
            ),
            "selected": pd.DataFrame({"row_id": [rows[3]], "case_rule": ["strong"]}),
        }

    def tearDown(self):
        plt.close("all")

    def _check_export(self, row_id=None, rule=None):
        number = "B2-999"
        with tempfile.TemporaryDirectory(dir=ROOT) as directory:
            root = Path(directory)
            with (
                patch.object(support, "display"),
                patch.object(
                    support, "figure", side_effect=lambda fig, *_: plt.close(fig)
                ) as output,
                patch.object(
                    support, "build_sample_readings", wraps=support.build_sample_readings
                ) as build,
                patch.object(
                    support, "add_sample_readings", wraps=support.add_sample_readings
                ) as decorate,
            ):
                support.change_network(self.data, number, row_id=row_id, case_rule=rule, root=root)
            output.assert_called_once()
            fig = output.call_args.args[0]
            summary = fig.change_graph_summary
            self.assertEqual(summary["n"], 147 if row_id is None else 1)
            self.assertEqual(summary["is_mean"], row_id is None)
            export = root / support.INTEGRATED / "change_graph_values"
            nodes = pd.read_csv(export / f"{number}_nodes.csv")
            edges = pd.read_csv(export / f"{number}_edges.csv")
            self.assertEqual(nodes.feature.tolist(), list(support.FEATURES))
            expected_pairs = list(combinations(support.FEATURES, 2))
            self.assertEqual(list(zip(edges.source, edges.target, strict=True)), expected_pairs)
            self.assertEqual(len(edges), 21)  # Non-displayed pairs remain auditable.
            selected = (
                None if row_id is None else int(np.flatnonzero(self.data["row_ids"] == row_id)[0])
            )
            readings_file = export / f"{number}_readings.json"
            if selected is None:
                build.assert_not_called()
                decorate.assert_not_called()
                self.assertIsNone(getattr(fig, "sample_readings", None))
                self.assertFalse(readings_file.exists())
            else:
                expected_point = self.data["points"].iloc[selected]
                build.assert_called_once()
                pd.testing.assert_series_equal(build.call_args.args[0], expected_point)
                self.assertEqual(build.call_args.args[1], "b2")
                decorate.assert_called_once()
                self.assertIs(decorate.call_args.args[0], fig)
                readings = sample_readings.build_sample_readings(expected_point, "b2")
                self.assertEqual(fig.sample_readings, readings)
                self.assertEqual(readings["row_id"], row_id)
                self.assertEqual(readings["observed_raw_values"]["temp"], expected_point.temp)
                self.assertEqual(readings["outcomes"]["y_true"], expected_point.y_true)
                self.assertEqual(json.loads(readings_file.read_text(encoding="utf-8")), readings)
            for frame, prefix in ((nodes, "phi"), (edges, "interaction")):
                base = self.data[f"base_{prefix}"]
                modified = self.data[f"variant_{prefix}"]
                if selected is not None:
                    base, modified = (
                        base[selected : selected + 1],
                        modified[selected : selected + 1],
                    )
                np.testing.assert_allclose(
                    frame.baseline, base.mean(axis=0), rtol=1e-12, atol=1e-10
                )
                np.testing.assert_allclose(
                    frame.modified, modified.mean(axis=0), rtol=1e-12, atol=1e-10
                )
                np.testing.assert_allclose(
                    frame.delta, (modified - base).mean(axis=0), rtol=1e-12, atol=1e-10
                )
                np.testing.assert_allclose(
                    frame.mean_absolute_sample_change,
                    np.abs(modified - base).mean(axis=0),
                    rtol=1e-12,
                    atol=1e-10,
                )
                self.assertTrue(frame.variant.eq("b2").all())
                self.assertTrue(frame.scope.eq("mean" if selected is None else "sample").all())
                if selected is None:
                    self.assertTrue(frame.row_id.isna().all())
                    self.assertTrue(frame.date.isna().all())
                else:
                    self.assertTrue(frame.row_id.eq(row_id).all())
                    self.assertTrue(frame.date.eq(self.data["points"].iloc[selected].date).all())

    def test_mean_export_uses_all_147_saved_rows(self):
        self._check_export()

    def test_sample_export_uses_original_row_id_not_position(self):
        strong = self.data["selected"].query("case_rule == 'strong'").iloc[0]
        # The original row ID exceeds the holdout length, catching accidental
        # direct positional indexing into the 147-row explanation matrices.
        self.assertGreater(int(strong.row_id), len(self.data["row_ids"]))
        self._check_export(int(strong.row_id), "strong")

    def test_unknown_row_or_incorrect_frozen_rule_is_rejected(self):
        with tempfile.TemporaryDirectory(dir=ROOT) as directory:
            root = Path(directory)
            row_id = int(self.data["selected"].iloc[0].row_id)
            for parameters in (
                {"row_id": -1},
                {"row_id": row_id, "case_rule": "not_frozen"},
                {"row_id": 1.5},
                {"row_id": True},
                {"case_rule": "strong"},
            ):
                with self.subTest(parameters=parameters), self.assertRaises(ValueError):
                    support.change_network(self.data, "B2-999", root=root, **parameters)
            self.assertEqual(list(root.iterdir()), [])

    def test_legacy_discovery_and_repeat_figures_show_the_same_original_sample_readings(self):
        row_id = int(self.data["selected"].iloc[0].row_id)
        point = self.data["points"].loc[self.data["points"].row_id.eq(row_id)].iloc[0]
        pair_labels = [" x ".join(support.FEATURES[i] for i in pair) for pair in support.PAIRS]
        seeds = [809, 907]
        data = {
            **self.data,
            "selected": pd.DataFrame(
                [
                    dict(
                        row_id=row_id,
                        date=point.date,
                        case_rule="strong",
                        effect_score_percentile=100.0,
                    )
                ]
            ),
            "protocol": dict(validation_seeds=seeds),
            "validation_details": pd.DataFrame(
                [
                    dict(
                        row_id=row_id,
                        case_rule="strong",
                        seed=seed,
                        order=order,
                        item=item,
                        delta=float(position + 1),
                    )
                    for seed in seeds
                    for order, items in ((1, support.FEATURES), (2, pair_labels))
                    for position, item in enumerate(items)
                ]
            ),
        }
        with tempfile.TemporaryDirectory(dir=ROOT) as directory:
            with (
                patch.object(
                    support, "figure", side_effect=lambda fig, *_: plt.close(fig)
                ) as output,
                patch.object(
                    support,
                    "plot_case_comparison",
                    side_effect=lambda *_, **__: plt.figure(figsize=(12, 4)),
                ),
                patch.object(
                    support, "add_sample_readings", wraps=support.add_sample_readings
                ) as decorate,
            ):
                support.case_plots(data, row_id, "strong", "B2-998", "B2-999", root=Path(directory))
        self.assertEqual(output.call_count, 2)
        self.assertEqual(decorate.call_count, 2)
        expected = sample_readings.build_sample_readings(point, "b2")
        for call in output.call_args_list:
            self.assertEqual(call.args[0].sample_readings, expected)
            self.assertEqual(call.args[0].sample_readings["row_id"], row_id)


if __name__ == "__main__":
    unittest.main()
