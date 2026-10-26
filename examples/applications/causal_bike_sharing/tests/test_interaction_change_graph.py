"""Verify signed-change semantics and paired plotting scales."""

from __future__ import annotations

import unittest

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from interaction_change_graph import plot_interaction_change_graph, summarize_change_graph


class InteractionChangeGraphTests(unittest.TestCase):
    def tearDown(self):
        plt.close("all")

    def test_signed_increase_does_not_imply_strengthening(self):
        result = summarize_change_graph(
            [1, 2, 3], [2, 3, 4], [-100, -100, 10], [-20, 100, 30], ["a", "b", "c"]
        )
        edges = result["edges"]
        np.testing.assert_allclose(edges.delta, [80, 200, 20])
        np.testing.assert_allclose(edges.magnitude_change, [-80, 0, 20])
        self.assertEqual(edges.magnitude_direction.tolist(), ["weaker", "unchanged", "stronger"])
        self.assertEqual(edges.sign_flip.tolist(), [False, True, False])

    def test_mean_change_does_not_replace_mean_absolute_sample_change(self):
        result = summarize_change_graph(
            [[10, 0], [-10, 0]], [[-10, 0], [10, 0]], [[100], [-100]], [[-100], [100]], ["a", "b"]
        )
        self.assertTrue(result["is_mean"])
        self.assertEqual(result["n"], 2)
        edge = result["edges"].iloc[0]
        self.assertEqual(edge.delta, 0)
        self.assertEqual(edge.mean_absolute_sample_change, 200)
        self.assertFalse(edge.sign_flip)
        self.assertFalse(edge.selected)
        self.assertEqual(result["nodes"].mean_absolute_sample_change.iloc[0], 20)

    def test_union_adds_largest_change_even_when_not_largest_arm_pair(self):
        result = summarize_change_graph(
            [1, 2, 3], [1, 2, 3], [100, 90, -80], [100, 200, 80], ["a", "b", "c"], top_k=1
        )
        self.assertEqual(result["edges"].query("selected").pair_index.tolist(), [0, 1, 2])
        self.assertEqual(result["edges"].query("top_delta").pair_index.tolist(), [2])

    def test_plot_keeps_paired_edges_and_scales(self):
        fig = plot_interaction_change_graph(
            [1, 2, 3], [1, 2, 9], [10, -20, 1], [10, -60, -1], ["a", "b", "c"], top_k=3
        )
        axes = fig.axes
        self.assertEqual(len(axes), 3)
        for ax in axes:
            self.assertEqual(
                {p.get_gid() for p in ax.patches},
                {"interaction-pair-0", "interaction-pair-1", "interaction-pair-2"},
            )
        left = next(p for p in axes[0].patches if p.get_gid() == "interaction-pair-0")
        right = next(p for p in axes[1].patches if p.get_gid() == "interaction-pair-0")
        self.assertEqual(left.get_linewidth(), right.get_linewidth())
        self.assertEqual(left.get_edgecolor(), right.get_edgecolor())
        self.assertEqual(axes[0].collections[0].get_clim(), axes[1].collections[0].get_clim())
        np.testing.assert_allclose(
            axes[0].collections[0].get_offsets(), axes[2].collections[0].get_offsets()
        )
        self.assertTrue(any("flip" in text.get_text() for text in axes[2].texts))
        self.assertTrue(any("ΔI -40.0" in text.get_text() for text in axes[2].texts))
        fig.canvas.draw()

    def test_invalid_or_unpaired_inputs_fail(self):
        with self.assertRaises(ValueError):
            summarize_change_graph([[1, 2]], [[1, 2], [3, 4]], [[1]], [[1]], ["a", "b"])
        with self.assertRaises(ValueError):
            summarize_change_graph([1, 2], [1, 2], [np.nan], [1], ["a", "b"])
        with self.assertRaises(ValueError):
            summarize_change_graph([1, 2], [1, 2], [1], [1], ["a", "b"], top_k=0)

    def test_all_zero_data_are_valid_and_not_fabricated_edges(self):
        fig = plot_interaction_change_graph([0, 0], [0, 0], [0], [0], ["a", "b"])
        self.assertEqual(len(fig.axes[0].patches), 0)
        self.assertTrue(np.isfinite(fig.axes[0].collections[0].get_sizes()).all())
        fig.canvas.draw()

    def test_focus_pair_below_top_k_is_present_without_truncating_full_table(self):
        names = [f"f{i}" for i in range(7)]
        interactions = np.arange(21, 0, -1, dtype=float)
        summary = summarize_change_graph(
            np.ones(7),
            np.ones(7),
            interactions,
            interactions,
            names,
            top_k=1,
            focus_features=["f6"],
            focus_pairs=[("f6", "f5"), "f5 × f6"],
        )
        self.assertEqual(len(summary["edges"]), 21)
        self.assertEqual(summary["edges"].query("is_focus").pair_index.tolist(), [20])
        self.assertEqual(summary["edges"].query("selected").pair_index.tolist(), [0, 20])
        self.assertFalse(summary["edges"].iloc[20].top_baseline)
        self.assertEqual(summary["nodes"].query("is_focus").feature.tolist(), ["f6"])
        fig = plot_interaction_change_graph(
            np.ones(7),
            np.ones(7),
            interactions,
            interactions,
            names,
            top_k=1,
            focus_features=["f6"],
            focus_pairs=["f6 × f5"],
        )
        for ax in fig.axes:
            self.assertIn("interaction-pair-20", {patch.get_gid() for patch in ax.patches})
            self.assertTrue(any(text.get_text().startswith("* f6") for text in ax.texts))
        fig.canvas.draw()

    def test_zero_focus_pair_has_label_but_no_false_weight(self):
        fig = plot_interaction_change_graph(
            [0, 0], [0, 0], [0], [0], ["a", "b"], focus_pairs=[("b", "a")]
        )
        self.assertTrue(fig.change_graph_summary["edges"].iloc[0].selected)
        for ax in fig.axes:
            self.assertEqual(ax.patches[0].get_linewidth(), 0)
            self.assertTrue(
                any(
                    "+0.0" in text.get_text() and text.get_text().startswith("* ")
                    for text in ax.texts
                )
            )
        fig.canvas.draw()

    def test_invalid_focus_targets_fail_instead_of_disappearing(self):
        for kwargs in (
            {"focus_features": ["missing"]},
            {"focus_pairs": ["a × missing"]},
            {"focus_pairs": [("a", "a")]},
            {"focus_pairs": ["a"]},
        ):
            with self.subTest(kwargs=kwargs), self.assertRaises(ValueError):
                summarize_change_graph([1, 2], [1, 2], [1], [1], ["a", "b"], **kwargs)


if __name__ == "__main__":
    unittest.main()
