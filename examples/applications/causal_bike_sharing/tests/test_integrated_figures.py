"""Check that visual summaries preserve paired estimands and comparison scales."""

from __future__ import annotations

import unittest

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from integrated_figures import (
    plot_first_order_distribution,
    plot_first_order_summary,
    plot_mean_interaction_matrices,
    plot_mean_interaction_networks,
    plot_sample_heterogeneity,
)


class IntegratedFigureTests(unittest.TestCase):
    def tearDown(self):
        plt.close("all")

    def test_cancellation_and_magnitude_are_separate_estimands(self):
        baseline = np.array([[10.0, 0.0], [-10.0, 0.0]])
        modified = -baseline
        figure = plot_first_order_summary(baseline, modified, feature_names=["a", "b"])

        def get_widths(ax):
            return np.array([bar.get_width() for bar in ax.patches])

        np.testing.assert_allclose(get_widths(figure.axes[1]), [0.0, 0.0])
        np.testing.assert_allclose(get_widths(figure.axes[2]), [20.0, 0.0])
        np.testing.assert_allclose(get_widths(figure.axes[3]), [0.0, 0.0])
        figure.canvas.draw()

    def test_distribution_preserves_all_147_samples_in_both_arms(self):
        baseline = np.arange(147 * 3, dtype=float).reshape(147, 3)
        figure = plot_first_order_distribution(
            baseline, -baseline, feature_names=["a", "b", "c"], title_prefix="Figure B2-1"
        )
        full_point_groups = [
            item for item in figure.axes[0].collections if len(item.get_offsets()) == 147
        ]
        self.assertEqual(len(full_point_groups), 6)
        self.assertIn("n = 147", figure.axes[0].get_title())
        self.assertTrue(figure.axes[0].get_title().startswith("Figure B2-1"))
        figure.canvas.draw()

    def test_heatmaps_share_arm_scales_and_difference_preserves_sign(self):
        baseline = np.array([[1.0, 2.0, 3.0], [1.0, 2.0, 3.0]])
        modified = baseline * 3
        figure = plot_mean_interaction_matrices(baseline, modified, feature_names=["a", "b", "c"])
        self.assertEqual(figure.axes[0].images[0].get_clim(), figure.axes[1].images[0].get_clim())
        self.assertEqual(figure.axes[0].images[0].get_clim(), (-9.0, 9.0))
        self.assertEqual(float(figure.axes[2].images[0].get_array()[0, 1]), 2.0)
        self.assertTrue(np.ma.is_masked(figure.axes[2].images[0].get_array()[0, 0]))
        figure.canvas.draw()

    def test_network_top_union_and_one_width_scale(self):
        base = np.array([[10.0, 2.0, 1.0]])
        variant = np.array([[10.0, 20.0, 1.0]])
        figure = plot_mean_interaction_networks(
            base, variant, feature_names=["a", "b", "c"], top_k=1
        )
        axes = figure.axes[:2]
        self.assertEqual([len(ax.lines) for ax in axes], [2, 2])
        # The interaction of value 10 occurs in both panels and keeps its width/color.
        left = next(line for line in axes[0].lines if np.isclose(line.get_linewidth(), 3.0))
        right = next(line for line in axes[1].lines if np.isclose(line.get_linewidth(), 3.0))
        np.testing.assert_allclose(left.get_color(), right.get_color())
        figure.canvas.draw()

    def test_heterogeneity_uses_matched_context_without_truncation(self):
        baseline = np.zeros((146, 2))
        modified = np.column_stack([np.arange(146), -np.arange(146)])
        figure = plot_sample_heterogeneity(
            baseline,
            modified,
            feature_names=["a", "b"],
            context_values=np.arange(146),
            context_label="Target day",
            color_values=np.arange(146),
            color_label="Day of year",
        )
        np.testing.assert_allclose(
            figure.axes[0].collections[0].get_offsets()[:, 1], np.arange(146)
        )
        np.testing.assert_allclose(
            figure.axes[1].collections[0].get_offsets()[:, 1], np.arange(146)
        )
        with self.assertRaises(ValueError):
            plot_sample_heterogeneity(
                baseline,
                modified,
                feature_names=["a", "b"],
                context_values=np.arange(145),
                context_label="Unpaired",
            )
        figure.canvas.draw()


if __name__ == "__main__":
    unittest.main()
