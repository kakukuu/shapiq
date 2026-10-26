"""Input context remains accurate and readable on existing case figures."""

from __future__ import annotations

import json
import unittest

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sample_readings import FEATURES, add_sample_readings, build_sample_readings


def _point():
    return {
        "date": "2012-03-21",
        "row_id": np.int64(445),
        "trend": 445.0,
        "cosyear": 0.21,
        "sinyear": 0.98,
        "temp": 19.53,
        "atemp": 22.82,
        "windspeed": 12.75,
        "hum": 71.5,
        "y_true": 6192.0,
        "prediction_base": 6011.5,
        "prediction_variant": 6011.5,
    }


class SampleReadingsTests(unittest.TestCase):
    def tearDown(self):
        plt.close("all")

    def test_exact_input_values_are_json_safe_and_not_rescaled(self):
        point = _point()
        point["date"] = pd.Timestamp(point["date"])
        readings = build_sample_readings(pd.Series(point), "b2")
        json.dumps(readings, allow_nan=False)
        self.assertEqual(set(readings["input_raw_values"]["baseline"]), set(FEATURES))
        self.assertEqual(
            readings["input_raw_values"]["baseline"], readings["input_raw_values"]["modified"]
        )
        self.assertEqual(readings["input_raw_values"]["baseline"]["temp"], 19.53)
        text = "\n".join(readings["lines"])
        for item in (
            "19.53 °C",
            "22.82 °C",
            "71.50%",
            "12.75 (dataset scale)",
            "trend 445",
            "cosyear 0.210",
            "sinyear 0.980",
            "6,192",
            "6,011.5",
        ):
            self.assertIn(item, text)
        self.assertEqual(readings["row_id"], 445)
        self.assertEqual(readings["date"], "2012-03-21")

    def test_lag_distinguishes_observation_input_dates_and_model_predictions(self):
        point = _point()
        point.update(
            temp_lag_design=10.25,
            hum_lag_design=60.0,
            windspeed_lag_design=8.0,
            lag_source_date="2012-03-20",
            prediction_variant=5600.25,
        )
        readings = build_sample_readings(point, "lag")
        self.assertEqual(readings["input_raw_values"]["baseline"]["temp"], 19.53)
        self.assertEqual(readings["input_raw_values"]["modified"]["temp"], 10.25)
        self.assertEqual(readings["input_raw_values"]["modified"]["atemp"], 22.82)
        text = "\n".join(readings["lines"])
        self.assertIn("previous day 2012-03-20", text)
        self.assertIn("atemp 22.82 °C remains same-day", text)
        self.assertIn("Lag prediction 5,600.2", text)

    def test_missing_values_show_na_and_aggregate_is_not_a_sample(self):
        point = _point()
        point.update(temp=np.nan, hum=pd.NA, prediction_variant=float("inf"))
        readings = build_sample_readings(point, "b2")
        json.dumps(readings, allow_nan=False)
        self.assertIsNone(readings["input_raw_values"]["baseline"]["temp"])
        self.assertIn("Temperature (temp) n/a °C", readings["lines"][0])
        lag = build_sample_readings(point, "lag")
        self.assertIsNone(lag["input_raw_values"]["modified"]["windspeed"])
        self.assertIsNone(build_sample_readings(None, "b2"))
        fig, _ = plt.subplots(figsize=(12, 3.8))
        original = fig.get_size_inches().copy()
        self.assertIs(add_sample_readings(fig, None), fig)
        np.testing.assert_array_equal(original, fig.get_size_inches())

    def test_header_grows_canvas_without_moving_or_shrinking_existing_plot(self):
        for size in ((20.5, 8.2), (13.5, 4.3), (12.0, 3.8), (8.0, 4.0)):
            with self.subTest(size=size):
                fig, ax = plt.subplots(figsize=size)
                ax.plot([0, 1], [0, 1], label="Series")
                title = fig.suptitle("Original title", y=0.98)
                footer = fig.text(0.5, 0.01, "Original footer", ha="center")
                legend = fig.legend(loc="lower center", bbox_to_anchor=(0.5, 0.05))
                fig.canvas.draw()
                bounds = ax.get_window_extent().bounds
                title_bounds = title.get_window_extent().bounds
                footer_bounds = footer.get_window_extent().bounds
                legend_bounds = legend.get_window_extent().bounds
                readings = build_sample_readings(_point(), "b2")
                self.assertIs(add_sample_readings(fig, readings), fig)
                fig.canvas.draw()
                np.testing.assert_allclose(ax.get_window_extent().bounds, bounds, atol=1e-8)
                np.testing.assert_allclose(
                    title.get_window_extent().bounds, title_bounds, atol=1e-8
                )
                np.testing.assert_allclose(
                    footer.get_window_extent().bounds, footer_bounds, atol=1e-8
                )
                np.testing.assert_allclose(
                    legend.get_window_extent().bounds, legend_bounds, atol=1e-8
                )
                renderer = fig.canvas.get_renderer()
                box = fig.sample_readings_band.get_window_extent(renderer)
                for text in fig.sample_readings_artists:
                    extent = text.get_window_extent(renderer)
                    self.assertGreaterEqual(extent.x0, box.x0)
                    self.assertLessEqual(extent.x1, box.x1)
                    self.assertGreaterEqual(extent.y0, box.y0)
                    self.assertLessEqual(extent.y1, box.y1)
                self.assertGreater(box.y0, title.get_window_extent(renderer).y1)
                height = fig.get_size_inches()[1]
                add_sample_readings(fig, readings)
                self.assertEqual(fig.get_size_inches()[1], height)


if __name__ == "__main__":
    unittest.main()
