"""Split, provenance and fitted-context invariants (no expensive explanations)."""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import data_utils as du
import experiment_context as ec
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split


class RecordingRegressor:
    """Tiny fitted predictor whose recorded inputs expose target leakage."""

    def fit(self, X, y):
        self.fitted_rows = X.index.copy()
        self.fitted_targets = y.copy()
        self.mean = float(y.mean())
        return self

    def predict(self, X):
        return np.full(len(X), self.mean)


class ContextTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        kwargs = {"model_cls": RecordingRegressor, "model_kwargs": {}}
        cls.full = ec.build_context(**kwargs)
        cls.same, cls.lag = ec.build_paired_lag_contexts(**kwargs)

    def test_full_split_preserves_historical_first_fifty(self):
        X, y = du.load_data()
        X_train, X_test, _, _ = train_test_split(X, y, test_size=0.2, random_state=42)
        pd.testing.assert_frame_equal(self.full.X_train, X_train)
        pd.testing.assert_frame_equal(self.full.X_test, X_test)
        self.assertEqual((len(self.full.X_train), len(self.full.X_explain)), (584, 147))
        self.assertIs(self.full.X_explain, self.full.X_test)

    def test_paired_target_rows_dates_and_outcomes(self):
        self.assertEqual((len(self.same.X_train), len(self.same.X_test)), (584, 146))
        pd.testing.assert_index_equal(self.same.X_train.index, self.lag.X_train.index)
        pd.testing.assert_index_equal(self.same.X_test.index, self.lag.X_test.index)
        pd.testing.assert_series_equal(self.same.y_train, self.lag.y_train)
        pd.testing.assert_series_equal(self.same.y_test, self.lag.y_test)
        for field in ("row_id", "date", "trend"):
            pd.testing.assert_series_equal(
                self.same.test_metadata[field], self.lag.test_metadata[field]
            )
        self.assertNotIn(0, self.same.X_train.index)
        self.assertNotIn(0, self.same.X_test.index)

    def test_lags_are_previous_original_rows(self):
        original, _ = du.load_data()
        for rows, meta in (
            (self.lag.X_train, self.lag.train_metadata),
            (self.lag.X_test, self.lag.test_metadata),
        ):
            ids = meta["row_id"].to_numpy()
            for feature in ec.LAG_FEATURES:
                np.testing.assert_array_equal(rows[feature], original.iloc[ids - 1][feature])
            for feature in set(du.FEATURE_NAMES) - set(ec.LAG_FEATURES):
                np.testing.assert_array_equal(rows[feature], original.iloc[ids][feature])
            np.testing.assert_array_equal(meta["lag_source_row_id"], ids - 1)
            dates = pd.to_datetime(meta["date"]) - pd.to_datetime(meta["lag_source_date"])
            self.assertTrue((dates == pd.Timedelta(days=1)).all())

    def test_train_only_fit_and_background(self):
        for context in (self.full, self.same, self.lag):
            self.assertFalse(set(context.X_train.index) & set(context.X_test.index))
            pd.testing.assert_index_equal(context.model.fitted_rows, context.X_train.index)
            pd.testing.assert_series_equal(context.model.fitted_targets, context.y_train)
            np.testing.assert_array_equal(context.background, context.X_train.to_numpy())
            self.assertEqual(len(context.predict()), len(context.X_test))

    def test_graph_variants_share_fitted_model_and_background(self):
        with patch.object(ec, "TabularExplainer") as constructor:
            self.full.explainer()
            self.full.explainer(ordering=du.ORDERING_B2, confounding=du.CONFOUNDING_B2, seed=7)
            self.full.explainer(imputer="marginal", seed=19)
        for call in constructor.call_args_list:
            self.assertIs(call.kwargs["data"], self.full.background)
            self.assertIs(call.kwargs["model"].__self__, self.full.model)
        self.assertEqual(constructor.call_args_list[1].kwargs["random_state"], 7)
        self.assertEqual(constructor.call_args_list[2].kwargs["random_state"], 19)

    def test_provenance_distinguishes_design_but_matches_target_rows(self):
        same, lag = self.same.fingerprints(), self.lag.fingerprints()
        self.assertEqual(same["train_rows_sha256"], lag["train_rows_sha256"])
        self.assertEqual(same["test_rows_sha256"], lag["test_rows_sha256"])
        self.assertEqual(same["test_targets_sha256"], lag["test_targets_sha256"])
        self.assertNotEqual(same["background_sha256"], lag["background_sha256"])
        self.assertNotEqual(same["test_features_sha256"], lag["test_features_sha256"])
        self.assertEqual(
            self.same.test_metadata["date"].iloc[0], self.lag.test_metadata["date"].iloc[0]
        )
        self.assertEqual(self.lag.feature_semantics["atemp"], "target-day (t)")
        self.assertEqual(self.lag.feature_semantics["temp"], "previous-day (t-1)")

    def test_sample_sizes_are_explicit_and_all_uses_training_background(self):
        with patch.object(ec, "TabularExplainer") as constructor:
            self.full.explainer()
            self.full.explainer(sample_size=1000)
            self.full.explainer(imputer="marginal", sample_size="all")
        calls = constructor.call_args_list
        self.assertEqual(calls[0].kwargs["sample_size"], 500)
        self.assertEqual(calls[1].kwargs["sample_size"], 1000)
        self.assertEqual(calls[2].kwargs["sample_size"], len(self.full.X_train))
        self.assertEqual(calls[2].kwargs["sampling_method"], "joint")
        with patch.object(ec, "TabularExplainer") as constructor:
            for size in ("all", 0, -1, True, 1.5):
                with self.subTest(size=size), self.assertRaises(ValueError):
                    self.full.explainer(sample_size=size)
        constructor.assert_not_called()

    def test_data_relocation_preserves_model_and_data_fingerprints(self):
        original = ec.build_context(model_cls=RecordingRegressor, model_kwargs={})
        with tempfile.TemporaryDirectory() as directory:
            relocated = Path(directory) / "bike_sharing_daily.csv"
            relocated.write_bytes(ec.bike_data.DATA_PATH.read_bytes())
            with patch.object(ec.bike_data, "DATA_PATH", relocated):
                copied = ec.build_context(model_cls=RecordingRegressor, model_kwargs={})
        self.assertEqual(original.fingerprints(), copied.fingerprints())
        self.assertEqual(
            copied.fingerprints()["metadata_source"]["source_csv"], "data/bike_sharing_daily.csv"
        )


if __name__ == "__main__":
    unittest.main()
