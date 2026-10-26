"""Check feature semantics against the bundled raw daily observations."""

from __future__ import annotations

import bike_data
import numpy as np
import pandas as pd


def test_daily_data_preserves_dates_targets_and_feature_units():
    raw = pd.read_csv(bike_data.DATA_PATH)
    X, y = bike_data.load_bike_sharing_daily()
    assert X.shape == (731, 7)
    assert list(X.columns) == bike_data.FEATURE_NAMES
    pd.testing.assert_series_equal(y, raw["cnt"])
    np.testing.assert_array_equal(X["trend"], np.arange(731, dtype=float))
    np.testing.assert_allclose(X["cosyear"] ** 2 + X["sinyear"] ** 2, 1.0)
    np.testing.assert_array_equal(X["temp"], raw["temp"] * 47 - 8)
    np.testing.assert_array_equal(X["atemp"], raw["atemp"] * 66 - 16)
    np.testing.assert_array_equal(X["hum"], raw["hum"] * 100)
    np.testing.assert_array_equal(X["windspeed"], raw["windspeed"] * 67)
