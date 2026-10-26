"""Local daily bike-sharing data and the seven-feature experimental design."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

DATA_PATH = Path(__file__).resolve().parents[1] / "data" / "bike_sharing_daily.csv"
FEATURE_NAMES = ["trend", "cosyear", "sinyear", "temp", "atemp", "windspeed", "hum"]


def load_bike_sharing_daily():
    """Return all 731 dates with the transformations used by this experiment.

    Temperatures are in degrees Celsius, humidity in percent, and wind speed
    on the original dataset scale. No model-specific normalization is applied.
    """
    dataset = pd.read_csv(DATA_PATH)
    dates = pd.to_datetime(dataset["dteday"], errors="raise")
    dataset["trend"] = (dates - dates.iloc[0]).dt.total_seconds() / (24 * 3600)
    dataset["cosyear"] = np.cos(dataset["trend"] / 365 * 2 * np.pi)
    dataset["sinyear"] = np.sin(dataset["trend"] / 365 * 2 * np.pi)
    dataset["temp"] = dataset["temp"] * 47 - 8
    dataset["atemp"] = dataset["atemp"] * 66 - 16
    dataset["windspeed"] = 67 * dataset["windspeed"]
    dataset["hum"] = 100 * dataset["hum"]
    return dataset[FEATURE_NAMES], dataset["cnt"]
