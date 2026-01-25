"""This module contains functions to load datasets."""

from __future__ import annotations

from pathlib import Path
from typing import cast

import numpy as np
import pandas as pd

GITHUB_DATA_URL = "https://raw.githubusercontent.com/mmschlk/shapiq/main/data/"

# csv files are located next to this file in a folder called "data"
SHAPIQ_DATASETS_FOLDER = Path(__file__).parent / "data"


def _create_folder() -> None:
    """Create the datasets folder if it does not exist."""
    Path(SHAPIQ_DATASETS_FOLDER).mkdir(parents=True, exist_ok=True)


def _try_load(csv_file_name: str) -> pd.DataFrame:
    """Try to load a dataset from the local folder.

    If it does not exist, load it from GitHub and save it to the local folder.

    Args:
        csv_file_name: The name of the csv file to load.

    Returns:
        The dataset as a pandas DataFrame.

    """
    _create_folder()
    path = Path(SHAPIQ_DATASETS_FOLDER) / csv_file_name
    try:
        return pd.read_csv(path)
    except FileNotFoundError:
        data = pd.read_csv(GITHUB_DATA_URL + csv_file_name)
        data.to_csv(path, index=False)
        return data


def load_california_housing(
    *,
    to_numpy: bool = False,
) -> tuple[pd.DataFrame, pd.Series | pd.DataFrame] | tuple[np.ndarray, np.ndarray]:
    """Load the California housing dataset.

    Args:
        to_numpy: Return numpy objects instead of pandas. Default is ``False``.

    Returns:
        The California housing dataset as a pandas DataFrame.

    Example:
        >>> from shapiq.datasets import load_california_housing
        >>> x_data, y_data = load_california_housing()
        >>> print(x_data.shape, y_data.shape)
        ((20640, 8), (20640,))

    """
    dataset = _try_load("california_housing.csv")
    class_label = "MedHouseVal"
    y_data = dataset[class_label]
    x_data = dataset.drop(columns=[class_label])

    if to_numpy:
        return x_data.to_numpy(), y_data.to_numpy()
    return x_data, y_data


def load_bike_sharing_daily(
    *, to_numpy: bool = False, preprocessed: bool = True
) -> tuple[pd.DataFrame, pd.Series | pd.DataFrame] | tuple[np.ndarray, np.ndarray]:
    """Load the daily bike-sharing dataset used in CauSHAPley paper.

    This is the daily aggregated version (731 rows) from UCI, used for
    reproducing Heskes et al. (2020) "Causal Shapley Values" NeurIPS paper.

    Args:
        to_numpy: Return numpy objects instead of pandas. Default is ``False``.
        preprocessed: Apply the same preprocessing as CauSHAPley paper. Default is ``True``.
            This includes computing trend, cosyear, sinyear and unnormalizing temp/atemp/etc.

    Returns:
        The daily bike-sharing dataset as a pandas DataFrame.

    Example:
        >>> from shapiq.datasets import load_bike_sharing_daily
        >>> x_data, y_data = load_bike_sharing_daily()
        >>> print(x_data.shape, y_data.shape)
        ((731, 7), (731,))

    Note:
        The features returned when preprocessed=True are:
        - trend: Days since 1 January 2011
        - cosyear: cos(2π × trend / 365) - seasonal cycle
        - sinyear: sin(2π × trend / 365) - seasonal cycle
        - temp: Temperature in Celsius (unnormalized from [-8, 39])
        - atemp: Feeling temperature in Celsius (unnormalized from [-16, 50])
        - windspeed: Wind speed (unnormalized, max 67)
        - hum: Humidity percentage (unnormalized, 0-100)

    """
    dataset = _try_load("bike_sharing_daily.csv")

    if preprocessed:
        # Parse dates and compute trend (Days since 1 January 2011)
        dataset['dteday'] = pd.to_datetime(dataset['dteday'])
        dataset['trend'] = (dataset['dteday'] - dataset['dteday'].iloc[0]).dt.total_seconds() / (24 * 3600)

        # Compute cyclical season variables (matching CauSHAPley paper)
        dataset['cosyear'] = np.cos(dataset['trend'] / 365 * 2 * np.pi)
        dataset['sinyear'] = np.sin(dataset['trend'] / 365 * 2 * np.pi)

        # Unnormalize variables (see UCI dataset documentation and CauSHAPley paper)
        dataset['temp'] = dataset['temp'] * (39 - (-8)) + (-8)
        dataset['atemp'] = dataset['atemp'] * (50 - (-16)) + (-16)
        dataset['windspeed'] = 67 * dataset['windspeed']
        dataset['hum'] = 100 * dataset['hum']

        # Select features used in CauSHAPley paper
        feature_names = ['trend', 'cosyear', 'sinyear', 'temp', 'atemp', 'windspeed', 'hum']
        x_data = dataset[feature_names]
        y_data = dataset['cnt']
    else:
        # Return raw data
        class_label = 'cnt'
        y_data = dataset[class_label]
        x_data = dataset.drop(columns=[class_label, 'instant', 'dteday', 'casual', 'registered'])

    if to_numpy:
        return x_data.to_numpy(), y_data.to_numpy()
    return x_data, y_data


def load_bike_sharing_daily_train_index() -> np.ndarray:
    """Load the training indices for bike_sharing_daily dataset.

    These indices are from the original CauSHAPley paper to ensure reproducibility.

    Returns:
        Array of training indices (587 samples out of 731 total).

    Example:
        >>> from shapiq.datasets import load_bike_sharing_daily_train_index
        >>> train_idx = load_bike_sharing_daily_train_index()
        >>> print(len(train_idx))
        587

    """
    path = Path(SHAPIQ_DATASETS_FOLDER) / "train_index.npy"
    return np.load(path)


def load_bike_sharing(
    *, to_numpy: bool = False
) -> tuple[pd.DataFrame, pd.Series | pd.DataFrame] | tuple[np.ndarray, np.ndarray]:
    """Load the bike-sharing dataset from openml and preprocess it.

    Note:
        The function requires the `sklearn` package to be installed.

    Args:
        to_numpy: Return numpy objects instead of pandas. ``Default is False.``

    Returns:
        The bike-sharing dataset as a pandas DataFrame.

    Example:
        >>> from shapiq.datasets import load_bike_sharing
        >>> x_data, y_data = load_bike_sharing()
        >>> print(x_data.shape, y_data.shape)
        ((17379, 12), (17379,))

    """
    from sklearn.compose import ColumnTransformer
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import OrdinalEncoder, RobustScaler

    dataset: pd.DataFrame = _try_load("bike.csv")
    class_label = "count"

    num_feature_names = [
        "hour",
        "temp",
        "feel_temp",
        "humidity",
        "windspeed",
        "year",
        "month",
        "holiday",
        "weekday",
        "workingday",
    ]
    cat_feature_names = [
        "season",
        "weather",
    ]
    dataset[num_feature_names] = dataset[num_feature_names].apply(pd.to_numeric)
    num_pipeline = Pipeline([("scaler", RobustScaler())])
    cat_pipeline = Pipeline(
        [
            ("ordinal_encoder", OrdinalEncoder()),
        ],
    )
    column_transformer = ColumnTransformer(
        [
            ("numerical", num_pipeline, num_feature_names),
            ("categorical", cat_pipeline, cat_feature_names),
        ],
        remainder="passthrough",
    )
    col_names = num_feature_names + cat_feature_names
    col_names += [feature for feature in dataset.columns if feature not in col_names]
    transformed_data: np.ndarray = cast(
        np.ndarray, column_transformer.fit_transform(dataset)
    )  # Transformations will always return a dense array
    dataset = pd.DataFrame(
        transformed_data,
        columns=np.asarray(col_names),
    )
    dataset = dataset.dropna()

    y_data = dataset.pop(class_label)
    x_data = dataset

    if to_numpy:
        return x_data.to_numpy(), y_data.to_numpy()
    return x_data, y_data


def load_adult_census(
    *, to_numpy: bool = False
) -> tuple[pd.DataFrame, pd.Series | pd.DataFrame] | tuple[np.ndarray, np.ndarray]:
    """Load the adult census dataset from the UCI Machine Learning Repository.

    Original source: https://archive.ics.uci.edu/ml/datasets/adult

    Note:
        The function requires the `sklearn` package to be installed.

    Args:
        to_numpy: Return numpy objects instead of pandas. Default is ``False``.

    Returns:
        The adult census dataset as a pandas DataFrame.

    Example:
        >>> from shapiq.datasets import load_adult_census
        >>> x_data, y_data = load_adult_census()
        >>> print(x_data.shape, y_data.shape)
        ((45222, 14), (45222,))

    """
    from sklearn.compose import ColumnTransformer
    from sklearn.impute import SimpleImputer
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import OrdinalEncoder, StandardScaler

    dataset = _try_load("adult_census.csv")
    class_label = "class"

    num_feature_names = ["age", "capital-gain", "capital-loss", "hours-per-week", "fnlwgt"]
    cat_feature_names = [
        "workclass",
        "education",
        "marital-status",
        "occupation",
        "relationship",
        "race",
        "sex",
        "native-country",
        "education-num",
    ]
    dataset[num_feature_names] = dataset[num_feature_names].apply(pd.to_numeric)
    num_pipeline = Pipeline(
        [("imputer", SimpleImputer(strategy="median")), ("std_scaler", StandardScaler())],
    )
    cat_pipeline = Pipeline(
        [
            ("ordinal_encoder", OrdinalEncoder()),
        ],
    )
    column_transformer = ColumnTransformer(
        [
            ("numerical", num_pipeline, num_feature_names),
            ("categorical", cat_pipeline, cat_feature_names),
        ],
        remainder="passthrough",
    )
    col_names = num_feature_names + cat_feature_names
    col_names += [feature for feature in dataset.columns if feature not in col_names]
    transformed_data = cast(
        np.ndarray, column_transformer.fit_transform(dataset)
    )  # Transformations will always return a dense array
    dataset = pd.DataFrame(
        transformed_data,
        columns=np.asarray(col_names),
    )
    dataset = dataset.dropna()

    y_data = dataset.pop(class_label)
    x_data = dataset.astype(float)

    # transform '>50K' to 1 and '<=50K' to 0
    y_data = y_data.apply(lambda x: 1 if x == ">50K" else 0)

    if to_numpy:
        return x_data.to_numpy(), y_data.to_numpy()
    return x_data, y_data
