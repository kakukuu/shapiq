"""Fitted contexts exposing every held-out row for explanation.

Graph variants share the fitted model and training-only background array.

The paired Lag comparison uses a common set of target days and a single split
before fitting the two feature designs separately.  It is a random-holdout
attribution comparison, not a prospective forecasting evaluation.
"""

from __future__ import annotations

import hashlib
import json
import pickle
from copy import deepcopy
from dataclasses import dataclass
from typing import Any, Literal

import bike_data
import data_utils as du
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from xgboost import XGBRegressor

from shapiq import TabularExplainer

LAG_FEATURES = ("temp", "hum", "windspeed")


def _frame_fingerprint(frame: pd.DataFrame | pd.Series) -> str:
    """Hash values, index and schema, retaining row order."""
    table = frame.to_frame() if isinstance(frame, pd.Series) else frame
    schema = json.dumps(
        {"columns": list(table.columns), "dtypes": [str(x) for x in table.dtypes]},
        sort_keys=True,
        default=str,
    ).encode("utf-8")
    values = pd.util.hash_pandas_object(table, index=True).to_numpy().tobytes()
    return hashlib.sha256(schema + values).hexdigest()


def _load_data_with_metadata() -> tuple[pd.DataFrame, pd.Series, pd.DataFrame]:
    """Recover dates from the *actual* source CSV after verifying its alignment."""
    X, y = du.load_data()
    X, y = X.copy(), y.copy()
    if not X.index.equals(y.index) or not X.index.is_unique:
        raise ValueError("Features and targets must have aligned unique source indices.")
    metadata = pd.DataFrame(
        {"row_id": np.arange(len(X)), "trend": X["trend"].to_numpy()},
        index=X.index,
    )
    raw_path = bike_data.DATA_PATH
    if raw_path.is_file():
        metadata.attrs["source_csv"] = "data/bike_sharing_daily.csv"
        metadata.attrs["source_csv_sha256"] = hashlib.sha256(raw_path.read_bytes()).hexdigest()
        raw = pd.read_csv(raw_path)
        if "dteday" in raw.columns and "cnt" in raw.columns:
            dates = pd.to_datetime(raw["dteday"], errors="raise")
            if len(raw) != len(X) or not np.array_equal(raw["cnt"].to_numpy(), y.to_numpy()):
                raise ValueError("Raw date metadata do not match the loaded targets.")
            expected_trend = (dates - dates.iloc[0]).dt.total_seconds() / 86400
            if not np.array_equal(expected_trend.to_numpy(), X["trend"].to_numpy()):
                raise ValueError("Raw date metadata do not match the loaded trend.")
            if not dates.is_unique or not dates.is_monotonic_increasing:
                raise ValueError("Daily source dates must be unique and ordered.")
            metadata["date"] = dates.dt.strftime("%Y-%m-%d").to_numpy()
    return X, y, metadata


@dataclass(frozen=True)
class ExperimentContext:
    """One fitted model with its training background and complete held-out rows.

    ``X_test.index`` and metadata ``row_id`` refer to original source rows, even
    for lagged features.  Do not mutate the frames/model/background after fitting.
    ``feature_names`` retain the historical column names; ``feature_semantics``
    states explicitly which design uses previous-day weather values.
    """

    label: str
    model: Any
    X_train: pd.DataFrame
    X_test: pd.DataFrame
    y_train: pd.Series
    y_test: pd.Series
    background: np.ndarray
    train_metadata: pd.DataFrame
    test_metadata: pd.DataFrame
    feature_semantics: dict[str, str]
    seed: int
    test_size: float

    @property
    def feature_names(self) -> list[str]:
        return list(self.X_train.columns)

    @property
    def X_explain(self) -> pd.DataFrame:
        """All held-out observations, with no hidden 50-row truncation."""
        return self.X_test

    def explainer(
        self,
        ordering=None,
        confounding=None,
        imputer: Literal["causal", "marginal"] = "causal",
        seed: int | None = None,
        sample_size: int | str = 500,
    ) -> TabularExplainer:
        """Construct an explainer without splitting data or fitting another model.

        Only ``seed`` controls this new explainer's random state.  Explanation
        budgets and per-explanation seeds are supplied to ``du.explain_multi``.
        """
        explain_seed = self.seed if seed is None else seed
        if sample_size == "all":
            if imputer != "marginal":
                raise ValueError("'all' is only defined for empirical marginal background rows.")
            sample_size = len(self.background)
        if (
            not isinstance(sample_size, (int, np.integer))
            or isinstance(sample_size, bool)
            or sample_size < 1
        ):
            raise ValueError("sample_size must be a positive integer, or 'all' for marginal.")
        if imputer == "causal":
            return TabularExplainer(
                model=self.model.predict,
                data=self.background,
                index="k-SII",
                max_order=2,
                imputer="causal",
                sampling_method="copula",
                sample_size=int(sample_size),
                random_state=explain_seed,
                ordering=deepcopy(du.ORDERING if ordering is None else ordering),
                confounding=deepcopy(du.CONFOUNDING if confounding is None else confounding),
            )
        if imputer == "marginal":
            return TabularExplainer(
                model=self.model.predict,
                data=self.background,
                index="k-SII",
                max_order=2,
                imputer="marginal",
                sampling_method="joint",
                sample_size=int(sample_size),
                random_state=explain_seed,
            )
        raise ValueError("imputer must be 'causal' or 'marginal'.")

    def predict(self, rows: pd.DataFrame | np.ndarray | None = None) -> np.ndarray:
        """Predict the supplied rows, or all held-out rows by default."""
        return np.asarray(self.model.predict(self.X_test if rows is None else rows))

    def metrics(self) -> dict[str, float | int]:
        """Predictive quality on the common held-out target outcome scale."""
        predictions = self.predict()
        return {
            "n_train": len(self.X_train),
            "n_test": len(self.X_test),
            "mae": float(mean_absolute_error(self.y_test, predictions)),
            "rmse": float(np.sqrt(mean_squared_error(self.y_test, predictions))),
            "r2": float(r2_score(self.y_test, predictions)),
        }

    def fingerprints(self) -> dict[str, Any]:
        """JSON-compatible provenance for detecting mixed splits/models/rows.

        The model digest describes its serialized fitted state.  The background
        digest includes the actual ndarray passed to every graph variant.
        """
        if hasattr(self.model, "get_booster"):
            model_bytes = bytes(self.model.get_booster().save_raw())
        else:
            model_bytes = pickle.dumps(self.model, protocol=pickle.HIGHEST_PROTOCOL)
        background_hash = hashlib.sha256()
        background_hash.update(str(self.background.shape).encode("ascii"))
        background_hash.update(str(self.background.dtype).encode("ascii"))
        background_hash.update(np.ascontiguousarray(self.background).tobytes())
        return {
            "label": self.label,
            "split_seed": self.seed,
            "test_size": self.test_size,
            "n_train": len(self.X_train),
            "n_test": len(self.X_test),
            "feature_names": self.feature_names,
            "feature_semantics": self.feature_semantics.copy(),
            "model_class": f"{type(self.model).__module__}.{type(self.model).__qualname__}",
            "model_sha256": hashlib.sha256(model_bytes).hexdigest(),
            "background_sha256": background_hash.hexdigest(),
            "train_rows_sha256": _frame_fingerprint(self.train_metadata["row_id"]),
            "test_rows_sha256": _frame_fingerprint(self.test_metadata["row_id"]),
            "train_features_sha256": _frame_fingerprint(self.X_train),
            "test_features_sha256": _frame_fingerprint(self.X_test),
            "train_targets_sha256": _frame_fingerprint(self.y_train),
            "test_targets_sha256": _frame_fingerprint(self.y_test),
            "train_metadata_sha256": _frame_fingerprint(self.train_metadata),
            "test_metadata_sha256": _frame_fingerprint(self.test_metadata),
            "date_metadata_available": "date" in self.test_metadata,
            "metadata_source": dict(self.test_metadata.attrs),
        }


def _fit_context(
    X: pd.DataFrame,
    y: pd.Series,
    metadata: pd.DataFrame,
    train_ids: np.ndarray,
    test_ids: np.ndarray,
    *,
    label: str,
    seed: int,
    test_size: float,
    model_cls,
    model_kwargs,
    lagged: bool = False,
) -> ExperimentContext:
    if len(np.intersect1d(train_ids, test_ids)):
        raise ValueError("Training and test target rows overlap.")
    X_train, X_test = X.loc[train_ids].copy(), X.loc[test_ids].copy()
    y_train, y_test = y.loc[train_ids].copy(), y.loc[test_ids].copy()
    if model_cls is None:
        model_cls = XGBRegressor
    if model_kwargs is None:
        model_kwargs = dict(n_estimators=200, max_depth=5, learning_rate=0.1, random_state=seed)
    model = model_cls(**dict(model_kwargs))
    model.fit(X_train, y_train)
    semantics = {
        col: "previous-day (t-1)" if lagged and col in LAG_FEATURES else "target-day (t)"
        for col in X.columns
    }
    return ExperimentContext(
        label=label,
        model=model,
        X_train=X_train,
        X_test=X_test,
        y_train=y_train,
        y_test=y_test,
        background=X_train.to_numpy(copy=True),
        train_metadata=metadata.loc[train_ids].copy(),
        test_metadata=metadata.loc[test_ids].copy(),
        feature_semantics=semantics,
        seed=seed,
        test_size=test_size,
    )


def build_context(
    *, seed: int = 42, test_size: float = 0.2, model_cls=None, model_kwargs=None
) -> ExperimentContext:
    """Fit the original 731-day design and expose all 147 held-out days.

    The default split/model settings match ``du.get_explainer`` exactly.  Its
    first 50 held-out rows therefore remain available as ``context.X_test.head(50)``.
    """
    X, y, metadata = _load_data_with_metadata()
    train_ids, test_ids = train_test_split(
        X.index.to_numpy(), test_size=test_size, random_state=seed
    )
    return _fit_context(
        X,
        y,
        metadata,
        train_ids,
        test_ids,
        label="same_day_full",
        seed=seed,
        test_size=test_size,
        model_cls=model_cls,
        model_kwargs=model_kwargs,
    )


def build_paired_lag_contexts(
    *, seed: int = 42, test_size: float = 0.2, model_cls=None, model_kwargs=None
) -> tuple[ExperimentContext, ExperimentContext]:
    """Return (same-day, lag) contexts for the same 730 target days and split.

    Exactly temp, hum and windspeed are shifted by one source row before any
    split.  Source dates, when available, must be consecutive calendar days.
    All target-day labels and unshifted columns retain their original row IDs.
    The first source row is removed from *both* arms, leaving 584 train/146 test
    target days.  Different fitted models make this a joint design/model change,
    not an isolated causal-structure intervention or a temporal forecast.
    """
    X, y, metadata = _load_data_with_metadata()
    if len(X) < 3:
        raise ValueError("At least three source days are needed for paired Lag contexts.")
    if "date" in metadata:
        differences = pd.to_datetime(metadata["date"]).diff().iloc[1:]
        if not (differences == pd.Timedelta(days=1)).all():
            raise ValueError("Lag data require consecutive source calendar days.")
    X_lag = X.copy()
    X_lag.loc[:, list(LAG_FEATURES)] = X.loc[:, list(LAG_FEATURES)].shift(1)
    common_ids = X.index[1:]
    if X_lag.loc[common_ids].isna().any().any() or X.loc[common_ids].isna().any().any():
        raise ValueError("Unexpected missing values in paired feature designs.")
    lag_metadata = metadata.copy()
    lag_metadata["lag_source_row_id"] = metadata["row_id"].shift(1)
    if "date" in metadata:
        lag_metadata["lag_source_date"] = metadata["date"].shift(1)
    lag_metadata = lag_metadata.loc[common_ids].copy()
    lag_metadata["lag_source_row_id"] = lag_metadata["lag_source_row_id"].astype(int)
    train_ids, test_ids = train_test_split(
        common_ids.to_numpy(), test_size=test_size, random_state=seed
    )
    same_day = _fit_context(
        X.loc[common_ids],
        y.loc[common_ids],
        metadata.loc[common_ids],
        train_ids,
        test_ids,
        label="paired_same_day",
        seed=seed,
        test_size=test_size,
        model_cls=model_cls,
        model_kwargs=model_kwargs,
    )
    lag = _fit_context(
        X_lag.loc[common_ids],
        y.loc[common_ids],
        lag_metadata,
        train_ids,
        test_ids,
        label="paired_lag",
        seed=seed,
        test_size=test_size,
        model_cls=model_cls,
        model_kwargs=model_kwargs,
        lagged=True,
    )
    return same_day, lag
