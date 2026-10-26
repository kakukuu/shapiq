"""Implementation of the marginal imputer."""

from __future__ import annotations

from typing import TYPE_CHECKING, Literal

import numpy as np
from scipy.stats import norm, rankdata

from .base import Imputer

if TYPE_CHECKING:
    from shapiq.typing import CoalitionMatrix, GameValues, Model

type MarginalSamplingMethod = Literal["joint", "gaussian", "copula"]

# maximum number of array elements (rows times features, ~8 MB as float64) per model call
_MAX_ELEMENTS_PER_PREDICTION = 2**20


class MarginalImputer(Imputer):
    """The marginal imputer for the shapiq package.

    The marginal imputer replaces missing features of the explanation point ``x`` by values
    sampled from the background data. When ``joint_marginal_distribution=True``, rows are sampled
    **jointly** (i.e., from the empirical joint marginal); when ``False``, each feature column is
    independently shuffled to break dependencies (feature-wise marginals).

    This corresponds to *interventional* imputation (often called *marginal fANOVA* in the
    literature), as opposed to *observational* imputers that condition on observed features.

    The default ``sampling_method="joint"`` uses empirical background rows. Alternatively,
    ``"gaussian"`` fits a multivariate Gaussian distribution and ``"copula"`` fits a Gaussian
    copula with empirical feature marginals. Both parametric methods draw unconditional samples
    shared by all coalitions; observed features are then replaced by their exact values in ``x``.

    Examples:
        >>> model = lambda x: np.sum(x, axis=1)  # some dummy model
        >>> data = np.random.rand(1000, 4)  # some background data
        >>> x_to_impute = np.array([[1, 1, 1, 1]])  # some data point to impute
        >>> imputer = MarginalImputer(model=model, data=data, x=x_to_impute, sample_size=100, random_state=42)
        >>> # get the model prediction with missing values
        >>> imputer(np.array([[True, False, True, False]]))
        np.array([2.01])  # some model prediction (might be different)
        >>> # exchange the background data
        >>> new_data = np.random.rand(1000, 4)
        >>> imputer.init_background(data=new_data)

    See Also:
        - :class:`shapiq.imputer.GenerativeConditionalImputer` for the conditional imputer.
        - :class:`shapiq.imputer.BaselineImputer` for the baseline imputer.
        - :class:`shapiq.imputer.base.Imputer` for the base imputer class.

    """

    joint_marginal_distribution: bool
    """A flag indicating whether to sample from the joint marginal distribution (``True``) or
    independently for each feature (``False``), for ``sampling_method="joint"``."""

    sampling_method: MarginalSamplingMethod
    """The unconditional sampling backend: empirical rows, Gaussian, or Gaussian copula."""

    def __init__(
        self,
        model: Model,
        data: np.ndarray,
        *,
        x: np.ndarray | None = None,
        sample_size: int | None = 100,
        categorical_features: list[int] | None = None,
        joint_marginal_distribution: bool = True,
        sampling_method: MarginalSamplingMethod = "joint",
        normalize: bool = True,
        random_state: int | None = None,
    ) -> None:
        """Initializes the marginal imputer.

        Args:
            model: The model to explain as a callable function expecting a data points as input and
                returning the model's predictions.

            data: The background data to use for the explainer as a two-dimensional array
                with shape ``(n_samples, n_features)``.

            x: The explanation point to use the imputer on either as a 2-dimensional array with
                shape ``(1, n_features)`` or as a vector with shape ``(n_features,)``. If ``None``,
                the imputer must be fitted before it can be used.

            sample_size: The maximum number of background rows used to impute missing features.
                The rows are drawn once without replacement whenever the background data or the
                random state is set, and are shared by all coalitions, including the empty
                coalition (whose value is the baseline value). If ``None`` or if the background
                data has at most ``sample_size`` rows, all rows are used. The runtime grows
                linearly with the number of rows used. For Gaussian and copula sampling, this is
                the number of Monte Carlo draws, with ``None`` using the number of background rows.
                Defaults to ``100``.

            categorical_features: A list of indices of the categorical features. If ``None``, all
                features are treated as continuous.

            joint_marginal_distribution: A flag to sample the replacement values from the joint
                marginal distribution. If ``False``, the replacement values are sampled
                independently for each feature. If ``True``, the replacement values are sampled from
                the joint marginal distribution. Only used for ``sampling_method="joint"``.

            sampling_method: The unconditional sampling backend. ``"joint"`` samples empirical
                rows, ``"gaussian"`` draws from a fitted multivariate Gaussian, and ``"copula"``
                draws from a Gaussian copula with empirical marginals. The parametric methods
                require finite numeric background data with at least two rows.

            normalize: A flag to normalize the game values. If ``True``, then the game values are
                normalized and centered to be zero for the empty set of features.

            random_state: The random state to use for sampling. If ``None``, the random state is not
                fixed.

        Raises:
            ValueError: If ``sample_size`` is not a positive integer or ``None``, the sampling
                method is unknown, or the background data is invalid.
        """
        if sample_size is not None and (
            isinstance(sample_size, bool)
            or not isinstance(sample_size, (int, np.integer))
            or sample_size < 1
        ):
            msg = f"The sample size must be a positive integer or None, but got {sample_size}."
            raise ValueError(msg)
        if sampling_method not in ("joint", "gaussian", "copula"):
            msg = f"Unknown sampling_method {sampling_method!r}; use 'joint', 'gaussian', or 'copula'."
            raise ValueError(msg)
        super().__init__(
            model=model,
            data=data,
            x=x,
            sample_size=sample_size,
            categorical_features=categorical_features,
            random_state=random_state,
        )

        # setup attributes
        self.joint_marginal_distribution = joint_marginal_distribution
        self.sampling_method = sampling_method
        self._normalize_by_empty = normalize
        self._max_sample_size = sample_size  # upper bound requested by the user (None = all rows)
        self._replacement_data: np.ndarray = np.zeros((1, self.n_features))
        self._sampled_replacement_data: np.ndarray = np.zeros((1, self.n_features))
        self.init_background(self.data)

        if normalize:  # update normalization value
            self.normalization_value = self.empty_prediction

    def value_function(self, coalitions: CoalitionMatrix) -> GameValues:
        """Imputes the missing values of a data point and calls the model.

        Args:
            coalitions: A boolean array indicating which features are present (``True``) and which
                are missing (``False``). The shape of the array must be ``(n_subsets, n_features)``.

        Returns:
            The model's predictions on the imputed data points. The shape of the array is
               ``(n_subsets, n_outputs)``.

        """
        x = self.x
        replacement_data = self._sampled_replacement_data
        n_samples, n_features = replacement_data.shape
        n_coalitions = coalitions.shape[0]
        outputs = np.empty(n_coalitions)
        # evaluate as many coalitions per model call as fit into the element budget
        chunk_size = max(1, _MAX_ELEMENTS_PER_PREDICTION // (n_samples * n_features))
        for start in range(0, n_coalitions, chunk_size):
            chunk = coalitions[start : start + chunk_size]
            # one row per (coalition, sample): present features from x, missing from the sample
            present = np.repeat(chunk, n_samples, axis=0)
            imputed_data = np.where(present, x, np.tile(replacement_data, (len(chunk), 1)))
            predictions = np.asarray(self.predict(imputed_data)).reshape(len(chunk), n_samples)
            outputs[start : start + len(chunk)] = predictions.mean(axis=1)  # average over samples
        # the empty prediction is computed on the same replacement samples, inserting it keeps the
        # value of the empty coalition exactly equal to ``empty_prediction``
        outputs[~np.any(coalitions, axis=1)] = self.empty_prediction
        return outputs

    def init_background(self, data: np.ndarray) -> MarginalImputer:
        """Initializes the imputer to a background data set.

        The background data is used to sample replacement values for the missing features. To change
        the background data, use this method.

        Args:
            data: The background data to use for the imputer. The shape of the array must
                be ``(n_samples, n_features)``.

        Returns:
            The initialized imputer.

        Examples:
            >>> model = lambda x: np.sum(x, axis=1)
            >>> data = np.random.rand(10, 3)
            >>> imputer = MarginalImputer(model=model, data=data, x=data[0])
            >>> new_data = np.random.rand(10, 3)
            >>> imputer.init_background(data=new_data)

        Note:
            At most ``sample_size`` rows of the new background data are used (all rows if
            ``sample_size`` is ``None``), independently of earlier background data. Parametric
            methods refit their distribution and redraw ``sample_size`` Monte Carlo samples.

        """
        data = np.asarray(data)
        if data.ndim != 2 or data.shape[0] == 0 or data.shape[1] != self.n_features:
            msg = f"Background data must have at least one row and {self.n_features} columns."
            raise ValueError(msg)
        if self.sampling_method != "joint":
            data = np.asarray(data, dtype=float)
            if data.shape[0] < 2 or not np.all(np.isfinite(data)):
                msg = "Parametric marginal sampling requires at least two finite numeric rows."
                raise ValueError(msg)
        self.data = np.copy(data)
        self._replacement_data = np.copy(data)
        n_rows = self._replacement_data.shape[0]
        if self._max_sample_size is None:
            self._sample_size = n_rows
        elif self.sampling_method == "joint":
            self._sample_size = min(self._max_sample_size, n_rows)
        else:
            self._sample_size = self._max_sample_size
        if self.sampling_method != "joint":
            self._fit_parametric_background()
        # draw the replacement samples once, so that all coalitions use the same samples
        self._sampled_replacement_data = self._sample_replacement_data(self._sample_size)
        self.calc_empty_prediction()  # reset the empty prediction to the new background data
        return self

    def set_random_state(self, random_state: int | None = None) -> None:
        """Sets the random state and redraws the replacement samples with it.

        Args:
            random_state: The random state to set. Defaults to ``None``, which will set a not
                deterministic random state.

        """
        super().set_random_state(random_state)
        self._sampled_replacement_data = self._sample_replacement_data(self._sample_size)
        self.calc_empty_prediction()  # reset the empty prediction to the new samples

    def _sample_replacement_data(self, sample_size: int | None = None) -> np.ndarray:
        """Samples replacement values from the background data.

        Args:
            sample_size: The number of replacement values to sample. If ``None``, all replacement
                rows are sampled for the empirical backend; parametric backends use the configured
                Monte Carlo sample size. Defaults to ``None``.

        Returns:
            The replacement values as a two-dimensional array with shape
                ``(sample_size, n_features)``.

        """
        if self.sampling_method != "joint":
            return self._sample_parametric_data(
                self.sample_size if sample_size is None else sample_size
            )
        replacement_data = np.copy(self._replacement_data)
        rng = np.random.default_rng(self.random_state)
        # shuffle data if not sampling from joint marginal distribution
        if not self.joint_marginal_distribution:
            for feature in range(self.n_features):
                rng.shuffle(replacement_data[:, feature])
        n_samples = replacement_data.shape[0]
        if sample_size is None or sample_size >= n_samples:
            return replacement_data
        # sample replacement values
        replacement_idx = rng.choice(n_samples, size=sample_size, replace=False)
        return replacement_data[replacement_idx]

    def calc_empty_prediction(self) -> float:
        """Runs the model on empty data points (all features missing) to get the empty prediction.

        Returns:
            The empty prediction of the model provided only missing features.

        """
        empty_predictions = self.predict(self._sampled_replacement_data)
        empty_prediction = float(np.mean(empty_predictions))
        self.empty_prediction = empty_prediction
        if self._normalize_by_empty:  # preserve normalization even when the old baseline was zero
            self.normalization_value = empty_prediction
        return empty_prediction

    def _fit_parametric_background(self) -> None:
        """Fit the unconditional distribution to the current replacement background."""
        data = self._replacement_data
        if self.sampling_method == "copula":
            self._data_sorted = np.sort(data, axis=0)
            quantiles = rankdata(data, axis=0, method="average") / data.shape[0]
            data = norm.ppf(np.clip(quantiles, 1e-10, 1 - 1e-10))
        self._gauss_mean = np.mean(data, axis=0)
        covariance = np.atleast_2d(np.cov(data, rowvar=False))
        covariance = 0.5 * (covariance + covariance.T)
        smallest_eigenvalue = float(np.min(np.linalg.eigvalsh(covariance)))
        covariance += max(0.0, 1e-6 - smallest_eigenvalue) * np.eye(self.n_features)
        self._gauss_cov = covariance

    def _sample_parametric_data(self, sample_size: int) -> np.ndarray:
        """Draw shared unconditional replacement samples from the fitted distribution."""
        rng = np.random.default_rng(self.random_state)
        samples = (
            rng.standard_normal((sample_size, self.n_features))
            @ np.linalg.cholesky(self._gauss_cov).T
            + self._gauss_mean
        )
        if self.sampling_method == "copula":
            n_rows = self._data_sorted.shape[0]
            ranks = norm.cdf(samples) * n_rows
            rank_indices = np.arange(1, n_rows + 1)
            for feature in range(self.n_features):
                samples[:, feature] = np.interp(
                    ranks[:, feature], rank_indices, self._data_sorted[:, feature]
                )
        return samples
