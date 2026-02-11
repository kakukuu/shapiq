"""Implementation of the marginal imputer."""

from __future__ import annotations

import warnings
from typing import TYPE_CHECKING, Literal

import numpy as np
from numpy.random import default_rng

from .base import Imputer

if TYPE_CHECKING:
    from shapiq.typing import CoalitionMatrix, GameValues, Model

MarginalSamplingMethod = Literal["joint", "gaussian", "copula"]

_too_large_sample_size_warning = (
    "The sample size is larger than the number of data points in the background set. "
    "Reducing the sample size to the number of background samples."
)


class MarginalImputer(Imputer):
    """The marginal imputer for the shapiq package.

    The marginal imputer replaces missing features of the explanation point ``x`` by values
    sampled from the unconditional marginal distribution $P(X_{\bar{S}})$, independent of the
    observed features $x_S$.

    Three sampling backends are supported via ``sampling_method``:

    - ``"joint"`` (default): Empirical resampling from training data rows. When
      ``joint_marginal_distribution=True``, rows are sampled **jointly**; when ``False``, each
      feature column is independently shuffled.
    - ``"gaussian"``: Parametric sampling from $\mathcal{N}(\mu_{\bar{S}}, \Sigma_{\bar{S}\bar{S}})$.
      Matches the Marginal Shapley Value (MSV) in Heskes et al. (2020, NeurIPS).
    - ``"copula"``: Gaussian copula sampling that preserves original marginal distributions
      while modeling dependencies through a Gaussian copula.

    All three methods compute **Marginal Shapley Values** (interventional imputation),
    as opposed to *observational/conditional* imputers that condition on observed features.

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
        - :class:`shapiq.imputer.ConditionalImputer` for the conditional imputer.
        - :class:`shapiq.imputer.BaselineImputer` for the baseline imputer.
        - :class:`shapiq.imputer.base.Imputer` for the base imputer class.

    """

    joint_marginal_distribution: bool
    """A flag indicating whether to sample from the joint marginal distribution (``True``) or
    independently for each feature (``False``). Only used when ``sampling_method="joint"``."""

    sampling_method: MarginalSamplingMethod
    """The sampling backend: ``"joint"`` (empirical), ``"gaussian"`` (parametric Gaussian),
    or ``"copula"`` (Gaussian copula)."""

    def __init__(
        self,
        model: Model,
        data: np.ndarray,
        *,
        x: np.ndarray | None = None,
        sample_size: int = 100,
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

            sample_size: The number of samples to draw from the background data. Increasing this
                value will linearly increase the runtime of the explainer.

            categorical_features: A list of indices of the categorical features. If ``None``, all
                features are treated as continuous.

            joint_marginal_distribution: A flag to sample the replacement values from the joint
                marginal distribution. If ``False``, the replacement values are sampled
                independently for each feature. If ``True``, the replacement values are sampled from
                the joint marginal distribution. Only used when ``sampling_method="joint"``.

            sampling_method: The sampling backend for the marginal distribution. Options:
                ``"joint"`` (default): empirical resampling from training data;
                ``"gaussian"``: parametric multivariate Gaussian;
                ``"copula"``: Gaussian copula preserving original marginals.

            normalize: A flag to normalize the game values. If ``True``, then the game values are
                normalized and centered to be zero for the empty set of features.

            random_state: The random state to use for sampling. If ``None``, the random state is not
                fixed.
        """
        valid_methods = ("joint", "gaussian", "copula")
        if sampling_method not in valid_methods:
            msg = f"sampling_method must be one of {valid_methods}, got '{sampling_method}'"
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
        self._replacement_data: np.ndarray = np.zeros((1, self.n_features))

        # Precompute statistics for parametric methods
        if sampling_method in ("gaussian", "copula"):
            if sampling_method == "copula":
                self._data_sorted: np.ndarray = np.sort(data, axis=0)
                transformed = self._transform_to_gaussian(data)
                self._gauss_mean: np.ndarray = np.mean(transformed, axis=0)
                self._gauss_cov: np.ndarray = self._ensure_positive_definite(
                    np.cov(transformed.T)
                )
            else:
                self._gauss_mean = np.mean(data, axis=0).astype(np.float64)
                self._gauss_cov = self._ensure_positive_definite(
                    np.cov(data.T).astype(np.float64)
                )

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
        if self.sampling_method in ("gaussian", "copula"):
            return self._value_function_parametric(coalitions)
        return self._value_function_empirical(coalitions)

    def _value_function_empirical(self, coalitions: CoalitionMatrix) -> GameValues:
        """Original empirical resampling value function."""
        n_coalitions = coalitions.shape[0]
        replacement_data = self._sample_replacement_data(self.sample_size)
        sample_size = replacement_data.shape[0]
        outputs = np.zeros((sample_size, n_coalitions))
        imputed_data = np.tile(self.x, (n_coalitions, 1))
        for i in range(self.sample_size):
            replacements = np.tile(replacement_data[i], (n_coalitions, 1))
            imputed_data[~coalitions] = replacements[~coalitions]
            predictions = self.predict(imputed_data)
            outputs[i] = predictions
        outputs = np.mean(outputs, axis=0)  # average over the samples
        # insert the better approximate empty prediction for the empty coalitions
        outputs[~np.any(coalitions, axis=1)] = self.empty_prediction
        return outputs

    def _value_function_parametric(self, coalitions: CoalitionMatrix) -> GameValues:
        """Parametric (Gaussian/Copula) marginal sampling value function.

        For each coalition, samples the unknown features from the marginal distribution
        N(μ_unk, Σ_unk_unk) — independent of the observed features x_S.
        For copula, samples are transformed back to original feature space.
        """
        x_explain = self.x.flatten()
        n_coalitions = coalitions.shape[0]
        rng = default_rng(self.random_state)
        outputs = np.zeros(n_coalitions)

        for i, coalition in enumerate(coalitions):
            known_indices = np.where(coalition)[0]
            unknown_indices = np.where(~coalition)[0]

            if len(unknown_indices) == 0:
                # All features known — just predict
                outputs[i] = float(self.predict(x_explain.reshape(1, -1))[0])
            elif len(known_indices) == 0:
                # No features known — use precomputed empty prediction
                outputs[i] = self.empty_prediction
            else:
                # Sample unknown features from MARGINAL (not conditional)
                mu_unk = self._gauss_mean[unknown_indices]
                cov_unk = self._gauss_cov[np.ix_(unknown_indices, unknown_indices)]
                cov_unk = self._ensure_positive_definite(0.5 * (cov_unk + cov_unk.T))

                Z = rng.standard_normal((self.sample_size, len(unknown_indices)))
                samples_unknown = Z @ np.linalg.cholesky(cov_unk).T + mu_unk

                if self.sampling_method == "copula":
                    samples_unknown = self._inverse_transform_copula(
                        samples_unknown, unknown_indices
                    )

                # Build full sample array: known = x_explain, unknown = sampled
                samples = np.tile(x_explain, (self.sample_size, 1))
                samples[:, unknown_indices] = samples_unknown
                outputs[i] = float(np.mean(self.predict(samples)))

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

        Raises:
            UserWarning: If the sample size is larger than the number of data points in the
                background data. In this case, the sample size is reduced to the number of data
                points in the background data.

        """
        self._replacement_data = np.copy(data)
        if self._sample_size > self._replacement_data.shape[0]:
            warnings.warn(UserWarning(_too_large_sample_size_warning), stacklevel=2)
            self._sample_size = self._replacement_data.shape[0]
        self.calc_empty_prediction()  # reset the empty prediction to the new background data
        return self

    def _sample_replacement_data(self, sample_size: int | None = None) -> np.ndarray:
        """Samples replacement values from the background data.

        Args:
            sample_size: The number of replacement values to sample. If ``None``, all replacement
                values are sampled. Defaults to ``None``.

        Returns:
            The replacement values as a two-dimensional array with shape
                ``(sample_size, n_features)``.

        """
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
        if self.sampling_method in ("gaussian", "copula"):
            return self._calc_empty_prediction_parametric()
        background_data = self._sample_replacement_data()
        empty_predictions = self.predict(background_data)
        empty_prediction = float(np.mean(empty_predictions))
        self.empty_prediction = empty_prediction
        if self.normalize:  # reset the normalization value
            self.normalization_value = empty_prediction
        return empty_prediction

    def _calc_empty_prediction_parametric(self) -> float:
        """Compute empty prediction for parametric methods using MC samples from full marginal."""
        rng = default_rng(self.random_state)
        Z = rng.standard_normal((self.sample_size, self.n_features))
        full_cov = self._ensure_positive_definite(0.5 * (self._gauss_cov + self._gauss_cov.T))
        samples = Z @ np.linalg.cholesky(full_cov).T + self._gauss_mean

        if self.sampling_method == "copula":
            all_indices = np.arange(self.n_features)
            samples = self._inverse_transform_copula(samples, all_indices)

        empty_prediction = float(np.mean(self.predict(samples)))
        self.empty_prediction = empty_prediction
        if self.normalize:
            self.normalization_value = empty_prediction
        return empty_prediction

    # -------------------------------------------------------------------------
    # Parametric helper methods
    # -------------------------------------------------------------------------

    @staticmethod
    def _ensure_positive_definite(
        cov_mat: np.ndarray,
        min_allowed_eigen_value: float = 1e-06,
    ) -> np.ndarray:
        """Ensure covariance matrix is positive definite by correcting eigenvalues if necessary.

        Args:
            cov_mat: The input covariance matrix.
            min_allowed_eigen_value: The minimum allowed eigenvalue. Defaults to ``1e-06``.

        Returns:
            The positive definite covariance matrix.
        """
        if cov_mat.ndim == 0:
            # Scalar (single feature) — just ensure positive
            return np.atleast_2d(max(float(cov_mat), min_allowed_eigen_value))

        eigen_values = np.linalg.eigvalsh(cov_mat)
        if np.any(eigen_values <= min_allowed_eigen_value):
            min_eigen_value = np.min(eigen_values)
            cov_mat = cov_mat + (min_allowed_eigen_value - min_eigen_value) * np.eye(
                cov_mat.shape[0]
            )
        return cov_mat

    # -------------------------------------------------------------------------
    # Copula helper methods
    # -------------------------------------------------------------------------

    _QUANTILE_CLIP_EPSILON: float = 1e-10

    def _transform_to_gaussian(self, data: np.ndarray) -> np.ndarray:
        """Transform each feature to standard normal via empirical CDF (rank-Gaussian).

        Args:
            data: Input data of shape ``(n_samples, n_features)``.

        Returns:
            Transformed data in Gaussian space.
        """
        from scipy.stats import norm, rankdata

        ranks = rankdata(data, axis=0, method="average")
        empirical_cdf = ranks / data.shape[0]
        empirical_cdf = np.clip(
            empirical_cdf, self._QUANTILE_CLIP_EPSILON, 1 - self._QUANTILE_CLIP_EPSILON
        )
        return norm.ppf(empirical_cdf)

    def _inverse_transform_copula(
        self, samples_gaussian: np.ndarray, feature_indices: np.ndarray
    ) -> np.ndarray:
        """Transform Gaussian-space samples back to original feature space for given features.

        Args:
            samples_gaussian: Samples in Gaussian space of shape ``(n_samples, n_features_subset)``.
            feature_indices: Indices of the features corresponding to columns of ``samples_gaussian``.

        Returns:
            Samples in original feature space of shape ``(n_samples, n_features_subset)``.
        """
        from scipy.stats import norm

        n_training = self.data.shape[0]
        quantiles = norm.cdf(samples_gaussian)
        ranks = quantiles * n_training
        rank_indices = np.arange(1, n_training + 1)

        result = np.zeros_like(samples_gaussian)
        for col_idx, feat_idx in enumerate(feature_indices):
            sorted_col = self._data_sorted[:, feat_idx]
            result[:, col_idx] = np.interp(ranks[:, col_idx], rank_indices, sorted_col)

        return result
