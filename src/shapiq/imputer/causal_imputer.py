"""Causal-aware imputer implementing Causal SHAP with multiple sampling backends."""

from __future__ import annotations

from typing import TYPE_CHECKING, Literal

import numpy as np
from scipy.stats import norm, rankdata

from .gaussian_imputer import GaussianImputer
from shapiq.causal.ordering_graph import OrderingGraph

if TYPE_CHECKING:
    import numpy.typing as npt
    from shapiq.typing import Model

# Type alias for sampling method
SamplingMethod = Literal["gaussian", "copula", "empirical"]


class CausalImputer(GaussianImputer):
    """Causal-aware imputer for computing Causal Shapley values.

    This imputer extends :class:`~shapiq.imputer.GaussianImputer` by respecting
    the causal structure when sampling missing feature values. Instead of sampling
    all missing features jointly, it samples them component-by-component following
    the causal ordering.

    Sampling Methods:

    - ``"gaussian"`` (default): Fast closed-form solution assuming multivariate
      normal distribution. Best for continuous, approximately normal data.

    - ``"copula"``: Gaussian copula approach that preserves original marginal
      distributions while capturing dependencies. Better for non-normal marginals.

    - ``"empirical"``: Non-parametric KNN-based sampling from training data.
      No distributional assumptions but requires sufficient training data.

    Attributes:
        causal_graph: The :class:`~shapiq.causal.OrderingGraph` defining the causal
            structure via ordering and confounding information.
        sampling_method: The sampling backend to use.

    Example:
        >>> import numpy as np
        >>> from shapiq.imputer import CausalImputer
        >>> from shapiq import TabularExplainer
        >>>
        >>> # Define causal structure: x0 -> x1,x2 -> x3,x4
        >>> ordering = [[0], [1, 2], [3, 4]]
        >>> confounding = [False, True, False]  # x1,x2 have unobserved confounder
        >>>
        >>> model = lambda x: x[:, 0] + x[:, 1] * x[:, 2]
        >>> X_train = np.random.rand(500, 5)
        >>>
        >>> # Using Gaussian (default, fastest)
        >>> explainer = TabularExplainer(
        ...     model=model, data=X_train, imputer="causal",
        ...     ordering=ordering, confounding=confounding,
        ... )
        >>>
        >>> # Using Copula (better for non-normal data)
        >>> from shapiq.imputer import CausalImputer
        >>> imputer = CausalImputer(
        ...     model=model, data=X_train,
        ...     ordering=ordering, confounding=confounding,
        ...     sampling_method="copula",
        ... )
        >>> explainer = TabularExplainer(model=model, data=X_train, imputer=imputer)

    See Also:
        - :class:`~shapiq.imputer.GaussianImputer`: The parent class.
        - :class:`~shapiq.imputer.GaussianCopulaImputer`: Copula-based imputer.
        - :class:`~shapiq.causal.OrderingGraph`: The ordering-based causal graph structure.

    References:
        Heskes, T., Sijben, E., Bucur, I. G., & Claassen, T. (2020).
        Causal Shapley Values: Exploiting Causal Knowledge to Explain Individual
        Predictions of Complex Models. NeurIPS 2020.
    """

    causal_graph: OrderingGraph
    """The causal graph defining ordering and confounding."""
    
    sampling_method: SamplingMethod
    """The sampling method to use: 'gaussian', 'copula', or 'empirical'."""
    
    # Copula parameters
    QUANTILE_CLIP_EPSILON = 1e-10
    """Clip epsilon for copula quantile function."""

    def __init__(
        self,
        model: Model,
        data: np.ndarray,
        x: np.ndarray | None = None,
        *,
        ordering: list[list[int]] | None = None,
        confounding: list[bool] | None = None,
        sampling_method: SamplingMethod = "copula",
        sample_size: int = 100,
        k_neighbors: int = 10,
        random_state: int | None = None,
        _model_accepts_gaussian: bool = False,
        _precomputed_empty_prediction: float | None = None,
        **kwargs,
    ) -> None:
        """Initialize the CausalImputer.

        Args:
            model: The model to explain as a callable function expecting data points
                as input and returning the model's predictions.

            data: The background data to use for computing mean and covariance,
                as a two-dimensional array with shape ``(n_samples, n_features)``.

            x: The explanation point as a ``np.ndarray`` of shape ``(1, n_features)``
                or ``(n_features,)``. If ``None``, must call ``fit()`` before use.

            ordering: The causal partial ordering as a list of lists. Each inner list
                contains feature indices that are in the same causal component.
                Earlier components are ancestors of later ones. For example,
                ``[[0], [1, 2], [3, 4]]`` means feature 0 is a root cause,
                features 1 and 2 depend on feature 0, and features 3 and 4 depend
                on all previous features. If ``None``, all features are placed in
                a single component (equivalent to standard Gaussian imputation).

            confounding: A list of boolean values, one per component in ordering.
                If ``confounding[i]`` is ``True``, there is an unobserved confounder
                among variables in component ``i``, so we should NOT condition on
                other intervened variables in the same component. If ``False``,
                we condition on all intervened variables including those in the
                same component. Defaults to all ``False``.

            sampling_method: The sampling backend to use. Options:
                - ``"copula"`` (default): Preserves marginals, better for mixed data.
                - ``"gaussian"``: Fast, assumes multivariate normal.
                - ``"empirical"``: KNN-based, no distributional assumptions.

            sample_size: The number of Monte Carlo samples to draw for imputation.
                Higher values give more accurate estimates but increase computation.

            k_neighbors: Number of neighbors for empirical (KNN) sampling.
                Only used when ``sampling_method="empirical"``. Defaults to 10.

            random_state: Random seed for reproducibility.

            _model_accepts_gaussian: Internal flag. When ``True``, the model expects
                inputs in Gaussian copula space, so ``_sample_causal`` skips the
                inverse copula transform. Used by ``NoMLCausalImputer.as_causal_imputer()``
                to eliminate the double copula round-trip.

            _precomputed_empty_prediction: Internal. Pre-computed empty prediction
                value. When provided, skips the expensive ``calc_empty_prediction()``
                call during initialization.

            **kwargs: Additional keyword arguments passed to the parent class.
        """
        # Skip parent's categorical check by not calling super().__init__ directly
        # Instead, initialize the Imputer base class and set up Gaussian parameters manually
        from .base import Imputer
        Imputer.__init__(
            self,
            model=model,
            data=data,
            x=x,
            categorical_features=[],
            sample_size=sample_size,
            random_state=random_state,
            **kwargs,
        )
        # Initialize Gaussian-specific attributes (skip categorical check)
        self._mean_per_feature = None
        self._cov_mat = None
        
        # Validate sampling method
        valid_methods = ("gaussian", "copula", "empirical")
        if sampling_method not in valid_methods:
            raise ValueError(
                f"sampling_method must be one of {valid_methods}, got '{sampling_method}'"
            )
        self.sampling_method = sampling_method
        self.k_neighbors = k_neighbors
        
        # Default: all variables in one component, no confounding
        if ordering is None:
            ordering = [list(range(self.n_features))]
        if confounding is None:
            confounding = [False] * len(ordering)
            
        self.causal_graph = OrderingGraph(ordering, confounding)
        self._rng = np.random.default_rng(random_state)
        
        # Flag: when True, skip _transform_from_gaussian_full in _sample_causal
        # because the model already expects Gaussian-space inputs.
        self._model_accepts_gaussian = _model_accepts_gaussian
        
        # Initialize method-specific data structures
        self._init_sampling_backend()
        
        # Calculate the empty prediction (model's mean prediction on background data)
        # This is critical for correct SII/Shapley value attribution
        if _precomputed_empty_prediction is not None:
            self.empty_prediction = _precomputed_empty_prediction
        else:
            self.empty_prediction = self.calc_empty_prediction()
    
    def calc_empty_prediction(self) -> float:
        """Calculate the empty prediction (baseline) for Shapley value computation.
        
        The empty prediction represents the model's expected output when no features
        are observed, computed as the mean prediction over the background data.
        This is essential for correct Shapley value attribution.
        
        Returns:
            The empty prediction value (mean of model predictions on background data).
        """
        empty_predictions = self.predict(self.data)
        return float(np.mean(empty_predictions))
    
    def _init_sampling_backend(self) -> None:
        """Initialize data structures specific to the chosen sampling method."""
        if self.sampling_method == "copula":
            # Transform data to Gaussian space for copula
            self._data_transformed = self._transform_to_gaussian(self.data)
            self._copula_mean = np.mean(self._data_transformed, axis=0)
            self._copula_cov = self._ensure_positive_definite(
                np.cov(self._data_transformed.T)
            )
            self._data_sorted = np.sort(self.data, axis=0)
        elif self.sampling_method == "empirical":
            # Pre-compute for KNN: just store data, no transformation needed
            self._empirical_data = self.data.copy()
    
    # --- Copula helper methods ---

    def _transform_to_gaussian(self, data: np.ndarray) -> np.ndarray:
        """Transform data to standard normal using empirical CDF (rank-Gaussian).
        
        Args:
            data: Input data of shape (n_samples, n_features).
            
        Returns:
            Transformed data in Gaussian space.
        """
        n_samples = data.shape[0]
        ranks = rankdata(data, axis=0, method="average")
        empirical_cdf = ranks / n_samples
        empirical_cdf = np.clip(
            empirical_cdf, 
            self.QUANTILE_CLIP_EPSILON, 
            1 - self.QUANTILE_CLIP_EPSILON
        )
        return norm.ppf(empirical_cdf)
    
    def _transform_point_to_gaussian(self, x: np.ndarray) -> np.ndarray:
        """Transform a single point to Gaussian space.
        
        Args:
            x: Point of shape (n_features,).
            
        Returns:
            Transformed point in Gaussian space.
        """
        n_samples = self.data.shape[0]
        ranks = np.sum(self.data <= x, axis=0)
        empirical_cdf = ranks / n_samples
        empirical_cdf = np.clip(
            empirical_cdf,
            self.QUANTILE_CLIP_EPSILON,
            1 - self.QUANTILE_CLIP_EPSILON
        )
        return norm.ppf(empirical_cdf)
    
    def _transform_from_gaussian(
        self, 
        samples_gaussian: np.ndarray, 
        feature_indices: np.ndarray
    ) -> np.ndarray:
        """Transform Gaussian samples back to original feature space.
        
        Args:
            samples_gaussian: Samples in Gaussian space, shape (n_samples, len(feature_indices)).
            feature_indices: Which features these samples correspond to.
            
        Returns:
            Samples in original space.
        """
        n_background = self.data.shape[0]
        quantiles = norm.cdf(samples_gaussian)
        ranks = quantiles * n_background
        rank_indices = np.arange(1, n_background + 1)
        
        result = np.zeros_like(samples_gaussian)
        for i, feat_idx in enumerate(feature_indices):
            sorted_col = self._data_sorted[:, feat_idx]
            result[:, i] = np.interp(ranks[:, i], rank_indices, sorted_col)
        
        return result
    
    # --- Main sampling methods ---

    def _draw_samples(
        self,
        x: npt.NDArray[np.floating],
        coalitions: npt.NDArray[np.bool_],
    ) -> npt.NDArray[np.floating]:
        """Override parent method to use causal sampling."""
        n_coalitions = coalitions.shape[0]
        all_samples = np.zeros((n_coalitions, self.sample_size, self.n_features))
        
        for i, coalition in enumerate(coalitions):
            all_samples[i] = self._sample_causal(x, coalition)
        
        return all_samples
    
    def _sample_causal(
        self, 
        x: np.ndarray, 
        coalition: np.ndarray
    ) -> np.ndarray:
        """Core: Sample by causal order respecting d-separation rules.
        
        This method dispatches to the appropriate sampling backend based on
        the ``sampling_method`` parameter.
        
        Args:
            x: Explanation point of shape (n_features,).
            coalition: Boolean array indicating known/intervened features.
        
        Returns:
            Samples of shape (sample_size, n_features).
        """
        # For copula, work in transformed space
        if self.sampling_method == "copula":
            x_transformed = self._transform_point_to_gaussian(x)
            samples = np.tile(x_transformed, (self.sample_size, 1))
        else:
            samples = np.tile(x, (self.sample_size, 1))
        
        for i, component in enumerate(self.causal_graph.ordering):
            # Variables to sample in this component
            to_sample = np.intersect1d(
                component, 
                np.where(~coalition)[0]  # Unknown variables
            )
            
            if len(to_sample) == 0:
                continue
            
            # Get conditioning set (d-separation logic)
            to_condition = self.causal_graph.get_conditioning_set(i, coalition)
            
            # Dispatch to appropriate sampling method
            if self.sampling_method == "gaussian":
                new_samples = self._sample_gaussian(
                    samples, to_sample, to_condition
                )
            elif self.sampling_method == "copula":
                new_samples = self._sample_copula(
                    samples, to_sample, to_condition
                )
            else:  # empirical
                new_samples = self._sample_empirical(
                    samples, to_sample, to_condition
                )
            
            samples[:, to_sample] = new_samples
        
        # For copula, transform back to original space — but skip if model
        # already accepts Gaussian-space inputs (e.g., NoMLCausalImputer.predict_gaussian)
        if self.sampling_method == "copula" and not self._model_accepts_gaussian:
            samples = self._transform_from_gaussian_full(samples)
        
        return samples
    
    # --- Backend: Gaussian ---

    def _sample_gaussian(
        self, 
        current_samples: np.ndarray, 
        to_sample: np.ndarray, 
        to_condition: np.ndarray
    ) -> np.ndarray:
        """Sample from conditional Gaussian distribution.
        
        P(X_s | X_c) ~ N(cond_mean, cond_cov)
        """
        if len(to_condition) == 0:
            # Marginal sampling
            return self._rng.multivariate_normal(
                self.mean_per_feature[to_sample],
                self.cov_mat[np.ix_(to_sample, to_sample)],
                self.sample_size
            )
        
        mu_s = self.mean_per_feature[to_sample]
        mu_c = self.mean_per_feature[to_condition]
        
        cov_ss = self.cov_mat[np.ix_(to_sample, to_sample)]
        cov_sc = self.cov_mat[np.ix_(to_sample, to_condition)]
        cov_cc = self.cov_mat[np.ix_(to_condition, to_condition)]
        
        cov_cc_inv = np.linalg.pinv(cov_cc)
        regression_coef = cov_sc @ cov_cc_inv
        
        deviation = current_samples[:, to_condition] - mu_c
        cond_mean = mu_s + (deviation @ regression_coef.T)
        
        cond_cov = cov_ss - regression_coef @ cov_sc.T
        cond_cov = 0.5 * (cond_cov + cond_cov.T)
        cond_cov = self._ensure_positive_definite(cond_cov)
        
        Z = self._rng.standard_normal((self.sample_size, len(to_sample)))
        try:
            L = np.linalg.cholesky(cond_cov)
            return cond_mean + Z @ L.T
        except np.linalg.LinAlgError:
            U, s, Vt = np.linalg.svd(cond_cov)
            L = U @ np.diag(np.sqrt(np.maximum(s, 1e-10)))
            return cond_mean + Z @ L.T
    
    # --- Backend: Copula ---

    def _sample_copula(
        self, 
        current_samples: np.ndarray, 
        to_sample: np.ndarray, 
        to_condition: np.ndarray
    ) -> np.ndarray:
        """Sample in Gaussian space (copula), will be back-transformed later.
        
        Uses the same conditional Gaussian formula but in the transformed space.
        """
        if len(to_condition) == 0:
            # Marginal sampling in Gaussian space
            return self._rng.multivariate_normal(
                self._copula_mean[to_sample],
                self._copula_cov[np.ix_(to_sample, to_sample)],
                self.sample_size
            )
        
        mu_s = self._copula_mean[to_sample]
        mu_c = self._copula_mean[to_condition]
        
        cov_ss = self._copula_cov[np.ix_(to_sample, to_sample)]
        cov_sc = self._copula_cov[np.ix_(to_sample, to_condition)]
        cov_cc = self._copula_cov[np.ix_(to_condition, to_condition)]
        
        cov_cc_inv = np.linalg.pinv(cov_cc)
        regression_coef = cov_sc @ cov_cc_inv
        
        deviation = current_samples[:, to_condition] - mu_c
        cond_mean = mu_s + (deviation @ regression_coef.T)
        
        cond_cov = cov_ss - regression_coef @ cov_sc.T
        cond_cov = 0.5 * (cond_cov + cond_cov.T)
        cond_cov = self._ensure_positive_definite(cond_cov)
        
        Z = self._rng.standard_normal((self.sample_size, len(to_sample)))
        try:
            L = np.linalg.cholesky(cond_cov)
            return cond_mean + Z @ L.T
        except np.linalg.LinAlgError:
            U, s, Vt = np.linalg.svd(cond_cov)
            L = U @ np.diag(np.sqrt(np.maximum(s, 1e-10)))
            return cond_mean + Z @ L.T
    
    def _transform_from_gaussian_full(self, samples: np.ndarray) -> np.ndarray:
        """Transform all features from Gaussian space back to original space.
        
        Args:
            samples: Samples in Gaussian space, shape (n_samples, n_features).
            
        Returns:
            Samples in original space.
        """
        return self._transform_from_gaussian(
            samples, np.arange(self.n_features)
        )
    
    # --- Backend: Empirical (KNN) ---

    def _sample_empirical(
        self, 
        current_samples: np.ndarray, 
        to_sample: np.ndarray, 
        to_condition: np.ndarray
    ) -> np.ndarray:
        """Sample using K-nearest neighbors from training data.
        
        For each sample, find k neighbors in the conditioning space,
        then sample from those neighbors' values for the target features.
        """
        if len(to_condition) == 0:
            # Marginal: just sample random rows from training data
            indices = self._rng.choice(
                len(self._empirical_data), 
                size=self.sample_size, 
                replace=True
            )
            return self._empirical_data[indices][:, to_sample]
        
        result = np.zeros((self.sample_size, len(to_sample)))
        
        for i in range(self.sample_size):
            # Find k nearest neighbors based on conditioning features
            query = current_samples[i, to_condition]
            distances = np.sum(
                (self._empirical_data[:, to_condition] - query) ** 2, 
                axis=1
            )
            k = min(self.k_neighbors, len(self._empirical_data))
            neighbor_indices = np.argpartition(distances, k)[:k]
            
            # Sample one neighbor uniformly
            chosen = self._rng.choice(neighbor_indices)
            result[i] = self._empirical_data[chosen, to_sample]
        
        return result
    
    # --- Legacy method for backwards compatibility ---

    def _conditional_sample(self, current_samples, to_sample, to_condition):
        """Legacy method - redirects to _sample_gaussian for backwards compatibility."""
        return self._sample_gaussian(current_samples, to_sample, to_condition)
