"""No-ML Causal Imputer for model-free Causal SHAP computation."""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
from numpy.linalg import pinv
from scipy.stats import norm, rankdata

from .causal_imputer import CausalImputer, SamplingMethod

if TYPE_CHECKING:
    import numpy.typing as npt


class NoMLCausalImputer:
    """Unified No-ML Causal Imputer with built-in Y estimation.

    This class provides a "model-free" approach to Causal SHAP by treating
    Y as the final node in the causal graph and using conditional Gaussian
    estimation to predict Y from X.

    The key insight is that both feature imputation and Y estimation use
    the same mathematical formula: mu_{s|c} = mu_s + Sigma_{sc} Sigma_{cc}^{-1} (x_c - mu_c).
    For Y estimation, s = Y and c = X (all features).
    
    **Important**: The Y estimation method is now consistent with the chosen
    sampling_method. When using 'copula', both X imputation and Y estimation
    work in the Gaussian copula space, preserving marginal distributions.

    Attributes:
        X: The feature data of shape ``(n_samples, n_features)``.
        y: The target values of shape ``(n_samples,)``.
        n_features: Number of features.
        ordering: Causal partial ordering of features.
        confounding: Confounding flags for each component.
        mu_X: Mean of features (in transformed space for copula).
        mu_Y: Mean of target (in transformed space for copula).
        beta_Y: Regression coefficients for Y estimation.

    Properties:
        empty_prediction: Returns the baseline value for SHAP (mean of Y in original space).
    """
    
    def __init__(
        self,
        X: npt.NDArray[np.floating],
        y: npt.NDArray[np.floating],
        ordering: list[list[int]] | None = None,
        confounding: list[bool] | None = None,
        sampling_method: SamplingMethod = "copula",
        sample_size: int = 200,
        k_neighbors: int = 10,
        random_state: int | None = None,
    ) -> None:
        """Initialize the NoMLCausalImputer.
        
        Args:
            X: Feature data of shape ``(n_samples, n_features)``.
            
            y: Target values of shape ``(n_samples,)``.
            
            ordering: The causal partial ordering as a list of lists. Each inner 
                list contains feature indices in the same causal component.
                If ``None``, all features are in a single component.
                
            confounding: A list of boolean values, one per component. If 
                ``confounding[i]`` is ``True``, there is an unobserved confounder
                among variables in component ``i``.
                
            sampling_method: Sampling method for feature imputation.
                - ``"copula"`` (default): Preserves marginal distributions.
                - ``"gaussian"``: Assumes multivariate normal.
                - ``"empirical"``: KNN-based sampling.
                
            sample_size: Number of Monte Carlo samples for imputation.
            
            k_neighbors: Number of neighbors for empirical sampling.
            
            random_state: Random seed for reproducibility.
        """
        self.X = np.atleast_2d(X)
        self.y = np.atleast_1d(y).flatten()
        
        if len(self.X) != len(self.y):
            raise ValueError(
                f"X and y must have same number of samples. "
                f"Got X: {len(self.X)}, y: {len(self.y)}"
            )
        
        self.n_samples, self.n_features = self.X.shape
        self.ordering = ordering if ordering is not None else [list(range(self.n_features))]
        self.confounding = confounding if confounding is not None else [False] * len(self.ordering)
        self.sampling_method: SamplingMethod = sampling_method
        self.sample_size = sample_size
        self.k_neighbors = k_neighbors
        self.random_state = random_state
        
        # Copula parameters
        self.QUANTILE_CLIP_EPSILON = 1e-10
        
        # Store original Y mean for baseline (before any transformation)
        self._original_y_mean = float(np.mean(self.y))
        
        # Compute unified covariance matrix (X, Y combined)
        # Method depends on sampling_method
        self._compute_unified_covariance()
        
        # For shapiq compatibility
        self.data = self.X
        self.n_players = self.n_features
        
    def _compute_unified_covariance(self) -> None:
        """Compute the unified covariance matrix for (X, Y).
        
        For 'copula' method, data is first transformed to Gaussian space.
        For 'gaussian' method, data is used directly.
        For 'empirical' method, KNN-based estimation is used.
        """
        XY = np.column_stack([self.X, self.y])
        
        if self.sampling_method == "copula":
            # Transform to Gaussian space for copula method
            XY_gaussian = self._transform_to_gaussian(XY)
            self._mu_XY = np.mean(XY_gaussian, axis=0)
            self._cov_XY = np.cov(XY_gaussian, rowvar=False)
            
            # Store sorted data for inverse transformation
            self._XY_sorted = np.sort(XY, axis=0)
            self._y_sorted = np.sort(self.y)
        else:
            # Gaussian and empirical: work in original space
            self._mu_XY = np.mean(XY, axis=0)
            self._cov_XY = np.cov(XY, rowvar=False)
        
        # Ensure positive definiteness
        self._cov_XY = self._ensure_positive_definite(self._cov_XY)
        
        # Extract components (in transformed space for copula)
        self.mu_X = self._mu_XY[:self.n_features]
        self.mu_Y = self._mu_XY[self.n_features]
        
        self.Sigma_XX = self._cov_XY[:self.n_features, :self.n_features]
        self.Sigma_YX = self._cov_XY[self.n_features, :self.n_features]
        self.Sigma_YY = self._cov_XY[self.n_features, self.n_features]
        
        # Compute regression coefficients: beta = Sigma_YX @ Sigma_XX^{-1}
        self.beta_Y = self.Sigma_YX @ pinv(self.Sigma_XX)
    
    # --- Copula transformation methods ---
    
    def _transform_to_gaussian(self, data: np.ndarray) -> np.ndarray:
        """Transform data to standard normal using empirical CDF (rank-Gaussian).
        
        Args:
            data: Input data of shape (n_samples, n_features) or (n_samples,).
            
        Returns:
            Transformed data in Gaussian space.
        """
        data = np.atleast_2d(data)
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
        """Transform points' X values to Gaussian space (vectorized).
        
        Args:
            x: Point of shape (n_features,) or (n_points, n_features).
            
        Returns:
            Transformed point(s) in Gaussian space.
        """
        x = np.atleast_2d(x)
        n_points = x.shape[0]
        
        # Vectorized: process all points at once using broadcasting
        # For large batches, chunk to avoid memory issues
        # x: (n_points, n_features), self.X: (n_samples, n_features)
        _CHUNK_SIZE = 200
        if n_points <= _CHUNK_SIZE:
            # (n_points, n_samples, n_features) boolean -> sum over n_samples axis
            ranks = np.sum(self.X[None, :, :] <= x[:, None, :], axis=1)  # (n_points, n_features)
        else:
            ranks = np.zeros((n_points, x.shape[1]), dtype=np.float64)
            for start in range(0, n_points, _CHUNK_SIZE):
                end = min(start + _CHUNK_SIZE, n_points)
                ranks[start:end] = np.sum(
                    self.X[None, :, :] <= x[start:end, None, :], axis=1
                )
        
        empirical_cdf = ranks / self.n_samples
        empirical_cdf = np.clip(
            empirical_cdf,
            self.QUANTILE_CLIP_EPSILON,
            1 - self.QUANTILE_CLIP_EPSILON
        )
        return norm.ppf(empirical_cdf)
    
    def _transform_y_from_gaussian(self, z_y: np.ndarray) -> np.ndarray:
        """Transform Y values from Gaussian space back to original space.
        
        Uses linear interpolation based on sorted Y values.
        
        Args:
            z_y: Y values in Gaussian space, shape (n_points,).
            
        Returns:
            Y values in original space.
        """
        z_y = np.atleast_1d(z_y)
        quantiles = norm.cdf(z_y)
        ranks = quantiles * self.n_samples
        
        # Linear interpolation using sorted Y values
        rank_indices = np.arange(1, self.n_samples + 1)
        y_original = np.interp(ranks, rank_indices, self._y_sorted)
        
        return y_original
        
    def _ensure_positive_definite(
        self, 
        cov_mat: np.ndarray, 
        min_eigenvalue: float = 1e-6
    ) -> np.ndarray:
        """Ensure covariance matrix is positive definite."""
        eigenvalues = np.linalg.eigvalsh(cov_mat)
        if np.any(eigenvalues <= min_eigenvalue):
            min_eig = np.min(eigenvalues)
            cov_mat = cov_mat + (min_eigenvalue - min_eig) * np.eye(len(cov_mat))
        return cov_mat
    
    @property
    def empty_prediction(self) -> float:
        """Return the baseline prediction (mean of Y in original space)."""
        return self._original_y_mean
    
    def predict(self, X_query: npt.NDArray[np.floating]) -> npt.NDArray[np.floating]:
        """Predict Y using the method consistent with sampling_method.

        For 'gaussian': Uses the formula Y_hat = mu_Y + beta_Y^T (X - mu_X) directly.
        For 'copula': Transforms X to Gaussian space, computes conditional expectation,
                      then transforms Y back to original space.
        For 'empirical': Uses KNN-based conditional expectation.

        Args:
            X_query: Query points of shape ``(n_points, n_features)`` or
                ``(n_features,)``.

        Returns:
            Predictions of shape ``(n_points,)`` or ``(1,)`` for single point.
        """
        X_query = np.atleast_2d(X_query)
        
        if self.sampling_method == "copula":
            # 1. Transform X to Gaussian space
            X_gaussian = self._transform_point_to_gaussian(X_query)
            # 2. Compute conditional expectation in Gaussian space
            deviation = X_gaussian - self.mu_X
            Z_Y = self.mu_Y + deviation @ self.beta_Y
            # 3. Transform Y back to original space
            result = self._transform_y_from_gaussian(Z_Y)
        elif self.sampling_method == "empirical":
            # KNN-based conditional expectation
            result = self._predict_empirical(X_query)
        else:
            # Gaussian: direct computation in original space
            deviation = X_query - self.mu_X
            result = self.mu_Y + deviation @ self.beta_Y
        
        return np.atleast_1d(result.squeeze())

    def predict_gaussian(self, X_gaussian: npt.NDArray[np.floating]) -> npt.NDArray[np.floating]:
        """Predict Y from X already in Gaussian copula space.

        This avoids the expensive ``_transform_point_to_gaussian`` step when the
        input is already in Gaussian space (e.g., from ``CausalImputer._sample_causal``).
        For non-copula methods, this falls back to ``predict()``.

        Args:
            X_gaussian: Query points already in Gaussian space, of shape
                ``(n_points, n_features)`` or ``(n_features,)``.

        Returns:
            Predictions of shape ``(n_points,)`` or ``(1,)`` for single point.
        """
        if self.sampling_method != "copula":
            # For non-copula, X_gaussian is in original space; delegate to predict
            return self.predict(X_gaussian)

        X_gaussian = np.atleast_2d(X_gaussian)
        # Conditional expectation in Gaussian space (just a matrix multiply)
        deviation = X_gaussian - self.mu_X
        Z_Y = self.mu_Y + deviation @ self.beta_Y
        # Transform Y back to original space
        result = self._transform_y_from_gaussian(Z_Y)
        return np.atleast_1d(result.squeeze())
    
    def _predict_empirical(self, X_query: np.ndarray) -> np.ndarray:
        """KNN-based Y estimation for empirical method.
        
        Args:
            X_query: Query points of shape (n_points, n_features).
            
        Returns:
            Predicted Y values based on k-nearest neighbors.
        """
        from scipy.spatial.distance import cdist
        
        n_points = X_query.shape[0]
        result = np.zeros(n_points)
        
        # Compute distances to all training points
        distances = cdist(X_query, self.X)
        
        for i in range(n_points):
            # Find k nearest neighbors
            k = min(self.k_neighbors, self.n_samples)
            neighbor_indices = np.argpartition(distances[i], k)[:k]
            # Return mean of neighbor Y values
            result[i] = np.mean(self.y[neighbor_indices])
        
        return result
    
    def __call__(self, X_query: npt.NDArray[np.floating]) -> npt.NDArray[np.floating]:
        """Make the object callable for compatibility with shapiq."""
        return self.predict(X_query)
    
    def as_causal_imputer(self) -> CausalImputer:
        """Create a CausalImputer using the same covariance structure.
        
        This ensures consistency between X imputation and Y estimation,
        as both use the same underlying covariance matrix.
        
        When ``sampling_method="copula"``, this passes ``predict_gaussian``
        as the model and sets ``_model_accepts_gaussian=True`` so that
        ``CausalImputer`` skips the redundant inverse copula transform,
        avoiding a costly Gaussian -> Original -> Gaussian round-trip.
        
        Returns:
            A CausalImputer configured with the same parameters.
        """
        # For copula: pass the Gaussian-space predict to avoid double transform
        use_gaussian = self.sampling_method == "copula"
        model_fn = self.predict_gaussian if use_gaussian else self.predict
        
        return CausalImputer(
            model=model_fn,
            data=self.X,
            ordering=self.ordering,
            confounding=self.confounding,
            sampling_method=self.sampling_method,
            sample_size=self.sample_size,
            k_neighbors=self.k_neighbors,
            random_state=self.random_state,
            _model_accepts_gaussian=use_gaussian,
            _precomputed_empty_prediction=self._original_y_mean,
        )
    
    def get_regression_info(self) -> dict:
        """Return information about the Y estimation model.
        
        Useful for understanding and debugging the Y estimation.
        
        Returns:
            Dictionary with sampling_method, mu_Y (original), beta_Y, R², etc.
        """
        # Compute R² for diagnostics (in original space)
        y_pred = self.predict(self.X)
        ss_res = np.sum((self.y - y_pred) ** 2)
        ss_tot = np.sum((self.y - self._original_y_mean) ** 2)
        r_squared = 1 - ss_res / ss_tot if ss_tot > 0 else 0.0
        
        return {
            'sampling_method': self.sampling_method,
            'mu_X': self.mu_X,
            'mu_Y': self.mu_Y,
            'mu_Y_original': self._original_y_mean,
            'Sigma_XX': self.Sigma_XX,
            'Sigma_YX': self.Sigma_YX,
            'beta_Y': self.beta_Y,
            'r_squared': r_squared,
            'n_samples': self.n_samples,
            'n_features': self.n_features,
        }
    
    def fit(self, x: npt.NDArray[np.floating]) -> "NoMLCausalImputer":
        """Fit method for compatibility with shapiq Imputer interface.
        
        This is a no-op since NoMLCausalImputer doesn't need per-point fitting.
        
        Args:
            x: Explanation point (ignored for this imputer).
            
        Returns:
            Self for method chaining.
        """
        return self
