"""do-Shapley Imputer implementing causal contributions via do-interventions.

This module implements the DML (Double Machine Learning) estimator for computing
causal Shapley values based on do-interventions, as described in:

    "On Measuring Causal Contributions via do-interventions"

The key difference from standard SHAP or Causal SHAP is that this imputer estimates
E[Y | do(X_S = x_S)] (interventional expectation) rather than E[Y | X_S = x_S]
(conditional expectation), which correctly handles confounding.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Literal

import numpy as np

from shapiq.causal import DAGGraph

from .base import Imputer

if TYPE_CHECKING:
    import numpy.typing as npt
    from shapiq.typing import Model

# Type alias for estimation method
EstimationMethod = Literal["dml", "ipw", "reg"]


class DoCausalImputer(Imputer):
    """Imputer for computing do-Shapley values via do-interventions.

    This imputer estimates E[Y | do(X_S = x_S)] using Double Machine Learning (DML),
    which combines inverse probability weighting (IPW) and regression to achieve
    doubly-robust estimation of causal effects.

    The do-intervention do(X_S = x_S) sets variables X_S to values x_S while
    cutting all incoming causal edges to X_S, unlike conditioning which preserves
    dependencies through confounders.

    Attributes:
        dag: The :class:`~shapiq.causal.DAGGraph` representing the causal structure.
        method: Estimation method ("dml", "ipw", or "reg").

    Example:
        >>> import numpy as np
        >>> from shapiq.imputer import DoCausalImputer
        >>> from shapiq import TabularExplainer
        >>>
        >>> # Define causal DAG: X0 -> X1, X0 -> X2, X1 -> X2
        >>> dag_edges = [(0, 1), (0, 2), (1, 2)]
        >>>
        >>> model = lambda x: x[:, 0] + 2 * x[:, 1] + x[:, 2]
        >>> X_train = np.random.rand(500, 3)
        >>>
        >>> imputer = DoCausalImputer(
        ...     model=model,
        ...     data=X_train,
        ...     dag_edges=dag_edges,
        ... )
        >>> explainer = TabularExplainer(model=model, data=X_train, imputer=imputer)

    See Also:
        :class:`~shapiq.causal.DAGGraph`: The DAG structure used by this imputer.
        :class:`~shapiq.imputer.CausalImputer`: For Causal SHAP (ordering-based).

    References:
        Jung, Y., Tian, J., & Bareinboim, E. (2022).
        On Measuring Causal Contributions via do-interventions.
        ICML 2022.
    """

    # Constants for numerical stability
    EPS = 1e-4
    """Small epsilon for numerical stability in probability estimates."""

    CLIP_VALUE = 3.0
    """Maximum value for IPW weights to prevent extreme values."""

    def __init__(
        self,
        model: Model,
        data: np.ndarray,
        dag_edges: list[tuple[int, int]],
        x: np.ndarray | None = None,
        *,
        confounding_pairs: set[tuple[int, int]] | None = None,
        method: EstimationMethod = "dml",
        cross_fitting: bool = False,
        sample_size: int = 100,
        random_state: int | None = None,
        categorical_features: list[int] | None = None,
        **kwargs,
    ) -> None:
        """Initialize the DoCausalImputer.

        Args:
            model: The model to explain as a callable function expecting data points
                as input and returning the model's predictions.

            data: The background data to use for estimation as a two-dimensional
                array with shape ``(n_samples, n_features)``.

            dag_edges: List of directed edges defining the causal DAG. Each edge
                is a tuple ``(parent, child)`` indicating parent -> child.
                For example, ``[(0, 1), (0, 2), (1, 2)]`` means:
                - Feature 0 causes features 1 and 2
                - Feature 1 causes feature 2

            x: The explanation point as a ``np.ndarray`` of shape ``(1, n_features)``
                or ``(n_features,)``. If ``None``, must call ``fit()`` before use.

            confounding_pairs: Set of variable pairs that have unobserved common
                causes. For example, ``{(0, 2)}`` means features 0 and 2 have a
                hidden confounder. This is represented as bidirectional edges
                (0 <-> 2) in the causal graph literature.

            method: Estimation method for E[Y | do(X_S)]:
                - ``"dml"`` (default): Double Machine Learning, doubly-robust.
                - ``"ipw"``: Inverse Probability Weighting only.
                - ``"reg"``: Regression/plug-in estimator only.

            cross_fitting: Whether to use 2-fold cross-fitting for DML estimation.
                Reduces bias but doubles computation. Defaults to ``False``.

            sample_size: Not directly used by DoCausalImputer but kept for
                interface compatibility.

            random_state: Random seed for reproducibility.

            categorical_features: List of indices of categorical features.
                These will use classification models for conditional probability.

            **kwargs: Additional keyword arguments passed to the parent class.
        """
        super().__init__(
            model=model,
            data=data,
            x=x,
            sample_size=sample_size,
            categorical_features=categorical_features,
            random_state=random_state,
            **kwargs,
        )

        # Validate estimation method
        valid_methods = ("dml", "ipw", "reg")
        if method not in valid_methods:
            raise ValueError(
                f"method must be one of {valid_methods}, got '{method}'"
            )

        # Build DAG structure using DAGGraph
        self.dag = DAGGraph(
            edges=dag_edges,
            confounding_pairs=confounding_pairs or set(),
            n_features=self.n_features,
        )

        self.method = method
        self.cross_fitting = cross_fitting

        # Expose some DAG properties for convenience
        self.dag_edges = dag_edges
        self.confounding_pairs = self.dag.confounding_pairs
        self.topo_order = self.dag.topo_order

        # Pre-train conditional probability models P(V_i | pre(V_i))
        self._cond_prob_models: dict = {}
        self._train_conditional_models()

        # Cache for computed do-effects
        self._cache: dict[str, float] = {}

        # Calculate empty prediction
        self.empty_prediction = self._calc_empty_prediction()

    def _get_parents(self, node: int) -> list[int]:
        """Get parents of a node from DAG.

        Args:
            node: The node index.

        Returns:
            List of parent node indices.
        """
        return self.dag.get_parents(node)

    def _get_predecessors(self, node: int) -> list[int]:
        """Get all predecessors of a node in topological order.

        Args:
            node: The node index.

        Returns:
            List of predecessor indices in topological order.
        """
        return self.dag.get_predecessors(node)

    def _sort_by_topo(self, S: list[int] | np.ndarray) -> list[int]:
        """Sort a subset S according to topological order.

        Args:
            S: Subset of feature indices.

        Returns:
            S sorted by topological order.
        """
        return self.dag.sort_by_topo(S)

    def _train_conditional_models(self) -> None:
        """Train conditional probability models P(V_i | pre(V_i)) for all variables.

        Uses GradientBoosting for regression/classification depending on whether
        the variable is categorical.
        """
        from sklearn.ensemble import GradientBoostingClassifier, GradientBoostingRegressor

        for idx, var_idx in enumerate(self.topo_order):
            if idx == 0:
                # First variable: compute marginal distribution
                values = self.data[:, var_idx]
                unique_vals = np.unique(values)
                probs = np.array([np.mean(values == v) for v in unique_vals])
                self._cond_prob_models[var_idx] = {
                    "type": "marginal",
                    "unique": unique_vals,
                    "probs": probs,
                }
            else:
                # Variables with predecessors: train conditional model
                pre_indices = self.topo_order[:idx]
                X_pre = self.data[:, pre_indices]
                y_var = self.data[:, var_idx]

                if var_idx in self._cat_features:
                    model = GradientBoostingClassifier(
                        n_estimators=50,
                        max_depth=3,
                        random_state=self.random_state,
                    )
                    model.fit(X_pre, y_var)
                    self._cond_prob_models[var_idx] = {
                        "type": "classifier",
                        "model": model,
                        "pre_indices": pre_indices,
                    }
                else:
                    model = GradientBoostingRegressor(
                        n_estimators=50,
                        max_depth=3,
                        random_state=self.random_state,
                    )
                    model.fit(X_pre, y_var)
                    self._cond_prob_models[var_idx] = {
                        "type": "regressor",
                        "model": model,
                        "pre_indices": pre_indices,
                    }

    def _estimate_cond_prob(
        self,
        S: list[int],
        X_eval: np.ndarray,
    ) -> dict[int, np.ndarray]:
        """Estimate conditional probabilities P(V_i | pre(V_i)) for variables in S.

        Args:
            S: Subset of variable indices to estimate probabilities for.
            X_eval: Data to evaluate probabilities on, shape (n_samples, n_features).

        Returns:
            Dictionary mapping variable index to array of probability estimates.
        """
        eval_probs: dict[int, np.ndarray] = {}
        S_sorted = self._sort_by_topo(S)

        for var_idx in S_sorted:
            model_info = self._cond_prob_models[var_idx]

            if model_info["type"] == "marginal":
                # Marginal probability
                values = X_eval[:, var_idx]
                unique_vals = model_info["unique"]
                probs_map = dict(zip(unique_vals, model_info["probs"]))
                # Default to mean probability for unseen values
                default_prob = np.mean(model_info["probs"])
                prob_array = np.array([
                    probs_map.get(v, default_prob) for v in values
                ])
                eval_probs[var_idx] = prob_array

            elif model_info["type"] == "classifier":
                # Classification probability
                pre_indices = model_info["pre_indices"]
                X_pre = X_eval[:, pre_indices]
                model = model_info["model"]
                # Get probability of the actual class
                proba = model.predict_proba(X_pre)
                classes = model.classes_
                values = X_eval[:, var_idx]
                prob_array = np.zeros(len(values))
                for i, v in enumerate(values):
                    if v in classes:
                        class_idx = list(classes).index(v)
                        prob_array[i] = proba[i, class_idx]
                    else:
                        prob_array[i] = 1.0 / len(classes)  # Uniform for unseen
                eval_probs[var_idx] = prob_array

            else:  # regressor
                # For continuous variables, use Gaussian density approximation
                pre_indices = model_info["pre_indices"]
                X_pre = X_eval[:, pre_indices]
                model = model_info["model"]
                predictions = model.predict(X_pre)
                # Compute residuals to estimate variance
                residuals = X_eval[:, var_idx] - predictions
                sigma = np.std(residuals) + self.EPS
                # Gaussian density (we use this as a proxy for probability)
                from scipy.stats import norm
                prob_array = norm.pdf(X_eval[:, var_idx], loc=predictions, scale=sigma)
                eval_probs[var_idx] = prob_array + self.EPS

        return eval_probs

    def _construct_omega(
        self,
        S: list[int],
        x_explain: np.ndarray,
        X_eval: np.ndarray,
        cond_probs: dict[int, np.ndarray],
    ) -> dict[int, np.ndarray]:
        """Construct IPW weights ω_S = ∏_{i∈S} I(V_i=v_i) / P(V_i | pre(V_i)).

        Args:
            S: Subset of variables being intervened on.
            x_explain: The explanation point values.
            X_eval: Data to evaluate weights on.
            cond_probs: Pre-computed conditional probabilities.

        Returns:
            Dictionary mapping variable index to cumulative IPW weights.
        """
        omega: dict[int, np.ndarray] = {}
        S_sorted = self._sort_by_topo(S)
        n_samples = X_eval.shape[0]

        prev_omega = np.ones(n_samples)

        for idx, k in enumerate(S_sorted):
            # Check if there are variables between k and next element in S
            if idx == len(S_sorted) - 1:
                # Last element: check remaining variables
                k_topo_idx = self.topo_order.index(k)
                Ck = self.topo_order[k_topo_idx + 1:]
            else:
                k_next = S_sorted[idx + 1]
                k_topo_idx = self.topo_order.index(k)
                k_next_topo_idx = self.topo_order.index(k_next)
                Ck = self.topo_order[k_topo_idx + 1:k_next_topo_idx]

            if not Ck:
                continue

            # Get P(V_k | pre(V_k))
            pred_k = cond_probs[k]

            # Compute indicator I(V_k = v_k)
            v_k = x_explain[k]
            V_k = X_eval[:, k]

            # For continuous variables, use soft indicator
            if k not in self._cat_features:
                # Gaussian kernel as soft indicator
                sigma = np.std(self.data[:, k]) + self.EPS
                indicator = np.exp(-0.5 * ((V_k - v_k) / sigma) ** 2)
            else:
                indicator = (V_k == v_k).astype(float)

            # Compute weight: I(V_k=v_k) / P(V_k | pre(V_k))
            weight = indicator / (pred_k + self.EPS)
            weight = np.clip(weight, 0, self.CLIP_VALUE)

            # Cumulative weight
            new_omega = weight * prev_omega
            new_omega = np.clip(new_omega, 0, self.CLIP_VALUE)

            omega[k] = new_omega
            prev_omega = new_omega

        return omega

    def _construct_theta(
        self,
        S: list[int],
        x_explain: np.ndarray,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_test: np.ndarray,
        y_test: np.ndarray,
    ) -> tuple[dict[int, np.ndarray], dict[int, np.ndarray]]:
        """Construct conditional expectations θ_{k,1} and θ_{k,2} for k ∈ S.

        θ_{k,1} = E[Y | v_{<k}, V_{≥k}]  (with intervention up to k-1)
        θ_{k,2} = E[Y | V_k, pre(V_k)]   (without intervention at k)

        Args:
            S: Subset of variables being intervened on.
            x_explain: The explanation point values.
            X_train: Training data for model fitting.
            y_train: Training labels.
            X_test: Test data for evaluation.
            y_test: Test labels.

        Returns:
            Tuple of (theta_1, theta_2) dictionaries.
        """
        from sklearn.ensemble import GradientBoostingRegressor

        S_sorted = self._sort_by_topo(S)
        S_sorted_reverse = list(reversed(S_sorted))

        theta_1: dict[int, np.ndarray] = {}
        theta_2: dict[int, np.ndarray] = {}

        # Initialize with y values
        theta_k1_train = y_train.copy()
        theta_k1_test = y_test.copy()

        for idx, k in enumerate(S_sorted_reverse):
            # Check Ck: variables between k and next element in S_reverse
            if idx == 0:
                k_topo_idx = self.topo_order.index(k)
                Ck = self.topo_order[k_topo_idx + 1:]
            else:
                k_prev = S_sorted_reverse[idx - 1]
                k_topo_idx = self.topo_order.index(k)
                k_prev_topo_idx = self.topo_order.index(k_prev)
                Ck = self.topo_order[k_topo_idx + 1:k_prev_topo_idx]

            if not Ck:
                continue

            theta_1[k] = theta_k1_test.copy()

            # Get columns: V_k and all predecessors
            k_topo_idx = self.topo_order.index(k)
            col_choice = self.topo_order[:k_topo_idx + 1]

            X_choice_train = X_train[:, col_choice]
            X_choice_test = X_test[:, col_choice]

            # Train E[θ_{k,1} | V_k, pre(V_k)]
            model = GradientBoostingRegressor(
                n_estimators=50,
                max_depth=3,
                random_state=self.random_state,
            )
            model.fit(X_choice_train, theta_k1_train)

            theta_k2_test = model.predict(X_choice_test)
            theta_2[k] = theta_k2_test

            # Compute θ_{k-1,1} = E[θ_{k,1} | v_k, pre(V_k)]
            # Replace V_k with intervention value v_k
            X_choice_test_intervened = X_choice_test.copy()
            X_choice_train_intervened = X_choice_train.copy()
            X_choice_test_intervened[:, -1] = x_explain[k]
            X_choice_train_intervened[:, -1] = x_explain[k]

            theta_k1_test = model.predict(X_choice_test_intervened)
            theta_k1_train = model.predict(X_choice_train_intervened)

        # θ_{0,1}: final value after all interventions
        theta_1[-1] = theta_k1_test

        return theta_1, theta_2

    def _compute_dml(
        self,
        omega: dict[int, np.ndarray],
        theta_1: dict[int, np.ndarray],
        theta_2: dict[int, np.ndarray],
    ) -> float:
        """Compute DML estimate given omega and theta.

        DML estimate: E[θ_{0,1}] + Σ_{k∈S} E[ω_k * (θ_{k,1} - θ_{k,2})]

        Args:
            omega: IPW weights from _construct_omega.
            theta_1: θ_{k,1} values from _construct_theta.
            theta_2: θ_{k,2} values from _construct_theta.

        Returns:
            DML estimate of E[Y | do(X_S = x_S)].
        """
        # Start with θ_{0,1}
        eif = theta_1[-1].copy()

        # Add IPW correction terms
        for k in omega:
            if k in theta_1 and k in theta_2:
                eif = eif + omega[k] * (theta_1[k] - theta_2[k])

        return float(np.mean(eif))

    def _compute_ipw(
        self,
        S: list[int],
        x_explain: np.ndarray,
        X_eval: np.ndarray,
        y_eval: np.ndarray,
    ) -> float:
        """Compute IPW estimate of E[Y | do(X_S = x_S)].

        Args:
            S: Subset of intervened variables.
            x_explain: Intervention values.
            X_eval: Evaluation data.
            y_eval: Evaluation labels.

        Returns:
            IPW estimate.
        """
        cond_probs = self._estimate_cond_prob(S, X_eval)
        omega = self._construct_omega(S, x_explain, X_eval, cond_probs)

        if not omega:
            return float(np.mean(y_eval))

        # Get final omega (product of all weights)
        final_omega = list(omega.values())[-1]
        return float(np.mean(final_omega * y_eval))

    def _compute_reg(
        self,
        S: list[int],
        x_explain: np.ndarray,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_test: np.ndarray,
    ) -> float:
        """Compute regression/plug-in estimate of E[Y | do(X_S = x_S)].

        Args:
            S: Subset of intervened variables.
            x_explain: Intervention values.
            X_train: Training data.
            y_train: Training labels.
            X_test: Test data for evaluation.

        Returns:
            Regression estimate.
        """
        from sklearn.ensemble import GradientBoostingRegressor

        # Create intervened data
        X_intervened = X_test.copy()
        for k in S:
            X_intervened[:, k] = x_explain[k]

        # Train outcome model
        model = GradientBoostingRegressor(
            n_estimators=50,
            max_depth=3,
            random_state=self.random_state,
        )
        model.fit(X_train, y_train)

        predictions = model.predict(X_intervened)
        return float(np.mean(predictions))

    def _compute_do_effect(self, S: list[int], x_explain: np.ndarray) -> float:
        """Compute E[Y | do(X_S = x_S)] using the specified method.

        Args:
            S: List of feature indices to intervene on.
            x_explain: The explanation point (intervention values).

        Returns:
            Estimated causal effect E[Y | do(X_S = x_S)].
        """
        # Check cache
        cache_key = self._keygen(S)
        if cache_key in self._cache:
            return self._cache[cache_key]

        # Handle boundary cases
        if len(S) == 0:
            result = self.empty_prediction
        elif len(S) == self.n_features:
            result = float(self.predict(x_explain.reshape(1, -1))[0])
        else:
            if self.cross_fitting:
                result = self._compute_do_effect_cross_fitting(S, x_explain)
            else:
                result = self._compute_do_effect_single(S, x_explain)

        self._cache[cache_key] = result
        return result

    def _compute_do_effect_single(self, S: list[int], x_explain: np.ndarray) -> float:
        """Compute do-effect without cross-fitting."""
        X_data = self.data
        y_data = self.predict(X_data)

        if self.method == "dml":
            cond_probs = self._estimate_cond_prob(S, X_data)
            omega = self._construct_omega(S, x_explain, X_data, cond_probs)
            theta_1, theta_2 = self._construct_theta(
                S, x_explain, X_data, y_data, X_data, y_data
            )
            return self._compute_dml(omega, theta_1, theta_2)

        elif self.method == "ipw":
            return self._compute_ipw(S, x_explain, X_data, y_data)

        else:  # reg
            return self._compute_reg(S, x_explain, X_data, y_data, X_data)

    def _compute_do_effect_cross_fitting(
        self, S: list[int], x_explain: np.ndarray
    ) -> float:
        """Compute do-effect with 2-fold cross-fitting for reduced bias."""
        n = len(self.data)
        indices = np.arange(n)
        self._rng.shuffle(indices)
        mid = n // 2

        fold1_idx, fold2_idx = indices[:mid], indices[mid:]
        X1, X2 = self.data[fold1_idx], self.data[fold2_idx]
        y1 = self.predict(X1)
        y2 = self.predict(X2)

        results = []
        for X_train, y_train, X_test, y_test in [(X1, y1, X2, y2), (X2, y2, X1, y1)]:
            if self.method == "dml":
                cond_probs = self._estimate_cond_prob(S, X_test)
                omega = self._construct_omega(S, x_explain, X_test, cond_probs)
                theta_1, theta_2 = self._construct_theta(
                    S, x_explain, X_train, y_train, X_test, y_test
                )
                results.append(self._compute_dml(omega, theta_1, theta_2))

            elif self.method == "ipw":
                results.append(self._compute_ipw(S, x_explain, X_test, y_test))

            else:  # reg
                results.append(self._compute_reg(S, x_explain, X_train, y_train, X_test))

        return float(np.mean(results))

    def _keygen(self, S: list[int]) -> str:
        """Generate cache key for subset S."""
        return "".join(str(x) for x in sorted(S))

    def _calc_empty_prediction(self) -> float:
        """Calculate E[Y] as the empty prediction baseline."""
        return float(np.mean(self.predict(self.data)))

    def value_function(self, coalitions: npt.NDArray[np.bool_]) -> npt.NDArray[np.float64]:
        """Compute the value function for given coalitions.

        For each coalition S, computes E[Y | do(X_S = x_S)].

        Args:
            coalitions: Boolean array of shape (n_coalitions, n_features) where
                True indicates the feature is in the coalition (intervened on).

        Returns:
            Array of shape (n_coalitions,) with causal effect estimates.
        """
        if self._x is None:
            raise ValueError("Imputer not fitted. Call fit(x) first.")

        x_explain = self._x.flatten()
        n_coalitions = len(coalitions)
        results = np.zeros(n_coalitions)

        for i, coalition in enumerate(coalitions):
            S = np.where(coalition)[0].tolist()
            results[i] = self._compute_do_effect(S, x_explain)

        return results

    def fit(self, x: np.ndarray) -> "DoCausalImputer":
        """Fit the imputer to the explanation point.

        Args:
            x: The explanation point to explain.

        Returns:
            The fitted imputer.
        """
        super().fit(x)
        # Clear cache when fitting to new point
        self._cache = {}
        return self

    def clear_cache(self) -> None:
        """Clear the cached do-effect computations."""
        self._cache = {}
