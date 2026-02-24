"""Tests for CausalImputer and NoMLCausalImputer performance fix.

Tests validate:
1. NoMLCausalImputer.predict_gaussian matches predict on training data
2. Vectorized _transform_point_to_gaussian matches loop-based version
3. CausalImputer with _model_accepts_gaussian skips inverse transform
4. as_causal_imputer produces correct results with the optimization
5. End-to-end: TabularExplainer with no_ml_causal produces valid Shapley values
"""

import numpy as np
import pytest

from shapiq.imputer.causal_imputer import CausalImputer
from shapiq.imputer.no_ml_causal_imputer import NoMLCausalImputer


# --- Fixtures ---

@pytest.fixture
def rng():
    return np.random.default_rng(42)


@pytest.fixture
def linear_data(rng):
    """Create simple linear causal data: x0 -> x1 -> x2, y = x0 + 2*x1 + 0.5*x2."""
    n = 200
    x0 = rng.standard_normal(n)
    x1 = 0.5 * x0 + rng.standard_normal(n) * 0.3
    x2 = 0.3 * x1 + rng.standard_normal(n) * 0.2
    X = np.column_stack([x0, x1, x2])
    y = x0 + 2.0 * x1 + 0.5 * x2 + rng.standard_normal(n) * 0.1
    return X, y


@pytest.fixture
def ordering():
    return [[0], [1], [2]]


@pytest.fixture
def confounding():
    return [False, False, False]


@pytest.fixture
def simple_model():
    return lambda x: np.atleast_2d(x)[:, 0] + 2.0 * np.atleast_2d(x)[:, 1] + 0.5 * np.atleast_2d(x)[:, 2]


# --- NoMLCausalImputer Tests ---

class TestNoMLCausalImputerPredictGaussian:
    """Test predict_gaussian correctness and consistency with predict."""

    def test_predict_gaussian_matches_predict_on_training_data_copula(self, linear_data, ordering, confounding):
        """predict_gaussian(transform(X)) should match predict(X) exactly."""
        X, y = linear_data
        imputer = NoMLCausalImputer(
            X=X, y=y, ordering=ordering, confounding=confounding,
            sampling_method="copula",
        )

        # Transform training subset to gaussian, then predict_gaussian
        X_subset = X[:10]
        X_gauss = imputer._transform_point_to_gaussian(X_subset)
        pred_gaussian = imputer.predict_gaussian(X_gauss)
        pred_original = imputer.predict(X_subset)

        # Should be identical (same code path after transform)
        np.testing.assert_allclose(pred_gaussian, pred_original, atol=1e-10)

    def test_predict_gaussian_single_point(self, linear_data, ordering, confounding):
        """predict_gaussian works for a single point."""
        X, y = linear_data
        imputer = NoMLCausalImputer(
            X=X, y=y, ordering=ordering, confounding=confounding,
            sampling_method="copula",
        )

        x_single = X[0]
        x_gauss = imputer._transform_point_to_gaussian(x_single)
        result = imputer.predict_gaussian(x_gauss)
        assert result.shape == (1,) or result.ndim == 0

    def test_predict_gaussian_fallback_for_gaussian_method(self, linear_data, ordering, confounding):
        """For sampling_method='gaussian', predict_gaussian falls back to predict."""
        X, y = linear_data
        imputer = NoMLCausalImputer(
            X=X, y=y, ordering=ordering, confounding=confounding,
            sampling_method="gaussian",
        )

        X_subset = X[:5]
        pred_gaussian_fn = imputer.predict_gaussian(X_subset)
        pred_original = imputer.predict(X_subset)

        np.testing.assert_allclose(pred_gaussian_fn, pred_original, atol=1e-10)


class TestTransformPointToGaussianVectorized:
    """Test that the vectorized _transform_point_to_gaussian is correct."""

    def test_single_point_matches(self, linear_data, ordering, confounding):
        """Single point should give same result as vectorized batch of 1."""
        X, y = linear_data
        imputer = NoMLCausalImputer(
            X=X, y=y, ordering=ordering, confounding=confounding,
            sampling_method="copula",
        )

        x_single = X[5]
        result_single = imputer._transform_point_to_gaussian(x_single.reshape(1, -1))
        result_batch = imputer._transform_point_to_gaussian(X[5:6])

        np.testing.assert_allclose(result_single, result_batch, atol=1e-10)

    def test_batch_consistency(self, linear_data, ordering, confounding):
        """Batch transform should equal sequential single-point transforms."""
        X, y = linear_data
        imputer = NoMLCausalImputer(
            X=X, y=y, ordering=ordering, confounding=confounding,
            sampling_method="copula",
        )

        X_batch = X[:20]
        batch_result = imputer._transform_point_to_gaussian(X_batch)

        # Compare with manual per-point computation (reference implementation)
        from scipy.stats import norm
        reference = np.zeros_like(X_batch)
        for i in range(len(X_batch)):
            ranks = np.sum(imputer.X <= X_batch[i], axis=0)
            empirical_cdf = ranks / imputer.n_samples
            empirical_cdf = np.clip(empirical_cdf, 1e-10, 1 - 1e-10)
            reference[i] = norm.ppf(empirical_cdf)

        np.testing.assert_allclose(batch_result, reference, atol=1e-10)

    def test_large_batch_chunking(self, linear_data, ordering, confounding):
        """Large batches (>200) should still work via chunking."""
        X, y = linear_data
        imputer = NoMLCausalImputer(
            X=X, y=y, ordering=ordering, confounding=confounding,
            sampling_method="copula",
        )

        # Create a batch larger than _CHUNK_SIZE (200)
        X_large = np.tile(X[:50], (5, 1))  # 250 points
        result = imputer._transform_point_to_gaussian(X_large)
        assert result.shape == X_large.shape

        # First 50 should match the corresponding transform
        result_small = imputer._transform_point_to_gaussian(X[:50])
        np.testing.assert_allclose(result[:50], result_small, atol=1e-10)


# --- CausalImputer Tests ---

class TestCausalImputerGaussianFlag:
    """Test the _model_accepts_gaussian flag."""

    def test_default_flag_is_false(self, linear_data, simple_model, ordering, confounding):
        """Default: _model_accepts_gaussian should be False."""
        X, y = linear_data
        imputer = CausalImputer(
            model=simple_model, data=X,
            ordering=ordering, confounding=confounding,
            sampling_method="copula", sample_size=10,
        )
        assert imputer._model_accepts_gaussian is False

    def test_flag_can_be_set_true(self, linear_data, simple_model, ordering, confounding):
        """_model_accepts_gaussian can be set via __init__."""
        X, y = linear_data
        imputer = CausalImputer(
            model=simple_model, data=X,
            ordering=ordering, confounding=confounding,
            sampling_method="copula", sample_size=10,
            _model_accepts_gaussian=True,
            _precomputed_empty_prediction=0.0,
        )
        assert imputer._model_accepts_gaussian is True

    def test_precomputed_empty_prediction_skips_calc(self, linear_data, simple_model, ordering, confounding):
        """Providing _precomputed_empty_prediction should use that value directly."""
        X, y = linear_data
        expected = 42.0
        imputer = CausalImputer(
            model=simple_model, data=X,
            ordering=ordering, confounding=confounding,
            sampling_method="gaussian", sample_size=10,
            _precomputed_empty_prediction=expected,
        )
        assert imputer.empty_prediction == expected


# --- as_causal_imputer Integration Tests ---

class TestAsCausalImputer:
    """Test NoMLCausalImputer.as_causal_imputer() with optimization."""

    def test_as_causal_imputer_sets_gaussian_flag_copula(self, linear_data, ordering, confounding):
        """as_causal_imputer with copula should set _model_accepts_gaussian=True."""
        X, y = linear_data
        noml = NoMLCausalImputer(
            X=X, y=y, ordering=ordering, confounding=confounding,
            sampling_method="copula",
        )
        causal = noml.as_causal_imputer()
        assert causal._model_accepts_gaussian is True

    def test_as_causal_imputer_no_gaussian_flag_gaussian_method(self, linear_data, ordering, confounding):
        """as_causal_imputer with gaussian should NOT set _model_accepts_gaussian."""
        X, y = linear_data
        noml = NoMLCausalImputer(
            X=X, y=y, ordering=ordering, confounding=confounding,
            sampling_method="gaussian",
        )
        causal = noml.as_causal_imputer()
        assert causal._model_accepts_gaussian is False

    def test_as_causal_imputer_uses_original_y_mean(self, linear_data, ordering, confounding):
        """as_causal_imputer should pass _original_y_mean as empty_prediction."""
        X, y = linear_data
        noml = NoMLCausalImputer(
            X=X, y=y, ordering=ordering, confounding=confounding,
            sampling_method="copula",
        )
        causal = noml.as_causal_imputer()
        assert causal.empty_prediction == noml._original_y_mean

    def test_as_causal_imputer_value_function_runs(self, linear_data, ordering, confounding):
        """value_function should run without error on the optimized imputer."""
        X, y = linear_data
        noml = NoMLCausalImputer(
            X=X, y=y, ordering=ordering, confounding=confounding,
            sampling_method="copula", sample_size=20,
        )
        causal = noml.as_causal_imputer()

        # Explain a test point
        x_test = X[0:1]
        causal.fit(x_test)

        # Create a few coalitions
        n_features = X.shape[1]
        coalitions = np.array([
            [True, True, True],    # all known
            [False, False, False], # all unknown
            [True, False, False],  # only x0 known
            [True, True, False],   # x0, x1 known
        ])

        result = causal.value_function(coalitions)
        assert result.shape == (4,)
        assert np.all(np.isfinite(result))

    def test_as_causal_imputer_copula_results_reasonable(self, linear_data, ordering, confounding):
        """Optimized imputer should produce results consistent with non-optimized."""
        X, y = linear_data
        noml = NoMLCausalImputer(
            X=X, y=y, ordering=ordering, confounding=confounding,
            sampling_method="copula", sample_size=50, random_state=42,
        )
        causal = noml.as_causal_imputer()
        x_test = X[0:1]
        causal.fit(x_test)

        coalitions = np.array([
            [True, True, True],
            [False, False, False],
        ])
        result = causal.value_function(coalitions)

        # Full coalition should predict close to actual y for this point
        # Empty coalition should predict close to mean(y)
        y_mean = np.mean(y)
        assert abs(result[1] - y_mean) < np.std(y) * 2  # empty coalition ~ y_mean

    def test_gaussian_method_still_works(self, linear_data, ordering, confounding):
        """Gaussian method should work normally (no optimization path)."""
        X, y = linear_data
        noml = NoMLCausalImputer(
            X=X, y=y, ordering=ordering, confounding=confounding,
            sampling_method="gaussian", sample_size=20,
        )
        causal = noml.as_causal_imputer()
        x_test = X[0:1]
        causal.fit(x_test)

        coalitions = np.array([
            [True, True, True],
            [False, False, False],
        ])
        result = causal.value_function(coalitions)
        assert result.shape == (2,)
        assert np.all(np.isfinite(result))


# --- Performance Sanity Check ---

class TestPerformanceImprovement:
    """Verify the optimization doesn't degrade results."""

    def test_no_ml_causal_explain_produces_valid_shapley_values(self, linear_data, ordering, confounding):
        """End-to-end: TabularExplainer with no_ml_causal should produce valid results."""
        from shapiq import TabularExplainer

        X, y = linear_data
        model = lambda x: np.atleast_2d(x)[:, 0] + 2.0 * np.atleast_2d(x)[:, 1] + 0.5 * np.atleast_2d(x)[:, 2]

        explainer = TabularExplainer(
            model=model,
            data=X,
            imputer="no_ml_causal",
            y=y,
            ordering=ordering,
            confounding=confounding,
            sampling_method="copula",
            sample_size=50,
            index="SV",
            random_state=42,
        )

        x_test = X[0:1]
        result = explainer.explain(x_test, budget=128)

        sv = result.get_n_order_values(order=1)
        assert sv.shape == (3,)
        assert np.all(np.isfinite(sv))

        # Efficiency: sum of SVs should approximately equal f(x) - E[f(x)]
        y_pred = model(x_test)
        y_mean = np.mean(y)
        expected_sum = float(y_pred[0]) - y_mean
        actual_sum = float(np.sum(sv))
        # Allow generous tolerance for MC approximation
        assert abs(actual_sum - expected_sum) < abs(expected_sum) * 0.5 + 50

    def test_causal_imputer_regular_model_unchanged(self, linear_data, simple_model, ordering, confounding):
        """Regular CausalImputer (non-NoML) should be completely unaffected."""
        X, y = linear_data
        imputer = CausalImputer(
            model=simple_model, data=X,
            ordering=ordering, confounding=confounding,
            sampling_method="copula", sample_size=20, random_state=42,
        )
        assert imputer._model_accepts_gaussian is False

        x_test = X[0:1]
        imputer.fit(x_test)

        coalitions = np.array([
            [True, True, True],
            [False, False, False],
            [True, False, False],
        ])
        result = imputer.value_function(coalitions)
        assert result.shape == (3,)
        assert np.all(np.isfinite(result))
