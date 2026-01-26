"""Tests for the DoCausalImputer class."""

import numpy as np
import pytest

from shapiq.imputer import DoCausalImputer


class TestDoCausalImputerBasic:
    """Basic tests for DoCausalImputer initialization and structure."""

    @pytest.fixture
    def simple_data(self):
        """Create simple test data."""
        np.random.seed(42)
        n = 100
        X = np.random.rand(n, 4)
        return X

    @pytest.fixture
    def simple_model(self):
        """Create a simple model."""
        return lambda x: x[:, 0] + 2 * x[:, 1] + x[:, 2] + 0.5 * x[:, 3]

    @pytest.fixture
    def simple_dag(self):
        """Create a simple DAG: X0 -> X1 -> X2, X0 -> X3."""
        return [(0, 1), (1, 2), (0, 3)]

    def test_initialization(self, simple_data, simple_model, simple_dag):
        """Test basic initialization."""
        imputer = DoCausalImputer(
            model=simple_model,
            data=simple_data,
            dag_edges=simple_dag,
        )

        assert imputer.n_features == 4
        assert imputer.dag_edges == simple_dag
        assert imputer.method == "dml"  # default
        assert len(imputer.topo_order) == 4

    def test_topological_sort(self, simple_data, simple_model, simple_dag):
        """Test topological sorting."""
        imputer = DoCausalImputer(
            model=simple_model,
            data=simple_data,
            dag_edges=simple_dag,
        )

        topo = imputer.topo_order

        # X0 should come before X1, X3
        assert topo.index(0) < topo.index(1)
        assert topo.index(0) < topo.index(3)
        # X1 should come before X2
        assert topo.index(1) < topo.index(2)

    def test_cycle_detection(self, simple_data, simple_model):
        """Test that cycles are detected."""
        cyclic_dag = [(0, 1), (1, 2), (2, 0)]  # Cycle!

        with pytest.raises(ValueError, match="cycle"):
            DoCausalImputer(
                model=simple_model,
                data=simple_data,
                dag_edges=cyclic_dag,
            )

    def test_invalid_edge(self, simple_data, simple_model):
        """Test that invalid edges are detected."""
        invalid_dag = [(0, 1), (1, 10)]  # Feature 10 doesn't exist

        with pytest.raises(ValueError, match="invalid.*(child|parent|feature).*index"):
            DoCausalImputer(
                model=simple_model,
                data=simple_data,
                dag_edges=invalid_dag,
            )

    def test_invalid_method(self, simple_data, simple_model, simple_dag):
        """Test that invalid methods are rejected."""
        with pytest.raises(ValueError, match="method must be one of"):
            DoCausalImputer(
                model=simple_model,
                data=simple_data,
                dag_edges=simple_dag,
                method="invalid",
            )


class TestDoCausalImputerMethods:
    """Test different estimation methods."""

    @pytest.fixture
    def causal_data(self):
        """Create data with known causal structure."""
        np.random.seed(42)
        n = 200

        X0 = np.random.uniform(0, 1, n)
        X1 = 0.8 * X0 + np.random.normal(0, 0.1, n)
        X2 = 0.6 * X1 + np.random.normal(0, 0.1, n)
        X3 = 0.5 * X1 + np.random.normal(0, 0.1, n)

        X = np.column_stack([X0, X1, X2, X3])
        return X

    @pytest.fixture
    def causal_model(self):
        """Model with known coefficients."""
        return lambda x: 2 * x[:, 0] + 3 * x[:, 2] + 0.5 * x[:, 3]

    @pytest.fixture
    def causal_dag(self):
        """DAG matching data generation."""
        return [(0, 1), (1, 2), (1, 3)]

    @pytest.mark.parametrize("method", ["dml", "ipw", "reg"])
    def test_methods_run(self, causal_data, causal_model, causal_dag, method):
        """Test that all methods run without error."""
        imputer = DoCausalImputer(
            model=causal_model,
            data=causal_data,
            dag_edges=causal_dag,
            method=method,
            random_state=42,
        )

        x_explain = causal_data[0]
        imputer.fit(x_explain)

        # Test with a simple coalition
        coalitions = np.array([
            [True, False, False, False],
            [True, True, False, False],
            [True, True, True, True],
        ])

        values = imputer.value_function(coalitions)
        assert len(values) == 3
        assert not np.any(np.isnan(values))

    def test_empty_coalition(self, causal_data, causal_model, causal_dag):
        """Test that empty coalition returns E[Y]."""
        imputer = DoCausalImputer(
            model=causal_model,
            data=causal_data,
            dag_edges=causal_dag,
        )

        x_explain = causal_data[0]
        imputer.fit(x_explain)

        coalitions = np.array([[False, False, False, False]])
        values = imputer.value_function(coalitions)

        expected = np.mean(causal_model(causal_data))
        np.testing.assert_almost_equal(values[0], expected, decimal=2)

    def test_full_coalition(self, causal_data, causal_model, causal_dag):
        """Test that full coalition returns f(x)."""
        imputer = DoCausalImputer(
            model=causal_model,
            data=causal_data,
            dag_edges=causal_dag,
        )

        x_explain = causal_data[0]
        imputer.fit(x_explain)

        coalitions = np.array([[True, True, True, True]])
        values = imputer.value_function(coalitions)

        expected = causal_model(x_explain.reshape(1, -1))[0]
        np.testing.assert_almost_equal(values[0], expected, decimal=2)


class TestDoCausalImputerCaching:
    """Test caching behavior."""

    @pytest.fixture
    def setup(self):
        """Setup test fixtures."""
        np.random.seed(42)
        X = np.random.rand(100, 3)
        model = lambda x: x[:, 0] + x[:, 1] + x[:, 2]
        dag = [(0, 1), (1, 2)]
        return X, model, dag

    def test_cache_works(self, setup):
        """Test that results are cached."""
        X, model, dag = setup
        imputer = DoCausalImputer(model=model, data=X, dag_edges=dag)
        imputer.fit(X[0])

        # First call
        coalitions = np.array([[True, False, False]])
        _ = imputer.value_function(coalitions)

        # Check cache has entry
        assert len(imputer._cache) > 0

    def test_cache_cleared_on_fit(self, setup):
        """Test that cache is cleared when fitting new point."""
        X, model, dag = setup
        imputer = DoCausalImputer(model=model, data=X, dag_edges=dag)

        imputer.fit(X[0])
        coalitions = np.array([[True, False, False]])
        _ = imputer.value_function(coalitions)
        assert len(imputer._cache) > 0

        # Fit to new point
        imputer.fit(X[1])
        assert len(imputer._cache) == 0


class TestDoCausalImputerCrossFitting:
    """Test cross-fitting option."""

    @pytest.fixture
    def setup(self):
        """Setup test fixtures."""
        np.random.seed(42)
        X = np.random.rand(200, 3)
        model = lambda x: x[:, 0] + x[:, 1] + x[:, 2]
        dag = [(0, 1), (1, 2)]
        return X, model, dag

    def test_cross_fitting_runs(self, setup):
        """Test that cross-fitting runs without error."""
        X, model, dag = setup
        imputer = DoCausalImputer(
            model=model,
            data=X,
            dag_edges=dag,
            cross_fitting=True,
            random_state=42,
        )

        imputer.fit(X[0])
        coalitions = np.array([[True, False, False], [True, True, False]])
        values = imputer.value_function(coalitions)

        assert len(values) == 2
        assert not np.any(np.isnan(values))


class TestDoCausalImputerIntegration:
    """Integration tests with TabularExplainer."""

    def test_with_explainer(self):
        """Test integration with TabularExplainer."""
        from shapiq import TabularExplainer

        np.random.seed(42)
        X = np.random.rand(200, 3)
        model = lambda x: x[:, 0] + 2 * x[:, 1] + x[:, 2]
        dag = [(0, 1), (1, 2)]

        imputer = DoCausalImputer(
            model=model,
            data=X,
            dag_edges=dag,
            random_state=42,
        )

        explainer = TabularExplainer(
            model=model,
            data=X,
            imputer=imputer,
            index="SV",
        )

        x_sample = X[0:1]
        sv = explainer.explain(x_sample, budget=100)

        assert sv is not None
        # SV values array includes only first-order effects (3 features)
        # but sv.values may include baseline, so check n_players
        assert sv.n_players == 3
        
        # Get only the first-order Shapley values
        first_order_values = [sv[(i,)] for i in range(3)]
        
        # Sum of Shapley values should approximately equal prediction - E[Y]
        prediction = model(x_sample)[0]
        baseline = np.mean(model(X))
        np.testing.assert_almost_equal(
            np.sum(first_order_values), prediction - baseline, decimal=1
        )
