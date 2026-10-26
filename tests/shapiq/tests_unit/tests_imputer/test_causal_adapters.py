"""Integration and cache regressions for the causal imputer adapters."""

from __future__ import annotations

import numpy as np
import pytest

from shapiq import CausalImputer, DoCausalImputer, NoMLCausalImputer, TabularExplainer
from shapiq.causal import CausalGraph, OrderingGraph
from shapiq.causal.graph import CausalGraph as LegacyCausalGraph


@pytest.fixture
def data():
    """A small continuous background with nonconstant features."""
    return np.random.default_rng(11).normal(size=(40, 3))


def model(values):
    """A prediction function with a nonzero baseline."""
    return 3.0 + values @ np.array([2.0, -0.5, 1.0])


def test_legacy_graph_is_the_same_class():
    """The compatibility import must not create a second graph implementation."""
    assert LegacyCausalGraph is CausalGraph is OrderingGraph


@pytest.mark.parametrize("sampling_method", ["gaussian", "copula", "empirical"])
def test_causal_selector_supports_exact_explainer(data, sampling_method):
    """The new selector composes with the current upstream exact approximator."""
    explainer = TabularExplainer(
        model,
        data,
        imputer="causal",
        approximator="exact",
        index="k-SII",
        max_order=2,
        ordering=[[0], [1, 2]],
        confounding=[False, True],
        sampling_method=sampling_method,
        sample_size=8,
        random_state=42,
    )
    assert isinstance(explainer.imputer, CausalImputer)
    x = np.array([2.3, -1.7, 0.2])
    result = explainer.explain(x)
    total = result.baseline_value + sum(
        value for interaction, value in result.dict_values.items() if interaction
    )
    np.testing.assert_allclose(total, model(x[None])[0], atol=1e-8)


@pytest.mark.parametrize("method", ["dml", "ipw", "reg"])
@pytest.mark.parametrize("cross_fitting", [False, True])
def test_do_causal_endpoints_and_partial_coalitions(data, method, cross_fitting):
    """Each estimator returns finite partial effects and exact game endpoints."""
    explainer = TabularExplainer(
        model,
        data,
        imputer="do_causal",
        dag_edges=[(0, 1), (1, 2)],
        method=method,
        cross_fitting=cross_fitting,
        random_state=42,
    )
    imputer = explainer.imputer
    assert isinstance(imputer, DoCausalImputer)
    x = np.array([2.3, -1.7, 0.2])
    imputer.fit(x)
    values = imputer(
        np.array(
            [
                [False, False, False],
                [True, False, False],
                [True, True, True],
            ]
        )
    )
    assert np.isfinite(values).all()
    assert values[0] == np.mean(model(data))
    np.testing.assert_allclose(values[-1], model(x[None])[0])
    assert imputer._cache
    imputer.set_random_state(7)
    assert not imputer._cache


def test_do_causal_cache_distinguishes_multidigit_indices():
    """Coalitions [1, 23] and [3, 12] must not share a cache entry."""
    imputer = object.__new__(DoCausalImputer)
    assert imputer._keygen([1, 23]) != imputer._keygen([3, 12])
    assert imputer._keygen([23, 1]) == imputer._keygen([1, 23])


def test_do_causal_rejects_unsupported_latent_confounding(data):
    """Latent-confounding annotations must not be silently ignored."""
    with pytest.raises(NotImplementedError, match="latent confounding_pairs"):
        DoCausalImputer(model, data, dag_edges=[(0, 1)], confounding_pairs={(0, 1)})


@pytest.mark.parametrize("sampling_method", ["gaussian", "copula", "empirical"])
@pytest.mark.parametrize("as_object", [False, True])
def test_no_ml_selector_and_adapter(data, sampling_method, as_object):
    """Both public entry points retain the target baseline and fitted prediction."""
    y = model(data)
    options = {
        "ordering": [[0], [1, 2]],
        "sampling_method": sampling_method,
        "sample_size": 8,
        "random_state": 42,
    }
    adapter = NoMLCausalImputer(data, y, **options)
    if as_object:
        explainer = TabularExplainer(adapter.predict, data, imputer=adapter)
    else:
        explainer = TabularExplainer(model, data, imputer="no_ml_causal", y=y, **options)
    x = np.array([0.1, 0.4, -0.2])
    explainer.imputer.fit(x)
    values = explainer.imputer(np.array([[False] * 3, [True] * 3]))
    assert values[0] == adapter.empty_prediction
    np.testing.assert_allclose(values[1:], adapter.predict(x[None]))


@pytest.mark.parametrize("k_neighbors", [40, 100])
def test_no_ml_empirical_accepts_entire_background(data, k_neighbors):
    """Using every background neighbor returns the target mean without an index error."""
    y = model(data)
    adapter = NoMLCausalImputer(data, y, sampling_method="empirical", k_neighbors=k_neighbors)
    np.testing.assert_allclose(adapter.predict(data[:2]), np.mean(y))


@pytest.mark.parametrize("imputer, required", [("do_causal", "dag_edges"), ("no_ml_causal", "y")])
def test_selector_reports_missing_causal_inputs(data, imputer, required):
    """Required causal inputs produce an actionable error."""
    with pytest.raises(ValueError, match=required):
        TabularExplainer(model, data, imputer=imputer)
