"""Regression tests for fixed interventions and causal game endpoints."""

from __future__ import annotations

from itertools import product

import numpy as np
import pytest

from shapiq.imputer.causal_imputer import CausalImputer
from shapiq.imputer.gaussian_copula_imputer import GaussianCopulaImputer
from shapiq.imputer.gaussian_imputer import GaussianImputer
from shapiq.imputer.no_ml_causal_imputer import NoMLCausalImputer


@pytest.fixture
def background():
    return np.array([[0.1, 1.2, 2.3], [1.4, 2.5, 3.6], [2.7, 3.8, 4.9], [3.2, 1.1, 5.4]])


def nonlinear_model(data):
    return data[:, 0] ** 2 + 2 * data[:, 1] - data[:, 2]


@pytest.mark.parametrize("sampling_method", ["copula", "gaussian", "empirical"])
@pytest.mark.parametrize("point", [[1.25, 2.05, 4.35], [-2.0, 8.0, 10.0]])
def test_causal_preserves_every_intervened_value(background, sampling_method, point):
    """Held-out and out-of-support inputs must never be projected onto background rows."""
    x = np.array(point)
    imputer = CausalImputer(
        model=nonlinear_model,
        data=background,
        x=x,
        sampling_method=sampling_method,
        ordering=[[0], [1, 2]],
        sample_size=7,
        k_neighbors=2,
        random_state=42,
    )
    coalitions = np.array(list(product([False, True], repeat=3)))
    samples = imputer._draw_samples(x, coalitions)
    assert np.all(np.isfinite(samples))
    for coalition, completed in zip(coalitions, samples, strict=False):
        np.testing.assert_array_equal(completed[:, coalition], np.tile(x[coalition], (7, 1)))
    np.testing.assert_allclose(imputer.value_function(coalitions[-1:]), nonlinear_model(x[None]))


@pytest.mark.parametrize("point", [[1.25, 2.05, 4.35], [-2.0, 8.0, 10.0]])
def test_gaussian_copula_preserves_every_intervened_value(background, point):
    x = np.array(point)
    imputer = GaussianCopulaImputer(
        model=nonlinear_model,
        data=background,
        x=x,
        sample_size=7,
        random_state=42,
    )
    coalitions = np.array(list(product([False, True], repeat=3)))
    samples = imputer._draw_samples(x, coalitions)
    for coalition, completed in zip(coalitions, samples, strict=False):
        np.testing.assert_array_equal(completed[:, coalition], np.tile(x[coalition], (7, 1)))
    np.testing.assert_allclose(imputer.value_function(coalitions[-1:]), nonlinear_model(x[None]))


@pytest.mark.parametrize("sampling_method", ["copula", "gaussian", "empirical"])
def test_causal_empty_coalition_matches_declared_baseline(background, sampling_method):
    imputer = CausalImputer(
        model=nonlinear_model,
        data=background,
        x=np.array([1.25, 2.05, 4.35]),
        sampling_method=sampling_method,
        sample_size=3,
        k_neighbors=2,
        random_state=42,
    )
    coalitions = np.array([[False, False, False], [True, False, True], [True, True, True]])
    for seed in [42, 809]:
        imputer.set_random_state(seed)
        values = imputer.value_function(coalitions)
        assert values[0] == imputer.empty_prediction == np.mean(nonlinear_model(background))


@pytest.mark.parametrize("sampling_method", ["gaussian", "empirical"])
def test_integer_input_does_not_truncate_sampled_values(background, sampling_method):
    x = np.array([1, 2, 3])
    imputer = CausalImputer(
        model=nonlinear_model,
        data=background,
        sampling_method=sampling_method,
        sample_size=7,
        k_neighbors=2,
        random_state=42,
    )
    coalition = np.array([True, False, False])
    integer_input_samples = imputer._sample_causal(x, coalition)
    imputer.set_random_state(42)
    float_input_samples = imputer._sample_causal(x.astype(float), coalition)
    np.testing.assert_array_equal(integer_input_samples, float_input_samples)


def test_gaussian_integer_input_does_not_truncate_sampled_values(background):
    imputer = GaussianImputer(
        model=nonlinear_model, data=background, sample_size=7, random_state=42
    )
    coalition = np.array([[True, False, False]])
    integer_input_samples = imputer._draw_samples(np.array([1, 2, 3]), coalition)
    float_input_samples = imputer._draw_samples(np.array([1.0, 2.0, 3.0]), coalition)
    np.testing.assert_array_equal(integer_input_samples, float_input_samples)


@pytest.mark.parametrize("k_neighbors", [4, 10])
def test_empirical_can_use_all_background_neighbors(background, k_neighbors):
    imputer = CausalImputer(
        model=nonlinear_model,
        data=background,
        sampling_method="empirical",
        sample_size=7,
        k_neighbors=k_neighbors,
        random_state=42,
    )
    samples = imputer._sample_causal(np.array([1.0, 2.0, 3.0]), np.array([True, False, False]))
    assert samples.shape == (7, 3)
    assert np.all(np.isin(samples[:, 1], background[:, 1]))


def test_empirical_rejects_zero_neighbors(background):
    with pytest.raises(ValueError, match="k_neighbors must be at least 1"):
        CausalImputer(
            model=nonlinear_model, data=background, sampling_method="empirical", k_neighbors=0
        )


@pytest.mark.parametrize("ordering", [[[0], [1]], [[0], [1, 1, 2]], [[0], [1, 3]]])
def test_causal_rejects_orderings_that_do_not_cover_features_once(background, ordering):
    with pytest.raises(ValueError, match="each feature index exactly once"):
        CausalImputer(model=nonlinear_model, data=background, ordering=ordering)


@pytest.mark.parametrize("imputer_class", [GaussianImputer, GaussianCopulaImputer, CausalImputer])
def test_single_feature_covariance_is_a_matrix(imputer_class):
    data = np.array([[0.1], [1.3], [2.7], [4.2]])
    x = np.array([2.25])
    imputer = imputer_class(
        model=lambda values: values[:, 0] ** 2,
        data=data,
        x=x,
        sample_size=7,
        random_state=42,
    )
    coalitions = np.array([[False], [True]])
    samples = imputer._draw_samples(x, coalitions)
    assert samples.shape == (2, 7, 1)
    assert np.all(np.isfinite(samples))
    np.testing.assert_array_equal(samples[1], np.tile(x, (7, 1)))


@pytest.mark.parametrize("sampling_method", ["copula", "gaussian"])
def test_no_ml_adapter_retains_its_prediction_space_and_baseline(sampling_method):
    data = np.random.default_rng(19).normal(size=(80, 3))
    noml = NoMLCausalImputer(
        X=data,
        y=nonlinear_model(data),
        ordering=[[0], [1, 2]],
        confounding=[False, False],
        sampling_method=sampling_method,
        sample_size=7,
        random_state=42,
    )
    causal = noml.as_causal_imputer()
    x = np.array([0.23, 0.87, -0.58])
    causal.fit(x)
    if sampling_method == "copula":
        coalition = np.array([True, False, True])
        samples = causal._sample_causal(x, coalition)
        gaussian_x = causal._transform_point_to_gaussian(x)
        np.testing.assert_array_equal(samples[:, coalition], np.tile(gaussian_x[coalition], (7, 1)))
    values = causal.value_function(np.array([[False, False, False], [True, True, True]]))
    assert values[0] == noml.empty_prediction
    np.testing.assert_allclose(values[1:], noml.predict(x[None]), rtol=1e-12)


@pytest.mark.parametrize("sampling_method", ["copula", "gaussian", "empirical"])
def test_k_sii_decomposition_reconstructs_original_prediction(background, sampling_method):
    """All 2**3 coalitions yield an efficient decomposition even with small MC draws."""
    from shapiq import TabularExplainer

    def model(data):
        return data @ np.array([2.0, -3.0, 0.5])

    imputer = CausalImputer(
        model=model,
        data=background,
        sampling_method=sampling_method,
        ordering=[[0], [1, 2]],
        sample_size=7,
        k_neighbors=2,
        random_state=42,
    )
    explainer = TabularExplainer(
        model=model,
        data=background,
        imputer=imputer,
        index="k-SII",
        max_order=2,
        random_state=42,
    )
    x = np.array([-2.0, 8.0, 10.0])
    result = explainer.explain(x, budget=8)
    contributions = sum(value for interaction, value in result.dict_values.items() if interaction)
    np.testing.assert_allclose(contributions + result.baseline_value, model(x[None])[0], atol=1e-8)
