"""Regression tests for unconditional Gaussian and copula replacement sampling."""

from __future__ import annotations

from itertools import product

import numpy as np
import pytest

from shapiq.imputer import MarginalImputer


def nonlinear_model(data: np.ndarray) -> np.ndarray:
    return data[:, 0] ** 2 + data[:, 1] * data[:, 2]


@pytest.mark.parametrize("sampling_method", ["joint", "gaussian", "copula"])
@pytest.mark.parametrize("normalize", [False, True])
def test_replaced_background_matches_fresh_imputer(sampling_method, normalize):
    """Background replacement must refresh moments, empirical marginals, and baseline."""
    rng = np.random.default_rng(71)
    old_data = rng.normal(size=(17, 3))
    new_data = rng.normal(size=(41, 3)) * [2.0, 4.0, 0.5] + [15.0, -20.0, 8.0]
    x = np.array([0.1, -0.3, 0.7])
    kwargs = {
        "model": nonlinear_model,
        "x": x,
        "sample_size": 53,
        "sampling_method": sampling_method,
        "random_state": 18,
        "normalize": normalize,
    }
    changed = MarginalImputer(data=old_data, **kwargs)
    old_baseline = changed.empty_prediction
    assert changed.init_background(new_data) is changed
    fresh = MarginalImputer(data=new_data, **kwargs)
    coalitions = np.array(list(product([False, True], repeat=3)))

    np.testing.assert_array_equal(changed.data, new_data)
    np.testing.assert_allclose(changed(coalitions), fresh(coalitions), rtol=0, atol=0)
    assert changed.empty_prediction == fresh.empty_prediction
    assert changed.normalization_value == fresh.normalization_value
    assert not np.isclose(changed.empty_prediction, old_baseline)
    assert changed.sample_size == (41 if sampling_method == "joint" else 53)


@pytest.mark.parametrize("sampling_method", ["gaussian", "copula"])
@pytest.mark.parametrize("random_state", [None, 42])
def test_shared_samples_preserve_null_players_and_batch_order(sampling_method, random_state):
    """Changing an unused observed feature cannot change the missing-feature draws."""
    data = np.random.default_rng(9).normal(size=(100, 3))

    def model(values):
        return values[:, 1] ** 2 + np.sin(values[:, 2])

    imputer = MarginalImputer(
        model=model,
        data=data,
        x=np.array([13.0, 0.2, -0.5]),
        sample_size=71,
        sampling_method=sampling_method,
        random_state=random_state,
        normalize=False,
    )
    without = np.array([[False, *pair] for pair in product([False, True], repeat=2)])
    with_unused_feature = without.copy()
    with_unused_feature[:, 0] = True
    values = imputer(without)
    np.testing.assert_allclose(imputer(with_unused_feature), values, rtol=0, atol=1e-12)
    np.testing.assert_array_equal(imputer(without), values)
    np.testing.assert_array_equal(imputer(without[::-1]), values[::-1])
    for coalition, expected in zip(without, values, strict=True):
        assert imputer(coalition[None])[0] == expected
    imputer.fit(np.array([-300.0, 0.2, -0.5]))
    np.testing.assert_allclose(imputer(with_unused_feature), values, rtol=0, atol=1e-12)


@pytest.mark.parametrize("sampling_method", ["gaussian", "copula"])
def test_seed_reset_matches_fresh_parametric_imputer(sampling_method):
    """Setting a seed redraws both replacement samples and the empty baseline."""
    data = np.random.default_rng(20).normal(size=(20, 3))
    kwargs = {
        "model": nonlinear_model,
        "data": data,
        "x": data[0],
        "sampling_method": sampling_method,
    }
    changed = MarginalImputer(random_state=1, **kwargs)
    before = changed.empty_prediction
    changed.set_random_state(29)
    fresh = MarginalImputer(random_state=29, **kwargs)
    coalitions = np.array(list(product([False, True], repeat=3)))
    np.testing.assert_array_equal(changed(coalitions), fresh(coalitions))
    assert changed.empty_prediction == fresh.empty_prediction
    assert changed.empty_prediction != before


@pytest.mark.parametrize("normalize", [False, True])
def test_background_update_preserves_zero_baseline_normalization_choice(normalize):
    """A zero baseline must not discard the caller's normalization preference."""
    data = np.array([[-1.0], [0.0], [1.0]])
    imputer = MarginalImputer(
        model=lambda values: values[:, 0], data=data, x=np.array([4.0]), normalize=normalize
    )
    assert imputer.empty_prediction == 0
    imputer.init_background(data + 10)
    assert imputer.empty_prediction == 10
    assert imputer(np.array([[False]]))[0] == (0 if normalize else 10)


@pytest.mark.parametrize("sampling_method", ["gaussian", "copula"])
def test_single_feature_parametric_background(sampling_method):
    """Scalar covariance is supported and the full coalition uses the exact held-out input."""
    data = np.array([[0.1], [0.5], [1.5], [2.7]])
    imputer = MarginalImputer(
        model=lambda values: values[:, 0] ** 2,
        data=data,
        x=np.array([7]),
        sample_size=None,
        sampling_method=sampling_method,
        random_state=7,
        normalize=False,
    )
    assert imputer.sample_size == len(data)
    values = imputer(np.array([[False], [True]]))
    assert np.isfinite(values).all()
    assert values[0] == imputer.empty_prediction
    assert values[1] == 49


def test_unknown_marginal_backend_is_rejected():
    with pytest.raises(ValueError, match="sampling_method"):
        MarginalImputer(model=nonlinear_model, data=np.ones((4, 3)), sampling_method="unknown")


@pytest.mark.parametrize("sampling_method", ["gaussian", "copula"])
@pytest.mark.parametrize("data", [np.ones((1, 3)), np.full((3, 3), np.nan)])
def test_invalid_parametric_background_is_rejected(sampling_method, data):
    with pytest.raises(ValueError, match="at least two finite numeric rows"):
        MarginalImputer(model=nonlinear_model, data=data, sampling_method=sampling_method)
