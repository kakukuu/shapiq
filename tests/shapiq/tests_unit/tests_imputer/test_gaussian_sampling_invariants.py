"""Regressions for exact observed values and one-dimensional Gaussian sampling."""

from __future__ import annotations

from itertools import product

import numpy as np
import pytest

from shapiq.imputer import GaussianCopulaImputer, GaussianImputer


@pytest.mark.parametrize("imputer_type", [GaussianImputer, GaussianCopulaImputer])
@pytest.mark.parametrize("x", [np.array([0.4, 1.7, 3.1]), np.array([-20, 40, 100])])
def test_observed_values_survive_imputation(imputer_type, x):
    """Observed held-out and out-of-support values must not be clipped or interpolated."""
    data = np.array([[0.1, 1.2, 2.3], [1.4, 2.5, 3.6], [2.7, 3.8, 4.9], [3.2, 1.1, 5.4]])

    def model(values):
        return values[:, 0] ** 2 + 2 * values[:, 1] - values[:, 2]

    imputer = imputer_type(model=model, data=data, x=x, sample_size=11, random_state=7)
    coalitions = np.array(list(product([False, True], repeat=3)))
    samples = imputer._draw_samples(x, coalitions)
    assert np.isfinite(samples).all()
    for coalition, completed in zip(coalitions, samples, strict=True):
        np.testing.assert_array_equal(completed[:, coalition], np.tile(x[coalition], (11, 1)))
    assert np.any(samples[1, :, :2] != np.floor(samples[1, :, :2]))
    np.testing.assert_allclose(imputer.value_function(coalitions[-1:]), model(x[None]))


@pytest.mark.parametrize("imputer_type", [GaussianImputer, GaussianCopulaImputer])
def test_one_feature_gaussian_sampling(imputer_type):
    data = np.array([[0.1], [0.5], [1.5], [2.7]])
    imputer = imputer_type(
        model=lambda values: values[:, 0],
        data=data,
        x=np.array([4]),
        sample_size=13,
        random_state=17,
    )
    samples = imputer._draw_samples(np.array([4]), np.array([[False], [True]]))
    assert samples.shape == (2, 13, 1)
    assert np.isfinite(samples).all()
    np.testing.assert_array_equal(samples[1], np.full((13, 1), 4))
    assert np.any(samples[0] != np.floor(samples[0]))
