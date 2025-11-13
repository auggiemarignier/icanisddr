"""Test the posterior module."""

import numpy as np
import pytest

from sddr.posterior import marginalise_samples, posterior_factory


def test_posterior_factory():
    """Test the posterior factory function."""

    def likelihood_fn(params: np.ndarray) -> float:
        return -0.5 * np.sum(params**2)

    def prior_fn(params: np.ndarray) -> float:
        return -np.sum(np.abs(params))

    posterior_fn = posterior_factory(likelihood_fn, prior_fn)

    params = np.array([1.0, 2.0, -1.5])
    log_posterior = posterior_fn(params)

    expected_log_likelihood = -0.5 * (1.0**2 + 2.0**2 + (-1.5) ** 2)
    expected_log_prior = -(np.abs(1.0) + np.abs(2.0) + np.abs(-1.5))
    expected_log_posterior = expected_log_likelihood + expected_log_prior

    assert np.isclose(log_posterior, expected_log_posterior)


@pytest.fixture
def samples() -> np.ndarray:
    """Fixture for sample posterior samples."""
    return np.array(
        [
            [0.5, 1.0, -0.5, 2.0],
            [1.5, -1.0, 0.5, -2.0],
            [-0.5, 0.0, 1.5, 3.0],
        ]
    )


def test_marginalise_posterior_samples_with_indices(samples):
    """Test the marginalisation of posterior samples using indices to select which parameters we want to keep."""

    marginal_samples = marginalise_samples(samples, [0, 1])

    expected_marginal_samples = np.array(
        [
            [0.5, 1.0],
            [1.5, -1.0],
            [-0.5, 0.0],
        ]
    )

    assert np.array_equal(marginal_samples, expected_marginal_samples)


def test_marginalise_posterior_samples_with_slice(samples):
    """Test the marginalisation of posterior samples using a slice to select which parameters we want to keep."""

    marginal_samples = marginalise_samples(samples, slice(1, 4))

    expected_marginal_samples = np.array(
        [
            [1.0, -0.5, 2.0],
            [-1.0, 0.5, -2.0],
            [0.0, 1.5, 3.0],
        ]
    )

    assert np.array_equal(marginal_samples, expected_marginal_samples)
