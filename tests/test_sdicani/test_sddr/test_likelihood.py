"""Test the likelihood functions."""

import numpy as np
import pytest

from sdicani.sddr.likelihood import gaussian_likelihood_factory


def _dummy_forward_fn(model_params: np.ndarray) -> np.ndarray:
    """A simple forward function for testing purposes."""
    return model_params * 2.0


def test_gaussian_likelihood_factory() -> None:
    """Test the Gaussian likelihood factory."""
    observed_data = np.array([1.0, 2.0, 3.0])
    covar = np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]])

    likelihood_fn = gaussian_likelihood_factory(_dummy_forward_fn, observed_data, covar)

    model_params = observed_data / 2.0  # So that predicted data matches observed data
    log_likelihood = likelihood_fn(model_params)

    expected_log_likelihood = 0.0  # Perfect match

    assert np.isclose(log_likelihood, expected_log_likelihood)


def test_invalid_asymmetric_covariance_matrix() -> None:
    """Test that an asymmetrical covariance matrix raises a ValueError."""
    observed_data = np.array([1.0, 2.0])
    covar = np.array([[1.0, 2.0], [0.0, 1.0]])  # Asymmetric

    with pytest.raises(ValueError, match="Covariance matrix must be symmetric."):
        gaussian_likelihood_factory(_dummy_forward_fn, observed_data, covar)


def test_invalid_non_positive_semidefinite_covariance_matrix() -> None:
    """Test that a non-positive semidefinite covariance matrix raises a ValueError."""
    observed_data = np.array([1.0, 2.0])
    covar = np.array([[1.0, 2.0], [2.0, 1.0]])  # Not positive semidefinite

    with pytest.raises(
        ValueError, match="Inverse covariance matrix must be positive semidefinite."
    ):
        gaussian_likelihood_factory(_dummy_forward_fn, observed_data, covar)


def test_invalid_data_vector_dimension() -> None:
    """Test that a non-one-dimensional data vector raises a ValueError."""
    observed_data = np.array([[1.0, 2.0], [3.0, 4.0]])  # 2D array
    covar = np.array([[1.0, 0.0], [0.0, 1.0]])

    with pytest.raises(ValueError, match="Data vector must be one-dimensional."):
        gaussian_likelihood_factory(_dummy_forward_fn, observed_data, covar)


def test_invalid_forward_function_output_dimension() -> None:
    """Test that a forward function returning incorrect output dimension raises a ValueError.

    This only happens when an example_model is provided to the factory.
    """
    observed_data = np.array([1.0, 2.0])
    covar = np.array([[1.0, 0.0], [0.0, 1.0]])

    def bad_forward_fn(model_params: np.ndarray) -> np.ndarray:
        return np.array([1.0, 2.0, 3.0])  # wrong length

    with pytest.raises(
        ValueError,
        match="shape",
    ):
        gaussian_likelihood_factory(
            bad_forward_fn, observed_data, covar, example_model=np.array([0.0, 0.0])
        )

    try:
        _ = gaussian_likelihood_factory(
            bad_forward_fn, observed_data, covar
        )  # a bad forward function but no example_model, so no check
    except Exception as e:
        pytest.fail(f"Unexpected exception: {e}")

    try:
        _ = gaussian_likelihood_factory(
            _dummy_forward_fn, observed_data, covar, example_model=np.array([0.0, 0.0])
        )  # valid forward function verification
    except Exception as e:
        pytest.fail(f"Unexpected exception: {e}")


def test_invalid_covariance_matrix_size() -> None:
    """Test that a covariance matrix with incorrect size raises a ValueError."""
    observed_data = np.array([1.0, 2.0, 3.0])
    covar = np.array([[1.0, 0.0], [0.0, 1.0]])  # 2x2 instead of 3x3

    with pytest.raises(
        ValueError,
        match="shape",
    ):
        gaussian_likelihood_factory(_dummy_forward_fn, observed_data, covar)
