"""Tests for prior functions."""

from collections.abc import Callable

import numpy as np
import pytest

from sddr.prior import (
    GaussianPrior,
    PriorComponent,
    UniformPrior,
    compound_prior_factory,
    gaussian_prior_factory,
    uniform_prior_factory,
)


class TestGaussianPriorFactory:
    """Tests for gaussian_prior_factory."""

    def test_gaussian_config_params_expose_mean_and_covar(self) -> None:
        """Gaussian prior should expose mean and covariance via config_params in order and by reference."""
        mean = np.array([1.0, -2.0, 3.0])
        covar = np.array(
            [
                [2.0, 0.1, 0.0],
                [0.1, 1.5, 0.2],
                [0.0, 0.2, 3.0],
            ]
        )
        prior = gaussian_prior_factory(mean, covar)
        assert isinstance(prior, GaussianPrior)

        cfg = prior.config_params
        assert isinstance(cfg, list)
        assert len(cfg) == 2

        # Check identity and values
        assert cfg[0] is mean
        assert cfg[1] is covar
        np.testing.assert_array_equal(cfg[0], mean)
        np.testing.assert_array_equal(cfg[1], covar)

        # Shape sanity
        assert cfg[0].shape == mean.shape
        assert cfg[1].shape == covar.shape

    def test_gaussian_prior_at_mean(self) -> None:
        """Test that log-prior is zero at the mean."""
        mean = np.array([1.0, 2.0, 3.0])
        covar = np.eye(3)
        prior_fn = gaussian_prior_factory(mean, covar)

        log_prior = prior_fn(mean)
        assert log_prior == 0.0

    def test_gaussian_prior_symmetric(self) -> None:
        """Test that log-prior is symmetric around the mean."""
        mean = np.array([0.0, 0.0])
        covar = np.eye(2)
        prior_fn = gaussian_prior_factory(mean, covar)

        delta = np.array([1.0, 1.0])
        log_prior_plus = prior_fn(mean + delta)
        log_prior_minus = prior_fn(mean - delta)

        np.testing.assert_allclose(log_prior_plus, log_prior_minus)

    def test_gaussian_prior_decreases_with_distance(self) -> None:
        """Test that log-prior decreases as we move away from the mean."""
        mean = np.array([0.0, 0.0])
        covar = np.eye(2)
        prior_fn = gaussian_prior_factory(mean, covar)

        close = np.array([0.1, 0.1])
        far = np.array([1.0, 1.0])

        log_prior_close = prior_fn(close)
        log_prior_far = prior_fn(far)

        assert log_prior_close > log_prior_far

    def test_gaussian_prior_with_correlation(self) -> None:
        """Test Gaussian prior with correlated covariance."""
        mean = np.array([0.0, 0.0])
        covar = np.array([[1.0, 0.5], [0.5, 1.0]])
        prior_fn = gaussian_prior_factory(mean, covar)

        params = np.array([1.0, 1.0])
        log_prior = prior_fn(params)

        # Should be finite and negative
        assert np.isfinite(log_prior)
        assert log_prior < 0

    def test_gaussian_prior_invalid_non_symmetric_covariance(self) -> None:
        """Test that non-symmetric covariance raises ValueError."""
        mean = np.array([0.0, 0.0])
        covar = np.array([[1.0, 0.5], [0.3, 1.0]])  # Non-symmetric

        with pytest.raises(ValueError, match="symmetric"):
            gaussian_prior_factory(mean, covar)

    def test_gaussian_prior_invalid_negative_eigenvalue(self) -> None:
        """Test that covariance with negative eigenvalues raises ValueError."""
        mean = np.array([0.0, 0.0])
        covar = np.array([[1.0, 2.0], [2.0, 1.0]])  # Not positive semidefinite

        with pytest.raises(ValueError, match="positive semidefinite"):
            gaussian_prior_factory(mean, covar)

    def test_gaussian_prior_invalid_shape_mismatch(self) -> None:
        """Test that mismatched mean and covariance shapes raise ValueError."""
        mean = np.array([0.0, 0.0])
        covar = np.eye(3)  # Wrong size

        with pytest.raises(ValueError, match="shape"):
            gaussian_prior_factory(mean, covar)

    def test_gaussian_prior_diagonal_covariance(self) -> None:
        """Test Gaussian prior with diagonal covariance (independent parameters)."""
        mean = np.array([1.0, 2.0, 3.0])
        variances = np.array([0.5, 1.0, 2.0])
        covar = np.diag(variances)
        prior_fn = gaussian_prior_factory(mean, covar)

        # Test a point one standard deviation away in first dimension
        params = mean.copy()
        params[0] += np.sqrt(variances[0])
        log_prior = prior_fn(params)

        # Expected: -0.5 * (1^2) = -0.5
        np.testing.assert_allclose(log_prior, -0.5, rtol=1e-10)


class TestUniformPriorFactory:
    """Tests for uniform_prior_factory."""

    @pytest.fixture
    def lower(self) -> np.ndarray:
        """Create lower bounds for testing."""
        return np.array([0.0, 0.0])

    @pytest.fixture
    def upper(self) -> np.ndarray:
        """Create upper bounds for testing."""
        return np.array([1.0, 1.0])

    @pytest.fixture
    def valid_uniform_prior(
        self, lower: np.ndarray, upper: np.ndarray
    ) -> Callable[[np.ndarray], float]:
        """Create a valid uniform prior function for testing."""
        return uniform_prior_factory(lower, upper)

    def test_uniform_config_params_expose_bounds(
        self,
        valid_uniform_prior: Callable[[np.ndarray], float],
        lower: np.ndarray,
        upper: np.ndarray,
    ) -> None:
        """Uniform prior should expose lower and upper bounds via config_params in order and by reference."""
        assert isinstance(valid_uniform_prior, UniformPrior)

        cfg = valid_uniform_prior.config_params
        assert isinstance(cfg, list)
        assert len(cfg) == 2

        # Check identity and values
        assert cfg[0] is lower
        assert cfg[1] is upper
        np.testing.assert_array_equal(cfg[0], lower)
        np.testing.assert_array_equal(cfg[1], upper)

        # Shape sanity
        assert cfg[0].shape == lower.shape
        assert cfg[1].shape == upper.shape

    def test_uniform_prior_in_bounds(
        self, valid_uniform_prior: Callable[[np.ndarray], float]
    ) -> None:
        """Test that log-prior is zero for parameters within bounds."""

        params = np.array([0.5, 0.5])
        log_prior = valid_uniform_prior(params)

        assert log_prior == 0.0

    def test_uniform_prior_at_lower_boundary(
        self, valid_uniform_prior: Callable[[np.ndarray], float], lower: np.ndarray
    ) -> None:
        """Test that log-prior is zero at the lower boundaries."""
        log_prior_lower = valid_uniform_prior(lower)
        assert log_prior_lower == 0.0

    def test_uniform_prior_at_upper_boundary(
        self, valid_uniform_prior: Callable[[np.ndarray], float], upper: np.ndarray
    ) -> None:
        """Test that log-prior is zero at the upper boundaries."""
        log_prior_upper = valid_uniform_prior(upper)
        assert log_prior_upper == 0.0

    def test_uniform_prior_out_of_bounds_below(
        self, valid_uniform_prior: Callable[[np.ndarray], float]
    ) -> None:
        """Test that log-prior is -inf for parameters below lower bound."""
        params = np.array([-0.1, 0.5])
        log_prior = valid_uniform_prior(params)

        assert log_prior == -np.inf

    def test_uniform_prior_out_of_bounds_above(
        self, valid_uniform_prior: Callable[[np.ndarray], float]
    ) -> None:
        """Test that log-prior is -inf for parameters above upper bound."""
        params = np.array([0.5, 1.1])
        log_prior = valid_uniform_prior(params)

        assert log_prior == -np.inf

    def test_uniform_prior_multiple_out_of_bounds(
        self, valid_uniform_prior: Callable[[np.ndarray], float]
    ) -> None:
        """Test that log-prior is -inf when multiple parameters are out of bounds."""
        params = np.array([-0.1, 1.1])
        log_prior = valid_uniform_prior(params)

        assert log_prior == -np.inf

    def test_uniform_prior_invalid_bounds(self) -> None:
        """Test that invalid bounds (lower >= upper) raise ValueError."""
        lower = np.array([0.0, 1.0])
        upper = np.array([1.0, 0.5])  # Second bound is invalid

        with pytest.raises(ValueError, match="lower bound must be less than"):
            uniform_prior_factory(lower, upper)

    def test_uniform_prior_equal_bounds(self) -> None:
        """Test that equal bounds raise ValueError."""
        lower = np.array([0.0, 1.0])
        upper = np.array([1.0, 1.0])  # Second bound is equal

        with pytest.raises(ValueError, match="lower bound must be less than"):
            uniform_prior_factory(lower, upper)

    def test_uniform_prior_different_ranges(self) -> None:
        """Test uniform prior with different ranges per dimension."""
        lower = np.array([0.0, -10.0, 5.0])
        upper = np.array([1.0, 10.0, 15.0])
        prior_fn = uniform_prior_factory(lower, upper)

        # In bounds
        params_in = np.array([0.5, 0.0, 10.0])
        assert prior_fn(params_in) == 0.0

        # Out of bounds in third dimension
        params_out = np.array([0.5, 0.0, 20.0])
        assert prior_fn(params_out) == -np.inf

    def test_uniform_prior_high_dimensional(self) -> None:
        """Test uniform prior in high-dimensional space."""
        n_dims = 10
        lower = np.zeros(n_dims)
        upper = np.ones(n_dims)
        prior_fn = uniform_prior_factory(lower, upper)

        # All in bounds
        params_in = np.full(n_dims, 0.5)
        assert prior_fn(params_in) == 0.0

        # One out of bounds
        params_out = np.full(n_dims, 0.5)
        params_out[5] = 1.1
        assert prior_fn(params_out) == -np.inf


class TestCompoundPriorFactory:
    """Tests for compound prior functions combining Gaussian and Uniform priors."""

    @pytest.fixture
    def compound_prior(self) -> Callable[[np.ndarray], float]:
        """Create a compound prior for testing."""
        # Gaussian prior on first two parameters
        mean = np.array([0.0, 0.0])
        covar = np.eye(2)
        gaussian_prior = gaussian_prior_factory(mean, covar)
        gaussian_component = PriorComponent(prior_fn=gaussian_prior, indices=[0, 1])

        # Uniform prior on last two parameters
        lower = np.array([-1.0, -1.0])
        upper = np.array([1.0, 1.0])
        uniform_prior = uniform_prior_factory(lower, upper)
        uniform_component = PriorComponent(prior_fn=uniform_prior, indices=[2, 3])

        # Combine into compound prior
        return compound_prior_factory([gaussian_component, uniform_component])

    def test_compound_prior_valid_model(
        self, compound_prior: Callable[[np.ndarray], float]
    ) -> None:
        """Test the compound prior with a valid model."""

        # Test point within both priors
        model = np.array(
            [1.0, -1.0, 0.0, 0.0]
        )  # 1 stddev away from the mean of the gaussian prior and within uniform prior
        log_prior = compound_prior(model)

        # Gaussian prior log-prob at [1.0, -1.0] is -1, uniform prior log-prob within bounds is 0
        expected_log_prior = -1.0
        np.testing.assert_almost_equal(log_prior, expected_log_prior)

    def test_compound_prior_invalid_model(
        self, compound_prior: Callable[[np.ndarray], float]
    ) -> None:
        """Test the compound prior with a model that has out-of-bounds parameters."""
        # Test point outside uniform prior
        model = np.array([0.1, -0.1, 2.0, -0.5])
        log_prior_out_uniform = compound_prior(model)
        assert log_prior_out_uniform == -np.inf
