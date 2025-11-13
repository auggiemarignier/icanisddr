"""Tests for prior functions."""

from collections.abc import Callable

import numpy as np
import pytest

from sdicani.sddr.prior import (
    CompoundPrior,
    GaussianPrior,
    PriorComponent,
    PriorFunction,
    UniformPrior,
    marginalise_prior,
)


class TestGaussianPrior:
    """Tests for GaussianPrior."""

    def test_gaussian_prior_n(self) -> None:
        """Test that Gaussian prior has correct number of parameters."""
        mean = np.array([1.0, 2.0, 3.0])
        covar = np.eye(3)
        prior_fn = GaussianPrior(mean, covar)
        assert prior_fn.n == 3

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
        prior = GaussianPrior(mean, covar)
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
        prior_fn = GaussianPrior(mean, covar)

        log_prior = prior_fn(mean)
        assert log_prior == 0.0

    def test_gaussian_prior_symmetric(self) -> None:
        """Test that log-prior is symmetric around the mean."""
        mean = np.array([0.0, 0.0])
        covar = np.eye(2)
        prior_fn = GaussianPrior(mean, covar)

        delta = np.array([1.0, 1.0])
        log_prior_plus = prior_fn(mean + delta)
        log_prior_minus = prior_fn(mean - delta)

        np.testing.assert_allclose(log_prior_plus, log_prior_minus)

    def test_gaussian_prior_decreases_with_distance(self) -> None:
        """Test that log-prior decreases as we move away from the mean."""
        mean = np.array([0.0, 0.0])
        covar = np.eye(2)
        prior_fn = GaussianPrior(mean, covar)

        close = np.array([0.1, 0.1])
        far = np.array([1.0, 1.0])

        log_prior_close = prior_fn(close)
        log_prior_far = prior_fn(far)

        assert log_prior_close > log_prior_far

    def test_gaussian_prior_with_correlation(self) -> None:
        """Test Gaussian prior with correlated covariance."""
        mean = np.array([0.0, 0.0])
        covar = np.array([[1.0, 0.5], [0.5, 1.0]])
        prior_fn = GaussianPrior(mean, covar)

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
            GaussianPrior(mean, covar)

    def test_gaussian_prior_invalid_negative_eigenvalue(self) -> None:
        """Test that covariance with negative eigenvalues raises ValueError."""
        mean = np.array([0.0, 0.0])
        covar = np.array([[1.0, 2.0], [2.0, 1.0]])  # Not positive semidefinite

        with pytest.raises(ValueError, match="positive semidefinite"):
            GaussianPrior(mean, covar)

    def test_gaussian_prior_invalid_shape_mismatch(self) -> None:
        """Test that mismatched mean and covariance shapes raise ValueError."""
        mean = np.array([0.0, 0.0])
        covar = np.eye(3)  # Wrong size

        with pytest.raises(ValueError, match="shape"):
            GaussianPrior(mean, covar)

    def test_gaussian_prior_diagonal_covariance(self) -> None:
        """Test Gaussian prior with diagonal covariance (independent parameters)."""
        mean = np.array([1.0, 2.0, 3.0])
        variances = np.array([0.5, 1.0, 2.0])
        covar = np.diag(variances)
        prior_fn = GaussianPrior(mean, covar)

        # Test a point one standard deviation away in first dimension
        params = mean.copy()
        params[0] += np.sqrt(variances[0])
        log_prior = prior_fn(params)

        # Expected: -0.5 * (1^2) = -0.5
        np.testing.assert_allclose(log_prior, -0.5, rtol=1e-10)


class TestUniformPrior:
    """Tests for UniformPrior."""

    @pytest.fixture
    def lower(self) -> np.ndarray:
        """Create lower bounds for testing."""
        return np.array([0.0, 0.0])

    @pytest.fixture
    def upper(self) -> np.ndarray:
        """Create upper bounds for testing."""
        return np.array([1.0, 1.0])

    @pytest.fixture
    def valid_uniform_prior(self, lower: np.ndarray, upper: np.ndarray) -> UniformPrior:
        """Create a valid uniform prior function for testing."""
        return UniformPrior(lower, upper)

    def test_uniform_prior_n(self, valid_uniform_prior: UniformPrior) -> None:
        """Test that uniform prior has correct number of parameters."""
        assert valid_uniform_prior.n == 2

    def test_uniform_config_params_expose_bounds(
        self,
        valid_uniform_prior: UniformPrior,
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
            UniformPrior(lower, upper)

    def test_uniform_prior_equal_bounds(self) -> None:
        """Test that equal bounds raise ValueError."""
        lower = np.array([0.0, 1.0])
        upper = np.array([1.0, 1.0])  # Second bound is equal

        with pytest.raises(ValueError, match="lower bound must be less than"):
            UniformPrior(lower, upper)

    def test_uniform_prior_different_ranges(self) -> None:
        """Test uniform prior with different ranges per dimension."""
        lower = np.array([0.0, -10.0, 5.0])
        upper = np.array([1.0, 10.0, 15.0])
        prior_fn = UniformPrior(lower, upper)

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
        prior_fn = UniformPrior(lower, upper)

        # All in bounds
        params_in = np.full(n_dims, 0.5)
        assert prior_fn(params_in) == 0.0

        # One out of bounds
        params_out = np.full(n_dims, 0.5)
        params_out[5] = 1.1
        assert prior_fn(params_out) == -np.inf


class TestMarginalisation:
    """Tests for marginalising the base distributions."""

    @pytest.fixture
    def gaussian_prior(self) -> Callable[[np.ndarray], float]:
        """Create a Gaussian prior for testing."""
        mean = np.array([0.0, 0.0, 0.0])
        covar = np.eye(3)
        return GaussianPrior(mean, covar)

    @pytest.fixture
    def uniform_prior(self) -> Callable[[np.ndarray], float]:
        """Create a Uniform prior for testing."""
        lower = np.array([-1.0, -1.0, -1.0])
        upper = np.array([1.0, 1.0, 1.0])
        return UniformPrior(lower, upper)

    @pytest.mark.parametrize(
        ("fixture_name", "return_type"),
        [("gaussian_prior", GaussianPrior), ("uniform_prior", UniformPrior)],
    )
    def test_marginalise_return_type(
        self,
        fixture_name: str,
        return_type: type[PriorFunction],
        request: pytest.FixtureRequest,
    ) -> None:
        """Test that marginalisation returns a callable prior function."""
        prior_fn = request.getfixturevalue(fixture_name)
        assert isinstance(marginalise_prior(prior_fn, [0, 2]), return_type)

    @pytest.mark.parametrize(
        ("fixture_name", "expected_result"),
        [("gaussian_prior", -1.0), ("uniform_prior", 0.0)],
    )
    def test_marginalise_log_prior_computation(
        self, fixture_name: str, expected_result: float, request: pytest.FixtureRequest
    ) -> None:
        """Test that marginalised prior computes correct log-prior values."""
        prior_fn = request.getfixturevalue(fixture_name)
        marginalised_prior = marginalise_prior(prior_fn, [0, 2])
        test_point = np.array([1.0, 1.0])

        log_prior = marginalised_prior(test_point)
        np.testing.assert_almost_equal(log_prior, expected_result)

    @pytest.mark.parametrize("fixture_name", ["gaussian_prior", "uniform_prior"])
    def test_marginalise_invalid_indices(
        self, fixture_name: str, request: pytest.FixtureRequest
    ) -> None:
        """Test that invalid indices raise ValueError."""
        prior_fn = request.getfixturevalue(fixture_name)

        with pytest.raises(IndexError, match="out of bounds"):
            marginalise_prior(prior_fn, [0, 5])  # 5 is out of bounds

    @pytest.mark.parametrize("fixture_name", ["gaussian_prior", "uniform_prior"])
    def test_marginalise_slice(
        self, fixture_name: str, request: pytest.FixtureRequest
    ) -> None:
        """Test that a slice gives the same as a list of indices."""
        prior_fn = request.getfixturevalue(fixture_name)
        marginalised_prior_list = marginalise_prior(prior_fn, [0, 2])
        marginalised_prior_slice = marginalise_prior(prior_fn, slice(0, 3, 2))

        test_point = np.array([1.0, 1.0])
        log_prior_list = marginalised_prior_list(test_point)
        log_prior_slice = marginalised_prior_slice(test_point)
        np.testing.assert_almost_equal(log_prior_list, log_prior_slice)

    @pytest.mark.parametrize("fixture_name", ["gaussian_prior", "uniform_prior"])
    def test_marginalise_all_indices(
        self, fixture_name: str, request: pytest.FixtureRequest
    ) -> None:
        """Test that marginalising all indices returns the original prior."""
        prior_fn = request.getfixturevalue(fixture_name)
        n_dims = len(
            prior_fn.config_params[0]
        )  # Assuming first config param is mean/lower bound
        all_indices = list(range(n_dims))

        marginalised_prior = marginalise_prior(prior_fn, all_indices)

        test_point = np.array([0.5] * n_dims)
        log_prior_original = prior_fn(test_point)
        log_prior_marginalised = marginalised_prior(test_point)

        np.testing.assert_almost_equal(log_prior_original, log_prior_marginalised)

    @pytest.mark.parametrize("fixture_name", ["gaussian_prior", "uniform_prior"])
    def test_marginalise_no_indices(
        self, fixture_name: str, request: pytest.FixtureRequest
    ) -> None:
        """Test that marginalising over all variables raises a ValueError."""
        prior_fn = request.getfixturevalue(fixture_name)

        with pytest.raises(ValueError, match="At least one index"):
            marginalise_prior(prior_fn, np.array([]))


class TestPriorComponent:
    """Tests for PriorComponent dataclass."""

    def test_prior_component_with_list_indices(self) -> None:
        """Test that PriorComponent stores prior function and indices correctly."""
        mean = np.array([0.0, 0.0])
        covar = np.eye(2)
        prior_fn = GaussianPrior(mean, covar)
        indices = [0, 1]

        component = PriorComponent(prior_fn=prior_fn, indices=indices)

        assert component.prior_fn is prior_fn
        np.testing.assert_array_equal(component.indices, np.array(indices))
        assert component.n == 2

    def test_prior_component_with_slice(self) -> None:
        """Test that PriorComponent can store indices as a slice."""
        lower = np.array([-1.0, -1.0])
        upper = np.array([1.0, 1.0])
        prior_fn = UniformPrior(lower, upper)
        indices = slice(0, 2)

        component = PriorComponent(prior_fn=prior_fn, indices=indices)

        assert component.prior_fn is prior_fn
        np.testing.assert_array_equal(component.indices, np.arange(0, 2))
        assert component.n == 2


class TestCompoundPrior:
    """Tests for compound prior functions combining Gaussian and Uniform priors."""

    @pytest.fixture
    def compound_prior(self) -> CompoundPrior:
        """Create a compound prior for testing."""
        # Gaussian prior on first two parameters
        mean = np.array([0.0, 0.0])
        covar = np.eye(2)
        gaussian_prior = GaussianPrior(mean, covar)
        gaussian_component = PriorComponent(prior_fn=gaussian_prior, indices=[0, 1])

        # Uniform prior on last two parameters
        lower = np.array([-1.0, -1.0])
        upper = np.array([1.0, 1.0])
        uniform_prior = UniformPrior(lower, upper)
        uniform_component = PriorComponent(prior_fn=uniform_prior, indices=[2, 3])

        # Combine into compound prior
        return CompoundPrior([gaussian_component, uniform_component])

    def test_compound_prior_n(self, compound_prior: CompoundPrior) -> None:
        """Test that compound prior infers correct number of parameters from components."""
        assert compound_prior.n == 4

    def test_compound_prior_valid_model(self, compound_prior: CompoundPrior) -> None:
        """Test the compound prior with a valid model."""

        # Test point within both priors
        model = np.array(
            [1.0, -1.0, 0.0, 0.0]
        )  # 1 stddev away from the mean of the gaussian prior and within uniform prior
        log_prior = compound_prior(model)

        # Gaussian prior log-prob at [1.0, -1.0] is -1, uniform prior log-prob within bounds is 0
        expected_log_prior = -1.0
        np.testing.assert_almost_equal(log_prior, expected_log_prior)

    def test_compound_prior_invalid_model(self, compound_prior: CompoundPrior) -> None:
        """Test the compound prior with a model that has out-of-bounds parameters."""
        # Test point outside uniform prior
        model = np.array([0.1, -0.1, 2.0, -0.5])
        log_prior_out_uniform = compound_prior(model)
        assert log_prior_out_uniform == -np.inf

    def test_compound_prior_marginalisation(
        self, compound_prior: CompoundPrior
    ) -> None:
        """Test marginalisation of the compound prior."""
        # Marginalise over the first and last parameters (one from Gaussian, one from Uniform)
        marginalised_prior = marginalise_prior(compound_prior, [1, 2])

        # test point 1 stdev away in Gaussian component, and within Uniform component
        test_point = np.array([1.0, 0.0])

        expected_log_prior = -0.5 * (1.0**2) + 0.0  # Gaussian part + Uniform part
        log_prior = marginalised_prior(test_point)
        np.testing.assert_almost_equal(log_prior, expected_log_prior)
