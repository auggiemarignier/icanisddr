"""Test the relative parametrisation functions."""

import numpy as np
import pytest

from tti.traveltimes._parametrisations.relative import (
    RelativeLoveDegreeAngles,
    _jacobian_to_dm,
    _unpack_relative_model_vector,
)


@pytest.fixture
def m() -> np.ndarray:
    """Fixture for a relative model vector m containing fractional perturbations and angles in degrees."""
    return np.array(
        [
            [
                [0.1, -0.2, 0.3, -0.4, 0.5, 10.0, 20.0],
                [0.1, -0.2, 0.3, -0.4, 0.5, 10.0, 20.0],
            ],
            [[0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]],
        ]
    )


@pytest.fixture
def grad(rng: np.random.Generator) -> np.ndarray:
    """Fixture for a random gradient array with respect to the Love parameters and angles."""
    batch_size = rng.integers(1, 5)
    n_cells = rng.integers(1, 5)
    n_tt = rng.integers(1, 5)
    return rng.normal(size=(batch_size, n_cells, 7, n_tt))


@pytest.fixture
def ref() -> np.ndarray:
    """Fixture for a reference model vector."""
    return np.array([100.0, 200.0, 300.0, 400.0, 500.0])


def test_unpack_relative_model_vector(m: np.ndarray, ref: np.ndarray):
    """Test the unpacking of the relative model vector."""
    A, C, F, L, N, eta1, eta2 = _unpack_relative_model_vector(m, ref)
    np.testing.assert_allclose(A, np.array([[110.0, 110.0], [100.0, 100.0]]))
    np.testing.assert_allclose(C, np.array([[160.0, 160.0], [200.0, 200.0]]))
    np.testing.assert_allclose(F, np.array([[390.0, 390.0], [300.0, 300.0]]))
    np.testing.assert_allclose(L, np.array([[240.0, 240.0], [400.0, 400.0]]))
    np.testing.assert_allclose(N, np.array([[750.0, 750.0], [500.0, 500.0]]))
    np.testing.assert_allclose(eta1, np.radians(np.array([[10.0, 10.0], [0.0, 0.0]])))
    np.testing.assert_allclose(eta2, np.radians(np.array([[20.0, 20.0], [0.0, 0.0]])))


def test_jacobian_to_dm(grad: np.ndarray, ref: np.ndarray) -> None:
    """Test the Jacobian conversion from dt_dparams to dt_dm."""
    grad_dm = _jacobian_to_dm(grad, ref)
    jac = np.concatenate([ref, np.array([np.pi / 180.0, np.pi / 180.0])])
    expected = grad * jac[None, None, :, None]

    np.testing.assert_allclose(grad_dm, expected)


class TestRelativeLoveDegreeAngles:
    """Test the RelativeLoveDegreeAngles parametriser."""

    @pytest.fixture(autouse=True)
    def _init_parametriser(self, ref):
        self.parametriser = RelativeLoveDegreeAngles(ref)

    def test_num_model_params_per_segment(self) -> None:
        """Test that the number of model parameters per segment is correct."""
        assert self.parametriser.n_model_params_per_segment == 7

    def test_to_parameters_delegation(self, assert_delegates_to_unpack, m: np.ndarray):
        """Test that to_parameters delegates to the unpacking function."""

        assert_delegates_to_unpack(
            "tti.traveltimes._parametrisations.relative._unpack_relative_model_vector",
            self.parametriser,
            m,
            expected_args=(self.parametriser.reference_model,),
        )

    def test_apply_jacobian_delegation(
        self, assert_delegates_to_jacobian, grad: np.ndarray
    ):
        """Test that apply_jacobian delegates to the Jacobian conversion function."""

        assert_delegates_to_jacobian(
            "tti.traveltimes._parametrisations.relative._jacobian_to_dm",
            self.parametriser,
            grad,
            expected_args=(self.parametriser.reference_model,),
        )
