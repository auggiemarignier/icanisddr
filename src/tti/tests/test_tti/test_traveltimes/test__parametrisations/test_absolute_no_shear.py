"""Tests for absolute without shear parametrisation."""

import numpy as np
import pytest

from tti.traveltimes._parametrisations.absolute_no_shear import (
    AbsoluteNoShearLoveDegreeAngles,
    _jacobian_to_dm,
    _unpack_model_vector_no_shear,
)


@pytest.fixture
def m(lv) -> np.ndarray:
    """Fixture for model vector m corresponding to the Love parameters and angles in degrees."""
    # Build m as (B, M, 5) then flatten to (B, 5M)
    B, M = lv.A.shape
    m = np.stack([lv.A, lv.C, lv.F, lv.eta1, lv.eta2], axis=-1).reshape(B, 5 * M)
    return m


def test__unpack_model_vector(lv, m: np.ndarray) -> None:
    """Test unpacking of model vector into Love parameters.

    Tests that the unpacking function correctly extracts each parameter
    from the model vector.
    The angles eta1 and eta2 should be converted from degrees to radians.
    """
    A, C, F, L, N, eta1, eta2 = _unpack_model_vector_no_shear(m)

    np.testing.assert_allclose(A, lv.A)
    np.testing.assert_allclose(C, lv.C)
    np.testing.assert_allclose(F, lv.F)
    np.testing.assert_allclose(L, np.zeros_like(lv.L))
    np.testing.assert_allclose(N, np.zeros_like(lv.N))
    np.testing.assert_allclose(eta1, np.radians(lv.eta1))
    np.testing.assert_allclose(eta2, np.radians(lv.eta2))


def test__jacobian_to_dm(grad_lv: np.ndarray) -> None:
    """Test conversion of gradient from dt_dparams to dt_dm."""
    result = _jacobian_to_dm(grad_lv)
    expected = grad_lv.copy()
    expected = np.delete(expected, [3, 4], axis=2)  # remove dL and dN
    expected[..., 3:5, :] *= np.pi / 180.0
    np.testing.assert_allclose(result, expected)


def test__jacobian_to_dm_finite_differences(
    grad_lv: np.ndarray, m: np.ndarray, numeric_apply_from_unpack
) -> None:
    """Finite-difference check that `_jacobian_to_dm` matches numeric chain-rule for no-shear case."""
    analytic = _jacobian_to_dm(grad_lv)

    numeric = numeric_apply_from_unpack(
        _unpack_model_vector_no_shear, m, grad_lv, eps=1e-6
    )

    np.testing.assert_allclose(analytic, numeric, rtol=1e-6, atol=1e-8)


class TestAbsoluteNoShearLoveDegreeAnglesParametriser:
    """Testing the Parametriser.

    It is mostly just a wrapper around the unpacking and jacobian functions.
    """

    parametriser = AbsoluteNoShearLoveDegreeAngles()

    def test_num_model_params_per_segment(self) -> None:
        """Test that the number of model parameters per segment is correct."""
        assert self.parametriser.n_model_params_per_segment == 5

    def test_to_parameters(self, m: np.ndarray, assert_delegates_to_unpack) -> None:
        """Test that to_parameters delegates to the unpacking function."""
        assert_delegates_to_unpack(
            "tti.traveltimes._parametrisations.absolute_no_shear._unpack_model_vector_no_shear",
            self.parametriser,
            m,
        )

    def test_apply_jacobian(
        self, grad_lv: np.ndarray, assert_delegates_to_jacobian
    ) -> None:
        """Test that apply_jacobian delegates to the jacobian conversion."""
        assert_delegates_to_jacobian(
            "tti.traveltimes._parametrisations.absolute_no_shear._jacobian_to_dm",
            self.parametriser,
            grad_lv,
        )
