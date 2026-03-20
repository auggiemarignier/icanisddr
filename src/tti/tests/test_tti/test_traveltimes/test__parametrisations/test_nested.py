"""Test the basic nested parameterisation."""

import numpy as np
import pytest

from tti.traveltimes._parametrisations.nested import (
    NestedLoveDegreeAngles,
    _jacobian_to_dm,
    _unpack_nested_model_vector,
)


@pytest.fixture
def m(lv) -> np.ndarray:
    """Fixture for model vector m corresponding to the nested Love parameters and angles in degrees."""
    dC = lv.C - lv.A
    dF = lv.F - (lv.A - 2 * lv.N)
    dN = lv.N - lv.L

    # Build m as (B, 7, M) then flatten to (B, 7*M) so reshape(batch, 7, -1)
    # will reconstruct the (B, 7, M) ordering used by unpackers that expect
    # param-major flattening.
    B, M = lv.A.shape
    m_nested = np.stack([lv.A, dC, dF, lv.L, dN, lv.eta1, lv.eta2], axis=1).reshape(
        B, 7 * M
    )
    return m_nested


def test__unpack_nested_model_vector(lv, m: np.ndarray) -> None:
    """Test unpacking of nested model vector into Love parameters.

    Tests that the nested unpacking correctly reconstructs Love parameters
    from their nested differences representation and angles in radians.
    """

    A, C, F, L, N, eta1, eta2 = _unpack_nested_model_vector(m)

    np.testing.assert_allclose(A, lv.A)
    np.testing.assert_allclose(C, lv.C)
    np.testing.assert_allclose(F, lv.F)
    np.testing.assert_allclose(L, lv.L)
    np.testing.assert_allclose(N, lv.N)
    np.testing.assert_allclose(eta1, np.radians(lv.eta1))
    np.testing.assert_allclose(eta2, np.radians(lv.eta2))


def test__jacobian_to_dm_finite_differences(
    m: np.ndarray, grad_lv: np.ndarray, numeric_apply_from_unpack
) -> None:
    """Finite-difference check that `_jacobian_to_dm` matches numeric chain-rule for nested parametrisation."""
    analytic = _jacobian_to_dm(grad_lv)

    numeric = numeric_apply_from_unpack(
        _unpack_nested_model_vector, m, grad_lv, eps=1e-6
    )

    np.testing.assert_allclose(analytic, numeric, rtol=1e-6, atol=1e-8)


class TestNestedLoveDegreeAnglesParametriser:
    """Testing the NestedLoveDegreeAngles Parametriser.

    It is mostly just a wrapper around the unpacking and jacobian functions.
    """

    parametriser = NestedLoveDegreeAngles()

    def test_num_model_params_per_segment(self) -> None:
        """Test that the number of model parameters per segment is correct."""
        assert self.parametriser.n_model_params_per_segment == 7

    def test_to_parameters(self, m: np.ndarray, assert_delegates_to_unpack) -> None:
        """Test that to_parameters delegates to the unpacking function."""
        assert_delegates_to_unpack(
            "tti.traveltimes._parametrisations.nested._unpack_nested_model_vector",
            self.parametriser,
            m,
        )

    def test_apply_jacobian(
        self, grad_lv: np.ndarray, assert_delegates_to_jacobian
    ) -> None:
        """Test that apply_jacobian delegates to the jacobian conversion."""
        assert_delegates_to_jacobian(
            "tti.traveltimes._parametrisations.nested._jacobian_to_dm",
            self.parametriser,
            grad_lv,
        )
