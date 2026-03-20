"""Test the relative parametrisation functions."""

from functools import partial

import numpy as np
import pytest

from tti.traveltimes._parametrisations.relative_no_shear import (
    RelativeNoShearLoveDegreeAngles,
    _jacobian_to_dm,
    _transform_model_vector,
)


@pytest.fixture
def m(lv, ref: np.ndarray) -> np.ndarray:
    """Fixture for a relative model vector m containing fractional perturbations and angles in degrees."""
    # Build m as (B, 5, M) then flatten to (B, 5*M) (param-major ordering)
    B, M = lv.A.shape
    return np.stack(
        [
            (lv.A - ref[0]) / ref[0],
            (lv.C - ref[1]) / ref[1],
            (lv.F - ref[2]) / ref[2],
            lv.eta1,
            lv.eta2,
        ],
        axis=1,
    ).reshape(B, 5 * M)


@pytest.fixture
def ref() -> np.ndarray:
    """Fixture for a reference model vector."""
    return np.array([100.0, 200.0, 300.0, 0.0, 0.0])


def test__transform_model_vector(lv, m: np.ndarray, ref: np.ndarray) -> None:
    """Test transformation of a relative model vector constructed from LoveValues."""

    arr = _transform_model_vector(m, ref)
    A, C, F, L, N, eta1, eta2 = arr.swapaxes(0, 1)  # swap to (7, B, M)

    np.testing.assert_allclose(A, lv.A)
    np.testing.assert_allclose(C, lv.C)
    np.testing.assert_allclose(F, lv.F)
    np.testing.assert_allclose(L, np.zeros_like(lv.L))
    np.testing.assert_allclose(N, np.zeros_like(lv.N))
    np.testing.assert_allclose(eta1, np.radians(lv.eta1))
    np.testing.assert_allclose(eta2, np.radians(lv.eta2))


def test__jacobian_to_dm(grad_lv: np.ndarray, ref: np.ndarray) -> None:
    """Test the Jacobian conversion from dt_dparams to dt_dm."""
    grad_dm = _jacobian_to_dm(grad_lv, ref)
    jac = np.concatenate([ref, np.array([np.pi / 180.0, np.pi / 180.0])])
    expected = grad_lv * jac[None, None, :, None]
    expected = np.delete(expected, [3, 4], axis=2)  # remove dL and dN

    np.testing.assert_allclose(grad_dm, expected)


def test_jacobian_to_dm_finite_differences(
    m: np.ndarray, ref: np.ndarray, grad_lv: np.ndarray, numeric_apply_from_transform
) -> None:
    """Finite-difference check that `_jacobian_to_dm` matches numeric chain-rule for relative no-shear parametrisation."""
    unpack_with_ref = partial(_transform_model_vector, ref=ref)

    analytic = _jacobian_to_dm(grad_lv, ref)
    numeric = numeric_apply_from_transform(unpack_with_ref, m, grad_lv, eps=1e-6)

    np.testing.assert_allclose(analytic, numeric, rtol=1e-6, atol=1e-8)


class TestRelativeNoShearLoveDegreeAngles:
    """Test the RelativeNoShearLoveDegreeAngles parametriser."""

    @pytest.fixture(autouse=True)
    def _init_parametriser(self, ref):
        self.parametriser = RelativeNoShearLoveDegreeAngles(
            ref[:3]
        )  # only pass A, C, F as reference model; L and N get set to 0 in the constructor

    def test_num_model_params_per_segment(self) -> None:
        """Test that the number of model parameters per segment is correct."""
        assert self.parametriser.n_model_params_per_segment == 5

    def test_to_parameters_delegation(
        self, assert_delegates_to_transform, m: np.ndarray
    ):
        """Test that to_parameters delegates to the unpacking function."""

        assert_delegates_to_transform(
            "tti.traveltimes._parametrisations.relative_no_shear._transform_model_vector",
            self.parametriser,
            m,
            expected_args=(self.parametriser.reference_model,),
        )

    def test_apply_jacobian_delegation(
        self, assert_delegates_to_jacobian, grad_lv: np.ndarray
    ):
        """Test that apply_jacobian delegates to the Jacobian conversion function."""

        assert_delegates_to_jacobian(
            "tti.traveltimes._parametrisations.relative_no_shear._jacobian_to_dm",
            self.parametriser,
            grad_lv,
            expected_args=(self.parametriser.reference_model,),
        )
