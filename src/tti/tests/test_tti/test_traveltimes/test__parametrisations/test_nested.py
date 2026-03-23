"""Test the basic nested parameterisation."""

import numpy as np
import pytest

from tti.traveltimes._parametrisations._abc import Parametriser
from tti.traveltimes._parametrisations.nested import NestedLoveRadianAngles


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


@pytest.fixture
def parametriser() -> NestedLoveRadianAngles:
    """Fixture for the NestedLoveDegreeAngles parametriser."""
    return NestedLoveRadianAngles()


def test_num_model_params_per_segment(parametriser: Parametriser) -> None:
    """Test that the number of model parameters per segment is correct."""
    assert parametriser.n_model_params_per_segment == 7


def test_to_parameters(
    parametriser: Parametriser,
    m: np.ndarray,
    lv,
    assert_parametriser_matches_love_values,
) -> None:
    """Test that the to_parameters method correctly transforms the model vector and unpacks the parameters."""
    assert_parametriser_matches_love_values(parametriser, m, lv, include_shear=True)


def test_apply_jacobian(
    parametriser: Parametriser,
    grad_lv: np.ndarray,
    m: np.ndarray,
    assert_jacobian_matches_finite_difference,
) -> None:
    """Test that the apply_jacobian method correctly applies the Jacobian to convert from dt_dparams to dt_dm.

    Uses a finite-difference approximation to check that the Jacobian is applied correctly.
    """
    assert_jacobian_matches_finite_difference(parametriser, grad_lv, m)
