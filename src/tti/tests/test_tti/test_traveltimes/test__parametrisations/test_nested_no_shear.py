"""Test the basic nested parameterisation."""

import numpy as np
import pytest

from tti.traveltimes.parametrisations._abc import BaseParametriser as Parametriser
from tti.traveltimes.parametrisations.nested_no_shear import (
    NestedNoShearDegreesParametriser,
)


@pytest.fixture
def m(lv) -> np.ndarray:
    """Fixture for model vector m corresponding to the nested Love parameters and angles in degrees."""
    dC = lv.C - lv.A
    dF = lv.F - lv.A

    # Build m as (B, 5, M) then flatten to (B, 5*M) so reshape(batch, 5, -1)
    # will reconstruct the (B, 5, M) ordering (param-major).
    B, M = lv.A.shape
    m_nested = np.stack([lv.A, dC, dF, lv.eta1, lv.eta2], axis=1).reshape(B, 5 * M)
    return m_nested


@pytest.fixture
def parametriser() -> NestedNoShearDegreesParametriser:
    """Fixture for the NestedNoShearDegreesParametriser parametriser."""
    return NestedNoShearDegreesParametriser()


def test_num_model_params_per_segment(parametriser: Parametriser) -> None:
    """Test that the number of model parameters per segment is correct."""
    assert parametriser.n_model_params_per_segment == 5


def test_to_parameters(
    parametriser: Parametriser,
    m: np.ndarray,
    lv,
    assert_parametriser_matches_love_values,
) -> None:
    """Test that the to_parameters method correctly transforms the model vector and unpacks the parameters."""
    assert_parametriser_matches_love_values(parametriser, m, lv, include_shear=False)


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
