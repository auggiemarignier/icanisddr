"""Tests for absolute parametrisation."""

import numpy as np
import pytest

from tti.traveltimes.parametrisations._abc import BaseParametriser as Parametriser
from tti.traveltimes.parametrisations.radians import AbsoluteDegreesParametriser


@pytest.fixture
def m(lv) -> np.ndarray:
    """Fixture for model vector m corresponding to the Love parameters and angles in degrees."""
    # Build m as (B, 7, M) then flatten to (B, 7*M) so reshape(batch, 7, -1)
    # will reconstruct the (B, 7, M) ordering used by unpackers that expect
    # param-major flattening.
    B, M = lv.A.shape
    m = np.stack([lv.A, lv.C, lv.F, lv.L, lv.N, lv.eta1, lv.eta2], axis=1).reshape(
        B, 7 * M
    )
    return m


@pytest.fixture
def parametriser() -> AbsoluteDegreesParametriser:
    """Fixture for the AbsoluteDegreesParametriser parametriser."""
    return AbsoluteDegreesParametriser()


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
