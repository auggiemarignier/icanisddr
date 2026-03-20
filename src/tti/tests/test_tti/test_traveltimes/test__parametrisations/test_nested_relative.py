"""Test the nested relative parameterisation."""

import numpy as np
import pytest

from tti.traveltimes._parametrisations._abc import Parametriser
from tti.traveltimes._parametrisations.nested_relative import (
    NestedRelativeLoveDegreeAngles,
)


@pytest.fixture
def m(lv, ref: np.ndarray) -> np.ndarray:
    """Fixture for a nested model vector m containing fractional perturbations and angles in degrees."""
    dC = (lv.C - ref[1]) / ref[1] - (lv.A - ref[0]) / ref[0]
    dF = (lv.F - ref[2]) / ref[2] - (
        (lv.A - ref[0]) / ref[0] - 2 * (lv.N - ref[4]) / ref[4]
    )
    dN = (lv.N - ref[4]) / ref[4] - (lv.L - ref[3]) / ref[3]

    # Build m as (B, 7, M) then flatten to (B, 7*M) so reshape(batch, 7, -1)
    # will reconstruct the (B, 7, M) ordering used by unpackers that expect
    # param-major flattening.
    B, M = lv.A.shape
    m_nested = np.stack(
        [
            (lv.A - ref[0]) / ref[0],
            dC,
            dF,
            (lv.L - ref[3]) / ref[3],
            dN,
            lv.eta1,
            lv.eta2,
        ],
        axis=1,
    ).reshape(B, 7 * M)
    return m_nested


@pytest.fixture
def ref() -> np.ndarray:
    """Fixture for a reference model vector."""
    return np.array([100.0, 200.0, 300.0, 400.0, 500.0])


@pytest.fixture
def parametriser(ref: np.ndarray) -> NestedRelativeLoveDegreeAngles:
    """Fixture for the NestedRelativeLoveDegreeAngles parametriser."""
    return NestedRelativeLoveDegreeAngles(reference_model=ref)


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
