"""Test the relative parametrisation functions."""

import numpy as np
import pytest

from tti.traveltimes._parametrisations._abc import Parametriser
from tti.traveltimes._parametrisations.no_shear import (
    TRANSFORMATION as NO_SHEAR_TRANSFORMATION,
)
from tti.traveltimes._parametrisations.radians import (
    TRANSFORMATION as DEGREES_TO_RADIANS_TRANSFORMATION,
)
from tti.traveltimes._parametrisations.relative import (
    build_relative_transformation_matrix,
)
from tti.traveltimes._parametrisations.relative_no_shear import (
    RelativeNoShearLoveDegreeAngles,
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
    return np.array(
        [
            100.0,
            200.0,
            300.0,
        ]
    )


@pytest.fixture
def parametriser(ref: np.ndarray) -> RelativeNoShearLoveDegreeAngles:
    """Fixture for the RelativeNoShearLoveDegreeAngles parametriser."""
    return RelativeNoShearLoveDegreeAngles(reference_model=ref)


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


def test_relative_no_shear_composed_transformation() -> None:
    """Test that the RelativeNoShearLoveDegreeAngles class uses the correct composed transformation matrix."""
    ref3 = np.array([1.0, 2.0, 3.0])
    # The RelativeNoShear class expands the 3-element ref into 5 by appending zeros
    ref5 = np.concatenate([ref3, np.zeros(2)])
    expected = (
        DEGREES_TO_RADIANS_TRANSFORMATION
        @ build_relative_transformation_matrix(ref5)
        @ NO_SHEAR_TRANSFORMATION
    )

    p = RelativeNoShearLoveDegreeAngles(reference_model=ref3)
    assert np.allclose(p.transformation, expected)


def test_relative_no_shear_invalid_reference_length_raises() -> None:
    """Constructing with a reference model of wrong length should raise ValueError for no-shear variant."""
    # too short
    with pytest.raises(ValueError):
        RelativeNoShearLoveDegreeAngles(reference_model=np.array([1.0, 2.0]))

    # too long
    with pytest.raises(ValueError):
        RelativeNoShearLoveDegreeAngles(reference_model=np.array([1.0, 2.0, 3.0, 4.0]))


def test_relative_no_shear_none_uses_zero_reference() -> None:
    """Passing None as reference_model should use a zero reference and build the composed transformation accordingly for no-shear variant."""
    p = RelativeNoShearLoveDegreeAngles(reference_model=None)
    expected = (
        DEGREES_TO_RADIANS_TRANSFORMATION
        @ build_relative_transformation_matrix(np.zeros(5))
        @ NO_SHEAR_TRANSFORMATION
    )
    assert np.allclose(p.transformation, expected)


def test_reference_model_property_none_returns_zeros_no_shear() -> None:
    """When constructed with `reference_model=None`, the `reference_model` property should be a zero vector of length 5 for no-shear variant."""
    p = RelativeNoShearLoveDegreeAngles(reference_model=None)
    assert np.allclose(p.reference_model, np.zeros(5))
