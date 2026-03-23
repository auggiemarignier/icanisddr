"""Test the relative parametrisation functions."""

import numpy as np
import pytest

from tti.traveltimes._parametrisations._abc import BaseParametriser as Parametriser
from tti.traveltimes._parametrisations.radians import (
    TRANSFORMATION as DEGREES_TO_RADIANS_TRANSFORMATION,
)
from tti.traveltimes._parametrisations.relative import (
    RelativeFractionalDegreesParametriser,
    build_relative_transformation_matrix,
)


@pytest.fixture
def m(lv, ref: np.ndarray) -> np.ndarray:
    """Fixture for a relative model vector m containing fractional perturbations and angles in degrees."""
    # Build m as (B, 7, M) then flatten to (B, 7*M) (param-major ordering)
    B, M = lv.A.shape
    return np.stack(
        [
            (lv.A - ref[0]) / ref[0],
            (lv.C - ref[1]) / ref[1],
            (lv.F - ref[2]) / ref[2],
            (lv.L - ref[3]) / ref[3],
            (lv.N - ref[4]) / ref[4],
            lv.eta1,
            lv.eta2,
        ],
        axis=1,
    ).reshape(B, 7 * M)


@pytest.fixture
def ref() -> np.ndarray:
    """Fixture for a reference model vector."""
    return np.array([100.0, 200.0, 300.0, 400.0, 500.0])


@pytest.fixture
def parametriser(ref: np.ndarray) -> RelativeFractionalDegreesParametriser:
    """Fixture for the RelativeFractionalDegreesParametriser parametriser."""
    return RelativeFractionalDegreesParametriser(reference_model=ref)


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


def test_build_relative_transformation_matrix_shape_and_values() -> None:
    """Test the shape and returned values of the build_relative_transformation_matrix function."""
    ref = np.array([10.0, 20.0, 30.0, 40.0, 50.0])
    T = build_relative_transformation_matrix(ref)

    assert T.shape == (7, 7)
    assert T.dtype == float
    # first five diagonal entries should match the reference
    assert np.allclose(np.diag(T)[:5], ref)
    # angle rows should be identity (1.0 on diagonal)
    assert T[5, 5] == 1.0 and T[6, 6] == 1.0


def test_relative_class_uses_radians_and_relative_builders() -> None:
    """Test that the RelativeFractionalDegreesParametriser uses the correct transformation matrix built from the radians and relative builders."""
    ref = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    expected = DEGREES_TO_RADIANS_TRANSFORMATION @ build_relative_transformation_matrix(
        ref
    )

    p = RelativeFractionalDegreesParametriser(reference_model=ref)
    assert np.allclose(p.transformation, expected)


def test_relative_invalid_reference_length_raises() -> None:
    """Constructing with a reference model of wrong length should raise ValueError."""
    # too short
    with pytest.raises(ValueError):
        RelativeFractionalDegreesParametriser(
            reference_model=np.array([1.0, 2.0, 3.0, 4.0])
        )

    # too long
    with pytest.raises(ValueError):
        RelativeFractionalDegreesParametriser(
            reference_model=np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0])
        )


def test_relative_none_uses_one_reference() -> None:
    """Passing None as reference_model should use an all-ones reference and build the composed transformation accordingly."""
    p = RelativeFractionalDegreesParametriser(reference_model=None)
    expected = DEGREES_TO_RADIANS_TRANSFORMATION @ build_relative_transformation_matrix(
        np.ones(5)
    )
    assert np.allclose(p.transformation, expected)


def test_reference_model_property_none_returns_ones() -> None:
    """When constructed with `reference_model=None`, the `reference_model` property should be an all-ones vector of length 5."""
    p = RelativeFractionalDegreesParametriser(reference_model=None)
    assert np.allclose(p.reference_model, np.ones(5))
