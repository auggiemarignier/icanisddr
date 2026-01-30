"""Tests for fourth-order elastic tensor utilities."""

import numpy as np
import pytest

from tti.elastic.fourth import (
    _check_elastic_tensor_symmetry,
    _check_major_symmetry,
    _check_minor_symmetry,
    tilted_transverse_isotropic_tensor,
    transformation_4th_order,
)
from tti.rotation import rotation_matrix_z


@pytest.mark.parametrize(
    "C_shape,symmetry_breaker_indices",
    [
        ((3, 3, 3, 3), (1, 0, 0, 2)),
        ((2, 3, 3, 3, 3), (0, 1, 0, 0, 2)),  # breaking symmetry of the first cell
        (
            (2, 4, 3, 3, 3, 3),
            (0, 0, 1, 0, 0, 2),
        ),  # breaking symmetry of the first cell in the first batch
    ],
    ids=["single", "cell", "batch_cell"],
)
def test_check_minor_symmetry(
    C_shape: tuple[int, ...], symmetry_breaker_indices: tuple[int, ...]
) -> None:
    """Test the minor symmetry checker."""

    C = np.zeros(C_shape)
    C[..., 0, 1, 2, 0] = 1.0
    C[..., 1, 0, 2, 0] = 1.0  # C_ijkl = C_jikl
    C[..., 0, 1, 0, 2] = 1.0  # C_ijkl = C_ijlk
    C[..., 1, 0, 0, 2] = 1.0  # C_jikl = C_jilk

    assert _check_minor_symmetry(C).all()

    C[symmetry_breaker_indices] = 2.0  # Break the symmetry
    minor_symmetry = _check_minor_symmetry(C)
    assert not minor_symmetry.all()
    assert minor_symmetry.shape == C_shape[:-4]
    assert np.sum(~minor_symmetry) == 1  # Only one tensor should fail the check

    if minor_symmetry.ndim == 0:
        assert not minor_symmetry.all()
    elif minor_symmetry.ndim == 1:
        assert not minor_symmetry[0]
        assert minor_symmetry[1:].all()
    elif minor_symmetry.ndim == 2:
        assert not minor_symmetry[0, 0]
        assert minor_symmetry[0, 1:].all()
        assert minor_symmetry[1, :].all()


@pytest.mark.parametrize(
    "C_shape,symmetry_breaker_indices",
    [
        ((3, 3, 3, 3), (2, 0, 1, 0)),
        ((2, 3, 3, 3, 3), (0, 2, 0, 1, 0)),  # breaking symmetry of the first cell
        (
            (2, 4, 3, 3, 3, 3),
            (0, 0, 2, 0, 1, 0),
        ),  # breaking symmetry of the first cell in the first batch
    ],
    ids=["single", "cell", "batch_cell"],
)
def test_check_major_symmetry(
    C_shape: tuple[int, ...], symmetry_breaker_indices: tuple[int, ...]
) -> None:
    """Test the major symmetry checker."""

    C = np.zeros(C_shape)
    C[..., 0, 1, 2, 0] = 1.0
    C[..., 2, 0, 0, 1] = 1.0  # C_ijkl = C_klij

    assert _check_major_symmetry(C).all()

    C[symmetry_breaker_indices] = 2.0  # Break the symmetry
    major_symmetry = _check_major_symmetry(C)
    assert not major_symmetry.all()
    assert major_symmetry.shape == C_shape[:-4]
    assert np.sum(~major_symmetry) == 1  # Only one tensor should fail the check

    if major_symmetry.ndim == 0:
        assert not major_symmetry.all()
    elif major_symmetry.ndim == 1:
        assert not major_symmetry[0]
        assert major_symmetry[1:].all()
    elif major_symmetry.ndim == 2:
        assert not major_symmetry[0, 0]
        assert major_symmetry[0, 1:].all()
        assert major_symmetry[1, :].all()


def test_construct_general_tti_tensor_shape(rng: np.random.Generator) -> None:
    """Test that the constructed TTI tensor has the correct shape.

    It should return a 4th-order tensor of shape (3, 3, 3, 3).
    """
    A = rng.uniform(1, 10, size=1)
    C = rng.uniform(1, 10, size=1)
    F = rng.uniform(1, 10, size=1)
    L = rng.uniform(1, 10, size=1)
    N = rng.uniform(1, 10, size=1)
    eta1 = rng.uniform(0, 2 * np.pi, size=1)
    eta2 = rng.uniform(0, 2 * np.pi, size=1)

    C4 = tilted_transverse_isotropic_tensor(A, C, F, L, N, eta1, eta2)

    assert C4.shape == (1, 3, 3, 3, 3)


def test_construct_general_tti_tensor_is_symmetric(rng: np.random.Generator) -> None:
    """Test that the symmetries are preserved under rotation."""
    A = rng.uniform(1, 10, size=3)
    C = rng.uniform(1, 10, size=3)
    F = rng.uniform(1, 10, size=3)
    L = rng.uniform(1, 10, size=3)
    N = rng.uniform(1, 10, size=3)
    eta1 = rng.uniform(0, 2 * np.pi, size=3)
    eta2 = rng.uniform(0, 2 * np.pi, size=3)

    C4 = tilted_transverse_isotropic_tensor(A, C, F, L, N, eta1, eta2)

    assert _check_elastic_tensor_symmetry(C4).all()


@pytest.mark.parametrize(
    "shape",
    [
        (),
        (2,),
        (2, 3),
    ],
    ids=["scalar", "1d", "2d"],
)
def test_transformation_4th_order(
    shape: tuple[int, ...], rng: np.random.Generator
) -> None:
    """Test the construction of a 4th order transformation tensor from a 3D rotation matrix."""

    angles = rng.uniform(0, 2 * np.pi, size=shape)
    R = rotation_matrix_z(angles)

    R4 = transformation_4th_order(R)

    expected_shape = shape + (3, 3, 3, 3)
    assert R4.shape == expected_shape

    c = np.cos(angles)
    s = np.sin(angles)

    # Build expected using broadcasting so it works for scalar and batched shapes
    expected = np.zeros(expected_shape, dtype=float)

    expected[..., 0, 0, 0, 0] = c * c
    expected[..., 0, 0, 0, 1] = -c * s
    expected[..., 0, 0, 1, 0] = -c * s
    expected[..., 0, 0, 1, 1] = s * s

    expected[..., 0, 1, 0, 0] = c * s
    expected[..., 0, 1, 0, 1] = c * c
    expected[..., 0, 1, 1, 0] = -s * s
    expected[..., 0, 1, 1, 1] = -c * s

    expected[..., 0, 2, 0, 2] = c
    expected[..., 0, 2, 1, 2] = -s

    expected[..., 1, 0, 0, 0] = s * c
    expected[..., 1, 0, 0, 1] = -s * s
    expected[..., 1, 0, 1, 0] = c * c
    expected[..., 1, 0, 1, 1] = -c * s

    expected[..., 1, 1, 0, 0] = s * s
    expected[..., 1, 1, 0, 1] = s * c
    expected[..., 1, 1, 1, 0] = s * c
    expected[..., 1, 1, 1, 1] = c * c

    expected[..., 1, 2, 0, 2] = s
    expected[..., 1, 2, 1, 2] = c

    expected[..., 2, 0, 2, 0] = c
    expected[..., 2, 0, 2, 1] = -s

    expected[..., 2, 1, 2, 0] = s
    expected[..., 2, 1, 2, 1] = c

    expected[..., 2, 2, 2, 2] = 1.0

    np.testing.assert_array_almost_equal(R4, expected)
