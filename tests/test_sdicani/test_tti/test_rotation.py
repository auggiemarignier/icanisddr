"""Test the rotation matrices."""

from collections.abc import Callable

import numpy as np
import pytest
from tti.rotation import (
    rotation_matrix_x,
    rotation_matrix_y,
    rotation_matrix_z,
    rotation_matrix_zy,
    transformation_4th_order,
)


@pytest.mark.parametrize("R", [rotation_matrix_z, rotation_matrix_y, rotation_matrix_x])
def test_rotation_matrix_0(R: Callable[[float], np.ndarray]) -> None:
    """Test rotation matrices at 0 radians."""

    expected = np.eye(3)
    np.testing.assert_array_almost_equal(R(0.0), expected)


@pytest.mark.parametrize(
    ["R", "expected"],
    [
        (rotation_matrix_z, np.array([[0, -1, 0], [1, 0, 0], [0, 0, 1]])),
        (rotation_matrix_y, np.array([[0, 0, 1], [0, 1, 0], [-1, 0, 0]])),
        (rotation_matrix_x, np.array([[1, 0, 0], [0, 0, -1], [0, 1, 0]])),
    ],
)
def test_rotation_matrix_90(
    R: Callable[[float], np.ndarray], expected: np.ndarray
) -> None:
    """Test rotation matrices at 90 degrees."""

    angle = np.pi / 2
    R_matrix = R(angle)

    np.testing.assert_array_almost_equal(R_matrix, expected)


@pytest.mark.parametrize("R", [rotation_matrix_z, rotation_matrix_y, rotation_matrix_x])
def test_rotation_matrices_orthogonal(
    R: Callable[[float], np.ndarray], rng: np.random.Generator
) -> None:
    """Test that rotation matrices are orthogonal."""

    angle = rng.uniform(0, 2 * np.pi)
    np.testing.assert_array_almost_equal(R(angle) @ R(angle).T, np.eye(3))


@pytest.mark.parametrize("R", [rotation_matrix_z, rotation_matrix_y, rotation_matrix_x])
def test_rotation_matrices_determinant(
    R: Callable[[float], np.ndarray], rng: np.random.Generator
) -> None:
    """Test that rotation matrices have determinant 1."""

    angle = rng.uniform(0, 2 * np.pi)
    assert np.isclose(np.linalg.det(R(angle)), 1.0)


@pytest.mark.parametrize("R", [rotation_matrix_z, rotation_matrix_y, rotation_matrix_x])
def test_rotation_matrices_composition(
    R: Callable[[float], np.ndarray], rng: np.random.Generator
) -> None:
    """Test that composition of rotation matrices is correct."""

    angle1 = rng.uniform(0, 2 * np.pi)
    angle2 = rng.uniform(0, 2 * np.pi)

    R1 = R(angle1)
    R2 = R(angle2)

    R_combined = R(angle1 + angle2)

    np.testing.assert_array_almost_equal(R1 @ R2, R_combined)


@pytest.mark.parametrize("R", [rotation_matrix_z, rotation_matrix_y, rotation_matrix_x])
def test_rotation_matrices_inverse(
    R: Callable[[float], np.ndarray], rng: np.random.Generator
) -> None:
    """Test that the inverse of rotation matrices is their transpose."""

    angle = rng.uniform(0, 2 * np.pi)
    np.testing.assert_array_almost_equal(R(angle).T, np.linalg.inv(R(angle)))


def test_rotation_matrix_zy(rng: np.random.Generator) -> None:
    """Test combined rotation matrices around z and y axes.

    The combined rotation matrix Rzy should be equivalent to first rotating
    around the z-axis by angle_z and then around the y-axis by angle_y.

    This should match what is given by Brett et al., 2024, Eq 8 (https://www.nature.com/articles/s41561-024-01539-6#Sec6) to ensure consistency in TTI rotation conventions.
    """

    angle_z = rng.uniform(0, 2 * np.pi)  # colatitude
    angle_y = rng.uniform(-2 * np.pi, 2 * np.pi)  # longitude

    Rzy = rotation_matrix_zy(angle_z, angle_y)

    c1 = np.cos(angle_z)
    s1 = np.sin(angle_z)
    c2 = np.cos(angle_y)
    s2 = np.sin(angle_y)

    expected = np.array(  # from Brett et al., 2024, Eq 8
        [
            [c1 * c2, -s1, s2 * c1],
            [s1 * c2, c1, s1 * s2],
            [-s2, 0, c2],
        ]
    )

    np.testing.assert_array_almost_equal(Rzy, expected)


def test_transformation_4th_order(rng: np.random.Generator) -> None:
    """Test the construction of a 4th order transformation tensor from a 3D rotation matrix."""

    angle = rng.uniform(0, 2 * np.pi)
    R = rotation_matrix_z(angle)

    R4 = transformation_4th_order(R)

    # write out the expected result manually
    c = np.cos(angle)
    s = np.sin(angle)

    R4_expected = np.array(
        [
            # i = 0 (row), j = 0,1,2 (columns)
            [
                # j = 0 -> B[0][0][k][l]
                [[c * c, -c * s, 0], [-c * s, s * s, 0], [0, 0, 0]],
                # j = 1 -> B[0][1][k][l]
                [[c * s, c * c, 0], [-s * s, -c * s, 0], [0, 0, 0]],
                # j = 2 -> B[0][2][k][l]
                [[0, 0, c], [0, 0, -s], [0, 0, 0]],
            ],
            # i = 1
            [
                # j = 0
                [[s * c, -s * s, 0], [c * c, -c * s, 0], [0, 0, 0]],
                # j = 1
                [[s * s, s * c, 0], [s * c, c * c, 0], [0, 0, 0]],
                # j = 2
                [[0, 0, s], [0, 0, c], [0, 0, 0]],
            ],
            # i = 2
            [
                # j = 0
                [[0, 0, 0], [0, 0, 0], [c, -s, 0]],
                # j = 1
                [[0, 0, 0], [0, 0, 0], [s, c, 0]],
                # j = 2
                [[0, 0, 0], [0, 0, 0], [0, 0, 1]],
            ],
        ]
    )
    np.testing.assert_array_equal(R4, R4_expected)
