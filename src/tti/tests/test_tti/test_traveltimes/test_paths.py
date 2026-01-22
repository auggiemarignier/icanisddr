"""Test path geometry functions for travel time calculations."""

import numpy as np
import pytest

from tti.traveltimes.paths import (
    _spherical_to_cartesian,
    calculate_path_direction_vector,
)


def test__spherical_to_cartesian() -> None:
    """Test conversion from spherical to cartesian coordinates."""

    lon, lat, r = 0.0, 0.0, 1.0
    x, y, z = _spherical_to_cartesian(lon, lat, r)
    np.testing.assert_allclose([x, y, z], [1.0, 0.0, 0.0], atol=1e-12)

    lon, lat, r = 90.0, 0.0, 1.0
    x, y, z = _spherical_to_cartesian(lon, lat, r)
    np.testing.assert_allclose([x, y, z], [0.0, 1.0, 0.0], atol=1e-12)

    lon, lat, r = 0.0, 90.0, 1.0
    x, y, z = _spherical_to_cartesian(lon, lat, r)
    np.testing.assert_allclose([x, y, z], [0.0, 0.0, 1.0], atol=1e-12)


def test__spherical_to_cartesian_batch() -> None:
    """Test conversion from spherical to cartesian coordinates with batch inputs."""

    # Batch of 3 coordinates
    coords_spherical = np.array([[0.0, 0.0, 1.0], [90.0, 0.0, 1.0], [0.0, 90.0, 1.0]])
    coords_cartesian = _spherical_to_cartesian(
        coords_spherical[:, 0], coords_spherical[:, 1], coords_spherical[:, 2]
    )
    expected = np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]])
    np.testing.assert_allclose(coords_cartesian, expected, atol=1e-12)

    # Single coordinate as 1-element arrays
    lon, lat, r = np.array([0.0]), np.array([0.0]), np.array([1.0])
    coords = _spherical_to_cartesian(lon, lat, r)
    expected = np.array([[1.0, 0.0, 0.0]])
    np.testing.assert_allclose(coords, expected, atol=1e-12)


@pytest.mark.parametrize(
    "ic_in, ic_out, expected",
    [
        (
            np.array([0.0, 0.0, 1.0]),  # in at equator prime meridian
            np.array([180.0, 0.0, 1.0]),  # out at the antipode
            np.array([-1.0, 0.0, 0.0]),
        ),
        (
            np.array([90.0, 0.0, 1.0]),  # in at equator 90 degrees east
            np.array([0.0, 0.0, 1.0]),  # out at the prime meridian
            np.array([1.0, -1.0, 0.0]) / np.sqrt(2),
        ),
        (
            np.array([0.0, 90.0, 1.0]),  # in at north pole
            np.array([0.0, -90.0, 1.0]),  # out at south pole
            np.array([0.0, 0.0, -1.0]),
        ),
        (  # off-centre polar path
            np.array([45.0, 45.0, 1.0]),  # in at 45N 45E
            np.array([45.0, -45.0, 1.0]),  # out at antipode
            np.array([0.0, 0.0, -1.0]),
        ),
    ],
)
def test_calculate_path_direction_vector(
    ic_in: np.ndarray, ic_out: np.ndarray, expected: np.ndarray
) -> None:
    """Test calculation of path direction unit vector."""
    n = calculate_path_direction_vector(ic_in, ic_out)
    np.testing.assert_allclose(n, expected, atol=1e-12)


@pytest.mark.parametrize(
    "ic_in_batch, ic_out_batch, expected_batch",
    [
        # Batch with 2 paths: east-west and north-south
        (
            np.array([[0.0, 0.0, 1.0], [90.0, 0.0, 1.0]]),
            np.array([[180.0, 0.0, 1.0], [270.0, 0.0, 1.0]]),
            np.array([[-1.0, 0.0, 0.0], [0.0, -1.0, 0.0]]),
        ),
        # Batch with 3 paths including diagonal
        (
            np.array([[0.0, 0.0, 1.0], [90.0, 0.0, 1.0], [45.0, 0.0, 1.0]]),
            np.array([[180.0, 0.0, 1.0], [270.0, 0.0, 1.0], [225.0, 0.0, 1.0]]),
            np.array(
                [
                    [-1.0, 0.0, 0.0],
                    [0.0, -1.0, 0.0],
                    [-1 / np.sqrt(2), -1 / np.sqrt(2), 0.0],
                ]
            ),
        ),
        # Single path as 1×3 array
        (
            np.array([[0.0, 0.0, 1.0]]),
            np.array([[180.0, 0.0, 1.0]]),
            np.array([[-1.0, 0.0, 0.0]]),
        ),
    ],
)
def test_calculate_path_direction_vector_batch(
    ic_in_batch: np.ndarray, ic_out_batch: np.ndarray, expected_batch: np.ndarray
) -> None:
    """Test calculation of path direction unit vectors for batch inputs."""
    n_batch = calculate_path_direction_vector(ic_in_batch, ic_out_batch)
    assert n_batch.shape == expected_batch.shape
    np.testing.assert_allclose(n_batch, expected_batch, atol=1e-12)
