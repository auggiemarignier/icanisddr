"""Tests for the creager module."""

from typing import Literal

import numpy as np
import pytest

from tti.elastic.creager import calculate_traveltime, love_to_creager


def test_traveltime_parallel(rng: np.random.Generator) -> None:
    """Test traveltime calculation for ray parallel to symmetry axis (theta=0).

    In this case, dt = a + b + c.
    """
    a = rng.uniform(-1, 1)
    b = rng.uniform(-1, 1)
    c = rng.uniform(-1, 1)

    # ray along symmetry axis
    theta = 0.0

    dt = calculate_traveltime(theta, a, b, c)
    expected_dt = a + b + c
    np.testing.assert_allclose(dt, expected_dt)


def test_traveltime_perpendicular(rng: np.random.Generator) -> None:
    """Test traveltime calculation for ray perpendicular to symmetry axis (theta=pi/2).

    In this case, dt = a.
    """
    a = rng.uniform(-1, 1)
    b = rng.uniform(-1, 1)
    c = rng.uniform(-1, 1)

    # ray perpendicular to symmetry axis
    theta = np.pi / 2

    dt = calculate_traveltime(theta, a, b, c)
    expected_dt = a
    np.testing.assert_allclose(dt, expected_dt)


def test_traveltime_isotropic(rng: np.random.Generator) -> None:
    """Test traveltime calculation for isotropic case (b=c=0).

    In this case, dt = a.
    """
    a = rng.uniform(-1, 1)
    b = 0.0
    c = 0.0

    # test for multiple angles
    for theta in np.linspace(0, np.pi, 5):
        dt = calculate_traveltime(theta, a, b, c)
        expected_dt = a
        np.testing.assert_allclose(dt, expected_dt)


@pytest.mark.parametrize("direction", ["polar", "equatorial"])
def test_love_to_creager_isotropic(direction: Literal["polar", "equatorial"]) -> None:
    """Test conversion from Love to Creager parameters for isotropic case.

    In the isotropic case, A = C = 2L + F, L = N, and we expect regardless of direction that:
        a = A = the P-wave modulus
        b = c = 0 since these correspond to anisotropic terms.
    """

    A = C = 10.0
    L = N = 3.0
    F = A - 2 * L

    a, b, c = love_to_creager(direction, A, C, F, L, N)

    expected_a = A
    expected_b = 0.0
    expected_c = 0.0

    np.testing.assert_allclose(a, expected_a)
    np.testing.assert_allclose(b, expected_b)
    np.testing.assert_allclose(c, expected_c)


def test_love_to_creager_polar() -> None:
    """Test conversion from Love to Creager parameters for polar path.

    To test the conversion, we check the predicted traveltime.
    In the polar case, the symmetry axis is vertical and we expect:
        dt(0) = a + b + c = C
    """

    A = 8.0
    C = 12.0
    F = 7.0
    L = 3.0
    N = 4.0

    a, b, c = love_to_creager("polar", A, C, F, L, N)
    dt = calculate_traveltime(0.0, a, b, c)

    expected = C

    np.testing.assert_allclose(dt, expected)


def test_love_to_creager_equatorial() -> None:
    """Test conversion from Love to Creager parameters for equatorial path.

    To test the conversion, we check the predicted traveltime.
    In the equatorial case, the symmetry axis is vertical and we expect:
        dt(pi/2) = a
    """

    A = 8.0
    C = 12.0
    F = 7.0
    L = 3.0
    N = 4.0

    a, b, c = love_to_creager("equatorial", A, C, F, L, N)
    dt = calculate_traveltime(np.pi / 2, a, b, c)

    expected = a

    np.testing.assert_allclose(dt, expected)


@pytest.mark.parametrize("direction", ["polar", "equatorial"])
def test_love_to_creager_N_irrelevant(
    direction: Literal["polar", "equatorial"],
) -> None:
    """Test that N parameter does not affect the Creager conversion."""
    A = 10.0
    C = 15.0
    F = 7.0
    L = 4.0

    a1, b1, c1 = love_to_creager(direction, A, C, F, L, N=0.0)
    a2, b2, c2 = love_to_creager(direction, A, C, F, L, N=100.0)

    np.testing.assert_allclose(a1, a2)
    np.testing.assert_allclose(b1, b2)
    np.testing.assert_allclose(c1, c2)


@pytest.mark.parametrize("direction", ["polar", "equatorial"])
def test_love_to_creager_vectorised(direction: Literal["polar", "equatorial"]) -> None:
    """Test that love_to_creager handles batched array inputs correctly."""
    # Test with batched array inputs
    A = np.array([8.0, 9.0, 10.0])
    C = np.array([12.0, 13.0, 14.0])
    F = np.array([7.0, 7.5, 8.0])
    L = np.array([3.0, 3.5, 4.0])
    N = np.array([4.0, 4.5, 5.0])

    a, b, c = love_to_creager(direction, A, C, F, L, N)

    # Check that output shapes are correct
    assert a.shape == (3,)
    assert b.shape == (3,)
    assert c.shape == (3,)

    # Verify each element matches scalar computation
    for i in range(3):
        a_i, b_i, c_i = love_to_creager(direction, A[i], C[i], F[i], L[i], N[i])
        np.testing.assert_allclose(a[i], a_i)
        np.testing.assert_allclose(b[i], b_i)
        np.testing.assert_allclose(c[i], c_i)


def test_calculate_traveltime_vectorised() -> None:
    """Test that calculate_traveltime handles broadcasting correctly."""
    # Test broadcasting with different shaped inputs
    theta = np.array([0.0, np.pi / 4, np.pi / 2])  # shape (3,)
    a = np.array([1.0, 2.0])  # shape (2,)
    b = np.array([0.1, 0.2])  # shape (2,)
    c = np.array([0.01, 0.02])  # shape (2,)

    # Reshape for broadcasting: theta (1, 3), a, b, c (2, 1)
    dt = calculate_traveltime(theta[None, :], a[:, None], b[:, None], c[:, None])

    # Check output shape is (2, 3) - 2 parameter sets x 3 angles
    assert dt.shape == (2, 3)

    # Verify each element matches scalar computation
    for i in range(2):
        for j in range(3):
            dt_ij = calculate_traveltime(theta[j], a[i], b[i], c[i])
            np.testing.assert_allclose(dt[i, j], dt_ij)


def test_calculate_traveltime_batched() -> None:
    """Test that calculate_traveltime handles batched inputs correctly."""
    # Test with batched inputs of same shape
    theta = np.array([0.0, np.pi / 4, np.pi / 2])
    a = np.array([1.0, 2.0, 3.0])
    b = np.array([0.1, 0.2, 0.3])
    c = np.array([0.01, 0.02, 0.03])

    dt = calculate_traveltime(theta, a, b, c)

    # Check output shape is (3,)
    assert dt.shape == (3,)

    # Verify each element matches scalar computation
    for i in range(3):
        dt_i = calculate_traveltime(theta[i], a[i], b[i], c[i])
        np.testing.assert_allclose(dt[i], dt_i)
