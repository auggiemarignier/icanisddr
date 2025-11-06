"""Tests for the creager module."""

import numpy as np

from tti.creager import calculate_traveltime, love_to_creager


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


def test_love_to_creager_isotropic() -> None:
    """Test conversion from Love to Creager parameters for isotropic case.

    In the isotropic case, A = C = 2L + F, L = N, and we expect:
        a = A = the P-wave modulus
        b = c = 0 since these correspond to anisotropic terms.
    """

    A = C = 10.0
    L = N = 3.0
    F = A - 2 * L

    a, b, c = love_to_creager(A, C, F, L, N)

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

    a, b, c = love_to_creager(A, C, F, L, N)
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

    a, b, c = love_to_creager(A, C, F, L, N)
    dt = calculate_traveltime(np.pi / 2, a, b, c)

    expected = a

    np.testing.assert_allclose(dt, expected)


def test_love_to_creager_known_anisotropic_case() -> None:
    """Test against hand-calculated Creager parameters for known Love parameters.

    Example: A=100, C=150, F=50, L=40

    Hand calculation:
    a = C - (3C - 5A + 4L + 2F)/(8A)
        = 150 - (450 - 500 + 160 + 100)/800
        = 150 - 210/800 = 150 - 0.2625 = 149.7375
    b = (C - A) / (2A) = (150 - 100) / 200 = 0.25
    c = (4L + 2F - A - C) / (8A) = (160 + 100 - 100 - 150) / 800 = 10/800 = 0.0125
    """

    A, C, F, L = 100.0, 150.0, 50.0, 40.0

    a, b, c = love_to_creager(A, C, F, L)

    expected_a = 149.7375
    expected_b = 0.25
    expected_c = 0.0125

    np.testing.assert_allclose(a, expected_a, rtol=1e-12)
    np.testing.assert_allclose(b, expected_b, rtol=1e-12)
    np.testing.assert_allclose(c, expected_c, rtol=1e-12)


def test_love_to_creager_N_irrelevant() -> None:
    """Test that N parameter does not affect the Creager conversion."""
    A = 10.0
    C = 15.0
    F = 7.0
    L = 4.0

    a1, b1, c1 = love_to_creager(A, C, F, L, N=0.0)
    a2, b2, c2 = love_to_creager(A, C, F, L, N=100.0)

    np.testing.assert_allclose(a1, a2)
    np.testing.assert_allclose(b1, b2)
    np.testing.assert_allclose(c1, c2)
