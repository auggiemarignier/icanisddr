"""Test the forward modelling functions for TTI media."""

import numpy as np
import pytest

from tti.elastic import (
    _check_elastic_tensor_symmetry,
    elastic_tensor_to_voigt,
    isotropic_tensor_4th,
    transformation_to_voigt,
    transverse_isotropic_tensor_4th,
    transverse_isotropic_tensor_voigt,
)
from tti.forward import calculate_relative_traveltime, construct_general_tti_tensor
from tti.rotation import rotation_matrix_zy, transformation_4th_order


@pytest.fixture
def A() -> float:
    """Fixture for elastic constant A."""
    return 10.0


@pytest.fixture
def C() -> float:
    """Fixture for elastic constant C."""
    return 15.0


@pytest.fixture
def F() -> float:
    """Fixture for elastic constant F."""
    return 8.0


@pytest.fixture
def L() -> float:
    """Fixture for elastic constant L."""
    return 0.15


@pytest.fixture
def N() -> float:
    """Fixture for elastic constant N."""
    return 0.1


@pytest.fixture
def eta1() -> float:
    """Fixture for rotation angle eta1."""
    return np.pi / 4


@pytest.fixture
def eta2() -> float:
    """Fixture for rotation angle eta2."""
    return np.pi / 6


@pytest.fixture
def C4(
    A: float, C: float, F: float, L: float, N: float, eta1: float, eta2: float
) -> np.ndarray:
    """Fixture for a sample rotated TTI elastic tensor."""
    return construct_general_tti_tensor(A, C, F, L, N, eta1, eta2)


@pytest.fixture
def C6(
    A: float, C: float, F: float, L: float, N: float, eta1: float, eta2: float
) -> np.ndarray:
    """Fixture for a sample rotated TTI elastic tensor in Voigt notation."""
    C_voigt = transverse_isotropic_tensor_voigt(A, C, F, L, N)
    rotation_matrix = rotation_matrix_zy(eta1, eta2)
    R = transformation_to_voigt(transformation_4th_order(rotation_matrix))
    return R @ C_voigt @ R.T


def test_construct_general_tti_tensor_shape(C4: np.ndarray) -> None:
    """Test that the constructed TTI tensor has the correct shape.

    It should return a 4th-order tensor of shape (3, 3, 3, 3).
    """

    assert C4.shape == (3, 3, 3, 3)


def test_construct_general_tti_tensor_is_symmetric(C4: np.ndarray) -> None:
    """Test that the symmetries are preserved under rotation."""

    assert _check_elastic_tensor_symmetry(C4)


def test_4th_order_gives_same_as_voigt(C4: np.ndarray, C6: np.ndarray) -> None:
    """Test that the 4th-order and Voigt implementations give the same result."""

    np.testing.assert_allclose(elastic_tensor_to_voigt(C4), C6)


def test_traveltime_zero_for_zero_perturbation() -> None:
    """Zero perturbation tensor gives zero traveltime anomaly."""

    D = np.zeros((3, 3, 3, 3))
    n = np.array([0.0, 0.0, 1.0])

    dt = calculate_relative_traveltime(n, D)

    assert dt == 0.0


def test_traveltime_isotropic_independent_of_direction(
    rng: np.random.Generator,
) -> None:
    """Isotropic perturbation gives same result for all ray directions."""

    lam, mu = 12.0, 5.0
    D = isotropic_tensor_4th(lam, mu)

    # Try several random directions
    directions = rng.normal(size=(10, 3))
    directions = directions / np.linalg.norm(directions, axis=1, keepdims=True)

    results = [calculate_relative_traveltime(n, D) for n in directions]

    # All should be equal for isotropic medium
    np.testing.assert_allclose(results, results[0], rtol=1e-12)


def test_traveltime_linear_in_perturbation(
    A: float, C: float, F: float, L: float, N: float
) -> None:
    """Doubling the perturbation tensor doubles the traveltime anomaly."""

    D = transverse_isotropic_tensor_4th(A, C, F, L, N)
    n = np.array([1.0, 0.0, 0.0])

    dt1 = calculate_relative_traveltime(n, D)
    dt2 = calculate_relative_traveltime(n, 2 * D)

    np.testing.assert_allclose(dt2, 2 * dt1, rtol=1e-12)


def test_traveltime_known_diagonal_tensor() -> None:
    """Test against hand-calculated result for simple diagonal tensor."""

    # D_iijj = δ_ij for i,j in {0,1,2}
    D = np.zeros((3, 3, 3, 3))
    D[0, 0, 0, 0] = 1.0
    D[1, 1, 1, 1] = 2.0
    D[2, 2, 2, 2] = 3.0

    # Only non-zero components are when i=j=k=l D_0000, D_1111, D_2222
    # For n = (nx, ny, nz), result should be nx^4 + 2*ny^4 + 3*nz^4
    n = np.array([0.6, 0.0, 0.8])  # normalised
    expected = 1.0 * (0.6**4) + 3.0 * (0.8**4)

    dt = calculate_relative_traveltime(n, D)

    np.testing.assert_allclose(dt, expected, rtol=1e-12)


def test_traveltime_antiparallel_rays_equal(
    A: float, C: float, F: float, L: float, N: float
) -> None:
    """Ray and its opposite give the same traveltime (even power)."""

    D = transverse_isotropic_tensor_4th(A, C, F, L, N)

    n = np.array([0.6, 0.8, 0.0])
    dt_forward = calculate_relative_traveltime(n, D)
    dt_backward = calculate_relative_traveltime(-n, D)

    np.testing.assert_allclose(dt_forward, dt_backward, rtol=1e-12)


def test_traveltime_shape_validation() -> None:
    """Function raises appropriate error for wrong input shapes."""

    D = np.zeros((3, 3, 3, 3))

    with pytest.raises((ValueError, IndexError)):
        calculate_relative_traveltime(np.array([1.0, 0.0]), D)  # n wrong shape

    with pytest.raises((ValueError, IndexError)):
        calculate_relative_traveltime(
            np.array([1.0, 0.0, 0.0]), np.zeros((3, 3))
        )  # D wrong shape


def calc_dt_creager(theta: float, a: float, b: float, c: float) -> float:
    """Calculate the traveltime anomaly according to Creager 1992 formula.

    delta t / t = a + b cos^2(theta) + c cos^4(theta)

    where a, b, c are functions of the elastic constants, theta is the angle from the symmetry axis (z-axis, ERA). Assumes symmetry axis is vertical.

    a = C11 (equatorial velocity perturbation)
    b = (C33 - C11) / (2C11) = (C - A) / (2A)
    c = (4C44 + 2C13 - C11 - C33) / (8C11) = (4L + 2F - A - C) / (8A)
    """
    return a + b * (np.cos(theta) ** 2) + c * (np.cos(theta) ** 4)


def test_traveltime_equivalence_with_creager_isotropic(
    F: float, L: float, rng: np.random.Generator
) -> None:
    """Test that the traveltime calculation matches Creager 1992 in the isotropic case.

    In the isotropic case,
        b = c = 0
        a = A = C = 2L + F = lambda + 2mu
        F = lambda
        L = N = mu
    The traveltime should be independent of direction and equal to a i.e the P-wave velocity.
    """
    lambda_ = F
    lambda_plus_two_mu = 2 * L + F
    mu = L
    D = construct_general_tti_tensor(
        lambda_plus_two_mu, lambda_plus_two_mu, lambda_, mu, mu, 0.0, 0.0
    )
    a = lambda_plus_two_mu
    b = 0
    c = 0

    theta = rng.uniform(0, 2 * np.pi)
    n = np.array([np.sin(theta), 0.0, np.cos(theta)])

    dt_tti = calculate_relative_traveltime(n, D)
    dt_creager = calc_dt_creager(theta, a, b, c)
    np.testing.assert_allclose(dt_tti, dt_creager, rtol=1e-12)


def test_traveltime_equivalence_with_creager_transverse_isotropic_parallel(
    A: float, C: float, F: float, L: float, N: float
) -> None:
    """Test that the traveltime calculation matches Creager 1992 in the TI case when parallel to symmetry axis.

    The symmetry axis is vertical (eta1 = eta2 = 0).

    If theta = 0 (ray along symmetry axis), then I would expect the traveltime perturbation to be C_33 i.e. the velocity perturbation along the symmetry axis.
    """

    D = construct_general_tti_tensor(A, C, F, L, N, 0.0, 0.0)
    a = C - (3 * C - 5 * A + 4 * L + 2 * F) / (8 * A)
    b = (C - A) / (2 * A)
    c = (4 * L + 2 * F - A - C) / (8 * A)

    theta = 0
    n = np.array([np.sin(theta), 0.0, np.cos(theta)])

    dt_tti = calculate_relative_traveltime(n, D)
    dt_creager = calc_dt_creager(theta, a, b, c)

    # what I expect intuitively
    expected = C
    np.testing.assert_allclose(dt_tti, expected, rtol=1e-12)

    np.testing.assert_allclose(dt_creager, expected, rtol=1e-12)

    np.testing.assert_allclose(dt_tti, dt_creager, rtol=1e-12)


def test_traveltime_equivalence_with_creager_transverse_isotropic_perpendicular(
    A: float, C: float, F: float, L: float, N: float
) -> None:
    """Test that the traveltime calculation matches Creager 1992 in the TI case when perpendicular to symmetry axis.

    The symmetry axis is vertical (eta1 = eta2 = 0).

    If theta = pi/2 (ray perpendicular to symmetry axis i.e. equatorial), then the traveltime should be equal to the velocity in that direction, C_11 = A.
    """

    D = construct_general_tti_tensor(A, C, F, L, N, 0.0, 0.0)
    a = A
    b = (C - A) / (2 * A)
    c = (4 * L + 2 * F - A - C) / (8 * A)

    theta = np.pi / 2
    n = np.array([np.sin(theta), 0.0, np.cos(theta)])

    dt_tti = calculate_relative_traveltime(n, D)
    dt_creager = calc_dt_creager(theta, a, b, c)

    # what I expect intuitively
    np.testing.assert_allclose(dt_creager, A, rtol=1e-12)

    np.testing.assert_allclose(dt_tti, dt_creager, rtol=1e-12)
