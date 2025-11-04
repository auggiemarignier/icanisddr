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
    return 5.0


@pytest.fixture
def N() -> float:
    """Fixture for elastic constant N."""
    return 7.0


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
