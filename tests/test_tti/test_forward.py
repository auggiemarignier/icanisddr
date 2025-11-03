"""Test the forward modelling functions for TTI media."""

import numpy as np
import pytest

from tti.elastic import (
    _check_elastic_tensor_symmetry,
    elastic_tensor_to_voigt,
    transformation_to_voigt,
    transverse_isotropic_tensor_voigt,
)
from tti.forward import construct_general_tti_tensor
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
