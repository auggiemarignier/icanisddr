"""Test the elastic module."""

import numpy as np
import pytest

from tti.elastic import (
    _check_major_symmetry,
    _check_minor_symmetry,
    VOIGT_MAP,
    VOIGT_MAP_INV,
    elastic_tensor_to_voigt,
    voigt_to_elastic_tensor,
)


def test_check_minor_symmetry() -> None:
    """Test the minor symmetry checker."""

    C = np.zeros((3, 3, 3, 3))
    C[0, 1, 2, 0] = 1.0
    C[1, 0, 2, 0] = 1.0  # C_ijkl = C_jikl
    C[0, 1, 0, 2] = 1.0  # C_ijkl = C_ijlk
    C[1, 0, 0, 2] = 1.0  # C_jikl = C_jilk

    assert _check_minor_symmetry(C)

    C[1, 0, 0, 2] = 2.0  # Break the symmetry
    assert not _check_minor_symmetry(C)


def test_check_major_symmetry() -> None:
    """Test the major symmetry checker."""

    C = np.zeros((3, 3, 3, 3))
    C[0, 1, 2, 0] = 1.0
    C[2, 0, 0, 1] = 1.0  # C_ijkl = C_klij

    assert _check_major_symmetry(C)

    C[2, 0, 1, 0] = 2.0  # Break the symmetry
    assert not _check_major_symmetry(C)



@pytest.fixture
def C4() -> np.ndarray:
    """Fixture for a sample 4th order elastic tensor."""
    C = np.zeros((3, 3, 3, 3))
    C[0, 0, 0, 0] = 1.0
    C[1, 1, 1, 1] = 2.0
    C[2, 2, 2, 2] = 3.0

    # A valid elastic tensor should have the symmetries
    # C_ijkl = C_jikl = C_ijlk = C_jilk = C_klij = C_lkij = C_klji = C_lkji
    i = 0
    j = 1  # (i,j) = (0, 1) -> Voigt index 5
    k = 2
    l = 0  # noqa: E741 # (k,l) = (2,0) -> Voigt index 4
    C[i, j, k, l] = 4.0
    C[j, i, k, l] = 4.0
    C[i, j, l, k] = 4.0
    C[j, i, l, k] = 4.0
    C[k, l, i, j] = 4.0
    C[l, k, i, j] = 4.0
    C[k, l, j, i] = 4.0
    C[l, k, j, i] = 4.0

    assert _check_minor_symmetry(C)
    assert _check_major_symmetry(C)

    return C


def test_voigt_map() -> None:
    """Test the Voigt mapping dictionaries."""

    for m in range(6):
        i, j = VOIGT_MAP[m]
        m_back = VOIGT_MAP_INV[(i, j)]
        assert m == m_back, f"VOIGT_MAP and VOIGT_MAP_INV are inconsistent for m={m}"


def test_elastic_tensor_to_voigt(C4: np.ndarray) -> None:
    """Test conversion from elastic tensor to Voigt notation."""

    C_voigt = elastic_tensor_to_voigt(C4)

    expected = np.zeros((6, 6))
    expected[0, 0] = 1.0
    expected[1, 1] = 2.0
    expected[2, 2] = 3.0

    # get Voigt indices for the component that was chosen to test symmetry in C4 fixture
    I = VOIGT_MAP_INV[(0, 1)]  # noqa: E741
    J = VOIGT_MAP_INV[(2, 0)]
    expected[I, J] = 4.0
    expected[J, I] = 4.0

    np.testing.assert_array_almost_equal(C_voigt, expected)


def test_elastic_to_voigt_and_back(C4: np.ndarray) -> None:
    """Test conversion from elastic tensor to Voigt notation and back."""

    C_voigt = elastic_tensor_to_voigt(C4)
    C_reconstructed = voigt_to_elastic_tensor(C_voigt)

    np.testing.assert_array_almost_equal(C4, C_reconstructed)
