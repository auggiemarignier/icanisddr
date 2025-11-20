# ruff: noqa: E741
# a fair bit of tensor notation is involved here so stop
# ruff from complaining about variable names like l

"""Test the elastic module."""

import numpy as np
import pytest

from sdicani.tti.elastic import (
    VOIGT_MAP,
    _check_elastic_tensor_symmetry,
    _check_major_symmetry,
    _check_minor_symmetry,
    elastic_tensor_to_voigt,
    isotropic_tensor_4th,
    isotropic_tensor_voigt,
    transformation_to_voigt,
    transverse_isotropic_tensor_4th,
    transverse_isotropic_tensor_voigt,
    voigt_to_elastic_tensor,
)

VOIGT_MAP_INV = {
    (0, 0): 0,
    (1, 1): 1,
    (2, 2): 2,
    (1, 2): 3,
    (2, 1): 3,
    (0, 2): 4,
    (2, 0): 4,
    (0, 1): 5,
    (1, 0): 5,
}


def _elastic_tensor_to_voigt_reference(C: np.ndarray) -> np.ndarray:
    """
    Reference loop implementation for validating vectorised elastic_tensor_to_voigt.

    Convert a 4th order elastic tensor (3x3x3x3) to Voigt notation (6x6).

    Parameters
    ----------
    C : ndarray, shape (n, 3, 3, 3, 3)
        Fourth order elastic tensor

    Returns
    -------
    C_voigt : ndarray, shape (n, 6, 6)
        Elastic tensor in Voigt notation
    """

    if not _check_elastic_tensor_symmetry(C):
        raise ValueError("Input elastic tensor does not have the required symmetries.")

    N = C.shape[0]
    C_voigt = np.zeros((N, 6, 6))

    for i in range(3):
        for j in range(3):
            m = VOIGT_MAP_INV[(i, j)]
            for k in range(3):
                for l in range(3):
                    n = VOIGT_MAP_INV[(k, l)]
                    C_voigt[:, m, n] = C[:, i, j, k, l]
    return C_voigt


def test_check_minor_symmetry() -> None:
    """Test the minor symmetry checker."""

    C = np.zeros((2, 3, 3, 3, 3))
    C[:, 0, 1, 2, 0] = 1.0
    C[:, 1, 0, 2, 0] = 1.0  # C_ijkl = C_jikl
    C[:, 0, 1, 0, 2] = 1.0  # C_ijkl = C_ijlk
    C[:, 1, 0, 0, 2] = 1.0  # C_jikl = C_jilk

    assert _check_minor_symmetry(C)

    C[0, 1, 0, 0, 2] = 2.0  # Break the symmetry
    assert not _check_minor_symmetry(C)


def test_check_major_symmetry() -> None:
    """Test the major symmetry checker."""

    C = np.zeros((2, 3, 3, 3, 3))
    C[:, 0, 1, 2, 0] = 1.0
    C[:, 2, 0, 0, 1] = 1.0  # C_ijkl = C_klij

    assert _check_major_symmetry(C)

    C[0, 2, 0, 1, 0] = 2.0  # Break the symmetry
    assert not _check_major_symmetry(C)


@pytest.fixture
def C4() -> np.ndarray:
    """Fixture for a sample 4th order elastic tensor."""
    C = np.zeros((2, 3, 3, 3, 3))
    C[:, 0, 0, 0, 0] = 1.0
    C[:, 1, 1, 1, 1] = 2.0
    C[:, 2, 2, 2, 2] = 3.0

    # A valid elastic tensor should have the symmetries
    # C_ijkl = C_jikl = C_ijlk = C_jilk = C_klij = C_lkij = C_klji = C_lkji
    i = 0
    j = 1  # (i,j) = (0, 1) -> Voigt index 5
    k = 2
    l = 0  # (k,l) = (2,0) -> Voigt index 4
    C[:, i, j, k, l] = 4.0
    C[:, j, i, k, l] = 4.0
    C[:, i, j, l, k] = 4.0
    C[:, j, i, l, k] = 4.0
    C[:, k, l, i, j] = 4.0
    C[:, l, k, i, j] = 4.0
    C[:, k, l, j, i] = 4.0
    C[:, l, k, j, i] = 4.0

    assert _check_minor_symmetry(C)
    assert _check_major_symmetry(C)

    return C


def test_voigt_map() -> None:
    """Test the Voigt mapping dictionaries."""

    for m in range(6):
        i, j = VOIGT_MAP[m]
        m_back = VOIGT_MAP_INV[(i, j)]
        assert m == m_back, f"VOIGT_MAP and VOIGT_MAP_INV are inconsistent for m={m}"


def test_elastic_tensor_to_voigt_correctness(C4: np.ndarray) -> None:
    """Test that the vectorised implementation matches reference loop implementation."""

    C_voigt_fast = elastic_tensor_to_voigt(C4)
    C_voigt_ref = _elastic_tensor_to_voigt_reference(C4)
    np.testing.assert_array_almost_equal(C_voigt_fast, C_voigt_ref)


def test_elastic_tensor_to_voigt(C4: np.ndarray) -> None:
    """Test conversion from elastic tensor to Voigt notation."""

    C_voigt = elastic_tensor_to_voigt(C4)

    expected = np.zeros((2, 6, 6))
    expected[:, 0, 0] = 1.0
    expected[:, 1, 1] = 2.0
    expected[:, 2, 2] = 3.0

    # get Voigt indices for the component that was chosen to test symmetry in C4 fixture
    I = VOIGT_MAP_INV[(0, 1)]  # noqa: E741
    J = VOIGT_MAP_INV[(2, 0)]
    expected[:, I, J] = 4.0
    expected[:, J, I] = 4.0

    np.testing.assert_array_almost_equal(C_voigt, expected)


def test_elastic_to_voigt_and_back(C4: np.ndarray) -> None:
    """Test conversion from elastic tensor to Voigt notation and back."""

    C_voigt = elastic_tensor_to_voigt(C4)
    C_reconstructed = voigt_to_elastic_tensor(C_voigt)

    np.testing.assert_array_almost_equal(C4, C_reconstructed)


def test_isotropic_symmetry(rng: np.random.Generator) -> None:
    """Test that an isotropic elastic tensor has the required symmetries."""

    lambda_ = rng.uniform(1, 10, size=3)
    mu = rng.uniform(1, 10, size=3)

    C_voigt = isotropic_tensor_voigt(lambda_, mu)

    np.testing.assert_array_equal(C_voigt, C_voigt.transpose(0, 2, 1))
    assert len(np.unique(C_voigt)) == 10  # (lambda, lambda+2mu, mu,)x3 0


def test_transverse_isotropic_symmetry(rng: np.random.Generator) -> None:
    """Test that a transverse isotropic elastic tensor has the required symmetries."""

    A = rng.uniform(1, 10, size=3)
    C = rng.uniform(1, 10, size=3)
    F = rng.uniform(1, 10, size=3)
    L = rng.uniform(1, 10, size=3)
    N = rng.uniform(1, 10, size=3)

    C_voigt = transverse_isotropic_tensor_voigt(A, C, F, L, N)
    np.testing.assert_array_equal(C_voigt, C_voigt.transpose(0, 2, 1))
    assert len(np.unique(C_voigt)) == 19  # (A, C, F, L, N, A-2N)x3, 0


def test_isotropic_4th_matches_voigt(rng: np.random.Generator) -> None:
    """Isotropic 4th-order constructor should match Voigt constructor after mapping."""

    lam = rng.uniform(1, 10, size=3)
    mu = rng.uniform(1, 10, size=3)

    C4 = isotropic_tensor_4th(lam, mu)
    C_voigt_from_4th = elastic_tensor_to_voigt(C4)
    C_voigt_direct = isotropic_tensor_voigt(lam, mu)

    np.testing.assert_array_almost_equal(C_voigt_from_4th, C_voigt_direct)


def test_tti_4th_matches_voigt(rng: np.random.Generator) -> None:
    """TTI 4th-order constructor should match Voigt constructor after mapping."""

    A = rng.uniform(1, 10, size=3)
    C = rng.uniform(1, 10, size=3)
    F = rng.uniform(1, 10, size=3)
    L = rng.uniform(1, 10, size=3)
    N = rng.uniform(1, 10, size=3)

    C4 = transverse_isotropic_tensor_4th(A, C, F, L, N)
    C_voigt_from_4th = elastic_tensor_to_voigt(C4)
    C_voigt_direct = transverse_isotropic_tensor_voigt(A, C, F, L, N)

    np.testing.assert_array_almost_equal(C_voigt_from_4th, C_voigt_direct)


def test_transformation_to_voigt(rng: np.random.Generator) -> None:
    """Test that the bond tensor in Voigt notation is symmetric."""

    from sdicani.tti.rotation import rotation_matrix_z, transformation_4th_order

    r = rotation_matrix_z(rng.uniform(0, 2 * np.pi, size=2))
    R = transformation_4th_order(r)
    R_voigt = transformation_to_voigt(R)

    # get the notation the same as in Brett et al., 2024
    r11 = r[:, 0, 0]
    r12 = r[:, 0, 1]
    r13 = r[:, 0, 2]
    r21 = r[:, 1, 0]
    r22 = r[:, 1, 1]
    r23 = r[:, 1, 2]
    r31 = r[:, 2, 0]
    r32 = r[:, 2, 1]
    r33 = r[:, 2, 2]

    expected = np.array(
        [
            [r11**2, r12**2, r13**2, 2 * r12 * r13, 2 * r11 * r13, 2 * r11 * r12],
            [r21**2, r22**2, r23**2, 2 * r22 * r23, 2 * r21 * r23, 2 * r21 * r22],
            [r31**2, r32**2, r33**2, 2 * r32 * r33, 2 * r31 * r33, 2 * r31 * r32],
            [
                r21 * r31,
                r22 * r32,
                r23 * r33,
                r22 * r33 + r23 * r32,
                r21 * r33 + r23 * r31,
                r21 * r32 + r22 * r31,
            ],
            [
                r11 * r31,
                r12 * r32,
                r13 * r33,
                r12 * r33 + r13 * r32,
                r11 * r33 + r13 * r31,
                r11 * r32 + r12 * r31,
            ],
            [
                r11 * r21,
                r12 * r22,
                r13 * r23,
                r12 * r23 + r13 * r22,
                r11 * r23 + r13 * r21,
                r11 * r22 + r12 * r21,
            ],
        ]
    ).transpose(2, 0, 1)  # to get the batch axis first (n, 6, 6)

    np.testing.assert_array_almost_equal(R_voigt, expected)
