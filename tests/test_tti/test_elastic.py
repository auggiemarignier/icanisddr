# ruff: noqa: E741
# a fair bit of tensor notation is involved here so stop
# ruff from complaining about variable names like l

"""Test the elastic module."""

import numpy as np
import pytest

from tti.elastic import (
    VOIGT_MAP,
    VOIGT_MAP_INV,
    _check_major_symmetry,
    _check_minor_symmetry,
    elastic_tensor_to_voigt,
    elastic_tensor_to_voigt_loop,
    elastic_tensor_to_voigt_vec,
    isotropic_tensor,
    isotropic_tensor_4th,
    transformation_to_voigt,
    transverse_isotropic_tensor,
    transverse_isotropic_tensor_4th,
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
    l = 0  # (k,l) = (2,0) -> Voigt index 4
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


def test_elastic_tensor_to_voigt_loop_vs_vec(C4: np.ndarray) -> None:
    """Test that the vectorised and naive implementations of elastic_tensor_to_voigt agree."""

    C_voigt_vec = elastic_tensor_to_voigt_vec(C4)

    C_voigt_loop = elastic_tensor_to_voigt_loop(C4)
    np.testing.assert_array_almost_equal(C_voigt_vec, C_voigt_loop)


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


def test_isotropic_symmetry(rng: np.random.Generator) -> None:
    """Test that an isotropic elastic tensor has the required symmetries."""

    lambda_ = rng.uniform(1, 10)
    mu = rng.uniform(1, 10)

    C_voigt = isotropic_tensor(lambda_, mu)

    np.testing.assert_array_equal(C_voigt, C_voigt.T)
    assert len(np.unique(C_voigt)) == 4  # lambda, lambda+2mu, mu, 0


def test_transverse_isotropic_symmetry(rng: np.random.Generator) -> None:
    """Test that a transverse isotropic elastic tensor has the required symmetries."""

    A = rng.uniform(1, 10)
    C = rng.uniform(1, 10)
    F = rng.uniform(1, 10)
    L = rng.uniform(1, 10)
    N = rng.uniform(1, 10)

    C_voigt = transverse_isotropic_tensor(A, C, F, L, N)
    np.testing.assert_array_equal(C_voigt, C_voigt.T)
    assert len(np.unique(C_voigt)) == 7  # A, C, F, L, N, A-2N, 0


def test_isotropic_4th_matches_voigt(rng: np.random.Generator) -> None:
    """Isotropic 4th-order constructor should match Voigt constructor after mapping."""

    lam = rng.uniform(1, 10)
    mu = rng.uniform(1, 10)

    C4 = isotropic_tensor_4th(lam, mu)
    C_voigt_from_4th = elastic_tensor_to_voigt(C4)
    C_voigt_direct = isotropic_tensor(lam, mu)

    np.testing.assert_array_almost_equal(C_voigt_from_4th, C_voigt_direct)


def test_tti_4th_matches_voigt(rng: np.random.Generator) -> None:
    """TTI 4th-order constructor should match Voigt constructor after mapping."""

    A = rng.uniform(1, 10)
    C = rng.uniform(1, 10)
    F = rng.uniform(1, 10)
    L = rng.uniform(1, 10)
    N = rng.uniform(1, 10)

    C4 = transverse_isotropic_tensor_4th(A, C, F, L, N)
    C_voigt_from_4th = elastic_tensor_to_voigt(C4)
    C_voigt_direct = transverse_isotropic_tensor(A, C, F, L, N)

    np.testing.assert_array_almost_equal(C_voigt_from_4th, C_voigt_direct)


def test_transformation_to_voigt(rng: np.random.Generator) -> None:
    """Test that the bond tensor in Voigt notation is symmetric."""

    from tti.rotation import rotation_matrix_z, transformation_4th_order

    r = rotation_matrix_z(rng.uniform(0, 2 * np.pi))
    R = transformation_4th_order(r)
    R_voigt = transformation_to_voigt(R)

    # get the notation the same as in Brett et al., 2024
    r11 = r[0, 0]
    r12 = r[0, 1]
    r13 = r[0, 2]
    r21 = r[1, 0]
    r22 = r[1, 1]
    r23 = r[1, 2]
    r31 = r[2, 0]
    r32 = r[2, 1]
    r33 = r[2, 2]

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
    )

    np.testing.assert_array_almost_equal(R_voigt, expected)
