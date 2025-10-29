"""Elastic tensor functions for TTI."""

import numpy as np

# Voigt mapping: (i,j) -> I
# (0,0)->0, (1,1)->1, (2,2)->2, (1,2)->3, (0,2)->4, (0,1)->5
VOIGT_MAP = {
    0: [0, 0],
    1: [1, 1],
    2: [2, 2],
    3: [1, 2],
    4: [0, 2],
    5: [0, 1],
}

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

_VMAP = np.array(
    [
        [0, 0],
        [1, 1],
        [2, 2],
        [1, 2],
        [0, 2],
        [0, 1],
    ]
)


def _check_minor_symmetry(C: np.ndarray) -> bool:
    """Check minor symmetries of a rank-4 tensor.

    C_ijkl = C_jikl and C_ijkl = C_ijlk

    => C_ijkl = C_jikl = C_ijlk = C_jilk

    Parameters
    ----------
    C : ndarray, shape (3, 3, 3, 3)
        Fourth order tensor to check

    Returns
    -------
    bool
        True if the tensor has the required symmetries, False otherwise.
    """
    ijkl_eq_jikl = np.array_equal(C, np.swapaxes(C, 0, 1))
    ijkl_eq_ijlk = np.array_equal(C, np.swapaxes(C, 2, 3))
    return ijkl_eq_jikl and ijkl_eq_ijlk


def _check_major_symmetry(C: np.ndarray) -> bool:
    """Check major symmetry of a rank-4 tensor.

    C_ijkl = C_klij

    Parameters
    ----------
    C : ndarray, shape (3, 3, 3, 3)
        Fourth order tensor to check

    Returns
    -------
    bool
        True if the tensor has the required symmetry, False otherwise.
    """
    return np.array_equal(C, np.transpose(C, (2, 3, 0, 1)))


def _check_elastic_tensor_symmetry(C: np.ndarray) -> bool:
    """Check if a rank-4 tensor has both major and minor symmetries.

    C_ijkl = C_jikl = C_ijlk = C_jilk = C_klij = C_lkij = C_klji = C_lkji

    Parameters
    ----------
    C : ndarray, shape (3, 3, 3, 3)
        Fourth order tensor to check

    Returns
    -------
    bool
        True if the tensor has both major and minor symmetries, False otherwise.
    """
    return _check_minor_symmetry(C) and _check_major_symmetry(C)


def elastic_tensor_to_voigt_loop(C: np.ndarray) -> np.ndarray:
    """
    Convert a 4th order elastic tensor (3x3x3x3) to Voigt notation (6x6).

    There is no imposition of symmetries in this implementation.
    As a result, the input tensor is expected to already have the major and minor symmetries of the elastic tensor.

    C_ijkl = C_jikl = C_ijlk = C_jilk = C_klij = C_lkij = C_klji = C_lkji

    The value of C_mn will be the last value assigned from the corresponding C_ijkl components.

    Parameters
    ----------
    C : ndarray, shape (3, 3, 3, 3)
        Fourth order elastic tensor

    Returns
    -------
    C_voigt : ndarray, shape (6, 6)
        Elastic tensor in Voigt notation
    """
    if not _check_elastic_tensor_symmetry(C):
        raise ValueError("Input elastic tensor does not have the required symmetries.")

    C_voigt = np.zeros((6, 6))

    for i in range(3):
        for j in range(3):
            m = VOIGT_MAP_INV[(i, j)]
            for k in range(3):
                for l in range(3):  # noqa: E741
                    n = VOIGT_MAP_INV[(k, l)]
                    C_voigt[m, n] = C[i, j, k, l]
    return C_voigt


def elastic_tensor_to_voigt_vec(C: np.ndarray) -> np.ndarray:
    """
    Convert a fully symmetric 4th-order elastic tensor (C_ijkl) to 6x6 Voigt notation (C_voigt).

    Assumes:
      - C has both minor and major symmetries:
        C_ijkl = C_jikl = C_ijlk = C_jilk = C_klij = C_lkij = C_klji = C_lkji
      - Voigt order: [11, 22, 33, 23, 13, 12]

    Parameters
    ----------
    C : ndarray, shape (3, 3, 3, 3)
        Fourth order elastic tensor

    Returns
    -------
    C_voigt : ndarray, shape (6, 6)
        Elastic tensor in Voigt notation
    """
    C_voigt = np.zeros((6, 6), dtype=C.dtype)

    # vectorised outer products of index pairs
    ij = _VMAP[:, None, :]  # shape (6,1,2)
    kl = _VMAP[None, :, :]  # shape (1,6,2)

    # gather C[i,j,k,l] for all Voigt combinations
    C_voigt[:, :] = C[ij[..., 0], ij[..., 1], kl[..., 0], kl[..., 1]]

    return C_voigt


elastic_tensor_to_voigt = elastic_tensor_to_voigt_vec


def voigt_to_elastic_tensor(C_voigt: np.ndarray) -> np.ndarray:
    """
    Convert an elastic tensor in Voigt notation (6x6) to a 4th order tensor (3x3x3x3).

    This imposes the major and minor symmetries of the elastic tensor.

    C_ijkl = C_jikl = C_ijlk = C_jilk = C_klij = C_lkij = C_klji = C_lkji

    Parameters
    ----------
    C_voigt : ndarray, shape (6, 6)
        Elastic tensor in Voigt notation

    Returns
    -------
    C : ndarray, shape (3, 3, 3, 3)
        Fourth order elastic tensor
    """

    C = np.zeros((3, 3, 3, 3))

    for m in range(6):
        i, j = VOIGT_MAP[m]
        for n in range(6):
            k, l = VOIGT_MAP[n]  # noqa: E741
            C[i, j, k, l] = C_voigt[m, n]
            C[j, i, k, l] = C_voigt[m, n]
            C[i, j, l, k] = C_voigt[m, n]
            C[j, i, l, k] = C_voigt[m, n]
            C[k, l, i, j] = C_voigt[m, n]
            C[l, k, i, j] = C_voigt[m, n]
            C[k, l, j, i] = C_voigt[m, n]
            C[l, k, j, i] = C_voigt[m, n]

    return C


def transverse_isotropic_tensor(
    A: float, C: float, L: float, N: float, F: float
) -> np.ndarray:
    """
    Construct a transverse isotropic elastic tensor in Voigt notation.

    C = [A, A-2N, F, 0, 0, 0],
        [A-2N, A, F, 0, 0, 0],
        [F, F, C, 0, 0, 0],
        [0, 0, 0, L, 0, 0],
        [0, 0, 0, 0, L, 0],
        [0, 0, 0, 0, 0, N]

    Parameters
    ----------
    A : float
        Elastic constant C11 = C22
    C : float
        Elastic constant C33
    L : float
        Elastic constant C44 = C55
    N : float
        Elastic constant C66
    F : float
        Elastic constant C13 = C23

    Returns
    -------
    C_voigt : ndarray, shape (6, 6)
        Transverse isotropic elastic tensor in Voigt notation
    """

    C_voigt = np.zeros((6, 6))

    C_voigt[0, 0] = A
    C_voigt[1, 1] = A
    C_voigt[2, 2] = C
    C_voigt[3, 3] = L
    C_voigt[4, 4] = L
    C_voigt[5, 5] = N

    C_voigt[0, 1] = A - 2 * N
    C_voigt[1, 0] = A - 2 * N

    C_voigt[0, 2] = F
    C_voigt[2, 0] = F

    C_voigt[1, 2] = F
    C_voigt[2, 1] = F

    return C_voigt


def isotropic_tensor(lam: float, mu: float) -> np.ndarray:
    """
    Construct an isotropic elastic tensor in Voigt notation.

    C = [lam + 2 mu, lam, lam, 0, 0, 0],
        [lam, lam + 2 mu, lam, 0, 0, 0],
        [lam, lam, lam + 2 mu, 0, 0, 0],
        [0, 0, 0, mu, 0, 0],
        [0, 0, 0, 0, mu, 0],
        [0, 0, 0, 0, 0, mu]

    Parameters
    ----------
    lam : float
        Lamé constant
    mu : float
        Shear modulus

    Returns
    -------
    C_voigt : ndarray, shape (6, 6)
        Isotropic elastic tensor in Voigt notation
    """

    C_voigt = np.zeros((6, 6))

    C_voigt[0, 0] = lam + 2 * mu
    C_voigt[1, 1] = lam + 2 * mu
    C_voigt[2, 2] = lam + 2 * mu
    C_voigt[3, 3] = mu
    C_voigt[4, 4] = mu
    C_voigt[5, 5] = mu

    C_voigt[0, 1] = lam
    C_voigt[1, 0] = lam

    C_voigt[0, 2] = lam
    C_voigt[2, 0] = lam

    C_voigt[1, 2] = lam
    C_voigt[2, 1] = lam

    return C_voigt
