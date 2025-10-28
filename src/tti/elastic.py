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


def elastic_tensor_to_voigt(C: np.ndarray) -> np.ndarray:
    """
    Convert a 4th order elastic tensor (3x3x3x3) to Voigt notation (6x6).

    Parameters
    ----------
    C : ndarray, shape (3, 3, 3, 3)
        Fourth order elastic tensor

    Returns
    -------
    C_voigt : ndarray, shape (6, 6)
        Elastic tensor in Voigt notation
    """

    C_voigt = np.zeros((6, 6))

    for i in range(3):
        for j in range(3):
            m = VOIGT_MAP_INV[(i, j)]
            for k in range(3):
                for l in range(3):  # noqa: E741
                    n = VOIGT_MAP_INV[(k, l)]
                    C_voigt[m, n] = C[i, j, k, l]
    return C_voigt


def voigt_to_elastic_tensor(C_voigt: np.ndarray) -> np.ndarray:
    """
    Convert an elastic tensor in Voigt notation (6x6) to a 4th order tensor (3x3x3x3).

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
            C[j, l, i, k] = C_voigt[m, n]
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
