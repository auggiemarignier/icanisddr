# ruff: noqa: E741
# a fair bit of tensor notation is involved here so stop
# ruff from complaining about variable names like l

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
    ijkl_eq_jikl = np.allclose(C, np.swapaxes(C, 0, 1))
    ijkl_eq_ijlk = np.allclose(C, np.swapaxes(C, 2, 3))
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
    return np.allclose(C, np.transpose(C, (2, 3, 0, 1)))


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


def elastic_tensor_to_voigt(C: np.ndarray) -> np.ndarray:
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
            k, l = VOIGT_MAP[n]
            C[i, j, k, l] = C_voigt[m, n]
            C[j, i, k, l] = C_voigt[m, n]
            C[i, j, l, k] = C_voigt[m, n]
            C[j, i, l, k] = C_voigt[m, n]
            C[k, l, i, j] = C_voigt[m, n]
            C[l, k, i, j] = C_voigt[m, n]
            C[k, l, j, i] = C_voigt[m, n]
            C[l, k, j, i] = C_voigt[m, n]

    return C


def transformation_to_voigt(T: np.ndarray) -> np.ndarray:
    """
    Convert a 4th order transformation tensor to Voigt notation.

    Enforces the minor symmetries of the transformation tensor expected by Voigt notation.

    Parameters
    ----------
    T : ndarray, shape (3, 3, 3, 3)
        Fourth order transformation tensor

    Returns
    -------
    T_voigt : ndarray, shape (6, 6)
        Transformation tensor in Voigt notation
    """

    T_voigt = np.zeros((6, 6))

    # Build indexing arrays: ij is (6,1,2), kl is (1,6,2)
    ij = _VMAP[:, None, :]  # shape (6, 1, 2)
    kl = _VMAP[None, :, :]  # shape (1, 6, 2)

    # Extract i, j, k, L indices with broadcasting
    i = ij[..., 0]  # shape (6, 1)
    j = ij[..., 1]  # shape (6, 1)
    k = kl[..., 0]  # shape (1, 6)
    l = kl[..., 1]  # shape (1, 6)

    # Base contribution: T[i, j, k, L]
    T_voigt = T[i, j, k, l]

    # Add symmetric contribution T[i, j, L, k] only where k != l
    mask = k != l
    T_voigt = np.where(mask, T_voigt + T[i, j, l, k], T_voigt)

    return T_voigt


def transverse_isotropic_tensor_voigt(
    A: float, C: float, F: float, L: float, N: float
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
    F : float
        Elastic constant C13 = C23
    L : float
        Elastic constant C44 = C55
    N : float
        Elastic constant C66

    Returns
    -------
    C_voigt : ndarray, shape (6, 6)
        Transverse isotropic elastic tensor in Voigt notation
    """
    A_2N = A - 2 * N
    C_voigt = np.array(
        [
            [A, A_2N, F, 0, 0, 0],
            [A_2N, A, F, 0, 0, 0],
            [F, F, C, 0, 0, 0],
            [0, 0, 0, L, 0, 0],
            [0, 0, 0, 0, L, 0],
            [0, 0, 0, 0, 0, N],
        ]
    )

    return C_voigt


def isotropic_tensor_voigt(lam: float, mu: float) -> np.ndarray:
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
    lam_2mu = lam + 2 * mu
    C_voigt = np.array(
        [
            [lam_2mu, lam, lam, 0, 0, 0],
            [lam, lam_2mu, lam, 0, 0, 0],
            [lam, lam, lam_2mu, 0, 0, 0],
            [0, 0, 0, mu, 0, 0],
            [0, 0, 0, 0, mu, 0],
            [0, 0, 0, 0, 0, mu],
        ]
    )

    return C_voigt


def transverse_isotropic_tensor_4th(
    A: float, C: float, F: float, L: float, N: float
) -> np.ndarray:
    """
    Construct a transverse isotropic elastic tensor directly as a 4th-order tensor.

    Assumes the symmetry axis is along the z-axis.

    Parameters
    ----------
    A : float
        Elastic constant C11 = C22
    C : float
        Elastic constant C33
    F : float
        Elastic constant C13 = C23
    L : float
        Elastic constant C44 = C55
    N : float
        Elastic constant C66

    Returns
    -------
    C : ndarray, shape (3, 3, 3, 3)
        Transverse isotropic elastic tensor (fully symmetric) in index form.
    """
    C_tensor = np.zeros((3, 3, 3, 3), dtype=float)

    # Normal components
    C_tensor[0, 0, 0, 0] = A
    C_tensor[1, 1, 1, 1] = A
    C_tensor[2, 2, 2, 2] = C

    # Cross normal terms implied by Voigt: C12 = A-2N, C13 = C23 = F
    A_2N = A - 2 * N

    # C1122 and symmetric permutations
    C_tensor[0, 0, 1, 1] = A_2N
    C_tensor[1, 1, 0, 0] = A_2N
    C_tensor[0, 1, 0, 1] = N
    C_tensor[1, 0, 1, 0] = N
    C_tensor[0, 1, 1, 0] = N
    C_tensor[1, 0, 0, 1] = N

    # Coupling with symmetry axis (F): C1133 = C2233 = F and symmetries
    C_tensor[0, 0, 2, 2] = F
    C_tensor[2, 2, 0, 0] = F
    C_tensor[1, 1, 2, 2] = F
    C_tensor[2, 2, 1, 1] = F

    # Shear components (minor symmetries enforced explicitly)
    # yz shear: C2323 = L and permutations
    C_tensor[1, 2, 1, 2] = L
    C_tensor[1, 2, 2, 1] = L
    C_tensor[2, 1, 1, 2] = L
    C_tensor[2, 1, 2, 1] = L

    # xz shear: C1313 = L and permutations
    C_tensor[0, 2, 0, 2] = L
    C_tensor[0, 2, 2, 0] = L
    C_tensor[2, 0, 0, 2] = L
    C_tensor[2, 0, 2, 0] = L

    # xy shear: C1212 = N and permutations
    C_tensor[0, 1, 0, 1] = N
    C_tensor[0, 1, 1, 0] = N
    C_tensor[1, 0, 0, 1] = N
    C_tensor[1, 0, 1, 0] = N

    # Major symmetry (ij <-> kl) is ensured by the explicit mirrored assignments above.
    return C_tensor


def isotropic_tensor_4th(lam: float, mu: float) -> np.ndarray:
    """
    Construct an isotropic elastic tensor directly as a 4th-order tensor.

    Uses the compact expression: C_ijkl = lam δ_ij δ_kl + mu (δ_ik δ_jl + δ_il δ_jk)

    Parameters
    ----------
    lam : float
        Lamé constant (λ)
    mu : float
        Shear modulus (μ)

    Returns
    -------
    C : ndarray, shape (3, 3, 3, 3)
        Isotropic elastic tensor in full index notation.
    """
    delta = np.eye(3)
    C = lam * np.einsum("ij,kl->ijkl", delta, delta) + mu * (
        np.einsum("ik,jl->ijkl", delta, delta) + np.einsum("il,jk->ijkl", delta, delta)
    )
    return C


def transverse_isotropic_tensor(
    A: float, C: float, F: float, L: float, N: float
) -> np.ndarray:
    """
    Construct a transverse isotropic elastic tensor (4th-order representation).

    This returns the tensor as a 3x3x3x3 array, suitable for direct tensor operations
    like rotations and contractions. For 6x6 Voigt notation, use
    transverse_isotropic_tensor_voigt() instead.

    Parameters
    ----------
    A : float
        Elastic constant C11 = C22
    C : float
        Elastic constant C33
    F : float
        Elastic constant C13 = C23
    L : float
        Elastic constant C44 = C55
    N : float
        Elastic constant C66

    Returns
    -------
    C : ndarray, shape (3, 3, 3, 3)
        Transverse isotropic elastic tensor in 4th-order form.

    See Also
    --------
    transverse_isotropic_tensor_voigt : Voigt notation (6x6) version
    transverse_isotropic_tensor_4th : Direct alias to this implementation
    """
    return transverse_isotropic_tensor_4th(A, C, F, L, N)


def isotropic_tensor(lam: float, mu: float) -> np.ndarray:
    """
    Construct an isotropic elastic tensor (4th-order representation).

    This returns the tensor as a 3x3x3x3 array, suitable for direct tensor operations
    like rotations and contractions. For 6x6 Voigt notation, use
    isotropic_tensor_voigt() instead.

    Parameters
    ----------
    lam : float
        Lamé constant (λ)
    mu : float
        Shear modulus (μ)

    Returns
    -------
    C : ndarray, shape (3, 3, 3, 3)
        Isotropic elastic tensor in 4th-order form.

    See Also
    --------
    isotropic_tensor_voigt : Voigt notation (6x6) version
    isotropic_tensor_4th : Direct alias to this implementation
    """
    return isotropic_tensor_4th(lam, mu)
