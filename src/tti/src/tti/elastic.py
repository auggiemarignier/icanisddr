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

type = BoolArray = np.typing.NDArray[np.bool_]


def _check_minor_symmetry(C: np.ndarray) -> BoolArray:
    """Check minor symmetries of a rank-4 tensor.

    C_ijkl = C_jikl and C_ijkl = C_ijlk

    => C_ijkl = C_jikl = C_ijlk = C_jilk

    Parameters
    ----------
    C : ndarray, shape (..., 3, 3, 3, 3)
        Fourth order tensor to check

    Returns
    -------
    BoolArray, shape (...,)
        True where each tensor in the batch and cell dimensions satisfies the minor symmetries.
    """
    ijkl_eq_jikl = np.isclose(C, np.swapaxes(C, -4, -3)).all(axis=(-4, -3, -2, -1))
    ijkl_eq_ijlk = np.isclose(C, np.swapaxes(C, -2, -1)).all(axis=(-4, -3, -2, -1))
    return ijkl_eq_jikl & ijkl_eq_ijlk


def _check_major_symmetry(C: np.ndarray) -> BoolArray:
    """Check major symmetry of a rank-4 tensor.

    C_ijkl = C_klij

    Parameters
    ----------
    C : ndarray, shape (..., 3, 3, 3, 3)
        Fourth order tensor to check

    Returns
    -------
    BoolArray, shape (...,)
        True where each tensor in the batch and cell dimensions satisfies the major symmetry.
    """
    return np.isclose(C, np.swapaxes(np.swapaxes(C, -4, -2), -3, -1)).all(
        axis=(-4, -3, -2, -1),
    )


def _check_elastic_tensor_symmetry(C: np.ndarray) -> BoolArray:
    """Check if a rank-4 tensor has both major and minor symmetries.

    C_ijkl = C_jikl = C_ijlk = C_jilk = C_klij = C_lkij = C_klji = C_lkji

    Parameters
    ----------
    C : ndarray, shape (..., 3, 3, 3, 3)
        Fourth order tensor to check

    Returns
    -------
    BoolArray, shape (...,)
        True where each tensor in the batch and cell dimensions satisfies both major and minor symmetries.
    """
    return _check_minor_symmetry(C) & _check_major_symmetry(C)


def elastic_tensor_to_voigt(C: np.ndarray) -> np.ndarray:
    """
    Convert a fully symmetric 4th-order elastic tensor (C_ijkl) to 6x6 Voigt notation (C_voigt).

    Assumes:
      - C has both minor and major symmetries:
        C_ijkl = C_jikl = C_ijlk = C_jilk = C_klij = C_lkij = C_klji = C_lkji
      - Voigt order: [11, 22, 33, 23, 13, 12]

    Parameters
    ----------
    C : ndarray, shape (..., 3, 3, 3, 3)
        Fourth order elastic tensor

    Returns
    -------
    C_voigt : ndarray, shape (..., 6, 6)
        Elastic tensor in Voigt notation
    """
    leading_shape = C.shape[:-4]
    C_voigt = np.zeros((*leading_shape, 6, 6), dtype=C.dtype)

    # vectorised outer products of index pairs
    ij = _VMAP[:, None, :]  # shape (6,1,2)
    kl = _VMAP[None, :, :]  # shape (1,6,2)

    # gather C[i,j,k,l] for all Voigt combinations
    C_voigt[..., :, :] = C[..., ij[..., 0], ij[..., 1], kl[..., 0], kl[..., 1]]

    return C_voigt


def voigt_to_elastic_tensor(C_voigt: np.ndarray) -> np.ndarray:
    """
    Convert an elastic tensor in Voigt notation (nx6x6) to a 4th order tensor (nx3x3x3x3).

    This imposes the major and minor symmetries of the elastic tensor.

    C_ijkl = C_jikl = C_ijlk = C_jilk = C_klij = C_lkij = C_klji = C_lkji

    Parameters
    ----------
    C_voigt : ndarray, shape (..., 6, 6)
        Elastic tensor in Voigt notation

    Returns
    -------
    C : ndarray, shape (..., 3, 3, 3, 3)
        Fourth order elastic tensor
    """

    C = np.zeros((*C_voigt.shape[:-2], 3, 3, 3, 3))
    for m in range(6):
        i, j = VOIGT_MAP[m]
        for n in range(6):
            k, l = VOIGT_MAP[n]
            C[..., i, j, k, l] = C_voigt[..., m, n]
            C[..., j, i, k, l] = C_voigt[..., m, n]
            C[..., i, j, l, k] = C_voigt[..., m, n]
            C[..., j, i, l, k] = C_voigt[..., m, n]
            C[..., k, l, i, j] = C_voigt[..., m, n]
            C[..., l, k, i, j] = C_voigt[..., m, n]
            C[..., k, l, j, i] = C_voigt[..., m, n]
            C[..., l, k, j, i] = C_voigt[..., m, n]

    return C


def transformation_to_voigt(T: np.ndarray) -> np.ndarray:
    """
    Convert a 4th order transformation tensor to Voigt notation.

    Enforces the minor symmetries of the transformation tensor expected by Voigt notation.

    Parameters
    ----------
    T : ndarray, shape (..., 3, 3, 3, 3)
        Fourth order transformation tensor

    Returns
    -------
    T_voigt : ndarray, shape (..., 6, 6)
        Transformation tensor in Voigt notation
    """

    # Build indexing arrays: ij is (6,1,2), kl is (1,6,2)
    ij = _VMAP[:, None, :]  # shape (6, 1, 2)
    kl = _VMAP[None, :, :]  # shape (1, 6, 2)

    # Extract i, j, k, l indices with broadcasting
    i = ij[..., 0]  # shape (6, 1)
    j = ij[..., 1]  # shape (6, 1)
    k = kl[..., 0]  # shape (1, 6)
    l = kl[..., 1]  # shape (1, 6)

    # Base contribution: T[i, j, k, l]
    T_voigt = T[..., i, j, k, l]

    # Add symmetric contribution T[i, j, l, k] only where k != l
    mask = k != l
    T_voigt = np.where(mask, T_voigt + T[..., i, j, l, k], T_voigt)
    return T_voigt


def transverse_isotropic_tensor_voigt(
    A: np.ndarray, C: np.ndarray, F: np.ndarray, L: np.ndarray, N: np.ndarray
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
    C_voigt : ndarray, shape (n, 6, 6)
        Batched transverse isotropic elastic tensor in Voigt notation. For
        scalar inputs n == 1 and the returned array has shape (1, 6, 6).
    """
    # Broadcast inputs and flatten to a batch dimension
    A_b, C_b, F_b, L_b, N_b = np.broadcast_arrays(A, C, F, L, N)
    leading_shape = A_b.shape

    # Compute A-2N per batch
    A_2N_b = A_b - 2 * N_b

    # Build batched Voigt matrices
    C_voigt = np.zeros((*leading_shape, 6, 6), dtype=float)

    # Top-left 3x3 normal block
    C_voigt[..., 0, 0] = A_b
    C_voigt[..., 0, 1] = A_2N_b
    C_voigt[..., 0, 2] = F_b

    C_voigt[..., 1, 0] = A_2N_b
    C_voigt[..., 1, 1] = A_b
    C_voigt[..., 1, 2] = F_b

    C_voigt[..., 2, 0] = F_b
    C_voigt[..., 2, 1] = F_b
    C_voigt[..., 2, 2] = C_b

    # Shear and remaining diagonal terms
    C_voigt[..., 3, 3] = L_b
    C_voigt[..., 4, 4] = L_b
    C_voigt[..., 5, 5] = N_b

    return C_voigt


def isotropic_tensor_voigt(lam: np.ndarray, mu: np.ndarray) -> np.ndarray:
    """
    Construct an isotropic elastic tensor in Voigt notation.

    C = [[lam + 2 mu, lam, lam, 0, 0, 0],
        [lam, lam + 2 mu, lam, 0, 0, 0],
        [lam, lam, lam + 2 mu, 0, 0, 0],
        [0, 0, 0, mu, 0, 0],
        [0, 0, 0, 0, mu, 0],
        [0, 0, 0, 0, 0, mu]]

    Parameters
    ----------
    lam : np.ndarray (...,)
        Lamé constant
    mu : np.ndarray (...,)
        Shear modulus

    Returns
    -------
    C_voigt : ndarray, shape (..., 6, 6)
        Isotropic elastic tensor in Voigt notation
    """

    # Broadcast inputs to a common shape and flatten to a batch dimension.
    lam_b, mu_b = np.broadcast_arrays(lam, mu)
    lam_2mu = lam_b + 2 * mu_b

    leading_shape = lam_b.shape
    C_voigt = np.zeros((*leading_shape, 6, 6), dtype=float)

    # Fill normal components
    C_voigt[..., 0, 0] = lam_2mu
    C_voigt[..., 0, 1] = lam_b
    C_voigt[..., 0, 2] = lam_b

    C_voigt[..., 1, 0] = lam_b
    C_voigt[..., 1, 1] = lam_2mu
    C_voigt[..., 1, 2] = lam_b

    C_voigt[..., 2, 0] = lam_b
    C_voigt[..., 2, 1] = lam_b
    C_voigt[..., 2, 2] = lam_2mu

    # Shear components
    C_voigt[..., 3, 3] = mu_b
    C_voigt[..., 4, 4] = mu_b
    C_voigt[..., 5, 5] = mu_b

    return C_voigt


def transverse_isotropic_tensor_4th(
    A: np.ndarray, C: np.ndarray, F: np.ndarray, L: np.ndarray, N: np.ndarray
) -> np.ndarray:
    """
    Construct a transverse isotropic elastic tensor directly as a 4th-order tensor.

    Assumes the symmetry axis is along the z-axis.

    Parameters
    ----------
    A : np.ndarray (...,)
        Elastic constant C11 = C22
    C : np.ndarray (...,)
        Elastic constant C33
    F : np.ndarray (...,)
        Elastic constant C13 = C23
    L : np.ndarray (...,)
        Elastic constant C44 = C55
    N : np.ndarray (...,)
        Elastic constant C66

    Returns
    -------
    C : ndarray, shape (..., 3, 3, 3, 3)
        Transverse isotropic elastic tensor (fully symmetric) in index form
        for each broadcast/batch of the input parameters.
    """

    # Broadcast to a common shape then flatten to (n,)
    A_b, C_b, F_b, L_b, N_b = np.broadcast_arrays(A, C, F, L, N)
    leading_shape = A_b.shape

    C_tensor = np.zeros((*leading_shape, 3, 3, 3, 3), dtype=float)

    # Normal components
    C_tensor[..., 0, 0, 0, 0] = A_b
    C_tensor[..., 1, 1, 1, 1] = A_b
    C_tensor[..., 2, 2, 2, 2] = C_b

    # Cross normal terms implied by Voigt: C12 = A-2N, C13 = C23 = F
    A_2N = A_b - 2 * N_b

    # C1122 and symmetric permutations
    C_tensor[..., 0, 0, 1, 1] = A_2N
    C_tensor[..., 1, 1, 0, 0] = A_2N

    # Coupling with symmetry axis (F): C1133 = C2233 = F and symmetries
    C_tensor[..., 0, 0, 2, 2] = F_b
    C_tensor[..., 2, 2, 0, 0] = F_b
    C_tensor[..., 1, 1, 2, 2] = F_b
    C_tensor[..., 2, 2, 1, 1] = F_b

    # Shear components (minor symmetries enforced explicitly)
    # yz shear: C2323 = L and permutations
    C_tensor[..., 1, 2, 1, 2] = L_b
    C_tensor[..., 1, 2, 2, 1] = L_b
    C_tensor[..., 2, 1, 1, 2] = L_b
    C_tensor[..., 2, 1, 2, 1] = L_b

    # xz shear: C1313 = L and permutations
    C_tensor[..., 0, 2, 0, 2] = L_b
    C_tensor[..., 0, 2, 2, 0] = L_b
    C_tensor[..., 2, 0, 0, 2] = L_b
    C_tensor[..., 2, 0, 2, 0] = L_b

    # xy shear: C1212 = N and permutations
    C_tensor[..., 0, 1, 0, 1] = N_b
    C_tensor[..., 0, 1, 1, 0] = N_b
    C_tensor[..., 1, 0, 0, 1] = N_b
    C_tensor[..., 1, 0, 1, 0] = N_b
    return C_tensor


def isotropic_tensor_4th(lam: np.ndarray, mu: np.ndarray) -> np.ndarray:
    """
    Construct an isotropic elastic tensor directly as a 4th-order tensor.

    Uses the compact expression: C_ijkl = lam δ_ij δ_kl + mu (δ_ik δ_jl + δ_il δ_jk)

    Parameters
    ----------
    lam : np.ndarray (...,)
        Lamé constant (λ)
    mu : np.ndarray (...,)
        Shear modulus (μ)

    Returns
    -------
    C : ndarray, shape (..., 3, 3, 3, 3)
        Isotropic elastic tensor in full index notation.
    """
    leading_shape = np.broadcast(lam, mu).shape
    delta = np.tile(np.eye(3), (*leading_shape, 1, 1))  # shape (..., 3, 3)
    C = lam[..., None, None, None, None] * np.einsum(
        "...ij,...kl->...ijkl", delta, delta
    ) + mu[..., None, None, None, None] * (
        np.einsum("...ik,...jl->...ijkl", delta, delta)
        + np.einsum("...il,...jk->...ijkl", delta, delta)
    )
    return C


def transverse_isotropic_tensor(
    A: np.ndarray, C: np.ndarray, F: np.ndarray, L: np.ndarray, N: np.ndarray
) -> np.ndarray:
    """
    Construct a transverse isotropic elastic tensor (4th-order representation).

    This returns the tensor as a 3x3x3x3 array, suitable for direct tensor operations
    like rotations and contractions. For 6x6 Voigt notation, use
    transverse_isotropic_tensor_voigt() instead.

    Parameters
    ----------
    A : np.ndarray (...,)
        Elastic constant C11 = C22
    C : np.ndarray (...,)
        Elastic constant C33
    F : np.ndarray (...,)
        Elastic constant C13 = C23
    L : np.ndarray (...,)
        Elastic constant C44 = C55
    N : np.ndarray (...,)
        Elastic constant C66

    Returns
    -------
    C : ndarray, shape (..., 3, 3, 3, 3)
        Transverse isotropic elastic tensor in 4th-order form.

    See Also
    --------
    transverse_isotropic_tensor_voigt : Voigt notation (6x6) version
    transverse_isotropic_tensor_4th : Direct alias to this implementation
    """
    return transverse_isotropic_tensor_4th(A, C, F, L, N)


def isotropic_tensor(lam: np.ndarray, mu: np.ndarray) -> np.ndarray:
    """
    Construct an isotropic elastic tensor (4th-order representation).

    This returns the tensor as a (n, 3, 3, 3, 3) array, suitable for direct tensor operations
    like rotations and contractions. For 6x6 Voigt notation, use
    isotropic_tensor_voigt() instead.

    Parameters
    ----------
    lam : np.ndarray (n,)
        Lamé constant (λ)
    mu : np.ndarray (n,)
        Shear modulus (μ)

    Returns
    -------
    C : ndarray, shape (n, 3, 3, 3, 3)
        Isotropic elastic tensor in 4th-order form.

    See Also
    --------
    isotropic_tensor_voigt : Voigt notation (6x6) version
    isotropic_tensor_4th : Direct alias to this implementation
    """
    return isotropic_tensor_4th(lam, mu)
