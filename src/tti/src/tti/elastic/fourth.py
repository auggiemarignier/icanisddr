"""Elastic tensors as full fourth-order tensors."""

import numpy as np

from ..rotation import rotation_matrix_zy
from .voigt_mapping import (
    elastic_tensor_to_voigt,
    transformation_to_voigt,
    voigt_to_elastic_tensor,
)

type BoolArray = np.typing.NDArray[np.bool_]


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


def transformation_4th_order(R: np.ndarray) -> np.ndarray:
    """
    Construct a 4th order tensor from a 3D transformation matrix.

    Given a transformation matrix R, a second order tensor A transforms as A' = R A R^T.
    In Einstein notation, this is A'_{ij} = R_{ik} R_{jl} A_{kl}.

    Parameters
    ----------
    R : ndarray, shape (..., 3, 3)
        Transformation matrix.

    Returns
    -------
    R4 : ndarray, shape (..., 3, 3, 3, 3)
        4th order transformation tensor.
    """
    return np.einsum("...ik,...jl->...ijkl", R, R)


def isotropic_tensor(lam: np.ndarray, mu: np.ndarray) -> np.ndarray:
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


def tilted_transverse_isotropic_tensor(
    A: np.ndarray,
    C: np.ndarray,
    F: np.ndarray,
    L: np.ndarray,
    N: np.ndarray,
    eta1: np.ndarray,
    eta2: np.ndarray,
) -> np.ndarray:
    """
    Construct a rotated transverse isotropic elastic tensor.

    Parameters
    ----------
    A : np.ndarray (n,)
        Elastic constant C11 = C22
    C : np.ndarray (n,)
        Elastic constant C33
    F : np.ndarray (n,)
        Elastic constant C13 = C23
    L : np.ndarray (n,)
        Elastic constant C44 = C55
    N : np.ndarray (n,)
        Elastic constant C66
    eta1 : np.ndarray (n,)
        Tilt angle in radians.
    eta2 : np.ndarray (n,)
        Azimuthal angle in radians.

    Returns
    -------
    C_rotated : ndarray, shape (n, 3, 3, 3, 3)
        Rotated transverse isotropic elastic tensor as a 4th-order tensor (not in Voigt notation).
    """
    C_ti = transverse_isotropic_tensor(A, C, F, L, N)

    C_voigt = elastic_tensor_to_voigt(C_ti)

    R = rotation_matrix_zy(eta1, eta2)
    R_voigt = transformation_to_voigt(transformation_4th_order(R))

    C_rotated_voigt = R_voigt @ C_voigt @ R_voigt.swapaxes(-2, -1)

    return voigt_to_elastic_tensor(C_rotated_voigt)
