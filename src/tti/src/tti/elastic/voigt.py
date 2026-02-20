"""Elastic tensors as Voigt notation matrices."""

import numpy as np

from ..rotation import rotation_matrix_zy
from .voigt_mapping import matrix_to_voigt


def isotropic_tensor(lam: np.ndarray, mu: np.ndarray) -> np.ndarray:
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


def transverse_isotropic_tensor(
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
    Construct a tilted transverse isotropic elastic tensor in Voigt notation.

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
    eta1 : np.ndarray (...,)
        Tilt angle around the y-axis (in radians)
    eta2 : np.ndarray (...,)
        Azimuthal angle around the z-axis (in radians)

    Returns
    -------
    C_voigt : ndarray, shape (..., 6, 6)
        Tilted transverse isotropic elastic tensor in Voigt notation
    """

    C_voigt = transverse_isotropic_tensor(A, C, F, L, N)

    R = rotation_matrix_zy(eta1, eta2)
    R_voigt = matrix_to_voigt(R)

    C_rotated_voigt = R_voigt @ C_voigt @ R_voigt.swapaxes(-2, -1)

    return C_rotated_voigt


def n_outer_n(n: np.ndarray) -> np.ndarray:
    """Compute the outer product of a vector with itself in Voigt notation.

    Parameters
    ----------
    n : ndarray, shape (..., 3)
        Vector(s) to compute outer product of with themselves.

    Returns
    -------
    n_outer_n : ndarray, shape (..., 6)
        Outer product of n with itself in Voigt notation.
    """
    non = np.zeros((*n.shape[:-1], 6))
    non[..., 0] = n[..., 0] ** 2
    non[..., 1] = n[..., 1] ** 2
    non[..., 2] = n[..., 2] ** 2
    non[..., 3] = 2 * n[..., 1] * n[..., 2]
    non[..., 4] = 2 * n[..., 0] * n[..., 2]
    non[..., 5] = 2 * n[..., 0] * n[..., 1]
    return non
