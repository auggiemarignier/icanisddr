"""Rotation matrices."""

import numpy as np

from .elastic import elastic_tensor_to_voigt, _VMAP


def rotation_matrix_z(angle: float) -> np.ndarray:
    """
    Create a 3D rotation matrix for a rotation around the z-axis.

    Parameters
    ----------
    angle : float
        Rotation angle in radians.

    Returns
    -------
    R : ndarray, shape (3, 3)
        Rotation matrix.
    """
    c = np.cos(angle)
    s = np.sin(angle)

    R = np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]])

    return R


def rotation_matrix_y(angle: float) -> np.ndarray:
    """
    Create a 3D rotation matrix for a rotation around the y-axis.

    Parameters
    ----------
    angle : float
        Rotation angle in radians.

    Returns
    -------
    R : ndarray, shape (3, 3)
        Rotation matrix.
    """
    c = np.cos(angle)
    s = np.sin(angle)

    R = np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]])

    return R


def rotation_matrix_x(angle: float) -> np.ndarray:
    """
    Create a 3D rotation matrix for a rotation around the x-axis.

    Parameters
    ----------
    angle : float
        Rotation angle in radians.

    Returns
    -------
    R : ndarray, shape (3, 3)
        Rotation matrix.
    """
    c = np.cos(angle)
    s = np.sin(angle)

    R = np.array([[1, 0, 0], [0, c, -s], [0, s, c]])
    return R


def rotation_matrix_zy(alpha: float, beta: float) -> np.ndarray:
    """
    Create a 3D rotation matrix for a rotation around the z-axis followed by a rotation around the y-axis.

    This is designed to match the convention used in Brett et al., 2024, Eq 8
    (https://www.nature.com/articles/s41561-024-01539-6#Sec6).

    Parameters
    ----------
    alpha : float
        Rotation angle around the z-axis in radians.
    beta : float
        Rotation angle around the y-axis in radians.

    Returns
    -------
    R : ndarray, shape (3, 3)
        Rotation matrix.
    """
    R_z = rotation_matrix_z(alpha)
    R_y = rotation_matrix_y(beta)

    R = R_z @ R_y

    return R

def transformation_tensor_to_voigt(B: np.ndarray) -> np.ndarray:
    """
    Map a 4th-order transformation tensor B[i,j,k,l] to a 6x6 matrix appropriate for acting on symmetric 2nd-order tensors.

    This function expects B to have (at least) minor symmetry in (k<->l)
    OR to have already been prepared as B_sum = B + B.swapaxes(2,3).
    It does NOT assume major symmetry.
    Voigt order: [11, 22, 33, 23, 13, 12].
    """
    # Ensure input is array and shape is correct
    B = np.ascontiguousarray(B)
    assert B.shape == (3, 3, 3, 3)

    # Build indexing arrays (6,1,2) and (1,6,2) and gather
    ij = _VMAP[:, None, :]   # (6,1,2)
    kl = _VMAP[None, :, :]   # (1,6,2)

    # result shape (6,6) where each entry is B[i,j,k,l] (no extra averaging)
    C6 = B[ij[..., 0], ij[..., 1], kl[..., 0], kl[..., 1]]

    return C6


def bonds_law_einsum(R: np.ndarray) -> np.ndarray:
    """
    Construct the Bond's law rotation tensor using Einstein summation.

    Parameters
    ----------
    R : ndarray, shape (3, 3)
        Rotation matrix.

    Returns
    -------
    A6 : ndarray, shape (6, 6)
        Bond's law rotation tensor.
    """
    A = np.einsum("ik,jl->ijkl", R, R)
    A6 = np.zeros((6, 6), dtype=R.dtype)

    for m, (i, j) in enumerate(_VMAP):
        for n, (k, l) in enumerate(_VMAP):
            if k == l:
                # normal column/row: only one contribution
                A6[m, n] = A[i, j, k, k]
            else:
                # shear column/row: contributions from (k,l) and (l,k)
                A6[m, n] = A[i, j, k, l] + A[i, j, l, k]
    return A6
