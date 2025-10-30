# ruff: noqa: E741
# a fair bit of tensor notation is involved here so stop
# ruff from complaining about variable names like l

"""Rotation matrices."""

import numpy as np

from .elastic import _VMAP


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

    # Build indexing arrays: ij is (6,1,2), kl is (1,6,2)
    ij = _VMAP[:, None, :]  # shape (6, 1, 2)
    kl = _VMAP[None, :, :]  # shape (1, 6, 2)

    # Extract i, j, k, L indices with broadcasting
    i = ij[..., 0]  # shape (6, 1)
    j = ij[..., 1]  # shape (6, 1)
    k = kl[..., 0]  # shape (1, 6)
    l = kl[..., 1]  # shape (1, 6)

    # Base contribution: A[i, j, k, L]
    A6 = A[i, j, k, l]

    # Add symmetric contribution A[i, j, L, k] only where k != L
    mask = k != l
    A6 = np.where(mask, A6 + A[i, j, l, k], A6)

    return A6
