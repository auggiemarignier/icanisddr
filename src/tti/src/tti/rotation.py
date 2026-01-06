# ruff: noqa: E741
# a fair bit of tensor notation is involved here so stop
# ruff from complaining about variable names like l

"""Rotation matrices."""

import numpy as np


def rotation_matrix_z(angle: float | np.ndarray) -> np.ndarray:
    """
    Create a 3D rotation matrix for a rotation around the z-axis.

    Parameters
    ----------
    angle : float | ndarray (n,)
        Rotation angle in radians.

    Returns
    -------
    R : ndarray, shape (n, 3, 3)
        Rotation matrix.
    """
    angle = np.atleast_1d(np.asarray(angle))
    c = np.cos(angle)
    s = np.sin(angle)

    R = np.zeros((angle.size, 3, 3), dtype=float)
    R[:, 0, 0] = c
    R[:, 0, 1] = -s
    R[:, 1, 0] = s
    R[:, 1, 1] = c
    R[:, 2, 2] = 1.0
    return R


def rotation_matrix_y(angle: float | np.ndarray) -> np.ndarray:
    """
    Create a 3D rotation matrix for a rotation around the y-axis.

    Parameters
    ----------
    angle : float | ndarray (n,)
        Rotation angle in radians.

    Returns
    -------
    R : ndarray, shape (n, 3, 3)
        Rotation matrix.
    """
    angle = np.atleast_1d(np.asarray(angle))
    c = np.cos(angle)
    s = np.sin(angle)

    R = np.zeros((angle.size, 3, 3), dtype=float)
    R[:, 0, 0] = c
    R[:, 0, 2] = s
    R[:, 1, 1] = 1.0
    R[:, 2, 0] = -s
    R[:, 2, 2] = c

    return R


def rotation_matrix_x(angle: float | np.ndarray) -> np.ndarray:
    """
    Create a 3D rotation matrix for a rotation around the x-axis.

    Parameters
    ----------
    angle : float | ndarray (n,)
        Rotation angle in radians.

    Returns
    -------
    R : ndarray, shape (n, 3, 3)
        Rotation matrix.
    """
    angle = np.asarray(angle)
    c = np.cos(angle)
    s = np.sin(angle)

    R = np.zeros((angle.size, 3, 3), dtype=float)
    R[:, 0, 0] = 1.0
    R[:, 1, 1] = c
    R[:, 1, 2] = -s
    R[:, 2, 1] = s
    R[:, 2, 2] = c
    return R


def rotation_matrix_zy(
    alpha: float | np.ndarray, beta: float | np.ndarray
) -> np.ndarray:
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
