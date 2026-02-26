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
    angle : float | ndarray (...,)
        Rotation angle in radians.

    Returns
    -------
    R : ndarray, shape (..., 3, 3)
        Rotation matrix.
    """
    angle = np.asarray(angle)
    c = np.cos(angle)
    s = np.sin(angle)

    R = np.zeros(angle.shape + (3, 3), dtype=float)
    R[..., 0, 0] = c
    R[..., 0, 1] = -s
    R[..., 1, 0] = s
    R[..., 1, 1] = c
    R[..., 2, 2] = 1.0
    return R


def gradient_rotation_matrix_z(angle: float | np.ndarray) -> np.ndarray:
    """
    Create the gradient of the 3D rotation matrix for a rotation around the z-axis with respect to the rotation angle.

    Parameters
    ----------
    angle : float | ndarray (...,)
        Rotation angle in radians.

    Returns
    -------
    dR_dangle : ndarray, shape (..., 3, 3)
        Gradient of the rotation matrix with respect to the rotation angle.
    """
    angle = np.asarray(angle)
    c = np.cos(angle)
    s = np.sin(angle)

    dR_dangle = np.zeros(angle.shape + (3, 3), dtype=float)
    dR_dangle[..., 0, 0] = -s
    dR_dangle[..., 0, 1] = -c
    dR_dangle[..., 1, 0] = c
    dR_dangle[..., 1, 1] = -s
    return dR_dangle


def rotation_matrix_y(angle: float | np.ndarray) -> np.ndarray:
    """
    Create a 3D rotation matrix for a rotation around the y-axis.

    Parameters
    ----------
    angle : float | ndarray (...,)
        Rotation angle in radians.

    Returns
    -------
    R : ndarray, shape (..., 3, 3)
        Rotation matrix.
    """
    angle = np.asarray(angle)
    c = np.cos(angle)
    s = np.sin(angle)

    R = np.zeros(angle.shape + (3, 3), dtype=float)
    R[..., 0, 0] = c
    R[..., 0, 2] = s
    R[..., 1, 1] = 1.0
    R[..., 2, 0] = -s
    R[..., 2, 2] = c
    return R


def gradient_rotation_matrix_y(angle: float | np.ndarray) -> np.ndarray:
    """
    Create the gradient of the 3D rotation matrix for a rotation around the y-axis with respect to the rotation angle.

    Parameters
    ----------
    angle : float | ndarray (...,)
        Rotation angle in radians.

    Returns
    -------
    dR_dangle : ndarray, shape (..., 3, 3)
        Gradient of the rotation matrix with respect to the rotation angle.
    """
    angle = np.asarray(angle)
    c = np.cos(angle)
    s = np.sin(angle)

    dR_dangle = np.zeros(angle.shape + (3, 3), dtype=float)
    dR_dangle[..., 0, 0] = -s
    dR_dangle[..., 0, 2] = c
    dR_dangle[..., 2, 0] = -c
    dR_dangle[..., 2, 2] = -s
    return dR_dangle


def rotation_matrix_x(angle: float | np.ndarray) -> np.ndarray:
    """
    Create a 3D rotation matrix for a rotation around the x-axis.

    Parameters
    ----------
    angle : float | ndarray (...,)
        Rotation angle in radians.

    Returns
    -------
    R : ndarray, shape (..., 3, 3)
        Rotation matrix.
    """
    angle = np.asarray(angle)
    c = np.cos(angle)
    s = np.sin(angle)

    R = np.zeros(angle.shape + (3, 3), dtype=float)
    R[..., 0, 0] = 1.0
    R[..., 1, 1] = c
    R[..., 1, 2] = -s
    R[..., 2, 1] = s
    R[..., 2, 2] = c
    return R


def gradient_rotation_matrix_x(angle: float | np.ndarray) -> np.ndarray:
    """
    Create the gradient of the 3D rotation matrix for a rotation around the x-axis with respect to the rotation angle.

    Parameters
    ----------
    angle : float | ndarray (...,)
        Rotation angle in radians.

    Returns
    -------
    dR_dangle : ndarray, shape (..., 3, 3)
        Gradient of the rotation matrix with respect to the rotation angle.
    """
    angle = np.asarray(angle)
    c = np.cos(angle)
    s = np.sin(angle)

    dR_dangle = np.zeros(angle.shape + (3, 3), dtype=float)
    dR_dangle[..., 1, 1] = -s
    dR_dangle[..., 1, 2] = -c
    dR_dangle[..., 2, 1] = c
    dR_dangle[..., 2, 2] = -s
    return dR_dangle


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


def gradient_rotation_matrix_zy(
    alpha: float | np.ndarray, beta: float | np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """
    Create the gradients of the 3D rotation matrix for a rotation around the z-axis followed by a rotation around the y-axis with respect to the rotation angles.

    Parameters
    ----------
    alpha : float or ndarray
        Rotation angle(s) around the z-axis in radians.
    beta : float or ndarray
        Rotation angle(s) around the y-axis in radians.

    Returns
    -------
    dR_dalpha : ndarray, shape (..., 3, 3)
        Gradient of the rotation matrix with respect to the rotation angle alpha.
    dR_dbeta : ndarray, shape (..., 3, 3)
        Gradient of the rotation matrix with respect to the rotation angle beta.
    """
    R_z = rotation_matrix_z(alpha)
    R_y = rotation_matrix_y(beta)
    dRz_dalpha = gradient_rotation_matrix_z(alpha)
    dRy_dbeta = gradient_rotation_matrix_y(beta)

    dR_dalpha = dRz_dalpha @ R_y
    dR_dbeta = R_z @ dRy_dbeta

    return dR_dalpha, dR_dbeta
