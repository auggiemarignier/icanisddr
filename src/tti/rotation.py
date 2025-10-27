"""Rotation matrices."""

import numpy as np


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

    R = R_y @ R_z

    return R
