"""Functions relating to Creager 1992 traveltime perturbation calculations.

The traveltime perturbation according to Creager 1992 is given by:

    dt(theta) = a + b * cos^2(theta) + c * cos^4(theta)

where theta is the angle between the ray and the symmetry axis, and a, b, c are parameters derived from the elastic constants.

From Brett et al. (2024) the parameters b and c are given in terms of the elastic constants as:
    b = (C_33 - C_11) / (2 * C_11)
    c = (4 * C_44 + 2 * C_13 - C_11 - C_33) / (8 * C_11)

So we have 3 equations and 4 unknowns (dt, a, b, c). We can determine a in terms of the elastic constants for the polar and equatorial paths, as for these paths we know the traveltime perturbation directly from the elastic constants.

For the polar path (theta = 0) the ray is along the symmetry axis (z, 3, ERA) so the traveltime perturbation should be equal to C_33, i.e.
    dt(0) = a + b + c = C_33
Solving for a gives:
    a = C_33 - b - c

For the equatorial path (theta = pi/2) the ray is perpendicular to the symmetry axis (x or y, 1 or 2) so the traveltime perturbation should be equal to C_11, i.e.
    dt(pi/2) = a = C_11
"""

from typing import Literal

import numpy as np


def love_to_creager(
    direction: Literal["polar", "equatorial"],
    A: np.ndarray | float,
    C: np.ndarray | float,
    F: np.ndarray | float,
    L: np.ndarray | float,
    N: None | np.ndarray | float = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Convert from Love parameters to Creager 1992 parameters a, b, c.

    Parameters
    ----------
    direction : Literal['polar', 'equatorial']
    A : np.ndarray | float
        Love parameter A (C_11). Can be scalar or array of shape (n,).
    C : np.ndarray | float
        Love parameter C (C_33). Can be scalar or array of shape (n,).
    F : np.ndarray | float
        Love parameter F (C_13). Can be scalar or array of shape (n,).
    L : np.ndarray | float
        Love parameter L (C_44). Can be scalar or array of shape (n,).
    N : np.ndarray | float, optional
        Love parameter N (C_66) (unused but included for completeness). Can be scalar or array of shape (n,).

    Returns
    -------
    a : np.ndarray
        Creager parameter a, shape (n,) where n is broadcast size
    b : np.ndarray
        Creager parameter b, shape (n,) where n is broadcast size
    c : np.ndarray
        Creager parameter c, shape (n,) where n is broadcast size
    """
    # Broadcast all inputs to a common shape and flatten to 1D
    A_b, C_b, F_b, L_b = np.broadcast_arrays(A, C, F, L)
    A_flat = np.asarray(A_b).ravel().astype(float)
    C_flat = np.asarray(C_b).ravel().astype(float)
    F_flat = np.asarray(F_b).ravel().astype(float)
    L_flat = np.asarray(L_b).ravel().astype(float)

    b = (C_flat - A_flat) / (2 * A_flat)
    c = (4 * L_flat + 2 * F_flat - A_flat - C_flat) / (8 * A_flat)

    match direction.lower():
        case "polar":
            a = C_flat - b - c
        case "equatorial":
            a = A_flat
        case _:
            raise ValueError("direction must be 'polar' or 'equatorial'")

    return a, b, c


def calculate_traveltime(
    theta: np.ndarray | float,
    a: np.ndarray | float,
    b: np.ndarray | float,
    c: np.ndarray | float,
) -> np.ndarray:
    """Calculate the traveltime perturbation according to Creager 1992.

    Parameters
    ----------
    theta : np.ndarray | float
        Angle between the ray and the symmetry axis (in radians).
        Can be scalar or array. Will be broadcast with a, b, c.
    a : np.ndarray | float
        Creager parameter a. Can be scalar or array. Will be broadcast with theta, b, c.
    b : np.ndarray | float
        Creager parameter b. Can be scalar or array. Will be broadcast with theta, a, c.
    c : np.ndarray | float
        Creager parameter c. Can be scalar or array. Will be broadcast with theta, a, b.

    Returns
    -------
    dt : np.ndarray
        Traveltime perturbation. Shape is the result of broadcasting all input shapes.
    """
    # Convert to arrays and use numpy broadcasting
    theta = np.asarray(theta)
    a = np.asarray(a)
    b = np.asarray(b)
    c = np.asarray(c)

    cos_theta = np.cos(theta)
    dt = a + b * cos_theta**2 + c * cos_theta**4
    return dt
