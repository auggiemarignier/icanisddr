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
        Love parameter A (C_11). Can be scalar or array of any shape.
    C : np.ndarray | float
        Love parameter C (C_33). Can be scalar or array of any shape.
    F : np.ndarray | float
        Love parameter F (C_13). Can be scalar or array of any shape.
    L : np.ndarray | float
        Love parameter L (C_44). Can be scalar or array of any shape.
    N : np.ndarray | float, optional
        Love parameter N (C_66) (unused but included for completeness). Can be scalar or array of any shape.

    Returns
    -------
    a : np.ndarray
        Creager parameter a, shape matches broadcast shape of inputs
    b : np.ndarray
        Creager parameter b, shape matches broadcast shape of inputs
    c : np.ndarray
        Creager parameter c, shape matches broadcast shape of inputs
    """
    # Broadcast all inputs to a common shape (preserve shape like elastic module)
    A_b, C_b, F_b, L_b = np.broadcast_arrays(A, C, F, L)

    b = (C_b - A_b) / (2 * A_b)
    c = (4 * L_b + 2 * F_b - A_b - C_b) / (8 * A_b)

    match direction.lower():
        case "polar":
            a = C_b - b - c
        case "equatorial":
            a = A_b
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
    # Convert to arrays
    theta = np.asarray(theta)
    a = np.asarray(a)
    b = np.asarray(b)
    c = np.asarray(c)

    # Expand dimensions so theta broadcasts with leading dimensions of a, b, c
    # If a, b, c have shape (..., ), and theta has shape (n,),
    # we want result shape (..., n)
    # Add trailing dimensions to a, b, c to match theta's dimensions
    if a.ndim > 0 and theta.ndim > 0:
        # Add trailing axis to a, b, c for broadcasting with theta
        a = a[..., np.newaxis]
        b = b[..., np.newaxis]
        c = c[..., np.newaxis]

    cos_theta = np.cos(theta)
    dt = a + b * cos_theta**2 + c * cos_theta**4
    return dt
