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
    A: float,
    C: float,
    F: float,
    L: float,
    N: float = 0.0,
) -> tuple[float, float, float]:
    """Convert from Love parameters to Creager 1992 parameters a, b, c.

    Parameters
    ----------
    direction : Literal['polar', 'equatorial']
    A : float
        Love parameter A (C_11)
    C : float
        Love parameter C (C_33)
    F : float
        Love parameter F (C_13)
    L : float
        Love parameter L (C_44)
    N : float
        Love parameter N (C_66) (unused but included for completeness)

    Returns
    -------
    a : float
        Creager parameter a
    b : float
        Creager parameter b
    c : float
        Creager parameter c
    """
    b = (C - A) / (2 * A)
    c = (4 * L + 2 * F - A - C) / (8 * A)

    match direction.lower():
        case "polar":
            a = C - b - c
        case "equatorial":
            a = A
        case _:
            raise ValueError("direction must be 'polar' or 'equatorial'")

    return a, b, c


def calculate_traveltime(theta: float, a: float, b: float, c: float) -> float:
    """Calculate the traveltime perturbation according to Creager 1992.

    Parameters
    ----------
    theta : float
        Angle between the ray and the symmetry axis (in radians)
    a : float
        Creager parameter a
    b : float
        Creager parameter b
    c : float
        Creager parameter c

    Returns
    -------
    dt : float
        Traveltime perturbation
    """
    cos_theta = np.cos(theta)
    dt = a + b * cos_theta**2 + c * cos_theta**4
    return dt
