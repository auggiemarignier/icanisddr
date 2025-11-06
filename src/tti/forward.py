"""Forward modelling of traveltimes in TTI media."""

import numpy as np

from .elastic import transverse_isotropic_tensor
from .rotation import rotation_matrix_zy


def construct_general_tti_tensor(
    A: float,
    C: float,
    F: float,
    L: float,
    N: float,
    eta1: float,
    eta2: float,
) -> np.ndarray:
    """
    Construct a rotated transverse isotropic elastic tensor in Voigt notation.

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
    eta1 : float
        Tilt angle in radians.
    eta2 : float
        Azimuthal angle in radians.

    Returns
    -------
    C_rotated : ndarray, shape (3, 3, 3, 3)
        Rotated transverse isotropic elastic tensor as a 4th-order tensor (not in Voigt notation).
    """
    C_tti = transverse_isotropic_tensor(A, C, F, L, N)
    R = rotation_matrix_zy(eta1, eta2)

    C_rotated = np.einsum("pi,qj,rk,sl,ijkl->pqrs", R, R, R, R, C_tti)

    return C_rotated


def calculate_relative_traveltime(n: np.ndarray, D: np.ndarray) -> float:
    r"""
    Calculate relative traveltime perturbation.

    .. math::
        \frac{\delta t}{t_{\mathrm{PREM}}} = \sum_{i,j,k,l = 1}^3 n_i n_j n_k n_l D_{ijkl}(\eta_1, \eta_2, \delta A, \delta C, \delta F | N = N_{\mathrm{PREM}}, L = L_{PREM})

    Parameters
    ----------
    n : ndarray, shape (3,)
        Ray direction unit vector.
    D : ndarray, shape (3, 3, 3, 3)
        4th-order perturbation tensor.

    Returns
    -------
    float
        Relative traveltime perturbation.
    """
    dt_over_t = np.einsum("ijkl,i,j,k,l", D, n, n, n, n)
    return dt_over_t
