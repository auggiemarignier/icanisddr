"""Forward modelling of traveltimes in TTI media."""

import numpy as np

from .elastic import transformation_to_voigt, transverse_isotropic_tensor
from .rotation import rotation_matrix_zy, transformation_4th_order


def construct_general_tti_tensor(
    A: float,
    C: float,
    L: float,
    N: float,
    F: float,
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
    L : float
        Elastic constant C44 = C55
    N : float
        Elastic constant C66
    F : float
        Elastic constant C13 = C23
    eta1 : float
        Tilt angle in radians.
    eta2 : float
        Azimuthal angle in radians.

    Returns
    -------
    C_rotated : ndarray, shape (6, 6)
        Rotated transverse isotropic elastic tensor in Voigt notation.
    """
    C_tti = transverse_isotropic_tensor(A, C, F, L, N)
    R6 = transformation_to_voigt(
        transformation_4th_order(rotation_matrix_zy(eta1, eta2))
    )
    C_rotated = R6 @ C_tti @ R6.T

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
