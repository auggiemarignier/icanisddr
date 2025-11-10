"""Forward modelling of traveltimes in TTI media."""

import numpy as np
from pydantic import BaseModel, Field

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
    Construct a rotated transverse isotropic elastic tensor.

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


def calculate_relative_traveltime(n: np.ndarray, D: np.ndarray) -> np.ndarray:
    r"""
    Calculate relative traveltime perturbation.

    .. math::
        \frac{\delta t}{t_{\mathrm{PREM}}} = \sum_{i,j,k,l = 1}^3 n_i n_j n_k n_l D_{ijkl}(\eta_1, \eta_2, \delta A, \delta C, \delta F | N = N_{\mathrm{PREM}}, L = L_{PREM})

    Parameters
    ----------
    n : ndarray, shape (..., 3)
        Ray direction unit vector.
    D : ndarray, shape (3, 3, 3, 3)
        4th-order perturbation tensor.

    Returns
    -------
    np.ndarray, shape (...)
        Relative traveltime perturbation.
    """
    return np.einsum("ijkl,...i,...j,...k,...l", D, n, n, n, n)


def calculate_path_direction_vector(
    ic_in: np.ndarray,
    ic_out: np.ndarray,
) -> np.ndarray:
    """
    Calculate the path direction unit vector from ic_in to ic_out.

    Parameters
    ----------
    ic_in : ndarray, shape (..., 3)
        Where the path enters the inner core (longitude (deg), latitude (deg), radius (km)).
    ic_out : ndarray, shape (..., 3)
        Where the path exits the inner core (longitude (deg), latitude (deg), radius (km)).

    Returns
    -------
    n : ndarray, shape (..., 3)
        Path direction unit vector.
    """
    ic_in_xyz = _spherical_to_cartesian(ic_in[..., 0], ic_in[..., 1], ic_in[..., 2])
    ic_out_xyz = _spherical_to_cartesian(ic_out[..., 0], ic_out[..., 1], ic_out[..., 2])
    path_vector = ic_out_xyz - ic_in_xyz
    return path_vector / np.linalg.norm(path_vector, axis=-1, keepdims=True)


def _spherical_to_cartesian(
    lon: float | np.ndarray, lat: float | np.ndarray, r: float | np.ndarray
) -> np.ndarray:
    """
    Convert spherical coordinates to Cartesian coordinates.

    Parameters
    ----------
    lon : float | np.ndarray
        Longitude in degrees.
    lat : float | np.ndarray
        Latitude in degrees.
    r : float | np.ndarray
        Radius.

    Returns
    -------
    ndarray, shape (3,)
        Cartesian coordinates (x, y, z).
    """
    lon = np.asarray(lon)
    lat = np.asarray(lat)
    r = np.asarray(r)

    colat = np.radians(90 - lat)
    lon = np.radians(lon)
    x = r * np.sin(colat) * np.cos(lon)
    y = r * np.sin(colat) * np.sin(lon)
    z = r * np.cos(colat)

    return np.stack((x, y, z), axis=-1)


class _Coordinate(BaseModel):
    lon: float = Field(..., ge=-180, le=180, description="Longitude in degrees.")
    lat: float = Field(..., ge=-90, le=90, description="Latitude in degrees.")
    r: float = Field(..., gt=0, description="Radius in km.")


class TravelTimeCalculator:
    """Class to calculate travel times in TTI media for a set of paths."""

    def __init__(self, ic_in: np.ndarray, ic_out: np.ndarray):
        """Initialise calculator.

        Parameters
        ----------
        ic_in : ndarray, shape (..., 3)
            Where the path enters the inner core (longitude (deg), latitude (deg), radius (km)).
        ic_out : ndarray, shape (..., 3)
            Where the path exits the inner core (longitude (deg), latitude (deg), radius (km)).
        """

        self._validate_paths(ic_in, ic_out)
        self._npaths = ic_in.shape[0]
        self.ic_in = ic_in
        self.ic_out = ic_out
        self.path_directions = calculate_path_direction_vector(ic_in, ic_out)

    def __call__(self, m: np.ndarray) -> np.ndarray:
        """
        Calculate relative traveltime perturbations for all paths given TTI model parameters.

        Parameters
        ----------
        m : ndarray, shape (7,)
            Model parameters: [A, C, F, L, N, eta1, eta2]

        Returns
        -------
        ndarray, shape (num_paths,)
            Relative traveltime perturbations for each path.
        """
        A, C, F, L, N, eta1, eta2 = m
        D = construct_general_tti_tensor(A, C, F, L, N, eta1, eta2)
        return self.calculate_traveltimes(D)

    def calculate_traveltimes(self, D: np.ndarray) -> np.ndarray:
        """
        Calculate relative traveltime perturbations for all paths.

        Parameters
        ----------
        D : ndarray, shape (3, 3, 3, 3)
            4th-order perturbation tensor.

        Returns
        -------
        ndarray, shape (num_paths,)
            Relative traveltime perturbations for each path.
        """
        return calculate_relative_traveltime(self.path_directions, D)

    @property
    def npaths(self) -> int:
        """Number of paths."""
        return self._npaths

    def _validate_paths(self, ic_in: np.ndarray, ic_out: np.ndarray) -> None:
        """Validate the in and out coordinates."""
        if ic_in.shape[-1] != 3 or ic_out.shape[-1] != 3:
            raise ValueError("In and out coordinates must have shape (..., 3)")

        if ic_in.shape != ic_out.shape:
            raise ValueError("In and out coordinates must have the same shape")

        ins = [_Coordinate(lon=ic[0], lat=ic[1], r=ic[2]) for ic in ic_in]
        outs = [_Coordinate(lon=ic[0], lat=ic[1], r=ic[2]) for ic in ic_out]
        for i, (in_, out_) in enumerate(zip(ins, outs)):
            if in_ == out_:
                raise ValueError(
                    f"In and out coordinates must be different for each path (path {i})"
                )
