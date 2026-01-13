"""Forward modelling of traveltimes in TTI media."""

from collections.abc import Callable

import numpy as np

from .elastic import (
    elastic_tensor_to_voigt,
    transformation_to_voigt,
    transverse_isotropic_tensor,
    voigt_to_elastic_tensor,
)
from .rotation import rotation_matrix_zy, transformation_4th_order


def construct_general_tti_tensor(
    A: np.ndarray,
    C: np.ndarray,
    F: np.ndarray,
    L: np.ndarray,
    N: np.ndarray,
    eta1: np.ndarray,
    eta2: np.ndarray,
) -> np.ndarray:
    """
    Construct a rotated transverse isotropic elastic tensor.

    Parameters
    ----------
    A : np.ndarray (n,)
        Elastic constant C11 = C22
    C : np.ndarray (n,)
        Elastic constant C33
    F : np.ndarray (n,)
        Elastic constant C13 = C23
    L : np.ndarray (n,)
        Elastic constant C44 = C55
    N : np.ndarray (n,)
        Elastic constant C66
    eta1 : np.ndarray (n,)
        Tilt angle in radians.
    eta2 : np.ndarray (n,)
        Azimuthal angle in radians.

    Returns
    -------
    C_rotated : ndarray, shape (n, 3, 3, 3, 3)
        Rotated transverse isotropic elastic tensor as a 4th-order tensor (not in Voigt notation).
    """
    C_tti = transverse_isotropic_tensor(A, C, F, L, N)
    C_voigt = elastic_tensor_to_voigt(C_tti)

    R = rotation_matrix_zy(eta1, eta2)
    R_voigt = transformation_to_voigt(transformation_4th_order(R))

    C_rotated_voigt = R_voigt @ C_voigt @ R_voigt.swapaxes(-2, -1)

    return voigt_to_elastic_tensor(C_rotated_voigt)


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
    return np.einsum("...ijkl,...i,...j,...k,...l", D, n, n, n, n)


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
    ndarray, shape (..., 3)
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


class TravelTimeCalculator:
    """Class to calculate travel times in TTI media for a set of paths."""

    def __init__(
        self,
        ic_in: np.ndarray,
        ic_out: np.ndarray,
        nested: bool = True,
        shear: bool = False,
    ) -> None:
        """Initialise calculator.

        Parameters
        ----------
        ic_in : ndarray, shape (..., 3)
            Where the path enters the inner core (longitude (deg), latitude (deg), radius (km)).
        ic_out : ndarray, shape (..., 3)
            Where the path exits the inner core (longitude (deg), latitude (deg), radius (km)).
        nested : bool, optional
            Whether model parameters are nested (default is True).
            The nested model parameter order is:
                [A, C-A, F-A+2N, L, N-L, eta1, eta2]
            The non-nested model parameter order is:
                [A, C, F, L, N, eta1, eta2]
        shear : bool, optional
            Whether the shear parameters L and N are included in the model (default is False).
        """

        self._validate_paths(ic_in, ic_out)
        self._npaths = ic_in.shape[0]
        self.ic_in = ic_in
        self.ic_out = ic_out
        self.path_directions = calculate_path_direction_vector(ic_in, ic_out)

        self._unpacking_function = _unpackings[(nested, shear)]

    def __call__(self, m: np.ndarray) -> np.ndarray:
        """
        Calculate relative traveltime perturbations for all paths given TTI model parameters.

        Parameters
        ----------
        m : ndarray, shape (7n,)
            Model parameters for n subregions, flattened as [A, C, F, L, N, eta1, eta2] repeated n times.
            That is, [A₁, C₁, F₁, L₁, N₁, eta1₁, eta2₁, ..., Aₙ, Cₙ, Fₙ, Lₙ, Nₙ, eta1ₙ, eta2ₙ].

        Returns
        -------
        ndarray, shape (num_paths,)
            Relative traveltime perturbations for each path.
        """
        A, C, F, L, N, eta1, eta2 = self._unpacking_function(m)
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

        # Check bounds for longitude, latitude, and radius
        if not np.all(
            (ic_in[:, 0] >= -180)
            & (ic_in[:, 0] <= 180)
            & (ic_out[:, 0] >= -180)
            & (ic_out[:, 0] <= 180)
        ):
            raise ValueError("Longitude must be in [-180, 180] degrees.")

        if not np.all(
            (ic_in[:, 1] >= -90)
            & (ic_in[:, 1] <= 90)
            & (ic_out[:, 1] >= -90)
            & (ic_out[:, 1] <= 90)
        ):
            raise ValueError("Latitude must be in [-90, 90] degrees.")

        if not np.all((ic_in[:, 2] > 0) & (ic_out[:, 2] > 0)):
            raise ValueError("Radius must be greater than 0 km.")

        # Ensure in and out coordinates are different for each path
        same_mask = np.all(ic_in == ic_out, axis=-1)
        if np.any(same_mask):
            idx = np.where(same_mask)[0][0]
            raise ValueError(
                f"In and out coordinates must be different for each path (path {idx})"
            )


type seven_arrays = tuple[
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
]


def _unpack_nested_model_vector(m: np.ndarray) -> seven_arrays:
    r"""Unpack nested model vector into individual Love parameters.

    Note:
        There is NO checking of the input shape here for performance reasons.

    Parameters
    ----------
    m : ndarray, shape (M * 7)
        Nested model parameters: [A, \delta_{CA}, \delta_{F,A+2L}, L, \delta_{LN}, eta1, eta2]
        M is the number of model vectors (e.g. number of pixels).

    Returns
    -------
    A : ndarray, shape (M,)
        Elastic constant C11 = C22
    C : ndarray, shape (M,)
        Elastic constant C33
    F : ndarray, shape (M,)
        Elastic constant C13 = C23
    L : ndarray, shape (M,)
        Elastic constant C44 = C55
    N : ndarray, shape (M,)
        Elastic constant C66
    eta1 : ndarray, shape (M,)
        Tilt angle in radians.
    eta2 : ndarray, shape (M,)
        Azimuthal angle in radians.
    """
    mT = m.reshape(7, -1)
    return (
        mT[0],
        mT[1] + mT[0],
        mT[2] + mT[0] - 2 * mT[3],
        mT[3],
        mT[4] + mT[3],
        np.radians(mT[5]),
        np.radians(mT[6]),
    )


def _unpack_nested_model_vector_no_shear(m: np.ndarray) -> seven_arrays:
    r"""Unpack nested model vector into individual Love parameters, with L and N fixed at 0.

    Note:
        There is NO checking of the input shape here for performance reasons.

    Parameters
    ----------
    m : ndarray, shape (M * 5)
        Nested model parameters: [A, \delta_{CA}, \delta_{F,A+2L}, eta1, eta2]
        M is the number of model vectors (e.g. number of pixels).

    Returns
    -------
    A : ndarray, shape (M,)
        Elastic constant C11 = C22
    C : ndarray, shape (M,)
        Elastic constant C33
    F : ndarray, shape (M,)
        Elastic constant C13 = C23
    L : ndarray, shape (M,)
        Elastic constant C44 = C55
        Fixed at 0.
    N : ndarray, shape (M,)
        Elastic constant C66
        Fixed at 0.
    eta1 : ndarray, shape (M,)
        Tilt angle in radians.
    eta2 : ndarray, shape (M,)
        Azimuthal angle in radians.
    """
    mT = m.reshape(5, -1)
    zeros = np.zeros_like(mT[0])
    return (
        mT[0],
        mT[1] + mT[0],
        mT[2] + mT[0],  # since L=0, F = A - 2*0 = A
        zeros,
        zeros,
        np.radians(mT[3]),
        np.radians(mT[4]),
    )


def _unpack_model_vector(m: np.ndarray) -> seven_arrays:
    r"""Unpack model vector into individual Love parameters.

    Note:
        There is NO checking of the input shape here for performance reasons.

    Parameters
    ----------
    m : ndarray, shape (M * 7)
        Model parameters: [A, C, F, L, N, eta1, eta2]
        M is the number of model vectors (e.g. number of pixels).

    Returns
    -------
    A : ndarray, shape (M,)
        Elastic constant C11 = C22
    C : ndarray, shape (M,)
        Elastic constant C33
    F : ndarray, shape (M,)
        Elastic constant C13 = C23
    L : ndarray, shape (M,)
        Elastic constant C44 = C55
    N : ndarray, shape (M,)
        Elastic constant C66
    eta1 : ndarray, shape (M,)
        Tilt angle in radians.
    eta2 : ndarray, shape (M,)
        Azimuthal angle in radians.
    """
    mT = m.reshape(7, -1)
    return mT[0], mT[1], mT[2], mT[3], mT[4], np.radians(mT[5]), np.radians(mT[6])


def _unpack_model_vector_no_shear(m: np.ndarray) -> seven_arrays:
    r"""Unpack model vector into individual Love parameters, with L and N fixed at 0.

    Note:
        There is NO checking of the input shape here for performance reasons.

    Parameters
    ----------
    m : ndarray, shape (M * 5)
        Model parameters: [A, C, F, eta1, eta2]
        M is the number of model vectors (e.g. number of pixels).

    Returns
    -------
    A : ndarray, shape (M,)
        Elastic constant C11 = C22
    C : ndarray, shape (M,)
        Elastic constant C33
    F : ndarray, shape (M,)
        Elastic constant C13 = C23
    L : ndarray, shape (M,)
        Elastic constant C44 = C55
        Fixed at 0.
    N : ndarray, shape (M,)
        Elastic constant C66
        Fixed at 0.
    eta1 : ndarray, shape (M,)
        Tilt angle in radians.
    eta2 : ndarray, shape (M,)
        Azimuthal angle in radians.
    """
    mT = m.reshape(5, -1)
    zeros = np.zeros_like(mT[0])
    return mT[0], mT[1], mT[2], zeros, zeros, np.radians(mT[3]), np.radians(mT[4])


_unpackings: dict[tuple[bool, bool], Callable[[np.ndarray], seven_arrays]] = {
    (True, True): _unpack_nested_model_vector,
    (True, False): _unpack_nested_model_vector_no_shear,
    (False, True): _unpack_model_vector,
    (False, False): _unpack_model_vector_no_shear,
}
