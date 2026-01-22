"""Main traveltime calculation routines for TTI media."""

import numpy as np

from ..elastic import tilted_transverse_isotropic_tensor
from ._unpackings import _unpackings
from .paths import calculate_path_direction_vector


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
                [A, C-A, F-A+2N, L-N, N, eta1, eta2]
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
        D = tilted_transverse_isotropic_tensor(A, C, F, L, N, eta1, eta2)
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
