"""Main traveltime calculation routines for TTI media."""

import numpy as np

from ..elastic import tilted_transverse_isotropic_tensor
from ._unpackings import _unpackings
from .paths import calculate_path_direction_vector


def calculate_relative_traveltime(
    n: np.ndarray, D: np.ndarray, normalisation: float = 1.0
) -> np.ndarray:
    r"""
    Calculate relative traveltime perturbation.

    .. math::
        \frac{\delta t}{t_{\mathrm{PREM}}} \propto{} \sum_{i,j,k,l = 1}^3 n_i n_j n_k n_l D_{ijkl}(\eta_1, \eta_2, \delta A, \delta C, \delta F | N = N_{\mathrm{PREM}}, L = L_{\mathrm{PREM}})

    where :math:`n` is the ray direction unit vector, :math:`D` is the 4th-order perturbation tensor.
    For inner core travel times, the proportionality constant is :math:`-1/(2 \rho_{\mathrm{PREM}} v_{\mathrm{PREM}}^2)`, where :math:`\rho_{\mathrm{PREM}}` and :math:`v_{\mathrm{PREM}}` are the average inner core density and seismic velocity in PREM.

    Parameters
    ----------
    n : ndarray, shape (npaths, 3)
        Ray direction unit vector(s).
    D : ndarray, shape (..., 3, 3, 3, 3)
        4th-order perturbation tensor.  Leading dimensions are arbitrary.
    normalisation : float, optional
        Normalisation constant to apply to the relative traveltime perturbation (default is 1.0).

    Returns
    -------
    np.ndarray, shape (..., npaths)
        Relative traveltime perturbation.  Batched according to the leading dimensions of D.
    """
    # Broadcast n to match leading dimensions of D
    leading_shape = D.shape[:-4]
    n = np.atleast_2d(n)
    n = np.broadcast_to(n, leading_shape + n.shape)

    # ijkl are the components of the 4th-order tensor D
    # p is the path index
    return normalisation * np.einsum(
        "...ijkl,...pi,...pj,...pk,...pl->...p", D, n, n, n, n
    )


class TravelTimeCalculator:
    """Class to calculate travel times in TTI media for a set of paths."""

    def __init__(
        self,
        ic_in: np.ndarray,
        ic_out: np.ndarray,
        reference_love: np.ndarray | None = None,
        weights: np.ndarray | None = None,
        normalisation: float = 1.0,
        nested: bool = True,
        shear: bool = False,
        N: bool = False,
    ) -> None:
        """Initialise calculator.

        Parameters
        ----------
        ic_in : ndarray, shape (npaths, 3)
            Where the path enters the inner core (longitude (deg), latitude (deg), radius (km)).
        ic_out : ndarray, shape (npaths, 3)
            Where the path exits the inner core (longitude (deg), latitude (deg), radius (km)).
        reference_love : ndarray, shape (5,), optional
            Reference Love parameters [A, C, F, L, N].
            Default is None, in which case 0 is used.
        weights : ndarray, shape (nsegments, npaths), optional
            Weights for each segment along each path (default is None, which gives equal weights).
        nested : bool, optional
            Whether model parameters are nested (default is True).
            The nested model parameter order is:
                [A, C-A, F-A+2N, L-N, N, eta1, eta2]
            The non-nested model parameter order is:
                [A, C, F, L, N, eta1, eta2]
        shear : bool, optional
            Whether the shear parameters L and N are included in the model (default is False).
        N : bool, optional
            Whether the N parameter is included in the model (default is False).
        """

        self._validate_paths(ic_in, ic_out)
        self._npaths = ic_in.shape[0]
        self.ic_in = ic_in
        self.ic_out = ic_out
        self.path_directions = calculate_path_direction_vector(ic_in, ic_out)
        self.weights = weights
        self._unpacking_function = _unpackings[nested][(shear, N)]

        if reference_love is None:
            self.reference_love = np.zeros(5)
        else:
            if reference_love.shape != (5,):
                raise ValueError(
                    "reference_love must have shape (5,) containing [A, C, F, L, N]"
                )
            self.reference_love = reference_love

        self.normalisation = normalisation

    def __call__(self, m: np.ndarray) -> np.ndarray:
        """
        Calculate relative traveltime perturbations for all paths given TTI model parameters.

        Computes the relative traveltime perturbation in each cell along each path,
        then performs a weighted sum along the cell axis (axis=1) to get the total relative traveltime perturbation for each path.

        Parameters
        ----------
        m : ndarray, shape ([batch], 7n,)
            Model parameters for n subregions, flattened as [A, C, F, L, N, eta1, eta2] repeated n times, potentially in a batch.
            That is, [A₁, C₁, F₁, L₁, N₁, eta1₁, eta2₁, ..., Aₙ, Cₙ, Fₙ, Lₙ, Nₙ, eta1ₙ, eta2ₙ].

        Returns
        -------
        ndarray, shape ([batch], npaths,)
            Relative traveltime perturbations for each path.
        """
        m = np.atleast_2d(m)
        A, C, F, L, N, eta1, eta2 = self._unpacking_function(
            m
        )  # each shape (batch, cells)
        A += self.reference_love[0]
        C += self.reference_love[1]
        F += self.reference_love[2]
        L += self.reference_love[3]
        N += self.reference_love[4]
        D = tilted_transverse_isotropic_tensor(A, C, F, L, N, eta1, eta2)
        dt = calculate_relative_traveltime(
            self.path_directions, D, normalisation=self.normalisation
        )  # shape (batch, cells, npaths)

        batch, cells, npaths = dt.shape
        weights = (
            np.ones((cells, npaths)) / cells if self.weights is None else self.weights
        )

        return np.sum(weights * dt, axis=-2)

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
