"""Create synthetic bulk IC data."""

from dataclasses import dataclass

import numpy as np

from tti.forward import TravelTimeCalculator


@dataclass(frozen=True)
class TrueBulkICParams:
    """True bulk IC model parameters.

    These parameters define a single elastic tensor for the entire inner core.
    This is cylindrically anisotropic with symmetry axis along the Earth's rotation axis.
    The Love parameters are relative perturbations, not absolute values.
    The values for A, C, F are from Brett et al. (2024), while L and N are set to 0 (i.e. the absolute values are equal to the reference model).
    """

    A: float = 0.0143
    C: float = 0.0909
    F: float = -0.0858
    L: float = 0.0
    N: float = 0.0
    eta1: float = 0.0
    eta2: float = 0.0

    def as_array(self) -> np.ndarray:
        """Return the parameters as a numpy array."""
        return np.array([self.A, self.C, self.F, self.L, self.N, self.eta1, self.eta2])


TRUE_IC = TrueBulkICParams()


def create_synthetic_bulk_ic_data(ic_in: np.ndarray, ic_out: np.ndarray) -> np.ndarray:
    """Create synthetic travel time data for bulk IC model.

    Parameters
    ----------
    ic_in : ndarray, shape (num_paths, 3)
        Entry points of paths into the inner core (longitude (deg), latitude (deg), radius (km)).
    ic_out : ndarray, shape (num_paths, 3)
        Exit points of paths from the inner core (longitude (deg), latitude (deg), radius (km)).

    Returns
    -------
    ndarray, shape (num_paths,)
        Synthetic relative travel time perturbations for each path.
    """
    calculator = TravelTimeCalculator(ic_in, ic_out)
    synthetic_data = calculator(TRUE_IC.as_array())
    return synthetic_data


def create_paths(source_spacing: float) -> tuple[np.ndarray, np.ndarray]:
    """Create entry and exit points for paths through the inner core.

    Entry and exit points are generated on a regular grid defined by the source spacing, with the receivers placed halfway between the sources.

    Parameters
    ----------
    source_spacing : float
        Spacing between sources in degrees.

    Returns
    -------
    ic_in : ndarray, shape (num_paths, 3)
        Entry points of paths into the inner core (longitude (deg), latitude (deg), radius (km)).
    ic_out : ndarray, shape (num_paths, 3)
        Exit points of paths from the inner core (longitude (deg), latitude (deg), radius (km)).
    """

    def _mw_sampling(L: int) -> tuple[np.ndarray, np.ndarray]:
        """Equally spaced points on a sphere using McEwen & Wiaux (2011) sampling."""

        t = np.arange(0, L).astype(np.float64)
        thetas = (2 * t + 1) * np.pi / (2 * L - 1)
        p = np.arange(0, 2 * L - 1).astype(np.float64)
        phis = 2 * p * np.pi / (2 * L - 1)

        lats = 90.0 - np.degrees(thetas)
        lons = np.degrees(phis) - 180.0
        return lons, lats

    lon_sources, lat_sources = _mw_sampling(int(180 / source_spacing))
    lon_receivers = lon_sources + source_spacing / 2
    lat_receivers = lat_sources + source_spacing / 2
    r_ic = 1221.5  # km
    ic_in = np.array([(lon, lat, r_ic) for lat in lat_sources for lon in lon_sources])
    ic_out = np.array(
        [(lon, lat, r_ic) for lat in lat_receivers for lon in lon_receivers]
    )
    return ic_in, ic_out


if __name__ == "__main__":
    from experiments.bulkic.plotting import plot_ic_paths

    ic_in, ic_out = create_paths(source_spacing=10.0)
    print("Number of paths:", ic_in.shape[0])
    synthetic_data = create_synthetic_bulk_ic_data(ic_in, ic_out)
    print("Synthetic data shape:", synthetic_data.shape)
    plot_ic_paths(ic_in, ic_out)
