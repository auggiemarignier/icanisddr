"""Create synthetic bulk IC data."""

import numpy as np

from tti.forward import TravelTimeCalculator

from .config import IC_RADIUS, TrueBulkICConfig

DEFAULT_TRUTH = TrueBulkICConfig().as_array()
RNG = np.random.default_rng(1234)


def create_synthetic_bulk_ic_data(
    ic_in: np.ndarray,
    ic_out: np.ndarray,
    truth: np.ndarray = DEFAULT_TRUTH,
    noise_level: float = 0.05,
) -> np.ndarray:
    """Create synthetic travel time data for bulk IC model.

    Parameters
    ----------
    ic_in : ndarray, shape (num_paths, 3)
        Entry points of paths into the inner core (longitude (deg), latitude (deg), radius (km)).
    ic_out : ndarray, shape (num_paths, 3)
        Exit points of paths from the inner core (longitude (deg), latitude (deg), radius (km)).
    truth : np.ndarray, shape (7,), optional
        True bulk IC model parameters.
    noise_level : float, optional
        Noise level for synthetic data as a fraction of the stddev of the clean data. Default is 0.05.

    Returns
    -------
    ndarray, shape (num_paths,)
        Synthetic relative travel time perturbations for each path.
    """
    calculator = TravelTimeCalculator(ic_in, ic_out, nested=True, shear=True)
    synthetic_data = calculator(truth)
    noise = RNG.normal(
        loc=0.0, scale=synthetic_data.std() * noise_level, size=synthetic_data.shape
    )
    return synthetic_data + noise


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
    sources = np.array(
        [(lon, lat, IC_RADIUS) for lat in lat_sources for lon in lon_sources]
    )
    receivers = np.array(
        [(lon, lat, IC_RADIUS) for lat in lat_receivers for lon in lon_receivers]
    )
    n_sources = sources.shape[0]
    n_receivers = receivers.shape[0]

    ic_in = np.repeat(sources, n_receivers, axis=0)
    ic_out = np.tile(receivers, (n_sources, 1))
    return ic_in, ic_out


if __name__ == "__main__":
    from bulkic.plotting import plot_ic_inout, plot_ic_paths

    ic_in, ic_out = create_paths(source_spacing=20)
    print("Number of paths:", ic_in.shape[0])
    synthetic_data = create_synthetic_bulk_ic_data(ic_in, ic_out)
    print("Synthetic data shape:", synthetic_data.shape)
    plot_ic_paths(ic_in, ic_out)
    plot_ic_inout(ic_in, ic_out)
