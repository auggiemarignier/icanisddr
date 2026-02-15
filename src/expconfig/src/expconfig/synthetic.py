"""Utilities for creating synthetic data for experiments."""

from collections.abc import Callable
from typing import Protocol

import numpy as np
from pydantic import BaseModel

from .geometry import IC_RADIUS

RNG = np.random.default_rng(1234)


class TrueBulkICConfig(BaseModel):
    """True bulk IC model parameters.

    These parameters define a single elastic tensor for the entire inner core.
    This is cylindrically anisotropic with symmetry axis along the Earth's rotation axis.
    The Love parameters are relative perturbations, not absolute values.
    The values for A, C, F are from Brett et al. (2024), while L and N are set to 0 (i.e. the absolute values are equal to the reference model).
    The angles eta1 and eta2 are in degrees.
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


DEFAULT_TRUTH = TrueBulkICConfig().as_array()


def mw_sampling(L: int) -> tuple[np.ndarray, np.ndarray]:
    """Equally spaced points on a sphere using McEwen & Wiaux (2011) sampling."""

    t = np.arange(0, L).astype(np.float64)[:-1]  # ignore south pole
    thetas = (2 * t + 1) * np.pi / (2 * L - 1)
    p = np.arange(0, 2 * L - 1).astype(np.float64)
    phis = 2 * p * np.pi / (2 * L - 1)

    lats = 90.0 - np.degrees(thetas)
    lons = np.degrees(phis) - 180.0
    return lons, lats


class NoiseModel(Protocol):
    """Protocol for noise models used to add noise to synthetic data."""

    def __call__(
        self,
        noise_level: float,
        rng: np.random.Generator,
        data: np.ndarray | None = None,
        **kwargs: object,
    ) -> np.ndarray:
        """Generate noise to add to synthetic data.

        Parameters
        ----------
        noise_level : float
            Noise level for synthetic data.
        rng : np.random.Generator
            Random number generator to use for noise generation.
        data : ndarray, shape (num_paths,), optional
            Data to which noise will be added. Some noise models may use the data to determine the scale of the noise.
        **kwargs : dict
            Additional keyword arguments for specific noise models.

        Returns
        -------
        ndarray, shape (num_paths,)
            Noise to add to synthetic data.
        """
        ...


def gaussian_noise_data_max(
    noise_level: float,
    rng: np.random.Generator,
    data: np.ndarray | None = None,
    **kwargs: object,
) -> np.ndarray:
    """Create Gaussian noise with a maximum noise level relative to the maximum absolute value of the data.

    Parameters
    ----------
    noise_level : float
        Noise level for synthetic data, defined as the standard deviation of the noise relative to the maximum absolute value of the data. For example, a noise_level of 0.1 means that the noise will have a standard deviation equal to 10% of the maximum absolute value of the data.
    rng : np.random.Generator
        Random number generator to use for noise generation.
    data : ndarray, shape (n,)
        Data to which noise will be added. The maximum absolute value of this data will be used to determine the scale of the noise. If all zeros, the noise level will be interpreted as an absolute scale rather than a relative scale.

    Returns
    -------
    ndarray, shape (n,)
        Gaussian noise to add to synthetic data.

    Raises
    ------
    ValueError
        If data is None, since this noise model requires data to determine the noise scale and the number of noise samples. data is None by default to conform to the NoiseModel protocol, but this specific noise model requires data to function properly.
    """
    if data is None:
        raise ValueError(
            "Data must be provided for gaussian_noise_data_max to determine noise scale and shape."
        )
    scale = noise_level
    if (data_max := np.abs(data).max()) == 0:
        # interpret noise_level as absolute scale when data is all zeros
        scale *= 1.0
    else:
        scale *= data_max
    return rng.normal(loc=0.0, scale=scale, size=data.shape)


noise_models: dict[str, NoiseModel] = {
    "gaussian_data_max": gaussian_noise_data_max,
}


def create_synthetic_data(
    calculator_fn: Callable[[np.ndarray], np.ndarray],
    truth: np.ndarray = DEFAULT_TRUTH,
    noise_level: float = 0.05,
    noise_model: str = "gaussian_data_max",
    noise_kwargs: dict[str, object] | None = None,
) -> np.ndarray:
    """Create synthetic travel time data for bulk IC model.

    Parameters
    ----------
    calculator_fn : Callable[[np.ndarray], ndarray]
        Function that computes travel time perturbations given IC model parameters.
    truth : np.ndarray, shape (7,), optional
        True bulk IC model parameters.
    noise_level : float, optional
        Noise level for synthetic data. Default is 0.05.
    noise_model : str, optional
        Noise model to use for synthetic data. Default is "gaussian_data_max".
    noise_kwargs : dict[str, object], optional
        Additional keyword arguments to pass to the selected noise model. If ``None``,
        no extra keyword arguments are forwarded (equivalent to an empty dict). The
        contents of this dictionary depend on the specific noise model and are passed
        through as ``**noise_kwargs`` to the noise model callable.

    Returns
    -------
    ndarray, shape (num_paths,)
        Synthetic relative travel time perturbations for each path.
    """
    synthetic_data = calculator_fn(truth)

    # If noise level is zero, always return the un-noised data regardless of noise_model.
    if noise_level == 0.0:
        return synthetic_data

    # Explicitly support "none"/"identity" as no-noise models.
    if noise_model in ("none", "identity"):
        return synthetic_data

    noise_model_fn = noise_models.get(noise_model)
    if noise_model_fn is None:
        available = ", ".join(sorted(noise_models.keys()))
        raise ValueError(
            f"Unknown noise model '{noise_model}'. "
            f"Available noise models are: {available}. "
            "Use noise_level=0.0 or noise_model='none'/'identity' to disable noise."
        )
    return synthetic_data + noise_model_fn(
        noise_level, RNG, synthetic_data, **(noise_kwargs or {})
    )


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

    lon_sources, lat_sources = mw_sampling(int(180 / source_spacing))
    lon_receivers = lon_sources + source_spacing / 2
    lat_receivers = lat_sources
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
