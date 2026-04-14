"""Synthetic IMIC experiment entry point."""

import logging
import pickle
from datetime import datetime
from functools import partial
from pathlib import Path

import numpy as np

from expconfig.config import PriorsConfig
from expconfig.synthetic import (
    SynthConfig,
    create_paths,
    gaussian_noise,
)
from sampling.likelihood import GaussianLikelihood
from sampling.priors import CompoundPrior
from sampling.sampling import MCMCConfig, ptmcmc
from tti.traveltimes import TravelTimeCalculator
from tti.traveltimes.parametrisations import (
    NestedNoShearDegreesParametriser,
)
from tti.traveltimes.paths import calculate_path_direction_vector


def lonlatrad_to_xyz(lonlatrad: np.ndarray) -> np.ndarray:
    """Convert (lon, lat, radius) to Cartesian (x, y, z) coordinates.

    Parameters
    ----------
    lonlatrad : np.ndarray, shape (..., 3)
        Array of longitude (degrees), latitude (degrees), and radius (km).

    Returns
    -------
    xyz : np.ndarray, shape (..., 3)
        Array of Cartesian coordinates in km.
    """
    lon = np.radians(lonlatrad[..., 0])
    lat = np.radians(lonlatrad[..., 1])
    r = lonlatrad[..., 2]

    x = r * np.cos(lat) * np.cos(lon)
    y = r * np.cos(lat) * np.sin(lon)
    z = r * np.sin(lat)

    return np.stack([x, y, z], axis=-1)


def determine_weights(region, ic_in: np.ndarray, ic_out: np.ndarray) -> np.ndarray:
    """Determine weights for each path based on the distance travelled in each region.

    Parameters
    ----------
    region : CompositeRegion
        The composite region defining the geometry and properties of the inner core.
    ic_in : ndarray, shape (num_paths, 3)
        Entry points of paths into the inner core (longitude (deg), latitude (deg), radius (km)).
    ic_out : ndarray, shape (num_paths, 3)
        Exit points of paths from the inner core (longitude (deg), latitude (deg), radius

    Returns
    -------
    weights : ndarray, shape (1, num_segments, num_paths)
        Fractional distance of each path in each segment.  Additional axis for broadcasting with travel time calculator.
    """
    path_directions = calculate_path_direction_vector(ic_in, ic_out)
    segment_distances = region.ray_distances_per_region(
        lonlatrad_to_xyz(ic_in), path_directions
    )
    total_distances = segment_distances.sum(axis=1)
    weights = segment_distances / total_distances[:, None]
    return weights.T[None, ...]


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)


CFG_FILE = Path(__file__).parent.parent / "config.yaml"
OUTPUT_DIR = (
    Path(__file__).parent.parent / "outputs" / datetime.now().strftime("%Y%m%d-%H%M%S")
)


def _setup_likelihood(
    synthetic_data: np.ndarray,
    sigma: float | np.ndarray,
    ttc: TravelTimeCalculator,
) -> GaussianLikelihood:
    logger.info("Setting up likelihood function...")
    inv_covar = 1 / sigma**2
    likelihood = GaussianLikelihood(ttc, synthetic_data, inv_covar)
    return likelihood


def _setup_prior(prior_cfg: PriorsConfig) -> CompoundPrior:
    logger.info("Setting up prior distributions...")
    prior = CompoundPrior.from_dict(prior_cfg.model_dump())
    return prior


def dump_results(samples: np.ndarray, lnprob: np.ndarray, output_dir: Path) -> None:
    """Dump the results to disk.

    Dumped files are
    - samples_full.pkl: the full (after burn and thin) MCMC samples
    - lnprob_full.pkl: the log-probabilities of the full (after burn and thin) MCMC samples

    Parameters
    ----------
    samples : np.ndarray
        MCMC samples.
    lnprob : np.ndarray
        Log-probabilities of the samples.
    output_dir : Path
        Directory to save the results.
    """
    with open(output_dir / "samples_full.pkl", "wb") as f:
        pickle.dump(samples, f)
    with open(output_dir / "lnprob_full.pkl", "wb") as f:
        pickle.dump(lnprob, f)

    logger.info(f"Results saved to {output_dir}")


def main() -> None:
    """Main function for synthetic IMIC experiment."""
    logger.info("Starting synthetic IMIC experiment")
    cfg = SynthConfig.load(CFG_FILE)

    rng = np.random.default_rng(42)

    ic_in, ic_out = create_paths(source_spacing=30.0)
    region = cfg.geometry.to_composite_region()
    weights = determine_weights(region, ic_in, ic_out)
    base_ttc_factory = partial(
        TravelTimeCalculator,
        ic_in=ic_in,
        ic_out=ic_out,
        normalisation=-0.5,
        weights=weights,
    )

    logger.info("Creating synthetic data...")
    synth_calculator = base_ttc_factory()
    synthetic_data_clean = synth_calculator(cfg.truth.as_array().flatten())[0]
    noise = gaussian_noise(cfg.data.noise_level, rng, synthetic_data_clean)
    synthetic_data = synthetic_data_clean + noise
    logger.info(f"Synthetic data shape: {synthetic_data.shape}")

    sigma = np.full_like(synthetic_data, cfg.data.noise_level)
    forward_calculator = base_ttc_factory(
        parametriser=NestedNoShearDegreesParametriser()
    )
    likelihood = _setup_likelihood(synthetic_data, sigma, forward_calculator)
    prior = _setup_prior(cfg.priors)

    logger.info("Running MCMC sampling")
    samples, lnprob = ptmcmc(
        prior.n, likelihood, prior, rng, MCMCConfig(**cfg.sampling.model_dump())
    )

    logger.info("MCMC sampling completed")

    logger.info("Saving samples to disk")
    OUTPUT_DIR.mkdir(parents=True, exist_ok=False)
    dump_results(samples, lnprob, OUTPUT_DIR)
    cfg.dump(OUTPUT_DIR / "config.yaml")


if __name__ == "__main__":
    main()
