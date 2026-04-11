"""Synthetic IMIC experiment entry point."""

import logging
import pickle
from collections.abc import Callable
from datetime import datetime
from functools import partial
from pathlib import Path

import numpy as np
import pandas as pd
from ptemcee import Sampler
from tqdm import tqdm

from expconfig.config import ExpConfig, PriorsConfig
from expconfig.geometry import IC_RADIUS
from raytracer import BallInShell
from sampling.likelihood import GaussianLikelihood
from sampling.priors import CompoundPrior, PriorFunction
from sampling.sampling import MCMCConfig
from tti.traveltimes import TravelTimeCalculator
from tti.traveltimes.parametrisations import NestedNoShearDegreesParametriser
from tti.traveltimes.paths import calculate_path_direction_vector


def mcmc(
    ndim: int,
    likelihood: Callable[[np.ndarray], float | np.ndarray],
    prior: PriorFunction,
    rng: np.random.Generator,
    config: MCMCConfig | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Run MCMC sampling using the ensemble sampler.

    Parameters
    ----------
    ndim : int
        Number of dimensions in the parameter space.
    likelihood : Callable[[ndarray], float | ndarray]
        Likelihood function that takes model parameters and returns log-likelihood.
        Should support both scalar (1D) and vectorised (2D batch) inputs if
        config.vectorise is True.
    prior : PriorFunction
        Prior function that takes model parameters and returns log-prior.
        Should support both scalar (1D) and vectorised (2D batch) inputs if
        config.vectorise is True.
    rng : np.random.Generator
        Random number generator for initializing walkers.
    config : MCMCConfig or None, optional
        MCMC configuration. If None, uses default configuration.

    Returns
    -------
    samples : ndarray, shape (num_samples, ndim)
        MCMC samples of the model parameters, after burn-in and thinning.
    lnprob : ndarray, shape (num_samples,)
        Log-probabilities of the MCMC samples, after burn-in and thinning.
    """
    if config is None:
        config = MCMCConfig()

    ntemps = 10

    initial_pos = prior.sample(ntemps * config.nwalkers, rng).reshape(
        (ntemps, config.nwalkers, ndim)
    )

    sampler = Sampler(
        config.nwalkers,
        ndim,
        likelihood,
        prior,
        threads=12,
        ntemps=ntemps,
    )
    for _ in tqdm(sampler.sample(initial_pos, config.nsteps), total=config.nsteps):
        pass
    return sampler.chain, sampler.logprobability


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)

CFG_FILE = Path(__file__).parent.parent / "config.yaml"
DATA_FILE = Path(__file__).parent.parent / "data" / "brett2024_ic_traveltimes.parquet"

# Hierarchical noise levels obtained by Brett et al., 2022
noise_levels: dict[str, float] = {
    "ab": 0.95,
    "bc": 0.63,
    "cd": 0.29,
    "df": 0.95,
}


def _setup_data(
    data_file: Path,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    logger.info("Loading real data...")
    df = pd.read_parquet(data_file)
    ic_in = np.stack(df.in_location.values)
    ic_out = np.stack(df.out_location.values)
    dt_over_t = (df.delta_t / df.inner_core_travel_time).values
    #  The noise levels for each reference phase are given in seconds, so we need to convert them to fractional traveltime perturbations by dividing by the inner core travel time.
    # In principle this gives a different sigma for each observation.
    sigma = (
        df["reference_phase"].map(noise_levels) / df["inner_core_travel_time"]
    ).values
    logger.info(f"Real data shape: {dt_over_t.shape}")
    return ic_in, ic_out, dt_over_t, sigma


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


def determine_weights(
    region, ic_in: np.ndarray, path_directions: np.ndarray
) -> np.ndarray:
    """Determine weights for each path based on the distance travelled in each region.

    Parameters
    ----------
    region : CompositeRegion
        The composite region defining the geometry and properties of the inner core.
    ic_in : ndarray, shape (num_paths, 3)
        Entry points of paths into the inner core (longitude (deg), latitude (deg), radius (km)).
    path_directions : ndarray, shape (num_paths, 3)
        Direction vectors for each path.

    Returns
    -------
    weights : ndarray, shape (1, num_segments, num_paths)
        Fractional distance of each path in each segment.  Additional axis for broadcasting with travel time calculator.
    """
    segment_distances = region.ray_distances_per_region(
        lonlatrad_to_xyz(ic_in), path_directions
    )
    total_distances = segment_distances.sum(axis=1)
    weights = segment_distances / total_distances[:, None]
    return weights.T[None, ...]


def forward(ttc: TravelTimeCalculator, params: np.ndarray) -> np.ndarray:
    """Forward model for synthetic IMIC experiment.

    Parameters
    ----------
    ttc : TravelTimeCalculator
        The travel time calculator.
    params : np.ndarray
        Model parameters. Shape (n_samples, n_parameters).
        Parameters are sorted as [A_1, A_2, C_1, C_2, F_1, F_2, eta1_1, eta2_1, eta1_2, eta2_2, r],
        where A, C, F, eta1, eta2 are the usual TTI parameters for each region (IMIC first, then OIC), and r is the radius of IMIC.

    Returns
    -------
    np.ndarray
        Predicted travel times. Shape (n_samples, n_rays).
    """
    tti_params = params[:, :-1]
    imic_radius = params[:, -1]

    if imic_radius < 0 or imic_radius > IC_RADIUS:
        # Return large travel times for unphysical radii to effectively zero out their likelihood
        # This case would normally be caught by the prior, but this allows us to skip the actual travel time calculation for these unphysical cases.
        # The logprior will be -inf anyway.
        # This also avoids having to try/catch errors when determining weights for unphysical radii.
        return np.full((params.shape[0], ttc.ic_in.shape[0]), 1e6)

    weights = determine_weights(
        BallInShell(imic_radius, IC_RADIUS), ttc.ic_in, ttc.path_directions
    )  # shape (1, n_cells, npaths)
    ttc.update_weights(weights)
    return ttc(tti_params)


OUTPUT_DIR = (
    Path(__file__).parent.parent / "outputs" / datetime.now().strftime("%Y%m%d-%H%M%S")
)


def _setup_likelihood(
    ttc: TravelTimeCalculator,
    dt_over_t: np.ndarray,
    sigma: float | np.ndarray,
) -> GaussianLikelihood:
    logger.info("Setting up likelihood function...")
    inv_covar = 1 / sigma**2
    forward_fn = partial(forward, ttc)
    likelihood = GaussianLikelihood(forward_fn, dt_over_t, inv_covar)
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
    """Main function for real data imic_r experiment."""
    logger.info("Starting real data IMIC_r experiment")
    cfg = ExpConfig.load(CFG_FILE)

    ic_in, ic_out, dt_over_t, sigma = _setup_data(DATA_FILE)
    path_directions = calculate_path_direction_vector(ic_in, ic_out)
    region = cfg.geometry.to_composite_region()
    initial_weights = determine_weights(region, ic_in, path_directions)
    ttc = TravelTimeCalculator(
        ic_in=ic_in,
        ic_out=ic_out,
        normalisation=-0.5,
        weights=initial_weights,
        parametriser=NestedNoShearDegreesParametriser(),
    )
    likelihood = _setup_likelihood(ttc, dt_over_t, sigma)
    prior = _setup_prior(cfg.priors)

    logger.info("Running MCMC sampling")
    rng = np.random.default_rng(42)
    samples, lnprob = mcmc(
        prior.n, likelihood, prior, rng, MCMCConfig(**cfg.sampling.model_dump())
    )

    logger.info("MCMC sampling completed")

    logger.info("Saving samples to disk")
    OUTPUT_DIR.mkdir(parents=True, exist_ok=False)
    dump_results(samples, lnprob, OUTPUT_DIR)
    cfg.dump(OUTPUT_DIR / "config.yaml")


if __name__ == "__main__":
    main()
