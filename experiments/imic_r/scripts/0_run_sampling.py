"""Synthetic IMIC experiment entry point."""

import logging
import os
import pickle
from datetime import datetime
from functools import partial
from pathlib import Path

os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"
os.environ["VECLIB_MAXIMUM_THREADS"] = "1"
import numpy as np

from expconfig.config import PriorsConfig
from expconfig.synthetic import (
    SynthConfig,
    create_paths,
    create_synthetic_data,
)
from icprem import PREM_IC_RHO, PREM_IC_VP
from raytracer import BallInShell, CompositeRegion
from sampling.likelihood import GaussianLikelihood
from sampling.priors import CompoundPrior
from sampling.sampling import MCMCConfig, mcmc
from tti.traveltimes import TravelTimeCalculator
from tti.traveltimes.paths import calculate_path_direction_vector

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)

CFG = SynthConfig.load(Path(__file__).parent.parent / "config.yaml")

region = CFG.geometry.to_composite_region()
IC_RADIUS = region.regions[1].radius_outer

IC_IN, IC_OUT = create_paths(source_spacing=30.0)
PATH_DIRECTIONS = calculate_path_direction_vector(IC_IN, IC_OUT)
total_distances = region.ray_distances(IC_IN, PATH_DIRECTIONS)

# Not sure why some rays have zero total distance.  Discard for now
valid_rays = total_distances > 0
IC_IN = IC_IN[valid_rays]
IC_OUT = IC_OUT[valid_rays]
PATH_DIRECTIONS = PATH_DIRECTIONS[valid_rays]
TOTAL_DISTANCES = total_distances[valid_rays, None]  # additional axis for broadcasting


def _calculate_weights(region: CompositeRegion) -> np.ndarray:
    segment_distances = region.ray_distances_per_region(IC_IN, PATH_DIRECTIONS)
    return segment_distances / TOTAL_DISTANCES


initial_weights = _calculate_weights(region)

NORMALISATION = -1 / (2 * PREM_IC_RHO * (PREM_IC_VP * 1e3) ** 2)
BASE_TTC_FACTORY = partial(
    TravelTimeCalculator,
    ic_in=IC_IN,
    ic_out=IC_OUT,
    normalisation=NORMALISATION,
    weights=initial_weights.T[np.newaxis],
)

# Synthetic data computed based on absolute perturbations from PREM including shear components
SYNTH_CALCULATOR = BASE_TTC_FACTORY(nested=False, shear=True, N=True)
# Forward model takes nested parameters and excludes both shear components.
FORWARD_CALCULATOR = BASE_TTC_FACTORY(nested=True)


def forward(params: np.ndarray) -> np.ndarray:
    """Forward model for synthetic IMIC experiment.

    Parameters
    ----------
    params : np.ndarray
        Model parameters. Shape (n_samples, n_parameters).
        Parameters are sorted as [A_1, C_1, F_1, eta1_1, eta2_1, A_2, C_2, F_2, eta1_2, eta2_2, r],
        where A, C, F, eta1, eta2 are the usual TTI parameters for each region (IMIC first, then OIC), and r is the radius of IMIC.

    Returns
    -------
    np.ndarray
        Predicted travel times. Shape (n_samples, n_rays).
    """
    tti_params = params[:, :-1]
    imic_radius = params[:, -1]
    batch_weights = np.stack(
        [_calculate_weights(BallInShell(r, IC_RADIUS)).T for r in imic_radius],
        axis=0,
    )  # shape (batch_size, n_cells, npaths)
    FORWARD_CALCULATOR.update_weights(batch_weights)
    return FORWARD_CALCULATOR(tti_params)


OUTPUT_DIR = (
    Path(__file__).parent.parent / "outputs" / datetime.now().strftime("%Y%m%d-%H%M%S")
)


def _setup_likelihood(
    synthetic_data: np.ndarray,
) -> GaussianLikelihood:
    logger.info("Setting up likelihood function...")
    inv_covar = np.array([1 / synthetic_data.std() ** 2])
    likelihood = GaussianLikelihood(forward, synthetic_data, inv_covar)
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

    rng = np.random.default_rng(42)

    logger.info("Creating synthetic data...")
    synthetic_data = create_synthetic_data(
        SYNTH_CALCULATOR,
        CFG.truth.as_array().flatten(),
        CFG.data.noise_level,
    )[0]
    logger.info(f"Synthetic data shape: {synthetic_data.shape}")

    likelihood = _setup_likelihood(synthetic_data)
    prior = _setup_prior(CFG.priors)

    logger.info("Running MCMC sampling")
    samples, lnprob = mcmc(
        prior.n, likelihood, prior, rng, MCMCConfig(**CFG.sampling.model_dump())
    )

    logger.info("MCMC sampling completed")

    logger.info("Saving samples to disk")
    OUTPUT_DIR.mkdir(parents=True, exist_ok=False)
    dump_results(samples, lnprob, OUTPUT_DIR)
    CFG.dump(OUTPUT_DIR / "config.yaml")


if __name__ == "__main__":
    main()
