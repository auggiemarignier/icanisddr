"""Real Data bulk IC experiment entry point."""

import logging
import pickle
from collections.abc import Callable
from datetime import datetime
from functools import partial
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from ptemcee import Sampler

from expconfig import ExpConfig
from sampling.likelihood import GaussianLikelihood
from sampling.priors import CompoundPrior, PriorFunction
from sampling.sampling import MCMCConfig
from tti.traveltimes import TravelTimeCalculator
from tti.traveltimes.parametrisations import NestedNoShearDegreesParametriser


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
        threads=8,
        ntemps=ntemps,
    )
    sampler.run_mcmc(initial_pos, config.nsteps)
    return sampler.chain, sampler.logprobability


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)


CFG_FILE = Path(__file__).parent.parent / "config.yaml"
DATA_FILE = Path(__file__).parent.parent / "data" / "brett2024_ic_traveltimes.parquet"
OUTPUT_DIR = (
    Path(__file__).parent.parent / "outputs" / datetime.now().strftime("%Y%m%d-%H%M%S")
)

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


def _gradient(ttc: TravelTimeCalculator, x: np.ndarray) -> np.ndarray:
    """Thin wrapper for the forward calculator gradient to swap axes for pints compatibility."""
    return ttc.gradient(x).swapaxes(-2, -1)


def _setup_likelihood(
    ttc: TravelTimeCalculator,
    dt_over_t: np.ndarray,
    sigma: np.ndarray,
) -> GaussianLikelihood:
    logger.info("Setting up likelihood function")
    gradient = partial(_gradient, ttc)
    inv_covar = 1 / sigma**2
    likelihood = GaussianLikelihood(
        ttc,
        dt_over_t,
        inv_covar,
        forward_fn_gradient=gradient,
    )
    return likelihood


def _setup_prior(prior_cfg: dict[str, Any]) -> CompoundPrior:
    logger.info("Setting up prior distributions")
    prior = CompoundPrior.from_dict(prior_cfg)
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
    """Main function for real data bulk IC experiment."""
    logger.info("Starting real data bulk IC experiment")
    cfg = ExpConfig.load(CFG_FILE)

    ic_in, ic_out, dt_over_t, sigma = _setup_data(DATA_FILE)
    ttc = TravelTimeCalculator(
        ic_in=ic_in,
        ic_out=ic_out,
        normalisation=-0.5,
        parametriser=NestedNoShearDegreesParametriser(),
    )
    prior = _setup_prior(cfg.priors.model_dump())
    likelihood = _setup_likelihood(ttc, dt_over_t, sigma)

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
