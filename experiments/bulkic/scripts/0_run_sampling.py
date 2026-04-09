"""Synthetic bulk IC experiment entry point."""

import logging
import pickle
from collections.abc import Callable
from datetime import datetime
from functools import partial
from pathlib import Path

import numpy as np
from ptemcee import Sampler

from expconfig.config import PriorsConfig
from expconfig.synthetic import (
    SynthConfig,
    create_paths,
    gaussian_noise,
)
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

IC_IN, IC_OUT = create_paths(source_spacing=30.0)
BASE_TTC_FACTORY = partial(
    TravelTimeCalculator,
    ic_in=IC_IN,
    ic_out=IC_OUT,
    normalisation=-0.5,
)

# Synthetic data computed based on absolute perturbations from PREM including shear components
SYNTH_CALCULATOR = BASE_TTC_FACTORY()
# Forward model takes nested parameters, excluding shear components
FORWARD_CALCULATOR = BASE_TTC_FACTORY(parametriser=NestedNoShearDegreesParametriser())

CFG_FILE = Path(__file__).parent.parent / "config.yaml"
OUTPUT_DIR = (
    Path(__file__).parent.parent / "outputs" / datetime.now().strftime("%Y%m%d-%H%M%S")
)


def _gradient(x: np.ndarray) -> np.ndarray:
    """Thin wrapper for the forward calculator gradient to swap axes for pints compatibility."""
    return FORWARD_CALCULATOR.gradient(x).swapaxes(-2, -1)


def _setup_likelihood(
    synthetic_data: np.ndarray,
    sigma: float | np.ndarray,
) -> GaussianLikelihood:
    logger.info("Setting up likelihood function...")
    inv_covar = 1 / sigma**2
    likelihood = GaussianLikelihood(
        FORWARD_CALCULATOR,
        synthetic_data,
        inv_covar,
        forward_fn_gradient=_gradient,
    )
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
    """Main function for synthetic bulk IC experiment."""
    logger.info("Starting synthetic bulk IC experiment")
    cfg = SynthConfig.load(CFG_FILE)

    rng = np.random.default_rng(42)

    logger.info("Creating synthetic data...")
    synthetic_data_clean = SYNTH_CALCULATOR(cfg.truth.as_array())[0]
    noise = gaussian_noise(cfg.data.noise_level, rng, synthetic_data_clean)
    synthetic_data = synthetic_data_clean + noise
    logger.info(f"Synthetic data shape: {synthetic_data.shape}")

    sigma = np.full_like(synthetic_data, cfg.data.noise_level)
    likelihood = _setup_likelihood(synthetic_data, sigma)
    prior = _setup_prior(cfg.priors)

    logger.info("Running MCMC sampling")
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
