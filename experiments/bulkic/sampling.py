"""Synthetic bulk IC experiment entry point."""

import logging
import pickle
from collections.abc import Callable
from pathlib import Path
from typing import Any

import numpy as np
from bulkic.config import load_config
from bulkic.data import create_paths, create_synthetic_bulk_ic_data
from sampling.likelihood import GaussianLikelihood
from sampling.posterior import Posterior
from sampling.priors import (
    CompoundPrior,
)
from sampling.sampling import MCMCConfig, mcmc

from tti.forward import TravelTimeCalculator

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)


def setup(
    prior_cfg: dict[str, Any],
) -> tuple[Callable[[np.ndarray], float], CompoundPrior, Callable[[np.ndarray], float]]:
    """Setup function for synthetic bulk IC experiment.

    Returns
    -------
    posterior : Callable[[np.ndarray], float]
        Posterior distribution function.
    prior : CompoundPrior
        Prior distribution.
    likelihood : Callable[[np.ndarray], float]
        Likelihood function.
    """
    logger.info("Setting up synthetic bulk IC experiment")

    logger.info("Creating synthetic data...")
    ic_in, ic_out = create_paths(source_spacing=20.0)
    logger.info(f"Number of paths: {ic_in.shape[0]}")
    synthetic_data = create_synthetic_bulk_ic_data(ic_in, ic_out)
    logger.info(f"Synthetic data shape: {synthetic_data.shape}")

    logger.info("Creating travel time calculator")
    ttc = TravelTimeCalculator(ic_in, ic_out)

    logger.info("Setting up likelihood function")
    inv_covar = np.eye(synthetic_data.shape[0]) * (1 / 0.1)  # Example covariance
    likelihood = GaussianLikelihood(ttc, synthetic_data, inv_covar)

    logger.info("Setting up prior distributions")
    prior = CompoundPrior.from_dict(prior_cfg)

    logger.info("Creating posterior distribution")
    posterior = Posterior(likelihood, prior)

    return posterior, prior, likelihood


CFG_FILE = Path(__file__).parent / "config.yaml"
SAMPLES_PATH = Path(__file__).parent / "samples.pkl"


def main() -> None:
    """Main function for synthetic bulk IC experiment."""
    logger.info("Starting synthetic bulk IC experiment")
    cfg = load_config(CFG_FILE)

    rng = np.random.default_rng(42)
    posterior, prior, likelihood = setup(cfg.priors.model_dump())

    logger.info("Running MCMC sampling")
    samples, lnprob = mcmc(
        prior.n, posterior, rng, MCMCConfig(**cfg.sampling.model_dump())
    )

    logger.info("MCMC sampling completed")
    logger.info("Saving samples to disk")
    with open(SAMPLES_PATH, "wb") as f:
        pickle.dump(samples, f)


if __name__ == "__main__":
    main()
