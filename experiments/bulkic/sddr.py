"""Synthetic bulk IC experiment entry point."""

import logging
import pickle
from pathlib import Path
from typing import Any

import numpy as np
from bulkic.config import load_config
from sampling.priors import CompoundPrior
from sddr.marginalisation import marginalise_prior
from sddr.sddr import (
    RealNVPConfig,
    TrainConfig,
    fit_marginalised_posterior,
    sddr,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)


def configure_posterior_fit(
    cfg: dict[str, Any],
) -> tuple[TrainConfig, RealNVPConfig]:
    """Configure the posterior fitting parameters.

    Parameters
    ----------
    cfg : dict[str, Any]
        Configuration dictionary for the prior.

    Returns
    -------
    train_cfg : TrainConfig
        Training configuration for the flow model.
    realnvp_cfg : RealNVPConfig
        RealNVP configuration for the flow model.
    """

    return TrainConfig(**cfg.get("training", {})), RealNVPConfig(
        **cfg.get("realnvp", {})
    )


CFG_FILE = Path(__file__).parent / "config.yaml"
SAMPLES_PATH = Path(__file__).parent / "samples.pkl"


def main() -> None:
    """Main function for synthetic bulk IC experiment."""
    logger.info("Starting synthetic bulk IC experiment")
    cfg = load_config(CFG_FILE)

    logger.info("Loading posterior samples from disk")
    samples = pickle.load(open(SAMPLES_PATH, "rb"))

    logger.info("Setting up prior")
    prior = CompoundPrior.from_dict(cfg.priors.model_dump())

    # Hypothesis 1: Vertical symmetry axis (eta1 = 0, eta2 = 0)
    # => margninalise out the love parameters, keeping the last two
    indices = [5, 6]  # Indices of eta1 and eta2

    logger.info("Configuring posterior fitting parameters")
    train_cfg, realnvp_cfg = configure_posterior_fit(cfg.model_dump())
    logger.info("Training flow model on marginalised samples")
    marg_posterior = fit_marginalised_posterior(
        samples, indices, realnvp_cfg, train_cfg
    )

    logger.info("Marginalising the prior")
    marg_prior = marginalise_prior(prior, [5, 6])

    nu = np.zeros(2)  # evaluation point
    logger.info("Calculating Savage-Dickey density ratio at point nu=0")
    sddr_h1 = sddr(marg_posterior, marg_prior, nu)
    logger.info(
        f"SDDR for hypothesis 1 (vertical symmetry axis): {np.exp(sddr_h1):.2f}"
    )

    logger.info("Experiment complete")


if __name__ == "__main__":
    main()
