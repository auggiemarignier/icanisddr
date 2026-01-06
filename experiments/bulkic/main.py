"""Synthetic bulk IC experiment entry point."""

import logging
from collections.abc import Callable
from typing import Any

import harmonic as hm
import hydra
import numpy as np
from bulkic.data import create_paths, create_synthetic_bulk_ic_data
from harmonic.model import RealNVPModel
from omegaconf import OmegaConf

from sdicani.sddr.likelihood import gaussian_likelihood_factory
from sdicani.sddr.posterior import marginalise_samples, posterior_factory
from sdicani.sddr.priors import (
    CompoundPrior,
    marginalise_prior,
)
from sdicani.sddr.sampling import MCMCConfig, mcmc
from sdicani.tti.forward import TravelTimeCalculator

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

    # Creating the synthetic data
    logger.info("Creating synthetic data...")
    ic_in, ic_out = create_paths(source_spacing=20.0)
    logger.info(f"Number of paths: {ic_in.shape[0]}")
    synthetic_data = create_synthetic_bulk_ic_data(ic_in, ic_out)
    logger.info(f"Synthetic data shape: {synthetic_data.shape}")

    # Create a forward travel time calculator
    logger.info("Creating travel time calculator")
    ttc = TravelTimeCalculator(ic_in, ic_out)

    # Create a likelihood function
    logger.info("Setting up likelihood function")
    inv_covar = np.eye(synthetic_data.shape[0]) * (1 / 0.1)  # Example covariance
    likelihood = gaussian_likelihood_factory(ttc, synthetic_data, inv_covar)

    # Create a prior
    logger.info("Setting up prior distributions")
    prior = CompoundPrior.from_dict(prior_cfg)

    # Create a posterior
    logger.info("Creating posterior distribution")
    posterior = posterior_factory(likelihood, prior)

    return posterior, prior, likelihood


def fit_marginalised_posterior(
    samples: np.ndarray, marginal_indices: list[int]
) -> hm.model.FlowModel:
    """Fit a flow model to the marginalised posterior samples.

    Parameters
    ----------
    samples : ndarray, shape (num_samples, ndim)
        MCMC samples of the model parameters.
    marginal_indices : list of int
        Indices of the parameters to keep after marginalisation.

    Returns
    -------
    model : FlowModel
        Fitted flow model to the marginalised posterior.
    """
    logger.info("Training flow model on marginalised samples")
    marginalised_samples = marginalise_samples(samples, marginal_indices)
    model = RealNVPModel(len(marginal_indices))
    model.fit(marginalised_samples, epochs=5, verbose=True)
    logger.info("Flow model training complete")
    return model


def sddr(
    marginalised_posterior: hm.model.FlowModel,
    marginalised_prior: CompoundPrior,
    nu: np.ndarray,
) -> float:
    """Calculate the Savage-Dickey density ratio (SDDR) for given marginalised posterior and prior.

    Parameters
    ----------
    marginalised_posterior : FlowModel
        Fitted flow model to the marginalised posterior.
    marginalised_prior : CompoundPrior
        Marginalised prior distribution.
    nu : ndarray, shape (k,)
        Point at which to evaluate the SDDR, where k is the number of marginalised parameters.

    Returns
    -------
    sddr : float
        Log SDDR value at the given point.
    """
    prior_log_prob = marginalised_prior(nu)
    posterior_log_prob = marginalised_posterior.predict(nu)
    return float(posterior_log_prob - prior_log_prob)


@hydra.main(version_base=None, config_path=".", config_name="config")
def main(cfg: OmegaConf) -> None:
    """Main function for synthetic bulk IC experiment."""
    logger.info("Starting synthetic bulk IC experiment")
    cfg: dict[str, Any] = OmegaConf.to_object(cfg)

    rng = np.random.default_rng(42)
    posterior, prior, likelihood = setup(cfg["priors"])

    samples, lnprob = mcmc(prior.n, posterior, rng, MCMCConfig(**cfg["sampling"]))

    # Hypothesis 1: Vertical symmetry axis (eta1 = 0, eta2 = 0)
    # => margninalise out the love parameters, keeping the last two
    logger.info("Training flow model on marginalised samples")
    marg_posterior = fit_marginalised_posterior(samples, [5, 6])
    logger.info("Marginalising the prior")
    marg_prior = marginalise_prior(prior, [5, 6])

    logger.info("Calculating Savage-Dickey density ratio")
    nu = np.zeros(2)  # evaluation point
    sddr_h1 = sddr(marg_posterior, marg_prior, nu)
    logger.info(
        f"SDDR for hypothesis 1 (vertical symmetry axis): {np.exp(sddr_h1):.2f}"
    )

    logger.info("Experiment complete")


if __name__ == "__main__":
    main()
