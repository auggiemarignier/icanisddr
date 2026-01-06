"""Synthetic bulk IC experiment entry point."""

import logging
from collections.abc import Callable

import harmonic as hm
import numpy as np
from emcee import EnsembleSampler
from harmonic.model import RealNVPModel

from bulkic.data import create_paths, create_synthetic_bulk_ic_data
from sdicani.sddr.likelihood import gaussian_likelihood_factory
from sdicani.sddr.posterior import marginalise_samples, posterior_factory
from sdicani.sddr.prior import (
    CompoundPrior,
    GaussianPrior,
    PriorComponent,
    UniformPrior,
    marginalise_prior,
)
from sdicani.tti.forward import TravelTimeCalculator

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)


def setup() -> tuple[
    Callable[[np.ndarray], float], CompoundPrior, Callable[[np.ndarray], float]
]:
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
    likelihood = gaussian_likelihood_factory(
        ttc, synthetic_data, inv_covar, skip_validation=True, covar_is_inverse=True
    )

    # Create a prior
    logger.info("Setting up prior distributions")
    eta1_prior = UniformPrior(np.array([-180.0]), np.array([180.0]))
    eta2_prior = UniformPrior(np.array([0.0]), np.array([90.0]))
    love_priors = GaussianPrior(
        mean=np.zeros(5), covar=np.eye(5) * 0.2
    )  # THIS INCLUDES L AND N!
    prior_components = [
        PriorComponent(love_priors, slice(0, 5)),
        PriorComponent(eta1_prior, slice(5, 6)),
        PriorComponent(eta2_prior, slice(6, 7)),
    ]
    prior = CompoundPrior(prior_components)

    # Create a posterior
    logger.info("Creating posterior distribution")
    posterior = posterior_factory(likelihood, prior)

    return posterior, prior, likelihood


def mcmc(
    posterior: Callable[[np.ndarray], float], rng: np.random.Generator
) -> tuple[np.ndarray, np.ndarray]:
    """Run MCMC sampling for the synthetic bulk IC experiment.

    Returns
    -------
    samples : ndarray, shape (num_samples, ndim)
        MCMC samples of the model parameters.
    lnprob : ndarray, shape (num_samples,)
        Log-probabilities of the MCMC samples.
    """
    logger.info("Starting MCMC sampling for synthetic bulk IC experiment")

    ndim = 7  # Number of model parameters: [A, C, F, L, N, eta1, eta2]
    nwalkers = 50
    nsteps = 1000
    logger.info(f"Running MCMC with {nwalkers} walkers for {nsteps} steps")
    initial_pos = rng.normal(
        0, 1, size=(nwalkers, ndim)
    )  # these should be drawn from the prior
    sampler = EnsembleSampler(nwalkers, ndim, posterior)
    sampler.run_mcmc(initial_pos, nsteps, progress=True)
    samples = np.ascontiguousarray(sampler.get_chain(flat=True)[200:, :])
    lnprob = np.ascontiguousarray(sampler.get_log_prob(flat=True)[200:])
    logger.info("MCMC sampling complete")

    return samples, lnprob


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


def main():
    """Main function for synthetic bulk IC experiment."""
    logger.info("Starting synthetic bulk IC experiment")

    rng = np.random.default_rng(42)
    posterior, prior, likelihood = setup()

    samples, lnprob = mcmc(posterior, rng)

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
