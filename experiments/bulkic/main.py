"""Synthetic bulk IC experiment entry point."""

import logging

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


def main():
    """Main function for synthetic bulk IC experiment."""
    logger.info("Starting synthetic bulk IC experiment")

    rng = np.random.default_rng(42)

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

    # Run MCMC sampling
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

    # Hypothesis 1: Vertical symmetry axis (eta1 = 0, eta2 = 0)
    # => margninalise out the love parameters, keeping the last two
    logger.info("Training flow model on marginalised samples")
    marginalised_samples = marginalise_samples(samples, [5, 6])
    model = RealNVPModel(2)
    chains = hm.Chains(ndim=2)
    chains.add_chains_2d(marginalised_samples, lnprob, sampler.nwalkers)
    chains_train, chains_test = hm.utils.split_data(chains, training_proportion=0.8)
    model.fit(chains_train.samples, epochs=5, verbose=True)

    # Calculate SDDRs
    logger.info("Calculating Savage-Dickey density ratio")
    prior_h1 = marginalise_prior(prior, [5, 6])(np.zeros(2))
    posterior_h1 = model.predict(np.zeros(2))
    sddr_h1 = posterior_h1 - prior_h1  # log SDDR for hypothesis 1
    logger.info(
        f"SDDR for hypothesis 1 (vertical symmetry axis): {np.exp(sddr_h1):.2f}"
    )
    logger.info("Experiment complete")


if __name__ == "__main__":
    main()
