"""Synthetic bulk IC experiment entry point."""

import harmonic as hm
import numpy as np
from emcee import EnsembleSampler
from harmonic.model import RealNVPModel

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

from .data import create_paths, create_synthetic_bulk_ic_data


def main():
    """Main function for synthetic bulk IC experiment."""

    rng = np.random.default_rng(42)

    # Creating the synthetic data
    print("Running synthetic bulk IC experiment...")
    ic_in, ic_out = create_paths(source_spacing=10.0)
    print("Number of paths:", ic_in.shape[0])
    synthetic_data = create_synthetic_bulk_ic_data(ic_in, ic_out)
    print("Synthetic data shape:", synthetic_data.shape)

    # Create a forward travel time calculator
    ttc = TravelTimeCalculator(ic_in, ic_out)

    # Create a likelihood function
    covar = np.eye(synthetic_data.shape[0]) * 0.1  # Example covariance
    likelihood = gaussian_likelihood_factory(ttc, synthetic_data, covar)

    # Create a prior
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
    posterior = posterior_factory(likelihood, prior)

    # Run MCMC sampling
    ndim = 7  # Number of model parameters: [A, C, F, L, N, eta1, eta2]
    nwalkers = 50
    nsteps = 500
    initial_pos = rng.normal(
        0, 1, size=(nwalkers, ndim)
    )  # these should be drawn from the prior
    sampler = EnsembleSampler(nwalkers, ndim, posterior)
    sampler.run_mcmc(initial_pos, nsteps, progress=True)
    samples = np.ascontiguousarray(sampler.get_chain(flat=True)[200:, :])
    lnprob = np.ascontiguousarray(sampler.get_log_prob(flat=True)[200:])

    # Hypothesis 1: Vertical symmetry axis (eta1 = 0, eta2 = 0)
    # => margninalise out the love parameters, keeping the last two
    marginalised_samples = marginalise_samples(samples, [5, 6])
    model = RealNVPModel(2)
    chains = hm.Chains(ndim=2)
    chains.add_chains_2d(marginalised_samples, lnprob, sampler.nwalkers)
    chains_train, chains_test = hm.utils.split_data(chains, training_proportion=0.8)
    model.fit(chains_train.samples, epochs=5, verbose=True)

    # Calculate SDDRs
    prior_h1 = marginalise_prior(prior, [5, 6])(np.zeros(2))
    posterior_h1 = model.predict(np.zeros(2))
    sddr_h1 = posterior_h1 - prior_h1  # log SDDR for hypothesis 1
    print(f"SDDR for hypothesis 1 (vertical symmetry axis): {np.exp(sddr_h1):.2f}")


if __name__ == "__main__":
    main()
