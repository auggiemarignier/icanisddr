"""Synthetic bulk IC experiment entry point."""

import numpy as np

from sddr.likelihood import gaussian_likelihood_factory
from sddr.prior import CompoundPrior, GaussianPrior, PriorComponent, UniformPrior
from tti.forward import TravelTimeCalculator

from .data import create_paths, create_synthetic_bulk_ic_data


def main():
    """Main function for synthetic bulk IC experiment."""

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
    _ = gaussian_likelihood_factory(ttc, synthetic_data, covar)

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
    _ = CompoundPrior(prior_components)

    # Create a posterior
    # posterior = posterior_factory(likelihood, prior)

    # Run MCMC sampling

    # Marginalise for each alternative hypothesis

    # Calculate SDDRs


if __name__ == "__main__":
    main()
