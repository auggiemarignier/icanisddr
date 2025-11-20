"""Functions for calculating the Savage-Dickey density ratio."""

import harmonic as hm
import numpy as np
from harmonic.model import RealNVPModel

from .posterior import marginalise_samples
from .prior import CompoundPrior


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
    marginalised_samples = marginalise_samples(samples, marginal_indices)
    model = RealNVPModel(len(marginal_indices))
    model.fit(marginalised_samples, epochs=5, verbose=True)
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
