"""Constructing and fitting the normalised posterior distribution."""

from collections.abc import Callable, Sequence

import numpy as np


def posterior_factory(
    likelihood_fn: Callable[[np.ndarray], float],
    prior_fn: Callable[[np.ndarray], float],
) -> Callable[[np.ndarray], float]:
    """
    Create a posterior function from likelihood and prior functions.

    Parameters
    ----------
    likelihood_fn : Callable[[np.ndarray], float]
        Likelihood function that takes model parameters and returns the log-likelihood.
    prior_fn : Callable[[np.ndarray], float]
        Prior function that takes model parameters and returns the log-prior.

    Returns
    -------
    posterior_fn : Callable[[np.ndarray], float]
        Posterior function that takes model parameters and returns the log-posterior.
    """

    def posterior_fn(model_params: np.ndarray) -> float:
        """Log-posterior function."""
        log_likelihood = likelihood_fn(model_params)
        log_prior = prior_fn(model_params)
        return log_likelihood + log_prior

    return posterior_fn


def marginalise_samples(
    samples: np.ndarray, param_indices: Sequence[int] | slice
) -> np.ndarray:
    """
    Marginalise posterior samples over specified parameter indices.

    Parameters
    ----------
    samples : ndarray, shape (n_samples, n_params)
        Posterior samples.
    param_indices : Sequence[int] | slice
        Indices of parameters to keep in the marginal distribution.

    Returns
    -------
    marginal_samples : ndarray, shape (n_samples, n_marginal_params)
        Marginalised posterior samples.
    """
    return samples[:, param_indices]
