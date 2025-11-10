"""Functions to calculate the log priors of model parameters.

In this application we'll deal with Gaussian and Uniform priors.
"""

from collections.abc import Callable

import numpy as np

from sddr.likelihood import _validate_covariance_matrix


def gaussian_prior_factory(
    mean: np.ndarray, covar: np.ndarray
) -> Callable[[np.ndarray], float]:
    """
    Create a Gaussian prior function.

    Parameters
    ----------
    mean : ndarray, shape (n,)
        Mean of the Gaussian prior e.g. a reference model.
    covar : ndarray, shape (n, n)
        Covariance matrix of the Gaussian prior.

    Returns
    -------
    prior_fn : Callable[[np.ndarray], float]
        Prior function that takes model parameters and returns the log-prior.

    Raises
    ------
    ValueError
        If the covariance matrix is not symmetric or not positive semidefinite.
    """
    _validate_covariance_matrix(covar, mean.size)

    inv_covar = np.linalg.inv(covar)

    def prior_fn(model_params: np.ndarray) -> float:
        """Gaussian log-prior."""
        diff = model_params - mean
        return -0.5 * diff.T @ inv_covar @ diff

    return prior_fn


def uniform_prior_factory(
    lower_bounds: np.ndarray, upper_bounds: np.ndarray
) -> Callable[[np.ndarray], float]:
    """
    Create a Uniform prior function.

    Parameters
    ----------
    lower_bounds : ndarray, shape (n,)
        Lower bounds of the uniform prior.
    upper_bounds : ndarray, shape (n,)
        Upper bounds of the uniform prior.

    Returns
    -------
    prior_fn : Callable[[np.ndarray], float]
        Prior function that takes model parameters and returns the log-prior.

    Raises
    ------
    ValueError
        If any lower bound is not less than the corresponding upper bound.
    """
    if np.any(lower_bounds >= upper_bounds):
        raise ValueError(
            "Each lower bound must be less than the corresponding upper bound."
        )

    def prior_fn(model_params: np.ndarray) -> float:
        """Uniform log-prior."""
        out_of_bounds = np.any(
            (model_params < lower_bounds) | (model_params > upper_bounds)
        )
        return np.where(out_of_bounds, -np.inf, 0.0)

    return prior_fn
