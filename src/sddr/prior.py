"""Functions to calculate the log priors of model parameters.

In this application we'll deal with Gaussian and Uniform priors.
"""

from collections.abc import Callable, Sequence
from dataclasses import dataclass

import numpy as np


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


@dataclass
class PriorComponent:
    """Class representing a prior component.

    Multiple prior components can be combined to form a joint prior over
    different subsets of model parameters.

    Parameters
    ----------
    prior_fn : Callable[[np.ndarray], float]
        Prior function that takes model parameters and returns the log-prior.
    indices : Sequence[int]
        Indices of the model parameters that this prior component applies to.
    """

    prior_fn: Callable[[np.ndarray], float]
    indices: Sequence[int] | slice | np.ndarray


def compound_prior_factory(
    prior_components: Sequence[PriorComponent],
) -> Callable[[np.ndarray], float]:
    """
    Create a compound prior function from multiple prior components.

    Parameters
    ----------
    prior_components : Sequence[PriorComponent]
        Sequence of PriorComponent instances.

    Returns
    -------
    prior_fn : Callable[[np.ndarray], float]
        Compound prior function that takes model parameters and returns the log-prior.
    """

    def prior_fn(model_params: np.ndarray) -> float:
        """Compound log-prior."""
        total_log_prior = 0.0
        for component in prior_components:
            params_subset = model_params[component.indices]
            component_log_prior = component.prior_fn(params_subset)

            if np.isinf(component_log_prior) and component_log_prior < 0:
                return -np.inf  # Early exit if any component is -inf

            total_log_prior += component_log_prior
        return total_log_prior

    return prior_fn


def _validate_covariance_matrix(covar: np.ndarray, N: int) -> None:
    """
    Validate that the covariance matrix is symmetric and positive semidefinite.

    Parameters
    ----------
    covar : ndarray, shape (n, n)
        Covariance matrix to validate.
    N : int
        Expected size of the covariance matrix.

    Raises
    ------
    ValueError
        If the covariance matrix is not symmetric or not positive semidefinite.
    """
    if covar.shape != (N, N):
        raise ValueError(f"Covariance matrix must be of shape ({N}, {N}).")

    if not np.allclose(covar, covar.T):
        raise ValueError("Covariance matrix must be symmetric.")

    eigenvalues = np.linalg.eigvalsh(covar)
    if np.any(eigenvalues < -1e-10):  # Allow small numerical tolerance
        raise ValueError("Covariance matrix must be positive semidefinite.")
