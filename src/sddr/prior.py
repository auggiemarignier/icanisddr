"""Functions to calculate the log priors of model parameters.

In this application we'll deal with Gaussian and Uniform priors.
"""

from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import Protocol

import numpy as np


class PriorFunction(Protocol):
    """Protocol for prior functions.

    Stores configuration parameters for the prior.
    Helpful for marginalisation routines that need to access these parameters.
    """

    config_params: list[np.ndarray]

    def __call__(self, model_params: np.ndarray) -> float:
        """Calculate the log-prior for given model parameters."""


class GaussianPrior:
    """Class representing a Gaussian prior.

    Parameters
    ----------
    mean : ndarray, shape (n,)
        Mean of the Gaussian prior e.g. a reference model.
    covar : ndarray, shape (n, n)
        Covariance matrix of the Gaussian prior.
    """

    def __init__(self, mean: np.ndarray, covar: np.ndarray) -> None:
        _validate_covariance_matrix(covar, mean.size)
        self.mean = mean
        self.covar = covar
        self.inv_covar = np.linalg.inv(covar)

    def __call__(self, model_params: np.ndarray) -> float:
        """Gaussian log-prior."""
        diff = model_params - self.mean
        return float(-0.5 * diff.T @ self.inv_covar @ diff)

    @property
    def config_params(self) -> list[np.ndarray]:
        """Configuration parameters of the prior."""
        return [self.mean, self.covar]


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
        If the covariance matrix shape doesn't match the mean dimension, is not symmetric, or is not positive semidefinite.
    """
    return GaussianPrior(mean, covar)


class UniformPrior:
    """Class representing a Uniform prior.

    Parameters
    ----------
    lower_bounds : ndarray, shape (n,)
        Lower bounds of the uniform prior.
    upper_bounds : ndarray, shape (n,)
        Upper bounds of the uniform prior.
    """

    def __init__(self, lower_bounds: np.ndarray, upper_bounds: np.ndarray) -> None:
        if lower_bounds.shape != upper_bounds.shape:
            raise ValueError(
                f"Shape mismatch: lower_bounds has shape {lower_bounds.shape}, upper_bounds has shape {upper_bounds.shape}."
            )
        if np.any(lower_bounds >= upper_bounds):
            raise ValueError(
                "Each lower bound must be less than the corresponding upper bound."
            )
        self.lower_bounds = lower_bounds
        self.upper_bounds = upper_bounds

    def __call__(self, model_params: np.ndarray) -> float:
        """Uniform log-prior."""
        out_of_bounds = np.any(
            (model_params < self.lower_bounds) | (model_params > self.upper_bounds)
        )
        return float(np.where(out_of_bounds, -np.inf, 0.0))

    @property
    def config_params(self) -> list[np.ndarray]:
        """Configuration parameters of the prior."""
        return [self.lower_bounds, self.upper_bounds]


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
    return UniformPrior(lower_bounds, upper_bounds)


@dataclass
class PriorComponent:
    """Class representing a prior component.

    Multiple prior components can be combined to form a joint prior over
    different subsets of model parameters.

    Parameters
    ----------
    prior_fn : Callable[[np.ndarray], float]
        Prior function that takes model parameters and returns the log-prior.
    indices : Sequence[int] | slice | np.ndarray
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

            if np.isneginf(component_log_prior):
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
        If the covariance matrix has incorrect shape, is not symmetric, or is not positive semidefinite.
    """
    if covar.shape != (N, N):
        raise ValueError(f"Covariance matrix must be of shape ({N}, {N}).")

    if not np.allclose(covar, covar.T):
        raise ValueError("Covariance matrix must be symmetric.")

    eigenvalues = np.linalg.eigvalsh(covar)
    if np.any(eigenvalues < -1e-10):  # Allow small numerical tolerance
        raise ValueError("Covariance matrix must be positive semidefinite.")
