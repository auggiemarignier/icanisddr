"""Functions to calculate the log priors of model parameters.

In this application we'll deal with Gaussian and Uniform priors.
"""

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Protocol

import numpy as np


class PriorFunction(Protocol):
    """Protocol for prior functions.

    Stores configuration parameters for the prior.
    Helpful for marginalisation routines that need to access these parameters.
    """

    config_params: list[np.ndarray]
    n: int

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
        self._n = mean.size

    def __call__(self, model_params: np.ndarray) -> float:
        """Gaussian log-prior."""
        diff = model_params - self.mean
        return float(-0.5 * diff.T @ self.inv_covar @ diff)

    @property
    def config_params(self) -> list[np.ndarray]:
        """Configuration parameters of the prior."""
        return [self.mean, self.covar]

    @property
    def n(self) -> int:
        """Number of parameters in the Gaussian prior."""
        return self._n


def gaussian_prior_factory(mean: np.ndarray, covar: np.ndarray) -> GaussianPrior:
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
    prior_fn : GaussianPrior
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
        self._n = lower_bounds.size

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

    @property
    def n(self) -> int:
        """Number of parameters in the Uniform prior."""
        return self._n


def uniform_prior_factory(
    lower_bounds: np.ndarray, upper_bounds: np.ndarray
) -> UniformPrior:
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
    prior_fn : UniformPrior
        Prior function that takes model parameters and returns the log-prior.

    Raises
    ------
    ValueError
        If any lower bound is not less than the corresponding upper bound.
    """
    return UniformPrior(lower_bounds, upper_bounds)


def marginalise_prior(
    prior: PriorFunction, indices: Sequence[int] | slice | np.ndarray
) -> PriorFunction:
    r"""
    Create a marginalised prior function over specified parameter indices.

    This assumes that the all the random variables are independent.

    .. math::
        p(m_{\text{remaining}}) = \int p(m_{\text{remaining}}, m_{\text{marginalised}}) \, dm_{\text{marginalised}}
        = \int p(m_{\text{remaining}}) p(m_{\text{marginalised}}) \, dm_{\text{marginalised}}
        = p(m_{\text{remaining}}) \int p(m_{\text{marginalised}}) \, dm_{\text{marginalised}}
        = p(m_{\text{remaining}})

    Parameters
    ----------
    prior : PriorFunction
        Original prior function to marginalise.
    indices : Sequence[int] | slice | np.ndarray
        Indices of the model parameters to keep after marginalisation i.e. $m_{\text{remaining}}$.

    Returns
    -------
    marginal_prior_fn : PriorFunction
        Marginalised prior function that takes model parameters and returns the log-prior.
    """
    idx = (
        np.arange(prior.config_params[0].size)[indices]
        if isinstance(indices, slice)
        else np.asarray(indices, dtype=int)
    )
    if idx.size == 0:
        raise ValueError(
            "At least one index should be kept after marginalisation.  Check the 'indices' parameter."
        )

    marginalised_params: list[np.ndarray] = []
    for param in prior.config_params:
        # repeat idx for each dimension of param
        # e.g. for a covariance matrix, we need to select both rows and columns
        ndim_repeat = tuple([idx] * param.ndim)
        # create the n-dimensional index selector
        selector = np.ix_(*ndim_repeat)
        marginalised_params.append(param[selector])

    return type(prior)(*marginalised_params)


@dataclass
class PriorComponent:
    """Class representing a prior component.

    Multiple prior components can be combined to form a joint prior over
    different subsets of model parameters.

    Parameters
    ----------
    prior_fn : PriorFunction
        Prior function that takes model parameters and returns the log-prior.
    indices : Sequence[int] | slice | np.ndarray
        Indices of the model parameters that this prior component applies to.
    """

    prior_fn: PriorFunction
    indices: Sequence[int] | slice | np.ndarray

    @property
    def n(self) -> int:
        """Number of parameters in this prior component."""
        if isinstance(self.indices, slice):
            start = self.indices.start or 0
            stop = self.indices.stop
            step = self.indices.step or 1
            return (stop - start + (step - 1)) // step
        else:
            return len(self.indices)


class CompoundPrior:
    """Class representing a compound prior from multiple prior components.

    Parameters
    ----------
    prior_components : Sequence[PriorComponent]
        Sequence of PriorComponent instances.
    """

    def __init__(self, prior_components: Sequence[PriorComponent]) -> None:
        # Bring any UniformPriors to the front for early exit
        self.prior_components = sorted(
            prior_components,
            key=lambda c: not isinstance(c.prior_fn, UniformPrior),
        )
        self._n = sum(c.n for c in prior_components)

    def __call__(self, model_params: np.ndarray) -> float:
        """Compound log-prior."""
        total_log_prior = 0.0
        for component in self.prior_components:
            params_subset = model_params[component.indices]
            component_log_prior = component.prior_fn(params_subset)

            if np.isneginf(component_log_prior):
                return -np.inf  # Early exit if any component is -inf

            total_log_prior += component_log_prior
        return total_log_prior

    @property
    def n(self) -> int:
        """Total number of parameters in the compound prior."""
        return self._n


def compound_prior_factory(
    prior_components: Sequence[PriorComponent],
) -> CompoundPrior:
    """
    Create a compound prior function from multiple prior components.

    Parameters
    ----------
    prior_components : Sequence[PriorComponent]
        Sequence of PriorComponent instances.

    Returns
    -------
    prior_fn : CompoundPrior
        Compound prior function that takes model parameters and returns the log-prior.
    """

    return CompoundPrior(prior_components)


def marginalise_compound_prior(
    compound_prior: CompoundPrior, indices: Sequence[int] | slice | np.ndarray
) -> CompoundPrior:
    """
    Create a marginalised compound prior function over specified parameter indices.

    This assumes that all the random variables are independent.

    Parameters
    ----------
    compound_prior : CompoundPrior
        Original compound prior function to marginalise.
    indices : Sequence[int] | slice | np.ndarray
        Indices of the model parameters to keep after marginalisation.

    Returns
    -------
    marginal_compound_prior : CompoundPrior
        Marginalised compound prior function that takes model parameters and returns the log-prior.
    """
    pass


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
