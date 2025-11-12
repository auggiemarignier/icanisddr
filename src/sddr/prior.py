"""Functions to calculate the log priors of model parameters.

In this application we'll deal with Gaussian and Uniform priors.
"""

from collections.abc import Sequence
from dataclasses import dataclass
from functools import singledispatch
from itertools import chain
from typing import Protocol, overload

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

    Raises
    ------
    ValueError
        If the covariance matrix is not symmetric, not positive semidefinite, or has a shape mismatch with the mean.
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


class UniformPrior:
    """Class representing a Uniform prior.

    Parameters
    ----------
    lower_bounds : ndarray, shape (n,)
        Lower bounds of the uniform prior.
    upper_bounds : ndarray, shape (n,)
        Upper bounds of the uniform prior.

    Raises
    ------
    ValueError
        If `lower_bounds` and `upper_bounds` have mismatched shapes,
        or if any lower bound is not less than the corresponding upper bound.
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
    """
    Represents a compound prior formed by combining multiple prior components.

    A compound prior is a joint prior distribution over model parameters, constructed
    by combining several prior components, each of which acts on a subset of the parameters.
    This is useful when different groups of parameters have different prior distributions,
    or when the overall prior can be factorized into independent components.

    When evaluating the compound prior, any UniformPrior components are reordered to be
    evaluated first. This allows for early exit optimization: if any UniformPrior component
    returns -inf (indicating the parameters are outside the allowed range), the evaluation
    stops immediately and -inf is returned for the whole compound prior.

    Parameters
    ----------
    prior_components : Sequence[PriorComponent]
        Sequence of PriorComponent instances, each specifying a prior and the indices
        of the model parameters it applies to.

    Raises
    ------
    IndexError
        If the indices specified in any PriorComponent are invalid for the given model parameters.
    TypeError
        If the input types for model parameters or prior components are incorrect.
    ValueError
        If the prior components do not cover the expected number of parameters, or if there is overlap.
    """

    def __init__(self, prior_components: Sequence[PriorComponent]) -> None:
        self.prior_components = prior_components
        self._n = sum(c.n for c in prior_components)

        self._uniform_components = [
            c for c in prior_components if isinstance(c.prior_fn, UniformPrior)
        ]
        self._non_uniform_components = [
            c for c in prior_components if not isinstance(c.prior_fn, UniformPrior)
        ]

    def __call__(self, model_params: np.ndarray) -> float:
        """Compound log-prior."""
        # Bring any UniformPriors to the front for early exit
        prior_components = chain(self._uniform_components, self._non_uniform_components)

        total_log_prior = 0.0
        for component in prior_components:
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


@overload
def marginalise_prior(
    prior: GaussianPrior, indices: Sequence[int] | slice | np.ndarray
) -> GaussianPrior: ...


@overload
def marginalise_prior(
    prior: UniformPrior, indices: Sequence[int] | slice | np.ndarray
) -> UniformPrior: ...


@overload
def marginalise_prior(
    prior: CompoundPrior, indices: Sequence[int] | slice | np.ndarray
) -> CompoundPrior: ...


@overload
def marginalise_prior(
    prior: PriorFunction, indices: Sequence[int] | slice | np.ndarray
) -> PriorFunction: ...


@singledispatch
def marginalise_prior(
    prior: PriorFunction, indices: Sequence[int] | slice | np.ndarray
) -> PriorFunction:
    r"""
    Create a marginalised prior function over specified parameter indices.

    This assumes that all the random variables are independent.

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

    Raises
    ------
    ValueError
        If no indices are provided (i.e., the indices array is empty).
    IndexError
        If any of the provided indices are out of bounds for the parameter array.
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


@marginalise_prior.register
def _(
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

    Raises
    ------
    ValueError
        If no indices are provided to keep after marginalisation, or if no prior components remain after marginalisation.
    """
    idx = (
        np.arange(compound_prior.n)[indices]
        if isinstance(indices, slice)
        else np.asarray(indices, dtype=int)
    )
    if idx.size == 0:
        raise ValueError(
            "At least one index should be kept after marginalisation. Check the 'indices' parameter."
        )

    # For each component, find which of the requested indices belong to it
    new_components = []
    next_index = 0  # Track the starting position in the new compound space

    for component in compound_prior.prior_components:
        # Convert component indices to array
        if isinstance(component.indices, slice):
            start = component.indices.start or 0
            stop = component.indices.stop
            step = component.indices.step or 1
            component_indices = np.array(range(start, stop, step))
        else:
            component_indices = np.asarray(component.indices)

        # Find which requested indices belong to this component
        mask = np.isin(idx, component_indices)
        if not np.any(mask):
            continue  # No indices from this component are being kept

        kept_global_indices = idx[mask]

        # Map global indices to local indices within the component
        local_indices = np.array(
            np.searchsorted(component_indices, kept_global_indices)
        )

        # Marginalise the prior function to just these parameters
        marginalised_prior = marginalise_prior(component.prior_fn, local_indices)

        # Create new component with indices in the new compound space
        n_kept = local_indices.size
        new_component = PriorComponent(
            prior_fn=marginalised_prior,
            indices=np.arange(next_index, next_index + n_kept),
        )
        new_components.append(new_component)
        next_index += n_kept

    if not new_components:
        raise ValueError(
            "No prior components remain after marginalisation. Check the 'indices' parameter."
        )

    return CompoundPrior(new_components)


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
