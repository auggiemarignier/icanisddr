"""Compound Prior combining multiple prior components."""

from collections.abc import Sequence
from dataclasses import InitVar, dataclass, field
from itertools import chain

import numpy as np

from ._protocols import PriorFunction
from ._utils import _normalise_indices
from .uniform import UniformPrior


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
    indices: InitVar[Sequence[int] | slice | np.ndarray]

    _indices: np.ndarray = field(default_factory=lambda: np.array([]))

    def __post_init__(self, indices: Sequence[int] | slice | np.ndarray) -> None:
        """Convert indices to a numpy array if it's a slice."""
        self._indices = _normalise_indices(indices, self.prior_fn.n)

    @property
    def n(self) -> int:
        """Number of parameters in this prior component."""
        return int(np.asarray(self.indices).size)

    @property
    def indices(self) -> np.ndarray:
        """Get the indices as a numpy array."""
        return self._indices


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
