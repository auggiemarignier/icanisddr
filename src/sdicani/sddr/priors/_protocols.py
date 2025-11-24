"""Common Prior Protocols."""

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
