"""An abstract base class defining the parameteriser objects."""

from abc import ABC, abstractmethod

import numpy as np

from .._types import seven_arrays


class Parametriser(ABC):
    """Abstract base class for parametrisation functions that transform input parameters into a form suitable for travel time calculations."""

    n_model_params_per_segment: int

    @abstractmethod
    def to_parameters(self, m: np.ndarray) -> seven_arrays:
        """Transform input model vector into individual Love parameters.

        Parameters
        ----------
        m : ndarray, shape (B, M * P)
            Input model vector, where B is the batch size, M is the number of model segments,
            and P is the number of parameters per segment.

        Returns
        -------
        A : ndarray, shape (B, M)
            Elastic constant C11 = C22
        C : ndarray, shape (B, M)
            Elastic constant C33
        F : ndarray, shape (B, M)
            Elastic constant C13 = C23
        L : ndarray, shape (B, M)
            Elastic constant C44 = C55
        N : ndarray, shape (B, M)
            Elastic constant C66
        eta1 : ndarray, shape (B, M)
            Tilt angle in radians.
        eta2 : ndarray, shape (B, M)
            Azimuthal angle in radians.
        """
        ...

    @abstractmethod
    def apply_jacobian(self, grad: np.ndarray) -> np.ndarray:
        """Convert from dt_dparams to dt_dm.

        Parameters
        ----------
        grad : ndarray, shape (B, M, 7, T)
            Gradient of travel times (T) with respect to the Love parameters and angles.

        Returns
        -------
        grad_dm : ndarray, shape (B, M, P, T)
            Gradient of travel times (T) with respect to the input model vector.
        """
        ...
