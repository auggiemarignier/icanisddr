"""An abstract base class defining the parameteriser objects."""

from abc import ABC, abstractmethod
from collections.abc import Callable

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


class LinearParametriser(Parametriser):
    """A simple linear parametriser that applies a fixed transformation matrix to the input model vector."""

    transformation: np.ndarray

    def to_parameters(self, m: np.ndarray) -> seven_arrays:
        lv = _transform_model_vector(
            m, self.n_model_params_per_segment, lambda x: self.transformation @ x
        )
        A, C, F, L, N, eta1, eta2 = (
            lv[:, 0, :],
            lv[:, 1, :],
            lv[:, 2, :],
            lv[:, 3, :],
            lv[:, 4, :],
            lv[:, 5, :],
            lv[:, 6, :],
        )
        return A, C, F, L, N, eta1, eta2

    def apply_jacobian(self, grad: np.ndarray) -> np.ndarray:
        return _jacobian_to_dm(grad, lambda x: self.transformation.T @ x)


def _transform_model_vector(
    m: np.ndarray, n: int, transform_fn: Callable[[np.ndarray], np.ndarray]
) -> np.ndarray:
    r"""Transform model vector into individual Love parameters.

    Note:
        There is NO checking of the input shape here for performance reasons.

    Parameters
    ----------
    m : ndarray, shape (B, M * n)
        Model parameters: [A, C, F, L, N, eta1, eta2]
        M is the number of model segments (e.g. number of pixels).
        B is the batch size (at least 1).
        n is the number of parameters per segment.
    n : int
        Number of parameters per segment (e.g. 7 for absolute Love parameters and angles).
    transform_fn : Callable[[ndarray], ndarray]
        Function to apply to the reshaped model vector, which performs any necessary transformations (e.g. scaling angles from degrees to radians).

    Returns
    -------
    arr: ndarray, shape (B, 7, M)
        Array containing the Love parameters and angles in radians, ordered along axis 1 as [A, C, F, L, N, eta1, eta2].
    """
    batch_size = m.shape[0]
    mT = m.reshape(batch_size, n, -1)
    return transform_fn(mT)


def _jacobian_to_dm(
    grad: np.ndarray, transform_fn: Callable[[np.ndarray], np.ndarray]
) -> np.ndarray:
    """Convert from dt_dparams to dt_dm.

    Chain rule to scale the angle derivatives to be wrt angles in degrees.
    The gradients with respect to the Love parameters (A, C, F, L, N) are unchanged.

    Parameters
    ----------
    grad : ndarray, shape (..., 7, T)
        Gradient of travel times (T) with respect to the Love parameters and angles.
        Gradient is expected to be ordered as [dA, dC, dF, dL, dN, deta1, deta2].
        deta1 and deta2 are with respect to the angles in radians.

    Returns
    -------
    grad_dm : ndarray, shape (..., 7, T)
        Gradient of travel times (T) with respect to the input model vector.
    """
    return transform_fn(grad)


def undo_double_degree_conversion(transformation: np.ndarray) -> np.ndarray:
    """Undo an extra degrees->radians conversion applied to the last two rows.

    Some parametrisation combinations multiply transformation matrices which causes
    the degrees->radians scaling (pi/180) for the angle rows to be applied twice.
    This helper multiplies the last two rows by 180/pi to revert that extra scaling
    and returns a new array.

    Parameters
    ----------
    transformation : ndarray
        Transformation matrix whose last two rows correspond to angles in radians.

    Returns
    -------
    ndarray
        A copy of `transformation` where the last two rows have been scaled by 180/pi.
    """
    t = transformation.copy()
    t[-2:, :] *= 180.0 / np.pi
    return t
