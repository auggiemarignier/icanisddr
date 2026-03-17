"""Model vector is nested differences of Love parameters and angles in degrees."""

import numpy as np

from .._types import seven_arrays
from . import Parametriser


def _unpack_nested_model_vector(m: np.ndarray) -> seven_arrays:
    r"""Unpack nested model vector into individual Love parameters.

    Note:
        There is NO checking of the input shape here for performance reasons.

    Parameters
    ----------
    m : ndarray, shape (B, M * 7)
        Nested model parameters: [A, \delta_{CA}, \delta_{F,A+2N}, L, \delta_{NL}, eta1, eta2]
        M is the number of model segments (e.g. number of pixels).
        B is the batch size (at least 1).

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
    batch_size = m.shape[0]
    mT = m.reshape(batch_size, -1, 7).copy()
    A = mT[..., 0]
    C = mT[..., 1] + A
    L = mT[..., 3]
    N = mT[..., 4] + L
    F = mT[..., 2] + A - 2 * N
    eta1 = np.radians(mT[..., 5])
    eta2 = np.radians(mT[..., 6])
    return (A, C, F, L, N, eta1, eta2)


def _jacobian_to_dm(grad: np.ndarray) -> np.ndarray:
    """Convert from dt_dparams to dt_dm for nested models.

    Parameters
    ----------
    grad : ndarray, shape (..., 7, T)
        Gradient of travel times (T) with respect to the nested model parameters.
        Gradient is expected to be ordered as [dA, dC, dF, dL, dN, deta1, deta2] where deta1 and deta2 are derivatives with respect to angles in radians.

    Returns
    -------
    grad_dm : ndarray, shape (..., 7, T)
        Gradient of travel times (T) with respect to the input model vector.
    """
    dA = grad[..., 0, :]
    dC = grad[..., 1, :]
    dF = grad[..., 2, :]
    dL = grad[..., 3, :]
    dN = grad[..., 4, :]
    deta1 = grad[..., 5, :] * np.pi / 180.0  # chain rule back to degrees
    deta2 = grad[..., 6, :] * np.pi / 180.0  # chain rule back to degrees
    return np.stack(
        [
            dA + dC + dF,
            dC,
            dF,
            dL + dN - 2 * dF,
            dN - 2 * dF,
            deta1,
            deta2,
        ],
        axis=-2,
    )


class NestedLoveDegreeAngles(Parametriser):
    """Parametriser for nested differences of Love parameters and angles in degrees."""

    n_model_params_per_segment = 7

    def to_parameters(self, m: np.ndarray) -> seven_arrays:
        return _unpack_nested_model_vector(m)

    def apply_jacobian(self, grad: np.ndarray) -> np.ndarray:
        return _jacobian_to_dm(grad)
