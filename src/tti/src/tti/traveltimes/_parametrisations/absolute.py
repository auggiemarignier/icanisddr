"""Model vector is absolute Love parameters."""

import numpy as np

from .._types import seven_arrays
from . import Parametriser


def _unpack_model_vector(m: np.ndarray) -> seven_arrays:
    r"""Unpack model vector into individual Love parameters.

    Note:
        There is NO checking of the input shape here for performance reasons.

    Parameters
    ----------
    m : ndarray, shape (B, M * 7)
        Model parameters: [A, C, F, L, N, eta1, eta2]
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
    return (
        mT[..., 0],
        mT[..., 1],
        mT[..., 2],
        mT[..., 3],
        mT[..., 4],
        np.radians(mT[..., 5]),
        np.radians(mT[..., 6]),
    )


def _jacobian_to_dm(grad: np.ndarray) -> np.ndarray:
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
    dA = grad[..., 0, :]
    dC = grad[..., 1, :]
    dF = grad[..., 2, :]
    dL = grad[..., 3, :]
    dN = grad[..., 4, :]
    deta1 = grad[..., 5, :] * np.pi / 180.0  # chain rule back to degrees
    deta2 = grad[..., 6, :] * np.pi / 180.0  # chain rule back to degrees
    return np.stack(
        [
            dA,
            dC,
            dF,
            dL,
            dN,
            deta1,
            deta2,
        ],
        axis=-2,
    )


class AbsoluteLoveDegreeAngles(Parametriser):
    """Parametriser for absolute Love parameters and angles in degrees."""

    n_model_params_per_segment = 7

    def to_parameters(self, m: np.ndarray) -> seven_arrays:
        return _unpack_model_vector(m)

    def apply_jacobian(self, grad: np.ndarray) -> np.ndarray:
        return _jacobian_to_dm(grad)
