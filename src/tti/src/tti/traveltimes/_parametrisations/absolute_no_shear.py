"""Model vector is absolute Love parameters without shear and angles in degrees."""

import numpy as np

from .._types import seven_arrays
from . import Parametriser


def _unpack_model_vector_no_shear(m: np.ndarray) -> seven_arrays:
    r"""Unpack model vector into individual Love parameters, with L and N fixed at 0.

    Note:
        There is NO checking of the input shape here for performance reasons.

    Parameters
    ----------
    m : ndarray, shape (B, M * 5)
        Model parameters: [A, C, F, eta1, eta2]
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
        Fixed at 0.
    N : ndarray, shape (B, M)
        Elastic constant C66
        Fixed at 0.
    eta1 : ndarray, shape (B, M)
        Tilt angle in radians.
    eta2 : ndarray, shape (B, M)
        Azimuthal angle in radians.
    """
    batch_size = m.shape[0]
    mT = m.reshape(batch_size, -1, 5).copy()
    zeros = np.zeros_like(mT[..., 0])
    return (
        mT[..., 0],
        mT[..., 1],
        mT[..., 2],
        zeros,
        zeros,
        np.radians(mT[..., 3]),
        np.radians(mT[..., 4]),
    )


def _jacobian_to_dm(grad: np.ndarray) -> np.ndarray:
    """Convert from dt_dparams to dt_dm.

    Chain rule to scale the angle derivatives to be wrt angles in degrees.
    The gradients with respect to the Love parameters (A, C, F) are unchanged.

    Parameters
    ----------
    grad : ndarray, shape (..., 7, T)
        Gradient of travel times (T) with respect to the Love parameters and angles.
        Gradient is expected to be ordered as [dA, dC, dF, dL, dN, deta1, deta2].
        deta1 and deta2 are with respect to the angles in radians.

    Returns
    -------
    grad_dm : ndarray, shape (..., 5, T)
        Gradient of travel times (T) with respect to the input model vector.
    """
    dA = grad[..., 0, :]
    dC = grad[..., 1, :]
    dF = grad[..., 2, :]
    _ = grad[..., 3, :]  # dL, should be zero
    _ = grad[..., 4, :]  # dN, should be zero
    deta1 = grad[..., 5, :] * np.pi / 180.0  # chain rule back to degrees
    deta2 = grad[..., 6, :] * np.pi / 180.0  # chain rule back to degrees
    return np.stack(
        [
            dA,
            dC,
            dF,
            deta1,
            deta2,
        ],
        axis=-2,
    )


class AbsoluteNoShearLoveDegreeAngles(Parametriser):
    """Parametriser for absolute Love parameters (no shear) and angles in degrees."""

    n_model_params_per_segment = 5

    def to_parameters(self, m: np.ndarray) -> seven_arrays:
        return _unpack_model_vector_no_shear(m)

    def apply_jacobian(self, grad: np.ndarray) -> np.ndarray:
        return _jacobian_to_dm(grad)
