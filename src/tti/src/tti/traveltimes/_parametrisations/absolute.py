"""Model vector is absolute Love parameters."""

import numpy as np

from .._types import seven_arrays
from . import Parametriser

TRANSFORMATION = np.array(
    [
        [1, 0, 0, 0, 0, 0, 0],
        [0, 1, 0, 0, 0, 0, 0],
        [0, 0, 1, 0, 0, 0, 0],
        [0, 0, 0, 1, 0, 0, 0],
        [0, 0, 0, 0, 1, 0, 0],
        [0, 0, 0, 0, 0, np.pi / 180.0, 0],
        [0, 0, 0, 0, 0, 0, np.pi / 180.0],
    ]
)


def _transform_model_vector(m: np.ndarray) -> np.ndarray:
    r"""Transform model vector into individual Love parameters.

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
    arr: ndarray, shape (B, 7, M)
        Array containing the Love parameters and angles in radians, ordered along axis 1 as [A, C, F, L, N, eta1, eta2].
    """
    batch_size = m.shape[0]
    mT = m.reshape(batch_size, 7, -1)
    return TRANSFORMATION @ mT


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
    return TRANSFORMATION.T @ grad


class AbsoluteLoveDegreeAngles(Parametriser):
    """Parametriser for absolute Love parameters and angles in degrees."""

    n_model_params_per_segment = 7

    def to_parameters(self, m: np.ndarray) -> seven_arrays:
        lv = _transform_model_vector(m)
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
        return _jacobian_to_dm(grad)
