"""Model vector is nested differences of Love parameters without shear and angles in degrees."""

import numpy as np

from .._types import seven_arrays
from . import Parametriser

TRANSFORMATION = np.array(
    [
        [1, 0, 0, 0, 0],
        [1, 1, 0, 0, 0],
        [1, 0, 1, 0, 0],
        [0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0],
        [0, 0, 0, np.pi / 180.0, 0],
        [0, 0, 0, 0, np.pi / 180.0],
    ]
)


def _unpack_nested_model_vector_no_shear(m: np.ndarray) -> seven_arrays:
    r"""Unpack nested model vector into individual Love parameters, with L and N fixed at 0.

    Note:
        There is NO checking of the input shape here for performance reasons.

    Parameters
    ----------
    m : ndarray, shape (B, M * 5)
        Nested model parameters: [A, \delta_{CA}, \delta_{F,A}, eta1, eta2]
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
    mT = m.reshape(batch_size, 5, -1)
    lv = TRANSFORMATION @ mT
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


def _jacobian_to_dm(grad: np.ndarray) -> np.ndarray:
    """Convert from dt_dparams to dt_dm for nested models.

    Parameters
    ----------
    grad : ndarray, shape (..., 7, T)
        Gradient of travel times (T) with respect to the nested model parameters.
        Gradient is expected to be ordered as [dA, dC, dF, dL, dN, deta1, deta2] where deta1 and deta2 are derivatives with respect to angles in radians.

    Returns
    -------
    grad_dm : ndarray, shape (..., 5, T)
        Gradient of travel times (T) with respect to the input model vector.
    """
    return TRANSFORMATION.T @ grad


class NestedNoShearLoveDegreeAngles(Parametriser):
    """Parametriser for nested differences of Love parameters (no shear) and angles in degrees."""

    n_model_params_per_segment = 5

    def to_parameters(self, m: np.ndarray) -> seven_arrays:
        return _unpack_nested_model_vector_no_shear(m)

    def apply_jacobian(self, grad: np.ndarray) -> np.ndarray:
        return _jacobian_to_dm(grad)
