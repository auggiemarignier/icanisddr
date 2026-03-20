"""Parameters are fractional perturbations from a reference model for the Love parameters, and angles in degrees."""

import numpy as np

from .._types import seven_arrays
from . import Parametriser


def _build_transformation_matrix(ref: np.ndarray) -> np.ndarray:
    """Build the transformation matrix for the relative parametrisation."""
    T = np.zeros((7, 7))
    T[0, 0] = ref[0]  # A_ref
    T[1, 1] = ref[1]  # C_ref
    T[2, 2] = ref[2]  # F_ref
    T[3, 3] = ref[3]  # L_ref
    T[4, 4] = ref[4]  # N_ref
    T[5, 5] = np.pi / 180.0  # radians
    T[6, 6] = np.pi / 180.0  # radians
    return T


def _unpack_relative_model_vector(m: np.ndarray, ref: np.ndarray) -> seven_arrays:
    """Unpack the model vector for the relative parametrisation.

    Parameters
    ----------
    m : ndarray, shape (B, M * 7)
        Model vector containing fractional perturbations for the Love parameters and angles in degrees.
        Expected order is [dA/A_ref, dC/C_ref, dF/F_ref, dL/L_ref, dN/N_ref, eta1_deg, eta2_deg].
        The first five parameters are fractional perturbations from the reference model.
        The last two parameters are angles in degrees.

    ref : ndarray, shape (5,)
        Reference model values for the Love parameters [A_ref, C_ref, F_ref, L_ref, N_ref].

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
    transformation = _build_transformation_matrix(ref)

    batch_size = m.shape[0]
    mT = m.reshape(batch_size, 7, -1)

    ref = np.concatenate(
        [ref, np.array([0.0, 0.0])]
    )  # add dummy values for angles for easy matrix operations
    lv = transformation @ mT + ref[None, :, None]
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


def _jacobian_to_dm(grad: np.ndarray, ref: np.ndarray) -> np.ndarray:
    """Convert from dt_dparams to dt_dm.

    Chain rule to scale the Love parameter derivatives to the reference model.
    Chain rule to scale the angle derivatives to be wrt angles in degrees.

    Parameters
    ----------
    grad : ndarray, shape (..., 7, T)
        Gradient of travel times (T) with respect to the Love parameters and angles.
        Gradient is expected to be ordered as [dA, dC, dF, dL, dN, deta1, deta2].
        deta1 and deta2 are with respect to the angles in radians.
    ref : ndarray, shape (5,)
        Reference model values for the Love parameters [A_ref, C_ref, F_ref, L_ref, N_ref].

    Returns
    -------
    grad_dm : ndarray, shape (..., 7, T)
        Gradient of travel times (T) with respect to the input model vector.
    """
    transformation = _build_transformation_matrix(ref)
    return transformation.T @ grad


class RelativeLoveDegreeAngles(Parametriser):
    """Parametriser for relative Love parameters and angles in degrees."""

    n_model_params_per_segment = 7

    def __init__(self, reference_model: np.ndarray | None = None) -> None:
        if reference_model is None:
            reference_model = np.array([0.0, 0.0, 0.0, 0.0, 0.0])
        elif len(reference_model) != 5:
            raise ValueError("Reference model must have 5 values for A, C, F, L, N.")
        self._reference_model = reference_model

    def to_parameters(self, m: np.ndarray) -> seven_arrays:
        return _unpack_relative_model_vector(m, self.reference_model)

    def apply_jacobian(self, grad: np.ndarray) -> np.ndarray:
        return _jacobian_to_dm(grad, self.reference_model)

    @property
    def reference_model(self) -> np.ndarray:
        """Reference model values for A, C, F, L, N."""
        return self._reference_model
