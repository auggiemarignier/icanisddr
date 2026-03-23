"""Parameters are fractional perturbations from a reference model for the Love parameters, and angles in radians."""

import numpy as np

from ._abc import RelativeLinearParametriser, _validate_reference
from .radians import TRANSFORMATION as DEGREES_TO_RADIANS_TRANSFORMATION


def build_relative_transformation_matrix(ref: np.ndarray) -> np.ndarray:
    """Build the transformation matrix for the relative parametrisation.

    Parameters
    ----------
    ref : np.ndarray
        Reference model values for A, C, F, L, N.

    Returns
    -------
    np.ndarray
        Transformation matrix for the relative parametrisation (7, 7).
        The first 5 rows scale the Love parameters and angles by the reference model values, and the last 2 rows are identity for the angles.
    """
    T = np.eye(7, 7)
    T[0, 0] = ref[0]  # A_ref
    T[1, 1] = ref[1]  # C_ref
    T[2, 2] = ref[2]  # F_ref
    T[3, 3] = ref[3]  # L_ref
    T[4, 4] = ref[4]  # N_ref
    return T


class RelativeLoveDegreeAngles(RelativeLinearParametriser):
    """Parametriser for relative Love parameters and angles in degrees."""

    n_model_params_per_segment = 7

    @staticmethod
    def build_transformation_matrix(ref: np.ndarray) -> np.ndarray:
        return DEGREES_TO_RADIANS_TRANSFORMATION @ build_relative_transformation_matrix(
            ref
        )

    def _normalise_reference(self, reference_model: np.ndarray | None) -> np.ndarray:
        reference_model = _validate_reference(reference_model)
        return reference_model
