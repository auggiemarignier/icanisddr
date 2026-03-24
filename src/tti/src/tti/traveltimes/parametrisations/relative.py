"""Parameters are fractional perturbations from a reference model for the Love parameters, and angles in degrees."""

import numpy as np

from ._abc import RelativeParametriser
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
        The first 5 rows scale the Love parameters by the reference model values, and the last 2 rows remain identity, leaving the angles unchanged.
    """
    T = np.eye(7, 7)
    T[0, 0] = ref[0]  # A_ref
    T[1, 1] = ref[1]  # C_ref
    T[2, 2] = ref[2]  # F_ref
    T[3, 3] = ref[3]  # L_ref
    T[4, 4] = ref[4]  # N_ref
    return T


class RelativeFractionalDegreesParametriser(RelativeParametriser):
    """Parametriser for relative Love parameters and angles in degrees."""

    n_model_params_per_segment = 7

    @classmethod
    def build_transformation_matrix(cls, ref: np.ndarray) -> np.ndarray:
        """Build the transformation matrix for the relative parametrisation.

        Transformation is defined as the product of two transformations:
        1. The transformation from the standard parametrisation to the relative parametrisation (build_relative_transformation_matrix(ref)).
        2. The transformation from angles in degrees to angles in radians (DEGREES_TO_RADIANS_TRANSFORMATION).

        Parameters
        ----------
        ref : np.ndarray
            Reference model values for Love parameters.

        Returns
        -------
        np.ndarray
            Transformation matrix for the relative parametrisation (7, 7).
            The first 5 rows scale the Love parameters by the reference model values; the last 2 rows act as an identity mapping for the angles (which are not scaled by the reference model).
        """
        return DEGREES_TO_RADIANS_TRANSFORMATION @ build_relative_transformation_matrix(
            ref
        )
