"""Parameters are fractional perturbations from a reference model for the Love parameters (no shear), and angles in degrees."""

import numpy as np

from ._abc import RelativeParametriser
from .no_shear import TRANSFORMATION as NO_SHEAR_TRANSFORMATION
from .radians import TRANSFORMATION as DEGREES_TO_RADIANS_TRANSFORMATION
from .relative import build_relative_transformation_matrix


class RelativeFractionalNoShearParametriser(RelativeParametriser):
    """Parametriser for relative Love parameters and angles in degrees (no shear)."""

    n_model_params_per_segment = 5

    @classmethod
    def build_transformation_matrix(cls, ref: np.ndarray) -> np.ndarray:
        """Build the transformation matrix for the relative parametrisation.

        Transformation is defined as the product of three transformations:
        1. Removing shear elements from the nested parametrisation (NO_SHEAR_TRANSFORMATION).
        2. The transformation from the standard parametrisation to the relative parametrisation (build_relative_transformation_matrix(ref)).
        3. The transformation from angles in degrees to angles in radians (DEGREES_TO_RADIANS_TRANSFORMATION).

        Parameters
        ----------
        ref : np.ndarray
            Reference model values for Love parameters.

        Returns
        -------
        np.ndarray
            Transformation matrix for the relative parametrisation (5, 5).
            The first 5 rows scale the Love parameters by the reference model values; the last 2 rows act as an identity mapping for the angles (which are not scaled by the reference model).
        """
        return (
            DEGREES_TO_RADIANS_TRANSFORMATION
            @ build_relative_transformation_matrix(ref)
            @ NO_SHEAR_TRANSFORMATION
        )
