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
        return (
            DEGREES_TO_RADIANS_TRANSFORMATION
            @ build_relative_transformation_matrix(ref)
            @ NO_SHEAR_TRANSFORMATION
        )
