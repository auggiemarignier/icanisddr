"""Nested relative no shear parametrisation."""

import numpy as np

from ._abc import RelativeParametriser
from .nested import TRANSFORMATION as NESTED_TRANSFORMATION
from .no_shear import TRANSFORMATION as NO_SHEAR_TRANSFORMATION
from .radians import TRANSFORMATION as DEGREES_TO_RADIANS_TRANSFORMATION
from .relative import build_relative_transformation_matrix


class NestedRelativeFractionalNoShearParametriser(RelativeParametriser):
    """Nested relative parametrisation for Love parameters and angles in degrees (no shear)."""

    n_model_params_per_segment = 5

    @classmethod
    def build_transformation_matrix(cls, ref: np.ndarray) -> np.ndarray:
        return (
            DEGREES_TO_RADIANS_TRANSFORMATION
            @ build_relative_transformation_matrix(ref)
            @ NESTED_TRANSFORMATION
            @ NO_SHEAR_TRANSFORMATION
        )
