"""Nested relative parametrisation."""

import numpy as np

from ._abc import RelativeLinearParametriser, _validate_reference
from .nested import TRANSFORMATION as NESTED_TRANSFORMATION
from .radians import TRANSFORMATION as DEGREES_TO_RADIANS_TRANSFORMATION
from .relative import build_relative_transformation_matrix


class NestedRelativeLoveDegreeAngles(RelativeLinearParametriser):
    """Nested relative parametrisation for Love parameters and angles in degrees."""

    n_model_params_per_segment = 7

    @classmethod
    def build_transformation_matrix(cls, ref: np.ndarray) -> np.ndarray:
        return (
            DEGREES_TO_RADIANS_TRANSFORMATION
            @ build_relative_transformation_matrix(ref)
            @ NESTED_TRANSFORMATION
        )

    def _normalise_reference(self, reference_model: np.ndarray | None) -> np.ndarray:
        reference_model = _validate_reference(reference_model)
        return reference_model
