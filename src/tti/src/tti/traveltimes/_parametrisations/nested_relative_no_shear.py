"""Nested relative no shear parametrisation."""

import numpy as np

from ._abc import undo_double_degree_conversion
from .nested_no_shear import TRANSFORMATION as NESTED_TRANSFORMATION
from .relative import _build_transformation_matrix
from .relative_no_shear import RelativeNoShearLoveDegreeAngles


class NestedRelativeNoShearLoveDegreeAngles(RelativeNoShearLoveDegreeAngles):
    """Nested relative parametrisation for Love parameters and angles in degrees."""

    def __init__(self, reference_model: np.ndarray | None = None) -> None:
        super().__init__(reference_model=reference_model)
        self.transformation = (
            _build_transformation_matrix(self.reference_model) @ NESTED_TRANSFORMATION
        )

        # Undo the duplicated degrees->radians conversion on the angle rows
        self.transformation = undo_double_degree_conversion(self.transformation)
