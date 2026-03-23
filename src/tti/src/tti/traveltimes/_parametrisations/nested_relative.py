"""Nested relative parametrisation."""

import numpy as np

from .nested import TRANSFORMATION as NESTED_TRANSFORMATION
from .relative import RelativeLoveDegreeAngles


class NestedRelativeLoveDegreeAngles(RelativeLoveDegreeAngles):
    """Nested relative parametrisation for Love parameters and angles in degrees."""

    def __init__(self, reference_model: np.ndarray | None = None) -> None:
        super().__init__(reference_model=reference_model)
        self.transformation = self.transformation @ NESTED_TRANSFORMATION
