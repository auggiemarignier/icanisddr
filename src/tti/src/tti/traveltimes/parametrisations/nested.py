"""Model vector is nested differences of Love parameters and angles in degrees."""

import numpy as np

from ._abc import LinearParametriser
from .radians import TRANSFORMATION as DEGREES_TO_RADIANS_TRANSFORMATION

TRANSFORMATION = np.array(
    [
        [1, 0, 0, 0, 0, 0, 0],
        [1, 1, 0, 0, 0, 0, 0],
        [1, 0, 1, -2, -2, 0, 0],
        [0, 0, 0, 1, 0, 0, 0],
        [0, 0, 0, 1, 1, 0, 0],
        [0, 0, 0, 0, 0, 1, 0],
        [0, 0, 0, 0, 0, 0, 1],
    ],
    dtype=float,
)


class NestedDegreesParametriser(LinearParametriser):
    """Parametriser for nested differences of Love parameters and angles in degrees."""

    n_model_params_per_segment = 7
    transformation = DEGREES_TO_RADIANS_TRANSFORMATION @ TRANSFORMATION
