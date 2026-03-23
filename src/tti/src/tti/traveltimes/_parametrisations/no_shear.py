"""Parametrisation of a model with no shear Love parameters, and angles in radians."""

import numpy as np

from ._abc import LinearParametriser

TRANSFORMATION = np.array(
    [
        [1, 0, 0, 0, 0],
        [0, 1, 0, 0, 0],
        [0, 0, 1, 0, 0],
        [0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0],
        [0, 0, 0, 1, 0],
        [0, 0, 0, 0, 1],
    ],
    dtype=float,
)


class NoShearLoveRadianAngles(LinearParametriser):
    """Parametriser for Love parameters without shear and angles in radians."""

    n_model_params_per_segment = 5
    transformation = TRANSFORMATION
