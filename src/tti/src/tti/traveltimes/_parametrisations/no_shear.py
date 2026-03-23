"""Parametrisation of a model with no shear Love parameters, and angles in degrees."""

import numpy as np

from ._abc import LinearParametriser

TRANSFORMATION = np.array(
    [
        [1, 0, 0, 0, 0],
        [0, 1, 0, 0, 0],
        [0, 0, 1, 0, 0],
        [0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0],
        [0, 0, 0, np.pi / 180.0, 0],
        [0, 0, 0, 0, np.pi / 180.0],
    ],
    dtype=float,
)


class NoShearLoveDegreeAngles(LinearParametriser):
    """Parametriser for Love parameters without shear and angles in degrees."""

    n_model_params_per_segment = 5
    transformation = TRANSFORMATION
