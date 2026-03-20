"""Model vector is absolute Love parameters."""

import numpy as np

from ._abc import LinearParametriser

TRANSFORMATION = np.array(
    [
        [1, 0, 0, 0, 0, 0, 0],
        [0, 1, 0, 0, 0, 0, 0],
        [0, 0, 1, 0, 0, 0, 0],
        [0, 0, 0, 1, 0, 0, 0],
        [0, 0, 0, 0, 1, 0, 0],
        [0, 0, 0, 0, 0, np.pi / 180.0, 0],
        [0, 0, 0, 0, 0, 0, np.pi / 180.0],
    ]
)


class AbsoluteLoveDegreeAngles(LinearParametriser):
    """Parametriser for absolute Love parameters and angles in degrees."""

    n_model_params_per_segment = 7
    transformation = TRANSFORMATION
