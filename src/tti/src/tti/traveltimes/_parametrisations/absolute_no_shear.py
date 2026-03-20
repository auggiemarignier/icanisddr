"""Model vector is absolute Love parameters without shear and angles in degrees."""

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
    ]
)


class AbsoluteNoShearLoveDegreeAngles(LinearParametriser):
    """Parametriser for absolute Love parameters (no shear) and angles in degrees."""

    n_model_params_per_segment = 5
    transformation = TRANSFORMATION
