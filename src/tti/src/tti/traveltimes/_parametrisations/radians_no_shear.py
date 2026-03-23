"""Model vector is absolute Love parameters without shear and angles in degrees."""

from ._abc import LinearParametriser
from .no_shear import TRANSFORMATION as NO_SHEAR_TRANSFORMATION
from .radians import TRANSFORMATION as DEGREES_TO_RADIANS_TRANSFORMATION


class RadiansNoShearLoveDegreeAngles(LinearParametriser):
    """Parametriser for absolute Love parameters (no shear) and angles in degrees."""

    n_model_params_per_segment = 5
    transformation = DEGREES_TO_RADIANS_TRANSFORMATION @ NO_SHEAR_TRANSFORMATION
