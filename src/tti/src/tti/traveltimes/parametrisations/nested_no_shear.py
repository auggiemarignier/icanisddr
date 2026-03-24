"""Model vector is nested differences of Love parameters without shear and angles in degrees."""

from ._abc import LinearParametriser
from .nested import TRANSFORMATION as NESTED_TRANSFORMATION
from .no_shear import TRANSFORMATION as NO_SHEAR_TRANSFORMATION
from .radians import TRANSFORMATION as DEGREES_TO_RADIANS_TRANSFORMATION


class NestedNoShearDegreesParametriser(LinearParametriser):
    """Parametriser for nested differences of Love parameters (no shear) and angles in degrees."""

    n_model_params_per_segment = 5
    transformation = (
        DEGREES_TO_RADIANS_TRANSFORMATION
        @ NESTED_TRANSFORMATION
        @ NO_SHEAR_TRANSFORMATION
    )
