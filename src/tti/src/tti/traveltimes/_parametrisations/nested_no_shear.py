"""Model vector is nested differences of Love parameters without shear and angles in degrees."""

from ._abc import LinearParametriser, undo_double_degree_conversion
from .nested import TRANSFORMATION as NESTED_TRANSFORMATION
from .no_shear import TRANSFORMATION as NO_SHEAR_TRANSFORMATION

TRANSFORMATION = undo_double_degree_conversion(
    NESTED_TRANSFORMATION @ NO_SHEAR_TRANSFORMATION
)


class NestedNoShearLoveDegreeAngles(LinearParametriser):
    """Parametriser for nested differences of Love parameters (no shear) and angles in degrees."""

    n_model_params_per_segment = 5
    transformation = TRANSFORMATION
