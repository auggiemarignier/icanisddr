"""Parameters are fractional perturbations from a reference model for the Love parameters (no shear), and angles in degrees."""

import numpy as np

from ._abc import RelativeLinearParametriser, _validate_reference_no_shear
from .no_shear import TRANSFORMATION as NO_SHEAR_TRANSFORMATION
from .radians import TRANSFORMATION as DEGREES_TO_RADIANS_TRANSFORMATION
from .relative import build_relative_transformation_matrix


class RelativeNoShearLoveDegreeAngles(RelativeLinearParametriser):
    """Parametriser for relative Love parameters and angles in degrees."""

    n_model_params_per_segment = 5

    @classmethod
    def build_transformation_matrix(cls, ref: np.ndarray) -> np.ndarray:
        return (
            DEGREES_TO_RADIANS_TRANSFORMATION
            @ build_relative_transformation_matrix(ref)
            @ NO_SHEAR_TRANSFORMATION
        )

    def _normalise_reference(self, reference_model: np.ndarray | None) -> np.ndarray:
        reference_model = _validate_reference_no_shear(reference_model)
        return np.concatenate(
            [reference_model, np.zeros(2)]
        )  # add L_ref and N_ref as 0 for unpacking
