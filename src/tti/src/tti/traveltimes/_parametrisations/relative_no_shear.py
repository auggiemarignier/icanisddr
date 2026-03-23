"""Parameters are fractional perturbations from a reference model for the Love parameters (no shear), and angles in degrees."""

import numpy as np

from .no_shear import TRANSFORMATION as NO_SHEAR_TRANSFORMATION
from .relative import RelativeLoveDegreeAngles


class RelativeNoShearLoveDegreeAngles(RelativeLoveDegreeAngles):
    """Parametriser for relative Love parameters and angles in degrees."""

    n_model_params_per_segment = 5

    def __init__(self, reference_model: np.ndarray | None = None) -> None:
        super().__init__(reference_model=reference_model)
        self.transformation = self.transformation @ NO_SHEAR_TRANSFORMATION

    def _normalise_reference(self, reference_model: np.ndarray | None) -> np.ndarray:
        if reference_model is None:
            reference_model = np.zeros(3)
        elif len(reference_model) != 3:
            raise ValueError("Reference model must have 3 values for A, C, F.")
        return np.concatenate(
            [reference_model, np.zeros(2)]
        )  # add L_ref and N_ref as 0 for unpacking
