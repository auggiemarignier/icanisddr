"""Parameters are fractional perturbations from a reference model for the Love parameters (no shear), and angles in degrees."""

import numpy as np

from .relative import RelativeLoveDegreeAngles


def _build_transformation_matrix(ref: np.ndarray) -> np.ndarray:
    """Build the transformation matrix for the relative parametrisation."""
    T = np.zeros((7, 5))
    T[0, 0] = ref[0]  # A_ref
    T[1, 1] = ref[1]  # C_ref
    T[2, 2] = ref[2]  # F_ref
    T[5, 3] = np.pi / 180.0  # radians
    T[6, 4] = np.pi / 180.0  # radians
    return T


class RelativeNoShearLoveDegreeAngles(RelativeLoveDegreeAngles):
    """Parametriser for relative Love parameters and angles in degrees."""

    n_model_params_per_segment = 5

    def __init__(self, reference_model: np.ndarray | None = None) -> None:
        if reference_model is None:
            reference_model = np.zeros(3)
        elif len(reference_model) != 3:
            raise ValueError("Reference model must have 3 values for A, C, F.")
        self._reference_model = np.concatenate(
            [reference_model, np.zeros(2)]
        )  # add L_ref and N_ref as 0 for unpacking
        self._useful_reference_model = np.concatenate(
            [self._reference_model, np.zeros(2)]
        )  # for angles
        self.transformation = _build_transformation_matrix(self._reference_model)
