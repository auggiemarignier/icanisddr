"""Parameters are fractional perturbations from a reference model for the Love parameters, and angles in degrees."""

import numpy as np

from .._types import seven_arrays
from ._abc import LinearParametriser


def _build_transformation_matrix(ref: np.ndarray) -> np.ndarray:
    """Build the transformation matrix for the relative parametrisation."""
    T = np.zeros((7, 7))
    T[0, 0] = ref[0]  # A_ref
    T[1, 1] = ref[1]  # C_ref
    T[2, 2] = ref[2]  # F_ref
    T[3, 3] = ref[3]  # L_ref
    T[4, 4] = ref[4]  # N_ref
    T[5, 5] = np.pi / 180.0  # radians
    T[6, 6] = np.pi / 180.0  # radians
    return T


class RelativeLoveDegreeAngles(LinearParametriser):
    """Parametriser for relative Love parameters and angles in degrees."""

    n_model_params_per_segment = 7

    def __init__(self, reference_model: np.ndarray | None = None) -> None:
        self._reference_model = self._normalise_reference(reference_model)
        self.transformation = _build_transformation_matrix(self._reference_model)

    def to_parameters(self, m: np.ndarray) -> seven_arrays:
        A, C, F, L, N, eta1, eta2 = super().to_parameters(m)
        return (
            A + self.ref_A,
            C + self.ref_C,
            F + self.ref_F,
            L + self.ref_L,
            N + self.ref_N,
            eta1,
            eta2,
        )

    @property
    def reference_model(self) -> np.ndarray:
        """Reference model values for A, C, F, L, N."""
        return self._reference_model

    @property
    def ref_A(self) -> float:
        """Reference model value for A."""
        return self.reference_model[0]

    @property
    def ref_C(self) -> float:
        """Reference model value for C."""
        return self.reference_model[1]

    @property
    def ref_F(self) -> float:
        """Reference model value for F."""
        return self.reference_model[2]

    @property
    def ref_L(self) -> float:
        """Reference model value for L."""
        return self.reference_model[3]

    @property
    def ref_N(self) -> float:
        """Reference model value for N."""
        return self.reference_model[4]

    def _normalise_reference(self, reference_model: np.ndarray | None) -> np.ndarray:
        if reference_model is None:
            reference_model = np.zeros(5)
        elif len(reference_model) != 5:
            raise ValueError("Reference model must have 5 values for A, C, F, L, N.")
        return reference_model
