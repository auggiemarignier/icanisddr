"""An abstract base class defining the parameteriser objects."""

from abc import ABC, abstractmethod
from collections.abc import Callable

import numpy as np

from .._types import seven_arrays


class BaseParametriser(ABC):
    """Abstract base class for parametrisation functions that transform input parameters into a form suitable for travel time calculations."""

    n_model_params_per_segment: int

    @abstractmethod
    def to_parameters(self, m: np.ndarray) -> seven_arrays:
        """Transform input model vector into individual Love parameters.

        Parameters
        ----------
        m : ndarray, shape (B, M * P)
            Input model vector, where B is the batch size, M is the number of model segments,
            and P is the number of parameters per segment.

        Returns
        -------
        A : ndarray, shape (B, M)
            Elastic constant C11 = C22
        C : ndarray, shape (B, M)
            Elastic constant C33
        F : ndarray, shape (B, M)
            Elastic constant C13 = C23
        L : ndarray, shape (B, M)
            Elastic constant C44 = C55
        N : ndarray, shape (B, M)
            Elastic constant C66
        eta1 : ndarray, shape (B, M)
            Tilt angle in radians.
        eta2 : ndarray, shape (B, M)
            Azimuthal angle in radians.
        """
        ...

    @abstractmethod
    def apply_jacobian(self, grad: np.ndarray) -> np.ndarray:
        """Convert from dt_dparams to dt_dm.

        Parameters
        ----------
        grad : ndarray, shape (B, M, 7, T)
            Gradient of travel times (T) with respect to the Love parameters and angles.

        Returns
        -------
        grad_dm : ndarray, shape (B, M, P, T)
            Gradient of travel times (T) with respect to the input model vector.
        """
        ...


class LinearParametriser(BaseParametriser):
    """A simple linear parametriser that applies a fixed transformation matrix to the input model vector."""

    transformation: np.ndarray

    def to_parameters(self, m: np.ndarray) -> seven_arrays:
        lv = _transform_model_vector(
            m, self.n_model_params_per_segment, lambda x: self.transformation @ x
        )
        A, C, F, L, N, eta1, eta2 = (
            lv[:, 0, :],
            lv[:, 1, :],
            lv[:, 2, :],
            lv[:, 3, :],
            lv[:, 4, :],
            lv[:, 5, :],
            lv[:, 6, :],
        )
        return A, C, F, L, N, eta1, eta2

    def apply_jacobian(self, grad: np.ndarray) -> np.ndarray:
        return _jacobian_to_dm(grad, lambda x: self.transformation.T @ x)


class RelativeParametriser(LinearParametriser):
    """A linear parametriser that applies a fixed transformation matrix to the input model vector, where the first 5 parameters are fractional perturbations from a reference model."""

    def __init__(self, reference_model: np.ndarray | None = None) -> None:
        self._reference_model = self._validate_reference_model(reference_model)
        self._useful_reference_model = np.concatenate(
            [self.reference_model, np.zeros(2)]
        )  # Pad with zeros for the angles so we can easily add the reference model to the transformed model vector in to_parameters.
        self.transformation = self.build_transformation_matrix(self._reference_model)

    def to_parameters(self, m: np.ndarray) -> seven_arrays:
        lv = (
            _transform_model_vector(
                m, self.n_model_params_per_segment, lambda x: self.transformation @ x
            )
            + self._useful_reference_model[None, :, None]
        )
        A, C, F, L, N, eta1, eta2 = (
            lv[:, 0, :],
            lv[:, 1, :],
            lv[:, 2, :],
            lv[:, 3, :],
            lv[:, 4, :],
            lv[:, 5, :],
            lv[:, 6, :],
        )
        return A, C, F, L, N, eta1, eta2

    @classmethod
    @abstractmethod
    def build_transformation_matrix(cls, ref: np.ndarray) -> np.ndarray:
        """Build the transformation matrix for the relative parametrisation.

        Parameters
        ----------
        ref : np.ndarray
            Reference model values for Love parameters.

        Returns
        -------
        np.ndarray
            Transformation matrix for the relative parametrisation (7, 7).
            The first 5 rows scale the Love parameters and angles by the reference model values, and the last 2 rows are identity for the angles.
        """
        ...

    def _validate_reference_model(
        self, reference_model: np.ndarray | None
    ) -> np.ndarray:
        """Validate the reference model to ensure it has the correct shape and values for unpacking.

        Parameters
        ----------
        reference_model : np.ndarray | None
            Reference model values for A, C, F, L, N. If None, defaults to ones.

        Raises
        ------
        ValueError
            If the reference model does not have 5 values for A, C, F, L, N.
        """
        if reference_model is None:
            reference_model = np.ones(5)
        elif len(reference_model) != 5:
            raise ValueError("Reference model must have 5 values for A, C, F, L, N.")
        return reference_model

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


def _transform_model_vector(
    m: np.ndarray, n: int, transform_fn: Callable[[np.ndarray], np.ndarray]
) -> np.ndarray:
    r"""Transform model vector into individual Love parameters.

    Note:
        There is NO checking of the input shape here for performance reasons.

    Parameters
    ----------
    m : ndarray, shape (B, M * n)
        Model parameters: [A, C, F, L, N, eta1, eta2]
        M is the number of model segments (e.g. number of pixels).
        B is the batch size (at least 1).
        n is the number of parameters per segment.
    n : int
        Number of parameters per segment (e.g. 7 for absolute Love parameters and angles).
    transform_fn : Callable[[ndarray], ndarray]
        Function to apply to the reshaped model vector, which performs any necessary transformations (e.g. scaling angles from degrees to radians).

    Returns
    -------
    arr: ndarray, shape (B, 7, M)
        Array containing the Love parameters and angles in radians, ordered along axis 1 as [A, C, F, L, N, eta1, eta2].
    """
    batch_size = m.shape[0]
    mT = m.reshape(batch_size, n, -1)
    return transform_fn(mT)


def _jacobian_to_dm(
    grad: np.ndarray, transform_fn: Callable[[np.ndarray], np.ndarray]
) -> np.ndarray:
    """Convert from dt_dparams to dt_dm by applying a supplied Jacobian transform.

    This helper does not implement any specific parameter or angle scaling itself;
    it simply applies ``transform_fn`` to the input gradient tensor. Any chain-rule
    operations (for example, unit conversions or linear reparameterisations) must
    be implemented inside ``transform_fn``.

    Parameters
    ----------
    grad : ndarray, shape (..., 7, T)
        Gradient of travel times (T) with respect to the parametrised model
        components. By convention this may be ordered as
        ``[dA, dC, dF, dL, dN, deta1, deta2]``, but no ordering is enforced here.
    transform_fn : Callable[[ndarray], ndarray]
        Function that maps ``grad`` to the gradient with respect to the input
        model vector.

    Returns
    -------
    grad_dm : ndarray
        Gradient of travel times (T) with respect to the input model vector,
        as returned by ``transform_fn``.
    """
    return transform_fn(grad)
