"""Contains functions for calculating the gradient of a tilted transversely isotropic (TTI) elastic with respect to the model parameters.

Basically wrappers around tti.elastic.voigt.gradient_D* functions to account for nesting and options regarding the shear Love parameters.

Similar idea to the tti.traveltimes._unpackings module.
"""

from typing import Protocol

import numpy as np

from tti.elastic.voigt import (
    gradient_D,
    gradient_D_wrt_A,
    gradient_D_wrt_C,
    gradient_D_wrt_eta1,
    gradient_D_wrt_eta2,
    gradient_D_wrt_F,
    gradient_D_wrt_L,
    gradient_D_wrt_N,
)


class _GradientD(Protocol):
    def __call__(
        self,
        A: np.ndarray,
        C: np.ndarray,
        F: np.ndarray,
        L: np.ndarray,
        N: np.ndarray,
        eta1: np.ndarray,
        eta2: np.ndarray,
    ) -> np.ndarray: ...


def _wrt_model_parameters(
    A: np.ndarray,
    C: np.ndarray,
    F: np.ndarray,
    L: np.ndarray,
    N: np.ndarray,
    eta1: np.ndarray,
    eta2: np.ndarray,
) -> np.ndarray:
    """Gradient of D with respect to A, C, F, L, N, eta1, eta2.

    Parameters
    ----------
    A : np.ndarray (...,)
        Elastic constant C11 = C22.
    C : np.ndarray (...,)
        Elastic constant C33.
    F : np.ndarray (...,)
        Elastic constant C13 = C23.
    L : np.ndarray (...,)
        Elastic constant C44 = C55.
    N : np.ndarray (...,)
        Elastic constant C66.
    eta1 : np.ndarray (...,)
        Tilt angle around the z-axis (in radians)
    eta2 : np.ndarray (...,)
        Azimuthal angle around the y-axis (in radians)

    Returns
    -------
    dD : np.ndarray (..., 7, 6, 6)
        Gradient of D with respect to A, C, F, L, N, eta1, eta2 in that order.
    """
    return gradient_D(A, C, F, L, N, eta1, eta2)


def _wrt_model_parameters_no_shear(
    A: np.ndarray,
    C: np.ndarray,
    F: np.ndarray,
    L: np.ndarray,
    N: np.ndarray,
    eta1: np.ndarray,
    eta2: np.ndarray,
) -> np.ndarray:
    """Gradient of D with respect to A, C, F, eta1, eta2.

    Parameters
    ----------
    A : np.ndarray (...,)
        Elastic constant C11 = C22.
    C : np.ndarray (...,)
        Elastic constant C33.
    F : np.ndarray (...,)
        Elastic constant C13 = C23.
    L : np.ndarray (...,)
        Elastic constant C44 = C55.
    N : np.ndarray (...,)
        Elastic constant C66.
    eta1 : np.ndarray (...,)
        Tilt angle around the z-axis (in radians)
    eta2 : np.ndarray (...,)
        Azimuthal angle around the y-axis (in radians)

    Returns
    -------
    dD : np.ndarray (..., 5, 6, 6)
        Gradient of D with respect to A, C, F, eta1, eta2 in that order.
    """
    return np.stack(
        [
            gradient_D_wrt_A(A, C, F, L, N, eta1, eta2),
            gradient_D_wrt_C(A, C, F, L, N, eta1, eta2),
            gradient_D_wrt_F(A, C, F, L, N, eta1, eta2),
            gradient_D_wrt_eta1(A, C, F, L, N, eta1, eta2),
            gradient_D_wrt_eta2(A, C, F, L, N, eta1, eta2),
        ],
        axis=-3,
    )


def _wrt_model_parameters_no_N(
    A: np.ndarray,
    C: np.ndarray,
    F: np.ndarray,
    L: np.ndarray,
    N: np.ndarray,
    eta1: np.ndarray,
    eta2: np.ndarray,
) -> np.ndarray:
    """Gradient of D with respect to A, C, F, L, eta1, eta2 (no N).

    Parameters
    ----------
    A : np.ndarray (...,)
        Elastic constant C11 = C22.
    C : np.ndarray (...,)
        Elastic constant C33.
    F : np.ndarray (...,)
        Elastic constant C13 = C23.
    L : np.ndarray (...,)
        Elastic constant C44 = C55.
    N : np.ndarray (...,)
        Elastic constant C66.
    eta1 : np.ndarray (...,)
        Tilt angle around the z-axis (in radians)
    eta2 : np.ndarray (...,)
        Azimuthal angle around the y-axis (in radians)

    Returns
    -------
    dD : np.ndarray (..., 6, 6, 6)
        Gradient of D with respect to A, C, F, L, eta1, eta2 in that order.
    """
    return np.stack(
        [
            gradient_D_wrt_A(A, C, F, L, N, eta1, eta2),
            gradient_D_wrt_C(A, C, F, L, N, eta1, eta2),
            gradient_D_wrt_F(A, C, F, L, N, eta1, eta2),
            gradient_D_wrt_L(A, C, F, L, N, eta1, eta2),
            gradient_D_wrt_eta1(A, C, F, L, N, eta1, eta2),
            gradient_D_wrt_eta2(A, C, F, L, N, eta1, eta2),
        ],
        axis=-3,
    )


def _wrt_nested_model_parameters(
    A: np.ndarray,
    C: np.ndarray,
    F: np.ndarray,
    L: np.ndarray,
    N: np.ndarray,
    eta1: np.ndarray,
    eta2: np.ndarray,
) -> np.ndarray:
    """Gradient of D with respect to A, C-A, F-A+2N, L, N-L, eta1, eta2 for nested models.

    Parameters
    ----------
    A : np.ndarray (...,)
        Elastic constant C11 = C22.
    C : np.ndarray (...,)
        Elastic constant C33.
    F : np.ndarray (...,)
        Elastic constant C13 = C23.
    L : np.ndarray (...,)
        Elastic constant C44 = C55.
    N : np.ndarray (...,)
        Elastic constant C66.
    eta1 : np.ndarray (...,)
        Tilt angle around the z-axis (in radians)
    eta2 : np.ndarray (...,)
        Azimuthal angle around the y-axis (in radians)

    Returns
    -------
    dD : np.ndarray (..., 7, 6, 6)
        Gradient of D with respect to A, C-A, F-A+2N, L, N-L, eta1, eta2 in that order.
    """
    dDdA = gradient_D_wrt_A(A, C, F, L, N, eta1, eta2)
    dDdC = gradient_D_wrt_C(A, C, F, L, N, eta1, eta2)
    dDdF = gradient_D_wrt_F(A, C, F, L, N, eta1, eta2)
    dDdL = gradient_D_wrt_L(A, C, F, L, N, eta1, eta2)
    dDdN = gradient_D_wrt_N(A, C, F, L, N, eta1, eta2)
    dDdeta1 = gradient_D_wrt_eta1(A, C, F, L, N, eta1, eta2)
    dDdeta2 = gradient_D_wrt_eta2(A, C, F, L, N, eta1, eta2)
    return np.stack(
        [
            dDdA + dDdC + dDdF,
            dDdC,
            dDdF,
            dDdL + dDdN - 2 * dDdF,
            dDdN - 2 * dDdF,
            dDdeta1,
            dDdeta2,
        ],
        axis=-3,
    )


def _wrt_nested_model_parameters_no_N(
    A: np.ndarray,
    C: np.ndarray,
    F: np.ndarray,
    L: np.ndarray,
    N: np.ndarray,
    eta1: np.ndarray,
    eta2: np.ndarray,
) -> np.ndarray:
    """Gradient of D with respect to A, C-A, F-A+2N, L, eta1, eta2 for nested models.

    Parameters
    ----------
    A : np.ndarray (...,)
        Elastic constant C11 = C22.
    C : np.ndarray (...,)
        Elastic constant C33.
    F : np.ndarray (...,)
        Elastic constant C13 = C23.
    L : np.ndarray (...,)
        Elastic constant C44 = C55.
    N : np.ndarray (...,)
        Elastic constant C66.
    eta1 : np.ndarray (...,)
        Tilt angle around the z-axis (in radians)
    eta2 : np.ndarray (...,)
        Azimuthal angle around the y-axis (in radians)

    Returns
    -------
    dD : np.ndarray (..., 6, 6, 6)
        Gradient of D with respect to A, C-A, F-A+2N, L, eta1, eta2 in that order.
    """
    dDdA = gradient_D_wrt_A(A, C, F, L, N, eta1, eta2)
    dDdC = gradient_D_wrt_C(A, C, F, L, N, eta1, eta2)
    dDdF = gradient_D_wrt_F(A, C, F, L, N, eta1, eta2)
    dDdL = gradient_D_wrt_L(A, C, F, L, N, eta1, eta2)
    dDdeta1 = gradient_D_wrt_eta1(A, C, F, L, N, eta1, eta2)
    dDdeta2 = gradient_D_wrt_eta2(A, C, F, L, N, eta1, eta2)
    return np.stack(
        [
            dDdA + dDdC + dDdF,
            dDdC,
            dDdF,
            dDdL - 2 * dDdF,
            dDdeta1,
            dDdeta2,
        ],
        axis=-3,
    )


def _wrt_nested_model_parameters_no_shear(
    A: np.ndarray,
    C: np.ndarray,
    F: np.ndarray,
    L: np.ndarray,
    N: np.ndarray,
    eta1: np.ndarray,
    eta2: np.ndarray,
) -> np.ndarray:
    """Gradient of D with respect to A, C-A, F-A+2N, eta1, eta2 for nested models.

    Parameters
    ----------
    A : np.ndarray (...,)
        Elastic constant C11 = C22.
    C : np.ndarray (...,)
        Elastic constant C33.
    F : np.ndarray (...,)
        Elastic constant C13 = C23.
    L : np.ndarray (...,)
        Elastic constant C44 = C55.
    N : np.ndarray (...,)
        Elastic constant C66.
    eta1 : np.ndarray (...,)
        Tilt angle around the z-axis (in radians)
    eta2 : np.ndarray (...,)
        Azimuthal angle around the y-axis (in radians)

    Returns
    -------
    dD : np.ndarray (..., 5, 6, 6)
        Gradient of D with respect to A, C-A, F-A+2N, eta1, eta2 in that order.
    """
    dDdA = gradient_D_wrt_A(A, C, F, L, N, eta1, eta2)
    dDdC = gradient_D_wrt_C(A, C, F, L, N, eta1, eta2)
    dDdF = gradient_D_wrt_F(A, C, F, L, N, eta1, eta2)
    dDdeta1 = gradient_D_wrt_eta1(A, C, F, L, N, eta1, eta2)
    dDdeta2 = gradient_D_wrt_eta2(A, C, F, L, N, eta1, eta2)
    return np.stack(
        [
            dDdA + dDdC + dDdF,
            dDdC,
            dDdF,
            dDdeta1,
            dDdeta2,
        ],
        axis=-3,
    )


# Mapping of gradient functions based on input format.
# The keys are (nested, (include_shear, include_N)).
# If include_shear is False, L and N are fixed at 0.
# If include_N is False, N is fixed at 0.
_gradient_D_functions: dict[bool, dict[tuple[bool, bool], _GradientD]] = {
    True: {
        (True, True): _wrt_nested_model_parameters,
        (True, False): _wrt_nested_model_parameters_no_N,
        # (False, True): This would be the case where we include N but not L. Not supported - KeyError will be raised if this combination is used.,
        (False, False): _wrt_nested_model_parameters_no_shear,
    },
    False: {
        (True, True): _wrt_model_parameters,
        (True, False): _wrt_model_parameters_no_N,
        # (False, True): This would be the case where we include N but not L. Not supported - KeyError will be raised if this combination is used.,
        (False, False): _wrt_model_parameters_no_shear,
    },
}
