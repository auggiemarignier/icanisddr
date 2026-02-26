"""Modules for constructing and manipulating elastic tensors."""

import numpy as np

from .voigt import isotropic_tensor as itv
from .voigt import tilted_transverse_isotropic_tensor
from .voigt import transverse_isotropic_tensor as titv


def isotropic_tensor(lam: np.ndarray, mu: np.ndarray) -> np.ndarray:
    """
    Construct an isotropic elastic tensor in Voigt notation (6x6 matrix).

    Returns the tensor as a (..., 6, 6) array in Voigt notation. For the full
    4th-order representation, use tti.elastic.fourth.isotropic_tensor() instead.

    Parameters
    ----------
    lam : np.ndarray (n,)
        Lamé constant (λ)
    mu : np.ndarray (n,)
        Shear modulus (μ)

    Returns
    -------
    C : ndarray, shape (..., 6, 6)
        Isotropic elastic tensor in Voigt notation.

    See Also
    --------
    tti.elastic.fourth.isotropic_tensor : 4th-order tensor representation
    """
    return itv(lam, mu)


def transverse_isotropic_tensor(
    A: np.ndarray, C: np.ndarray, F: np.ndarray, L: np.ndarray, N: np.ndarray
) -> np.ndarray:
    """
    Construct a transverse isotropic elastic tensor in Voigt notation (6x6 matrix).

    Returns the tensor as a (..., 6, 6) array in Voigt notation. For the full
    4th-order representation, use tti.elastic.fourth.transverse_isotropic_tensor()
    instead.

    Parameters
    ----------
    A : np.ndarray (...,)
        Elastic constant C11 = C22
    C : np.ndarray (...,)
        Elastic constant C33
    F : np.ndarray (...,)
        Elastic constant C13 = C23
    L : np.ndarray (...,)
        Elastic constant C44 = C55
    N : np.ndarray (...,)
        Elastic constant C66

    Returns
    -------
    C : ndarray, shape (..., 6, 6)
        Transverse isotropic elastic tensor in Voigt notation.

    See Also
    --------
    tti.elastic.fourth.transverse_isotropic_tensor : 4th-order tensor representation
    """
    return titv(A, C, F, L, N)


__all__ = [
    "isotropic_tensor",
    "transverse_isotropic_tensor",
    "tilted_transverse_isotropic_tensor",
]
