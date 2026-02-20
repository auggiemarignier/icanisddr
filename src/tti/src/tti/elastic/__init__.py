"""Modules for constructing and manipulating elastic tensors."""

import numpy as np

from .voigt import isotropic_tensor as itv
from .voigt import tilted_transverse_isotropic_tensor
from .voigt import transverse_isotropic_tensor as titv


def isotropic_tensor(lam: np.ndarray, mu: np.ndarray) -> np.ndarray:
    """
    Construct an isotropic elastic tensor (4th-order representation).

    This returns the tensor as a (n, 3, 3, 3, 3) array, suitable for direct tensor operations
    like rotations and contractions. For 6x6 Voigt notation, use
    isotropic_tensor_voigt() instead.

    Parameters
    ----------
    lam : np.ndarray (n,)
        Lamé constant (λ)
    mu : np.ndarray (n,)
        Shear modulus (μ)

    Returns
    -------
    C : ndarray, shape (n, 3, 3, 3, 3)
        Isotropic elastic tensor in 4th-order form.

    See Also
    --------
    tti.elastic.voigt.isotropic_tensor : Voigt notation (6x6) version
    tti.elastic.fourth.isotropic_tensor : Direct alias to this implementation
    """
    return itv(lam, mu)


def transverse_isotropic_tensor(
    A: np.ndarray, C: np.ndarray, F: np.ndarray, L: np.ndarray, N: np.ndarray
) -> np.ndarray:
    """
    Construct a transverse isotropic elastic tensor (4th-order representation).

    This returns the tensor as a 3x3x3x3 array, suitable for direct tensor operations
    like rotations and contractions. For 6x6 Voigt notation, use
    transverse_isotropic_tensor_voigt() instead.

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
    C : ndarray, shape (..., 3, 3, 3, 3)
        Transverse isotropic elastic tensor in 4th-order form.

    See Also
    --------
    tti.elastic.voigt.transverse_isotropic_tensor : Voigt notation (6x6) version
    tti.elastic.fourth.transverse_isotropic_tensor : Direct alias to this implementation
    """
    return titv(A, C, F, L, N)


__all__ = [
    "isotropic_tensor",
    "transverse_isotropic_tensor",
    "tilted_transverse_isotropic_tensor",
]
