"""Unpacking an array of TTI parameters into individual parameters.

Parameters can be provided in absolute or nested format, and with or without the shear Love parameters.

These methods unpack the parameters into A, C, F, L, N, eta1, eta2 for use in the methods in the `tti.elastic` module.
"""

from collections.abc import Callable

import numpy as np

type seven_arrays = tuple[
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
]


def _unpack_nested_model_vector(m: np.ndarray) -> seven_arrays:
    r"""Unpack nested model vector into individual Love parameters.

    Note:
        There is NO checking of the input shape here for performance reasons.

    Parameters
    ----------
    m : ndarray, shape (M * 7)
        Nested model parameters: [A, \delta_{CA}, \delta_{F,A+2N}, \delta_{LN}, N, eta1, eta2]
        M is the number of model vectors (e.g. number of pixels).

    Returns
    -------
    A : ndarray, shape (M,)
        Elastic constant C11 = C22
    C : ndarray, shape (M,)
        Elastic constant C33
    F : ndarray, shape (M,)
        Elastic constant C13 = C23
    L : ndarray, shape (M,)
        Elastic constant C44 = C55
    N : ndarray, shape (M,)
        Elastic constant C66
    eta1 : ndarray, shape (M,)
        Tilt angle in radians.
    eta2 : ndarray, shape (M,)
        Azimuthal angle in radians.
    """
    mT = m.reshape(-1, 7).T
    return (
        mT[0],
        mT[1] + mT[0],
        mT[2] + mT[0] - 2 * mT[4],
        mT[3] + mT[4],
        mT[4],
        np.radians(mT[5]),
        np.radians(mT[6]),
    )


def _unpack_nested_model_vector_no_shear(m: np.ndarray) -> seven_arrays:
    r"""Unpack nested model vector into individual Love parameters, with L and N fixed at 0.

    Note:
        There is NO checking of the input shape here for performance reasons.

    Parameters
    ----------
    m : ndarray, shape (M * 5)
        Nested model parameters: [A, \delta_{CA}, \delta_{F,A+2N}, eta1, eta2]
        M is the number of model vectors (e.g. number of pixels).

    Returns
    -------
    A : ndarray, shape (M,)
        Elastic constant C11 = C22
    C : ndarray, shape (M,)
        Elastic constant C33
    F : ndarray, shape (M,)
        Elastic constant C13 = C23
    L : ndarray, shape (M,)
        Elastic constant C44 = C55
        Fixed at 0.
    N : ndarray, shape (M,)
        Elastic constant C66
        Fixed at 0.
    eta1 : ndarray, shape (M,)
        Tilt angle in radians.
    eta2 : ndarray, shape (M,)
        Azimuthal angle in radians.
    """
    mT = m.reshape(-1, 5).T
    zeros = np.zeros_like(mT[0])
    return (
        mT[0],
        mT[1] + mT[0],
        mT[2] + mT[0],  # since L=0, F = A - 2*0 = A
        zeros,
        zeros,
        np.radians(mT[3]),
        np.radians(mT[4]),
    )


def _unpack_model_vector(m: np.ndarray) -> seven_arrays:
    r"""Unpack model vector into individual Love parameters.

    Note:
        There is NO checking of the input shape here for performance reasons.

    Parameters
    ----------
    m : ndarray, shape (M * 7)
        Model parameters: [A, C, F, L, N, eta1, eta2]
        M is the number of model vectors (e.g. number of pixels).

    Returns
    -------
    A : ndarray, shape (M,)
        Elastic constant C11 = C22
    C : ndarray, shape (M,)
        Elastic constant C33
    F : ndarray, shape (M,)
        Elastic constant C13 = C23
    L : ndarray, shape (M,)
        Elastic constant C44 = C55
    N : ndarray, shape (M,)
        Elastic constant C66
    eta1 : ndarray, shape (M,)
        Tilt angle in radians.
    eta2 : ndarray, shape (M,)
        Azimuthal angle in radians.
    """
    mT = m.reshape(-1, 7).T
    return mT[0], mT[1], mT[2], mT[3], mT[4], np.radians(mT[5]), np.radians(mT[6])


def _unpack_model_vector_no_shear(m: np.ndarray) -> seven_arrays:
    r"""Unpack model vector into individual Love parameters, with L and N fixed at 0.

    Note:
        There is NO checking of the input shape here for performance reasons.

    Parameters
    ----------
    m : ndarray, shape (M * 5)
        Model parameters: [A, C, F, eta1, eta2]
        M is the number of model vectors (e.g. number of pixels).

    Returns
    -------
    A : ndarray, shape (M,)
        Elastic constant C11 = C22
    C : ndarray, shape (M,)
        Elastic constant C33
    F : ndarray, shape (M,)
        Elastic constant C13 = C23
    L : ndarray, shape (M,)
        Elastic constant C44 = C55
        Fixed at 0.
    N : ndarray, shape (M,)
        Elastic constant C66
        Fixed at 0.
    eta1 : ndarray, shape (M,)
        Tilt angle in radians.
    eta2 : ndarray, shape (M,)
        Azimuthal angle in radians.
    """
    mT = m.reshape(-1, 5).T
    zeros = np.zeros_like(mT[0])
    return mT[0], mT[1], mT[2], zeros, zeros, np.radians(mT[3]), np.radians(mT[4])


_unpackings: dict[tuple[bool, bool], Callable[[np.ndarray], seven_arrays]] = {
    (True, True): _unpack_nested_model_vector,
    (True, False): _unpack_nested_model_vector_no_shear,
    (False, True): _unpack_model_vector,
    (False, False): _unpack_model_vector_no_shear,
}
