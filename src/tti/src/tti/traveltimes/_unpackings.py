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
    m : ndarray, shape (B, M * 7)
        Nested model parameters: [A, \delta_{CA}, \delta_{F,A+2N}, L, \delta_{NL}, eta1, eta2]
        M is the number of model segments (e.g. number of pixels).
        B is the batch size (at least 1).

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
    batch_size = m.shape[0]
    mT = m.reshape(batch_size, -1, 7).copy()
    A = mT[..., 0]
    C = mT[..., 1] + A
    L = mT[..., 3]
    N = mT[..., 4] + L
    F = mT[..., 2] + A - 2 * N
    eta1 = np.radians(mT[..., 5])
    eta2 = np.radians(mT[..., 6])
    return (A, C, F, L, N, eta1, eta2)


def _unpack_nested_model_vector_no_shear(m: np.ndarray) -> seven_arrays:
    r"""Unpack nested model vector into individual Love parameters, with L and N fixed at 0.

    Note:
        There is NO checking of the input shape here for performance reasons.

    Parameters
    ----------
    m : ndarray, shape (B, M * 5)
        Nested model parameters: [A, \delta_{CA}, \delta_{F,A}, eta1, eta2]
        M is the number of model segments (e.g. number of pixels).
        B is the batch size (at least 1).

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
        Fixed at 0.
    N : ndarray, shape (B, M)
        Elastic constant C66
        Fixed at 0.
    eta1 : ndarray, shape (B, M)
        Tilt angle in radians.
    eta2 : ndarray, shape (B, M)
        Azimuthal angle in radians.
    """
    batch_size = m.shape[0]
    mT = m.reshape(batch_size, -1, 5).copy()
    zeros = np.zeros_like(mT[..., 0])
    return (
        mT[..., 0],
        mT[..., 1] + mT[..., 0],
        mT[..., 2] + mT[..., 0],
        zeros,
        zeros,
        np.radians(mT[..., 3]),
        np.radians(mT[..., 4]),
    )


def _unpack_model_vector(m: np.ndarray) -> seven_arrays:
    r"""Unpack model vector into individual Love parameters.

    Note:
        There is NO checking of the input shape here for performance reasons.

    Parameters
    ----------
    m : ndarray, shape (B, M * 7)
        Model parameters: [A, C, F, L, N, eta1, eta2]
        M is the number of model segments (e.g. number of pixels).
        B is the batch size (at least 1).

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
    batch_size = m.shape[0]
    mT = m.reshape(batch_size, -1, 7).copy()
    return (
        mT[..., 0],
        mT[..., 1],
        mT[..., 2],
        mT[..., 3],
        mT[..., 4],
        np.radians(mT[..., 5]),
        np.radians(mT[..., 6]),
    )


def _unpack_model_vector_no_shear(m: np.ndarray) -> seven_arrays:
    r"""Unpack model vector into individual Love parameters, with L and N fixed at 0.

    Note:
        There is NO checking of the input shape here for performance reasons.

    Parameters
    ----------
    m : ndarray, shape (B, M * 5)
        Model parameters: [A, C, F, eta1, eta2]
        M is the number of model segments (e.g. number of pixels).
        B is the batch size (at least 1).

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
        Fixed at 0.
    N : ndarray, shape (B, M)
        Elastic constant C66
        Fixed at 0.
    eta1 : ndarray, shape (B, M)
        Tilt angle in radians.
    eta2 : ndarray, shape (B, M)
        Azimuthal angle in radians.
    """
    batch_size = m.shape[0]
    mT = m.reshape(batch_size, -1, 5).copy()
    zeros = np.zeros_like(mT[..., 0])
    return (
        mT[..., 0],
        mT[..., 1],
        mT[..., 2],
        zeros,
        zeros,
        np.radians(mT[..., 3]),
        np.radians(mT[..., 4]),
    )


def _unpack_model_vector_no_N(m: np.ndarray) -> seven_arrays:
    r"""Unpack model vector into individual Love parameters, with N fixed at 0.

    Note:
        There is NO checking of the input shape here for performance reasons.

    Parameters
    ----------
    m : ndarray, shape (B, M * 6)
        Model parameters: [A, C, F, L, eta1, eta2]
        M is the number of model segments (e.g. number of pixels).
        B is the batch size (at least 1).

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
        Fixed at 0.
    eta1 : ndarray, shape (B, M)
        Tilt angle in radians.
    eta2 : ndarray, shape (B, M)
        Azimuthal angle in radians.
    """
    batch_size = m.shape[0]
    mT = m.reshape(batch_size, -1, 6).copy()
    zeros = np.zeros_like(mT[..., 0])
    return (
        mT[..., 0],
        mT[..., 1],
        mT[..., 2],
        mT[..., 3],
        zeros,
        np.radians(mT[..., 4]),
        np.radians(mT[..., 5]),
    )


def _unpack_nested_model_vector_no_N(m: np.ndarray) -> seven_arrays:
    r"""Unpack nested model vector into individual Love parameters, with N fixed at 0.

    Note:
        There is NO checking of the input shape here for performance reasons.

    Parameters
    ----------
    m : ndarray, shape (B, M * 6)
        Nested model parameters: [A, \delta_{CA}, \delta_{F,A}, L, eta1, eta2]
        M is the number of model segments (e.g. number of pixels).
        B is the batch size (at least 1).

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
        Fixed at 0.
    eta1 : ndarray, shape (B, M)
        Tilt angle in radians.
    eta2 : ndarray, shape (B, M)
        Azimuthal angle in radians.
    """
    batch_size = m.shape[0]
    mT = m.reshape(batch_size, -1, 6).copy()
    zeros = np.zeros_like(mT[..., 0])
    return (
        mT[..., 0],
        mT[..., 1] + mT[..., 0],
        mT[..., 2] + mT[..., 0],
        mT[..., 3],
        zeros,
        np.radians(mT[..., 4]),
        np.radians(mT[..., 5]),
    )


# Mapping of unpacking functions based on input format.
# The keys are (nested, (include_shear, include_N)).
# If include_shear is False, L and N are fixed at 0.
# If include_N is False, N is fixed at 0.
_unpackings: dict[
    bool, dict[tuple[bool, bool], Callable[[np.ndarray], seven_arrays]]
] = {
    True: {
        (True, True): _unpack_nested_model_vector,
        (True, False): _unpack_nested_model_vector_no_N,
        # (False, True): This would be the case where we include N but not L. Not supported - KeyError will be raised if this combination is used.,
        (False, False): _unpack_nested_model_vector_no_shear,
    },
    False: {
        (True, True): _unpack_model_vector,
        (True, False): _unpack_model_vector_no_N,
        # (False, True): This would be the case where we include N but not L. Not supported - KeyError will be raised if this combination is used.,
        (False, False): _unpack_model_vector_no_shear,
    },
}
