# ruff: noqa: E741
# a fair bit of tensor notation is involved here so stop
# ruff from complaining about variable names like l

"""Voigt notation mapping for elastic tensors and transformations."""

import numpy as np

# Voigt mapping: (i,j) -> I
# (0,0)->0, (1,1)->1, (2,2)->2, (1,2)->3, (0,2)->4, (0,1)->5
VOIGT_MAP = {
    0: [0, 0],
    1: [1, 1],
    2: [2, 2],
    3: [1, 2],
    4: [0, 2],
    5: [0, 1],
}

_VMAP = np.array(
    [
        [0, 0],
        [1, 1],
        [2, 2],
        [1, 2],
        [0, 2],
        [0, 1],
    ]
)


def elastic_tensor_to_voigt(C: np.ndarray) -> np.ndarray:
    """
    Convert a fully symmetric 4th-order elastic tensor (C_ijkl) to 6x6 Voigt notation (C_voigt).

    Assumes:
      - C has both minor and major symmetries:
        C_ijkl = C_jikl = C_ijlk = C_jilk = C_klij = C_lkij = C_klji = C_lkji
      - Voigt order: [11, 22, 33, 23, 13, 12]

    Parameters
    ----------
    C : ndarray, shape (..., 3, 3, 3, 3)
        Fourth order elastic tensor

    Returns
    -------
    C_voigt : ndarray, shape (..., 6, 6)
        Elastic tensor in Voigt notation
    """
    leading_shape = C.shape[:-4]
    C_voigt = np.zeros((*leading_shape, 6, 6), dtype=C.dtype)

    # vectorised outer products of index pairs
    ij = _VMAP[:, None, :]  # shape (6,1,2)
    kl = _VMAP[None, :, :]  # shape (1,6,2)

    # gather C[i,j,k,l] for all Voigt combinations
    C_voigt[..., :, :] = C[..., ij[..., 0], ij[..., 1], kl[..., 0], kl[..., 1]]

    return C_voigt


def voigt_to_elastic_tensor(C_voigt: np.ndarray) -> np.ndarray:
    """
    Convert an elastic tensor in Voigt notation (nx6x6) to a 4th order tensor (nx3x3x3x3).

    This imposes the major and minor symmetries of the elastic tensor.

    C_ijkl = C_jikl = C_ijlk = C_jilk = C_klij = C_lkij = C_klji = C_lkji

    Parameters
    ----------
    C_voigt : ndarray, shape (..., 6, 6)
        Elastic tensor in Voigt notation

    Returns
    -------
    C : ndarray, shape (..., 3, 3, 3, 3)
        Fourth order elastic tensor
    """

    C = np.zeros((*C_voigt.shape[:-2], 3, 3, 3, 3))
    for m in range(6):
        i, j = VOIGT_MAP[m]
        for n in range(6):
            k, l = VOIGT_MAP[n]
            C[..., i, j, k, l] = C_voigt[..., m, n]
            C[..., j, i, k, l] = C_voigt[..., m, n]
            C[..., i, j, l, k] = C_voigt[..., m, n]
            C[..., j, i, l, k] = C_voigt[..., m, n]
            C[..., k, l, i, j] = C_voigt[..., m, n]
            C[..., l, k, i, j] = C_voigt[..., m, n]
            C[..., k, l, j, i] = C_voigt[..., m, n]
            C[..., l, k, j, i] = C_voigt[..., m, n]

    return C


def transformation_to_voigt(T: np.ndarray) -> np.ndarray:
    """
    Convert a 4th order transformation tensor to Voigt notation.

    Enforces the minor symmetries of the transformation tensor expected by Voigt notation.

    Parameters
    ----------
    T : ndarray, shape (..., 3, 3, 3, 3)
        Fourth order transformation tensor

    Returns
    -------
    T_voigt : ndarray, shape (..., 6, 6)
        Transformation tensor in Voigt notation
    """

    # Build indexing arrays: ij is (6,1,2), kl is (1,6,2)
    ij = _VMAP[:, None, :]  # shape (6, 1, 2)
    kl = _VMAP[None, :, :]  # shape (1, 6, 2)

    # Extract i, j, k, l indices with broadcasting
    i = ij[..., 0]  # shape (6, 1)
    j = ij[..., 1]  # shape (6, 1)
    k = kl[..., 0]  # shape (1, 6)
    l = kl[..., 1]  # shape (1, 6)

    # Base contribution: T[i, j, k, l]
    T_voigt = T[..., i, j, k, l]

    # Add symmetric contribution T[i, j, l, k] only where k != l
    mask = k != l
    T_voigt = np.where(mask, T_voigt + T[..., i, j, l, k], T_voigt)
    return T_voigt


def matrix_to_voigt(R: np.ndarray) -> np.ndarray:
    """Convert a 3x3 transformation matrix to Voigt notation using Bond's law.

    Parameters
    ----------
    R : ndarray, shape (..., 3, 3)
        3x3 transformation matrix

    Returns
    -------
    R_voigt : ndarray, shape (..., 6, 6)
        Transformation matrix in Voigt notation
    """

    # get the notation the same as in Brett et al., 2024
    r11 = R[..., 0, 0]
    r12 = R[..., 0, 1]
    r13 = R[..., 0, 2]
    r21 = R[..., 1, 0]
    r22 = R[..., 1, 1]
    r23 = R[..., 1, 2]
    r31 = R[..., 2, 0]
    r32 = R[..., 2, 1]
    r33 = R[..., 2, 2]

    # Build the 6x6 Bond transformation matrix in Voigt notation
    row0 = np.stack(
        [r11**2, r12**2, r13**2, 2 * r12 * r13, 2 * r11 * r13, 2 * r11 * r12], axis=-1
    )
    row1 = np.stack(
        [r21**2, r22**2, r23**2, 2 * r22 * r23, 2 * r21 * r23, 2 * r21 * r22], axis=-1
    )
    row2 = np.stack(
        [r31**2, r32**2, r33**2, 2 * r32 * r33, 2 * r31 * r33, 2 * r31 * r32], axis=-1
    )
    row3 = np.stack(
        [
            r21 * r31,
            r22 * r32,
            r23 * r33,
            r22 * r33 + r23 * r32,
            r21 * r33 + r23 * r31,
            r21 * r32 + r22 * r31,
        ],
        axis=-1,
    )
    row4 = np.stack(
        [
            r11 * r31,
            r12 * r32,
            r13 * r33,
            r12 * r33 + r13 * r32,
            r11 * r33 + r13 * r31,
            r11 * r32 + r12 * r31,
        ],
        axis=-1,
    )
    row5 = np.stack(
        [
            r11 * r21,
            r12 * r22,
            r13 * r23,
            r12 * r23 + r13 * r22,
            r11 * r23 + r13 * r21,
            r11 * r22 + r12 * r21,
        ],
        axis=-1,
    )
    return np.stack([row0, row1, row2, row3, row4, row5], axis=-2)
