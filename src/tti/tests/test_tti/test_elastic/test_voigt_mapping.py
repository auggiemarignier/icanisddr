# ruff: noqa: E741
# a fair bit of tensor notation is involved here so stop
# ruff from complaining about variable names like l

"""Tests for Voigt notation mapping of elastic tensors and transformations."""

import numpy as np
import pytest

from tti.elastic.fourth import _check_elastic_tensor_symmetry, transformation_4th_order
from tti.elastic.fourth import isotropic_tensor as it4
from tti.elastic.fourth import tilted_transverse_isotropic_tensor as tti4
from tti.elastic.fourth import transverse_isotropic_tensor as tit4
from tti.elastic.voigt import isotropic_tensor as itv
from tti.elastic.voigt import tilted_transverse_isotropic_tensor as ttiv
from tti.elastic.voigt import transverse_isotropic_tensor as titv
from tti.elastic.voigt_mapping import (
    VOIGT_MAP,
    elastic_tensor_to_voigt,
    transformation_to_voigt,
    voigt_to_elastic_tensor,
)
from tti.rotation import rotation_matrix_z

VOIGT_MAP_INV = {
    (0, 0): 0,
    (1, 1): 1,
    (2, 2): 2,
    (1, 2): 3,
    (2, 1): 3,
    (0, 2): 4,
    (2, 0): 4,
    (0, 1): 5,
    (1, 0): 5,
}


def _build_symmetric_elastic_tensor(shape: tuple[int, ...]) -> np.ndarray:
    """Create symmetric 4th-order tensors for arbitrary leading dimensions."""

    leading_shape = shape[:-4]
    n_elements = int(np.prod(leading_shape) or 1)
    C_flat = np.zeros((n_elements, 3, 3, 3, 3))

    for i in range(n_elements):
        offset = i + 1.0
        d0, d1, d2 = 1.0 + offset, 2.0 + offset, 3.0 + offset
        val = 4.0 + offset

        C_flat[i, 0, 0, 0, 0] = d0
        C_flat[i, 1, 1, 1, 1] = d1
        C_flat[i, 2, 2, 2, 2] = d2

        C_flat[i, 0, 1, 2, 0] = val
        C_flat[i, 1, 0, 2, 0] = val
        C_flat[i, 0, 1, 0, 2] = val
        C_flat[i, 1, 0, 0, 2] = val
        C_flat[i, 2, 0, 0, 1] = val
        C_flat[i, 2, 0, 1, 0] = val
        C_flat[i, 0, 2, 0, 1] = val
        C_flat[i, 0, 2, 1, 0] = val

    return C_flat.reshape(shape)


def _elastic_tensor_to_voigt_reference(C: np.ndarray) -> np.ndarray:
    """
    Reference loop implementation for validating vectorised elastic_tensor_to_voigt.

    Convert a 4th order elastic tensor (3x3x3x3) to Voigt notation (6x6).

    Parameters
    ----------
    C : ndarray, shape (..., 3, 3, 3, 3)
        Fourth order elastic tensor with batch and cell dimensions

    Returns
    -------
    C_voigt : ndarray, shape (..., 6, 6)
        Elastic tensor in Voigt notation
    """

    if not _check_elastic_tensor_symmetry(C).all():
        raise ValueError("Input elastic tensor does not have the required symmetries.")

    leading_shape = C.shape[:-4]
    n_elements = int(np.prod(leading_shape) or 1)
    C_voigt = np.zeros((n_elements, 6, 6))

    def _elastic_tensor_to_voigt_single(C: np.ndarray) -> np.ndarray:
        C_voigt_single = np.zeros((6, 6))
        for i in range(3):
            for j in range(3):
                m = VOIGT_MAP_INV[(i, j)]
                for k in range(3):
                    for l in range(3):
                        n = VOIGT_MAP_INV[(k, l)]
                        C_voigt_single[m, n] = C[i, j, k, l]
        return C_voigt_single

    C_flat = C.reshape((n_elements, 3, 3, 3, 3))
    for i in range(n_elements):
        C_voigt[i] = _elastic_tensor_to_voigt_single(C_flat[i])
    return C_voigt.reshape((*leading_shape, 6, 6))


def test_voigt_map() -> None:
    """Test the Voigt mapping dictionaries."""

    for m in range(6):
        i, j = VOIGT_MAP[m]
        m_back = VOIGT_MAP_INV[(i, j)]
        assert m == m_back, f"VOIGT_MAP and VOIGT_MAP_INV are inconsistent for m={m}"


@pytest.mark.parametrize(
    "shape",
    [
        (3, 3, 3, 3),
        (2, 3, 3, 3, 3),
        (2, 4, 3, 3, 3, 3),
    ],
    ids=["single", "cells", "batch_cells"],
)
def test_elastic_tensor_to_voigt_correctness(shape: tuple[int, ...]) -> None:
    """Test that the vectorised implementation matches reference loop implementation."""
    C = _build_symmetric_elastic_tensor(shape)

    C_voigt_fast = elastic_tensor_to_voigt(C)
    C_voigt_ref = _elastic_tensor_to_voigt_reference(C)
    np.testing.assert_array_almost_equal(C_voigt_fast, C_voigt_ref)


@pytest.mark.parametrize(
    "shape",
    [
        (3, 3, 3, 3),
        (2, 3, 3, 3, 3),
        (2, 4, 3, 3, 3, 3),
    ],
    ids=["single", "cells", "batch_cells"],
)
def test_elastic_tensor_to_voigt(shape: tuple[int, ...]) -> None:
    """Test conversion from elastic tensor to Voigt notation for 4D/5D/6D inputs."""

    C = _build_symmetric_elastic_tensor(shape)

    C_voigt_fast = elastic_tensor_to_voigt(C)
    C_voigt_ref = _elastic_tensor_to_voigt_reference(C)
    np.testing.assert_array_almost_equal(C_voigt_fast, C_voigt_ref)


@pytest.mark.parametrize(
    "shape",
    [
        (3, 3, 3, 3),
        (2, 3, 3, 3, 3),
        (2, 4, 3, 3, 3, 3),
    ],
    ids=["single", "cells", "batch_cells"],
)
def test_elastic_to_voigt_and_back(shape: tuple[int, ...]) -> None:
    """Round-trip elastic tensor -> Voigt -> elastic for 4D/5D/6D inputs."""

    C = _build_symmetric_elastic_tensor(shape)

    C_voigt = elastic_tensor_to_voigt(C)
    C_reconstructed = voigt_to_elastic_tensor(C_voigt)

    assert C_reconstructed.shape == shape
    np.testing.assert_array_almost_equal(C, C_reconstructed)


@pytest.mark.parametrize(
    "shape",
    [
        (3, 3, 3, 3),
        (2, 3, 3, 3, 3),
        (2, 4, 3, 3, 3, 3),
    ],
    ids=["single", "cells", "batch_cells"],
)
def test_isotropic_4th_matches_voigt(
    shape: tuple[int, ...], rng: np.random.Generator
) -> None:
    """Isotropic 4th-order constructor should match Voigt constructor after mapping."""
    leading_shape = shape[:-4]
    lam = rng.uniform(1, 10, size=leading_shape)
    mu = rng.uniform(1, 10, size=leading_shape)

    C4 = it4(lam, mu)
    assert C4.shape == shape
    C_voigt_from_4th = elastic_tensor_to_voigt(C4)
    C_voigt_direct = itv(lam, mu)

    assert C_voigt_from_4th.shape == leading_shape + (6, 6)
    np.testing.assert_array_almost_equal(C_voigt_from_4th, C_voigt_direct)


@pytest.mark.parametrize(
    "shape",
    [
        (3, 3, 3, 3),
        (2, 3, 3, 3, 3),
        (2, 4, 3, 3, 3, 3),
    ],
    ids=["single", "cells", "batch_cells"],
)
def test_transverse_isotropic_4th_matches_voigt(
    shape: tuple[int, ...], rng: np.random.Generator
) -> None:
    """TI 4th-order constructor should match Voigt constructor after mapping."""
    leading_shape = shape[:-4]
    A = rng.uniform(1, 10, size=leading_shape)
    C = rng.uniform(1, 10, size=leading_shape)
    F = rng.uniform(1, 10, size=leading_shape)
    L = rng.uniform(1, 10, size=leading_shape)
    N = rng.uniform(1, 10, size=leading_shape)

    C4 = tit4(A, C, F, L, N)
    assert C4.shape == shape
    C_voigt_from_4th = elastic_tensor_to_voigt(C4)
    C_voigt_direct = titv(A, C, F, L, N)

    assert C_voigt_from_4th.shape == leading_shape + (6, 6)
    np.testing.assert_array_almost_equal(C_voigt_from_4th, C_voigt_direct)


def test_tti_4th_matches_voigt(rng: np.random.Generator) -> None:
    """TTI 4th-order constructor should match Voigt constructor after mapping."""

    A = rng.uniform(1, 10, size=3)
    C = rng.uniform(1, 10, size=3)
    F = rng.uniform(1, 10, size=3)
    L = rng.uniform(1, 10, size=3)
    N = rng.uniform(1, 10, size=3)
    eta1 = rng.uniform(0, 2 * np.pi, size=3)
    eta2 = rng.uniform(0, 2 * np.pi, size=3)

    C4 = tti4(A, C, F, L, N, eta1, eta2)
    C_voigt_from_4th = elastic_tensor_to_voigt(C4)

    C_voigt = ttiv(A, C, F, L, N, eta1, eta2)

    np.testing.assert_array_almost_equal(C_voigt_from_4th, C_voigt)


@pytest.mark.parametrize(
    "shape",
    [
        (3, 3),
        (2, 3, 3),
        (2, 4, 3, 3),
    ],
    ids=["single", "cells", "batch_cells"],
)
def test_transformation_to_voigt(
    rng: np.random.Generator, shape: tuple[int, ...]
) -> None:
    """Test that the bond tensor in Voigt notation is symmetric."""

    leading_shape = shape[:-2]
    r = rotation_matrix_z(rng.uniform(0, 2 * np.pi, size=leading_shape))
    R = transformation_4th_order(r)
    R_voigt = transformation_to_voigt(R)

    expected_voigt_shape = leading_shape + (6, 6)
    assert R_voigt.shape == expected_voigt_shape

    # get the notation the same as in Brett et al., 2024
    r11 = r[..., 0, 0]
    r12 = r[..., 0, 1]
    r13 = r[..., 0, 2]
    r21 = r[..., 1, 0]
    r22 = r[..., 1, 1]
    r23 = r[..., 1, 2]
    r31 = r[..., 2, 0]
    r32 = r[..., 2, 1]
    r33 = r[..., 2, 2]

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
    expected = np.stack([row0, row1, row2, row3, row4, row5], axis=-2)
    np.testing.assert_array_almost_equal(R_voigt, expected)
