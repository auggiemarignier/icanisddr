# ruff: noqa: E741
# a fair bit of tensor notation is involved here so stop
# ruff from complaining about variable names like l


"""Test Voigt notation tensors."""

import numpy as np
import pytest

from tti.elastic.voigt import (
    isotropic_tensor,
    n_outer_n,
    transverse_isotropic_tensor,
)


@pytest.mark.parametrize(
    "shape",
    [(6, 6), (2, 6, 6), (2, 4, 6, 6)],
    ids=["single", "cells", "batch_cells"],
)
def test_isotropic_symmetry(shape: tuple[int, ...], rng: np.random.Generator) -> None:
    """Test that an isotropic elastic tensor has the required symmetries."""
    leading_shape = shape[:-2]
    lambda_ = rng.uniform(1, 10, size=leading_shape)
    mu = rng.uniform(1, 10, size=leading_shape)

    C_voigt = isotropic_tensor(lambda_, mu)

    assert C_voigt.shape == shape

    # Check symmetry of the 6x6 matrices for each batch/cell
    if C_voigt.ndim == 2:
        np.testing.assert_array_equal(C_voigt, C_voigt.T)
    else:
        for idx in np.ndindex(C_voigt.shape[:-2]):
            np.testing.assert_array_equal(C_voigt[idx], C_voigt[idx].T)


@pytest.mark.parametrize(
    "shape,",
    [(6, 6), (2, 6, 6), (2, 4, 6, 6)],
    ids=["single", "cells", "batch_cells"],
)
def test_transverse_isotropic_symmetry(
    shape: tuple[int, ...], rng: np.random.Generator
) -> None:
    """Test that a transverse isotropic elastic tensor has the required symmetries."""
    leading_shape = shape[:-2]
    A = rng.uniform(1, 10, size=leading_shape)
    C = rng.uniform(1, 10, size=leading_shape)
    F = rng.uniform(1, 10, size=leading_shape)
    L = rng.uniform(1, 10, size=leading_shape)
    N = rng.uniform(1, 10, size=leading_shape)

    C_voigt = transverse_isotropic_tensor(A, C, F, L, N)

    assert C_voigt.shape == shape

    # Check symmetry of the 6x6 matrices for each batch/cell
    if C_voigt.ndim == 2:
        np.testing.assert_array_equal(C_voigt, C_voigt.T)
    else:
        for idx in np.ndindex(C_voigt.shape[:-2]):
            np.testing.assert_array_equal(C_voigt[idx], C_voigt[idx].T)


@pytest.mark.parametrize(
    "shape,",
    [(3,), (2, 3)],
    ids=["single", "batch"],
)
def test_n_outer_n(shape: tuple[int, ...], rng: np.random.Generator) -> None:
    """Test that n_outer_n produces the correct outer product in Voigt notation."""

    leading_shape = shape[:-1]

    n = rng.uniform(-1, 1, size=(*leading_shape, 3))
    n = n / np.linalg.norm(n, axis=-1, keepdims=True)

    n_outer_n_result = n_outer_n(n)
    assert n_outer_n_result.shape == (*leading_shape, 6)

    expected = []
    for _n in np.atleast_2d(n):
        outer = np.outer(_n, _n)
        expected.append(
            [
                outer[0, 0],
                outer[1, 1],
                outer[2, 2],
                2 * outer[1, 2],
                2 * outer[0, 2],
                2 * outer[0, 1],
            ]
        )

    expected = np.array(expected).squeeze()
    np.testing.assert_allclose(n_outer_n_result, expected)
