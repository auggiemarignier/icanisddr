# ruff: noqa: E741
# a fair bit of tensor notation is involved here so stop
# ruff from complaining about variable names like l


"""Test Voigt notation tensors."""

import numpy as np
import pytest

from tti.elastic.voigt import (
    isotropic_tensor,
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
