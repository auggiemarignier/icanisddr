# ruff: noqa: E741
# a fair bit of tensor notation is involved here so stop
# ruff from complaining about variable names like l


"""Test Voigt notation tensors."""

from collections.abc import Callable

import numpy as np
import pytest

from tti.elastic import tilted_transverse_isotropic_tensor
from tti.elastic.voigt import (
    dCdA,
    dCdC,
    dCdF,
    dCdL,
    dCdN,
    gradient_C_wrt_A,
    gradient_C_wrt_C,
    gradient_C_wrt_F,
    gradient_C_wrt_L,
    gradient_C_wrt_N,
    gradient_D,
    gradient_D_wrt_A,
    gradient_D_wrt_C,
    gradient_D_wrt_eta1,
    gradient_D_wrt_eta2,
    gradient_D_wrt_F,
    gradient_D_wrt_L,
    gradient_D_wrt_N,
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


class TestGradients:
    """Test all the gradient functionality."""

    @pytest.mark.parametrize(
        "gradient_constant",
        [dCdA, dCdC, dCdF, dCdL, dCdN],
        ids=["A", "C", "F", "L", "N"],
    )
    def test_unwritable_gradients(
        self, gradient_constant: np.ndarray, rng: np.random.Generator
    ) -> None:
        """Test that dCd* gradient constants are not writable."""
        idx = rng.integers(0, 6, 2)
        with pytest.raises(ValueError):
            gradient_constant[idx[0], idx[1]] = 0

    @pytest.mark.parametrize(
        "gradient_func,expected",
        [
            (gradient_C_wrt_A, dCdA),
            (gradient_C_wrt_C, dCdC),
            (gradient_C_wrt_F, dCdF),
            (gradient_C_wrt_L, dCdL),
            (gradient_C_wrt_N, dCdN),
        ],
        ids=["A", "C", "F", "L", "N"],
    )
    def test_public_api_C_gradients(
        self, gradient_func: Callable[[], np.ndarray], expected: np.ndarray
    ) -> None:
        """Test the public API for C gradients."""
        grad = gradient_func()

        assert grad.shape == (6, 6)
        np.testing.assert_array_equal(grad, expected)

    @pytest.mark.parametrize(
        "gradient_func,expected",
        [
            (gradient_D_wrt_A, dCdA),
            (gradient_D_wrt_C, dCdC),
            (gradient_D_wrt_F, dCdF),
            (gradient_D_wrt_L, dCdL),
            (gradient_D_wrt_N, dCdN),
        ],
        ids=["A", "C", "F", "L", "N"],
    )
    @pytest.mark.parametrize(
        "shape,",
        [(6, 6), (2, 6, 6), (2, 4, 6, 6)],
        ids=["single", "cells", "batch_cells"],
    )
    def test_gradient_D_wrt_Love_no_rotation(
        self,
        gradient_func: Callable[
            [
                np.ndarray,
                np.ndarray,
                np.ndarray,
                np.ndarray,
                np.ndarray,
                np.ndarray,
                np.ndarray,
            ],
            np.ndarray,
        ],
        expected: np.ndarray,
        shape: tuple[int, ...],
        rng: np.random.Generator,
    ) -> None:
        """Test that the gradient of D with respect to Love when no rotation is applied is the same as the gradient of C."""
        leading_shape = shape[:-2]

        A = rng.random(size=leading_shape)
        C = rng.random(size=leading_shape)
        F = rng.random(size=leading_shape)
        L = rng.random(size=leading_shape)
        N = rng.random(size=leading_shape)
        eta1 = np.zeros(leading_shape)
        eta2 = np.zeros(leading_shape)

        grad = gradient_func(A, C, F, L, N, eta1, eta2)

        assert grad.shape == (*leading_shape, 6, 6)

        expected_b = np.broadcast_to(expected, (*leading_shape, 6, 6))
        np.testing.assert_array_equal(grad, expected_b)

    @staticmethod
    def finite_diff_D(args: list[np.ndarray], idx: int) -> np.ndarray:
        """Helper function computing gradients of D by finite difference to compare with analytical solutions in testing.

        Parameters
        ----------
        args: list[np.ndarray]
            Love parameters in a list
        idx: int
            Index of the parameter with which to differentiate wrt
        """
        dx = 1e-8
        # Make shallow copies of the argument list but replace the indexed
        # element with a bumped value (avoid in-place modification of arrays).
        args_minus = args.copy()
        args_minus[idx] = args[idx] - dx
        args_plus = args.copy()
        args_plus[idx] = args[idx] + dx

        D_minus = tilted_transverse_isotropic_tensor(*args_minus)
        D_plus = tilted_transverse_isotropic_tensor(*args_plus)

        return (D_plus - D_minus) / (2 * dx)

    @pytest.mark.parametrize(
        "gradient_func,idx",
        [
            (gradient_D_wrt_A, 0),
            (gradient_D_wrt_C, 1),
            (gradient_D_wrt_F, 2),
            (gradient_D_wrt_L, 3),
            (gradient_D_wrt_N, 4),
            (gradient_D_wrt_eta1, 5),
            (gradient_D_wrt_eta2, 6),
        ],
        ids=["A", "C", "F", "L", "N", "eta1", "eta2"],
    )
    @pytest.mark.parametrize(
        "shape,",
        [(6, 6), (2, 6, 6), (2, 4, 6, 6)],
        ids=["single", "cells", "batch_cells"],
    )
    def test_gradient_D_finite_diff(
        self,
        gradient_func: Callable[
            [
                np.ndarray,
                np.ndarray,
                np.ndarray,
                np.ndarray,
                np.ndarray,
                np.ndarray,
                np.ndarray,
            ],
            np.ndarray,
        ],
        idx: int,
        shape: tuple[int, ...],
        rng: np.random.Generator,
    ) -> None:
        """Compare analytical solution with finite differences."""
        leading_shape = shape[:-2]

        A = rng.random(size=leading_shape)
        C = rng.random(size=leading_shape)
        F = rng.random(size=leading_shape)
        L = rng.random(size=leading_shape)
        N = rng.random(size=leading_shape)
        eta1 = rng.random(size=leading_shape)
        eta2 = rng.random(size=leading_shape)

        analytical = gradient_func(A, C, F, L, N, eta1, eta2)
        numerical = TestGradients.finite_diff_D([A, C, F, L, N, eta1, eta2], idx)

        np.testing.assert_allclose(analytical, numerical, atol=1e-6)

    @pytest.mark.parametrize(
        "shape,",
        [(6, 6), (2, 6, 6), (2, 4, 6, 6)],
        ids=["single", "cells", "batch_cells"],
    )
    def test_full_gradient_D_finite_diff(
        self,
        shape: tuple[int, ...],
        rng: np.random.Generator,
    ) -> None:
        """Compare analytical solution with finite differences."""
        leading_shape = shape[:-2]

        A = rng.random(size=leading_shape)
        C = rng.random(size=leading_shape)
        F = rng.random(size=leading_shape)
        L = rng.random(size=leading_shape)
        N = rng.random(size=leading_shape)
        eta1 = rng.random(size=leading_shape)
        eta2 = rng.random(size=leading_shape)

        analytical = gradient_D(A, C, F, L, N, eta1, eta2)
        assert analytical.shape == (
            *leading_shape,
            7,
            6,
            6,
        )  # (batch, cells, param, 6, 6)

        numerical = np.stack(
            [
                TestGradients.finite_diff_D([A, C, F, L, N, eta1, eta2], idx)
                for idx in range(7)
            ],
            axis=-3,
        )  # shape (batch, cells, param, 6, 6)

        np.testing.assert_allclose(analytical, numerical, atol=1e-6)
