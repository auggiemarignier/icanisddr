"""Common configuration for parametrisation tests."""

from collections.abc import Callable
from dataclasses import dataclass

import numpy as np
import pytest

from tti.traveltimes._parametrisations import BaseParametriser as Parametriser


@dataclass
class LoveValues:
    """Container for Love parameter arrays."""

    A: np.ndarray
    C: np.ndarray
    F: np.ndarray
    L: np.ndarray
    N: np.ndarray
    eta1: np.ndarray  # angle in degrees
    eta2: np.ndarray  # angle in degrees


@pytest.fixture
def lv(rng: np.random.Generator) -> LoveValues:
    """Fixture for random Love parameter values and angles."""
    n_models = rng.integers(2, 10, size=2)
    A_values = rng.uniform(5.0, 15.0, size=n_models)
    C_values = rng.uniform(5.0, 15.0, size=n_models)
    F_values = rng.uniform(3.0, 10.0, size=n_models)
    L_values = rng.uniform(0.1, 0.5, size=n_models)
    N_values = rng.uniform(0.1, 0.5, size=n_models)
    eta1_values = rng.uniform(-180.0, 180.0, size=n_models)
    eta2_values = rng.uniform(-180.0, 180.0, size=n_models)
    return LoveValues(
        A=A_values,
        C=C_values,
        F=F_values,
        L=L_values,
        N=N_values,
        eta1=eta1_values,
        eta2=eta2_values,
    )


@pytest.fixture
def grad_lv(rng: np.random.Generator, lv: LoveValues) -> np.ndarray:
    """Fixture for random gradient values with respect to the Love parameters and angles."""
    B, M = lv.A.shape
    T = rng.integers(1, 5)  # number of travel time measurements
    return rng.normal(size=(B, M, 7, T))


@pytest.fixture
def numeric_apply_from_transform() -> Callable:
    """Return a helper that applies finite-difference chain-rule using an transform fn.

    Usage:
        apply_fd = numeric_apply_from_transform()
        apply_fd(transform_fn, m, grad_lv, N=7, eps=1e-6) -> dt_dm (shape like grad_lv)

    The transform function should accept `m` with shape (B, M*N) and return a 7-tuple of arrays each shaped (B, M) in the same ordering as the analytic
    `apply_jacobian` expects: (A, C, F, L, N, eta1, eta2).
    """

    def _fn(
        transform_fn, m: np.ndarray, grad_lv: np.ndarray, eps: float = 1e-6
    ) -> np.ndarray:
        B, M, _, T = (
            grad_lv.shape
        )  # grad_lv shape is (B batch, M segments, 7 love parameters, T traveltimes)

        m = m.astype(float, copy=True)
        batch_size, n_flat_params = m.shape
        if batch_size != B:
            raise ValueError(
                f"Batch size of m ({batch_size}) does not match batch size of grad_lv ({B}). Adjust the test fixture or this function as needed."
            )
        N = n_flat_params // M  # number of parameters per segment
        if N != 7 and N != 5:
            raise ValueError(
                f"Expected 5 or 7 parameters per segment for nested relative, got {N}. Adjust the test fixture or this function as needed."
            )

        dt_dm_result = np.zeros((B, M, N, T))

        def _stack_unpacked_parameters(m_in):
            parts = transform_fn(m_in)
            return np.stack(parts, axis=1)  # shape (B, 7, M)

        # iterate over each model element and compute local dp/dm via central FD
        for batch_index in range(B):
            for segment_index in range(M):
                for param_index in range(N):
                    # The test fixtures flatten `m` as (B, N, M) -> (B, N*M)
                    # (params major), so the flat index for parameter `param_index`
                    # and segment `segment_index` is `param_index * M + segment_index`.
                    flat_param_index = param_index * M + segment_index

                    m_plus = m.copy()
                    m_minus = m.copy()
                    m_plus[batch_index, flat_param_index] += eps
                    m_minus[batch_index, flat_param_index] -= eps

                    lv_plus = _stack_unpacked_parameters(m_plus)
                    lv_minus = _stack_unpacked_parameters(m_minus)  # (B, 7, M)

                    # dlv_dmi: shape (B, 7, M) -> partials of each lv param w.r.t this model param
                    dlv_dmi = (lv_plus - lv_minus) / (2.0 * eps)

                    # dlv_dmi for this batch/segment: shape (7,)
                    dlv_wrt_model = dlv_dmi[batch_index, :, segment_index]

                    # grad w.r.t parameters for this batch/segment: shape (7, T)
                    grad_wrt_lv = grad_lv[batch_index, segment_index, :, :]

                    # Chain rule: sum over parameters to get dt/dm for this model param: shape (T,)
                    dt_wrt_model_param = dlv_wrt_model @ grad_wrt_lv

                    dt_dm_result[batch_index, segment_index, param_index, :] = (
                        dt_wrt_model_param
                    )

        return dt_dm_result

    return _fn


@pytest.fixture
def assert_jacobian_matches_finite_difference(
    numeric_apply_from_transform: Callable,
) -> Callable:
    """Fixture returning a callable to assert that the Jacobian matches a finite-difference approximation."""

    def _fn(parametriser: Parametriser, grad_lv: np.ndarray, m: np.ndarray) -> None:
        grad_dm = parametriser.apply_jacobian(grad_lv)
        grad_fd = numeric_apply_from_transform(
            parametriser.to_parameters, m, grad_lv, eps=1e-6
        )
        np.testing.assert_allclose(grad_dm, grad_fd, rtol=1e-6, atol=1e-8)

    return _fn


@pytest.fixture
def assert_parametriser_matches_love_values() -> Callable:
    """Fixture returning a callable to assert that the parametriser's to_parameters output matches the original Love values.

    The returned callable accepts an `include_shear: bool = True` argument. When
    `include_shear` is False the shear parameters `L` and `N` are compared to
    zeros instead of `lv` values (useful for no-shear parametrisers).
    """

    def _fn(
        parametriser: Parametriser,
        m: np.ndarray,
        lv: LoveValues,
        include_shear: bool = True,
    ) -> None:
        A, C, F, L, N, eta1, eta2 = parametriser.to_parameters(m)

        np.testing.assert_allclose(A, lv.A)
        np.testing.assert_allclose(C, lv.C)
        np.testing.assert_allclose(F, lv.F)
        if include_shear:
            np.testing.assert_allclose(L, lv.L)
            np.testing.assert_allclose(N, lv.N)
        else:
            np.testing.assert_allclose(L, np.zeros_like(lv.L))
            np.testing.assert_allclose(N, np.zeros_like(lv.N))
        np.testing.assert_allclose(eta1, np.radians(lv.eta1))
        np.testing.assert_allclose(eta2, np.radians(lv.eta2))

    return _fn
