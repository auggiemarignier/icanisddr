"""Test unpacking functions for model vectors."""

from dataclasses import dataclass

import numpy as np
import pytest

from tti.traveltimes._unpackings import (
    _unpack_model_vector,
    _unpack_model_vector_no_N,
    _unpack_model_vector_no_shear,
    _unpack_nested_model_vector,
    _unpack_nested_model_vector_no_N,
    _unpack_nested_model_vector_no_shear,
)


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
    """Fixture for random Love parameter values."""
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


def test__unpack_nested_model_vector(lv: LoveValues) -> None:
    """Test unpacking of nested model vector into Love parameters.

    Tests that the nested unpacking correctly reconstructs Love parameters
    from their nested differences representation and angles in radians.
    """
    dC = lv.C - lv.A
    dF = lv.F - (lv.A - 2 * lv.N)
    dN = lv.N - lv.L

    # Build m as (B, M, 7) then flatten to (B, 7M)
    B, M = lv.A.shape
    m_nested = np.stack([lv.A, dC, dF, lv.L, dN, lv.eta1, lv.eta2], axis=-1).reshape(
        B, 7 * M
    )

    A, C, F, L, N, eta1, eta2 = _unpack_nested_model_vector(m_nested)

    np.testing.assert_allclose(A, lv.A)
    np.testing.assert_allclose(C, lv.C)
    np.testing.assert_allclose(F, lv.F)
    np.testing.assert_allclose(L, lv.L)
    np.testing.assert_allclose(N, lv.N)
    np.testing.assert_allclose(eta1, np.radians(lv.eta1))
    np.testing.assert_allclose(eta2, np.radians(lv.eta2))


def test__unpack_model_vector(lv) -> None:
    """Test unpacking of model vector into Love parameters.

    Tests that the unpacking function correctly extracts each parameter
    from the model vector.
    The angles eta1 and eta2 should be converted from degrees to radians.
    """
    # Build m as (B, M, 7) then flatten to (B, 7M)
    B, M = lv.A.shape
    m = np.stack([lv.A, lv.C, lv.F, lv.L, lv.N, lv.eta1, lv.eta2], axis=-1).reshape(
        B, 7 * M
    )

    A, C, F, L, N, eta1, eta2 = _unpack_model_vector(m)

    np.testing.assert_allclose(A, lv.A)
    np.testing.assert_allclose(C, lv.C)
    np.testing.assert_allclose(F, lv.F)
    np.testing.assert_allclose(L, lv.L)
    np.testing.assert_allclose(N, lv.N)
    np.testing.assert_allclose(eta1, np.radians(lv.eta1))
    np.testing.assert_allclose(eta2, np.radians(lv.eta2))


def test__unpack_nested_model_vector_no_shear(lv) -> None:
    """Test unpacking of nested model vector with no shear into Love parameters.

    Tests that the unpacking function correctly extracts Love parameters
    when the model vector doesn't include shear anisotropy.
    The angles eta1 and eta2 should be converted from degrees to radians.
    """

    dC = lv.C - lv.A
    dF = lv.F - lv.A  # no shear anisotropy term

    # Build m as (B, M, 5) then flatten to (B, 5M)
    B, M = lv.A.shape
    m_nested = np.stack([lv.A, dC, dF, lv.eta1, lv.eta2], axis=-1).reshape(B, 5 * M)

    A, C, F, L, N, eta1, eta2 = _unpack_nested_model_vector_no_shear(m_nested)

    np.testing.assert_allclose(A, lv.A)
    np.testing.assert_allclose(C, lv.C)
    np.testing.assert_allclose(F, lv.F)
    np.testing.assert_allclose(L, np.zeros_like(lv.L))
    np.testing.assert_allclose(N, np.zeros_like(lv.N))
    np.testing.assert_allclose(eta1, np.radians(lv.eta1))
    np.testing.assert_allclose(eta2, np.radians(lv.eta2))


def test__unpack_model_vector_no_shear(lv) -> None:
    """Test unpacking of model vector with no shear into Love parameters.

    Tests that the unpacking function correctly extracts Love parameters
    when the model vector doesn't include shear anisotropy.
    The angles eta1 and eta2 should be converted from degrees to radians.
    """

    # Build m as (B, M, 5) then flatten to (B, 5M)
    B, M = lv.A.shape
    m = np.stack([lv.A, lv.C, lv.F, lv.eta1, lv.eta2], axis=-1).reshape(B, 5 * M)

    A, C, F, L, N, eta1, eta2 = _unpack_model_vector_no_shear(m)

    np.testing.assert_allclose(A, lv.A)
    np.testing.assert_allclose(C, lv.C)
    np.testing.assert_allclose(F, lv.F)
    np.testing.assert_allclose(L, np.zeros_like(lv.L))
    np.testing.assert_allclose(N, np.zeros_like(lv.N))
    np.testing.assert_allclose(eta1, np.radians(lv.eta1))
    np.testing.assert_allclose(eta2, np.radians(lv.eta2))


def test__unpack_model_vector_no_N(lv) -> None:
    """Test unpacking of model vector with no N parameter into Love parameters.

    Tests that the unpacking function correctly extracts Love parameters
    when the model vector doesn't include the N parameter.
    The angles eta1 and eta2 should be converted from degrees to radians.
    """

    # Build m as (B, M, 6) then flatten to (B, 6M)
    B, M = lv.A.shape
    m = np.stack([lv.A, lv.C, lv.F, lv.L, lv.eta1, lv.eta2], axis=-1).reshape(B, 6 * M)

    A, C, F, L, N, eta1, eta2 = _unpack_model_vector_no_N(m)

    np.testing.assert_allclose(A, lv.A)
    np.testing.assert_allclose(C, lv.C)
    np.testing.assert_allclose(F, lv.F)
    np.testing.assert_allclose(L, lv.L)
    np.testing.assert_allclose(N, np.zeros_like(lv.N))
    np.testing.assert_allclose(eta1, np.radians(lv.eta1))
    np.testing.assert_allclose(eta2, np.radians(lv.eta2))


def test__unpack_nested_model_vector_no_N(lv) -> None:
    """Test unpacking of nested model vector with no N parameter into Love parameters.

    Tests that the unpacking function correctly extracts Love parameters
    when the model vector doesn't include the N parameter.
    The angles eta1 and eta2 should be converted from degrees to radians.
    """

    dC = lv.C - lv.A
    dF = lv.F - lv.A  # no shear anisotropy term

    # Build m as (B, M, 6) then flatten to (B, 6M)
    B, M = lv.A.shape
    m_nested = np.stack([lv.A, dC, dF, lv.L, lv.eta1, lv.eta2], axis=-1).reshape(
        B, 6 * M
    )

    A, C, F, L, N, eta1, eta2 = _unpack_nested_model_vector_no_N(m_nested)

    np.testing.assert_allclose(A, lv.A)
    np.testing.assert_allclose(C, lv.C)
    np.testing.assert_allclose(F, lv.F)
    np.testing.assert_allclose(L, lv.L)
    np.testing.assert_allclose(N, np.zeros_like(lv.N))
    np.testing.assert_allclose(eta1, np.radians(lv.eta1))
    np.testing.assert_allclose(eta2, np.radians(lv.eta2))
