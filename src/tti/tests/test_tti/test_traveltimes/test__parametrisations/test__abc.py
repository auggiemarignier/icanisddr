"""Test the base class and helper functions for parametrisations."""

from unittest.mock import Mock

import numpy as np
import pytest

from tti.traveltimes._parametrisations._abc import (
    _jacobian_to_dm,
    _transform_model_vector,
)


def test__jacobian_to_dm_identity(grad_lv: np.ndarray) -> None:
    """Test that `_jacobian_to_dm` with identity transform returns the input gradient."""
    identity_transform = lambda x: x
    result = _jacobian_to_dm(grad_lv, identity_transform)
    np.testing.assert_allclose(result, grad_lv)


def test__jacobian_to_dm_arg_is_called(grad_lv: np.ndarray) -> None:
    """Test that the transform function passed to `_jacobian_to_dm` is called."""
    transform = Mock(side_effect=lambda x: x)

    _jacobian_to_dm(grad_lv, transform)
    transform.assert_called()


@pytest.fixture(params=[5, 7])
def m(rng: np.random.Generator, request) -> np.ndarray:
    """Fixture for a random model vector `m` parameterised over `N`.

    `N` is the number of parameters per segment; tests will run for
    `N=5` and `N=7`.
    """
    B, M = 2, 3  # batch size and number of segments
    N = request.param
    return rng.random((B, N * M))


def test__transform_model_vector_identity(m: np.ndarray) -> None:
    """Test that `_transform_model_vector` with identity transform returns the input model vector."""
    B, NM = m.shape
    M = 3  # number of segments, taken directly from the fixture
    N = NM // M

    identity_transform = lambda x: x
    result = _transform_model_vector(m, N, identity_transform)
    np.testing.assert_allclose(result, m.reshape(B, N, M))


def test__transform_model_vector_arg_is_called(m: np.ndarray) -> None:
    """Test that the transform function passed to `_transform_model_vector` is called."""
    B, NM = m.shape
    M = 3  # number of segments, taken directly from the fixture
    N = NM // M

    transform = Mock(side_effect=lambda x: x)
    _transform_model_vector(m, N, transform)
    transform.assert_called()
