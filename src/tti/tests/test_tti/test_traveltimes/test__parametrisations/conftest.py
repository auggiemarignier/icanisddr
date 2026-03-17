"""Common configuration for parametrisation tests."""

from dataclasses import dataclass
from typing import Any
from unittest import mock

import numpy as np
import pytest

from tti.traveltimes._parametrisations import Parametriser


def _compare_potential_arrays(
    actual: np.ndarray | Any, expected: np.ndarray | Any
) -> None:
    """Helper function to compare arrays that may be numpy arrays or scalars."""
    if isinstance(expected, np.ndarray):
        np.testing.assert_allclose(actual, expected)
    else:
        assert actual == expected


def _compare_called_args_to_expected(
    called_args: tuple,
    expected_args: tuple | None,
    called_kwargs: dict,
    expected_kwargs: dict | None,
) -> None:
    """Helper function to compare called args/kwargs to expected args/kwargs."""
    if expected_args is None:
        expected_args = ()
    if expected_kwargs is None:
        expected_kwargs = {}

    for idx, expected in enumerate(expected_args):
        _compare_potential_arrays(called_args[idx], expected)

    for k, expected in expected_kwargs.items():
        assert k in called_kwargs
        _compare_potential_arrays(called_kwargs[k], expected)


@pytest.fixture
def assert_delegates_to_unpack(monkeypatch):
    """Fixture returning a callable to assert `to_parameters` delegation.

    `to_parameters` is called with a single model vector argument.
    Any extra arguments of the dependency function (e.g. reference model) should be instance attributes only passed internally.
    This fixture passes m to the dependency function and then checks that the unpacking function was called with the same m and any expected extra args/kwargs.
    Usage:
        def test(..., assert_delegates_to_unpack):
            assert_delegates_to_unpack(fn_to_mock, parametriser, m, *args, **kwargs)
    """

    def _fn(
        fn_to_be_mocked: str,
        parametriser: Parametriser,
        m: np.ndarray,
        expected_args=None,
        expected_kwargs=None,
    ) -> None:
        sentinel = tuple(object() for _ in range(7))
        fake_unpack = mock.MagicMock(return_value=sentinel)
        monkeypatch.setattr(fn_to_be_mocked, fake_unpack)

        A, C, F, L, N, eta1, eta2 = parametriser.to_parameters(m)

        # Compare call args robustly (numpy arrays need elementwise checks).
        called_args, called_kwargs = fake_unpack.call_args

        # The first positional argument should be m.
        assert called_args[0] is m

        # remaining positional args should match expected_args
        # These would have been copied on instantiation of the parametriser, so we can't check identity, but we can check value.
        _compare_called_args_to_expected(
            called_args[1:], expected_args, called_kwargs, expected_kwargs
        )

        assert A is sentinel[0]
        assert C is sentinel[1]
        assert F is sentinel[2]
        assert L is sentinel[3]
        assert N is sentinel[4]
        assert eta1 is sentinel[5]
        assert eta2 is sentinel[6]

    return _fn


@pytest.fixture
def assert_delegates_to_jacobian(monkeypatch):
    """Fixture returning a callable to assert `apply_jacobian` delegation.

    For functions with signature ``f(grad, *args, **kwargs)`` the fixture
    forwards any extra positional/keyword args and asserts the mock was
    called with the same arguments.
    """

    def _fn(
        fn_to_be_mocked: str,
        parametriser: Parametriser,
        grad_lv: np.ndarray,
        expected_args=None,
        expected_kwargs=None,
    ) -> None:
        sentinel = object()
        fake_apply_jac = mock.MagicMock(return_value=sentinel)
        monkeypatch.setattr(fn_to_be_mocked, fake_apply_jac)

        result = parametriser.apply_jacobian(grad_lv)

        called = fake_apply_jac.call_args
        called_args, called_kwargs = called

        # The first positional argument should be grad_lv.
        assert called_args[0] is grad_lv

        _compare_called_args_to_expected(
            called_args[1:], expected_args, called_kwargs, expected_kwargs
        )

        assert result is sentinel

    return _fn


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
