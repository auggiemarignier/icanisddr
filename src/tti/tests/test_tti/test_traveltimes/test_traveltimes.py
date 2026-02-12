"""Test the TravelTimeCalculator class."""

from typing import Literal

import numpy as np
import pytest

from tti.elastic import (
    isotropic_tensor,
    transverse_isotropic_tensor,
)
from tti.elastic.creager import calculate_traveltime as calc_dt_creager
from tti.elastic.creager import love_to_creager
from tti.traveltimes import TravelTimeCalculator
from tti.traveltimes.traveltimes import calculate_relative_traveltime


def test_traveltime_zero_for_zero_perturbation() -> None:
    """Zero perturbation tensor gives zero traveltime anomaly."""

    D = np.zeros((3, 3, 3, 3))
    n = np.array([0.0, 0.0, 1.0])

    dt = calculate_relative_traveltime(n, D)

    assert dt == 0.0


def test_traveltime_batch() -> None:
    """Test traveltime calculation for a batch of ray directions."""

    D = np.zeros((3, 3, 3, 3))
    D[0, 0, 0, 0] = 1.0
    D[1, 1, 1, 1] = 2.0
    D[2, 2, 2, 2] = 3.0
    D = np.broadcast_to(D, (2, 4, 3, 3, 3, 3))

    n_batch = np.array(
        [
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0],
            [0.0, 1.0, 0.0],
            [1.0, 0.0, 0.0],
        ]
    )
    n_paths = n_batch.shape[0]

    dt_batch = calculate_relative_traveltime(n_batch, D)

    expected_per_path = np.array([1.0, 2.0, 3.0, 2.0, 1.0])
    expected = np.broadcast_to(expected_per_path, (2, 4, n_paths))

    assert dt_batch.shape == (2, 4, n_paths)
    np.testing.assert_allclose(dt_batch, expected)


def test_traveltime_batch_with_normalisation() -> None:
    """Test traveltime calculation for a batch of ray directions with an explicit normalisation."""

    D = np.zeros((3, 3, 3, 3))
    D[0, 0, 0, 0] = 1.0
    D[1, 1, 1, 1] = 2.0
    D[2, 2, 2, 2] = 3.0
    D = np.broadcast_to(D, (2, 4, 3, 3, 3, 3))

    n_batch = np.array(
        [
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0],
            [0.0, 1.0, 0.0],
            [1.0, 0.0, 0.0],
        ]
    )
    n_paths = n_batch.shape[0]

    dt_batch = calculate_relative_traveltime(n_batch, D, normalisation=2.0)

    expected_per_path = np.array([1.0, 2.0, 3.0, 2.0, 1.0])
    expected = np.broadcast_to(2.0 * expected_per_path, (2, 4, n_paths))

    assert dt_batch.shape == (2, 4, n_paths)
    np.testing.assert_allclose(dt_batch, expected)


def test_traveltime_isotropic_independent_of_direction(
    rng: np.random.Generator,
) -> None:
    """Isotropic perturbation gives same result for all ray directions."""

    lam = np.array([12.0])
    mu = np.array([5.0])
    D = isotropic_tensor(lam, mu)

    directions = rng.normal(size=(10, 3))
    directions = directions / np.linalg.norm(directions, axis=1, keepdims=True)

    results = calculate_relative_traveltime(directions, D)
    expected = (lam + 2 * mu).item()

    np.testing.assert_allclose(results, expected)


def test_traveltime_linear_in_perturbation(rng: np.random.Generator) -> None:
    """Doubling the perturbation tensor doubles the traveltime anomaly."""

    A, C, F, L, N = rng.random(size=(5, 2, 4))
    D = transverse_isotropic_tensor(A, C, F, L, N)
    n = np.array([[1.0, 0.0, 0.0]])

    dt1 = calculate_relative_traveltime(n, D)
    dt2 = calculate_relative_traveltime(n, 2 * D)

    np.testing.assert_allclose(dt2, 2 * dt1)


def test_traveltime_known_diagonal_tensor() -> None:
    """Test against hand-calculated result for simple diagonal tensor."""

    # D_iijj = δ_ij for i,j in {0,1,2}
    D = np.zeros((3, 3, 3, 3))
    D[0, 0, 0, 0] = 1.0
    D[1, 1, 1, 1] = 2.0
    D[2, 2, 2, 2] = 3.0

    # Only non-zero components are when i=j=k=l D_0000, D_1111, D_2222
    # For n = (nx, ny, nz), result should be nx^4 + 2*ny^4 + 3*nz^4
    n = np.array([[0.6, 0.0, 0.8]])  # normalised
    expected = 1.0 * (0.6**4) + 3.0 * (0.8**4)

    dt = calculate_relative_traveltime(n, D)

    np.testing.assert_allclose(dt, expected)


def test_traveltime_antiparallel_rays_equal(rng: np.random.Generator) -> None:
    """Ray and its opposite give the same traveltime (even power)."""

    A, C, F, L, N = rng.random(size=(5, 2, 4))
    D = transverse_isotropic_tensor(A, C, F, L, N)

    n = np.array([[0.6, 0.8, 0.0]])
    dt_forward = calculate_relative_traveltime(n, D)
    dt_backward = calculate_relative_traveltime(-n, D)

    np.testing.assert_allclose(dt_forward, dt_backward)


def test_traveltime_shape_validation() -> None:
    """Function raises appropriate error for wrong input shapes."""

    D = np.zeros((3, 3, 3, 3))

    with pytest.raises((ValueError, IndexError)):
        calculate_relative_traveltime(np.array([1.0, 0.0]), D)  # n wrong shape

    with pytest.raises((ValueError, IndexError)):
        calculate_relative_traveltime(
            np.array([1.0, 0.0, 0.0]), np.zeros((3, 3))
        )  # D wrong shape


def test_traveltime_transverse_isotropic_polar(rng: np.random.Generator) -> None:
    """Test traveltime calculation for a polar path in a TI medium.

    Expected result is the C_33 component of the elastic tensor.
    """

    A, C, F, L, N = rng.random(size=(5, 2, 4))
    D = transverse_isotropic_tensor(A, C, F, L, N)
    n = np.array([[0.0, 0.0, 1.0]])  # along symmetry axis

    dt = calculate_relative_traveltime(n, D)

    expected = C[..., None]

    np.testing.assert_allclose(dt, expected)


@pytest.mark.parametrize(
    "n", [np.array([1.0, 0.0, 0.0]), np.array([0.0, 1.0, 0.0])]
)  # perpendicular directions
def test_traveltime_transverse_isotropic_equatorial(
    n: np.ndarray, rng: np.random.Generator
) -> None:
    """Test traveltime calculation for an equatorial path in a TI medium.

    Expected result is the C_11 component of the elastic tensor.
    """

    A, C, F, L, N = rng.random(size=(5, 2, 4))
    D = transverse_isotropic_tensor(A, C, F, L, N)

    dt = calculate_relative_traveltime(n, D)

    expected = A[..., None]

    np.testing.assert_allclose(dt, expected)


@pytest.mark.parametrize("direction", ["polar", "equatorial"])
def test_traveltime_equivalence_with_creager_isotropic(
    direction: Literal["polar", "equatorial"], rng: np.random.Generator
) -> None:
    """Test that the traveltime calculation matches Creager 1992 in the isotropic case.

    In the isotropic case,
        b = c = 0
        a = A = C = 2L + F = lambda + 2mu
        F = lambda
        L = N = mu
    The traveltime should be independent of direction and equal to a i.e the P-wave velocity.
    """
    lambda_, mu = rng.random(size=(2, 2, 4))
    lambda_plus_two_mu = lambda_ + 2 * mu
    A = lambda_plus_two_mu
    C = lambda_plus_two_mu
    F = lambda_
    L = mu
    N = mu

    D = isotropic_tensor(lambda_, mu)
    a, b, c = love_to_creager(direction, A, C, F, L, N)

    theta = rng.uniform(0, 2 * np.pi, size=10)
    n = np.stack([np.sin(theta), np.zeros_like(theta), np.cos(theta)], axis=-1)

    dt_tti = calculate_relative_traveltime(n, D)
    dt_creager = calc_dt_creager(theta, a, b, c)
    np.testing.assert_allclose(dt_tti, dt_creager)


def test_traveltime_equivalence_with_creager_transverse_isotropic_parallel(
    rng: np.random.Generator,
) -> None:
    """Test that the traveltime calculation matches Creager 1992 in the TI case when parallel to symmetry axis."""

    A, C, F, L, N = rng.random(size=(5, 2, 4))
    D = transverse_isotropic_tensor(A, C, F, L, N)
    a, b, c = love_to_creager("polar", A, C, F, L, N)

    theta = np.zeros(10)
    n = np.stack([np.sin(theta), np.zeros_like(theta), np.cos(theta)], axis=-1)

    dt_tti = calculate_relative_traveltime(n, D)
    dt_creager = calc_dt_creager(theta, a, b, c)

    np.testing.assert_allclose(dt_tti, dt_creager)


def test_traveltime_equivalence_with_creager_transverse_isotropic_perpendicular(
    rng: np.random.Generator,
) -> None:
    """Test that the traveltime calculation matches Creager 1992 in the TI case when perpendicular to symmetry axis.

    Creager is only valid for the case where the symmetry axis is vertical (eta1 = eta2 = 0).
    """
    A, C, F, L, N = rng.random(size=(5, 2, 4))
    D = transverse_isotropic_tensor(A, C, F, L, N)
    a, b, c = love_to_creager("equatorial", A, C, F, L, N)

    theta = np.array([np.pi / 2])
    n = np.stack([np.sin(theta), np.zeros_like(theta), np.cos(theta)], axis=-1)

    dt_tti = calculate_relative_traveltime(n, D)
    dt_creager = calc_dt_creager(theta, a, b, c)

    np.testing.assert_allclose(dt_tti, dt_creager)


class TestTravelTimeCalculator:
    """Test the TravelTimeCalculator class."""

    @pytest.fixture
    def valid_paths(self) -> tuple[np.ndarray, np.ndarray]:
        """Fixture for valid input paths."""
        ic_in = np.array(
            [
                [0.0, 0.0, 1.0],
                [90.0, 0.0, 1.0],
                [45.0, 45.0, 1.0],
                [180.0, -30.0, 1.0],
                [-90.0, 60.0, 1.0],
            ]
        )
        ic_out = np.array(
            [
                [180.0, 0.0, 1.0],
                [-90.0, 0.0, 1.0],
                [-180.0, -45.0, 1.0],
                [0.0, 30.0, 1.0],
                [90.0, -60.0, 1.0],
            ]
        )
        return ic_in, ic_out

    @pytest.fixture
    def calculator(
        self, valid_paths: tuple[np.ndarray, np.ndarray]
    ) -> TravelTimeCalculator:
        """Fixture for a TravelTimeCalculator instance with valid paths."""
        ic_in, ic_out = valid_paths
        return TravelTimeCalculator(ic_in, ic_out, nested=False, shear=True, N=True)

    def test_initialisation_npaths(self, calculator: TravelTimeCalculator) -> None:
        """Test that the class initialises correctly with valid inputs."""
        assert calculator.npaths == 5

    def test_initialisation_direction_vectors(
        self, calculator: TravelTimeCalculator
    ) -> None:
        """Test that the direction vectors are calculated correctly upon initialisation."""
        # Hard-coded expected unit direction vectors for the fixture paths
        expected_directions = np.array(
            [
                [-1.0, 0.0, 0.0],
                [0.0, -1.0, 0.0],
                [-0.626943121322, -0.259688343688, -0.734509555268],
                [0.866025403784, 0.0, 0.5],
                [0.0, 0.5, -0.866025403784],
            ]
        )
        np.testing.assert_allclose(
            calculator.path_directions, expected_directions, atol=1e-12
        )

    def test_initialisation_invalid_in_out_same(self) -> None:
        """Test that initialisation fails if an in coordinate is the same as the corresponding out coordinate."""
        ic_in = np.array([[1.0, 2.0, 3.0], [0.0, 0.0, 1.0]])
        ic_out = np.array([[0.0, 0.0, 1.0], [0.0, 0.0, 1.0]])
        with pytest.raises(ValueError):
            TravelTimeCalculator(ic_in, ic_out)

    def test_initialisation_inconsistent_npaths(self) -> None:
        """Test that initialisation fails if the number of in and out coordinates differ."""
        ic_in = np.array([[0.0, 0.0, 1.0]])
        ic_out = np.array([[180.0, 0.0, 1.0], [-90.0, 0.0, 1.0]])
        with pytest.raises(ValueError):
            TravelTimeCalculator(ic_in, ic_out)

    @pytest.mark.parametrize(
        "nsegments,batch_size",
        [(1, 1), (4, 1), (4, 2)],
        ids=["single_segment", "multiple_segments", "multiple_segments_batch"],
    )
    def test_call_isotropic_medium_single_segment(
        self, nsegments: int, batch_size: int, calculator: TravelTimeCalculator
    ) -> None:
        """Test traveltime calculation for isotropic medium."""
        lam, mu = 12.0, 5.0
        a = lam + 2 * mu
        m = np.stack(
            [np.tile(np.array([a, a, lam, mu, mu, 0.0, 0.0]), nsegments)] * batch_size
        )

        expected = lam + 2 * mu

        dt = calculator(m)
        assert dt.shape == (batch_size, calculator.npaths)
        np.testing.assert_allclose(dt, expected, atol=1e-12)

    def test_traveltime_calclulator_with_reference_love(
        self,
        calculator: TravelTimeCalculator,
        rng: np.random.Generator,
    ) -> None:
        """Test that the TravelTimeCalculator correctly adds reference Love parameters."""
        reference_love = rng.random(size=5)  # random reference Love parameters
        # In this test we reuse the shared `calculator` fixture and explicitly override
        # its `reference_love` attribute instead of constructing a new instance with
        # `reference_love=...` in __init__. This keeps the fixture usage simple while
        # still exercising the behavior with non-zero reference Love parameters.
        calculator.reference_love = reference_love  # override default zeros
        lam = 12.0
        mu = 5.0

        # determine perturbations that would yield isotropic medium when added to reference
        dA = (lam + 2 * mu) - reference_love[0]
        dC = (lam + 2 * mu) - reference_love[1]
        dF = lam - reference_love[2]
        dL = mu - reference_love[3]
        dN = mu - reference_love[4]

        nsegments = 3
        batch_size = 2
        m = np.stack(
            [np.tile(np.array([dA, dC, dF, dL, dN, 0.0, 0.0]), nsegments)] * batch_size
        )

        dt = calculator(m)
        expected = lam + 2 * mu

        assert dt.shape == (batch_size, calculator.npaths)
        np.testing.assert_allclose(dt, expected, atol=1e-12)

    @pytest.mark.parametrize(
        "nsegments,batch_size",
        [(1, 1), (4, 1), (4, 2)],
        ids=["single_segment", "multiple_segments", "multiple_segments_batch"],
    )
    def test_traveltime_calculator_with_normalisation(
        self,
        nsegments: int,
        batch_size: int,
        calculator: TravelTimeCalculator,
    ) -> None:
        """Test that the normalisation factor is applied correctly in the traveltime calculation.

        Testing the isotropic case.
        """
        # override fixture's default normalisation with a different value
        n = 2.0
        calculator.normalisation = n

        lam, mu = 12.0, 5.0
        a = lam + 2 * mu
        m = np.stack(
            [np.tile(np.array([a, a, lam, mu, mu, 0.0, 0.0]), nsegments)] * batch_size
        )

        expected = n * (lam + 2 * mu)

        dt = calculator(m)
        assert dt.shape == (batch_size, calculator.npaths)
        np.testing.assert_allclose(dt, expected, atol=1e-12)
