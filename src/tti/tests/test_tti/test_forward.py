"""Test the forward modelling functions for TTI media."""

from typing import Literal

import numpy as np
import pytest

from tti.creager import calculate_traveltime as calc_dt_creager
from tti.creager import love_to_creager
from tti.elastic import (
    _check_elastic_tensor_symmetry,
    elastic_tensor_to_voigt,
    isotropic_tensor_4th,
    transformation_to_voigt,
    transverse_isotropic_tensor_4th,
    transverse_isotropic_tensor_voigt,
)
from tti.forward import (
    TravelTimeCalculator,
    _spherical_to_cartesian,
    _unpack_model_vector,
    _unpack_nested_model_vector,
    calculate_path_direction_vector,
    calculate_relative_traveltime,
    construct_general_tti_tensor,
)
from tti.rotation import rotation_matrix_zy, transformation_4th_order


@pytest.fixture
def A() -> np.ndarray:
    """Fixture for elastic constant A."""
    return np.array([10.0])


@pytest.fixture
def C() -> np.ndarray:
    """Fixture for elastic constant C."""
    return np.array([15.0])


@pytest.fixture
def F() -> np.ndarray:
    """Fixture for elastic constant F."""
    return np.array([8.0])


@pytest.fixture
def L() -> np.ndarray:
    """Fixture for elastic constant L."""
    return np.array([0.15])


@pytest.fixture
def N() -> np.ndarray:
    """Fixture for elastic constant N."""
    return np.array([0.1])


@pytest.fixture
def eta1() -> np.ndarray:
    """Fixture for rotation angle eta1."""
    return np.array([np.pi / 4])


@pytest.fixture
def eta2() -> np.ndarray:
    """Fixture for rotation angle eta2."""
    return np.array([np.pi / 6])


@pytest.fixture
def C4(
    A: np.ndarray,
    C: np.ndarray,
    F: np.ndarray,
    L: np.ndarray,
    N: np.ndarray,
    eta1: np.ndarray,
    eta2: np.ndarray,
) -> np.ndarray:
    """Fixture for a sample rotated TTI elastic tensor."""
    return construct_general_tti_tensor(A, C, F, L, N, eta1, eta2)


@pytest.fixture
def C6(
    A: np.ndarray,
    C: np.ndarray,
    F: np.ndarray,
    L: np.ndarray,
    N: np.ndarray,
    eta1: np.ndarray,
    eta2: np.ndarray,
) -> np.ndarray:
    """Fixture for a sample rotated TTI elastic tensor in Voigt notation."""
    C_voigt = transverse_isotropic_tensor_voigt(A, C, F, L, N)
    rotation_matrix = rotation_matrix_zy(eta1, eta2)
    R = transformation_to_voigt(transformation_4th_order(rotation_matrix))
    return np.einsum("...ij,...jk,...kl->...il", R, C_voigt, R.transpose(0, 2, 1))


def test_construct_general_tti_tensor_shape(C4: np.ndarray) -> None:
    """Test that the constructed TTI tensor has the correct shape.

    It should return a 4th-order tensor of shape (3, 3, 3, 3).
    """

    assert C4.shape == (1, 3, 3, 3, 3)


def test_construct_general_tti_tensor_is_symmetric(C4: np.ndarray) -> None:
    """Test that the symmetries are preserved under rotation."""

    assert _check_elastic_tensor_symmetry(C4)


def test_4th_order_gives_same_as_voigt(C4: np.ndarray, C6: np.ndarray) -> None:
    """Test that the 4th-order and Voigt implementations give the same result."""

    np.testing.assert_allclose(elastic_tensor_to_voigt(C4), C6)


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

    # Batch of 3 normalised ray directions
    n_batch = np.array(
        [
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0],
        ]
    )

    dt_batch = calculate_relative_traveltime(n_batch, D)

    expected = np.array([1.0, 2.0, 3.0])

    assert dt_batch.shape == (3,)
    np.testing.assert_allclose(dt_batch, expected)


def test_traveltime_isotropic_independent_of_direction(
    rng: np.random.Generator,
) -> None:
    """Isotropic perturbation gives same result for all ray directions."""

    lam = np.array([12.0])
    mu = np.array([5.0])
    D = isotropic_tensor_4th(lam, mu)

    # Try several random directions
    directions = rng.normal(size=(10, 3))
    directions = directions / np.linalg.norm(directions, axis=1, keepdims=True)

    results = calculate_relative_traveltime(directions, D)

    # All should be equal for isotropic medium
    np.testing.assert_allclose(results, results[0])


def test_traveltime_linear_in_perturbation(
    A: float, C: float, F: float, L: float, N: float
) -> None:
    """Doubling the perturbation tensor doubles the traveltime anomaly."""

    D = transverse_isotropic_tensor_4th(A, C, F, L, N)
    n = np.array([1.0, 0.0, 0.0])

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
    n = np.array([0.6, 0.0, 0.8])  # normalised
    expected = 1.0 * (0.6**4) + 3.0 * (0.8**4)

    dt = calculate_relative_traveltime(n, D)

    np.testing.assert_allclose(dt, expected)


def test_traveltime_antiparallel_rays_equal(
    A: float, C: float, F: float, L: float, N: float
) -> None:
    """Ray and its opposite give the same traveltime (even power)."""

    D = transverse_isotropic_tensor_4th(A, C, F, L, N)

    n = np.array([0.6, 0.8, 0.0])
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


def test_traveltime_transverse_isotropic_polar(
    A: float, C: float, F: float, L: float, N: float
) -> None:
    """Test traveltime calculation for a polar path in a TI medium.

    TI medium => eta1 = eta2 = 0

    Expected result is the C_33 component of the elastic tensor.
    """

    D = construct_general_tti_tensor(A, C, F, L, N, 0.0, 0.0)
    n = np.array([0.0, 0.0, 1.0])  # along symmetry axis

    dt = calculate_relative_traveltime(n, D)

    expected = C

    np.testing.assert_allclose(dt, expected)


@pytest.mark.parametrize(
    "n", [np.array([1.0, 0.0, 0.0]), np.array([0.0, 1.0, 0.0])]
)  # perpendicular directions
def test_traveltime_transverse_isotropic_equatorial(
    n: np.ndarray, A: float, C: float, F: float, L: float, N: float
) -> None:
    """Test traveltime calculation for an equatorial path in a TI medium.

    TI medium => eta1 = eta2 = 0

    Expected result is the C_11 component of the elastic tensor.
    """

    D = construct_general_tti_tensor(A, C, F, L, N, 0.0, 0.0)

    dt = calculate_relative_traveltime(n, D)

    expected = A

    np.testing.assert_allclose(dt, expected)


@pytest.mark.parametrize("direction", ["polar", "equatorial"])
def test_traveltime_equivalence_with_creager_isotropic(
    direction: Literal["polar", "equatorial"],
    F: float,
    L: float,
    rng: np.random.Generator,
) -> None:
    """Test that the traveltime calculation matches Creager 1992 in the isotropic case.

    Creager is only valid for the case where the symmetry axis is vertical (eta1 = eta2 = 0).

    In the isotropic case,
        b = c = 0
        a = A = C = 2L + F = lambda + 2mu
        F = lambda
        L = N = mu
    The traveltime should be independent of direction and equal to a i.e the P-wave velocity.
    """
    lambda_ = F
    lambda_plus_two_mu = 2 * L + F
    mu = L
    D = construct_general_tti_tensor(
        lambda_plus_two_mu, lambda_plus_two_mu, lambda_, mu, mu, 0.0, 0.0
    )
    a, b, c = love_to_creager(
        direction, lambda_plus_two_mu, lambda_plus_two_mu, lambda_, mu, mu
    )

    theta = rng.uniform(0, 2 * np.pi)
    n = np.array([np.sin(theta), 0.0, np.cos(theta)])

    dt_tti = calculate_relative_traveltime(n, D)
    dt_creager = calc_dt_creager(theta, a, b, c)
    np.testing.assert_allclose(dt_tti, dt_creager)


def test_traveltime_equivalence_with_creager_transverse_isotropic_parallel(
    A: float, C: float, F: float, L: float, N: float
) -> None:
    """Test that the traveltime calculation matches Creager 1992 in the TI case when parallel to symmetry axis.

    Creager is only valid for the case where the symmetry axis is vertical (eta1 = eta2 = 0).
    """

    D = construct_general_tti_tensor(A, C, F, L, N, 0.0, 0.0)
    a, b, c = love_to_creager("polar", A, C, F, L, N)

    theta = 0
    n = np.array([np.sin(theta), 0.0, np.cos(theta)])

    dt_tti = calculate_relative_traveltime(n, D)
    dt_creager = calc_dt_creager(theta, a, b, c)

    np.testing.assert_allclose(dt_tti, dt_creager)


def test_traveltime_equivalence_with_creager_transverse_isotropic_perpendicular(
    A: float, C: float, F: float, L: float, N: float
) -> None:
    """Test that the traveltime calculation matches Creager 1992 in the TI case when perpendicular to symmetry axis.

    Creager is only valid for the case where the symmetry axis is vertical (eta1 = eta2 = 0).
    """

    D = construct_general_tti_tensor(A, C, F, L, N, 0.0, 0.0)
    a, b, c = love_to_creager("equatorial", A, C, F, L, N)

    theta = np.pi / 2
    n = np.array([np.sin(theta), 0.0, np.cos(theta)])

    dt_tti = calculate_relative_traveltime(n, D)
    dt_creager = calc_dt_creager(theta, a, b, c)

    np.testing.assert_allclose(dt_tti, dt_creager)


def test__spherical_to_cartesian() -> None:
    """Test conversion from spherical to cartesian coordinates."""

    lon, lat, r = 0.0, 0.0, 1.0
    x, y, z = _spherical_to_cartesian(lon, lat, r)
    np.testing.assert_allclose([x, y, z], [1.0, 0.0, 0.0], atol=1e-12)

    lon, lat, r = 90.0, 0.0, 1.0
    x, y, z = _spherical_to_cartesian(lon, lat, r)
    np.testing.assert_allclose([x, y, z], [0.0, 1.0, 0.0], atol=1e-12)

    lon, lat, r = 0.0, 90.0, 1.0
    x, y, z = _spherical_to_cartesian(lon, lat, r)
    np.testing.assert_allclose([x, y, z], [0.0, 0.0, 1.0], atol=1e-12)


def test__spherical_to_cartesian_batch() -> None:
    """Test conversion from spherical to cartesian coordinates with batch inputs."""

    # Batch of 3 coordinates
    coords_spherical = np.array([[0.0, 0.0, 1.0], [90.0, 0.0, 1.0], [0.0, 90.0, 1.0]])
    coords_cartesian = _spherical_to_cartesian(
        coords_spherical[:, 0], coords_spherical[:, 1], coords_spherical[:, 2]
    )
    expected = np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]])
    np.testing.assert_allclose(coords_cartesian, expected, atol=1e-12)

    # Single coordinate as 1-element arrays
    lon, lat, r = np.array([0.0]), np.array([0.0]), np.array([1.0])
    coords = _spherical_to_cartesian(lon, lat, r)
    expected = np.array([[1.0, 0.0, 0.0]])
    np.testing.assert_allclose(coords, expected, atol=1e-12)


@pytest.mark.parametrize(
    "ic_in, ic_out, expected",
    [
        (
            np.array([0.0, 0.0, 1.0]),  # in at equator prime meridian
            np.array([180.0, 0.0, 1.0]),  # out at the antipode
            np.array([-1.0, 0.0, 0.0]),
        ),
        (
            np.array([90.0, 0.0, 1.0]),  # in at equator 90 degrees east
            np.array([0.0, 0.0, 1.0]),  # out at the prime meridian
            np.array([1.0, -1.0, 0.0]) / np.sqrt(2),
        ),
        (
            np.array([0.0, 90.0, 1.0]),  # in at north pole
            np.array([0.0, -90.0, 1.0]),  # out at south pole
            np.array([0.0, 0.0, -1.0]),
        ),
        (  # off-centre polar path
            np.array([45.0, 45.0, 1.0]),  # in at 45N 45E
            np.array([45.0, -45.0, 1.0]),  # out at antipode
            np.array([0.0, 0.0, -1.0]),
        ),
    ],
)
def test_calculate_path_direction_vector(
    ic_in: np.ndarray, ic_out: np.ndarray, expected: np.ndarray
) -> None:
    """Test calculation of path direction unit vector."""
    n = calculate_path_direction_vector(ic_in, ic_out)
    np.testing.assert_allclose(n, expected, atol=1e-12)


@pytest.mark.parametrize(
    "ic_in_batch, ic_out_batch, expected_batch",
    [
        # Batch with 2 paths: east-west and north-south
        (
            np.array([[0.0, 0.0, 1.0], [90.0, 0.0, 1.0]]),
            np.array([[180.0, 0.0, 1.0], [270.0, 0.0, 1.0]]),
            np.array([[-1.0, 0.0, 0.0], [0.0, -1.0, 0.0]]),
        ),
        # Batch with 3 paths including diagonal
        (
            np.array([[0.0, 0.0, 1.0], [90.0, 0.0, 1.0], [45.0, 0.0, 1.0]]),
            np.array([[180.0, 0.0, 1.0], [270.0, 0.0, 1.0], [225.0, 0.0, 1.0]]),
            np.array(
                [
                    [-1.0, 0.0, 0.0],
                    [0.0, -1.0, 0.0],
                    [-1 / np.sqrt(2), -1 / np.sqrt(2), 0.0],
                ]
            ),
        ),
        # Single path as 1×3 array
        (
            np.array([[0.0, 0.0, 1.0]]),
            np.array([[180.0, 0.0, 1.0]]),
            np.array([[-1.0, 0.0, 0.0]]),
        ),
    ],
)
def test_calculate_path_direction_vector_batch(
    ic_in_batch: np.ndarray, ic_out_batch: np.ndarray, expected_batch: np.ndarray
) -> None:
    """Test calculation of path direction unit vectors for batch inputs."""
    n_batch = calculate_path_direction_vector(ic_in_batch, ic_out_batch)
    assert n_batch.shape == expected_batch.shape
    np.testing.assert_allclose(n_batch, expected_batch, atol=1e-12)


class TestTravelTimeCalculator:
    """Test the TravelTimeCalculator class."""

    @pytest.fixture
    def valid_paths(self) -> tuple[np.ndarray, np.ndarray]:
        """Fixture for valid input paths."""
        ic_in = np.array([[0.0, 0.0, 1.0], [90.0, 0.0, 1.0]])
        ic_out = np.array([[180.0, 0.0, 1.0], [-90.0, 0.0, 1.0]])
        return ic_in, ic_out

    @pytest.fixture
    def calculator(
        self, valid_paths: tuple[np.ndarray, np.ndarray]
    ) -> TravelTimeCalculator:
        """Fixture for a TravelTimeCalculator instance with valid paths."""
        ic_in, ic_out = valid_paths
        return TravelTimeCalculator(ic_in, ic_out)

    def test_initialisation_npaths(self, calculator: TravelTimeCalculator) -> None:
        """Test that the class initialises correctly with valid inputs."""
        assert calculator.npaths == 2

    def test_initialisation_direction_vectors(
        self, calculator: TravelTimeCalculator
    ) -> None:
        """Test that the direction vectors are calculated correctly upon initialisation."""
        expected_directions = np.array([[-1.0, 0.0, 0.0], [0.0, -1.0, 0.0]])
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

    def test_call_isotropic_medium(self, calculator: TravelTimeCalculator) -> None:
        """Test traveltime calculation for isotropic medium."""
        # Isotropic medium parameters
        lam, mu = 12.0, 5.0
        a = lam + 2 * mu
        m = np.array([a, a, lam, mu, mu, 0.0, 0.0] * 2)

        dt = calculator(m)

        # For isotropic medium, traveltime should be the same for all paths
        np.testing.assert_allclose(dt, dt[0], atol=1e-12)


def test__unpack_nested_model_vector() -> None:
    """Test unpacking of nested model vector into Love parameters.

    Testing the case where the model is isotropic, so C=A, F=A-2N, N=L, eta1=whatever, eta2=whatever.
    """

    m_nested = np.array(
        [
            [10.0, 0.0, 0.0, 0.0, 0.15, 45, 90],
            [12.0, 0.0, 0.0, 0.0, 0.2, 60, -90],
        ]
    )

    A, C, F, L, N, eta1, eta2 = _unpack_nested_model_vector(m_nested)

    expected_A = np.array([10.0, 12.0])
    expected_N = np.array([0.15, 0.2])
    expected_C = expected_A
    expected_F = expected_A - 2 * expected_N
    expected_L = expected_N
    expected_eta1 = np.array([np.pi / 4, np.pi / 3])
    expected_eta2 = np.array([np.pi / 2, -np.pi / 2])

    np.testing.assert_allclose(A, expected_A)
    np.testing.assert_allclose(C, expected_C)
    np.testing.assert_allclose(F, expected_F)
    np.testing.assert_allclose(L, expected_L)
    np.testing.assert_allclose(N, expected_N)
    np.testing.assert_allclose(eta1, expected_eta1)
    np.testing.assert_allclose(eta2, expected_eta2)


def test__unpack_model_vector() -> None:
    """Test unpacking of model vector into Love parameters.

    Testing the case where the model is isotropic, so C=A, F=A-2N, N=L.
    This function just unpacks the model directly without nested differences.
    """

    m = np.array(
        [
            [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0],  # dummy row to check indexing
            [8.0, 9.0, 10.0, 11.0, 12.0, 13.0, 14.0],  # dummy row to check indexing
        ]
    )

    A, C, F, L, N, eta1, eta2 = _unpack_model_vector(m)

    expected_A = np.array([1.0, 8.0])
    expected_C = np.array([2.0, 9.0])
    expected_F = np.array([3.0, 10.0])
    expected_L = np.array([4.0, 11.0])
    expected_N = np.array([5.0, 12.0])
    expected_eta1 = np.radians(np.array([6.0, 13.0]))
    expected_eta2 = np.radians(np.array([7.0, 14.0]))

    np.testing.assert_allclose(A, expected_A)
    np.testing.assert_allclose(C, expected_C)
    np.testing.assert_allclose(F, expected_F)
    np.testing.assert_allclose(L, expected_L)
    np.testing.assert_allclose(N, expected_N)
    np.testing.assert_allclose(eta1, expected_eta1)
    np.testing.assert_allclose(eta2, expected_eta2)
