"""Test the rotation matrices."""

from collections.abc import Callable

import numpy as np
import pytest

from tti.rotation import (
    gradient_rotation_matrix_x,
    gradient_rotation_matrix_y,
    gradient_rotation_matrix_z,
    gradient_rotation_matrix_zy,
    rotation_matrix_x,
    rotation_matrix_y,
    rotation_matrix_z,
    rotation_matrix_zy,
)


@pytest.mark.parametrize("R", [rotation_matrix_z, rotation_matrix_y, rotation_matrix_x])
@pytest.mark.parametrize(
    "shape",
    [
        (),
        (5,),
        (2, 3),
    ],
    ids=["scalar", "1d", "2d"],
)
def test_rotation_matrix_0(
    R: Callable[[float | np.ndarray], np.ndarray], shape: tuple[int, ...]
) -> None:
    """Test rotation matrices at 0 radians for various input shapes."""

    angles = np.zeros(shape)
    result = R(angles)

    expected_shape = shape + (3, 3)
    assert result.shape == expected_shape
    expected = np.broadcast_to(np.eye(3), expected_shape)
    np.testing.assert_array_almost_equal(result, expected)


@pytest.mark.parametrize(
    ["R", "expected_base"],
    [
        (rotation_matrix_z, np.array([[0, -1, 0], [1, 0, 0], [0, 0, 1]])),
        (rotation_matrix_y, np.array([[0, 0, 1], [0, 1, 0], [-1, 0, 0]])),
        (rotation_matrix_x, np.array([[1, 0, 0], [0, 0, -1], [0, 1, 0]])),
    ],
)
@pytest.mark.parametrize(
    "shape",
    [
        (),
        (5,),
        (2, 3),
    ],
    ids=["scalar", "1d", "2d"],
)
def test_rotation_matrix_90(
    R: Callable[[float | np.ndarray], np.ndarray],
    expected_base: np.ndarray,
    shape: tuple[int, ...],
) -> None:
    """Test rotation matrices at 90 degrees for various input shapes."""

    angle = np.pi / 2
    angles = np.full(shape, angle)
    result = R(angles)

    expected_shape = shape + (3, 3)
    assert result.shape == expected_shape

    # Broadcast expected matrix to match output shape
    expected = np.broadcast_to(expected_base, expected_shape)
    np.testing.assert_array_almost_equal(result, expected)


@pytest.mark.parametrize("R", [rotation_matrix_z, rotation_matrix_y, rotation_matrix_x])
@pytest.mark.parametrize(
    "shape",
    [
        (),
        (5,),
        (2, 3),
    ],
    ids=["scalar", "1d", "2d"],
)
def test_rotation_matrices_orthogonal(
    R: Callable[[float | np.ndarray], np.ndarray],
    shape: tuple[int, ...],
    rng: np.random.Generator,
) -> None:
    """Test that rotation matrices are orthogonal for various input shapes."""

    angles = rng.uniform(0, 2 * np.pi, size=shape)
    R_mat = R(angles)
    RRT = np.einsum("...ij,...jk->...ik", R_mat, np.swapaxes(R_mat, -2, -1))

    expected_shape = shape + (3, 3)
    assert R_mat.shape == expected_shape
    expected = np.broadcast_to(np.eye(3), expected_shape)
    np.testing.assert_array_almost_equal(RRT, expected)


@pytest.mark.parametrize("R", [rotation_matrix_z, rotation_matrix_y, rotation_matrix_x])
@pytest.mark.parametrize(
    "shape",
    [
        (),
        (5,),
        (2, 3),
    ],
    ids=["scalar", "1d", "2d"],
)
def test_rotation_matrices_determinant(
    R: Callable[[float | np.ndarray], np.ndarray],
    shape: tuple[int, ...],
    rng: np.random.Generator,
) -> None:
    """Test that rotation matrices have determinant 1 for various input shapes."""

    angles = rng.uniform(0, 2 * np.pi, size=shape)
    R_mat = R(angles)
    dets = np.linalg.det(R_mat)

    np.testing.assert_array_almost_equal(dets, np.ones(shape))


@pytest.mark.parametrize("R", [rotation_matrix_z, rotation_matrix_y, rotation_matrix_x])
@pytest.mark.parametrize(
    "shape",
    [
        (),
        (5,),
        (2, 3),
    ],
    ids=["scalar", "1d", "2d"],
)
def test_rotation_matrices_composition(
    R: Callable[[float | np.ndarray], np.ndarray],
    shape: tuple[int, ...],
    rng: np.random.Generator,
) -> None:
    """Test that composition of rotation matrices is correct for various input shapes."""

    angle1 = rng.uniform(0, 2 * np.pi, size=shape)
    angle2 = rng.uniform(0, 2 * np.pi, size=shape)

    R1 = R(angle1)
    R2 = R(angle2)
    R_combined = R(angle1 + angle2)

    R_composed = np.einsum("...ij,...jk->...ik", R1, R2)
    np.testing.assert_array_almost_equal(R_composed, R_combined)


@pytest.mark.parametrize("R", [rotation_matrix_z, rotation_matrix_y, rotation_matrix_x])
@pytest.mark.parametrize(
    "shape",
    [
        (),
        (5,),
        (2, 3),
    ],
    ids=["scalar", "1d", "2d"],
)
def test_rotation_matrices_inverse(
    R: Callable[[float | np.ndarray], np.ndarray],
    shape: tuple[int, ...],
    rng: np.random.Generator,
) -> None:
    """Test that the inverse of rotation matrices is their transpose for various input shapes."""

    angles = rng.uniform(0, 2 * np.pi, size=shape)
    R_mat = R(angles)
    R_inv = np.linalg.inv(R_mat)
    R_T = np.swapaxes(R_mat, -2, -1)

    np.testing.assert_array_almost_equal(R_inv, R_T)


@pytest.mark.parametrize(
    "shape",
    [
        (),
        (5,),
        (2, 3),
    ],
    ids=["scalar", "1d", "2d"],
)
def test_rotation_matrix_zy(shape: tuple[int, ...], rng: np.random.Generator) -> None:
    """Test combined rotation matrices around z and y axes for various input shapes.

    The combined rotation matrix Rzy should be equivalent to first rotating
    around the z-axis by angle_z and then around the y-axis by angle_y.

    This should match what is given by Brett et al., 2024, Eq 8 (https://www.nature.com/articles/s41561-024-01539-6#Sec6) to ensure consistency in TTI rotation conventions.
    """

    angle_z = rng.uniform(0, 2 * np.pi, size=shape)  # colatitude
    angle_y = rng.uniform(-2 * np.pi, 2 * np.pi, size=shape)  # longitude

    Rzy = rotation_matrix_zy(angle_z, angle_y)

    expected_shape = shape + (3, 3)
    assert Rzy.shape == expected_shape

    # from Brett et al., 2024, Eq 8
    c1 = np.cos(angle_z)
    s1 = np.sin(angle_z)
    c2 = np.cos(angle_y)
    s2 = np.sin(angle_y)
    expected = np.zeros(expected_shape, dtype=float)
    expected[..., 0, 0] = c1 * c2
    expected[..., 0, 1] = -s1
    expected[..., 0, 2] = s2 * c1
    expected[..., 1, 0] = s1 * c2
    expected[..., 1, 1] = c1
    expected[..., 1, 2] = s1 * s2
    expected[..., 2, 0] = -s2
    expected[..., 2, 1] = 0.0
    expected[..., 2, 2] = c2

    np.testing.assert_array_almost_equal(Rzy, expected)


@pytest.mark.parametrize("R", [rotation_matrix_z, rotation_matrix_y, rotation_matrix_x])
def test_batch_rotations(
    R: Callable[[np.ndarray], np.ndarray], rng: np.random.Generator
) -> None:
    """Test that rotation matrices can handle batches of angles.

    Batches can be ND, with leading dimensions handled flexibly.
    """

    leading_shape = (2, 5)
    angles = rng.uniform(0, 2 * np.pi, size=leading_shape)

    R_batch = R(angles)
    assert R_batch.shape == leading_shape + (3, 3)


def fd_grad(fun, x, eps=1e-6):
    """Finite-difference derivative of a function `fun` with respect to each element of `x`.

    `fun` should accept an array `x` and return an array of shape `x.shape + (3,3)`.
    The return value has shape `x.shape + (3,3)` containing the partial derivative
    of `fun` with respect to each element of `x` at the corresponding position.
    """
    x = np.asarray(x, dtype=float)
    out = np.zeros(x.shape + (3, 3), dtype=float)
    for idx, _ in np.ndenumerate(x):
        x_plus = x.copy()
        x_minus = x.copy()
        x_plus[idx] += eps
        x_minus[idx] -= eps
        f_plus = fun(x_plus)
        f_minus = fun(x_minus)
        out[idx] = (f_plus[idx] - f_minus[idx]) / (2 * eps)
    return out


def fd_grad_twoarg(fun, a, b, eps=1e-6):
    """Finite-difference partials of `fun(a, b)` with respect to `a` and `b`.

    Returns tuple `(d_fun_da, d_fun_db)` each shaped `a.shape + (3,3)`.
    """
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    d_a = np.zeros(a.shape + (3, 3), dtype=float)
    d_b = np.zeros(a.shape + (3, 3), dtype=float)
    for idx, _ in np.ndenumerate(a):
        a_plus = a.copy()
        a_minus = a.copy()
        a_plus[idx] += eps
        a_minus[idx] -= eps
        f_plus = fun(a_plus, b)
        f_minus = fun(a_minus, b)
        d_a[idx] = (f_plus[idx] - f_minus[idx]) / (2 * eps)

        b_plus = b.copy()
        b_minus = b.copy()
        b_plus[idx] += eps
        b_minus[idx] -= eps
        f_plus = fun(a, b_plus)
        f_minus = fun(a, b_minus)
        d_b[idx] = (f_plus[idx] - f_minus[idx]) / (2 * eps)

    return d_a, d_b


@pytest.mark.parametrize(
    "shape",
    [
        (),
        (5,),
        (2, 3),
    ],
    ids=["scalar", "1d", "2d"],
)
def test_gradient_rotation_matrix_z_matches_finite_difference(
    shape: tuple[int, ...], rng: np.random.Generator
):
    """Test that the gradient of the z-rotation matrix matches finite-difference gradients for various input shapes."""
    angles = rng.uniform(-2.0, 2.0, size=shape)
    dR_numeric = fd_grad(lambda th: rotation_matrix_z(th), angles)
    dR_analytic = gradient_rotation_matrix_z(angles)
    assert np.allclose(dR_numeric, dR_analytic, atol=1e-6)


@pytest.mark.parametrize(
    "shape",
    [
        (),
        (5,),
        (2, 3),
    ],
    ids=["scalar", "1d", "2d"],
)
def test_gradient_rotation_matrix_y_matches_finite_difference(
    shape: tuple[int, ...], rng: np.random.Generator
):
    """Test that the gradient of the y-rotation matrix matches finite-difference gradients for various input shapes."""
    angles = rng.uniform(-2.0, 2.0, size=shape)
    dR_numeric = fd_grad(lambda th: rotation_matrix_y(th), angles)
    dR_analytic = gradient_rotation_matrix_y(angles)
    assert np.allclose(dR_numeric, dR_analytic, atol=1e-6)


@pytest.mark.parametrize(
    "shape",
    [
        (),
        (5,),
        (2, 3),
    ],
    ids=["scalar", "1d", "2d"],
)
def test_gradient_rotation_matrix_x_matches_finite_difference(
    shape: tuple[int, ...], rng: np.random.Generator
):
    """Test that the gradient of the x-rotation matrix matches finite-difference gradients for various input shapes."""
    angles = rng.uniform(-2.0, 2.0, size=shape)
    dR_numeric = fd_grad(lambda th: rotation_matrix_x(th), angles)
    dR_analytic = gradient_rotation_matrix_x(angles)
    assert np.allclose(dR_numeric, dR_analytic, atol=1e-6)


@pytest.mark.parametrize(
    "shape",
    [
        (),
        (5,),
        (2, 3),
    ],
    ids=["scalar", "1d", "2d"],
)
def test_gradient_rotation_matrix_zy_matches_finite_difference(
    shape: tuple[int, ...], rng: np.random.Generator
) -> None:
    """Test that the gradient of the combined z-y rotation matrix matches finite-difference gradients for various input shapes."""
    alpha = rng.uniform(-2.0, 2.0, size=shape)
    beta = rng.uniform(-2.0, 2.0, size=shape)
    d_alpha_num, d_beta_num = fd_grad_twoarg(
        lambda a, b: rotation_matrix_zy(a, b), alpha, beta
    )
    d_alpha_analytic, d_beta_analytic = gradient_rotation_matrix_zy(alpha, beta)
    assert np.allclose(d_alpha_num, d_alpha_analytic, atol=1e-6)
    assert np.allclose(d_beta_num, d_beta_analytic, atol=1e-6)
