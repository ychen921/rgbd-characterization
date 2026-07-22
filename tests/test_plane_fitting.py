"""Tests for deterministic plane fitting and residual geometry."""

import numpy as np
import pytest

from src.geometry.plane_fitting import (
    PlaneModel,
    fit_plane_svd,
    point_to_plane_distances,
)


@pytest.fixture
def horizontal_points() -> np.ndarray:
    return np.array(
        [
            [-1.0, -1.0, 1.0],
            [1.0, -1.0, 1.0],
            [-1.0, 1.0, 1.0],
            [1.0, 1.0, 1.0],
        ],
        dtype=np.float64,
    )


def test_fit_plane_svd_fits_perfect_horizontal_plane(
    horizontal_points: np.ndarray,
) -> None:
    plane = fit_plane_svd(horizontal_points)

    assert np.allclose(plane.normal, [0.0, 0.0, 1.0])
    assert plane.d == pytest.approx(-1.0)
    assert np.allclose(plane.centroid, [0.0, 0.0, 1.0])
    assert np.linalg.norm(plane.normal) == pytest.approx(1.0)
    assert plane.normal[2] >= 0.0


def test_fit_plane_svd_fits_known_tilted_plane() -> None:
    x, y = np.meshgrid(
        np.linspace(-1.0, 1.0, 5),
        np.linspace(-1.0, 1.0, 4),
    )
    z = 1.0 + 0.1 * x
    points = np.column_stack((x.ravel(), y.ravel(), z.ravel()))
    expected_normal = np.array([-0.1, 0.0, 1.0])
    expected_normal /= np.linalg.norm(expected_normal)

    plane = fit_plane_svd(points)

    assert np.allclose(plane.normal, expected_normal, atol=1e-12)
    assert plane.d == pytest.approx(-expected_normal[2])
    assert np.max(np.abs(point_to_plane_distances(points, plane))) < 1e-12


def test_fit_plane_svd_estimates_noisy_plane_residual() -> None:
    random = np.random.default_rng(42)
    x, y = np.meshgrid(
        np.linspace(-0.5, 0.5, 40),
        np.linspace(-0.5, 0.5, 30),
    )
    z = 1.0 + random.normal(0.0, 0.002, size=x.shape)
    points = np.column_stack((x.ravel(), y.ravel(), z.ravel()))

    plane = fit_plane_svd(points)
    residuals = point_to_plane_distances(points, plane)
    rmse_m = np.sqrt(np.mean(residuals ** 2))

    assert np.allclose(plane.normal, [0.0, 0.0, 1.0], atol=5e-4)
    assert rmse_m == pytest.approx(0.002, rel=0.1)


def test_fit_plane_svd_returns_float64_geometry() -> None:
    points = np.array(
        [
            [0, 0, 1],
            [1, 0, 1],
            [0, 1, 1],
        ],
        dtype=np.int32,
    )

    plane = fit_plane_svd(points)

    assert plane.normal.dtype == np.float64
    assert plane.centroid.dtype == np.float64


@pytest.mark.parametrize(
    "points",
    [
        np.array(
            [
                [0.0, 0.0, 0.0],
                [0.0, 0.0, 0.0],
                [0.0, 0.0, 0.0],
            ]
        ),
        np.array(
            [
                [0.0, 0.0, 0.0],
                [1.0, 0.0, 0.0],
                [2.0, 0.0, 0.0],
            ]
        ),
    ],
)
def test_fit_plane_svd_rejects_degenerate_points(
    points: np.ndarray,
) -> None:
    with pytest.raises(ValueError, match="non-collinear"):
        fit_plane_svd(points)


def test_fit_plane_svd_rejects_fewer_than_three_points() -> None:
    points = np.zeros((2, 3), dtype=np.float64)

    with pytest.raises(ValueError, match="At least 3"):
        fit_plane_svd(points)


@pytest.mark.parametrize(
    "points",
    [
        np.zeros((3,), dtype=np.float64),
        np.zeros((3, 2), dtype=np.float64),
        np.zeros((3, 4), dtype=np.float64),
    ],
)
def test_fit_plane_svd_rejects_invalid_shape(points: np.ndarray) -> None:
    with pytest.raises(ValueError, match=r"shape \(N, 3\)"):
        fit_plane_svd(points)


def test_fit_plane_svd_rejects_non_array_input() -> None:
    with pytest.raises(TypeError, match="numpy.ndarray"):
        fit_plane_svd([[0, 0, 1], [1, 0, 1], [0, 1, 1]])


@pytest.mark.parametrize("invalid_value", [np.nan, np.inf, -np.inf])
def test_fit_plane_svd_rejects_non_finite_points(
    invalid_value: float,
) -> None:
    points = np.array(
        [
            [0.0, 0.0, 1.0],
            [1.0, 0.0, 1.0],
            [0.0, 1.0, invalid_value],
        ]
    )

    with pytest.raises(ValueError, match="finite"):
        fit_plane_svd(points)


def test_fit_plane_svd_rejects_non_numeric_points() -> None:
    points = np.array(
        [["0", "0", "1"], ["1", "0", "1"], ["0", "1", "1"]]
    )

    with pytest.raises(TypeError, match="numeric"):
        fit_plane_svd(points)


def test_fit_plane_svd_rejects_complex_points() -> None:
    points = np.array(
        [[0, 0, 1], [1, 0, 1], [0, 1, 1]],
        dtype=np.complex128,
    )

    with pytest.raises(TypeError, match="real-valued"):
        fit_plane_svd(points)


def test_point_to_plane_distances_returns_signed_metres() -> None:
    plane = PlaneModel(
        normal=np.array([0.0, 0.0, 1.0]),
        d=-1.0,
        centroid=np.array([0.0, 0.0, 1.0]),
    )
    points = np.array(
        [
            [0.0, 0.0, 0.99],
            [0.0, 0.0, 1.00],
            [0.0, 0.0, 1.01],
        ]
    )

    distances = point_to_plane_distances(points, plane)

    assert distances.shape == (3,)
    assert np.allclose(distances, [-0.01, 0.0, 0.01])


def test_point_to_plane_distances_accepts_empty_point_array() -> None:
    plane = PlaneModel(
        normal=np.array([0.0, 0.0, 1.0]),
        d=-1.0,
        centroid=np.array([0.0, 0.0, 1.0]),
    )

    distances = point_to_plane_distances(
        np.empty((0, 3), dtype=np.float64),
        plane,
    )

    assert distances.shape == (0,)


def test_point_to_plane_distances_rejects_invalid_plane(
    horizontal_points: np.ndarray,
) -> None:
    with pytest.raises(TypeError, match="PlaneModel"):
        point_to_plane_distances(horizontal_points, object())


@pytest.mark.parametrize(
    ("arguments", "expected_exception", "expected_message"),
    [
        (
            {
                "normal": [0.0, 0.0, 1.0],
                "d": -1.0,
                "centroid": np.array([0.0, 0.0, 1.0]),
            },
            TypeError,
            "normal must be a numpy.ndarray",
        ),
        (
            {
                "normal": np.array([0.0, 1.0]),
                "d": -1.0,
                "centroid": np.array([0.0, 0.0, 1.0]),
            },
            ValueError,
            "normal must have shape",
        ),
        (
            {
                "normal": np.array([0.0, 0.0, 2.0]),
                "d": -1.0,
                "centroid": np.array([0.0, 0.0, 1.0]),
            },
            ValueError,
            "unit length",
        ),
        (
            {
                "normal": np.array([0.0, 0.0, -1.0]),
                "d": 1.0,
                "centroid": np.array([0.0, 0.0, 1.0]),
            },
            ValueError,
            "non-negative camera Z",
        ),
        (
            {
                "normal": np.array([0.0, 0.0, 1.0]),
                "d": np.nan,
                "centroid": np.array([0.0, 0.0, 1.0]),
            },
            ValueError,
            "d must be a finite number",
        ),
        (
            {
                "normal": np.array([0.0, 0.0, 1.0]),
                "d": -1.0,
                "centroid": np.array([0.0, np.inf, 1.0]),
            },
            ValueError,
            "centroid must contain only finite values",
        ),
    ],
)
def test_plane_model_rejects_invalid_representation(
    arguments: dict[str, object],
    expected_exception: type[Exception],
    expected_message: str,
) -> None:
    with pytest.raises(expected_exception, match=expected_message):
        PlaneModel(**arguments)
