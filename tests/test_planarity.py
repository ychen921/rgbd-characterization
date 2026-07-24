"""Tests for single-frame geometric planarity metrics."""

import numpy as np
import pytest

from src.geometry.camera import CameraIntrinsics
from src.metrics.planarity import (
    DEFAULT_INLIER_THRESHOLD_MM,
    DEFAULT_MIN_VALID_POINTS,
    FramePlaneResult,
    PlanarityResult,
    compute_frame_planarity,
    compute_planarity,
)


@pytest.fixture
def intrinsics() -> CameraIntrinsics:
    return CameraIntrinsics(
        width=12,
        height=10,
        fx=100.0,
        fy=120.0,
        cx=5.5,
        cy=4.5,
        frame_id="camera_depth_optical_frame",
    )


def _depth_roi_for_plane(
    *,
    intrinsics: CameraIntrinsics,
    normal: np.ndarray,
    d: float,
    roi_x: int,
    roi_y: int,
    height: int,
    width: int,
) -> np.ndarray:
    """Render ideal depth values for a camera-space plane."""
    v_local, u_local = np.indices(
        (height, width),
        dtype=np.float64,
    )
    u = u_local + roi_x
    v = v_local + roi_y

    denominator = (
        normal[0] * (u - intrinsics.cx) / intrinsics.fx
        + normal[1] * (v - intrinsics.cy) / intrinsics.fy
        + normal[2]
    )
    depth_m = -d / denominator
    return (depth_m * 1000.0).astype(np.float32)


def test_compute_frame_planarity_fits_horizontal_plane(
    intrinsics: CameraIntrinsics,
) -> None:
    depth_mm = np.full((10, 12), 1000.0, dtype=np.float32)

    result = compute_frame_planarity(
        depth_mm,
        intrinsics=intrinsics,
        roi_x=0,
        roi_y=0,
    )

    assert isinstance(result, FramePlaneResult)
    assert np.allclose(result.normal, [0.0, 0.0, 1.0])
    assert result.distance_m == pytest.approx(1.0)
    assert result.tilt_deg == pytest.approx(0.0)
    assert result.rmse_mm == pytest.approx(0.0, abs=1e-10)
    assert result.residual_std_mm == pytest.approx(0.0, abs=1e-10)
    assert result.residual_p95_abs_mm == pytest.approx(0.0, abs=1e-10)
    assert result.inlier_ratio == pytest.approx(1.0)
    assert result.valid_points == 120


def test_compute_frame_planarity_fits_tilted_plane_with_roi_offset(
    intrinsics: CameraIntrinsics,
) -> None:
    expected_normal = np.array([-0.1, 0.05, 1.0])
    expected_normal /= np.linalg.norm(expected_normal)
    depth_mm = _depth_roi_for_plane(
        intrinsics=intrinsics,
        normal=expected_normal,
        d=-1.0,
        roi_x=3,
        roi_y=2,
        height=6,
        width=7,
    )

    result = compute_frame_planarity(
        depth_mm,
        intrinsics=intrinsics,
        roi_x=3,
        roi_y=2,
        min_valid_points=3,
    )

    expected_tilt = np.degrees(np.arccos(expected_normal[2]))
    assert np.allclose(result.normal, expected_normal, atol=1e-6)
    assert result.distance_m == pytest.approx(1.0, abs=1e-6)
    assert result.tilt_deg == pytest.approx(expected_tilt, abs=1e-5)
    assert result.rmse_mm < 1e-4
    assert result.inlier_ratio == pytest.approx(1.0)
    assert result.valid_points == 42


def test_compute_frame_planarity_excludes_nan_depth(
    intrinsics: CameraIntrinsics,
) -> None:
    depth_mm = np.full((4, 4), 1000.0, dtype=np.float32)
    depth_mm[0, 0] = np.nan
    depth_mm[3, 3] = np.nan

    result = compute_frame_planarity(
        depth_mm,
        intrinsics=intrinsics,
        roi_x=0,
        roi_y=0,
        min_valid_points=3,
    )

    assert result.valid_points == 14
    assert result.distance_m == pytest.approx(1.0)
    assert result.inlier_ratio == pytest.approx(1.0)


def test_compute_frame_planarity_reports_outlier_metrics(
    intrinsics: CameraIntrinsics,
) -> None:
    depth_mm = np.full((10, 12), 1000.0, dtype=np.float32)
    depth_mm[0, 0] = 1050.0

    strict = compute_frame_planarity(
        depth_mm,
        intrinsics=intrinsics,
        roi_x=0,
        roi_y=0,
        inlier_threshold_mm=1.0,
    )
    permissive = compute_frame_planarity(
        depth_mm,
        intrinsics=intrinsics,
        roi_x=0,
        roi_y=0,
        inlier_threshold_mm=100.0,
    )

    assert strict.rmse_mm > 0.0
    assert strict.residual_std_mm > 0.0
    assert strict.residual_p95_abs_mm > 0.0
    assert 0.0 < strict.inlier_ratio < 1.0
    assert permissive.inlier_ratio == pytest.approx(1.0)


def test_compute_frame_planarity_rejects_insufficient_valid_points(
    intrinsics: CameraIntrinsics,
) -> None:
    depth_mm = np.full((2, 2), np.nan, dtype=np.float32)
    depth_mm[0, 0] = 1000.0
    depth_mm[0, 1] = 1000.0

    with pytest.raises(
        ValueError,
        match=r"got 2, require at least 3",
    ):
        compute_frame_planarity(
            depth_mm,
            intrinsics=intrinsics,
            roi_x=0,
            roi_y=0,
            min_valid_points=3,
        )


def test_compute_frame_planarity_propagates_degenerate_plane_error(
    intrinsics: CameraIntrinsics,
) -> None:
    depth_mm = np.full((1, 3), 1000.0, dtype=np.float32)

    with pytest.raises(ValueError, match="non-collinear"):
        compute_frame_planarity(
            depth_mm,
            intrinsics=intrinsics,
            roi_x=0,
            roi_y=0,
            min_valid_points=3,
        )


def test_compute_frame_planarity_rejects_non_array_depth(
    intrinsics: CameraIntrinsics,
) -> None:
    with pytest.raises(TypeError, match="numpy.ndarray"):
        compute_frame_planarity(
            [[1000.0] * 3] * 3,
            intrinsics=intrinsics,
            roi_x=0,
            roi_y=0,
            min_valid_points=3,
        )


@pytest.mark.parametrize(
    "depth_mm",
    [
        np.zeros((2, 2, 2), dtype=np.float32),
        np.zeros((0, 2), dtype=np.float32),
        np.zeros((2, 0), dtype=np.float32),
    ],
)
def test_compute_frame_planarity_rejects_invalid_depth_shape(
    depth_mm: np.ndarray,
    intrinsics: CameraIntrinsics,
) -> None:
    with pytest.raises(ValueError, match="shape|dimensions"):
        compute_frame_planarity(
            depth_mm,
            intrinsics=intrinsics,
            roi_x=0,
            roi_y=0,
            min_valid_points=3,
        )


def test_compute_frame_planarity_rejects_wrong_depth_dtype(
    intrinsics: CameraIntrinsics,
) -> None:
    depth_mm = np.full((3, 3), 1000.0, dtype=np.float64)

    with pytest.raises(ValueError, match="dtype float32"):
        compute_frame_planarity(
            depth_mm,
            intrinsics=intrinsics,
            roi_x=0,
            roi_y=0,
            min_valid_points=3,
        )


@pytest.mark.parametrize("invalid_value", [np.inf, -np.inf])
def test_compute_frame_planarity_rejects_infinite_depth(
    invalid_value: float,
    intrinsics: CameraIntrinsics,
) -> None:
    depth_mm = np.full((3, 3), 1000.0, dtype=np.float32)
    depth_mm[0, 0] = invalid_value

    with pytest.raises(ValueError, match="finite values or NaN"):
        compute_frame_planarity(
            depth_mm,
            intrinsics=intrinsics,
            roi_x=0,
            roi_y=0,
            min_valid_points=3,
        )


@pytest.mark.parametrize(
    ("threshold", "expected_exception"),
    [
        (True, TypeError),
        ("5.0", TypeError),
        (0.0, ValueError),
        (-1.0, ValueError),
        (np.nan, ValueError),
        (np.inf, ValueError),
    ],
)
def test_compute_frame_planarity_rejects_invalid_inlier_threshold(
    threshold: object,
    expected_exception: type[Exception],
    intrinsics: CameraIntrinsics,
) -> None:
    depth_mm = np.full((3, 3), 1000.0, dtype=np.float32)

    with pytest.raises(
        expected_exception,
        match="inlier_threshold_mm",
    ):
        compute_frame_planarity(
            depth_mm,
            intrinsics=intrinsics,
            roi_x=0,
            roi_y=0,
            inlier_threshold_mm=threshold,
            min_valid_points=3,
        )


@pytest.mark.parametrize(
    ("minimum", "expected_exception"),
    [
        (True, TypeError),
        (3.0, TypeError),
        ("3", TypeError),
        (2, ValueError),
        (0, ValueError),
        (-1, ValueError),
    ],
)
def test_compute_frame_planarity_rejects_invalid_minimum_points(
    minimum: object,
    expected_exception: type[Exception],
    intrinsics: CameraIntrinsics,
) -> None:
    depth_mm = np.full((3, 3), 1000.0, dtype=np.float32)

    with pytest.raises(
        expected_exception,
        match="min_valid_points",
    ):
        compute_frame_planarity(
            depth_mm,
            intrinsics=intrinsics,
            roi_x=0,
            roi_y=0,
            min_valid_points=minimum,
        )


def test_planarity_defaults_match_baseline_configuration() -> None:
    assert DEFAULT_INLIER_THRESHOLD_MM == 5.0
    assert DEFAULT_MIN_VALID_POINTS == 100


def test_compute_planarity_aggregates_multiple_distances(
    intrinsics: CameraIntrinsics,
) -> None:
    depth = np.stack(
        [
            np.full((10, 12), 999.0, dtype=np.float32),
            np.full((10, 12), 1000.0, dtype=np.float32),
            np.full((10, 12), 1001.0, dtype=np.float32),
        ]
    )

    result = compute_planarity(
        depth,
        intrinsics=intrinsics,
        roi_x=0,
        roi_y=0,
    )

    assert isinstance(result, PlanarityResult)
    assert result.frame_normal.shape == (3, 3)
    assert result.frame_distance_m.shape == (3,)
    assert result.frame_tilt_deg.shape == (3,)
    assert result.frame_rmse_mm.shape == (3,)
    assert result.frame_residual_std_mm.shape == (3,)
    assert result.frame_p95_abs_mm.shape == (3,)
    assert result.frame_inlier_ratio.shape == (3,)
    assert result.frame_valid_points.dtype == np.int64
    assert result.frame_fit_succeeded.dtype == np.bool_
    assert np.allclose(result.frame_normal, [0.0, 0.0, 1.0])
    assert np.allclose(
        result.frame_distance_m,
        [0.999, 1.000, 1.001],
    )
    assert np.allclose(result.frame_tilt_deg, 0.0)
    assert np.allclose(result.frame_rmse_mm, 0.0, atol=1e-10)
    assert np.allclose(result.frame_residual_std_mm, 0.0, atol=1e-10)
    assert np.allclose(result.frame_p95_abs_mm, 0.0, atol=1e-10)
    assert np.allclose(result.frame_inlier_ratio, 1.0)
    assert np.array_equal(result.frame_valid_points, [120, 120, 120])
    assert np.array_equal(result.frame_fit_succeeded, [True, True, True])
    assert result.median_distance_m == pytest.approx(1.0)
    assert result.distance_std_mm == pytest.approx(
        np.std([0.999, 1.000, 1.001]) * 1000.0
    )
    assert result.median_tilt_deg == pytest.approx(0.0)
    assert result.tilt_std_deg == pytest.approx(0.0)
    assert result.median_rmse_mm == pytest.approx(0.0, abs=1e-10)
    assert result.p95_rmse_mm == pytest.approx(0.0, abs=1e-10)
    assert result.median_p95_abs_mm == pytest.approx(0.0, abs=1e-10)
    assert result.median_inlier_ratio == pytest.approx(1.0)
    assert result.successful_frames == 3
    assert result.failed_frames == 0


def test_compute_planarity_preserves_alignment_for_insufficient_frame(
    intrinsics: CameraIntrinsics,
) -> None:
    depth = np.stack(
        [
            np.full((4, 4), 999.0, dtype=np.float32),
            np.full((4, 4), np.nan, dtype=np.float32),
            np.full((4, 4), 1001.0, dtype=np.float32),
        ]
    )
    depth[1, 0, 0] = 1000.0
    depth[1, 0, 1] = 1000.0

    result = compute_planarity(
        depth,
        intrinsics=intrinsics,
        roi_x=0,
        roi_y=0,
        min_valid_points=3,
    )

    assert np.array_equal(result.frame_valid_points, [16, 2, 16])
    assert np.array_equal(result.frame_fit_succeeded, [True, False, True])
    assert np.all(np.isnan(result.frame_normal[1]))
    assert np.isnan(result.frame_distance_m[1])
    assert np.isnan(result.frame_tilt_deg[1])
    assert np.isnan(result.frame_rmse_mm[1])
    assert np.isnan(result.frame_residual_std_mm[1])
    assert np.isnan(result.frame_p95_abs_mm[1])
    assert np.isnan(result.frame_inlier_ratio[1])
    assert result.median_distance_m == pytest.approx(1.0)
    assert result.distance_std_mm == pytest.approx(1.0)
    assert result.successful_frames == 2
    assert result.failed_frames == 1


def test_compute_planarity_aggregates_tilted_planes_with_roi_offset(
    intrinsics: CameraIntrinsics,
) -> None:
    expected_normal = np.array([-0.1, 0.05, 1.0])
    expected_normal /= np.linalg.norm(expected_normal)
    depth = np.stack(
        [
            _depth_roi_for_plane(
                intrinsics=intrinsics,
                normal=expected_normal,
                d=-distance,
                roi_x=3,
                roi_y=2,
                height=6,
                width=7,
            )
            for distance in (0.9, 1.0, 1.1)
        ]
    )

    result = compute_planarity(
        depth,
        intrinsics=intrinsics,
        roi_x=3,
        roi_y=2,
        min_valid_points=3,
    )

    expected_tilt = np.degrees(np.arccos(expected_normal[2]))
    assert np.allclose(result.frame_normal, expected_normal, atol=1e-6)
    assert np.allclose(result.frame_distance_m, [0.9, 1.0, 1.1], atol=1e-6)
    assert np.allclose(result.frame_tilt_deg, expected_tilt, atol=1e-5)
    assert result.median_tilt_deg == pytest.approx(expected_tilt, abs=1e-5)
    assert result.tilt_std_deg == pytest.approx(0.0, abs=1e-5)
    assert result.median_distance_m == pytest.approx(1.0, abs=1e-6)


def test_compute_planarity_returns_undefined_empty_sequence(
    intrinsics: CameraIntrinsics,
) -> None:
    depth = np.empty((0, 4, 4), dtype=np.float32)

    result = compute_planarity(
        depth,
        intrinsics=intrinsics,
        roi_x=1,
        roi_y=1,
        min_valid_points=3,
    )

    assert result.frame_normal.shape == (0, 3)
    assert result.frame_distance_m.shape == (0,)
    assert result.frame_valid_points.shape == (0,)
    assert result.frame_fit_succeeded.shape == (0,)
    assert np.isnan(result.median_distance_m)
    assert np.isnan(result.distance_std_mm)
    assert np.isnan(result.median_tilt_deg)
    assert np.isnan(result.tilt_std_deg)
    assert np.isnan(result.median_rmse_mm)
    assert np.isnan(result.p95_rmse_mm)
    assert np.isnan(result.median_p95_abs_mm)
    assert np.isnan(result.median_inlier_ratio)
    assert result.successful_frames == 0
    assert result.failed_frames == 0


def test_compute_planarity_returns_undefined_when_all_frames_fail(
    intrinsics: CameraIntrinsics,
) -> None:
    depth = np.full((2, 4, 4), np.nan, dtype=np.float32)

    result = compute_planarity(
        depth,
        intrinsics=intrinsics,
        roi_x=0,
        roi_y=0,
        min_valid_points=3,
    )

    assert np.all(np.isnan(result.frame_normal))
    assert np.all(np.isnan(result.frame_distance_m))
    assert np.array_equal(result.frame_valid_points, [0, 0])
    assert np.array_equal(result.frame_fit_succeeded, [False, False])
    assert np.isnan(result.median_distance_m)
    assert np.isnan(result.median_rmse_mm)
    assert result.successful_frames == 0
    assert result.failed_frames == 2


def test_compute_planarity_propagates_degenerate_plane_error(
    intrinsics: CameraIntrinsics,
) -> None:
    depth = np.full((1, 1, 3), 1000.0, dtype=np.float32)

    with pytest.raises(ValueError, match="non-collinear"):
        compute_planarity(
            depth,
            intrinsics=intrinsics,
            roi_x=0,
            roi_y=0,
            min_valid_points=3,
        )


def test_compute_planarity_rejects_non_array_depth(
    intrinsics: CameraIntrinsics,
) -> None:
    with pytest.raises(TypeError, match="numpy.ndarray"):
        compute_planarity(
            [[[1000.0]]],
            intrinsics=intrinsics,
            roi_x=0,
            roi_y=0,
            min_valid_points=3,
        )


@pytest.mark.parametrize(
    "depth",
    [
        np.zeros((3, 3), dtype=np.float32),
        np.zeros((1, 0, 3), dtype=np.float32),
        np.zeros((1, 3, 0), dtype=np.float32),
    ],
)
def test_compute_planarity_rejects_invalid_depth_shape(
    depth: np.ndarray,
    intrinsics: CameraIntrinsics,
) -> None:
    with pytest.raises(ValueError, match="shape|dimensions"):
        compute_planarity(
            depth,
            intrinsics=intrinsics,
            roi_x=0,
            roi_y=0,
            min_valid_points=3,
        )


def test_compute_planarity_rejects_wrong_depth_dtype(
    intrinsics: CameraIntrinsics,
) -> None:
    depth = np.full((1, 3, 3), 1000.0, dtype=np.float64)

    with pytest.raises(ValueError, match="dtype float32"):
        compute_planarity(
            depth,
            intrinsics=intrinsics,
            roi_x=0,
            roi_y=0,
            min_valid_points=3,
        )


def test_compute_planarity_rejects_infinite_depth(
    intrinsics: CameraIntrinsics,
) -> None:
    depth = np.full((1, 3, 3), 1000.0, dtype=np.float32)
    depth[0, 0, 0] = np.inf

    with pytest.raises(ValueError, match="finite values or NaN"):
        compute_planarity(
            depth,
            intrinsics=intrinsics,
            roi_x=0,
            roi_y=0,
            min_valid_points=3,
        )


@pytest.mark.parametrize(
    ("roi_x", "roi_y", "expected_message"),
    [
        (-1, 0, "roi_x must be non-negative"),
        (0, -1, "roi_y must be non-negative"),
        (1.5, 0, "roi_x must be an integer"),
        (0, True, "roi_y must be an integer"),
        (9, 0, "image width"),
        (0, 7, "image height"),
    ],
)
def test_compute_planarity_validates_roi_for_empty_sequence(
    roi_x: object,
    roi_y: object,
    expected_message: str,
    intrinsics: CameraIntrinsics,
) -> None:
    depth = np.empty((0, 4, 4), dtype=np.float32)

    with pytest.raises(ValueError, match=expected_message):
        compute_planarity(
            depth,
            intrinsics=intrinsics,
            roi_x=roi_x,
            roi_y=roi_y,
            min_valid_points=3,
        )


def test_compute_planarity_validates_intrinsics_for_empty_sequence() -> None:
    depth = np.empty((0, 4, 4), dtype=np.float32)

    with pytest.raises(TypeError, match="CameraIntrinsics"):
        compute_planarity(
            depth,
            intrinsics=object(),
            roi_x=0,
            roi_y=0,
            min_valid_points=3,
        )


@pytest.mark.parametrize(
    ("keyword", "value", "expected_exception"),
    [
        ("inlier_threshold_mm", 0.0, ValueError),
        ("inlier_threshold_mm", True, TypeError),
        ("min_valid_points", 2, ValueError),
        ("min_valid_points", 3.0, TypeError),
    ],
)
def test_compute_planarity_validates_parameters_for_empty_sequence(
    keyword: str,
    value: object,
    expected_exception: type[Exception],
    intrinsics: CameraIntrinsics,
) -> None:
    depth = np.empty((0, 4, 4), dtype=np.float32)
    arguments = {
        "intrinsics": intrinsics,
        "roi_x": 0,
        "roi_y": 0,
        "min_valid_points": 3,
    }
    arguments[keyword] = value

    with pytest.raises(expected_exception):
        compute_planarity(depth, **arguments)
