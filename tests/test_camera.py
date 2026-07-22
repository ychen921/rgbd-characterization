"""Tests for depth-camera calibration loading and validation."""

from pathlib import Path

import numpy as np
import pytest
import yaml

from src.geometry.camera import (
    CameraIntrinsics,
    depth_roi_to_points,
    load_camera_intrinsics,
    validate_depth_resolution,
)


def _camera_info_document() -> dict[str, object]:
    return {
        "header": {"frame_id": "camera_depth_optical_frame"},
        "height": 480,
        "width": 848,
        "k": [
            412.45037841796875,
            0.0,
            422.34375,
            0.0,
            412.45037841796875,
            241.32501220703125,
            0.0,
            0.0,
            1.0,
        ],
    }


def _write_camera_info(path: Path, document: object) -> None:
    with path.open("w", encoding="utf-8") as stream:
        yaml.safe_dump(document, stream)


@pytest.fixture
def simple_intrinsics() -> CameraIntrinsics:
    return CameraIntrinsics(
        width=5,
        height=4,
        fx=100.0,
        fy=100.0,
        cx=2.0,
        cy=1.0,
        frame_id="camera_depth_optical_frame",
    )


def test_load_project_depth_camera_info() -> None:
    intrinsics = load_camera_intrinsics(
        Path("config/calib/depth_camera_info.yaml")
    )

    assert intrinsics == CameraIntrinsics(
        width=848,
        height=480,
        fx=412.45037841796875,
        fy=412.45037841796875,
        cx=422.34375,
        cy=241.32501220703125,
        frame_id="camera_depth_optical_frame",
    )


def test_load_allows_trailing_empty_yaml_document(tmp_path: Path) -> None:
    path = tmp_path / "camera_info.yaml"
    _write_camera_info(path, _camera_info_document())
    with path.open("a", encoding="utf-8") as stream:
        stream.write("---\n")

    assert load_camera_intrinsics(path).width == 848


@pytest.mark.parametrize(
    ("field", "value", "expected_message"),
    [
        ("width", None, "width"),
        ("height", None, "height"),
        ("k", [1.0] * 8, "9 values"),
    ],
)
def test_load_rejects_missing_or_invalid_fields(
    tmp_path: Path,
    field: str,
    value: object,
    expected_message: str,
) -> None:
    document = _camera_info_document()
    if value is None:
        document.pop(field)
    else:
        document[field] = value
    path = tmp_path / "invalid.yaml"
    _write_camera_info(path, document)

    with pytest.raises(ValueError, match=expected_message):
        load_camera_intrinsics(path)


@pytest.mark.parametrize(("index", "value"), [(0, 0.0), (4, -1.0)])
def test_load_rejects_non_positive_focal_length(
    tmp_path: Path,
    index: int,
    value: float,
) -> None:
    document = _camera_info_document()
    matrix = document["k"]
    assert isinstance(matrix, list)
    matrix[index] = value
    path = tmp_path / "invalid_focal_length.yaml"
    _write_camera_info(path, document)

    with pytest.raises(ValueError, match="fx and fy must be positive"):
        load_camera_intrinsics(path)


def test_validate_depth_resolution_accepts_image_and_sequence() -> None:
    intrinsics = load_camera_intrinsics(
        Path("config/calib/depth_camera_info.yaml")
    )

    validate_depth_resolution(np.zeros((480, 848), dtype=np.uint16), intrinsics)
    validate_depth_resolution(
        np.zeros((2, 480, 848), dtype=np.uint16),
        intrinsics,
    )


def test_validate_depth_resolution_rejects_mismatch() -> None:
    intrinsics = load_camera_intrinsics(
        Path("config/calib/depth_camera_info.yaml")
    )
    depth = np.zeros((480, 640), dtype=np.uint16)

    with pytest.raises(ValueError, match="640x480.*848x480"):
        validate_depth_resolution(depth, intrinsics)


def test_project_dataset_resolution_matches_calibration() -> None:
    intrinsics = load_camera_intrinsics(
        Path("config/calib/depth_camera_info.yaml")
    )
    with np.load(
        "data/scene01_white_d050_r01/depth.npz",
        allow_pickle=False,
    ) as archive:
        validate_depth_resolution(archive["depth"], intrinsics)


def test_depth_roi_to_points_back_projects_principal_point(
    simple_intrinsics: CameraIntrinsics,
) -> None:
    depth_mm = np.array([[1000.0]], dtype=np.float32)

    points = depth_roi_to_points(
        depth_mm,
        simple_intrinsics,
        roi_x=2,
        roi_y=1,
    )

    assert points.shape == (1, 3)
    assert points.dtype == np.float64
    assert np.allclose(points, [[0.0, 0.0, 1.0]])


def test_depth_roi_to_points_computes_expected_coordinates(
    simple_intrinsics: CameraIntrinsics,
) -> None:
    depth_mm = np.array([[2000.0]], dtype=np.float32)

    points = depth_roi_to_points(
        depth_mm,
        simple_intrinsics,
        roi_x=3,
        roi_y=2,
    )

    assert np.allclose(points, [[0.02, 0.02, 2.0]])


def test_depth_roi_to_points_applies_roi_offset(
    simple_intrinsics: CameraIntrinsics,
) -> None:
    depth_mm = np.array([[1000.0]], dtype=np.float32)

    without_offset = depth_roi_to_points(
        depth_mm,
        simple_intrinsics,
        roi_x=0,
        roi_y=0,
    )
    with_offset = depth_roi_to_points(
        depth_mm,
        simple_intrinsics,
        roi_x=2,
        roi_y=1,
    )

    assert np.allclose(without_offset, [[-0.02, -0.01, 1.0]])
    assert np.allclose(with_offset, [[0.0, 0.0, 1.0]])


def test_depth_roi_to_points_excludes_invalid_depth_and_preserves_order(
    simple_intrinsics: CameraIntrinsics,
) -> None:
    depth_mm = np.array(
        [
            [1000.0, np.nan, 2000.0],
            [np.inf, 0.0, -1.0],
        ],
        dtype=np.float32,
    )

    points = depth_roi_to_points(
        depth_mm,
        simple_intrinsics,
        roi_x=0,
        roi_y=0,
    )

    assert points.shape == (2, 3)
    assert np.all(np.isfinite(points))
    assert np.allclose(
        points,
        [
            [-0.02, -0.01, 1.0],
            [0.0, -0.02, 2.0],
        ],
    )


def test_depth_roi_to_points_returns_empty_array_for_no_valid_depth(
    simple_intrinsics: CameraIntrinsics,
) -> None:
    depth_mm = np.full((2, 2), np.nan, dtype=np.float32)

    points = depth_roi_to_points(
        depth_mm,
        simple_intrinsics,
        roi_x=0,
        roi_y=0,
    )

    assert points.shape == (0, 3)
    assert points.dtype == np.float64


def test_depth_roi_to_points_rejects_non_array_input(
    simple_intrinsics: CameraIntrinsics,
) -> None:
    with pytest.raises(TypeError, match="numpy.ndarray"):
        depth_roi_to_points(
            [[1000.0]],
            simple_intrinsics,
            roi_x=0,
            roi_y=0,
        )


def test_depth_roi_to_points_rejects_frame_sequence(
    simple_intrinsics: CameraIntrinsics,
) -> None:
    depth_mm = np.zeros((2, 4, 5), dtype=np.float32)

    with pytest.raises(ValueError, match=r"shape \(H, W\)"):
        depth_roi_to_points(
            depth_mm,
            simple_intrinsics,
            roi_x=0,
            roi_y=0,
        )


@pytest.mark.parametrize(
    ("roi_x", "roi_y", "expected_message"),
    [
        (-1, 0, "roi_x must be non-negative"),
        (0, -1, "roi_y must be non-negative"),
        (1.5, 0, "roi_x must be an integer"),
        (0, True, "roi_y must be an integer"),
    ],
)
def test_depth_roi_to_points_rejects_invalid_roi_origin(
    simple_intrinsics: CameraIntrinsics,
    roi_x: object,
    roi_y: object,
    expected_message: str,
) -> None:
    depth_mm = np.ones((1, 1), dtype=np.float32)

    with pytest.raises(ValueError, match=expected_message):
        depth_roi_to_points(
            depth_mm,
            simple_intrinsics,
            roi_x=roi_x,
            roi_y=roi_y,
        )


@pytest.mark.parametrize(
    ("shape", "roi_x", "roi_y", "expected_message"),
    [
        ((1, 2), 4, 0, "image width"),
        ((2, 1), 0, 3, "image height"),
    ],
)
def test_depth_roi_to_points_rejects_out_of_bounds_roi(
    simple_intrinsics: CameraIntrinsics,
    shape: tuple[int, int],
    roi_x: int,
    roi_y: int,
    expected_message: str,
) -> None:
    depth_mm = np.ones(shape, dtype=np.float32)

    with pytest.raises(ValueError, match=expected_message):
        depth_roi_to_points(
            depth_mm,
            simple_intrinsics,
            roi_x=roi_x,
            roi_y=roi_y,
        )


def test_depth_roi_to_points_accepts_roi_touching_image_boundaries(
    simple_intrinsics: CameraIntrinsics,
) -> None:
    depth_mm = np.full((2, 2), 1000.0, dtype=np.float32)

    points = depth_roi_to_points(
        depth_mm,
        simple_intrinsics,
        roi_x=3,
        roi_y=2,
    )

    assert points.shape == (4, 3)
