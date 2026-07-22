"""Tests for depth-camera calibration loading and validation."""

from pathlib import Path

import numpy as np
import pytest
import yaml

from src.geometry.camera import (
    CameraIntrinsics,
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
