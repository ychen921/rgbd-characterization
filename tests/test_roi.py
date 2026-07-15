"""Tests for rectangular ROI handling and configuration persistence."""

from pathlib import Path

import numpy as np
import pytest
import yaml

from src.preprocessing.roi import (
    RectROI,
    derive_roi_key,
    get_roi_path,
    load_roi,
    save_roi,
)


@pytest.fixture
def frames() -> np.ndarray:
    return np.arange(2 * 4 * 5, dtype=np.uint16).reshape(2, 4, 5)


def test_pixel_count() -> None:
    roi = RectROI(x=1, y=2, width=3, height=4)

    assert roi.pixel_count == 12


def test_crop_preserves_frames_and_selects_expected_region(
    frames: np.ndarray,
) -> None:
    roi = RectROI(x=1, y=1, width=3, height=2)

    cropped = roi.crop(frames)

    assert cropped.shape == (2, 2, 3)
    assert np.array_equal(cropped, frames[:, 1:3, 1:4])


def test_crop_accepts_roi_touching_image_boundaries(
    frames: np.ndarray,
) -> None:
    roi = RectROI(x=2, y=2, width=3, height=2)

    assert np.array_equal(roi.crop(frames), frames[:, 2:4, 2:5])


@pytest.mark.parametrize(
    ("arguments", "expected_message"),
    [
        ({"x": -1, "y": 0, "width": 1, "height": 1}, "non-negative"),
        ({"x": 0, "y": -1, "width": 1, "height": 1}, "non-negative"),
        ({"x": 0, "y": 0, "width": 0, "height": 1}, "positive"),
        ({"x": 0, "y": 0, "width": 1, "height": 0}, "positive"),
        ({"x": 0.5, "y": 0, "width": 1, "height": 1}, "integer"),
    ],
)
def test_rejects_invalid_roi_geometry(
    arguments: dict[str, object],
    expected_message: str,
) -> None:
    with pytest.raises(ValueError, match=expected_message):
        RectROI(**arguments)


def test_crop_rejects_non_array_input() -> None:
    roi = RectROI(x=0, y=0, width=1, height=1)

    with pytest.raises(TypeError, match="numpy.ndarray"):
        roi.crop([[[1]]])


def test_crop_rejects_non_frame_array() -> None:
    roi = RectROI(x=0, y=0, width=1, height=1)
    image = np.zeros((4, 5), dtype=np.uint16)

    with pytest.raises(ValueError, match=r"shape \(N, H, W\)"):
        roi.crop(image)


@pytest.mark.parametrize(
    ("roi", "expected_message"),
    [
        (RectROI(x=4, y=0, width=2, height=1), "image width"),
        (RectROI(x=0, y=3, width=1, height=2), "image height"),
    ],
)
def test_crop_rejects_out_of_bounds_roi(
    roi: RectROI,
    expected_message: str,
    frames: np.ndarray,
) -> None:
    with pytest.raises(ValueError, match=expected_message):
        roi.crop(frames)


@pytest.mark.parametrize(
    ("experiment_name", "expected_key"),
    [
        ("scene01_white_d050_r01", "scene01_white_d050"),
        ("scene01_white_d050_r12", "scene01_white_d050"),
        ("scene01_white_d050", "scene01_white_d050"),
        ("scene01_white_d050_r01_extra", "scene01_white_d050_r01_extra"),
    ],
)
def test_derive_roi_key(experiment_name: str, expected_key: str) -> None:
    assert derive_roi_key(experiment_name) == expected_key


def test_get_roi_path() -> None:
    path = get_roi_path(Path("config/roi"), "scene01_white_d050_r02")

    assert path == Path("config/roi/scene01_white_d050.yaml")


def test_save_load_round_trip(tmp_path: Path) -> None:
    output_path = tmp_path / "config" / "roi" / "scene01_white_d050.yaml"
    roi = RectROI(x=280, y=210, width=80, height=60)

    save_roi(
        output_path,
        roi,
        name="scene01_white_d050",
        source_experiment="scene01_white_d050_r01",
        source_frame_index=421,
    )

    assert load_roi(output_path) == roi

    with output_path.open("r", encoding="utf-8") as stream:
        document = yaml.safe_load(stream)

    assert document["name"] == "scene01_white_d050"
    assert document["source"] == {
        "experiment": "scene01_white_d050_r01",
        "frame_index": 421,
    }
    assert document["roi"] == {
        "type": "rectangle",
        "x": 280,
        "y": 210,
        "width": 80,
        "height": 60,
    }


@pytest.mark.parametrize(
    ("document", "expected_message"),
    [
        ({}, "name"),
        ({"name": "test"}, "source"),
        (
            {
                "name": "test",
                "source": {"experiment": "test_r01", "frame_index": 0},
                "roi": {
                    "type": "ellipse",
                    "x": 0,
                    "y": 0,
                    "width": 1,
                    "height": 1,
                },
            },
            "roi type",
        ),
        (
            {
                "name": "test",
                "source": {"experiment": "test_r01", "frame_index": 0},
                "roi": {"type": "rectangle", "y": 0, "width": 1, "height": 1},
            },
            "ROI x",
        ),
    ],
)
def test_load_rejects_invalid_configuration(
    tmp_path: Path,
    document: object,
    expected_message: str,
) -> None:
    input_path = tmp_path / "invalid.yaml"
    with input_path.open("w", encoding="utf-8") as stream:
        yaml.safe_dump(document, stream)

    with pytest.raises(ValueError, match=expected_message):
        load_roi(input_path)
