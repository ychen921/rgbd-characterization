"""Tests for the interactive ROI selection tool."""

from pathlib import Path

import cv2
import numpy as np
import pytest
import yaml

from src.io.dataset import DepthDataset
from tools import select_roi as select_roi_tool


def save_dataset(dataset_dir: Path) -> None:
    """Save a small valid dataset for ROI selection tests."""
    dataset_dir.mkdir(parents=True)
    depth = np.arange(3 * 4 * 5, dtype=np.uint16).reshape(3, 4, 5) + 1
    timestamps_ns = np.array([100, 200, 300], dtype=np.int64)
    DepthDataset(depth=depth, timestamps_ns=timestamps_ns).save(
        dataset_dir / "depth.npz"
    )


def test_depth_to_display_returns_bgr_uint8_image() -> None:
    depth = np.array([[0, 100, 200], [300, 400, 65535]], dtype=np.uint16)

    display = select_roi_tool.depth_to_display(depth)

    assert display.shape == (2, 3, 3)
    assert display.dtype == np.uint8
    assert np.array_equal(display[:, :, 0], display[:, :, 1])
    assert np.array_equal(display[:, :, 1], display[:, :, 2])


@pytest.mark.parametrize(
    "depth",
    [
        np.array([[0, 65535]], dtype=np.uint16),
        np.full((2, 2), 500, dtype=np.uint16),
    ],
)
def test_depth_to_display_rejects_unusable_frame(depth: np.ndarray) -> None:
    with pytest.raises(ValueError):
        select_roi_tool.depth_to_display(depth)


def test_existing_roi_skips_dataset_loading(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    dataset_dir = tmp_path / "data" / "scene01_white_d050_r02"
    roi_root = tmp_path / "config" / "roi"
    roi_root.mkdir(parents=True)
    expected_path = roi_root / "scene01_white_d050.yaml"
    expected_path.write_text("existing", encoding="utf-8")

    def fail_load(path: Path) -> DepthDataset:
        raise AssertionError(f"Dataset should not be loaded: {path}")

    monkeypatch.setattr(select_roi_tool.DepthDataset, "load", fail_load)

    assert select_roi_tool.select_roi(dataset_dir, roi_root) == expected_path


def test_successful_selection_saves_shared_roi_yaml(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    dataset_dir = tmp_path / "data" / "scene01_white_d050_r01"
    roi_root = tmp_path / "config" / "roi"
    save_dataset(dataset_dir)
    monkeypatch.setattr(cv2, "selectROI", lambda *args, **kwargs: (1, 1, 3, 2))
    monkeypatch.setattr(cv2, "destroyAllWindows", lambda: None)

    output_path = select_roi_tool.select_roi(dataset_dir, roi_root)

    assert output_path == roi_root / "scene01_white_d050.yaml"
    with output_path.open("r", encoding="utf-8") as stream:
        document = yaml.safe_load(stream)
    assert document == {
        "name": "scene01_white_d050",
        "source": {
            "experiment": "scene01_white_d050_r01",
            "frame_index": 1,
        },
        "roi": {
            "type": "rectangle",
            "x": 1,
            "y": 1,
            "width": 3,
            "height": 2,
        },
    }


def test_cancelled_selection_does_not_save_yaml(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    dataset_dir = tmp_path / "data" / "scene01_white_d050_r01"
    roi_root = tmp_path / "config" / "roi"
    save_dataset(dataset_dir)
    monkeypatch.setattr(cv2, "selectROI", lambda *args, **kwargs: (0, 0, 0, 0))
    monkeypatch.setattr(cv2, "destroyAllWindows", lambda: None)

    with pytest.raises(ValueError, match="cancelled"):
        select_roi_tool.select_roi(dataset_dir, roi_root)

    assert not (roi_root / "scene01_white_d050.yaml").exists()


def test_out_of_bounds_selection_does_not_save_yaml(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    dataset_dir = tmp_path / "data" / "scene01_white_d050_r01"
    roi_root = tmp_path / "config" / "roi"
    save_dataset(dataset_dir)
    monkeypatch.setattr(cv2, "selectROI", lambda *args, **kwargs: (4, 0, 2, 1))
    monkeypatch.setattr(cv2, "destroyAllWindows", lambda: None)

    with pytest.raises(ValueError, match="image width"):
        select_roi_tool.select_roi(dataset_dir, roi_root)

    assert not (roi_root / "scene01_white_d050.yaml").exists()
