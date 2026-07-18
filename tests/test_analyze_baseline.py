"""Tests for baseline analysis orchestration."""

from pathlib import Path

import numpy as np
import pytest

from src.io.dataset import DepthDataset
from src.preprocessing.roi import RectROI, save_roi
from tools.analyze_baseline import (
    compute_baseline_metrics,
    load_baseline_input,
)


EXPERIMENT_NAME = "scene01_white_d050_r01"
ROI_KEY = "scene01_white_d050"


def _write_dataset(dataset_dir: Path, depth: np.ndarray) -> Path:
    dataset_dir.mkdir(parents=True, exist_ok=True)
    dataset_path = dataset_dir / "depth.npz"
    timestamps_ns = np.arange(depth.shape[0], dtype=np.int64)
    DepthDataset(
        depth=depth,
        timestamps_ns=timestamps_ns,
    ).save(dataset_path)
    return dataset_path


def _write_roi(roi_root: Path, roi: RectROI) -> Path:
    roi_path = roi_root / f"{ROI_KEY}.yaml"
    save_roi(
        roi_path,
        roi,
        name=ROI_KEY,
        source_experiment=EXPERIMENT_NAME,
        source_frame_index=0,
    )
    return roi_path


def test_compute_baseline_metrics_combines_all_metrics() -> None:
    raw_roi = np.array(
        [
            [[0, 10, 65535]],
            [[20, 14, 30]],
        ],
        dtype=np.uint16,
    )

    result = compute_baseline_metrics(raw_roi)

    assert result.depth_quality.zero_ratio == pytest.approx(1 / 6)
    assert result.depth_quality.max_uint16_ratio == pytest.approx(1 / 6)

    np.testing.assert_allclose(
        result.measured_depth.frame_median,
        [10.0, 20.0],
    )
    assert result.measured_depth.median_depth == pytest.approx(15.0)

    np.testing.assert_allclose(
        result.temporal_noise.std_map,
        [[np.nan, 2.0, np.nan]],
        equal_nan=True,
    )
    assert result.temporal_noise.median_std == pytest.approx(2.0)


def test_compute_baseline_metrics_forwards_min_valid_ratio() -> None:
    raw_roi = np.array(
        [
            [[0, 10, 65535]],
            [[20, 14, 30]],
        ],
        dtype=np.uint16,
    )

    result = compute_baseline_metrics(
        raw_roi,
        min_valid_ratio=0.5,
    )

    np.testing.assert_allclose(
        result.temporal_noise.std_map,
        [[0.0, 2.0, 0.0]],
    )


def test_load_baseline_input_loads_dataset_and_crops_roi(
    tmp_path: Path,
) -> None:
    dataset_dir = tmp_path / "data" / EXPERIMENT_NAME
    roi_root = tmp_path / "config" / "roi"
    depth = np.arange(24, dtype=np.uint16).reshape(2, 3, 4)
    roi = RectROI(x=1, y=1, width=2, height=2)
    dataset_path = _write_dataset(dataset_dir, depth)
    roi_path = _write_roi(roi_root, roi)

    result = load_baseline_input(dataset_dir, roi_root)

    assert result.experiment_name == EXPERIMENT_NAME
    assert result.dataset_path == dataset_path
    assert result.roi_key == ROI_KEY
    assert result.roi_path == roi_path
    assert result.roi == roi
    np.testing.assert_array_equal(
        result.raw_roi,
        depth[:, 1:3, 1:3],
    )
    assert result.raw_roi.dtype == np.uint16
    assert result.raw_roi.shape == (2, 2, 2)
    assert np.shares_memory(result.raw_roi, result.dataset.depth)


def test_load_baseline_input_rejects_missing_dataset(
    tmp_path: Path,
) -> None:
    dataset_dir = tmp_path / "data" / EXPERIMENT_NAME
    roi_root = tmp_path / "config" / "roi"

    with pytest.raises(FileNotFoundError, match="Cannot find dataset file"):
        load_baseline_input(dataset_dir, roi_root)


def test_load_baseline_input_reports_missing_roi(
    tmp_path: Path,
) -> None:
    dataset_dir = tmp_path / "data" / EXPERIMENT_NAME
    roi_root = tmp_path / "config" / "roi"
    depth = np.ones((1, 2, 2), dtype=np.uint16)
    _write_dataset(dataset_dir, depth)

    with pytest.raises(
        FileNotFoundError,
        match="ROI configuration not found",
    ) as error_info:
        load_baseline_input(dataset_dir, roi_root)

    message = str(error_info.value)
    assert f"{ROI_KEY}.yaml" in message
    assert f"python3 tools/select_roi.py {dataset_dir}" in message


def test_load_baseline_input_rejects_empty_dataset(
    tmp_path: Path,
) -> None:
    dataset_dir = tmp_path / "data" / EXPERIMENT_NAME
    roi_root = tmp_path / "config" / "roi"
    depth = np.empty((0, 3, 4), dtype=np.uint16)
    _write_dataset(dataset_dir, depth)
    _write_roi(
        roi_root,
        RectROI(x=1, y=1, width=2, height=2),
    )

    with pytest.raises(ValueError, match="contains no depth frames"):
        load_baseline_input(dataset_dir, roi_root)
