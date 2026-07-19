"""Tests for baseline analysis orchestration."""

import csv
from pathlib import Path

import numpy as np
import pytest
import yaml

from src.io.dataset import DepthDataset
from src.preprocessing.roi import RectROI, save_roi
from tools.analyze_baseline import (
    analyze_baseline,
    build_summary,
    compute_baseline_metrics,
    load_baseline_input,
    save_baseline_analysis,
    save_frame_median_csv,
    save_metric_maps,
    save_summary,
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


def test_analyze_baseline_loads_input_and_computes_metrics(
    tmp_path: Path,
) -> None:
    dataset_dir = tmp_path / "data" / EXPERIMENT_NAME
    roi_root = tmp_path / "config" / "roi"
    depth = np.array(
        [
            [[0, 10, 65535]],
            [[20, 14, 30]],
        ],
        dtype=np.uint16,
    )
    _write_dataset(dataset_dir, depth)
    _write_roi(
        roi_root,
        RectROI(x=0, y=0, width=3, height=1),
    )

    result = analyze_baseline(dataset_dir, roi_root)

    assert result.source.experiment_name == EXPERIMENT_NAME
    assert result.source.raw_roi.dtype == np.uint16
    assert result.metrics.depth_quality.zero_ratio == pytest.approx(1 / 6)
    assert result.metrics.measured_depth.median_depth == pytest.approx(15.0)
    np.testing.assert_allclose(
        result.metrics.temporal_noise.std_map,
        [[np.nan, 2.0, np.nan]],
        equal_nan=True,
    )


def test_build_summary_contains_dataset_roi_and_metrics(
    tmp_path: Path,
) -> None:
    dataset_dir = tmp_path / "data" / EXPERIMENT_NAME
    roi_root = tmp_path / "config" / "roi"
    depth = np.array(
        [
            [[0, 10, 65535]],
            [[20, 14, 30]],
        ],
        dtype=np.uint16,
    )
    dataset_path = _write_dataset(dataset_dir, depth)
    roi_path = _write_roi(
        roi_root,
        RectROI(x=0, y=0, width=3, height=1),
    )
    result = analyze_baseline(dataset_dir, roi_root)

    summary = build_summary(result)

    assert summary["dataset"] == {
        "experiment": EXPERIMENT_NAME,
        "path": str(dataset_path.resolve()),
        "num_frames": 2,
        "width": 3,
        "height": 1,
    }
    assert summary["roi"] == {
        "key": ROI_KEY,
        "config": str(roi_path.resolve()),
        "x": 0,
        "y": 0,
        "width": 3,
        "height": 1,
        "pixel_count": 3,
    }
    assert summary["depth_preprocessing"] == {
        "excluded_raw_values": [0, 65535],
        "depth_scale_to_mm": 1.0,
    }
    assert summary["depth_quality"] == {
        "zero_ratio": pytest.approx(1 / 6),
        "max_uint16": {
            "ratio": pytest.approx(1 / 6),
            "affected_frames": 1,
            "max_pixels_per_frame": 1,
        },
    }
    assert summary["temporal_noise"] == {
        "min_valid_ratio": 0.9,
        "median_std_mm": 2.0,
        "mean_std_mm": 2.0,
        "p95_std_mm": 2.0,
    }
    assert summary["measured_depth"] == {
        "median_mm": 15.0,
        "mean_mm": 15.0,
        "std_mm": 5.0,
        "p05_mm": pytest.approx(10.5),
        "p95_mm": pytest.approx(19.5),
    }


def test_build_summary_converts_undefined_metrics_to_none(
    tmp_path: Path,
) -> None:
    dataset_dir = tmp_path / "data" / EXPERIMENT_NAME
    roi_root = tmp_path / "config" / "roi"
    depth = np.zeros((2, 1, 2), dtype=np.uint16)
    _write_dataset(dataset_dir, depth)
    _write_roi(
        roi_root,
        RectROI(x=0, y=0, width=2, height=1),
    )
    result = analyze_baseline(dataset_dir, roi_root)

    summary = build_summary(result)

    assert summary["depth_quality"]["zero_ratio"] == 1.0
    assert summary["temporal_noise"]["median_std_mm"] is None
    assert summary["temporal_noise"]["mean_std_mm"] is None
    assert summary["temporal_noise"]["p95_std_mm"] is None
    assert summary["measured_depth"]["median_mm"] is None
    assert summary["measured_depth"]["mean_mm"] is None
    assert summary["measured_depth"]["std_mm"] is None
    assert summary["measured_depth"]["p05_mm"] is None
    assert summary["measured_depth"]["p95_mm"] is None


def test_save_summary_creates_parent_and_writes_yaml(
    tmp_path: Path,
) -> None:
    summary_path = (
        tmp_path
        / "results"
        / EXPERIMENT_NAME
        / "baseline"
        / "summary.yaml"
    )
    summary = {
        "dataset": {
            "experiment": EXPERIMENT_NAME,
        },
        "measured_depth": {
            "median_mm": None,
        },
    }

    save_summary(summary_path, summary)

    with summary_path.open("r", encoding="utf-8") as stream:
        loaded = yaml.safe_load(stream)

    assert loaded == summary


def test_save_summary_rejects_existing_file(
    tmp_path: Path,
) -> None:
    summary_path = tmp_path / "summary.yaml"
    original = {"value": 1}

    save_summary(summary_path, original)

    with pytest.raises(FileExistsError):
        save_summary(summary_path, {"value": 2})

    with summary_path.open("r", encoding="utf-8") as stream:
        loaded = yaml.safe_load(stream)

    assert loaded == original


def test_save_frame_median_csv_writes_timestamp_aligned_rows(
    tmp_path: Path,
) -> None:
    dataset_dir = tmp_path / "data" / EXPERIMENT_NAME
    roi_root = tmp_path / "config" / "roi"
    depth = np.array(
        [
            [[10, 12]],
            [[0, 65535]],
            [[20, 24]],
        ],
        dtype=np.uint16,
    )
    _write_dataset(dataset_dir, depth)
    _write_roi(
        roi_root,
        RectROI(x=0, y=0, width=2, height=1),
    )
    result = analyze_baseline(dataset_dir, roi_root)
    csv_path = (
        tmp_path
        / "results"
        / EXPERIMENT_NAME
        / "baseline"
        / "frame_median_depth.csv"
    )

    save_frame_median_csv(csv_path, result)

    with csv_path.open("r", encoding="utf-8", newline="") as stream:
        rows = list(csv.reader(stream))

    assert rows == [
        ["frame_index", "timestamp_ns", "median_depth_mm"],
        ["0", "0", "11.0"],
        ["1", "1", ""],
        ["2", "2", "22.0"],
    ]


def test_save_frame_median_csv_rejects_existing_file(
    tmp_path: Path,
) -> None:
    dataset_dir = tmp_path / "data" / EXPERIMENT_NAME
    roi_root = tmp_path / "config" / "roi"
    depth = np.array([[[10, 12]]], dtype=np.uint16)
    _write_dataset(dataset_dir, depth)
    _write_roi(
        roi_root,
        RectROI(x=0, y=0, width=2, height=1),
    )
    result = analyze_baseline(dataset_dir, roi_root)
    csv_path = tmp_path / "frame_median_depth.csv"

    save_frame_median_csv(csv_path, result)

    with pytest.raises(FileExistsError):
        save_frame_median_csv(csv_path, result)

    with csv_path.open("r", encoding="utf-8", newline="") as stream:
        rows = list(csv.reader(stream))

    assert rows == [
        ["frame_index", "timestamp_ns", "median_depth_mm"],
        ["0", "0", "11.0"],
    ]


def test_save_metric_maps_writes_expected_npy_arrays(
    tmp_path: Path,
) -> None:
    dataset_dir = tmp_path / "data" / EXPERIMENT_NAME
    roi_root = tmp_path / "config" / "roi"
    depth = np.array(
        [
            [[0, 10, 65535]],
            [[20, 14, 30]],
        ],
        dtype=np.uint16,
    )
    _write_dataset(dataset_dir, depth)
    _write_roi(
        roi_root,
        RectROI(x=0, y=0, width=3, height=1),
    )
    result = analyze_baseline(dataset_dir, roi_root)
    output_dir = tmp_path / "results" / EXPERIMENT_NAME / "baseline"

    save_metric_maps(output_dir, result)

    temporal_std = np.load(
        output_dir / "temporal_std.npy",
        allow_pickle=False,
    )
    zero_ratio = np.load(
        output_dir / "zero_ratio_map.npy",
        allow_pickle=False,
    )
    max_uint16_ratio = np.load(
        output_dir / "max_uint16_ratio_map.npy",
        allow_pickle=False,
    )

    np.testing.assert_allclose(
        temporal_std,
        [[np.nan, 2.0, np.nan]],
        equal_nan=True,
    )
    np.testing.assert_allclose(
        zero_ratio,
        [[0.5, 0.0, 0.0]],
    )
    np.testing.assert_allclose(
        max_uint16_ratio,
        [[0.0, 0.0, 0.5]],
    )
    assert temporal_std.shape == (1, 3)
    assert temporal_std.dtype == np.float64
    assert zero_ratio.dtype == np.float64
    assert max_uint16_ratio.dtype == np.float64


def test_save_metric_maps_checks_all_conflicts_before_writing(
    tmp_path: Path,
) -> None:
    dataset_dir = tmp_path / "data" / EXPERIMENT_NAME
    roi_root = tmp_path / "config" / "roi"
    depth = np.array(
        [
            [[0, 10, 65535]],
            [[20, 14, 30]],
        ],
        dtype=np.uint16,
    )
    _write_dataset(dataset_dir, depth)
    _write_roi(
        roi_root,
        RectROI(x=0, y=0, width=3, height=1),
    )
    result = analyze_baseline(dataset_dir, roi_root)
    output_dir = tmp_path / "results" / EXPERIMENT_NAME / "baseline"
    output_dir.mkdir(parents=True)
    existing_path = output_dir / "zero_ratio_map.npy"
    with existing_path.open("xb") as stream:
        np.save(
            stream,
            np.array([[99.0]], dtype=np.float64),
            allow_pickle=False,
        )

    with pytest.raises(FileExistsError, match="zero_ratio_map.npy"):
        save_metric_maps(output_dir, result)

    assert not (output_dir / "temporal_std.npy").exists()
    assert not (output_dir / "max_uint16_ratio_map.npy").exists()
    np.testing.assert_array_equal(
        np.load(existing_path, allow_pickle=False),
        [[99.0]],
    )


def test_save_baseline_analysis_writes_all_artifacts(
    tmp_path: Path,
) -> None:
    dataset_dir = tmp_path / "data" / EXPERIMENT_NAME
    roi_root = tmp_path / "config" / "roi"
    depth = np.array(
        [
            [[0, 10, 65535]],
            [[20, 14, 30]],
        ],
        dtype=np.uint16,
    )
    _write_dataset(dataset_dir, depth)
    _write_roi(
        roi_root,
        RectROI(x=0, y=0, width=3, height=1),
    )
    result = analyze_baseline(dataset_dir, roi_root)
    output_dir = tmp_path / "results" / EXPERIMENT_NAME / "baseline"

    saved_dir = save_baseline_analysis(output_dir, result)

    assert saved_dir == output_dir
    assert {
        path.name
        for path in output_dir.iterdir()
    } == {
        "summary.yaml",
        "frame_median_depth.csv",
        "temporal_std.npy",
        "zero_ratio_map.npy",
        "max_uint16_ratio_map.npy",
    }
    with (output_dir / "summary.yaml").open(
        "r",
        encoding="utf-8",
    ) as stream:
        summary = yaml.safe_load(stream)
    assert summary["dataset"]["experiment"] == EXPERIMENT_NAME

    temporal_std = np.load(
        output_dir / "temporal_std.npy",
        allow_pickle=False,
    )
    assert temporal_std.shape == (1, 3)


def test_save_baseline_analysis_checks_all_conflicts_before_writing(
    tmp_path: Path,
) -> None:
    dataset_dir = tmp_path / "data" / EXPERIMENT_NAME
    roi_root = tmp_path / "config" / "roi"
    depth = np.array([[[10, 12]]], dtype=np.uint16)
    _write_dataset(dataset_dir, depth)
    _write_roi(
        roi_root,
        RectROI(x=0, y=0, width=2, height=1),
    )
    result = analyze_baseline(dataset_dir, roi_root)
    output_dir = tmp_path / "results" / EXPERIMENT_NAME / "baseline"
    output_dir.mkdir(parents=True)
    summary_path = output_dir / "summary.yaml"
    summary_path.write_text("existing: true\n", encoding="utf-8")

    with pytest.raises(FileExistsError, match="summary.yaml"):
        save_baseline_analysis(output_dir, result)

    assert summary_path.read_text(encoding="utf-8") == "existing: true\n"
    assert not (output_dir / "frame_median_depth.csv").exists()
    assert not (output_dir / "temporal_std.npy").exists()
    assert not (output_dir / "zero_ratio_map.npy").exists()
    assert not (output_dir / "max_uint16_ratio_map.npy").exists()
