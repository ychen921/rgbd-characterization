"""Tests for baseline analysis orchestration."""

import csv
from pathlib import Path
import sys

import numpy as np
import pytest
import yaml

from src.io.dataset import DepthDataset
from src.geometry.camera import CameraIntrinsics
from src.preprocessing.roi import RectROI, save_roi
from tools.analyze_baseline import (
    DEFAULT_DEPTH_CAMERA_INFO_PATH,
    DEFAULT_RESULTS_ROOT,
    analyze_baseline,
    build_summary,
    compute_baseline_metrics,
    load_baseline_input,
    main,
    resolve_output_dir,
    save_baseline_analysis,
    save_frame_median_csv,
    save_frame_plane_metrics_csv,
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


def _write_camera_info(
    path: Path,
    *,
    width: int,
    height: int,
) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    document = {
        "header": {
            "frame_id": "camera_depth_optical_frame",
        },
        "height": height,
        "width": width,
        "k": [
            100.0,
            0.0,
            (width - 1) / 2.0,
            0.0,
            100.0,
            (height - 1) / 2.0,
            0.0,
            0.0,
            1.0,
        ],
    }
    with path.open("w", encoding="utf-8") as stream:
        yaml.safe_dump(document, stream)
    return path


def _intrinsics(width: int, height: int) -> CameraIntrinsics:
    return CameraIntrinsics(
        width=width,
        height=height,
        fx=100.0,
        fy=100.0,
        cx=(width - 1) / 2.0,
        cy=(height - 1) / 2.0,
        frame_id="camera_depth_optical_frame",
    )


def _analyze_baseline(
    dataset_dir: Path,
    roi_root: Path,
    **kwargs: object,
):
    with np.load(
        dataset_dir / "depth.npz",
        allow_pickle=False,
    ) as archive:
        _, height, width = archive["depth"].shape

    camera_info_path = _write_camera_info(
        dataset_dir.parents[1]
        / "config"
        / "calib"
        / "depth_camera_info.yaml",
        width=width,
        height=height,
    )

    return analyze_baseline(
        dataset_dir,
        roi_root,
        depth_camera_info_path=camera_info_path,
        **kwargs,
    )


def test_compute_baseline_metrics_combines_all_metrics() -> None:
    raw_roi = np.array(
        [
            [[0, 10, 65535]],
            [[20, 14, 30]],
        ],
        dtype=np.uint16,
    )

    result = compute_baseline_metrics(
        raw_roi,
        intrinsics=_intrinsics(width=3, height=1),
        roi=RectROI(x=0, y=0, width=3, height=1),
    )

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
        intrinsics=_intrinsics(width=3, height=1),
        roi=RectROI(x=0, y=0, width=3, height=1),
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

    result = _analyze_baseline(dataset_dir, roi_root)

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
    result = _analyze_baseline(dataset_dir, roi_root)

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
    assert summary["depth_camera"] == {
        "config": str(result.depth_camera_info_path.resolve()),
        "frame_id": "camera_depth_optical_frame",
        "width": 3,
        "height": 1,
        "fx": 100.0,
        "fy": 100.0,
        "cx": 1.0,
        "cy": 0.0,
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
    assert summary["planarity"] == {
        "fitting_method": "svd",
        "inlier_threshold_mm": 5.0,
        "min_valid_points": 100,
        "successful_frames": 0,
        "failed_frames": 2,
        "plane_distance": {
            "median_m": None,
            "std_mm": None,
        },
        "tilt": {
            "median_deg": None,
            "std_deg": None,
        },
        "residual": {
            "median_rmse_mm": None,
            "p95_rmse_mm": None,
            "median_p95_abs_mm": None,
        },
        "inlier_ratio": {
            "median": None,
        },
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
    result = _analyze_baseline(dataset_dir, roi_root)

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
    result = _analyze_baseline(dataset_dir, roi_root)
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
    result = _analyze_baseline(dataset_dir, roi_root)
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
    result = _analyze_baseline(dataset_dir, roi_root)
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
    result = _analyze_baseline(dataset_dir, roi_root)
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
    result = _analyze_baseline(dataset_dir, roi_root)
    output_dir = tmp_path / "results" / EXPERIMENT_NAME / "baseline"

    saved_dir = save_baseline_analysis(output_dir, result)

    assert saved_dir == output_dir
    assert {
        path.name
        for path in output_dir.iterdir()
    } == {
        "summary.yaml",
        "frame_median_depth.csv",
        "frame_plane_metrics.csv",
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
    result = _analyze_baseline(dataset_dir, roi_root)
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


def test_resolve_output_dir_uses_default_results_location() -> None:
    output_dir = resolve_output_dir(
        experiment_name=EXPERIMENT_NAME,
        output_dir=None,
    )

    assert output_dir == (
        DEFAULT_RESULTS_ROOT
        / EXPERIMENT_NAME
        / "baseline"
    )


def test_resolve_output_dir_uses_explicit_path(tmp_path: Path) -> None:
    explicit_output_dir = tmp_path / "custom" / "baseline"

    output_dir = resolve_output_dir(
        experiment_name=EXPERIMENT_NAME,
        output_dir=explicit_output_dir,
    )

    assert output_dir == explicit_output_dir


def test_main_runs_complete_cli_pipeline(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    dataset_dir = tmp_path / "data" / EXPERIMENT_NAME
    roi_root = tmp_path / "config" / "roi"
    output_dir = tmp_path / "results" / EXPERIMENT_NAME / "baseline"
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
    camera_info_path = _write_camera_info(
        tmp_path / "config" / "calib" / "depth_camera_info.yaml",
        width=3,
        height=1,
    )
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "analyze_baseline.py",
            str(dataset_dir),
            "--roi-root",
            str(roi_root),
            "--output-dir",
            str(output_dir),
            "--min-valid-ratio",
            "0.5",
            "--depth-camera-info",
            str(camera_info_path),
            "--plane-inlier-threshold-mm",
            "4.0",
            "--plane-min-valid-points",
            "4",
        ],
    )

    exit_code = main()

    assert exit_code == 0
    assert {
        path.name
        for path in output_dir.iterdir()
    } == {
        "summary.yaml",
        "frame_median_depth.csv",
        "frame_plane_metrics.csv",
        "temporal_std.npy",
        "zero_ratio_map.npy",
        "max_uint16_ratio_map.npy",
    }
    with (output_dir / "summary.yaml").open(
        "r",
        encoding="utf-8",
    ) as stream:
        summary = yaml.safe_load(stream)
    assert summary["temporal_noise"]["min_valid_ratio"] == 0.5
    assert summary["planarity"]["inlier_threshold_mm"] == 4.0
    assert summary["planarity"]["min_valid_points"] == 4

    captured = capsys.readouterr()
    assert "Baseline analysis complete." in captured.out
    assert "Saved:" in captured.out
    assert str(output_dir) in captured.out


def test_analyze_baseline_computes_planarity_with_calibration_and_roi_offset(
    tmp_path: Path,
) -> None:
    dataset_dir = tmp_path / "data" / EXPERIMENT_NAME
    roi_root = tmp_path / "config" / "roi"
    depth = np.stack(
        [
            np.full((6, 6), 1000, dtype=np.uint16),
            np.full((6, 6), 1010, dtype=np.uint16),
        ]
    )
    _write_dataset(dataset_dir, depth)
    _write_roi(
        roi_root,
        RectROI(x=1, y=1, width=4, height=4),
    )

    result = _analyze_baseline(
        dataset_dir,
        roi_root,
        plane_inlier_threshold_mm=4.0,
        plane_min_valid_points=3,
    )

    assert result.intrinsics.width == 6
    assert result.intrinsics.height == 6
    assert result.plane_inlier_threshold_mm == 4.0
    assert result.plane_min_valid_points == 3
    assert result.metrics.planarity.successful_frames == 2
    assert result.metrics.planarity.failed_frames == 0
    np.testing.assert_allclose(
        result.metrics.planarity.frame_distance_m,
        [1.0, 1.01],
    )
    np.testing.assert_allclose(
        result.metrics.planarity.frame_normal,
        [[0.0, 0.0, 1.0], [0.0, 0.0, 1.0]],
        atol=1e-12,
    )


def test_analyze_baseline_rejects_camera_info_resolution_mismatch(
    tmp_path: Path,
) -> None:
    dataset_dir = tmp_path / "data" / EXPERIMENT_NAME
    roi_root = tmp_path / "config" / "roi"
    depth = np.full((1, 4, 5), 1000, dtype=np.uint16)
    _write_dataset(dataset_dir, depth)
    _write_roi(
        roi_root,
        RectROI(x=0, y=0, width=4, height=4),
    )
    camera_info_path = _write_camera_info(
        tmp_path / "config" / "calib" / "wrong.yaml",
        width=6,
        height=4,
    )

    with pytest.raises(
        ValueError,
        match=r"depth=5x4, intrinsics=6x4",
    ):
        analyze_baseline(
            dataset_dir,
            roi_root,
            depth_camera_info_path=camera_info_path,
        )


def test_save_frame_plane_metrics_csv_preserves_failed_frame_alignment(
    tmp_path: Path,
) -> None:
    dataset_dir = tmp_path / "data" / EXPERIMENT_NAME
    roi_root = tmp_path / "config" / "roi"
    depth = np.stack(
        [
            np.full((4, 4), 1000, dtype=np.uint16),
            np.zeros((4, 4), dtype=np.uint16),
            np.full((4, 4), 1020, dtype=np.uint16),
        ]
    )
    _write_dataset(dataset_dir, depth)
    _write_roi(
        roi_root,
        RectROI(x=0, y=0, width=4, height=4),
    )
    result = _analyze_baseline(
        dataset_dir,
        roi_root,
        plane_min_valid_points=3,
    )
    csv_path = tmp_path / "frame_plane_metrics.csv"

    save_frame_plane_metrics_csv(csv_path, result)

    with csv_path.open("r", encoding="utf-8", newline="") as stream:
        rows = list(csv.DictReader(stream))

    assert len(rows) == 3
    assert rows[0]["frame_index"] == "0"
    assert rows[0]["timestamp_ns"] == "0"
    assert rows[0]["fit_succeeded"] == "true"
    assert rows[0]["valid_points"] == "16"
    assert float(rows[0]["normal_z"]) == pytest.approx(1.0)
    assert float(rows[0]["plane_distance_m"]) == pytest.approx(1.0)

    assert rows[1]["frame_index"] == "1"
    assert rows[1]["timestamp_ns"] == "1"
    assert rows[1]["fit_succeeded"] == "false"
    assert rows[1]["valid_points"] == "0"
    assert rows[1]["normal_x"] == ""
    assert rows[1]["normal_y"] == ""
    assert rows[1]["normal_z"] == ""
    assert rows[1]["plane_distance_m"] == ""
    assert rows[1]["tilt_deg"] == ""
    assert rows[1]["residual_rmse_mm"] == ""
    assert rows[1]["residual_std_mm"] == ""
    assert rows[1]["residual_p95_abs_mm"] == ""
    assert rows[1]["inlier_ratio"] == ""

    assert rows[2]["frame_index"] == "2"
    assert rows[2]["timestamp_ns"] == "2"
    assert rows[2]["fit_succeeded"] == "true"
    assert rows[2]["valid_points"] == "16"
    assert float(rows[2]["plane_distance_m"]) == pytest.approx(1.02)


def test_save_baseline_analysis_preflights_existing_plane_csv(
    tmp_path: Path,
) -> None:
    dataset_dir = tmp_path / "data" / EXPERIMENT_NAME
    roi_root = tmp_path / "config" / "roi"
    depth = np.full((1, 4, 4), 1000, dtype=np.uint16)
    _write_dataset(dataset_dir, depth)
    _write_roi(
        roi_root,
        RectROI(x=0, y=0, width=4, height=4),
    )
    result = _analyze_baseline(
        dataset_dir,
        roi_root,
        plane_min_valid_points=3,
    )
    output_dir = tmp_path / "results" / EXPERIMENT_NAME / "baseline"
    output_dir.mkdir(parents=True)
    existing_path = output_dir / "frame_plane_metrics.csv"
    existing_path.write_text("existing\n", encoding="utf-8")

    with pytest.raises(
        FileExistsError,
        match="frame_plane_metrics.csv",
    ):
        save_baseline_analysis(output_dir, result)

    assert existing_path.read_text(encoding="utf-8") == "existing\n"
    assert {
        path.name
        for path in output_dir.iterdir()
    } == {
        "frame_plane_metrics.csv",
    }


def test_default_depth_camera_info_path_uses_project_calibration() -> None:
    assert DEFAULT_DEPTH_CAMERA_INFO_PATH == (
        Path(__file__).resolve().parents[1]
        / "config"
        / "calib"
        / "depth_camera_info.yaml"
    )
