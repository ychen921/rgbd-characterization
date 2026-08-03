"""Tests for controlled Scene 01--03 baseline input validation."""

import csv
from io import StringIO
from pathlib import Path

import numpy as np
import pytest
import yaml

import tools.summarize_baseline as summarize_baseline


FRAME_MEDIAN_COLUMNS = (
    "frame_index",
    "timestamp_ns",
    "median_depth_mm",
)
FRAME_PLANE_COLUMNS = (
    "frame_index",
    "timestamp_ns",
    "fit_succeeded",
    "valid_points",
    "normal_x",
    "normal_y",
    "normal_z",
    "plane_distance_m",
    "tilt_deg",
    "residual_rmse_mm",
    "residual_std_mm",
    "residual_p95_abs_mm",
    "inlier_ratio",
)


def _summary(
    experiment: str,
    roi_path: Path,
) -> dict[str, object]:
    identity = summarize_baseline.parse_experiment_name(experiment)
    distance_m = identity.nominal_distance_mm / 1000.0
    return {
        "dataset": {
            "experiment": experiment,
            "path": f"data/{experiment}/depth.npz",
            "num_frames": 2,
            "width": 8,
            "height": 6,
        },
        "roi": {
            "key": identity.condition,
            "config": str(roi_path),
            "x": 1,
            "y": 2,
            "width": 3,
            "height": 2,
            "pixel_count": 6,
        },
        "depth_camera": {
            "config": "config/calib/depth_camera_info.yaml",
            "frame_id": "camera_depth_optical_frame",
            "width": 8,
            "height": 6,
            "fx": 100.0,
            "fy": 101.0,
            "cx": 4.0,
            "cy": 3.0,
        },
        "depth_preprocessing": {
            "excluded_raw_values": [0, 65535],
            "depth_scale_to_mm": 1.0,
        },
        "depth_quality": {
            "zero_ratio": 0.01,
            "max_uint16": {
                "ratio": 0.0,
                "affected_frames": 0,
                "max_pixels_per_frame": 0,
            },
        },
        "temporal_noise": {
            "min_valid_ratio": 0.9,
            "median_std_mm": 0.5,
            "mean_std_mm": 0.6,
            "p95_std_mm": 0.8,
        },
        "measured_depth": {
            "median_mm": identity.nominal_distance_mm + 2.0,
            "mean_mm": identity.nominal_distance_mm + 2.1,
            "std_mm": 0.2,
            "p05_mm": identity.nominal_distance_mm + 1.0,
            "p95_mm": identity.nominal_distance_mm + 3.0,
        },
        "planarity": {
            "fitting_method": "svd",
            "inlier_threshold_mm": 5.0,
            "min_valid_points": 100,
            "successful_frames": 2,
            "failed_frames": 0,
            "plane_distance": {
                "median_m": distance_m,
                "std_mm": 0.1,
            },
            "tilt": {
                "median_deg": float(identity.yaw_deg or 1),
                "std_deg": 0.1,
            },
            "residual": {
                "median_rmse_mm": 0.5,
                "p95_rmse_mm": 0.7,
                "median_p95_abs_mm": 0.9,
            },
            "inlier_ratio": {
                "median": 0.99,
            },
        },
    }


def _write_csv(
    path: Path,
    fieldnames: tuple[str, ...],
    rows: list[dict[str, object]],
) -> None:
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _write_result(
    results_root: Path,
    roi_root: Path,
    experiment: str,
) -> Path:
    identity = summarize_baseline.parse_experiment_name(experiment)
    roi_path = roi_root / f"{identity.condition}.yaml"
    if not roi_path.exists():
        roi_path.parent.mkdir(parents=True, exist_ok=True)
        roi_path.write_text(
            yaml.safe_dump(
                {
                    "name": identity.condition,
                    "source": {
                        "experiment": experiment,
                        "frame_index": 1,
                    },
                    "roi": {
                        "type": "rectangle",
                        "x": 1,
                        "y": 2,
                        "width": 3,
                        "height": 2,
                    },
                },
                sort_keys=False,
            ),
            encoding="utf-8",
        )

    result_dir = results_root / experiment / "baseline"
    result_dir.mkdir(parents=True, exist_ok=True)
    (result_dir / summarize_baseline.SUMMARY_FILENAME).write_text(
        yaml.safe_dump(
            _summary(experiment, roi_path),
            sort_keys=False,
        ),
        encoding="utf-8",
    )

    identity = summarize_baseline.parse_experiment_name(experiment)
    medians = [
        {
            "frame_index": index,
            "timestamp_ns": 1000 + index,
            "median_depth_mm": (
                identity.nominal_distance_mm + index
            ),
        }
        for index in range(2)
    ]
    _write_csv(
        result_dir / summarize_baseline.FRAME_MEDIAN_FILENAME,
        FRAME_MEDIAN_COLUMNS,
        medians,
    )
    planes = [
        {
            "frame_index": index,
            "timestamp_ns": 1000 + index,
            "fit_succeeded": "true",
            "valid_points": 6,
            "normal_x": 0.0,
            "normal_y": 0.0,
            "normal_z": 1.0,
            "plane_distance_m": (
                identity.nominal_distance_mm / 1000.0
            ),
            "tilt_deg": float(identity.yaw_deg or 1),
            "residual_rmse_mm": 0.5,
            "residual_std_mm": 0.5,
            "residual_p95_abs_mm": 0.9,
            "inlier_ratio": 0.99,
        }
        for index in range(2)
    ]
    _write_csv(
        result_dir / summarize_baseline.FRAME_PLANE_FILENAME,
        FRAME_PLANE_COLUMNS,
        planes,
    )
    np.save(
        result_dir / summarize_baseline.TEMPORAL_MAP_FILENAME,
        np.full((2, 3), 0.5, dtype=np.float32),
        allow_pickle=False,
    )
    np.save(
        result_dir / summarize_baseline.ZERO_RATIO_MAP_FILENAME,
        np.zeros((2, 3), dtype=np.float64),
        allow_pickle=False,
    )
    np.save(
        result_dir / summarize_baseline.MAX_UINT16_MAP_FILENAME,
        np.zeros((2, 3), dtype=np.float64),
        allow_pickle=False,
    )
    return result_dir


def _write_matrix(
    root: Path,
    *,
    omit: str | None = None,
) -> tuple[Path, Path]:
    results_root = root / "results"
    roi_root = root / "config" / "roi"
    results_root.mkdir(parents=True, exist_ok=True)
    for experiment in summarize_baseline.EXPECTED_EXPERIMENTS:
        if experiment != omit:
            _write_result(results_root, roi_root, experiment)
    return results_root, roi_root


def _load_summary(path: Path) -> dict[str, object]:
    loaded = yaml.safe_load(path.read_text(encoding="utf-8"))
    assert isinstance(loaded, dict)
    return loaded


def _save_summary(path: Path, summary: dict[str, object]) -> None:
    path.write_text(
        yaml.safe_dump(summary, sort_keys=False),
        encoding="utf-8",
    )


def test_expected_matrix_and_name_parsing() -> None:
    experiments = summarize_baseline.build_expected_experiments()

    assert len(experiments) == 42
    assert len(set(experiments)) == 42
    assert experiments[0] == "scene01_white_d050_r01"
    assert experiments[-1] == "scene03_transparent_d100_r03"

    scene01 = summarize_baseline.parse_experiment_name(
        "scene01_white_d150_r02"
    )
    assert scene01.scene == "scene01"
    assert scene01.nominal_distance_mm == 1500
    assert scene01.yaw_deg is None
    assert scene01.repeat_index == 2

    yaw00 = summarize_baseline.parse_experiment_name(
        "scene02_white_d100_r01"
    )
    assert yaw00.yaw_deg == 0
    yaw30 = summarize_baseline.parse_experiment_name(
        "scene02_yaw30_white_d100_r03"
    )
    assert yaw30.yaw_deg == 30

    scene03 = summarize_baseline.parse_experiment_name(
        "scene03_reflection_d100_r02"
    )
    assert scene03.target == "reflection"


@pytest.mark.parametrize(
    "experiment",
    [
        "scene01_white_d250_r01",
        "scene02_yaw20_white_d100_r01",
        "scene03_metal_d100_r01",
        "scene01_white_d050_r04",
        "scene04_gap030_vertical_white_d100_r01",
    ],
)
def test_parse_rejects_names_outside_controlled_matrix(
    experiment: str,
) -> None:
    with pytest.raises(ValueError, match="controlled baseline matrix"):
        summarize_baseline.parse_experiment_name(experiment)


def test_load_and_validate_complete_matrix_orders_records(
    tmp_path: Path,
) -> None:
    results_root, _ = _write_matrix(tmp_path)

    comparison = (
        summarize_baseline.load_and_validate_baseline_comparison(
            results_root
        )
    )

    assert len(comparison.records) == 42
    assert [
        record.identity.experiment
        for record in comparison.records
    ] == list(summarize_baseline.EXPECTED_EXPERIMENTS)
    assert comparison.records[0].temporal_map_shape == (2, 3)


def test_discovery_rejects_missing_and_unexpected_results(
    tmp_path: Path,
) -> None:
    missing_experiment = "scene03_transparent_d100_r03"
    results_root, roi_root = _write_matrix(
        tmp_path,
        omit=missing_experiment,
    )

    with pytest.raises(ValueError, match="missing"):
        summarize_baseline.discover_baseline_result_dirs(results_root)

    _write_result(results_root, roi_root, missing_experiment)
    unexpected = results_root / "scene01_white_d250_r01" / "baseline"
    unexpected.mkdir(parents=True)
    with pytest.raises(ValueError, match="controlled baseline matrix"):
        summarize_baseline.discover_baseline_result_dirs(results_root)


def test_load_rejects_missing_artifact(tmp_path: Path) -> None:
    results_root, roi_root = _write_matrix(
        tmp_path,
        omit=summarize_baseline.EXPECTED_EXPERIMENTS[1],
    )
    result_dir = _write_result(
        results_root,
        roi_root,
        summarize_baseline.EXPECTED_EXPERIMENTS[1],
    )
    (result_dir / summarize_baseline.TEMPORAL_MAP_FILENAME).unlink()

    with pytest.raises(FileNotFoundError, match="temporal_std.npy"):
        summarize_baseline.load_baseline_summary_record(result_dir)


def test_load_rejects_roi_csv_and_map_mismatches(
    tmp_path: Path,
) -> None:
    results_root, roi_root = _write_matrix(
        tmp_path,
        omit=summarize_baseline.EXPECTED_EXPERIMENTS[0],
    )
    experiment = summarize_baseline.EXPECTED_EXPERIMENTS[0]
    result_dir = _write_result(results_root, roi_root, experiment)
    summary_path = result_dir / summarize_baseline.SUMMARY_FILENAME

    summary = _load_summary(summary_path)
    summary["roi"]["width"] = 4
    summary["roi"]["pixel_count"] = 8
    _save_summary(summary_path, summary)
    with pytest.raises(ValueError, match="does not match summary"):
        summarize_baseline.load_baseline_summary_record(result_dir)

    _save_summary(
        summary_path,
        _summary(
            experiment,
            roi_root
            / f"{summarize_baseline.parse_experiment_name(experiment).condition}.yaml",
        ),
    )
    frame_path = result_dir / summarize_baseline.FRAME_MEDIAN_FILENAME
    with frame_path.open(encoding="utf-8", newline="") as stream:
        rows = list(csv.DictReader(stream))
    _write_csv(frame_path, FRAME_MEDIAN_COLUMNS, rows[:1])
    with pytest.raises(ValueError, match="row count"):
        summarize_baseline.load_baseline_summary_record(result_dir)

    _write_csv(
        frame_path,
        FRAME_MEDIAN_COLUMNS,
        [
            {
                "frame_index": index,
                "timestamp_ns": 1000 + index,
                "median_depth_mm": 500 + index,
            }
            for index in range(2)
        ],
    )
    np.save(
        result_dir / summarize_baseline.ZERO_RATIO_MAP_FILENAME,
        np.zeros((1, 3), dtype=np.float64),
        allow_pickle=False,
    )
    with pytest.raises(ValueError, match="Map shape"):
        summarize_baseline.load_baseline_summary_record(result_dir)


def test_load_rejects_frame_alignment_and_summary_count_mismatch(
    tmp_path: Path,
) -> None:
    results_root, roi_root = _write_matrix(
        tmp_path,
        omit=summarize_baseline.EXPECTED_EXPERIMENTS[0],
    )
    experiment = summarize_baseline.EXPECTED_EXPERIMENTS[0]
    result_dir = _write_result(results_root, roi_root, experiment)
    plane_path = result_dir / summarize_baseline.FRAME_PLANE_FILENAME
    with plane_path.open(encoding="utf-8", newline="") as stream:
        rows = list(csv.DictReader(stream))
    rows[1]["timestamp_ns"] = "9999"
    _write_csv(plane_path, FRAME_PLANE_COLUMNS, rows)
    with pytest.raises(ValueError, match="timestamps are not aligned"):
        summarize_baseline.load_baseline_summary_record(result_dir)

    rows[1]["timestamp_ns"] = "1001"
    _write_csv(plane_path, FRAME_PLANE_COLUMNS, rows)
    summary_path = result_dir / summarize_baseline.SUMMARY_FILENAME
    summary = _load_summary(summary_path)
    summary["planarity"]["successful_frames"] = 1
    summary["planarity"]["failed_frames"] = 1
    _save_summary(summary_path, summary)
    with pytest.raises(ValueError, match="fit_succeeded count"):
        summarize_baseline.load_baseline_summary_record(result_dir)


def test_comparison_rejects_repeat_roi_and_common_setting_mismatch(
    tmp_path: Path,
) -> None:
    results_root, roi_root = _write_matrix(tmp_path)
    records = [
        summarize_baseline.load_baseline_summary_record(path)
        for path in summarize_baseline.discover_baseline_result_dirs(
            results_root
        )
    ]

    changed = records[2]
    alternate_roi = roi_root / "alternate.yaml"
    original_roi = Path(changed.summary["roi"]["config"])
    alternate_roi.write_text(
        original_roi.read_text(encoding="utf-8"),
        encoding="utf-8",
    )
    changed.summary["roi"]["config"] = str(alternate_roi)
    with pytest.raises(ValueError, match="inconsistent ROI"):
        summarize_baseline.validate_baseline_comparison(records)

    changed.summary["roi"]["config"] = str(original_roi)
    changed.summary["depth_camera"]["fx"] = 999.0
    with pytest.raises(ValueError, match="common settings"):
        summarize_baseline.validate_baseline_comparison(records)


def test_load_rejects_invalid_ratio_map_values(tmp_path: Path) -> None:
    results_root, roi_root = _write_matrix(
        tmp_path,
        omit=summarize_baseline.EXPECTED_EXPERIMENTS[0],
    )
    result_dir = _write_result(
        results_root,
        roi_root,
        summarize_baseline.EXPECTED_EXPERIMENTS[0],
    )
    invalid = np.zeros((2, 3), dtype=np.float64)
    invalid[0, 0] = 1.1
    np.save(
        result_dir / summarize_baseline.ZERO_RATIO_MAP_FILENAME,
        invalid,
        allow_pickle=False,
    )

    with pytest.raises(ValueError, match="above one"):
        summarize_baseline.load_baseline_summary_record(result_dir)


def test_build_recording_rows_derives_identity_units_and_offsets(
    tmp_path: Path,
) -> None:
    results_root, _ = _write_matrix(tmp_path)
    comparison = (
        summarize_baseline.load_and_validate_baseline_comparison(
            results_root
        )
    )

    rows = summarize_baseline.build_recording_rows(comparison)

    assert len(rows) == 42
    first = rows[0]
    assert tuple(first) == summarize_baseline.RECORDING_COLUMNS
    assert first["experiment"] == "scene01_white_d050_r01"
    assert first["scene"] == "scene01"
    assert first["condition"] == "scene01_white_d050"
    assert first["target"] == "white"
    assert first["nominal_distance_mm"] == 500
    assert first["yaw_deg"] is None
    assert first["repeat_index"] == 1
    assert first["roi_width"] == 3
    assert first["roi_height"] == 2
    assert first["roi_pixels"] == 6
    assert first["measured_median_mm"] == 502.0
    assert first["measured_offset_from_nominal_mm"] == 2.0
    assert first["plane_distance_median_mm"] == 500.0
    assert first["plane_distance_offset_from_nominal_mm"] == 0.0
    assert first["plane_success_ratio"] == 1.0
    assert first["tilt_error_from_nominal_deg"] is None

    yaw30 = next(
        row
        for row in rows
        if row["experiment"]
        == "scene02_yaw30_white_d100_r01"
    )
    assert yaw30["yaw_deg"] == 30
    assert yaw30["tilt_median_deg"] == 30.0
    assert yaw30["tilt_error_from_nominal_deg"] == 0.0


def test_recording_csv_is_deterministic_and_preserves_empty_values(
    tmp_path: Path,
) -> None:
    results_root, _ = _write_matrix(tmp_path)
    comparison = (
        summarize_baseline.load_and_validate_baseline_comparison(
            results_root
        )
    )

    first_csv = summarize_baseline.build_baseline_summary_csv(
        comparison
    )
    second_csv = summarize_baseline.build_baseline_summary_csv(
        comparison
    )
    rows = list(csv.DictReader(StringIO(first_csv)))

    assert first_csv == second_csv
    assert len(rows) == 42
    assert tuple(rows[0]) == summarize_baseline.RECORDING_COLUMNS
    assert rows[0]["yaw_deg"] == ""
    assert rows[0]["tilt_error_from_nominal_deg"] == ""
    assert rows[-1]["experiment"] == (
        "scene03_transparent_d100_r03"
    )


def test_aggregate_metric_uses_sample_std_and_tracks_coverage() -> None:
    complete = summarize_baseline._aggregate_metric(
        [1.0, 2.0, 3.0]
    )
    assert complete.mean == pytest.approx(2.0)
    assert complete.repeat_std == pytest.approx(1.0)
    assert complete.valid_count == 3

    partial = summarize_baseline._aggregate_metric(
        [1.0, None, 3.0]
    )
    assert partial.mean == pytest.approx(2.0)
    assert partial.repeat_std == pytest.approx(np.sqrt(2.0))
    assert partial.valid_count == 2

    single = summarize_baseline._aggregate_metric(
        [None, np.float64(4.0), np.nan]
    )
    assert single.mean == 4.0
    assert single.repeat_std is None
    assert single.valid_count == 1

    empty = summarize_baseline._aggregate_metric(
        [None, np.nan, None]
    )
    assert empty.mean is None
    assert empty.repeat_std is None
    assert empty.valid_count == 0


def test_build_condition_rows_aggregates_repeats_in_fixed_order(
    tmp_path: Path,
) -> None:
    results_root, _ = _write_matrix(tmp_path)
    comparison = (
        summarize_baseline.load_and_validate_baseline_comparison(
            results_root
        )
    )
    for value, record in zip(
        (0.1, 0.2, 0.3),
        comparison.records[:3],
        strict=True,
    ):
        record.summary["depth_quality"]["zero_ratio"] = value

    rows = summarize_baseline.build_condition_rows(comparison)

    assert len(rows) == 14
    first = rows[0]
    assert tuple(first) == summarize_baseline.CONDITION_COLUMNS
    assert first["condition"] == "scene01_white_d050"
    assert first["repeat_count"] == 3
    assert first["total_frames"] == 6
    assert first["min_frames_per_repeat"] == 2
    assert first["max_frames_per_repeat"] == 2
    assert first["zero_ratio_mean"] == pytest.approx(0.2)
    assert first["zero_ratio_repeat_std"] == pytest.approx(0.1)
    assert first["zero_ratio_valid_count"] == 3
    assert "max_uint16_affected_frames_mean" not in first

    assert [
        row["yaw_deg"]
        for row in rows
        if row["scene"] == "scene02"
    ] == [0, 15, 30, 45, 60]
    assert [
        row["target"]
        for row in rows
        if row["scene"] == "scene03"
    ] == list(summarize_baseline.SCENE03_TARGETS)


def test_condition_rows_preserve_undefined_metric_coverage(
    tmp_path: Path,
) -> None:
    results_root, _ = _write_matrix(tmp_path)
    comparison = (
        summarize_baseline.load_and_validate_baseline_comparison(
            results_root
        )
    )
    comparison.records[0].summary["measured_depth"]["median_mm"] = None
    comparison.records[1].summary["measured_depth"]["median_mm"] = None

    condition = summarize_baseline.build_condition_rows(comparison)[0]

    assert condition["measured_median_mm_mean"] == 502.0
    assert condition["measured_median_mm_repeat_std"] is None
    assert condition["measured_median_mm_valid_count"] == 1
    assert condition["measured_offset_from_nominal_mm_mean"] == 2.0
    assert condition[
        "measured_offset_from_nominal_mm_repeat_std"
    ] is None

    csv_text = summarize_baseline.build_condition_summary_csv(
        comparison
    )
    rows = list(csv.DictReader(StringIO(csv_text)))
    assert len(rows) == 14
    assert tuple(rows[0]) == summarize_baseline.CONDITION_COLUMNS
    assert rows[0]["measured_median_mm_repeat_std"] == ""
    assert rows[0]["measured_median_mm_valid_count"] == "1"
