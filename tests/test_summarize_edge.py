"""Tests for Scene 04 cross-dataset edge summarization."""

import csv
from io import StringIO
from pathlib import Path

import cv2
import numpy as np
import pytest
import yaml

import tools.summarize_edge as summarize_edge


CONDITIONS = (
    ("horizon", "horizontal", 1000),
    ("vertical", "vertical", 500),
    ("vertical", "vertical", 1000),
    ("vertical", "vertical", 2000),
)


def _experiment(
    orientation_token: str,
    distance_mm: int,
) -> str:
    return (
        f"scene04_gap030_{orientation_token}_white_"
        f"d{distance_mm // 10:03d}_r01"
    )


def _summary(
    *,
    orientation_token: str,
    orientation: str,
    distance_mm: int,
) -> dict[str, object]:
    experiment = _experiment(
        orientation_token,
        distance_mm,
    )
    return {
        "dataset": {
            "experiment": experiment,
            "path": f"data/{experiment}/depth.npz",
            "num_frames": 4,
            "width": 848,
            "height": 480,
        },
        "setup": {
            "orientation_token": orientation_token,
            "orientation": orientation,
            "target": "white",
            "repeat_index": 1,
            "distance_reference": (
                "camera_optical_reference_plane"
            ),
            "nominal_foreground_distance_mm": distance_mm,
            "nominal_gap_mm": 300,
            "nominal_background_distance_mm": (
                distance_mm + 300
            ),
        },
        "roi": {
            "key": experiment.removesuffix("_r01"),
            "config": "config/roi/example.yaml",
            "source_experiment": experiment,
            "source_frame_index": 2,
            "foreground_pixels": 101,
            "background_pixels": 102,
            "edge_pixels": 103,
        },
        "edge_geometry": {
            "nominal_edge_p1": [4.0, 1.0],
            "nominal_edge_p2": [4.0, 7.0],
            "foreground_side": "left",
            "distance_bin_px": 2.0,
            "max_edge_distance_px": 20.0,
        },
        "analysis_parameters": {
            "reference": {
                "minimum_tolerance_mm": 10.0,
                "mad_scale": 3.0,
                "minimum_valid_ratio": 0.9,
                "minimum_valid_count": 100,
            },
            "bleeding_probability_threshold": 0.05,
            "invalid_ratio_threshold": 0.5,
            "transition_high_probability": 0.9,
            "transition_low_probability": 0.1,
        },
        "depth_preprocessing": {
            "excluded_raw_values": [0, 65535],
            "depth_scale_to_mm": 1.0,
        },
        "frames": {
            "valid": 3,
            "rejected": 1,
            "transition_valid": 2,
            "transition_failed": 1,
        },
        "diagnostics": {
            "aggregate_profile_available": True,
            "representative_label_map_available": True,
        },
    }


def _frame_rows(
    distance_mm: int,
) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    statuses = (
        ("ok", "ok"),
        ("ok", "ok"),
        ("ok", "insufficient_probability"),
        ("reference_invalid", "not_analyzed"),
    )
    for index, (analysis_status, transition_status) in enumerate(
        statuses
    ):
        valid = analysis_status == "ok"
        transition_valid = transition_status == "ok"
        rows.append(
            {
                "frame_index": index,
                "timestamp_ns": 1000 + index,
                "foreground_reference_mm": (
                    distance_mm + index if valid else 9999
                ),
                "background_reference_mm": (
                    distance_mm + 300 + index
                    if valid
                    else 9999
                ),
                "measured_gap_mm": (
                    300 + 2 * index if valid else 9999
                ),
                "foreground_bleeding_ratio": (
                    0.01 * index if valid else 0.99
                ),
                "background_bleeding_ratio": (
                    0.02 * index if valid else 0.99
                ),
                "mixed_ratio": (
                    0.03 * index if valid else 0.99
                ),
                "outlier_ratio": (
                    0.04 * index if valid else 0.99
                ),
                "invalid_ratio": (
                    0.05 * index if valid else 0.99
                ),
                "transition_width_px": (
                    2.0 + index if transition_valid else ""
                ),
                "nominal_edge_offset_px": (
                    -1.0 + index
                    if transition_valid
                    else ""
                ),
                "analysis_status": analysis_status,
                "transition_status": transition_status,
            }
        )
    return rows


def _write_frame_metrics(
    path: Path,
    distance_mm: int,
) -> None:
    rows = _frame_rows(distance_mm)
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(
            stream,
            fieldnames=list(rows[0]),
        )
        writer.writeheader()
        writer.writerows(rows)


def _write_profile(
    path: Path,
    *,
    zero_centers: tuple[float, ...] = (),
) -> None:
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(
            stream,
            fieldnames=summarize_edge.PROFILE_COLUMNS,
        )
        writer.writeheader()
        for center in np.arange(-20.0, 22.0, 2.0):
            zero = center in zero_centers
            pixel_count = 0 if zero else 10
            foreground_count = (
                0
                if zero or center > 0
                else pixel_count
            )
            background_count = (
                0
                if zero or center <= 0
                else pixel_count
            )
            writer.writerow(
                {
                    "distance_min_px": (
                        center
                        if center == -20.0
                        else center - 1.0
                    ),
                    "distance_max_px": (
                        center
                        if center == 20.0
                        else center + 1.0
                    ),
                    "distance_center_px": center,
                    "pixel_count": pixel_count,
                    "valid_count": pixel_count,
                    "foreground_count": foreground_count,
                    "background_count": background_count,
                    "mixed_count": 0,
                    "outlier_count": 0,
                    "invalid_count": 0,
                    "foreground_ratio": (
                        ""
                        if zero
                        else foreground_count / pixel_count
                    ),
                    "background_ratio": (
                        ""
                        if zero
                        else background_count / pixel_count
                    ),
                    "mixed_ratio": "" if zero else 0.0,
                    "outlier_ratio": "" if zero else 0.0,
                    "invalid_ratio": "" if zero else 0.0,
                }
            )


def _write_result(
    root: Path,
    orientation_token: str,
    orientation: str,
    distance_mm: int,
    *,
    zero_centers: tuple[float, ...] = (),
) -> Path:
    experiment = _experiment(
        orientation_token,
        distance_mm,
    )
    result_dir = root / experiment / "edge_discontinuity"
    result_dir.mkdir(parents=True)
    summary = _summary(
        orientation_token=orientation_token,
        orientation=orientation,
        distance_mm=distance_mm,
    )
    (result_dir / summarize_edge.SUMMARY_FILENAME).write_text(
        yaml.safe_dump(summary, sort_keys=False),
        encoding="utf-8",
    )
    _write_frame_metrics(
        result_dir / summarize_edge.FRAME_METRICS_FILENAME,
        distance_mm,
    )
    _write_profile(
        result_dir / summarize_edge.PROFILE_FILENAME,
        zero_centers=zero_centers,
    )
    return result_dir


def _write_matrix(
    root: Path,
    *,
    d200_zero_centers: tuple[float, ...] = (),
) -> list[Path]:
    return [
        _write_result(
            root,
            orientation_token,
            orientation,
            distance_mm,
            zero_centers=(
                d200_zero_centers
                if distance_mm == 2000
                else ()
            ),
        )
        for (
            orientation_token,
            orientation,
            distance_mm,
        ) in CONDITIONS
    ]


def _comparison(
    root: Path,
    *,
    d200_zero_centers: tuple[float, ...] = (),
) -> summarize_edge.EdgeComparisonResult:
    paths = _write_matrix(
        root,
        d200_zero_centers=d200_zero_centers,
    )
    records = [
        summarize_edge.load_edge_summary_record(path)
        for path in reversed(paths)
    ]
    return summarize_edge.validate_edge_comparison(records)


def test_parse_args_uses_default_output() -> None:
    args = summarize_edge.parse_args(["one", "two"])

    assert args.result_dirs == [Path("one"), Path("two")]
    assert args.output_dir == summarize_edge.DEFAULT_OUTPUT_DIR


def test_load_and_validate_orders_expected_matrix(
    tmp_path: Path,
) -> None:
    comparison = _comparison(tmp_path)

    assert [
        record.summary["dataset"]["experiment"]
        for record in comparison.records
    ] == [
        "scene04_gap030_horizon_white_d100_r01",
        "scene04_gap030_vertical_white_d050_r01",
        "scene04_gap030_vertical_white_d100_r01",
        "scene04_gap030_vertical_white_d200_r01",
    ]


def test_load_rejects_missing_or_incomplete_inputs(
    tmp_path: Path,
) -> None:
    result_dir = _write_result(
        tmp_path,
        "vertical",
        "vertical",
        500,
    )
    (result_dir / summarize_edge.PROFILE_FILENAME).unlink()

    with pytest.raises(
        FileNotFoundError,
        match="aggregate_edge_profile.csv",
    ):
        summarize_edge.load_edge_summary_record(result_dir)


def test_comparison_rejects_condition_and_setting_mismatch(
    tmp_path: Path,
) -> None:
    paths = _write_matrix(tmp_path)
    records = [
        summarize_edge.load_edge_summary_record(path)
        for path in paths
    ]

    with pytest.raises(ValueError, match="exactly four"):
        summarize_edge.validate_edge_comparison(records[:3])

    changed = dict(records[-1].summary)
    changed["edge_geometry"] = {
        **records[-1].summary["edge_geometry"],
        "max_edge_distance_px": 30.0,
    }
    mismatched = summarize_edge.EdgeSummaryRecord(
        **{
            **records[-1].__dict__,
            "summary": changed,
        }
    )
    with pytest.raises(ValueError, match="common settings"):
        summarize_edge.validate_edge_comparison(
            [*records[:-1], mismatched]
        )


def test_load_rejects_frame_status_count_mismatch(
    tmp_path: Path,
) -> None:
    result_dir = _write_result(
        tmp_path,
        "vertical",
        "vertical",
        500,
    )
    summary_path = result_dir / summarize_edge.SUMMARY_FILENAME
    summary = yaml.safe_load(
        summary_path.read_text(encoding="utf-8")
    )
    summary["frames"]["valid"] = 4
    summary_path.write_text(
        yaml.safe_dump(summary, sort_keys=False),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="Valid-frame count"):
        summarize_edge.load_edge_summary_record(result_dir)


def test_load_rejects_inconsistent_profile_counts(
    tmp_path: Path,
) -> None:
    result_dir = _write_result(
        tmp_path,
        "vertical",
        "vertical",
        500,
    )
    profile_path = result_dir / summarize_edge.PROFILE_FILENAME
    with profile_path.open(
        "r",
        encoding="utf-8",
        newline="",
    ) as stream:
        rows = list(csv.DictReader(stream))
    rows[0]["valid_count"] = "9"
    with profile_path.open(
        "w",
        encoding="utf-8",
        newline="",
    ) as stream:
        writer = csv.DictWriter(
            stream,
            fieldnames=summarize_edge.PROFILE_COLUMNS,
        )
        writer.writeheader()
        writer.writerows(rows)

    with pytest.raises(
        ValueError,
        match="valid and invalid counts",
    ):
        summarize_edge.load_edge_summary_record(result_dir)


def test_zero_pixel_profile_bins_are_reported_not_rejected(
    tmp_path: Path,
) -> None:
    comparison = _comparison(
        tmp_path,
        d200_zero_centers=(-20.0, 20.0),
    )

    summary = summarize_edge.build_comparison_summary(
        comparison
    )
    d200 = summary["profile_coverage"]["experiments"][-1]
    assert d200["populated_bins"] == 19
    assert d200["zero_pixel_distance_centers_px"] == [
        -20.0,
        20.0,
    ]
    assert len(summary["profile_coverage"]["warnings"]) == 1


def test_summary_csv_uses_only_eligible_frames(
    tmp_path: Path,
) -> None:
    comparison = _comparison(tmp_path)

    rows = list(
        csv.DictReader(
            StringIO(
                summarize_edge.build_edge_summary_csv(
                    comparison
                )
            )
        )
    )

    assert len(rows) == 4
    row = rows[0]
    assert row["valid_frames"] == "3"
    assert row["rejected_frames"] == "1"
    assert float(row["measured_gap_median_mm"]) == 302.0
    assert float(row["gap_error_mm"]) == 2.0
    assert float(row["foreground_reference_median_mm"]) == 1001.0
    assert float(row["transition_width_median_px"]) == 2.5
    assert float(row["nominal_offset_std_px"]) == 0.5
    assert float(row["transition_success_ratio"]) == pytest.approx(
        2 / 3
    )


def test_build_artifacts_returns_csv_yaml_and_valid_pngs(
    tmp_path: Path,
) -> None:
    comparison = _comparison(tmp_path)

    artifacts = summarize_edge.build_edge_summary_artifacts(
        comparison
    )

    assert set(artifacts) == {
        summarize_edge.OUTPUT_CSV_FILENAME,
        summarize_edge.OUTPUT_YAML_FILENAME,
        summarize_edge.VERTICAL_METRICS_FILENAME,
        summarize_edge.VERTICAL_PROFILES_FILENAME,
        summarize_edge.ORIENTATION_METRICS_FILENAME,
        summarize_edge.ORIENTATION_PROFILES_FILENAME,
    }
    assert yaml.safe_load(
        artifacts[summarize_edge.OUTPUT_YAML_FILENAME]
    )["repeat_scope"]["repeatability_analysis_available"] is False
    for filename in (
        summarize_edge.VERTICAL_METRICS_FILENAME,
        summarize_edge.VERTICAL_PROFILES_FILENAME,
        summarize_edge.ORIENTATION_METRICS_FILENAME,
        summarize_edge.ORIENTATION_PROFILES_FILENAME,
    ):
        decoded = cv2.imdecode(
            np.frombuffer(artifacts[filename], dtype=np.uint8),
            cv2.IMREAD_COLOR,
        )
        assert decoded is not None
        assert decoded.size > 0


def test_save_is_non_overwriting(
    tmp_path: Path,
) -> None:
    comparison = _comparison(tmp_path / "inputs")
    output_dir = tmp_path / "summary"

    saved = summarize_edge.save_edge_summary(
        output_dir,
        comparison,
    )

    assert saved == output_dir
    assert len(tuple(output_dir.iterdir())) == 6
    with pytest.raises(FileExistsError, match="already exists"):
        summarize_edge.save_edge_summary(
            output_dir,
            comparison,
        )


def test_save_rolls_back_files_created_by_failed_write(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    comparison = _comparison(tmp_path / "inputs")
    output_dir = tmp_path / "summary"
    original_open = Path.open

    def failing_open(
        path: Path,
        mode: str = "r",
        *args: object,
        **kwargs: object,
    ):
        if (
            path.name
            == summarize_edge.VERTICAL_PROFILES_FILENAME
            and mode == "xb"
        ):
            raise OSError("simulated failure")
        return original_open(path, mode, *args, **kwargs)

    monkeypatch.setattr(Path, "open", failing_open)

    with pytest.raises(OSError, match="simulated failure"):
        summarize_edge.save_edge_summary(
            output_dir,
            comparison,
        )
    assert output_dir.is_dir()
    assert not tuple(output_dir.iterdir())


def test_main_writes_explicit_output_and_reports_scope(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    result_dirs = _write_matrix(tmp_path / "inputs")
    output_dir = tmp_path / "summary"

    status = summarize_edge.main(
        [
            *(str(path) for path in result_dirs),
            "--output-dir",
            str(output_dir),
        ]
    )

    assert status == 0
    assert (
        output_dir / summarize_edge.OUTPUT_CSV_FILENAME
    ).is_file()
    output = capsys.readouterr().out
    assert "Edge summary complete." in output
    assert "descriptive comparison only" in output
