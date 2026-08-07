"""Tests for atomic frame-pairing CSV and YAML persistence."""

from __future__ import annotations

import csv
from pathlib import Path
from typing import Any

import pytest
import yaml

import src.io.frame_pairing_artifacts as artifacts_module
from src.io.frame_pairing_artifacts import (
    CSV_HEADER,
    PAIRING_CSV_FILENAME,
    PAIRING_DIRECTORY,
    SUMMARY_FILENAME,
    StoredFramePairing,
    load_frame_pairing_artifacts,
    save_frame_pairing_artifacts,
)
from src.preprocessing.frame_pairing import (
    FramePair,
    FramePairingConfig,
    FramePairingResult,
)


EXPERIMENT = "scene05_alignment_d100_center_yaw00_r01"


def make_result() -> FramePairingResult:
    return FramePairingResult(
        config=FramePairingConfig(max_abs_delta_ms=5.0),
        rgb_frame_count=5,
        depth_frame_count=6,
        pairs=(
            FramePair(0, 0, 1_000_000_000, 1_001_000_000),
            FramePair(2, 1, 1_100_000_000, 1_099_500_000),
            FramePair(3, 3, 1_300_000_000, 1_302_000_000),
        ),
    )


def make_dataset_directory(tmp_path: Path) -> Path:
    dataset = tmp_path / EXPERIMENT
    dataset.mkdir()
    (dataset / "timestamps.npz").write_bytes(b"source timestamps marker")
    (dataset / "experiment.yaml").write_text("source metadata\n", encoding="utf-8")
    return dataset


def read_yaml(path: Path) -> dict[str, Any]:
    document = yaml.safe_load(path.read_text(encoding="utf-8"))
    assert isinstance(document, dict)
    return document


def write_yaml(path: Path, document: dict[str, Any]) -> None:
    path.write_text(yaml.safe_dump(document, sort_keys=False), encoding="utf-8")


def read_csv_rows(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as stream:
        return list(csv.DictReader(stream))


def write_csv_rows(path: Path, rows: list[dict[str, str]]) -> None:
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=CSV_HEADER, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def save_valid_artifacts(tmp_path: Path) -> tuple[Path, FramePairingResult]:
    dataset = make_dataset_directory(tmp_path)
    result = make_result()
    output = save_frame_pairing_artifacts(
        dataset,
        experiment=EXPERIMENT,
        result=result,
    )
    return output, result


def test_save_load_round_trip_preserves_result_and_source_files(
    tmp_path: Path,
) -> None:
    dataset = make_dataset_directory(tmp_path)
    source_timestamps = (dataset / "timestamps.npz").read_bytes()
    source_metadata = (dataset / "experiment.yaml").read_text(encoding="utf-8")
    result = make_result()

    output = save_frame_pairing_artifacts(
        dataset,
        experiment=EXPERIMENT,
        result=result,
    )
    loaded = load_frame_pairing_artifacts(output)

    assert output == dataset / PAIRING_DIRECTORY
    assert loaded == StoredFramePairing(EXPERIMENT, dataset.resolve(), result)
    assert (dataset / "timestamps.npz").read_bytes() == source_timestamps
    assert (dataset / "experiment.yaml").read_text(encoding="utf-8") == source_metadata
    assert sorted(path.name for path in output.iterdir()) == [
        PAIRING_CSV_FILENAME,
        SUMMARY_FILENAME,
    ]


def test_csv_uses_canonical_schema_and_six_decimal_delta_ms(tmp_path: Path) -> None:
    output, _ = save_valid_artifacts(tmp_path)

    with (output / PAIRING_CSV_FILENAME).open(
        "r", encoding="utf-8", newline=""
    ) as stream:
        reader = csv.DictReader(stream)
        rows = list(reader)

    assert reader.fieldnames == list(CSV_HEADER)
    assert rows == [
        {
            "pair_index": "0",
            "rgb_index": "0",
            "depth_index": "0",
            "rgb_timestamp_ns": "1000000000",
            "depth_timestamp_ns": "1001000000",
            "delta_ns": "1000000",
            "delta_ms": "1.000000",
        },
        {
            "pair_index": "1",
            "rgb_index": "2",
            "depth_index": "1",
            "rgb_timestamp_ns": "1100000000",
            "depth_timestamp_ns": "1099500000",
            "delta_ns": "-500000",
            "delta_ms": "-0.500000",
        },
        {
            "pair_index": "2",
            "rgb_index": "3",
            "depth_index": "3",
            "rgb_timestamp_ns": "1300000000",
            "depth_timestamp_ns": "1302000000",
            "delta_ns": "2000000",
            "delta_ms": "2.000000",
        },
    ]


def test_summary_records_config_counts_and_recomputed_statistics(
    tmp_path: Path,
) -> None:
    output, _ = save_valid_artifacts(tmp_path)

    summary = read_yaml(output / SUMMARY_FILENAME)

    assert summary["frame_pairing"] == {
        "method": "nearest_timestamp",
        "delta_definition": "depth_minus_rgb",
        "max_abs_delta_ms": 5.0,
        "max_abs_delta_ns": 5_000_000,
        "threshold_inclusive": True,
        "cardinality": "one_to_one",
        "preserve_order": True,
        "tie_breaker": "earlier_depth",
    }
    assert summary["counts"] == {
        "rgb_frames": 5,
        "depth_frames": 6,
        "accepted_pairs": 3,
        "rejected_rgb_frames": 2,
        "unmatched_depth_frames": 3,
    }
    assert summary["delta_ms"] == {
        "minimum_signed": -0.5,
        "median_signed": 1.0,
        "maximum_signed": 2.0,
        "median_absolute": 1.0,
        "p95_absolute": 1.9,
        "maximum_absolute": 2.0,
        "percentile_method": "linear",
    }


def test_zero_accepted_pairs_write_null_statistics(tmp_path: Path) -> None:
    dataset = make_dataset_directory(tmp_path)
    result = FramePairingResult(FramePairingConfig(), 2, 3, ())

    output = save_frame_pairing_artifacts(
        dataset,
        experiment=EXPERIMENT,
        result=result,
    )
    summary = read_yaml(output / SUMMARY_FILENAME)

    assert read_csv_rows(output / PAIRING_CSV_FILENAME) == []
    assert summary["delta_ms"] == {
        "minimum_signed": None,
        "median_signed": None,
        "maximum_signed": None,
        "median_absolute": None,
        "p95_absolute": None,
        "maximum_absolute": None,
        "percentile_method": "linear",
    }
    assert load_frame_pairing_artifacts(output).result == result


def test_save_rejects_existing_pairing_directory_without_modifying_it(
    tmp_path: Path,
) -> None:
    dataset = make_dataset_directory(tmp_path)
    output = dataset / PAIRING_DIRECTORY
    output.mkdir()
    marker = output / "keep.txt"
    marker.write_text("keep", encoding="utf-8")

    with pytest.raises(FileExistsError, match="already exists"):
        save_frame_pairing_artifacts(
            dataset,
            experiment=EXPERIMENT,
            result=make_result(),
        )

    assert marker.read_text(encoding="utf-8") == "keep"


def test_save_cleans_staging_directory_after_failure(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    dataset = make_dataset_directory(tmp_path)

    def fail_summary(path: Path, stored: StoredFramePairing) -> None:
        raise RuntimeError(f"forced failure for {path} and {stored.experiment}")

    monkeypatch.setattr(artifacts_module, "_write_summary", fail_summary)

    with pytest.raises(RuntimeError, match="forced failure"):
        save_frame_pairing_artifacts(
            dataset,
            experiment=EXPERIMENT,
            result=make_result(),
        )

    assert sorted(path.name for path in dataset.iterdir()) == [
        "experiment.yaml",
        "timestamps.npz",
    ]


@pytest.mark.parametrize(
    ("dataset_setup", "exception", "expected_message"),
    [
        ("missing", FileNotFoundError, "does not exist"),
        ("file", NotADirectoryError, "not a directory"),
    ],
)
def test_save_rejects_invalid_dataset_path(
    tmp_path: Path,
    dataset_setup: str,
    exception: type[Exception],
    expected_message: str,
) -> None:
    dataset = tmp_path / EXPERIMENT
    if dataset_setup == "file":
        dataset.write_text("not a directory", encoding="utf-8")

    with pytest.raises(exception, match=expected_message):
        save_frame_pairing_artifacts(
            dataset,
            experiment=EXPERIMENT,
            result=make_result(),
        )


def test_save_rejects_experiment_directory_name_mismatch(tmp_path: Path) -> None:
    dataset = make_dataset_directory(tmp_path)

    with pytest.raises(ValueError, match="does not match"):
        save_frame_pairing_artifacts(
            dataset,
            experiment="different_experiment",
            result=make_result(),
        )


def test_save_rejects_non_result(tmp_path: Path) -> None:
    dataset = make_dataset_directory(tmp_path)

    with pytest.raises(TypeError, match="FramePairingResult"):
        save_frame_pairing_artifacts(
            dataset,
            experiment=EXPERIMENT,
            result=None,
        )


@pytest.mark.parametrize(
    ("field", "replacement", "expected_message"),
    [
        ("pair_index", "2", "consecutive from zero"),
        ("rgb_index", "01", "canonical integer"),
        ("delta_ns", "999", "depth minus RGB"),
        ("delta_ms", "1.0", "does not match delta_ns"),
    ],
)
def test_load_rejects_corrupt_csv_rows(
    tmp_path: Path,
    field: str,
    replacement: str,
    expected_message: str,
) -> None:
    output, _ = save_valid_artifacts(tmp_path)
    csv_path = output / PAIRING_CSV_FILENAME
    rows = read_csv_rows(csv_path)
    rows[0][field] = replacement
    write_csv_rows(csv_path, rows)

    with pytest.raises(ValueError, match=expected_message):
        load_frame_pairing_artifacts(output)


def test_load_rejects_wrong_csv_header(tmp_path: Path) -> None:
    output, _ = save_valid_artifacts(tmp_path)
    csv_path = output / PAIRING_CSV_FILENAME
    csv_path.write_text("rgb_index,depth_index\n0,0\n", encoding="utf-8")

    with pytest.raises(ValueError, match="CSV header"):
        load_frame_pairing_artifacts(output)


@pytest.mark.parametrize(
    ("path", "replacement", "expected_message"),
    [
        (("schema_version",), 2, "schema_version"),
        (("frame_pairing", "method"), "other", "method"),
        (("frame_pairing", "max_abs_delta_ns"), 1, "max_abs_delta_ns"),
        (("counts", "accepted_pairs"), 2, "recomputed CSV result"),
        (("delta_ms", "p95_absolute"), 999.0, "recomputed CSV result"),
        (("source", "timestamp_source"), "storage", "timestamp_source"),
        (("source", "dataset_directory"), "relative/path", "absolute path"),
    ],
)
def test_load_rejects_corrupt_summary(
    tmp_path: Path,
    path: tuple[str, ...],
    replacement: Any,
    expected_message: str,
) -> None:
    output, _ = save_valid_artifacts(tmp_path)
    summary_path = output / SUMMARY_FILENAME
    document = read_yaml(summary_path)
    target: dict[str, Any] = document
    for key in path[:-1]:
        target = target[key]
    target[path[-1]] = replacement
    write_yaml(summary_path, document)

    with pytest.raises(ValueError, match=expected_message):
        load_frame_pairing_artifacts(output)


@pytest.mark.parametrize(
    ("missing_filename", "expected_message"),
    [
        (PAIRING_CSV_FILENAME, "CSV does not exist"),
        (SUMMARY_FILENAME, "summary does not exist"),
    ],
)
def test_load_rejects_missing_artifact(
    tmp_path: Path,
    missing_filename: str,
    expected_message: str,
) -> None:
    output, _ = save_valid_artifacts(tmp_path)
    (output / missing_filename).unlink()

    with pytest.raises(FileNotFoundError, match=expected_message):
        load_frame_pairing_artifacts(output)


def test_load_rejects_source_experiment_mismatch(tmp_path: Path) -> None:
    output, _ = save_valid_artifacts(tmp_path)
    summary_path = output / SUMMARY_FILENAME
    document = read_yaml(summary_path)
    document["experiment"] = "different_experiment"
    write_yaml(summary_path, document)

    with pytest.raises(ValueError, match="does not match"):
        load_frame_pairing_artifacts(output)


@pytest.mark.parametrize(
    ("path_type", "exception", "expected_message"),
    [
        ("missing", FileNotFoundError, "does not exist"),
        ("file", NotADirectoryError, "not a directory"),
    ],
)
def test_load_rejects_invalid_artifact_directory(
    tmp_path: Path,
    path_type: str,
    exception: type[Exception],
    expected_message: str,
) -> None:
    path = tmp_path / "pairing"
    if path_type == "file":
        path.write_text("not a directory", encoding="utf-8")

    with pytest.raises(exception, match=expected_message):
        load_frame_pairing_artifacts(path)
