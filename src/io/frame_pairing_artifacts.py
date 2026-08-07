"""Atomic CSV and YAML persistence for frame-pairing results."""

from __future__ import annotations

import csv
import re
import shutil
import tempfile
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import yaml

from src.io.alignment_dataset import AlignmentDataset
from src.preprocessing.frame_pairing import (
    FramePair,
    FramePairingConfig,
    FramePairingResult,
)


PAIRING_DIRECTORY = "pairing"
PAIRING_CSV_FILENAME = "frame_pairing.csv"
SUMMARY_FILENAME = "summary.yaml"
ARTIFACT_SCHEMA_VERSION = 1
PERCENTILE_METHOD = "linear"
CSV_HEADER = (
    "pair_index",
    "rgb_index",
    "depth_index",
    "rgb_timestamp_ns",
    "depth_timestamp_ns",
    "delta_ns",
    "delta_ms",
)


@dataclass(frozen=True)
class StoredFramePairing:
    """Pair a validated result with its source-dataset provenance."""

    experiment: str
    source_dataset_directory: Path
    result: FramePairingResult

    def __post_init__(self) -> None:
        """Normalize provenance and reject inconsistent experiment names."""
        if not isinstance(self.experiment, str) or not self.experiment:
            raise ValueError("experiment must be a non-empty string")
        source = Path(self.source_dataset_directory).expanduser()
        if not source.is_absolute():
            raise ValueError("source_dataset_directory must be an absolute path")
        if source.name != self.experiment:
            raise ValueError(
                f"Experiment {self.experiment!r} does not match source dataset "
                f"directory {source.name!r}"
            )
        if not isinstance(self.result, FramePairingResult):
            raise TypeError(
                "result must be a FramePairingResult; got "
                f"{type(self.result).__name__}"
            )
        object.__setattr__(self, "source_dataset_directory", source)


def save_frame_pairing_artifacts(
    dataset_directory: Path,
    *,
    experiment: str,
    result: FramePairingResult,
) -> Path:
    """Atomically save one non-overwriting pairing artifact directory."""
    dataset_path = Path(dataset_directory).expanduser()
    if not dataset_path.exists():
        raise FileNotFoundError(f"Dataset directory does not exist: {dataset_path}")
    if not dataset_path.is_dir():
        raise NotADirectoryError(f"Dataset path is not a directory: {dataset_path}")

    stored = StoredFramePairing(
        experiment=experiment,
        source_dataset_directory=dataset_path.resolve(),
        result=result,
    )
    output_path = dataset_path / PAIRING_DIRECTORY
    if output_path.exists():
        raise FileExistsError(f"Frame-pairing output already exists: {output_path}")

    staging_path = Path(
        tempfile.mkdtemp(
            prefix=f".{PAIRING_DIRECTORY}.staging-",
            dir=dataset_path,
        )
    )
    try:
        _write_pairing_csv(staging_path / PAIRING_CSV_FILENAME, result)
        _write_summary(staging_path / SUMMARY_FILENAME, stored)

        reloaded = load_frame_pairing_artifacts(staging_path)
        if reloaded != stored:
            raise ValueError("Reloaded frame-pairing artifacts do not match input")
        if output_path.exists():
            raise FileExistsError(
                f"Frame-pairing output appeared during save: {output_path}"
            )
        staging_path.replace(output_path)
        return output_path
    except BaseException:
        shutil.rmtree(staging_path, ignore_errors=True)
        raise


def load_frame_pairing_artifacts(pairing_directory: Path) -> StoredFramePairing:
    """Load CSV and YAML artifacts and verify complete cross-file agreement."""
    input_path = Path(pairing_directory).expanduser()
    if not input_path.exists():
        raise FileNotFoundError(
            f"Frame-pairing artifact directory does not exist: {input_path}"
        )
    if not input_path.is_dir():
        raise NotADirectoryError(
            f"Frame-pairing artifact path is not a directory: {input_path}"
        )

    document = _load_summary(input_path / SUMMARY_FILENAME)
    _validate_fixed_summary_contract(document)
    experiment = _require_non_empty_string(document, "experiment")
    source = _require_mapping(document, "source")
    source_directory = Path(
        _require_non_empty_string(source, "dataset_directory")
    ).expanduser()

    config_document = _require_mapping(document, "frame_pairing")
    config = FramePairingConfig(
        max_abs_delta_ms=_require_real(
            config_document,
            "max_abs_delta_ms",
        )
    )
    _require_exact(
        config_document,
        "max_abs_delta_ns",
        config.max_abs_delta_ns,
    )

    counts = _require_mapping(document, "counts")
    rgb_frame_count = _require_integer(counts, "rgb_frames")
    depth_frame_count = _require_integer(counts, "depth_frames")
    pairs = _load_pairing_csv(input_path / PAIRING_CSV_FILENAME)
    result = FramePairingResult(
        config=config,
        rgb_frame_count=rgb_frame_count,
        depth_frame_count=depth_frame_count,
        pairs=pairs,
    )
    stored = StoredFramePairing(
        experiment=experiment,
        source_dataset_directory=source_directory,
        result=result,
    )

    expected_document = _summary_document(stored)
    if document != expected_document:
        raise ValueError(
            "Frame-pairing summary does not match the recomputed CSV result"
        )
    return stored


def _write_pairing_csv(path: Path, result: FramePairingResult) -> None:
    with path.open("x", encoding="utf-8", newline="") as stream:
        writer = csv.writer(stream, lineterminator="\n")
        writer.writerow(CSV_HEADER)
        for pair_index, pair in enumerate(result.pairs):
            writer.writerow(
                (
                    pair_index,
                    pair.rgb_index,
                    pair.depth_index,
                    pair.rgb_timestamp_ns,
                    pair.depth_timestamp_ns,
                    pair.delta_ns,
                    _format_delta_ms(pair.delta_ns),
                )
            )


def _load_pairing_csv(path: Path) -> tuple[FramePair, ...]:
    if not path.is_file():
        raise FileNotFoundError(f"Frame-pairing CSV does not exist: {path}")
    with path.open("r", encoding="utf-8", newline="") as stream:
        reader = csv.DictReader(stream)
        if reader.fieldnames != list(CSV_HEADER):
            raise ValueError(
                f"Frame-pairing CSV header must equal {list(CSV_HEADER)!r}; "
                f"got {reader.fieldnames!r}"
            )

        pairs: list[FramePair] = []
        for expected_pair_index, row in enumerate(reader):
            pair_index = _parse_canonical_integer(
                row["pair_index"],
                "pair_index",
            )
            if pair_index != expected_pair_index:
                raise ValueError(
                    "Frame-pairing CSV pair_index must be consecutive from zero"
                )
            pair = FramePair(
                rgb_index=_parse_canonical_integer(row["rgb_index"], "rgb_index"),
                depth_index=_parse_canonical_integer(
                    row["depth_index"],
                    "depth_index",
                ),
                rgb_timestamp_ns=_parse_canonical_integer(
                    row["rgb_timestamp_ns"],
                    "rgb_timestamp_ns",
                ),
                depth_timestamp_ns=_parse_canonical_integer(
                    row["depth_timestamp_ns"],
                    "depth_timestamp_ns",
                ),
            )
            delta_ns = _parse_canonical_integer(row["delta_ns"], "delta_ns")
            if delta_ns != pair.delta_ns:
                raise ValueError(
                    "Frame-pairing CSV delta_ns does not equal depth minus RGB"
                )
            expected_delta_ms = _format_delta_ms(pair.delta_ns)
            if row["delta_ms"] != expected_delta_ms:
                raise ValueError(
                    "Frame-pairing CSV delta_ms does not match delta_ns; "
                    f"expected {expected_delta_ms!r}"
                )
            pairs.append(pair)
    return tuple(pairs)


def _summary_document(stored: StoredFramePairing) -> dict[str, Any]:
    result = stored.result
    config = result.config
    return {
        "schema_version": ARTIFACT_SCHEMA_VERSION,
        "status": "success",
        "experiment": stored.experiment,
        "source": {
            "dataset_directory": str(stored.source_dataset_directory),
            "timestamps_artifact": AlignmentDataset.TIMESTAMPS_FILENAME,
            "dataset_schema_version": AlignmentDataset.SCHEMA_VERSION,
            "timestamp_source": AlignmentDataset.PRIMARY_TIMESTAMP_SOURCE,
            "timestamp_unit": AlignmentDataset.TIMESTAMP_UNIT,
        },
        "frame_pairing": {
            "method": config.METHOD,
            "delta_definition": config.DELTA_DEFINITION,
            "max_abs_delta_ms": config.max_abs_delta_ms,
            "max_abs_delta_ns": config.max_abs_delta_ns,
            "threshold_inclusive": config.THRESHOLD_INCLUSIVE,
            "cardinality": config.CARDINALITY,
            "preserve_order": config.PRESERVE_ORDER,
            "tie_breaker": config.TIE_BREAKER,
        },
        "counts": {
            "rgb_frames": result.rgb_frame_count,
            "depth_frames": result.depth_frame_count,
            "accepted_pairs": result.accepted_pair_count,
            "rejected_rgb_frames": result.rejected_rgb_count,
            "unmatched_depth_frames": result.unmatched_depth_count,
        },
        "delta_ms": _delta_statistics(result),
        "artifacts": {"frame_pairing_csv": PAIRING_CSV_FILENAME},
    }


def _delta_statistics(result: FramePairingResult) -> dict[str, Any]:
    if not result.pairs:
        return {
            "minimum_signed": None,
            "median_signed": None,
            "maximum_signed": None,
            "median_absolute": None,
            "p95_absolute": None,
            "maximum_absolute": None,
            "percentile_method": PERCENTILE_METHOD,
        }

    signed = np.asarray([pair.delta_ms for pair in result.pairs], dtype=np.float64)
    absolute = np.abs(signed)
    return {
        "minimum_signed": float(np.min(signed)),
        "median_signed": float(np.median(signed)),
        "maximum_signed": float(np.max(signed)),
        "median_absolute": float(np.median(absolute)),
        "p95_absolute": float(np.percentile(absolute, 95, method=PERCENTILE_METHOD)),
        "maximum_absolute": float(np.max(absolute)),
        "percentile_method": PERCENTILE_METHOD,
    }


def _write_summary(path: Path, stored: StoredFramePairing) -> None:
    with path.open("x", encoding="utf-8") as stream:
        yaml.safe_dump(_summary_document(stored), stream, sort_keys=False)


def _load_summary(path: Path) -> Mapping[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(f"Frame-pairing summary does not exist: {path}")
    try:
        with path.open("r", encoding="utf-8") as stream:
            document = yaml.safe_load(stream)
    except yaml.YAMLError as exc:
        raise ValueError(f"Invalid frame-pairing summary YAML: {path}") from exc
    if not isinstance(document, Mapping):
        raise ValueError("Frame-pairing summary must be a mapping")
    return document


def _validate_fixed_summary_contract(document: Mapping[str, Any]) -> None:
    _require_exact(document, "schema_version", ARTIFACT_SCHEMA_VERSION)
    _require_exact(document, "status", "success")

    source = _require_mapping(document, "source")
    _require_exact(
        source,
        "timestamps_artifact",
        AlignmentDataset.TIMESTAMPS_FILENAME,
    )
    _require_exact(
        source,
        "dataset_schema_version",
        AlignmentDataset.SCHEMA_VERSION,
    )
    _require_exact(
        source,
        "timestamp_source",
        AlignmentDataset.PRIMARY_TIMESTAMP_SOURCE,
    )
    _require_exact(source, "timestamp_unit", AlignmentDataset.TIMESTAMP_UNIT)

    config = _require_mapping(document, "frame_pairing")
    fixed_config = {
        "method": FramePairingConfig.METHOD,
        "delta_definition": FramePairingConfig.DELTA_DEFINITION,
        "threshold_inclusive": FramePairingConfig.THRESHOLD_INCLUSIVE,
        "cardinality": FramePairingConfig.CARDINALITY,
        "preserve_order": FramePairingConfig.PRESERVE_ORDER,
        "tie_breaker": FramePairingConfig.TIE_BREAKER,
    }
    for key, expected in fixed_config.items():
        _require_exact(config, key, expected)

    artifacts = _require_mapping(document, "artifacts")
    _require_exact(artifacts, "frame_pairing_csv", PAIRING_CSV_FILENAME)


def _format_delta_ms(delta_ns: int) -> str:
    sign = "-" if delta_ns < 0 else ""
    milliseconds, nanosecond_remainder = divmod(abs(delta_ns), 1_000_000)
    return f"{sign}{milliseconds}.{nanosecond_remainder:06d}"


def _parse_canonical_integer(value: Any, field_name: str) -> int:
    if not isinstance(value, str) or re.fullmatch(r"-?(0|[1-9]\d*)", value) is None:
        raise ValueError(f"CSV field {field_name} must be a canonical integer")
    return int(value)


def _require_mapping(
    document: Mapping[str, Any],
    key: str,
) -> Mapping[str, Any]:
    value = document.get(key)
    if not isinstance(value, Mapping):
        raise ValueError(f"Summary field {key} must be a mapping")
    return value


def _require_non_empty_string(document: Mapping[str, Any], key: str) -> str:
    value = document.get(key)
    if not isinstance(value, str) or not value:
        raise ValueError(f"Summary field {key} must be a non-empty string")
    return value


def _require_integer(document: Mapping[str, Any], key: str) -> int:
    value = document.get(key)
    if not isinstance(value, int) or isinstance(value, bool):
        raise ValueError(f"Summary field {key} must be an integer")
    return value


def _require_real(document: Mapping[str, Any], key: str) -> float:
    value = document.get(key)
    if not isinstance(value, (int, float)) or isinstance(value, bool):
        raise ValueError(f"Summary field {key} must be a real number")
    return float(value)


def _require_exact(document: Mapping[str, Any], key: str, expected: Any) -> None:
    if key not in document:
        raise ValueError(f"Summary is missing required field {key}")
    actual = document[key]
    if actual != expected or type(actual) is not type(expected):
        raise ValueError(f"Summary field {key} must equal {expected!r}; got {actual!r}")
