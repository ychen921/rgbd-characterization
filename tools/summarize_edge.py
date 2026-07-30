"""Summarize and compare completed Scene 04 edge analyses."""

import argparse
from collections.abc import Mapping, Sequence
import csv
from dataclasses import dataclass
from io import BytesIO, StringIO
from pathlib import Path
import sys

import numpy as np
import yaml


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "results" / "edge_summary"

SUMMARY_FILENAME = "summary.yaml"
FRAME_METRICS_FILENAME = "frame_edge_metrics.csv"
PROFILE_FILENAME = "aggregate_edge_profile.csv"

OUTPUT_CSV_FILENAME = "edge_summary.csv"
OUTPUT_YAML_FILENAME = "comparison_summary.yaml"
VERTICAL_METRICS_FILENAME = "vertical_distance_metrics.png"
VERTICAL_PROFILES_FILENAME = "vertical_distance_profiles.png"
ORIENTATION_METRICS_FILENAME = "d100_orientation_metrics.png"
ORIENTATION_PROFILES_FILENAME = "d100_orientation_profiles.png"

REQUIRED_FRAME_COLUMNS = frozenset(
    {
        "frame_index",
        "timestamp_ns",
        "foreground_reference_mm",
        "background_reference_mm",
        "measured_gap_mm",
        "foreground_bleeding_ratio",
        "background_bleeding_ratio",
        "mixed_ratio",
        "outlier_ratio",
        "invalid_ratio",
        "transition_width_px",
        "nominal_edge_offset_px",
        "analysis_status",
        "transition_status",
    }
)
PROFILE_COLUMNS = (
    "distance_min_px",
    "distance_max_px",
    "distance_center_px",
    "pixel_count",
    "valid_count",
    "foreground_count",
    "background_count",
    "mixed_count",
    "outlier_count",
    "invalid_count",
    "foreground_ratio",
    "background_ratio",
    "mixed_ratio",
    "outlier_ratio",
    "invalid_ratio",
)
COUNT_PROFILE_COLUMNS = frozenset(
    column
    for column in PROFILE_COLUMNS
    if column.endswith("_count")
)

IDENTIFIER_COLUMNS = (
    "experiment",
    "orientation_token",
    "orientation",
    "target",
    "repeat_index",
    "source_summary",
    "num_frames",
)
SETUP_COLUMNS = (
    "nominal_foreground_distance_mm",
    "nominal_background_distance_mm",
    "nominal_gap_mm",
    "distance_reference",
)
ROI_COLUMNS = (
    "foreground_roi_pixels",
    "background_roi_pixels",
    "edge_roi_pixels",
    "distance_bin_px",
    "max_edge_distance_px",
)
REFERENCE_COLUMNS = tuple(
    f"{prefix}_{stat}_mm"
    for prefix in (
        "foreground_reference",
        "background_reference",
        "measured_gap",
    )
    for stat in ("p05", "median", "p95")
) + ("gap_error_mm",)
RATIO_COLUMNS = tuple(
    f"{prefix}_{stat}"
    for prefix in (
        "foreground_bleeding_ratio",
        "background_bleeding_ratio",
        "mixed_ratio",
        "outlier_ratio",
        "invalid_ratio",
    )
    for stat in ("p05", "median", "p95")
)
TRANSITION_COLUMNS = (
    "transition_width_p05_px",
    "transition_width_median_px",
    "transition_width_p95_px",
    "nominal_offset_p05_px",
    "nominal_offset_median_px",
    "nominal_offset_p95_px",
    "nominal_offset_std_px",
)
FRAME_COLUMNS = (
    "valid_frames",
    "rejected_frames",
    "valid_frame_ratio",
    "transition_valid_frames",
    "transition_failed_frames",
    "transition_success_ratio",
)
OUTPUT_COLUMNS = (
    IDENTIFIER_COLUMNS
    + SETUP_COLUMNS
    + ROI_COLUMNS
    + REFERENCE_COLUMNS
    + RATIO_COLUMNS
    + TRANSITION_COLUMNS
    + FRAME_COLUMNS
)

EXPECTED_CONDITIONS = frozenset(
    {
        ("horizontal", 1000, 1),
        ("vertical", 500, 1),
        ("vertical", 1000, 1),
        ("vertical", 2000, 1),
    }
)
EXPECTED_EXPERIMENTS = frozenset(
    {
        "scene04_gap030_horizon_white_d100_r01",
        "scene04_gap030_vertical_white_d050_r01",
        "scene04_gap030_vertical_white_d100_r01",
        "scene04_gap030_vertical_white_d200_r01",
    }
)

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


@dataclass(frozen=True)
class EdgeSummaryRecord:
    """Store one validated dataset summary and its detailed tables."""

    result_dir: Path
    summary_path: Path
    frame_metrics_path: Path
    profile_path: Path
    summary: Mapping[str, object]
    frame_metrics: tuple[Mapping[str, str], ...]
    profile: Mapping[str, np.ndarray]


@dataclass(frozen=True)
class EdgeComparisonResult:
    """Store the four validated records in deterministic order."""

    records: tuple[EdgeSummaryRecord, ...]


def parse_args(
    argv: Sequence[str] | None = None,
) -> argparse.Namespace:
    """Parse command-line arguments for Scene 04 summarization."""
    parser = argparse.ArgumentParser(
        description=(
            "Summarize four completed Scene 04 edge analyses."
        )
    )
    parser.add_argument(
        "result_dirs",
        type=Path,
        nargs="+",
        help=(
            "Completed edge_discontinuity result directories."
        ),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help=(
            "Summary artifact directory "
            "(default: results/edge_summary)."
        ),
    )
    return parser.parse_args(argv)


def load_edge_summary_record(
    result_dir: Path,
) -> EdgeSummaryRecord:
    """Load and validate one completed edge-analysis directory."""
    resolved_dir = Path(result_dir).expanduser()
    summary_path = resolved_dir / SUMMARY_FILENAME
    frame_path = resolved_dir / FRAME_METRICS_FILENAME
    profile_path = resolved_dir / PROFILE_FILENAME
    for path in (summary_path, frame_path, profile_path):
        if not path.is_file():
            raise FileNotFoundError(
                f"Required edge summary input not found: {path}"
            )

    with summary_path.open("r", encoding="utf-8") as stream:
        loaded_summary = yaml.safe_load(stream)
    summary = _require_mapping(
        loaded_summary,
        f"summary {summary_path}",
    )

    with frame_path.open(
        "r",
        encoding="utf-8",
        newline="",
    ) as stream:
        reader = csv.DictReader(stream)
        if reader.fieldnames is None:
            raise ValueError(
                f"Frame metrics CSV has no header: {frame_path}"
            )
        missing = REQUIRED_FRAME_COLUMNS.difference(
            reader.fieldnames
        )
        if missing:
            raise ValueError(
                "Frame metrics CSV is missing columns: "
                + ", ".join(sorted(missing))
            )
        frame_metrics = tuple(dict(row) for row in reader)

    profile = _load_profile_csv(profile_path)
    record = EdgeSummaryRecord(
        result_dir=resolved_dir,
        summary_path=summary_path,
        frame_metrics_path=frame_path,
        profile_path=profile_path,
        summary=summary,
        frame_metrics=frame_metrics,
        profile=profile,
    )
    _validate_record(record)
    _validate_profile_counts(record)
    return record


def validate_edge_comparison(
    records: Sequence[EdgeSummaryRecord],
) -> EdgeComparisonResult:
    """Validate the controlled four-condition comparison matrix."""
    normalized = tuple(records)
    if len(normalized) != 4:
        raise ValueError(
            "Scene 04 comparison requires exactly four result directories"
        )
    if any(
        not isinstance(record, EdgeSummaryRecord)
        for record in normalized
    ):
        raise TypeError(
            "records must contain EdgeSummaryRecord values"
        )

    conditions = [
        (
            _summary_text(record, "setup", "orientation"),
            _summary_int(
                record,
                "setup",
                "nominal_foreground_distance_mm",
            ),
            _summary_int(record, "setup", "repeat_index"),
        )
        for record in normalized
    ]
    if len(set(conditions)) != len(conditions):
        raise ValueError(
            "Scene 04 comparison contains a duplicate condition"
        )
    if frozenset(conditions) != EXPECTED_CONDITIONS:
        raise ValueError(
            "Scene 04 comparison must contain horizontal d100 and "
            "vertical d050/d100/d200, all at repeat 1"
        )
    if {
        _experiment(record)
        for record in normalized
    } != EXPECTED_EXPERIMENTS:
        raise ValueError(
            "Scene 04 comparison must use the approved gap030 white "
            "r01 experiment names"
        )

    signatures = [
        _comparison_signature(record)
        for record in normalized
    ]
    if any(signature != signatures[0] for signature in signatures[1:]):
        raise ValueError(
            "Scene 04 result summaries use inconsistent common settings"
        )

    first_centers = normalized[0].profile[
        "distance_center_px"
    ]
    for record in normalized:
        centers = record.profile["distance_center_px"]
        if not np.array_equal(centers, first_centers):
            raise ValueError(
                "Aggregate edge profiles use inconsistent distance bins"
            )
        if np.any(record.profile["pixel_count"] < 0):
            raise ValueError(
                f"Aggregate profile contains negative pixel counts: "
                f"{_experiment(record)}"
            )

    expected_max = _summary_float(
        normalized[0],
        "edge_geometry",
        "max_edge_distance_px",
    )
    expected_bin = _summary_float(
        normalized[0],
        "edge_geometry",
        "distance_bin_px",
    )
    bins_per_side = int(round(expected_max / expected_bin))
    expected_centers = (
        np.arange(
            -bins_per_side,
            bins_per_side + 1,
            dtype=np.float64,
        )
        * expected_bin
    )
    if not np.array_equal(first_centers, expected_centers):
        raise ValueError(
            "Aggregate profile does not cover the configured "
            "signed-distance range"
        )

    ordered = tuple(
        sorted(
            normalized,
            key=lambda record: (
                0
                if _summary_text(
                    record,
                    "setup",
                    "orientation",
                )
                == "horizontal"
                else 1,
                _summary_int(
                    record,
                    "setup",
                    "nominal_foreground_distance_mm",
                ),
            ),
        )
    )
    return EdgeComparisonResult(records=ordered)


def build_edge_summary_csv(
    comparison: EdgeComparisonResult,
) -> str:
    """Build one row per recording as deterministic CSV text."""
    _validate_comparison_result(comparison)
    stream = StringIO(newline="")
    writer = csv.DictWriter(
        stream,
        fieldnames=OUTPUT_COLUMNS,
    )
    writer.writeheader()
    for record in comparison.records:
        writer.writerow(_summary_row(record))
    return stream.getvalue()


def build_comparison_summary(
    comparison: EdgeComparisonResult,
) -> dict[str, object]:
    """Build a YAML-safe comparison provenance document."""
    _validate_comparison_result(comparison)
    records = comparison.records
    horizontal = _select_record(
        records,
        orientation="horizontal",
        distance_mm=1000,
    )
    vertical = [
        record
        for record in records
        if _summary_text(record, "setup", "orientation")
        == "vertical"
    ]
    first = records[0]
    return {
        "inputs": {
            "dataset_count": len(records),
            "experiments": [
                _experiment(record)
                for record in records
            ],
            "summary_paths": [
                _summary_path(record.summary_path)
                for record in records
            ],
        },
        "common_setup": {
            "target": _summary_text(first, "setup", "target"),
            "nominal_gap_mm": _summary_int(
                first,
                "setup",
                "nominal_gap_mm",
            ),
            "distance_reference": _summary_text(
                first,
                "setup",
                "distance_reference",
            ),
        },
        "common_analysis": {
            "distance_bin_px": _summary_float(
                first,
                "edge_geometry",
                "distance_bin_px",
            ),
            "max_edge_distance_px": _summary_float(
                first,
                "edge_geometry",
                "max_edge_distance_px",
            ),
            "reference": _nested_mapping(
                first.summary,
                "analysis_parameters",
                "reference",
            ),
            "bleeding_probability_threshold": _summary_float(
                first,
                "analysis_parameters",
                "bleeding_probability_threshold",
            ),
            "invalid_ratio_threshold": _summary_float(
                first,
                "analysis_parameters",
                "invalid_ratio_threshold",
            ),
            "transition_high_probability": _summary_float(
                first,
                "analysis_parameters",
                "transition_high_probability",
            ),
            "transition_low_probability": _summary_float(
                first,
                "analysis_parameters",
                "transition_low_probability",
            ),
        },
        "comparison_groups": {
            "orientation_d100": {
                "nominal_foreground_distance_mm": 1000,
                "experiments": [
                    _experiment(horizontal),
                    _experiment(
                        _select_record(
                            records,
                            orientation="vertical",
                            distance_mm=1000,
                        )
                    ),
                ],
            },
            "vertical_distance": {
                "nominal_foreground_distances_mm": [
                    _summary_int(
                        record,
                        "setup",
                        "nominal_foreground_distance_mm",
                    )
                    for record in vertical
                ],
                "experiments": [
                    _experiment(record)
                    for record in vertical
                ],
            },
        },
        "repeat_scope": {
            "repeat_count_per_condition": 1,
            "repeatability_analysis_available": False,
        },
        "profile_coverage": {
            "zero_pixel_bins_are_missing_data": True,
            "experiments": [
                _profile_coverage(record)
                for record in records
            ],
            "warnings": [
                (
                    f"{_experiment(record)} has zero-pixel "
                    "signed-distance bins"
                )
                for record in records
                if np.any(record.profile["pixel_count"] == 0)
            ],
        },
        "interpretation_limits": [
            "descriptive comparison only",
            "temporal ranges are not confidence intervals",
            "zero-pixel profile bins are not interpolated",
            (
                "nominal offset is relative to the manual "
                "nominal-edge annotation"
            ),
        ],
    }


def build_edge_summary_artifacts(
    comparison: EdgeComparisonResult,
) -> dict[str, bytes]:
    """Serialize the CSV, provenance YAML, and four plots in memory."""
    _validate_comparison_result(comparison)
    return {
        OUTPUT_CSV_FILENAME: build_edge_summary_csv(
            comparison
        ).encode("utf-8"),
        OUTPUT_YAML_FILENAME: yaml.safe_dump(
            build_comparison_summary(comparison),
            sort_keys=False,
            allow_unicode=True,
        ).encode("utf-8"),
        VERTICAL_METRICS_FILENAME: _figure_png_bytes(
            plot_vertical_distance_metrics(comparison)
        ),
        VERTICAL_PROFILES_FILENAME: _figure_png_bytes(
            plot_vertical_distance_profiles(comparison)
        ),
        ORIENTATION_METRICS_FILENAME: _figure_png_bytes(
            plot_d100_orientation_metrics(comparison)
        ),
        ORIENTATION_PROFILES_FILENAME: _figure_png_bytes(
            plot_d100_orientation_profiles(comparison)
        ),
    }


def plot_vertical_distance_metrics(
    comparison: EdgeComparisonResult,
):
    """Plot vertical-edge metrics across foreground distance."""
    _validate_comparison_result(comparison)
    records = [
        record
        for record in comparison.records
        if _summary_text(record, "setup", "orientation")
        == "vertical"
    ]
    x = np.asarray(
        [
            _summary_int(
                record,
                "setup",
                "nominal_foreground_distance_mm",
            )
            for record in records
        ],
        dtype=np.float64,
    )
    return _plot_metric_panels(
        records,
        x=x,
        x_labels=None,
        title="Vertical edge: distance comparison",
        x_label="Nominal foreground distance (mm)",
    )


def plot_d100_orientation_metrics(
    comparison: EdgeComparisonResult,
):
    """Plot horizontal and vertical metrics at nominal d100."""
    _validate_comparison_result(comparison)
    records = [
        _select_record(
            comparison.records,
            orientation=orientation,
            distance_mm=1000,
        )
        for orientation in ("horizontal", "vertical")
    ]
    return _plot_metric_panels(
        records,
        x=np.asarray([0.0, 1.0]),
        x_labels=("Horizontal", "Vertical"),
        title="d100: orientation comparison",
        x_label="Edge orientation",
    )


def plot_vertical_distance_profiles(
    comparison: EdgeComparisonResult,
):
    """Plot signed-distance profiles for vertical d050/d100/d200."""
    _validate_comparison_result(comparison)
    records = [
        record
        for record in comparison.records
        if _summary_text(record, "setup", "orientation")
        == "vertical"
    ]
    labels = [
        f"d{_summary_int(record, 'setup', 'nominal_foreground_distance_mm')}"
        for record in records
    ]
    return _plot_profile_panels(
        records,
        labels=labels,
        title="Vertical edge: signed-distance profiles",
    )


def plot_d100_orientation_profiles(
    comparison: EdgeComparisonResult,
):
    """Plot signed-distance profiles for horizontal/vertical d100."""
    _validate_comparison_result(comparison)
    records = [
        _select_record(
            comparison.records,
            orientation=orientation,
            distance_mm=1000,
        )
        for orientation in ("horizontal", "vertical")
    ]
    return _plot_profile_panels(
        records,
        labels=("Horizontal", "Vertical"),
        title="d100: signed-distance profiles by orientation",
    )


def save_edge_summary(
    output_dir: Path,
    comparison: EdgeComparisonResult,
) -> Path:
    """Save all B6 artifacts without overwrite or partial output."""
    resolved_output = Path(output_dir).expanduser()
    artifacts = build_edge_summary_artifacts(comparison)
    paths = {
        filename: resolved_output / filename
        for filename in artifacts
    }
    existing = [
        path
        for path in paths.values()
        if path.exists()
    ]
    if existing:
        raise FileExistsError(
            "Edge summary output already exists: "
            + ", ".join(str(path) for path in existing)
        )

    resolved_output.mkdir(parents=True, exist_ok=True)
    created: list[Path] = []
    try:
        for filename, payload in artifacts.items():
            path = paths[filename]
            with path.open("xb") as stream:
                created.append(path)
                stream.write(payload)
    except BaseException:
        for path in reversed(created):
            _remove_created_output(path)
        raise
    return resolved_output


def print_completion(
    comparison: EdgeComparisonResult,
    output_dir: Path,
) -> None:
    """Print one concise B6 completion report."""
    print("Edge summary complete.")
    print()
    print("Experiments:")
    for record in comparison.records:
        print(f"  {_experiment(record)}")
    print()
    print("Repeat scope:")
    print("  one recording per condition; descriptive comparison only")
    coverage_warnings = [
        record
        for record in comparison.records
        if np.any(record.profile["pixel_count"] == 0)
    ]
    if coverage_warnings:
        print()
        print("Profile coverage warnings:")
        for record in coverage_warnings:
            coverage = _profile_coverage(record)
            centers = ", ".join(
                str(value)
                for value in coverage[
                    "zero_pixel_distance_centers_px"
                ]
            )
            print(
                f"  {_experiment(record)}: "
                f"zero-pixel bins at {centers} px"
            )
    print()
    print("Saved:")
    print(f"  {_summary_path(output_dir)}")


def main(
    argv: Sequence[str] | None = None,
) -> int:
    """Run the Scene 04 cross-dataset summarizer."""
    args = parse_args(argv)
    records = tuple(
        load_edge_summary_record(path)
        for path in args.result_dirs
    )
    comparison = validate_edge_comparison(records)
    output_dir = save_edge_summary(
        args.output_dir,
        comparison,
    )
    print_completion(comparison, output_dir)
    return 0


def _validate_record(record: EdgeSummaryRecord) -> None:
    """Validate cross-file identity, counts, status, and diagnostics."""
    experiment = _experiment(record)
    if record.result_dir.name != "edge_discontinuity":
        raise ValueError(
            "Result directory must be named edge_discontinuity"
        )
    if record.result_dir.parent.name != experiment:
        raise ValueError(
            "Summary experiment does not match result directory"
        )
    num_frames = _summary_int(
        record,
        "dataset",
        "num_frames",
    )
    foreground_distance = _summary_int(
        record,
        "setup",
        "nominal_foreground_distance_mm",
    )
    background_distance = _summary_int(
        record,
        "setup",
        "nominal_background_distance_mm",
    )
    nominal_gap = _summary_int(
        record,
        "setup",
        "nominal_gap_mm",
    )
    if background_distance != foreground_distance + nominal_gap:
        raise ValueError(
            "Nominal background distance must equal foreground "
            f"distance plus gap: {experiment}"
        )
    if len(record.frame_metrics) != num_frames:
        raise ValueError(
            f"Frame metric count does not match summary: {experiment}"
        )
    for index, row in enumerate(record.frame_metrics):
        try:
            frame_index = int(row["frame_index"])
        except (TypeError, ValueError) as error:
            raise ValueError(
                f"Invalid frame_index in {experiment}"
            ) from error
        if frame_index != index:
            raise ValueError(
                f"Frame indices are not contiguous in {experiment}"
            )

    valid = sum(
        row["analysis_status"] == "ok"
        for row in record.frame_metrics
    )
    transition_valid = sum(
        row["transition_status"] == "ok"
        for row in record.frame_metrics
    )
    if any(
        row["transition_status"] == "ok"
        and row["analysis_status"] != "ok"
        for row in record.frame_metrics
    ):
        raise ValueError(
            "Rejected frame cannot have a valid transition: "
            f"{experiment}"
        )
    if valid != _summary_int(record, "frames", "valid"):
        raise ValueError(
            f"Valid-frame count does not match summary: {experiment}"
        )
    if (
        num_frames - valid
        != _summary_int(record, "frames", "rejected")
    ):
        raise ValueError(
            f"Rejected-frame count does not match summary: {experiment}"
        )
    if transition_valid != _summary_int(
        record,
        "frames",
        "transition_valid",
    ):
        raise ValueError(
            "Transition-valid count does not match summary: "
            f"{experiment}"
        )
    if (
        valid - transition_valid
        != _summary_int(
            record,
            "frames",
            "transition_failed",
        )
    ):
        raise ValueError(
            "Transition-failed count does not match summary: "
            f"{experiment}"
        )
    if not _summary_bool(
        record,
        "diagnostics",
        "aggregate_profile_available",
    ):
        raise ValueError(
            f"Aggregate profile is unavailable: {experiment}"
        )
    if not _summary_bool(
        record,
        "diagnostics",
        "representative_label_map_available",
    ):
        raise ValueError(
            f"Representative label map is unavailable: {experiment}"
        )


def _validate_profile_counts(
    record: EdgeSummaryRecord,
) -> None:
    """Validate aggregate profile count and ratio identities."""
    profile = record.profile
    experiment = _experiment(record)
    for field_name in (
        "distance_min_px",
        "distance_max_px",
        "distance_center_px",
    ):
        if not np.all(np.isfinite(profile[field_name])):
            raise ValueError(
                f"Aggregate profile has invalid distances: {experiment}"
            )

    count_fields = tuple(COUNT_PROFILE_COLUMNS)
    if any(
        np.any(profile[field_name] < 0)
        for field_name in count_fields
    ):
        raise ValueError(
            f"Aggregate profile has negative counts: {experiment}"
        )

    pixel_count = profile["pixel_count"]
    valid_count = profile["valid_count"]
    invalid_count = profile["invalid_count"]
    classified_count = sum(
        (
            profile[field_name]
            for field_name in (
                "foreground_count",
                "background_count",
                "mixed_count",
                "outlier_count",
            )
        ),
        start=np.zeros_like(valid_count),
    )
    if not np.array_equal(
        valid_count + invalid_count,
        pixel_count,
    ):
        raise ValueError(
            "Aggregate profile valid and invalid counts do not "
            f"match pixel counts: {experiment}"
        )
    if not np.array_equal(classified_count, valid_count):
        raise ValueError(
            "Aggregate profile class counts do not match valid "
            f"counts: {experiment}"
        )

    ratio_specs = (
        ("foreground_ratio", "foreground_count", valid_count),
        ("background_ratio", "background_count", valid_count),
        ("mixed_ratio", "mixed_count", valid_count),
        ("outlier_ratio", "outlier_count", valid_count),
        ("invalid_ratio", "invalid_count", pixel_count),
    )
    for ratio_name, count_name, denominator in ratio_specs:
        ratio = profile[ratio_name]
        expected = np.full(
            ratio.shape,
            np.nan,
            dtype=np.float64,
        )
        np.divide(
            profile[count_name],
            denominator,
            out=expected,
            where=denominator > 0,
        )
        if not np.allclose(
            ratio,
            expected,
            rtol=1e-9,
            atol=1e-12,
            equal_nan=True,
        ):
            raise ValueError(
                f"Aggregate profile {ratio_name} does not match "
                f"counts: {experiment}"
            )


def _load_profile_csv(
    path: Path,
) -> Mapping[str, np.ndarray]:
    """Load one aggregate profile into typed one-dimensional arrays."""
    with path.open(
        "r",
        encoding="utf-8",
        newline="",
    ) as stream:
        reader = csv.DictReader(stream)
        if reader.fieldnames != list(PROFILE_COLUMNS):
            raise ValueError(
                f"Unexpected aggregate profile columns: {path}"
            )
        rows = tuple(reader)
    if not rows:
        raise ValueError(f"Aggregate profile is empty: {path}")

    arrays: dict[str, np.ndarray] = {}
    for column in PROFILE_COLUMNS:
        try:
            if column in COUNT_PROFILE_COLUMNS:
                arrays[column] = np.asarray(
                    [int(row[column]) for row in rows],
                    dtype=np.int64,
                )
            else:
                arrays[column] = np.asarray(
                    [
                        (
                            np.nan
                            if row[column] == ""
                            else float(row[column])
                        )
                        for row in rows
                    ],
                    dtype=np.float64,
                )
        except (TypeError, ValueError) as error:
            raise ValueError(
                f"Invalid aggregate profile value in {path}"
            ) from error
    return arrays


def _comparison_signature(
    record: EdgeSummaryRecord,
) -> tuple[object, ...]:
    """Return common settings that must match across all records."""
    reference = _nested_mapping(
        record.summary,
        "analysis_parameters",
        "reference",
    )
    preprocessing = _nested_mapping(
        record.summary,
        "depth_preprocessing",
    )
    excluded_values = preprocessing.get("excluded_raw_values")
    if (
        not isinstance(excluded_values, Sequence)
        or isinstance(excluded_values, (str, bytes))
    ):
        raise ValueError(
            "depth_preprocessing.excluded_raw_values "
            "must be a sequence"
        )
    depth_scale = preprocessing.get("depth_scale_to_mm")
    if (
        not isinstance(depth_scale, (int, float))
        or isinstance(depth_scale, bool)
        or not np.isfinite(depth_scale)
    ):
        raise ValueError(
            "depth_preprocessing.depth_scale_to_mm must be finite"
        )
    return (
        _summary_text(record, "setup", "target"),
        _summary_int(record, "setup", "nominal_gap_mm"),
        _summary_text(
            record,
            "setup",
            "distance_reference",
        ),
        _summary_int(record, "dataset", "width"),
        _summary_int(record, "dataset", "height"),
        _summary_float(
            record,
            "edge_geometry",
            "distance_bin_px",
        ),
        _summary_float(
            record,
            "edge_geometry",
            "max_edge_distance_px",
        ),
        tuple(sorted(reference.items())),
        tuple(excluded_values),
        float(depth_scale),
        _summary_float(
            record,
            "analysis_parameters",
            "bleeding_probability_threshold",
        ),
        _summary_float(
            record,
            "analysis_parameters",
            "invalid_ratio_threshold",
        ),
        _summary_float(
            record,
            "analysis_parameters",
            "transition_high_probability",
        ),
        _summary_float(
            record,
            "analysis_parameters",
            "transition_low_probability",
        ),
    )


def _summary_row(
    record: EdgeSummaryRecord,
) -> dict[str, object]:
    """Build one flat output row from summary and frame-level metrics."""
    valid_rows = [
        row
        for row in record.frame_metrics
        if row["analysis_status"] == "ok"
    ]
    transition_rows = [
        row
        for row in valid_rows
        if row["transition_status"] == "ok"
    ]
    num_frames = len(record.frame_metrics)
    valid_frames = len(valid_rows)
    transition_valid = len(transition_rows)

    row: dict[str, object] = {
        "experiment": _experiment(record),
        "orientation_token": _summary_text(
            record,
            "setup",
            "orientation_token",
        ),
        "orientation": _summary_text(
            record,
            "setup",
            "orientation",
        ),
        "target": _summary_text(record, "setup", "target"),
        "repeat_index": _summary_int(
            record,
            "setup",
            "repeat_index",
        ),
        "source_summary": _summary_path(record.summary_path),
        "num_frames": num_frames,
        "nominal_foreground_distance_mm": _summary_int(
            record,
            "setup",
            "nominal_foreground_distance_mm",
        ),
        "nominal_background_distance_mm": _summary_int(
            record,
            "setup",
            "nominal_background_distance_mm",
        ),
        "nominal_gap_mm": _summary_int(
            record,
            "setup",
            "nominal_gap_mm",
        ),
        "distance_reference": _summary_text(
            record,
            "setup",
            "distance_reference",
        ),
        "foreground_roi_pixels": _summary_int(
            record,
            "roi",
            "foreground_pixels",
        ),
        "background_roi_pixels": _summary_int(
            record,
            "roi",
            "background_pixels",
        ),
        "edge_roi_pixels": _summary_int(
            record,
            "roi",
            "edge_pixels",
        ),
        "distance_bin_px": _summary_float(
            record,
            "edge_geometry",
            "distance_bin_px",
        ),
        "max_edge_distance_px": _summary_float(
            record,
            "edge_geometry",
            "max_edge_distance_px",
        ),
        "valid_frames": valid_frames,
        "rejected_frames": num_frames - valid_frames,
        "valid_frame_ratio": valid_frames / num_frames,
        "transition_valid_frames": transition_valid,
        "transition_failed_frames": (
            valid_frames - transition_valid
        ),
        "transition_success_ratio": (
            transition_valid / valid_frames
            if valid_frames
            else ""
        ),
    }

    temporal_fields = {
        "foreground_reference": (
            valid_rows,
            "foreground_reference_mm",
        ),
        "background_reference": (
            valid_rows,
            "background_reference_mm",
        ),
        "measured_gap": (valid_rows, "measured_gap_mm"),
        "foreground_bleeding_ratio": (
            valid_rows,
            "foreground_bleeding_ratio",
        ),
        "background_bleeding_ratio": (
            valid_rows,
            "background_bleeding_ratio",
        ),
        "mixed_ratio": (valid_rows, "mixed_ratio"),
        "outlier_ratio": (valid_rows, "outlier_ratio"),
        "invalid_ratio": (valid_rows, "invalid_ratio"),
        "transition_width": (
            transition_rows,
            "transition_width_px",
        ),
        "nominal_offset": (
            transition_rows,
            "nominal_edge_offset_px",
        ),
    }
    statistics: dict[str, tuple[float | str, ...]] = {}
    for prefix, (rows, field_name) in temporal_fields.items():
        statistics[prefix] = _temporal_percentiles(
            rows,
            field_name,
        )
    for prefix, values in statistics.items():
        suffix = "mm" if prefix in {
            "foreground_reference",
            "background_reference",
            "measured_gap",
        } else ""
        if prefix in {"transition_width", "nominal_offset"}:
            suffix = "px"
        for stat, value in zip(
            ("p05", "median", "p95"),
            values,
            strict=True,
        ):
            key = f"{prefix}_{stat}"
            if suffix:
                key += f"_{suffix}"
            row[key] = value

    measured_gap_median = row["measured_gap_median_mm"]
    row["gap_error_mm"] = (
        ""
        if measured_gap_median == ""
        else (
            float(measured_gap_median)
            - _summary_int(record, "setup", "nominal_gap_mm")
        )
    )
    offset_values = _finite_frame_values(
        transition_rows,
        "nominal_edge_offset_px",
    )
    row["nominal_offset_std_px"] = (
        ""
        if offset_values.size == 0
        else float(np.std(offset_values))
    )

    missing = set(OUTPUT_COLUMNS).difference(row)
    if missing:
        raise RuntimeError(
            "Internal edge summary row is missing columns: "
            + ", ".join(sorted(missing))
        )
    return row


def _plot_metric_panels(
    records: Sequence[EdgeSummaryRecord],
    *,
    x: np.ndarray,
    x_labels: Sequence[str] | None,
    title: str,
    x_label: str,
):
    """Build the shared six-panel comparison figure."""
    from matplotlib.figure import Figure

    rows = [_summary_row(record) for record in records]
    figure = Figure(figsize=(12.0, 12.0))
    axes = figure.subplots(3, 2, sharex=True)
    flat = axes.ravel()

    _plot_temporal_series(
        flat[0],
        x,
        rows,
        "measured_gap",
        label="Measured gap",
        nominal=np.asarray(
            [row["nominal_gap_mm"] for row in rows],
            dtype=np.float64,
        ),
    )
    flat[0].set_ylabel("Gap error (mm)")
    flat[0].set_title("Measured gap error")

    _plot_ratio_series(
        flat[1],
        x,
        rows,
        (
            ("foreground_bleeding_ratio", "Foreground bleeding"),
            ("background_bleeding_ratio", "Background bleeding"),
        ),
    )
    flat[1].set_title("Bleeding ratios")

    _plot_ratio_series(
        flat[2],
        x,
        rows,
        (
            ("mixed_ratio", "Mixed"),
            ("outlier_ratio", "Outlier"),
            ("invalid_ratio", "Invalid"),
        ),
    )
    flat[2].set_title("Mixed, outlier, and invalid ratios")

    _plot_temporal_series(
        flat[3],
        x,
        rows,
        "transition_width",
        label="Transition width",
    )
    flat[3].set_ylabel("Width (px)")
    flat[3].set_title("Transition width")

    _plot_temporal_series(
        flat[4],
        x,
        rows,
        "nominal_offset",
        label="Offset from nominal line",
    )
    flat[4].axhline(0.0, color="0.4", linestyle=":")
    flat[4].set_ylabel("Offset (px)")
    flat[4].set_title("Nominal-line offset")

    success = np.asarray(
        [
            _plot_float(row["transition_success_ratio"])
            for row in rows
        ],
        dtype=np.float64,
    )
    flat[5].plot(
        x,
        success,
        marker="o",
        label="Transition success",
    )
    flat[5].set_ylim(0.0, 1.05)
    flat[5].set_ylabel("Ratio")
    flat[5].set_title("Transition success ratio")

    for axis in flat:
        axis.grid(True, alpha=0.25)
        axis.legend(loc="best")
        axis.set_xlabel(x_label)
        if x_labels is not None:
            axis.set_xticks(x, labels=x_labels)
    figure.suptitle(title)
    figure.text(
        0.5,
        0.005,
        (
            "Whiskers show within-recording p05–p95, "
            "not confidence intervals."
        ),
        ha="center",
        fontsize=9,
    )
    figure.tight_layout(rect=(0.0, 0.025, 1.0, 0.97))
    return figure


def _plot_profile_panels(
    records: Sequence[EdgeSummaryRecord],
    *,
    labels: Sequence[str],
    title: str,
):
    """Build four signed-distance profile panels."""
    from matplotlib.figure import Figure

    figure = Figure(figsize=(11.0, 8.0))
    axes = figure.subplots(2, 2, sharex=True, sharey=True)
    fields = (
        ("foreground_ratio", "Foreground ratio"),
        ("background_ratio", "Background ratio"),
        ("mixed_ratio", "Mixed ratio"),
        ("invalid_ratio", "Invalid ratio"),
    )
    for axis, (field_name, panel_title) in zip(
        axes.ravel(),
        fields,
        strict=True,
    ):
        for record, label in zip(records, labels, strict=True):
            axis.plot(
                record.profile["distance_center_px"],
                record.profile[field_name],
                marker="o",
                markersize=3,
                label=label,
            )
        axis.axvline(0.0, color="0.4", linestyle=":")
        axis.set_ylim(0.0, 1.0)
        axis.set_title(panel_title)
        axis.set_xlabel("Signed distance (px)")
        axis.set_ylabel("Ratio")
        axis.grid(True, alpha=0.25)
        axis.legend(loc="best")
    figure.suptitle(title)
    figure.tight_layout(rect=(0.0, 0.0, 1.0, 0.96))
    return figure


def _plot_temporal_series(
    axis: object,
    x: np.ndarray,
    rows: Sequence[Mapping[str, object]],
    prefix: str,
    *,
    label: str,
    nominal: np.ndarray | None = None,
) -> None:
    """Plot medians and asymmetric within-recording p05–p95."""
    median = np.asarray(
        [
            _plot_float(
                row[
                    f"{prefix}_median_"
                    f"{_metric_unit(prefix)}"
                ]
            )
            for row in rows
        ],
        dtype=np.float64,
    )
    p05 = np.asarray(
        [
            _plot_float(
                row[
                    f"{prefix}_p05_"
                    f"{_metric_unit(prefix)}"
                ]
            )
            for row in rows
        ],
        dtype=np.float64,
    )
    p95 = np.asarray(
        [
            _plot_float(
                row[
                    f"{prefix}_p95_"
                    f"{_metric_unit(prefix)}"
                ]
            )
            for row in rows
        ],
        dtype=np.float64,
    )
    if nominal is not None:
        median = median - nominal
        p05 = p05 - nominal
        p95 = p95 - nominal
    axis.errorbar(
        x,
        median,
        yerr=np.vstack((median - p05, p95 - median)),
        marker="o",
        capsize=4,
        label=label,
    )


def _plot_ratio_series(
    axis: object,
    x: np.ndarray,
    rows: Sequence[Mapping[str, object]],
    series: Sequence[tuple[str, str]],
) -> None:
    """Plot ratio medians with temporal p05–p95 whiskers."""
    for prefix, label in series:
        median = np.asarray(
            [
                _plot_float(row[f"{prefix}_median"])
                for row in rows
            ],
            dtype=np.float64,
        )
        p05 = np.asarray(
            [
                _plot_float(row[f"{prefix}_p05"])
                for row in rows
            ],
            dtype=np.float64,
        )
        p95 = np.asarray(
            [
                _plot_float(row[f"{prefix}_p95"])
                for row in rows
            ],
            dtype=np.float64,
        )
        axis.errorbar(
            x,
            median,
            yerr=np.vstack((median - p05, p95 - median)),
            marker="o",
            capsize=4,
            label=label,
        )
    axis.set_ylim(bottom=0.0)
    axis.set_ylabel("Ratio")


def _metric_unit(prefix: str) -> str:
    """Return the unit suffix used by one temporal output prefix."""
    if prefix == "measured_gap":
        return "mm"
    return "px"


def _plot_float(value: object) -> float:
    """Convert a summary value to float, preserving blanks as NaN."""
    if value == "":
        return float("nan")
    return float(value)


def _temporal_percentiles(
    rows: Sequence[Mapping[str, str]],
    field_name: str,
) -> tuple[float | str, float | str, float | str]:
    """Return finite p05/median/p95 or blank values."""
    values = _finite_frame_values(rows, field_name)
    if values.size == 0:
        return ("", "", "")
    percentiles = np.percentile(
        values,
        [5, 50, 95],
    )
    return tuple(float(value) for value in percentiles)


def _finite_frame_values(
    rows: Sequence[Mapping[str, str]],
    field_name: str,
) -> np.ndarray:
    """Parse one finite frame-level numeric field."""
    values: list[float] = []
    for row in rows:
        text = row[field_name]
        if text == "":
            continue
        try:
            value = float(text)
        except ValueError as error:
            raise ValueError(
                f"Invalid frame metric {field_name}: {text!r}"
            ) from error
        if np.isfinite(value):
            values.append(value)
    return np.asarray(values, dtype=np.float64)


def _select_record(
    records: Sequence[EdgeSummaryRecord],
    *,
    orientation: str,
    distance_mm: int,
) -> EdgeSummaryRecord:
    """Return the unique record for one comparison condition."""
    selected = [
        record
        for record in records
        if (
            _summary_text(record, "setup", "orientation")
            == orientation
            and _summary_int(
                record,
                "setup",
                "nominal_foreground_distance_mm",
            )
            == distance_mm
        )
    ]
    if len(selected) != 1:
        raise ValueError(
            f"Expected one {orientation} d{distance_mm} record"
        )
    return selected[0]


def _profile_coverage(
    record: EdgeSummaryRecord,
) -> dict[str, object]:
    """Describe populated and zero-pixel signed-distance bins."""
    counts = record.profile["pixel_count"]
    centers = record.profile["distance_center_px"]
    populated = counts > 0
    return {
        "experiment": _experiment(record),
        "total_bins": int(counts.size),
        "populated_bins": int(np.count_nonzero(populated)),
        "zero_pixel_distance_centers_px": [
            float(value)
            for value in centers[~populated]
        ],
    }


def _experiment(record: EdgeSummaryRecord) -> str:
    """Return the experiment name from one record."""
    return _summary_text(record, "dataset", "experiment")


def _summary_value(
    record: EdgeSummaryRecord,
    *keys: str,
) -> object:
    """Read one required nested summary value."""
    current: object = record.summary
    for key in keys:
        if not isinstance(current, Mapping) or key not in current:
            raise ValueError(
                f"Summary {_experiment_name_hint(record)} "
                f"is missing {'.'.join(keys)}"
            )
        current = current[key]
    return current


def _summary_text(
    record: EdgeSummaryRecord,
    *keys: str,
) -> str:
    value = _summary_value(record, *keys)
    if not isinstance(value, str) or not value:
        raise ValueError(
            f"Summary {'.'.join(keys)} must be a non-empty string"
        )
    return value


def _summary_int(
    record: EdgeSummaryRecord,
    *keys: str,
) -> int:
    value = _summary_value(record, *keys)
    if (
        not isinstance(value, int)
        or isinstance(value, bool)
    ):
        raise ValueError(
            f"Summary {'.'.join(keys)} must be an integer"
        )
    return value


def _summary_float(
    record: EdgeSummaryRecord,
    *keys: str,
) -> float:
    value = _summary_value(record, *keys)
    if (
        not isinstance(value, (int, float))
        or isinstance(value, bool)
        or not np.isfinite(value)
    ):
        raise ValueError(
            f"Summary {'.'.join(keys)} must be finite"
        )
    return float(value)


def _summary_bool(
    record: EdgeSummaryRecord,
    *keys: str,
) -> bool:
    value = _summary_value(record, *keys)
    if not isinstance(value, bool):
        raise ValueError(
            f"Summary {'.'.join(keys)} must be boolean"
        )
    return value


def _nested_mapping(
    mapping: Mapping[str, object],
    *keys: str,
) -> Mapping[str, object]:
    current: object = mapping
    for key in keys:
        if not isinstance(current, Mapping) or key not in current:
            raise ValueError(
                f"Missing required mapping {'.'.join(keys)}"
            )
        current = current[key]
    return _require_mapping(current, ".".join(keys))


def _require_mapping(
    value: object,
    name: str,
) -> Mapping[str, object]:
    if not isinstance(value, Mapping):
        raise ValueError(f"{name} must be a mapping")
    return value


def _experiment_name_hint(
    record: EdgeSummaryRecord,
) -> str:
    dataset = record.summary.get("dataset")
    if isinstance(dataset, Mapping):
        experiment = dataset.get("experiment")
        if isinstance(experiment, str):
            return experiment
    return str(record.summary_path)


def _validate_comparison_result(
    comparison: EdgeComparisonResult,
) -> None:
    if not isinstance(comparison, EdgeComparisonResult):
        raise TypeError(
            "comparison must be an EdgeComparisonResult"
        )
    if len(comparison.records) != 4:
        raise ValueError(
            "comparison must contain four records"
        )


def _figure_png_bytes(figure: object) -> bytes:
    from matplotlib.figure import Figure

    if not isinstance(figure, Figure):
        raise TypeError("figure must be a Matplotlib Figure")
    stream = BytesIO()
    try:
        figure.savefig(stream, format="png", dpi=150)
        return stream.getvalue()
    finally:
        figure.clear()


def _summary_path(path: Path) -> str:
    resolved = Path(path).expanduser().resolve()
    try:
        return str(resolved.relative_to(PROJECT_ROOT))
    except ValueError:
        return str(resolved)


def _remove_created_output(path: Path) -> None:
    try:
        path.unlink(missing_ok=True)
    except OSError:
        pass


if __name__ == "__main__":
    raise SystemExit(main())
