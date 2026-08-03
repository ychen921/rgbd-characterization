"""Load and validate the controlled Scene 01--03 baseline matrix."""

from collections.abc import Mapping, Sequence
import csv
from dataclasses import dataclass
from io import BytesIO, StringIO
from pathlib import Path
import re

import numpy as np
import yaml


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_RESULTS_ROOT = PROJECT_ROOT / "results"
DEFAULT_OUTPUT_DIR = DEFAULT_RESULTS_ROOT / "baseline_summary"

SUMMARY_FILENAME = "summary.yaml"
FRAME_MEDIAN_FILENAME = "frame_median_depth.csv"
FRAME_PLANE_FILENAME = "frame_plane_metrics.csv"
TEMPORAL_MAP_FILENAME = "temporal_std.npy"
ZERO_RATIO_MAP_FILENAME = "zero_ratio_map.npy"
MAX_UINT16_MAP_FILENAME = "max_uint16_ratio_map.npy"

OUTPUT_RECORDING_CSV_FILENAME = "baseline_summary.csv"
OUTPUT_CONDITION_CSV_FILENAME = "condition_summary.csv"
OUTPUT_YAML_FILENAME = "comparison_summary.yaml"
SCENE01_DEPTH_QUALITY_FILENAME = (
    "scene01_distance_depth_quality.png"
)
SCENE01_PLANARITY_FILENAME = "scene01_distance_planarity.png"
SCENE02_DEPTH_QUALITY_FILENAME = "scene02_angle_depth_quality.png"
SCENE02_PLANARITY_FILENAME = "scene02_angle_planarity.png"
SCENE03_DEPTH_QUALITY_FILENAME = "scene03_target_depth_quality.png"
SCENE03_PLANARITY_FILENAME = "scene03_target_planarity.png"

REQUIRED_FRAME_MEDIAN_COLUMNS = frozenset(
    {
        "frame_index",
        "timestamp_ns",
        "median_depth_mm",
    }
)
REQUIRED_FRAME_PLANE_COLUMNS = frozenset(
    {
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
    }
)

IDENTITY_COLUMNS = (
    "experiment",
    "scene",
    "condition",
    "target",
    "nominal_distance_mm",
    "yaw_deg",
    "repeat_index",
    "source_summary",
    "num_frames",
)
ROI_COLUMNS = (
    "roi_width",
    "roi_height",
    "roi_pixels",
)
DEPTH_QUALITY_COLUMNS = (
    "zero_ratio",
    "max_uint16_ratio",
    "max_uint16_affected_frames",
    "max_uint16_max_pixels_per_frame",
)
TEMPORAL_MEASURED_COLUMNS = (
    "temporal_median_std_mm",
    "temporal_mean_std_mm",
    "temporal_p95_std_mm",
    "measured_median_mm",
    "measured_mean_mm",
    "measured_std_mm",
    "measured_p05_mm",
    "measured_p95_mm",
    "measured_offset_from_nominal_mm",
)
PLANARITY_COLUMNS = (
    "plane_successful_frames",
    "plane_failed_frames",
    "plane_success_ratio",
    "plane_distance_median_mm",
    "plane_distance_offset_from_nominal_mm",
    "plane_distance_temporal_std_mm",
    "tilt_median_deg",
    "tilt_temporal_std_deg",
    "tilt_error_from_nominal_deg",
    "plane_rmse_median_mm",
    "plane_rmse_p95_mm",
    "plane_p95_abs_median_mm",
    "plane_inlier_ratio_median",
)
RECORDING_COLUMNS = (
    IDENTITY_COLUMNS
    + ROI_COLUMNS
    + DEPTH_QUALITY_COLUMNS
    + TEMPORAL_MEASURED_COLUMNS
    + PLANARITY_COLUMNS
)

CONDITION_IDENTITY_COLUMNS = (
    "scene",
    "condition",
    "target",
    "nominal_distance_mm",
    "yaw_deg",
    "repeat_count",
    "total_frames",
    "min_frames_per_repeat",
    "max_frames_per_repeat",
    "roi_width",
    "roi_height",
    "roi_pixels",
)
AGGREGATED_METRICS = (
    "zero_ratio",
    "max_uint16_ratio",
    "temporal_median_std_mm",
    "temporal_mean_std_mm",
    "temporal_p95_std_mm",
    "measured_median_mm",
    "measured_mean_mm",
    "measured_std_mm",
    "measured_p05_mm",
    "measured_p95_mm",
    "measured_offset_from_nominal_mm",
    "plane_success_ratio",
    "plane_distance_median_mm",
    "plane_distance_offset_from_nominal_mm",
    "plane_distance_temporal_std_mm",
    "tilt_median_deg",
    "tilt_temporal_std_deg",
    "tilt_error_from_nominal_deg",
    "plane_rmse_median_mm",
    "plane_rmse_p95_mm",
    "plane_p95_abs_median_mm",
    "plane_inlier_ratio_median",
)
CONDITION_COLUMNS = CONDITION_IDENTITY_COLUMNS + tuple(
    column
    for metric in AGGREGATED_METRICS
    for column in (
        f"{metric}_mean",
        f"{metric}_repeat_std",
        f"{metric}_valid_count",
    )
)

SCENE01_DISTANCES_MM = (500, 1000, 1500, 2000, 3000)
SCENE02_YAWS_DEG = (0, 15, 30, 45, 60)
SCENE03_TARGETS = (
    "black",
    "cbd",
    "reflection",
    "transparent",
)
REPEAT_INDICES = (1, 2, 3)

_SCENE01_PATTERN = re.compile(
    r"^scene01_white_d(?P<distance>\d{3})_r(?P<repeat>\d{2})$"
)
_SCENE02_PATTERN = re.compile(
    r"^scene02_(?:yaw(?P<yaw>\d+)_)?white_d100_"
    r"r(?P<repeat>\d{2})$"
)
_SCENE03_PATTERN = re.compile(
    r"^scene03_(?P<target>black|cbd|reflection|transparent)_"
    r"d100_r(?P<repeat>\d{2})$"
)


@dataclass(frozen=True)
class ExperimentIdentity:
    """Store metadata parsed from one controlled experiment name."""

    experiment: str
    scene: str
    condition: str
    target: str
    nominal_distance_mm: int
    yaw_deg: int | None
    repeat_index: int


@dataclass(frozen=True)
class BaselineSummaryRecord:
    """Store one validated baseline result and its detailed inputs."""

    result_dir: Path
    summary_path: Path
    frame_median_path: Path
    frame_plane_path: Path
    temporal_map_path: Path
    zero_ratio_map_path: Path
    max_uint16_map_path: Path
    identity: ExperimentIdentity
    summary: Mapping[str, object]
    frame_medians: tuple[Mapping[str, str], ...]
    frame_planes: tuple[Mapping[str, str], ...]
    temporal_map_shape: tuple[int, ...]
    zero_ratio_map_shape: tuple[int, ...]
    max_uint16_map_shape: tuple[int, ...]


@dataclass(frozen=True)
class BaselineComparison:
    """Store the validated 42-recording matrix in deterministic order."""

    records: tuple[BaselineSummaryRecord, ...]


@dataclass(frozen=True)
class MetricAggregate:
    """Store one metric's repeat-level descriptive statistics."""

    mean: float | None
    repeat_std: float | None
    valid_count: int


@dataclass(frozen=True)
class PlotMetric:
    """Define one repeat-aggregated metric panel."""

    metric: str
    title: str
    ylabel: str
    percent: bool = False
    zero_reference: bool = False


def _distance_token(distance_mm: int) -> str:
    if distance_mm <= 0 or distance_mm % 10 != 0:
        raise ValueError("distance_mm must be a positive multiple of 10")
    return f"d{distance_mm // 10:03d}"


def build_expected_experiments() -> tuple[str, ...]:
    """Return the fixed Scene 01--03 experiment matrix in output order."""
    experiments: list[str] = []
    for distance_mm in SCENE01_DISTANCES_MM:
        distance = _distance_token(distance_mm)
        for repeat in REPEAT_INDICES:
            experiments.append(
                f"scene01_white_{distance}_r{repeat:02d}"
            )
    for yaw_deg in SCENE02_YAWS_DEG:
        condition = (
            "scene02_white_d100"
            if yaw_deg == 0
            else f"scene02_yaw{yaw_deg}_white_d100"
        )
        for repeat in REPEAT_INDICES:
            experiments.append(f"{condition}_r{repeat:02d}")
    for target in SCENE03_TARGETS:
        for repeat in REPEAT_INDICES:
            experiments.append(
                f"scene03_{target}_d100_r{repeat:02d}"
            )
    return tuple(experiments)


EXPECTED_EXPERIMENTS = build_expected_experiments()
EXPECTED_EXPERIMENT_SET = frozenset(EXPECTED_EXPERIMENTS)
_EXPECTED_ORDER = {
    experiment: index
    for index, experiment in enumerate(EXPECTED_EXPERIMENTS)
}


def parse_experiment_name(experiment: str) -> ExperimentIdentity:
    """Parse one approved Scene 01--03 experiment name."""
    if not isinstance(experiment, str) or not experiment:
        raise ValueError("experiment name must be a non-empty string")
    if experiment not in EXPECTED_EXPERIMENT_SET:
        raise ValueError(
            "Experiment is not part of the controlled baseline matrix: "
            f"{experiment}"
        )

    match = _SCENE01_PATTERN.fullmatch(experiment)
    if match is not None:
        distance_mm = int(match.group("distance")) * 10
        repeat = int(match.group("repeat"))
        return ExperimentIdentity(
            experiment=experiment,
            scene="scene01",
            condition=_condition_name(experiment),
            target="white",
            nominal_distance_mm=distance_mm,
            yaw_deg=None,
            repeat_index=repeat,
        )

    match = _SCENE02_PATTERN.fullmatch(experiment)
    if match is not None:
        yaw_text = match.group("yaw")
        return ExperimentIdentity(
            experiment=experiment,
            scene="scene02",
            condition=_condition_name(experiment),
            target="white",
            nominal_distance_mm=1000,
            yaw_deg=0 if yaw_text is None else int(yaw_text),
            repeat_index=int(match.group("repeat")),
        )

    match = _SCENE03_PATTERN.fullmatch(experiment)
    if match is not None:
        return ExperimentIdentity(
            experiment=experiment,
            scene="scene03",
            condition=_condition_name(experiment),
            target=match.group("target"),
            nominal_distance_mm=1000,
            yaw_deg=None,
            repeat_index=int(match.group("repeat")),
        )

    raise ValueError(
        f"Unsupported controlled experiment name: {experiment}"
    )


def discover_baseline_result_dirs(
    results_root: Path,
) -> tuple[Path, ...]:
    """Discover and validate the fixed matrix beneath a results root."""
    resolved_root = Path(results_root).expanduser()
    if not resolved_root.is_dir():
        raise FileNotFoundError(
            f"Baseline results root not found: {resolved_root}"
        )

    discovered: dict[str, Path] = {}
    for result_dir in resolved_root.glob("scene0[1-3]_*/baseline"):
        if not result_dir.is_dir():
            continue
        experiment = result_dir.parent.name
        if experiment in discovered:
            raise ValueError(
                f"Duplicate baseline experiment directory: {experiment}"
            )
        parse_experiment_name(experiment)
        discovered[experiment] = result_dir

    observed = frozenset(discovered)
    missing = sorted(EXPECTED_EXPERIMENT_SET - observed)
    unexpected = sorted(observed - EXPECTED_EXPERIMENT_SET)
    if missing or unexpected:
        details: list[str] = []
        if missing:
            details.append("missing: " + ", ".join(missing))
        if unexpected:
            details.append("unexpected: " + ", ".join(unexpected))
        raise ValueError(
            "Baseline comparison requires the fixed 42-result matrix ("
            + "; ".join(details)
            + ")"
        )

    return tuple(
        discovered[experiment]
        for experiment in EXPECTED_EXPERIMENTS
    )


def load_baseline_summary_record(
    result_dir: Path,
) -> BaselineSummaryRecord:
    """Load and validate one completed baseline-analysis directory."""
    resolved_dir = Path(result_dir).expanduser()
    paths = {
        SUMMARY_FILENAME: resolved_dir / SUMMARY_FILENAME,
        FRAME_MEDIAN_FILENAME: resolved_dir / FRAME_MEDIAN_FILENAME,
        FRAME_PLANE_FILENAME: resolved_dir / FRAME_PLANE_FILENAME,
        TEMPORAL_MAP_FILENAME: resolved_dir / TEMPORAL_MAP_FILENAME,
        ZERO_RATIO_MAP_FILENAME: resolved_dir / ZERO_RATIO_MAP_FILENAME,
        MAX_UINT16_MAP_FILENAME: resolved_dir / MAX_UINT16_MAP_FILENAME,
    }
    for path in paths.values():
        if not path.is_file():
            raise FileNotFoundError(
                f"Required baseline summary input not found: {path}"
            )

    summary_path = paths[SUMMARY_FILENAME]
    with summary_path.open("r", encoding="utf-8") as stream:
        summary = _require_mapping(
            yaml.safe_load(stream),
            f"summary {summary_path}",
        )

    experiment = resolved_dir.parent.name
    identity = parse_experiment_name(experiment)
    frame_medians = _load_csv(
        paths[FRAME_MEDIAN_FILENAME],
        REQUIRED_FRAME_MEDIAN_COLUMNS,
        "Frame-median",
    )
    frame_planes = _load_csv(
        paths[FRAME_PLANE_FILENAME],
        REQUIRED_FRAME_PLANE_COLUMNS,
        "Frame-plane",
    )
    temporal_map = _load_map(paths[TEMPORAL_MAP_FILENAME])
    zero_ratio_map = _load_map(paths[ZERO_RATIO_MAP_FILENAME])
    max_uint16_map = _load_map(paths[MAX_UINT16_MAP_FILENAME])

    record = BaselineSummaryRecord(
        result_dir=resolved_dir,
        summary_path=summary_path,
        frame_median_path=paths[FRAME_MEDIAN_FILENAME],
        frame_plane_path=paths[FRAME_PLANE_FILENAME],
        temporal_map_path=paths[TEMPORAL_MAP_FILENAME],
        zero_ratio_map_path=paths[ZERO_RATIO_MAP_FILENAME],
        max_uint16_map_path=paths[MAX_UINT16_MAP_FILENAME],
        identity=identity,
        summary=summary,
        frame_medians=frame_medians,
        frame_planes=frame_planes,
        temporal_map_shape=temporal_map.shape,
        zero_ratio_map_shape=zero_ratio_map.shape,
        max_uint16_map_shape=max_uint16_map.shape,
    )
    _validate_record(
        record,
        temporal_map=temporal_map,
        zero_ratio_map=zero_ratio_map,
        max_uint16_map=max_uint16_map,
    )
    return record


def validate_baseline_comparison(
    records: Sequence[BaselineSummaryRecord],
) -> BaselineComparison:
    """Validate and order the fixed 42-recording baseline matrix."""
    normalized = tuple(records)
    if len(normalized) != len(EXPECTED_EXPERIMENTS):
        raise ValueError(
            "Baseline comparison requires exactly 42 result records"
        )
    if any(
        not isinstance(record, BaselineSummaryRecord)
        for record in normalized
    ):
        raise TypeError(
            "records must contain BaselineSummaryRecord values"
        )

    experiments = [record.identity.experiment for record in normalized]
    if len(set(experiments)) != len(experiments):
        raise ValueError("Baseline comparison contains duplicate experiments")
    if frozenset(experiments) != EXPECTED_EXPERIMENT_SET:
        raise ValueError(
            "Baseline comparison must contain the fixed Scene 01--03 "
            "experiment matrix"
        )

    by_condition: dict[str, list[BaselineSummaryRecord]] = {}
    for record in normalized:
        by_condition.setdefault(
            record.identity.condition,
            [],
        ).append(record)
    if len(by_condition) != 14:
        raise ValueError(
            "Baseline comparison must contain exactly 14 conditions"
        )
    for condition, condition_records in by_condition.items():
        repeats = {
            record.identity.repeat_index
            for record in condition_records
        }
        if repeats != set(REPEAT_INDICES):
            raise ValueError(
                f"Condition must contain r01/r02/r03: {condition}"
            )
        roi_signatures = {
            _roi_signature(record)
            for record in condition_records
        }
        if len(roi_signatures) != 1:
            raise ValueError(
                f"Condition repeats use inconsistent ROI settings: "
                f"{condition}"
            )

    common_signatures = {
        _common_analysis_signature(record)
        for record in normalized
    }
    if len(common_signatures) != 1:
        raise ValueError(
            "Baseline result summaries use inconsistent common settings"
        )

    ordered = tuple(
        sorted(
            normalized,
            key=lambda record: _EXPECTED_ORDER[
                record.identity.experiment
            ],
        )
    )
    return BaselineComparison(records=ordered)


def load_and_validate_baseline_comparison(
    results_root: Path,
) -> BaselineComparison:
    """Discover, load, validate, and order all controlled results."""
    result_dirs = discover_baseline_result_dirs(results_root)
    records = tuple(
        load_baseline_summary_record(result_dir)
        for result_dir in result_dirs
    )
    return validate_baseline_comparison(records)


def build_recording_rows(
    comparison: BaselineComparison,
) -> tuple[Mapping[str, object], ...]:
    """Build one deterministic summary row per validated recording."""
    validated = _validate_comparison_result(comparison)
    return tuple(
        _recording_row(record)
        for record in validated.records
    )


def build_baseline_summary_csv(
    comparison: BaselineComparison,
) -> str:
    """Serialize the 42 per-recording rows as deterministic CSV."""
    return _build_csv_text(
        RECORDING_COLUMNS,
        build_recording_rows(comparison),
    )


def build_condition_rows(
    comparison: BaselineComparison,
) -> tuple[Mapping[str, object], ...]:
    """Aggregate r01/r02/r03 into one row per condition."""
    validated = _validate_comparison_result(comparison)
    recording_rows = build_recording_rows(validated)
    grouped: dict[str, list[Mapping[str, object]]] = {}
    for row in recording_rows:
        condition = row["condition"]
        if not isinstance(condition, str):
            raise TypeError("recording condition must be a string")
        grouped.setdefault(condition, []).append(row)

    condition_rows = tuple(
        _condition_row(rows)
        for rows in grouped.values()
    )
    if len(condition_rows) != 14:
        raise ValueError(
            "Condition summary requires exactly 14 condition rows"
        )
    return condition_rows


def build_condition_summary_csv(
    comparison: BaselineComparison,
) -> str:
    """Serialize the 14 repeat-aggregated rows as deterministic CSV."""
    return _build_csv_text(
        CONDITION_COLUMNS,
        build_condition_rows(comparison),
    )


def build_comparison_summary(
    comparison: BaselineComparison,
) -> dict[str, object]:
    """Build the planar comparison provenance and coverage document."""
    validated = _validate_comparison_result(comparison)
    condition_rows = build_condition_rows(validated)
    first = validated.records[0]
    camera = _nested_mapping(first.summary, "depth_camera")
    preprocessing = _nested_mapping(
        first.summary,
        "depth_preprocessing",
    )
    temporal = _nested_mapping(first.summary, "temporal_noise")
    planarity = _nested_mapping(first.summary, "planarity")
    excluded = preprocessing.get("excluded_raw_values")
    if not isinstance(excluded, list):
        raise ValueError("excluded_raw_values must be a list")

    scene01 = _rows_for_scene(condition_rows, "scene01")
    scene02 = _rows_for_scene(condition_rows, "scene02")
    scene03 = _rows_for_scene(condition_rows, "scene03")
    return {
        "inputs": {
            "expected_dataset_count": len(EXPECTED_EXPERIMENTS),
            "observed_dataset_count": len(validated.records),
            "condition_count": len(condition_rows),
            "experiments": [
                record.identity.experiment
                for record in validated.records
            ],
            "summary_paths": [
                _summary_path(record.summary_path)
                for record in validated.records
            ],
        },
        "comparison_groups": {
            "scene01_distance": {
                "scene": "scene01",
                "independent_variable": "nominal_distance_mm",
                "values": [
                    _condition_int(row, "nominal_distance_mm")
                    for row in scene01
                ],
                "conditions": [
                    _condition_text(row, "condition")
                    for row in scene01
                ],
            },
            "scene02_angle": {
                "scene": "scene02",
                "independent_variable": "yaw_deg",
                "values": [
                    _condition_int(row, "yaw_deg")
                    for row in scene02
                ],
                "conditions": [
                    _condition_text(row, "condition")
                    for row in scene02
                ],
            },
            "scene03_target": {
                "scene": "scene03",
                "independent_variable": "target",
                "values": [
                    _condition_text(row, "target")
                    for row in scene03
                ],
                "conditions": [
                    _condition_text(row, "condition")
                    for row in scene03
                ],
            },
        },
        "common_analysis": {
            "depth_camera": {
                "config": _required_text(
                    camera,
                    "config",
                    "depth_camera",
                ),
                "frame_id": _required_text(
                    camera,
                    "frame_id",
                    "depth_camera",
                ),
                "width": _required_int(
                    camera,
                    "width",
                    "depth_camera",
                ),
                "height": _required_int(
                    camera,
                    "height",
                    "depth_camera",
                ),
                "fx": _finite_float(camera, "fx", "depth_camera"),
                "fy": _finite_float(camera, "fy", "depth_camera"),
                "cx": _finite_float(camera, "cx", "depth_camera"),
                "cy": _finite_float(camera, "cy", "depth_camera"),
            },
            "depth_preprocessing": {
                "excluded_raw_values": list(excluded),
                "depth_scale_to_mm": _finite_float(
                    preprocessing,
                    "depth_scale_to_mm",
                    "depth_preprocessing",
                ),
            },
            "temporal_noise": {
                "min_valid_ratio": _finite_float(
                    temporal,
                    "min_valid_ratio",
                    "temporal_noise",
                ),
            },
            "plane_fitting": {
                "method": _required_text(
                    planarity,
                    "fitting_method",
                    "planarity",
                ),
                "inlier_threshold_mm": _finite_float(
                    planarity,
                    "inlier_threshold_mm",
                    "planarity",
                ),
                "min_valid_points": _required_int(
                    planarity,
                    "min_valid_points",
                    "planarity",
                ),
            },
        },
        "repeat_scope": {
            "repeat_count_per_condition": len(REPEAT_INDICES),
            "statistic": "sample_standard_deviation",
            "ddof": 1,
            "confidence_intervals_available": False,
            "error_bar_definition": (
                "repeat-to-repeat sample standard deviation"
            ),
        },
        "roi_coverage": {
            "conditions": [
                {
                    "condition": _condition_text(row, "condition"),
                    "width": _condition_int(row, "roi_width"),
                    "height": _condition_int(row, "roi_height"),
                    "pixel_count": _condition_int(row, "roi_pixels"),
                }
                for row in condition_rows
            ],
        },
        "metric_coverage": _metric_coverage(condition_rows),
        "interpretation_limits": [
            (
                "nominal offsets are setup-relative differences, "
                "not sensor bias"
            ),
            (
                "repeat error bars are sample standard deviations, "
                "not confidence intervals"
            ),
            (
                "within-recording temporal variation is distinct "
                "from repeat variation"
            ),
            "ROI dimensions differ between conditions",
            (
                "Scene 02 tilt error depends on nominal setup-angle "
                "accuracy"
            ),
            "metrics are not combined into a generic quality score",
        ],
    }


def plot_scene01_depth_quality(
    comparison: BaselineComparison,
):
    """Plot Scene 01 distance-dependent depth-quality metrics."""
    rows = _scene_rows(comparison, "scene01")
    x = _numeric_x(rows, "nominal_distance_mm")
    return _plot_condition_panels(
        rows,
        x=x,
        x_labels=None,
        x_label="Nominal distance (mm)",
        title="Scene 01: distance-dependent depth quality",
        panels=(
            PlotMetric(
                "measured_offset_from_nominal_mm",
                "Measured-depth offset",
                "Offset (mm)",
                zero_reference=True,
            ),
            PlotMetric(
                "temporal_median_std_mm",
                "Temporal median noise",
                "Standard deviation (mm)",
            ),
            PlotMetric(
                "zero_ratio",
                "Zero-depth ratio",
                "Ratio (%)",
                percent=True,
            ),
            PlotMetric(
                "max_uint16_ratio",
                "Maximum-uint16 ratio",
                "Ratio (%)",
                percent=True,
            ),
        ),
    )


def plot_scene01_planarity(
    comparison: BaselineComparison,
):
    """Plot Scene 01 distance-dependent planarity metrics."""
    rows = _scene_rows(comparison, "scene01")
    x = _numeric_x(rows, "nominal_distance_mm")
    return _plot_condition_panels(
        rows,
        x=x,
        x_labels=None,
        x_label="Nominal distance (mm)",
        title="Scene 01: distance-dependent planarity",
        panels=_distance_planarity_panels(),
    )


def plot_scene02_depth_quality(
    comparison: BaselineComparison,
):
    """Plot Scene 02 yaw-dependent depth-quality metrics."""
    rows = _scene_rows(comparison, "scene02")
    x = _numeric_x(rows, "yaw_deg")
    return _plot_condition_panels(
        rows,
        x=x,
        x_labels=None,
        x_label="Nominal yaw (deg)",
        title="Scene 02: angle-dependent depth quality",
        panels=(
            PlotMetric(
                "measured_offset_from_nominal_mm",
                "Measured-depth offset",
                "Offset (mm)",
                zero_reference=True,
            ),
            PlotMetric(
                "temporal_median_std_mm",
                "Temporal median noise",
                "Standard deviation (mm)",
            ),
            PlotMetric(
                "zero_ratio",
                "Zero-depth ratio",
                "Ratio (%)",
                percent=True,
            ),
            PlotMetric(
                "max_uint16_ratio",
                "Maximum-uint16 ratio",
                "Ratio (%)",
                percent=True,
            ),
        ),
    )


def plot_scene02_planarity(
    comparison: BaselineComparison,
):
    """Plot Scene 02 yaw-dependent planarity metrics."""
    rows = _scene_rows(comparison, "scene02")
    x = _numeric_x(rows, "yaw_deg")
    figure = _plot_condition_panels(
        rows,
        x=x,
        x_labels=None,
        x_label="Nominal yaw (deg)",
        title="Scene 02: angle-dependent planarity",
        panels=(
            PlotMetric(
                "tilt_median_deg",
                "Fitted tilt",
                "Tilt (deg)",
            ),
            PlotMetric(
                "tilt_error_from_nominal_deg",
                "Tilt error from nominal",
                "Error (deg)",
                zero_reference=True,
            ),
            PlotMetric(
                "plane_distance_offset_from_nominal_mm",
                "Plane-distance offset",
                "Offset (mm)",
                zero_reference=True,
            ),
            PlotMetric(
                "plane_rmse_median_mm",
                "Median residual RMSE",
                "RMSE (mm)",
            ),
            PlotMetric(
                "plane_p95_abs_median_mm",
                "Median p95 absolute residual",
                "Residual (mm)",
            ),
            PlotMetric(
                "plane_inlier_ratio_median",
                "Median plane inlier ratio",
                "Ratio (%)",
                percent=True,
            ),
        ),
    )
    fitted_tilt_axis = figure.axes[0]
    fitted_tilt_axis.plot(
        x,
        x,
        color="0.35",
        linestyle="--",
        label="Nominal yaw reference",
    )
    fitted_tilt_axis.legend(loc="best")
    return figure


def plot_scene03_depth_quality(
    comparison: BaselineComparison,
):
    """Plot Scene 03 target-dependent depth-quality metrics."""
    rows = _scene_rows(comparison, "scene03")
    x = np.arange(len(rows), dtype=np.float64)
    return _plot_condition_panels(
        rows,
        x=x,
        x_labels=_target_labels(rows),
        x_label="Target",
        title="Scene 03: target-dependent depth quality",
        panels=(
            PlotMetric(
                "measured_offset_from_nominal_mm",
                "Measured-depth offset",
                "Offset (mm)",
                zero_reference=True,
            ),
            PlotMetric(
                "temporal_median_std_mm",
                "Temporal median noise",
                "Standard deviation (mm)",
            ),
            PlotMetric(
                "zero_ratio",
                "Zero-depth ratio",
                "Ratio (%)",
                percent=True,
            ),
            PlotMetric(
                "max_uint16_ratio",
                "Maximum-uint16 ratio",
                "Ratio (%)",
                percent=True,
            ),
        ),
    )


def plot_scene03_planarity(
    comparison: BaselineComparison,
):
    """Plot Scene 03 target-dependent planarity metrics."""
    rows = _scene_rows(comparison, "scene03")
    x = np.arange(len(rows), dtype=np.float64)
    return _plot_condition_panels(
        rows,
        x=x,
        x_labels=_target_labels(rows),
        x_label="Target",
        title="Scene 03: target-dependent planarity",
        panels=_distance_planarity_panels(),
    )


def build_baseline_summary_artifacts(
    comparison: BaselineComparison,
) -> dict[str, bytes]:
    """Build both CSVs, provenance YAML, and six plots in memory."""
    validated = _validate_comparison_result(comparison)
    return {
        OUTPUT_RECORDING_CSV_FILENAME: build_baseline_summary_csv(
            validated
        ).encode("utf-8"),
        OUTPUT_CONDITION_CSV_FILENAME: build_condition_summary_csv(
            validated
        ).encode("utf-8"),
        OUTPUT_YAML_FILENAME: yaml.safe_dump(
            build_comparison_summary(validated),
            sort_keys=False,
            allow_unicode=True,
        ).encode("utf-8"),
        SCENE01_DEPTH_QUALITY_FILENAME: _figure_png_bytes(
            plot_scene01_depth_quality(validated)
        ),
        SCENE01_PLANARITY_FILENAME: _figure_png_bytes(
            plot_scene01_planarity(validated)
        ),
        SCENE02_DEPTH_QUALITY_FILENAME: _figure_png_bytes(
            plot_scene02_depth_quality(validated)
        ),
        SCENE02_PLANARITY_FILENAME: _figure_png_bytes(
            plot_scene02_planarity(validated)
        ),
        SCENE03_DEPTH_QUALITY_FILENAME: _figure_png_bytes(
            plot_scene03_depth_quality(validated)
        ),
        SCENE03_PLANARITY_FILENAME: _figure_png_bytes(
            plot_scene03_planarity(validated)
        ),
    }


def _recording_row(
    record: BaselineSummaryRecord,
) -> dict[str, object]:
    summary = record.summary
    identity = record.identity
    dataset = _nested_mapping(summary, "dataset")
    roi = _nested_mapping(summary, "roi")
    quality = _nested_mapping(summary, "depth_quality")
    max_uint16 = _nested_mapping(quality, "max_uint16")
    temporal = _nested_mapping(summary, "temporal_noise")
    measured = _nested_mapping(summary, "measured_depth")
    planarity = _nested_mapping(summary, "planarity")
    plane_distance = _nested_mapping(planarity, "plane_distance")
    tilt = _nested_mapping(planarity, "tilt")
    residual = _nested_mapping(planarity, "residual")
    inlier = _nested_mapping(planarity, "inlier_ratio")

    measured_median = _optional_non_negative(
        measured,
        "median_mm",
        "measured_depth",
    )
    plane_distance_m = _optional_non_negative(
        plane_distance,
        "median_m",
        "plane_distance",
    )
    plane_distance_mm = _optional_scale(
        plane_distance_m,
        1000.0,
    )
    tilt_median = _optional_non_negative(
        tilt,
        "median_deg",
        "tilt",
    )
    successful_frames = _non_negative_int(
        planarity,
        "successful_frames",
        "planarity",
    )
    failed_frames = _non_negative_int(
        planarity,
        "failed_frames",
        "planarity",
    )
    num_frames = _positive_int(dataset, "num_frames", "dataset")

    return {
        "experiment": identity.experiment,
        "scene": identity.scene,
        "condition": identity.condition,
        "target": identity.target,
        "nominal_distance_mm": identity.nominal_distance_mm,
        "yaw_deg": identity.yaw_deg,
        "repeat_index": identity.repeat_index,
        "source_summary": _summary_path(record.summary_path),
        "num_frames": num_frames,
        "roi_width": _positive_int(roi, "width", "roi"),
        "roi_height": _positive_int(roi, "height", "roi"),
        "roi_pixels": _positive_int(roi, "pixel_count", "roi"),
        "zero_ratio": _optional_ratio(
            quality,
            "zero_ratio",
            "depth_quality",
        ),
        "max_uint16_ratio": _optional_ratio(
            max_uint16,
            "ratio",
            "depth_quality.max_uint16",
        ),
        "max_uint16_affected_frames": _non_negative_int(
            max_uint16,
            "affected_frames",
            "depth_quality.max_uint16",
        ),
        "max_uint16_max_pixels_per_frame": _non_negative_int(
            max_uint16,
            "max_pixels_per_frame",
            "depth_quality.max_uint16",
        ),
        "temporal_median_std_mm": _optional_non_negative(
            temporal,
            "median_std_mm",
            "temporal_noise",
        ),
        "temporal_mean_std_mm": _optional_non_negative(
            temporal,
            "mean_std_mm",
            "temporal_noise",
        ),
        "temporal_p95_std_mm": _optional_non_negative(
            temporal,
            "p95_std_mm",
            "temporal_noise",
        ),
        "measured_median_mm": measured_median,
        "measured_mean_mm": _optional_non_negative(
            measured,
            "mean_mm",
            "measured_depth",
        ),
        "measured_std_mm": _optional_non_negative(
            measured,
            "std_mm",
            "measured_depth",
        ),
        "measured_p05_mm": _optional_non_negative(
            measured,
            "p05_mm",
            "measured_depth",
        ),
        "measured_p95_mm": _optional_non_negative(
            measured,
            "p95_mm",
            "measured_depth",
        ),
        "measured_offset_from_nominal_mm": _optional_offset(
            measured_median,
            identity.nominal_distance_mm,
        ),
        "plane_successful_frames": successful_frames,
        "plane_failed_frames": failed_frames,
        "plane_success_ratio": successful_frames / num_frames,
        "plane_distance_median_mm": plane_distance_mm,
        "plane_distance_offset_from_nominal_mm": _optional_offset(
            plane_distance_mm,
            identity.nominal_distance_mm,
        ),
        "plane_distance_temporal_std_mm": _optional_non_negative(
            plane_distance,
            "std_mm",
            "plane_distance",
        ),
        "tilt_median_deg": tilt_median,
        "tilt_temporal_std_deg": _optional_non_negative(
            tilt,
            "std_deg",
            "tilt",
        ),
        "tilt_error_from_nominal_deg": (
            _optional_offset(tilt_median, identity.yaw_deg)
            if identity.scene == "scene02"
            else None
        ),
        "plane_rmse_median_mm": _optional_non_negative(
            residual,
            "median_rmse_mm",
            "residual",
        ),
        "plane_rmse_p95_mm": _optional_non_negative(
            residual,
            "p95_rmse_mm",
            "residual",
        ),
        "plane_p95_abs_median_mm": _optional_non_negative(
            residual,
            "median_p95_abs_mm",
            "residual",
        ),
        "plane_inlier_ratio_median": _optional_ratio(
            inlier,
            "median",
            "inlier_ratio",
        ),
    }


def _condition_row(
    rows: Sequence[Mapping[str, object]],
) -> dict[str, object]:
    if len(rows) != len(REPEAT_INDICES):
        raise ValueError("Condition aggregation requires three repeats")
    first = rows[0]
    condition = first["condition"]
    identity_columns = (
        "scene",
        "condition",
        "target",
        "nominal_distance_mm",
        "yaw_deg",
        "roi_width",
        "roi_height",
        "roi_pixels",
    )
    for row in rows[1:]:
        for column in identity_columns:
            if row[column] != first[column]:
                raise ValueError(
                    f"Condition rows disagree on {column}: {condition}"
                )
    repeats = {row["repeat_index"] for row in rows}
    if repeats != set(REPEAT_INDICES):
        raise ValueError(
            f"Condition aggregation requires r01/r02/r03: {condition}"
        )

    frame_counts = [_row_int(row, "num_frames") for row in rows]
    result: dict[str, object] = {
        "scene": first["scene"],
        "condition": condition,
        "target": first["target"],
        "nominal_distance_mm": first["nominal_distance_mm"],
        "yaw_deg": first["yaw_deg"],
        "repeat_count": len(rows),
        "total_frames": sum(frame_counts),
        "min_frames_per_repeat": min(frame_counts),
        "max_frames_per_repeat": max(frame_counts),
        "roi_width": first["roi_width"],
        "roi_height": first["roi_height"],
        "roi_pixels": first["roi_pixels"],
    }
    for metric in AGGREGATED_METRICS:
        aggregate = _aggregate_metric([row[metric] for row in rows])
        result[f"{metric}_mean"] = aggregate.mean
        result[f"{metric}_repeat_std"] = aggregate.repeat_std
        result[f"{metric}_valid_count"] = aggregate.valid_count
    return result


def _aggregate_metric(values: Sequence[object]) -> MetricAggregate:
    finite: list[float] = []
    for value in values:
        if value is None:
            continue
        if isinstance(value, bool) or not isinstance(
            value,
            (int, float, np.integer, np.floating),
        ):
            raise TypeError("Aggregate metric values must be numeric or None")
        normalized = float(value)
        if np.isfinite(normalized):
            finite.append(normalized)

    valid_count = len(finite)
    if not finite:
        return MetricAggregate(
            mean=None,
            repeat_std=None,
            valid_count=0,
        )
    mean = float(np.mean(finite))
    repeat_std = (
        float(np.std(finite, ddof=1))
        if valid_count >= 2
        else None
    )
    return MetricAggregate(
        mean=mean,
        repeat_std=repeat_std,
        valid_count=valid_count,
    )


def _metric_coverage(
    condition_rows: Sequence[Mapping[str, object]],
) -> dict[str, object]:
    incomplete: list[dict[str, object]] = []
    applicable_count = 0
    for row in condition_rows:
        scene = _condition_text(row, "scene")
        condition = _condition_text(row, "condition")
        for metric in AGGREGATED_METRICS:
            if (
                metric == "tilt_error_from_nominal_deg"
                and scene != "scene02"
            ):
                continue
            applicable_count += 1
            valid_count = _condition_int(
                row,
                f"{metric}_valid_count",
            )
            if valid_count < len(REPEAT_INDICES):
                incomplete.append(
                    {
                        "condition": condition,
                        "metric": metric,
                        "valid_count": valid_count,
                    }
                )
    return {
        "expected_valid_count_per_condition": len(REPEAT_INDICES),
        "applicable_metric_condition_pairs": applicable_count,
        "complete_metric_condition_pairs": (
            applicable_count - len(incomplete)
        ),
        "incomplete": incomplete,
        "not_applicable": [
            {
                "scene": "scene01",
                "metric": "tilt_error_from_nominal_deg",
            },
            {
                "scene": "scene03",
                "metric": "tilt_error_from_nominal_deg",
            },
        ],
        "warnings": [
            (
                f"{item['condition']} has "
                f"{item['valid_count']}/3 valid repeat values for "
                f"{item['metric']}"
            )
            for item in incomplete
        ],
    }


def _distance_planarity_panels() -> tuple[PlotMetric, ...]:
    return (
        PlotMetric(
            "plane_distance_offset_from_nominal_mm",
            "Plane-distance offset",
            "Offset (mm)",
            zero_reference=True,
        ),
        PlotMetric(
            "plane_distance_temporal_std_mm",
            "Plane-distance temporal variation",
            "Standard deviation (mm)",
        ),
        PlotMetric(
            "plane_rmse_median_mm",
            "Median residual RMSE",
            "RMSE (mm)",
        ),
        PlotMetric(
            "plane_p95_abs_median_mm",
            "Median p95 absolute residual",
            "Residual (mm)",
        ),
        PlotMetric(
            "tilt_median_deg",
            "Fitted tilt",
            "Tilt (deg)",
        ),
        PlotMetric(
            "plane_inlier_ratio_median",
            "Median plane inlier ratio",
            "Ratio (%)",
            percent=True,
        ),
    )


def _scene_rows(
    comparison: BaselineComparison,
    scene: str,
) -> tuple[Mapping[str, object], ...]:
    rows = _rows_for_scene(build_condition_rows(comparison), scene)
    expected_count = 4 if scene == "scene03" else 5
    if len(rows) != expected_count:
        raise ValueError(
            f"{scene} comparison requires {expected_count} conditions"
        )
    return rows


def _rows_for_scene(
    rows: Sequence[Mapping[str, object]],
    scene: str,
) -> tuple[Mapping[str, object], ...]:
    return tuple(
        row
        for row in rows
        if _condition_text(row, "scene") == scene
    )


def _numeric_x(
    rows: Sequence[Mapping[str, object]],
    key: str,
) -> np.ndarray:
    return np.asarray(
        [_condition_int(row, key) for row in rows],
        dtype=np.float64,
    )


def _target_labels(
    rows: Sequence[Mapping[str, object]],
) -> tuple[str, ...]:
    return tuple(
        (
            "CBD"
            if _condition_text(row, "target") == "cbd"
            else _condition_text(row, "target").title()
        )
        for row in rows
    )


def _plot_condition_panels(
    rows: Sequence[Mapping[str, object]],
    *,
    x: np.ndarray,
    x_labels: Sequence[str] | None,
    x_label: str,
    title: str,
    panels: Sequence[PlotMetric],
):
    from matplotlib.figure import Figure

    if len(panels) not in {4, 6}:
        raise ValueError("Comparison figures require four or six panels")
    if x.shape != (len(rows),):
        raise ValueError("Plot x values must align with condition rows")

    figure = Figure(
        figsize=(12.0, 8.0 if len(panels) == 4 else 12.0),
        constrained_layout=True,
    )
    axes = figure.subplots(2, 2) if len(panels) == 4 else (
        figure.subplots(3, 2)
    )
    flat = np.asarray(axes, dtype=object).ravel()
    for axis, panel in zip(flat, panels, strict=True):
        means, errors = _metric_mean_and_error(
            rows,
            panel.metric,
            percent=panel.percent,
        )
        valid = np.isfinite(means)
        if np.any(valid):
            axis.plot(
                x[valid],
                means[valid],
                color="tab:blue",
                marker="o",
                label="Condition mean",
            )
            with_error = valid & np.isfinite(errors)
            if np.any(with_error):
                axis.errorbar(
                    x[with_error],
                    means[with_error],
                    yerr=errors[with_error],
                    color="tab:blue",
                    fmt="none",
                    capsize=4,
                    label="Repeat SD",
                )
            axis.legend(loc="best")
        else:
            axis.text(
                0.5,
                0.5,
                "No valid values",
                ha="center",
                va="center",
                transform=axis.transAxes,
            )
        if panel.zero_reference:
            axis.axhline(
                0.0,
                color="0.4",
                linestyle=":",
                label="Zero reference",
            )
        if panel.percent:
            axis.set_ylim(bottom=0.0)
        axis.set_title(panel.title)
        axis.set_ylabel(panel.ylabel)
        axis.set_xlabel(x_label)
        axis.grid(True, alpha=0.25)
        if x_labels is not None:
            axis.set_xticks(x, labels=x_labels, rotation=15)
    figure.suptitle(title)
    figure.text(
        0.5,
        0.005,
        (
            "Error bars show repeat-to-repeat sample SD (n ≤ 3), "
            "not confidence intervals. Missing metrics are not plotted."
        ),
        ha="center",
        fontsize=9,
    )
    return figure


def _metric_mean_and_error(
    rows: Sequence[Mapping[str, object]],
    metric: str,
    *,
    percent: bool,
) -> tuple[np.ndarray, np.ndarray]:
    scale = 100.0 if percent else 1.0
    means = np.asarray(
        [
            _condition_optional_float(row, f"{metric}_mean")
            * scale
            for row in rows
        ],
        dtype=np.float64,
    )
    errors = np.asarray(
        [
            _condition_optional_float(
                row,
                f"{metric}_repeat_std",
            )
            * scale
            for row in rows
        ],
        dtype=np.float64,
    )
    return means, errors


def _condition_optional_float(
    row: Mapping[str, object],
    key: str,
) -> float:
    value = row.get(key)
    if value is None:
        return float("nan")
    if isinstance(value, bool) or not isinstance(
        value,
        (int, float, np.integer, np.floating),
    ):
        raise TypeError(f"Condition row {key} must be numeric or None")
    result = float(value)
    if not np.isfinite(result):
        return float("nan")
    return result


def _condition_int(row: Mapping[str, object], key: str) -> int:
    value = row.get(key)
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError(f"Condition row {key} must be an integer")
    return value


def _condition_text(row: Mapping[str, object], key: str) -> str:
    value = row.get(key)
    if not isinstance(value, str) or not value:
        raise TypeError(f"Condition row {key} must be a non-empty string")
    return value


def _figure_png_bytes(figure: object) -> bytes:
    from matplotlib.figure import Figure

    if not isinstance(figure, Figure):
        raise TypeError("figure must be a Matplotlib Figure")
    stream = BytesIO()
    try:
        figure.savefig(stream, format="png", dpi=160)
        return stream.getvalue()
    finally:
        figure.clear()


def _build_csv_text(
    columns: Sequence[str],
    rows: Sequence[Mapping[str, object]],
) -> str:
    stream = StringIO(newline="")
    writer = csv.DictWriter(stream, fieldnames=columns)
    writer.writeheader()
    writer.writerows(rows)
    return stream.getvalue()


def _optional_scale(
    value: float | None,
    scale: float,
) -> float | None:
    if value is None:
        return None
    return float(value * scale)


def _optional_offset(
    value: float | None,
    nominal: int | None,
) -> float | None:
    if value is None or nominal is None:
        return None
    return float(value - nominal)


def _row_int(row: Mapping[str, object], key: str) -> int:
    value = row.get(key)
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError(f"Recording row {key} must be an integer")
    return value


def _summary_path(path: Path) -> str:
    resolved = path.resolve()
    try:
        return str(resolved.relative_to(PROJECT_ROOT))
    except ValueError:
        return str(resolved)


def _validate_comparison_result(
    comparison: BaselineComparison,
) -> BaselineComparison:
    if not isinstance(comparison, BaselineComparison):
        raise TypeError("comparison must be a BaselineComparison")
    return validate_baseline_comparison(comparison.records)


def _validate_record(
    record: BaselineSummaryRecord,
    *,
    temporal_map: np.ndarray,
    zero_ratio_map: np.ndarray,
    max_uint16_map: np.ndarray,
) -> None:
    summary = record.summary
    dataset = _nested_mapping(summary, "dataset")
    roi = _nested_mapping(summary, "roi")
    camera = _nested_mapping(summary, "depth_camera")
    preprocessing = _nested_mapping(summary, "depth_preprocessing")
    quality = _nested_mapping(summary, "depth_quality")
    temporal = _nested_mapping(summary, "temporal_noise")
    measured = _nested_mapping(summary, "measured_depth")
    planarity = _nested_mapping(summary, "planarity")

    experiment = _required_text(dataset, "experiment", "dataset")
    if experiment != record.identity.experiment:
        raise ValueError(
            "Summary experiment does not match result directory: "
            f"{experiment} != {record.identity.experiment}"
        )
    num_frames = _positive_int(dataset, "num_frames", "dataset")
    image_width = _positive_int(dataset, "width", "dataset")
    image_height = _positive_int(dataset, "height", "dataset")

    roi_key = _required_text(roi, "key", "roi")
    if roi_key != record.identity.condition:
        raise ValueError(
            f"ROI key does not match condition: {roi_key}"
        )
    roi_x = _non_negative_int(roi, "x", "roi")
    roi_y = _non_negative_int(roi, "y", "roi")
    roi_width = _positive_int(roi, "width", "roi")
    roi_height = _positive_int(roi, "height", "roi")
    roi_pixels = _positive_int(roi, "pixel_count", "roi")
    if roi_width * roi_height != roi_pixels:
        raise ValueError("ROI pixel_count does not match width x height")
    if roi_x + roi_width > image_width or roi_y + roi_height > image_height:
        raise ValueError("ROI exceeds dataset image bounds")
    _validate_roi_config(record, roi)

    camera_width = _positive_int(camera, "width", "depth_camera")
    camera_height = _positive_int(camera, "height", "depth_camera")
    if (camera_width, camera_height) != (image_width, image_height):
        raise ValueError("Depth-camera resolution does not match dataset")
    _required_text(camera, "config", "depth_camera")
    _required_text(camera, "frame_id", "depth_camera")
    for key in ("fx", "fy"):
        if _finite_float(camera, key, "depth_camera") <= 0:
            raise ValueError(f"depth_camera.{key} must be positive")
    for key in ("cx", "cy"):
        _finite_float(camera, key, "depth_camera")

    excluded = preprocessing.get("excluded_raw_values")
    if excluded != [0, 65535]:
        raise ValueError(
            "depth_preprocessing.excluded_raw_values must be [0, 65535]"
        )
    if _finite_float(
        preprocessing,
        "depth_scale_to_mm",
        "depth_preprocessing",
    ) <= 0:
        raise ValueError("depth scale must be positive")

    _optional_ratio(quality, "zero_ratio", "depth_quality")
    max_uint16 = _nested_mapping(quality, "max_uint16")
    _optional_ratio(max_uint16, "ratio", "depth_quality.max_uint16")
    affected_frames = _non_negative_int(
        max_uint16,
        "affected_frames",
        "depth_quality.max_uint16",
    )
    if affected_frames > num_frames:
        raise ValueError("max_uint16 affected_frames exceeds num_frames")
    max_pixels = _non_negative_int(
        max_uint16,
        "max_pixels_per_frame",
        "depth_quality.max_uint16",
    )
    if max_pixels > roi_pixels:
        raise ValueError("max_uint16 max_pixels_per_frame exceeds ROI")

    _ratio(temporal, "min_valid_ratio", "temporal_noise")
    for key in ("median_std_mm", "mean_std_mm", "p95_std_mm"):
        _optional_non_negative(temporal, key, "temporal_noise")
    for key in ("median_mm", "mean_mm", "p05_mm", "p95_mm"):
        _optional_non_negative(measured, key, "measured_depth")
    _optional_non_negative(measured, "std_mm", "measured_depth")

    _required_text(planarity, "fitting_method", "planarity")
    _non_negative_float(
        planarity,
        "inlier_threshold_mm",
        "planarity",
    )
    _positive_int(planarity, "min_valid_points", "planarity")
    successful = _non_negative_int(
        planarity,
        "successful_frames",
        "planarity",
    )
    failed = _non_negative_int(
        planarity,
        "failed_frames",
        "planarity",
    )
    if successful + failed != num_frames:
        raise ValueError(
            "Planarity successful_frames + failed_frames must equal "
            "num_frames"
        )
    plane_distance = _nested_mapping(planarity, "plane_distance")
    tilt = _nested_mapping(planarity, "tilt")
    residual = _nested_mapping(planarity, "residual")
    inlier = _nested_mapping(planarity, "inlier_ratio")
    _optional_non_negative(plane_distance, "median_m", "plane_distance")
    _optional_non_negative(plane_distance, "std_mm", "plane_distance")
    _optional_non_negative(tilt, "median_deg", "tilt")
    _optional_non_negative(tilt, "std_deg", "tilt")
    for key in ("median_rmse_mm", "p95_rmse_mm", "median_p95_abs_mm"):
        _optional_non_negative(residual, key, "residual")
    _optional_ratio(inlier, "median", "inlier_ratio")

    _validate_frame_tables(
        record,
        num_frames=num_frames,
        successful_frames=successful,
    )
    expected_shape = (roi_height, roi_width)
    _validate_map(
        temporal_map,
        expected_shape,
        record.temporal_map_path,
        ratio=False,
        allow_nan=True,
    )
    _validate_map(
        zero_ratio_map,
        expected_shape,
        record.zero_ratio_map_path,
        ratio=True,
        allow_nan=False,
    )
    _validate_map(
        max_uint16_map,
        expected_shape,
        record.max_uint16_map_path,
        ratio=True,
        allow_nan=False,
    )


def _validate_roi_config(
    record: BaselineSummaryRecord,
    roi: Mapping[str, object],
) -> None:
    config_text = _required_text(roi, "config", "roi")
    config_path = _resolve_project_path(config_text)
    if not config_path.is_file():
        raise FileNotFoundError(
            f"ROI configuration not found: {config_path}"
        )
    with config_path.open("r", encoding="utf-8") as stream:
        config = _require_mapping(
            yaml.safe_load(stream),
            f"ROI configuration {config_path}",
        )
    if _required_text(config, "name", "ROI configuration") != (
        record.identity.condition
    ):
        raise ValueError("ROI configuration name does not match condition")
    rectangle = _nested_mapping(config, "roi")
    if _required_text(rectangle, "type", "ROI rectangle") != "rectangle":
        raise ValueError("ROI configuration must contain a rectangle")
    for key in ("x", "y", "width", "height"):
        if _required_int(rectangle, key, "ROI rectangle") != (
            _required_int(roi, key, "roi")
        ):
            raise ValueError(
                f"ROI configuration {key} does not match summary"
            )


def _validate_frame_tables(
    record: BaselineSummaryRecord,
    *,
    num_frames: int,
    successful_frames: int,
) -> None:
    if len(record.frame_medians) != num_frames:
        raise ValueError(
            "Frame-median CSV row count does not match num_frames"
        )
    if len(record.frame_planes) != num_frames:
        raise ValueError(
            "Frame-plane CSV row count does not match num_frames"
        )

    successful_count = 0
    success_float_columns = (
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
    for expected_index, (median_row, plane_row) in enumerate(
        zip(record.frame_medians, record.frame_planes, strict=True)
    ):
        median_index = _csv_int(
            median_row,
            "frame_index",
            record.frame_median_path,
        )
        plane_index = _csv_int(
            plane_row,
            "frame_index",
            record.frame_plane_path,
        )
        if median_index != expected_index or plane_index != expected_index:
            raise ValueError("Frame CSV indices must be sequential from zero")
        median_timestamp = _csv_int(
            median_row,
            "timestamp_ns",
            record.frame_median_path,
        )
        plane_timestamp = _csv_int(
            plane_row,
            "timestamp_ns",
            record.frame_plane_path,
        )
        if median_timestamp != plane_timestamp:
            raise ValueError("Frame CSV timestamps are not aligned")
        median_depth = median_row["median_depth_mm"].strip()
        if median_depth:
            _csv_non_negative_float(
                median_depth,
                "median_depth_mm",
                record.frame_median_path,
            )

        fit_text = plane_row["fit_succeeded"].strip().lower()
        if fit_text not in {"true", "false"}:
            raise ValueError(
                "fit_succeeded must contain true or false: "
                f"{record.frame_plane_path}"
            )
        _csv_non_negative_int(
            plane_row,
            "valid_points",
            record.frame_plane_path,
        )
        if fit_text == "true":
            successful_count += 1
            for column in success_float_columns:
                value = _csv_float(
                    plane_row,
                    column,
                    record.frame_plane_path,
                )
                if column in {
                    "plane_distance_m",
                    "tilt_deg",
                    "residual_rmse_mm",
                    "residual_std_mm",
                    "residual_p95_abs_mm",
                } and value < 0:
                    raise ValueError(
                        f"{column} must be non-negative: "
                        f"{record.frame_plane_path}"
                    )
                if column == "inlier_ratio" and not 0 <= value <= 1:
                    raise ValueError(
                        f"inlier_ratio must be in [0, 1]: "
                        f"{record.frame_plane_path}"
                    )
    if successful_count != successful_frames:
        raise ValueError(
            "Frame-plane fit_succeeded count does not match summary"
        )


def _validate_map(
    array: np.ndarray,
    expected_shape: tuple[int, int],
    path: Path,
    *,
    ratio: bool,
    allow_nan: bool,
) -> None:
    if array.shape != expected_shape:
        raise ValueError(
            f"Map shape does not match ROI for {path}: "
            f"{array.shape} != {expected_shape}"
        )
    if not np.issubdtype(array.dtype, np.number):
        raise ValueError(f"Map must use a numeric dtype: {path}")
    if np.any(np.isinf(array)):
        raise ValueError(f"Map contains infinite values: {path}")
    if not allow_nan and np.any(np.isnan(array)):
        raise ValueError(f"Map contains NaN values: {path}")
    finite = array[np.isfinite(array)]
    if np.any(finite < 0):
        raise ValueError(f"Map contains negative values: {path}")
    if ratio and np.any(finite > 1):
        raise ValueError(f"Ratio map contains values above one: {path}")


def _roi_signature(record: BaselineSummaryRecord) -> tuple[object, ...]:
    roi = _nested_mapping(record.summary, "roi")
    return (
        _required_text(roi, "key", "roi"),
        _required_text(roi, "config", "roi"),
        _required_int(roi, "x", "roi"),
        _required_int(roi, "y", "roi"),
        _required_int(roi, "width", "roi"),
        _required_int(roi, "height", "roi"),
        _required_int(roi, "pixel_count", "roi"),
    )


def _common_analysis_signature(
    record: BaselineSummaryRecord,
) -> tuple[object, ...]:
    summary = record.summary
    camera = _nested_mapping(summary, "depth_camera")
    preprocessing = _nested_mapping(summary, "depth_preprocessing")
    temporal = _nested_mapping(summary, "temporal_noise")
    planarity = _nested_mapping(summary, "planarity")
    excluded = preprocessing.get("excluded_raw_values")
    if not isinstance(excluded, list):
        raise ValueError("excluded_raw_values must be a list")
    return (
        _required_text(camera, "config", "depth_camera"),
        _required_text(camera, "frame_id", "depth_camera"),
        _required_int(camera, "width", "depth_camera"),
        _required_int(camera, "height", "depth_camera"),
        _finite_float(camera, "fx", "depth_camera"),
        _finite_float(camera, "fy", "depth_camera"),
        _finite_float(camera, "cx", "depth_camera"),
        _finite_float(camera, "cy", "depth_camera"),
        tuple(excluded),
        _finite_float(
            preprocessing,
            "depth_scale_to_mm",
            "depth_preprocessing",
        ),
        _finite_float(temporal, "min_valid_ratio", "temporal_noise"),
        _required_text(planarity, "fitting_method", "planarity"),
        _finite_float(
            planarity,
            "inlier_threshold_mm",
            "planarity",
        ),
        _required_int(planarity, "min_valid_points", "planarity"),
    )


def _load_csv(
    path: Path,
    required_columns: frozenset[str],
    label: str,
) -> tuple[Mapping[str, str], ...]:
    with path.open("r", encoding="utf-8", newline="") as stream:
        reader = csv.DictReader(stream)
        if reader.fieldnames is None:
            raise ValueError(f"{label} CSV has no header: {path}")
        missing = required_columns.difference(reader.fieldnames)
        if missing:
            raise ValueError(
                f"{label} CSV is missing columns: "
                + ", ".join(sorted(missing))
            )
        return tuple(dict(row) for row in reader)


def _load_map(path: Path) -> np.ndarray:
    try:
        return np.load(path, allow_pickle=False)
    except (OSError, ValueError) as error:
        raise ValueError(f"Unable to load metric map: {path}") from error


def _condition_name(experiment: str) -> str:
    return re.sub(r"_r\d{2}$", "", experiment)


def _resolve_project_path(value: str) -> Path:
    path = Path(value).expanduser()
    if path.is_absolute():
        return path
    return PROJECT_ROOT / path


def _nested_mapping(
    mapping: Mapping[str, object],
    key: str,
) -> Mapping[str, object]:
    return _require_mapping(mapping.get(key), key)


def _require_mapping(value: object, label: str) -> Mapping[str, object]:
    if not isinstance(value, Mapping):
        raise ValueError(f"{label} must be a mapping")
    return value


def _required_text(
    mapping: Mapping[str, object],
    key: str,
    label: str,
) -> str:
    value = mapping.get(key)
    if not isinstance(value, str) or not value:
        raise ValueError(f"{label}.{key} must be a non-empty string")
    return value


def _required_int(
    mapping: Mapping[str, object],
    key: str,
    label: str,
) -> int:
    value = mapping.get(key)
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError(f"{label}.{key} must be an integer")
    return value


def _positive_int(
    mapping: Mapping[str, object],
    key: str,
    label: str,
) -> int:
    value = _required_int(mapping, key, label)
    if value <= 0:
        raise ValueError(f"{label}.{key} must be positive")
    return value


def _non_negative_int(
    mapping: Mapping[str, object],
    key: str,
    label: str,
) -> int:
    value = _required_int(mapping, key, label)
    if value < 0:
        raise ValueError(f"{label}.{key} must be non-negative")
    return value


def _finite_float(
    mapping: Mapping[str, object],
    key: str,
    label: str,
) -> float:
    value = mapping.get(key)
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{label}.{key} must be numeric")
    result = float(value)
    if not np.isfinite(result):
        raise ValueError(f"{label}.{key} must be finite")
    return result


def _non_negative_float(
    mapping: Mapping[str, object],
    key: str,
    label: str,
) -> float:
    value = _finite_float(mapping, key, label)
    if value < 0:
        raise ValueError(f"{label}.{key} must be non-negative")
    return value


def _ratio(
    mapping: Mapping[str, object],
    key: str,
    label: str,
) -> float:
    value = _finite_float(mapping, key, label)
    if not 0 <= value <= 1:
        raise ValueError(f"{label}.{key} must be in [0, 1]")
    return value


def _optional_non_negative(
    mapping: Mapping[str, object],
    key: str,
    label: str,
) -> float | None:
    if key not in mapping:
        raise ValueError(f"{label}.{key} is required")
    if mapping[key] is None:
        return None
    return _non_negative_float(mapping, key, label)


def _optional_ratio(
    mapping: Mapping[str, object],
    key: str,
    label: str,
) -> float | None:
    if key not in mapping:
        raise ValueError(f"{label}.{key} is required")
    if mapping[key] is None:
        return None
    return _ratio(mapping, key, label)


def _csv_int(
    row: Mapping[str, str],
    key: str,
    path: Path,
) -> int:
    try:
        return int(row[key])
    except (KeyError, TypeError, ValueError) as error:
        raise ValueError(f"Invalid integer {key} in {path}") from error


def _csv_non_negative_int(
    row: Mapping[str, str],
    key: str,
    path: Path,
) -> int:
    value = _csv_int(row, key, path)
    if value < 0:
        raise ValueError(f"{key} must be non-negative: {path}")
    return value


def _csv_float(
    row: Mapping[str, str],
    key: str,
    path: Path,
) -> float:
    try:
        value = float(row[key])
    except (KeyError, TypeError, ValueError) as error:
        raise ValueError(f"Invalid float {key} in {path}") from error
    if not np.isfinite(value):
        raise ValueError(f"Non-finite float {key} in {path}")
    return value


def _csv_non_negative_float(
    value: str,
    key: str,
    path: Path,
) -> float:
    try:
        result = float(value)
    except ValueError as error:
        raise ValueError(f"Invalid float {key} in {path}") from error
    if not np.isfinite(result) or result < 0:
        raise ValueError(f"{key} must be finite and non-negative: {path}")
    return result
