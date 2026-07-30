"""Analyze one Scene 04 depth-discontinuity dataset."""

import argparse
from collections.abc import Sequence
import csv
from dataclasses import dataclass, replace
from io import BytesIO, StringIO
from numbers import Integral
from pathlib import Path
import re
import sys

import cv2
import numpy as np
import yaml


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_ROI_ROOT = PROJECT_ROOT / "config" / "roi"
DEFAULT_RESULTS_ROOT = PROJECT_ROOT / "results"

FRAME_METRICS_FILENAME = "frame_edge_metrics.csv"
PROFILE_FILENAME = "aggregate_edge_profile.csv"
SUMMARY_FILENAME = "summary.yaml"
LABEL_MAP_FILENAME = "representative_label_map.npy"
ROI_OVERLAY_FILENAME = "roi_overlay.png"
LABEL_OVERLAY_FILENAME = "label_overlay.png"
PROFILE_PLOT_FILENAME = "edge_probability_profile.png"
TEMPORAL_PLOT_FILENAME = "temporal_edge_metrics.png"

_EDGE_EXPERIMENT_PATTERN = re.compile(
    r"^scene04_"
    r"gap(?P<gap_cm>\d+)_"
    r"(?P<orientation>horizon|horizontal|vertical)_"
    r"(?P<target>[a-z0-9]+(?:_[a-z0-9]+)*)_"
    r"d(?P<foreground_cm>\d+)_"
    r"r(?P<repeat>\d+)$"
)

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.geometry.edge_geometry import (  # noqa: E402
    compute_signed_distance_map,
)
from src.io.dataset import DepthDataset  # noqa: E402
from src.metrics.edge_discontinuity import (  # noqa: E402
    DistanceProfileResult,
    EdgeDiscontinuityResult,
    EdgeFrameAnalysis,
    aggregate_edge_dataset,
    analyze_edge_frame,
)
from src.preprocessing.depth import prepare_depth  # noqa: E402
from src.preprocessing.edge_roi import (  # noqa: E402
    EdgeROIConfig,
    load_edge_roi_config,
    validate_edge_roi_config,
)
from src.preprocessing.roi import (  # noqa: E402
    derive_roi_key,
    get_roi_path,
)


@dataclass(frozen=True)
class EdgeAnalysisInput:
    """Store one loaded Scene 04 dataset and edge ROI configuration."""

    experiment_name: str
    dataset_dir: Path
    dataset_path: Path

    roi_key: str
    roi_path: Path

    dataset: DepthDataset
    config: EdgeROIConfig


@dataclass(frozen=True)
class EdgeMetricResults:
    """Store aggregate metrics and one retained diagnostic frame."""

    discontinuity: EdgeDiscontinuityResult
    representative_target_frame_index: int
    representative_analysis: EdgeFrameAnalysis | None


@dataclass(frozen=True)
class EdgeExperimentMetadata:
    """Store physical setup metadata parsed from an experiment name."""

    orientation_token: str
    orientation: str
    target: str
    nominal_foreground_distance_mm: int
    nominal_gap_mm: int
    nominal_background_distance_mm: int
    distance_reference: str
    repeat_index: int


@dataclass(frozen=True)
class EdgeAnalysisResult:
    """Store loaded input metadata and computed Scene 04 metrics."""

    source: EdgeAnalysisInput
    metadata: EdgeExperimentMetadata
    metrics: EdgeMetricResults


def parse_args(
    argv: Sequence[str] | None = None,
) -> argparse.Namespace:
    """Parse command-line arguments for Scene 04 edge analysis."""
    parser = argparse.ArgumentParser(
        description=(
            "Analyze one extracted Scene 04 depth-discontinuity "
            "dataset."
        )
    )
    parser.add_argument(
        "dataset_dir",
        type=Path,
        help="Experiment directory containing depth.npz.",
    )
    parser.add_argument(
        "--roi-root",
        type=Path,
        default=DEFAULT_ROI_ROOT,
        help=(
            "Edge ROI configuration directory "
            "(default: config/roi)."
        ),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        help=(
            "Artifact output directory. Defaults to "
            "results/<experiment>/edge_discontinuity."
        ),
    )
    parser.add_argument(
        "--frame-index",
        type=int,
        help=(
            "Preferred representative frame. Defaults to the "
            "middle frame; the nearest valid frame is used if needed."
        ),
    )
    return parser.parse_args(argv)


def parse_edge_experiment_name(
    experiment_name: str,
) -> EdgeExperimentMetadata:
    """Parse the controlled Scene 04 experiment naming convention."""
    if not isinstance(experiment_name, str):
        raise TypeError("experiment_name must be a string")

    match = _EDGE_EXPERIMENT_PATTERN.fullmatch(
        experiment_name
    )
    if match is None:
        raise ValueError(
            "Scene 04 experiment name must match "
            "'scene04_gap<cm>_<horizon|horizontal|vertical>_"
            "<target>_d<foreground-cm>_r<repeat>'"
        )

    gap_cm = int(match.group("gap_cm"))
    foreground_cm = int(match.group("foreground_cm"))
    repeat_index = int(match.group("repeat"))
    if gap_cm <= 0:
        raise ValueError(
            "Scene 04 nominal gap must be positive"
        )
    if foreground_cm <= 0:
        raise ValueError(
            "Scene 04 foreground distance must be positive"
        )
    if repeat_index <= 0:
        raise ValueError(
            "Scene 04 repeat index must be positive"
        )

    orientation_token = match.group("orientation")
    orientation = (
        "horizontal"
        if orientation_token in {"horizon", "horizontal"}
        else "vertical"
    )
    foreground_mm = foreground_cm * 10
    gap_mm = gap_cm * 10
    return EdgeExperimentMetadata(
        orientation_token=orientation_token,
        orientation=orientation,
        target=match.group("target"),
        nominal_foreground_distance_mm=foreground_mm,
        nominal_gap_mm=gap_mm,
        nominal_background_distance_mm=(
            foreground_mm + gap_mm
        ),
        distance_reference="camera_optical_reference_plane",
        repeat_index=repeat_index,
    )


def load_edge_input(
    dataset_dir: Path,
    roi_root: Path = DEFAULT_ROI_ROOT,
) -> EdgeAnalysisInput:
    """Load and validate one Scene 04 dataset and shared ROI config."""
    resolved_dataset_dir = Path(dataset_dir).expanduser()
    resolved_roi_root = Path(roi_root).expanduser()
    experiment_name = resolved_dataset_dir.name
    if not experiment_name:
        raise ValueError(
            f"Cannot derive experiment name from {resolved_dataset_dir}"
        )

    dataset_path = resolved_dataset_dir / "depth.npz"
    if not dataset_path.is_file():
        raise FileNotFoundError(
            f"Cannot find dataset file {dataset_path}"
        )

    roi_key = derive_roi_key(experiment_name)
    roi_path = get_roi_path(
        resolved_roi_root,
        experiment_name,
    )
    if not roi_path.is_file():
        raise FileNotFoundError(
            f"Edge ROI configuration not found: {roi_path}\n\n"
            "Run:\n"
            f"python3 tools/select_edge_roi.py {resolved_dataset_dir}"
        )

    dataset = DepthDataset.load(dataset_path)
    if dataset.num_frames == 0:
        raise ValueError(
            f"Dataset {dataset_path} contains no depth frames"
        )

    config = load_edge_roi_config(roi_path)
    if config.name != roi_key:
        raise ValueError(
            "Edge ROI config name does not match dataset ROI key: "
            f"config name {config.name!r}, expected {roi_key!r}"
        )

    source_roi_key = derive_roi_key(
        config.source_experiment
    )
    if source_roi_key != roi_key:
        raise ValueError(
            "Edge ROI config source experiment does not match "
            "dataset ROI key: "
            f"source key {source_roi_key!r}, expected {roi_key!r}"
        )

    validate_edge_roi_config(
        config,
        image_shape=(dataset.height, dataset.width),
    )

    return EdgeAnalysisInput(
        experiment_name=experiment_name,
        dataset_dir=resolved_dataset_dir,
        dataset_path=dataset_path,
        roi_key=roi_key,
        roi_path=roi_path,
        dataset=dataset,
        config=config,
    )


def compute_edge_metrics(
    raw_depth: np.ndarray,
    config: EdgeROIConfig,
    representative_frame_index: int | None = None,
) -> EdgeMetricResults:
    """Compute all Scene 04 frame metrics without retaining every label map."""
    _validate_raw_depth(raw_depth)
    validate_edge_roi_config(
        config,
        image_shape=raw_depth.shape[1:],
    )

    target_frame_index = _resolve_representative_target(
        num_frames=raw_depth.shape[0],
        requested_frame_index=representative_frame_index,
    )
    signed_distance_map = compute_signed_distance_map(
        image_shape=raw_depth.shape[1:],
        line=config.nominal_edge,
        foreground_side=config.foreground_side,
    )

    compact_analyses: list[EdgeFrameAnalysis] = []
    representative_analysis: EdgeFrameAnalysis | None = None
    representative_key: tuple[int, int] | None = None

    for frame_index in range(raw_depth.shape[0]):
        prepared_frame = prepare_depth(
            raw_depth[frame_index:frame_index + 1]
        )[0]
        analysis = analyze_edge_frame(
            frame_index=frame_index,
            prepared_depth_frame=prepared_frame,
            config=config,
            signed_distance_map=signed_distance_map,
        )

        if (
            analysis.result.analysis_status == "ok"
            and analysis.label_map is not None
        ):
            candidate_key = (
                abs(frame_index - target_frame_index),
                frame_index,
            )
            if (
                representative_key is None
                or candidate_key < representative_key
            ):
                representative_key = candidate_key
                representative_analysis = analysis

        # Dataset aggregation needs scalar results and profiles, not every
        # full-resolution label map. Retain only the selected diagnostic map.
        compact_analyses.append(
            replace(
                analysis,
                label_map=None,
            )
        )

    discontinuity = aggregate_edge_dataset(
        compact_analyses
    )
    return EdgeMetricResults(
        discontinuity=discontinuity,
        representative_target_frame_index=target_frame_index,
        representative_analysis=representative_analysis,
    )


def analyze_edge(
    dataset_dir: Path,
    roi_root: Path = DEFAULT_ROI_ROOT,
    representative_frame_index: int | None = None,
) -> EdgeAnalysisResult:
    """Load one Scene 04 dataset and compute its edge metrics."""
    source = load_edge_input(
        dataset_dir=dataset_dir,
        roi_root=roi_root,
    )
    metadata = parse_edge_experiment_name(
        source.experiment_name
    )
    metrics = compute_edge_metrics(
        raw_depth=source.dataset.depth,
        config=source.config,
        representative_frame_index=representative_frame_index,
    )
    return EdgeAnalysisResult(
        source=source,
        metadata=metadata,
        metrics=metrics,
    )


def build_summary(
    result: EdgeAnalysisResult,
) -> dict[str, object]:
    """Build a YAML-safe summary for one Scene 04 analysis."""
    _validate_analysis_result(result)
    source = result.source
    metadata = result.metadata
    config = source.config
    discontinuity = result.metrics.discontinuity
    representative = result.metrics.representative_analysis

    valid_results = [
        frame
        for frame in discontinuity.frame_results
        if frame.analysis_status == "ok"
    ]
    foreground_values = [
        frame.foreground_reference_mm
        for frame in valid_results
    ]
    background_values = [
        frame.background_reference_mm
        for frame in valid_results
    ]
    gap_values = [
        (
            frame.background_reference_mm
            - frame.foreground_reference_mm
        )
        for frame in valid_results
        if (
            np.isfinite(frame.foreground_reference_mm)
            and np.isfinite(frame.background_reference_mm)
        )
    ]
    foreground_median = _finite_median_or_none(
        foreground_values
    )
    background_median = _finite_median_or_none(
        background_values
    )
    gap_median = _finite_median_or_none(gap_values)
    gap_error = (
        None
        if gap_median is None
        else (
            gap_median
            - metadata.nominal_gap_mm
        )
    )

    return {
        "dataset": {
            "experiment": source.experiment_name,
            "path": _summary_path(source.dataset_path),
            "num_frames": int(source.dataset.num_frames),
            "width": int(source.dataset.width),
            "height": int(source.dataset.height),
        },
        "setup": {
            "orientation_token": metadata.orientation_token,
            "orientation": metadata.orientation,
            "target": metadata.target,
            "repeat_index": metadata.repeat_index,
            "distance_reference": metadata.distance_reference,
            "nominal_foreground_distance_mm": (
                metadata.nominal_foreground_distance_mm
            ),
            "nominal_gap_mm": metadata.nominal_gap_mm,
            "nominal_background_distance_mm": (
                metadata.nominal_background_distance_mm
            ),
        },
        "roi": {
            "key": source.roi_key,
            "config": _summary_path(source.roi_path),
            "source_experiment": config.source_experiment,
            "source_frame_index": config.source_frame_index,
            "foreground_pixels": (
                config.foreground_roi.pixel_count
            ),
            "background_pixels": (
                config.background_roi.pixel_count
            ),
            "edge_pixels": config.edge_roi.pixel_count,
        },
        "edge_geometry": {
            "nominal_edge_p1": [
                float(value)
                for value in config.nominal_edge.p1
            ],
            "nominal_edge_p2": [
                float(value)
                for value in config.nominal_edge.p2
            ],
            "foreground_side": config.foreground_side,
            "distance_bin_px": config.distance_bin_px,
            "max_edge_distance_px": (
                config.max_edge_distance_px
            ),
        },
        "analysis_parameters": {
            "reference": {
                "minimum_tolerance_mm": (
                    config.reference.minimum_tolerance_mm
                ),
                "mad_scale": config.reference.mad_scale,
                "minimum_valid_ratio": (
                    config.reference.minimum_valid_ratio
                ),
                "minimum_valid_count": (
                    config.reference.minimum_valid_count
                ),
            },
            "bleeding_probability_threshold": (
                config.bleeding.probability_threshold
            ),
            "invalid_ratio_threshold": (
                config.invalid.ratio_threshold
            ),
            "transition_high_probability": (
                config.transition.high_probability
            ),
            "transition_low_probability": (
                config.transition.low_probability
            ),
        },
        "depth_preprocessing": {
            "excluded_raw_values": [
                0,
                int(np.iinfo(np.uint16).max),
            ],
            "depth_scale_to_mm": 1.0,
        },
        "representative": {
            "target_frame_index": (
                result.metrics.representative_target_frame_index
            ),
            "selected_frame_index": (
                None
                if representative is None
                else representative.result.frame_index
            ),
            "available": representative is not None,
        },
        "reference_depth": {
            "foreground_median_mm": foreground_median,
            "background_median_mm": background_median,
            "measured_gap_median_mm": gap_median,
            "nominal_gap_mm": metadata.nominal_gap_mm,
            "gap_error_mm": gap_error,
        },
        "edge_quality": {
            "foreground_bleeding_ratio_median": (
                _finite_or_none(
                    discontinuity
                    .median_foreground_bleeding_ratio
                )
            ),
            "background_bleeding_ratio_median": (
                _finite_or_none(
                    discontinuity
                    .median_background_bleeding_ratio
                )
            ),
            "mixed_ratio_median": _finite_or_none(
                discontinuity.median_mixed_ratio
            ),
            "outlier_ratio_median": _finite_or_none(
                discontinuity.median_outlier_ratio
            ),
            "invalid_ratio_median": _finite_or_none(
                discontinuity.median_invalid_ratio
            ),
        },
        "transition": {
            "width_median_px": _finite_or_none(
                discontinuity.median_transition_width_px
            ),
            "width_p95_px": _finite_or_none(
                discontinuity.transition_width_p95_px
            ),
            "nominal_offset_median_px": _finite_or_none(
                discontinuity.median_nominal_edge_offset_px
            ),
            "nominal_offset_std_px": _finite_or_none(
                discontinuity.nominal_edge_offset_std_px
            ),
        },
        "frames": {
            "valid": discontinuity.valid_frames,
            "rejected": discontinuity.rejected_frames,
            "transition_valid": (
                discontinuity.valid_transition_frames
            ),
            "transition_failed": (
                discontinuity.failed_transition_frames
            ),
        },
        "diagnostics": {
            "aggregate_profile_available": (
                discontinuity.aggregate_profile is not None
            ),
            "representative_label_map_available": (
                representative is not None
                and representative.label_map is not None
            ),
        },
    }


def build_frame_edge_metrics_csv(
    result: EdgeAnalysisResult,
) -> str:
    """Build timestamp-aligned per-frame edge metrics as CSV text."""
    _validate_analysis_result(result)
    frame_results = result.metrics.discontinuity.frame_results
    timestamps_ns = result.source.dataset.timestamps_ns
    num_frames = result.source.dataset.num_frames
    if len(frame_results) != num_frames:
        raise ValueError(
            "frame result count does not match dataset frame count"
        )
    if timestamps_ns.shape != (num_frames,):
        raise ValueError(
            "timestamps_ns must match dataset frame count"
        )

    stream = StringIO(newline="")
    writer = csv.writer(stream)
    writer.writerow(
        [
            "frame_index",
            "timestamp_ns",
            "foreground_reference_mm",
            "background_reference_mm",
            "measured_gap_mm",
            "foreground_bleeding_ratio",
            "foreground_bleeding_max_distance_px",
            "background_bleeding_ratio",
            "background_bleeding_max_distance_px",
            "mixed_ratio",
            "peak_mixed_ratio",
            "peak_mixed_distance_px",
            "outlier_ratio",
            "invalid_ratio",
            "invalid_band_width_px",
            "transition_width_px",
            "nominal_edge_offset_px",
            "analysis_status",
            "transition_status",
        ]
    )
    for expected_index, frame in enumerate(frame_results):
        if frame.frame_index != expected_index:
            raise ValueError(
                "frame results must have contiguous indices "
                "starting at zero"
            )
        measured_gap = (
            frame.background_reference_mm
            - frame.foreground_reference_mm
        )
        writer.writerow(
            [
                frame.frame_index,
                int(timestamps_ns[expected_index]),
                _csv_float_or_blank(
                    frame.foreground_reference_mm
                ),
                _csv_float_or_blank(
                    frame.background_reference_mm
                ),
                _csv_float_or_blank(measured_gap),
                _csv_float_or_blank(
                    frame.foreground_bleeding_ratio
                ),
                _csv_float_or_blank(
                    frame.foreground_bleeding_max_distance_px
                ),
                _csv_float_or_blank(
                    frame.background_bleeding_ratio
                ),
                _csv_float_or_blank(
                    frame.background_bleeding_max_distance_px
                ),
                _csv_float_or_blank(frame.mixed_ratio),
                _csv_float_or_blank(frame.peak_mixed_ratio),
                _csv_float_or_blank(
                    frame.peak_mixed_distance_px
                ),
                _csv_float_or_blank(frame.outlier_ratio),
                _csv_float_or_blank(frame.invalid_ratio),
                _csv_float_or_blank(
                    frame.invalid_band_width_px
                ),
                _csv_float_or_blank(
                    frame.transition_width_px
                ),
                _csv_float_or_blank(
                    frame.nominal_edge_offset_px
                ),
                frame.analysis_status,
                frame.transition_status,
            ]
        )
    return stream.getvalue()


def build_aggregate_edge_profile_csv(
    profile: DistanceProfileResult | None,
) -> str | None:
    """Build one aggregate signed-distance profile as CSV text."""
    if profile is None:
        return None
    fields = tuple(
        DistanceProfileResult.__dataclass_fields__
    )
    arrays: dict[str, np.ndarray] = {}
    expected_length: int | None = None
    for field_name in fields:
        value = getattr(profile, field_name)
        if not isinstance(value, np.ndarray) or value.ndim != 1:
            raise ValueError(
                f"profile {field_name} must be one-dimensional"
            )
        if expected_length is None:
            expected_length = len(value)
        elif len(value) != expected_length:
            raise ValueError(
                "all profile arrays must have the same length"
            )
        arrays[field_name] = value

    stream = StringIO(newline="")
    writer = csv.writer(stream)
    writer.writerow(fields)
    assert expected_length is not None
    for index in range(expected_length):
        row: list[int | float | str] = []
        for field_name in fields:
            value = arrays[field_name][index]
            if field_name.endswith("_count"):
                row.append(int(value))
            else:
                row.append(
                    _csv_float_or_blank(float(value))
                )
        writer.writerow(row)
    return stream.getvalue()


def build_edge_artifacts(
    result: EdgeAnalysisResult,
) -> dict[str, bytes]:
    """Serialize every available analysis artifact in memory."""
    from src.visualization.edge import (
        depth_to_edge_display,
        draw_edge_roi_overlay,
        plot_edge_label_map,
        plot_edge_probability_profile,
        plot_edge_temporal_metrics,
    )

    _validate_analysis_result(result)
    source = result.source
    discontinuity = result.metrics.discontinuity
    representative = result.metrics.representative_analysis

    artifacts: dict[str, bytes] = {
        SUMMARY_FILENAME: yaml.safe_dump(
            build_summary(result),
            sort_keys=False,
            allow_unicode=True,
        ).encode("utf-8"),
        FRAME_METRICS_FILENAME: (
            build_frame_edge_metrics_csv(result).encode("utf-8")
        ),
    }

    visualization_frame_index = (
        result.metrics.representative_target_frame_index
        if representative is None
        else representative.result.frame_index
    )
    display_image = depth_to_edge_display(
        source.dataset.depth[visualization_frame_index]
    )
    roi_overlay = draw_edge_roi_overlay(
        display_image,
        source.config,
    )
    artifacts[ROI_OVERLAY_FILENAME] = _rgb_png_bytes(
        roi_overlay
    )
    artifacts[TEMPORAL_PLOT_FILENAME] = _figure_png_bytes(
        plot_edge_temporal_metrics(
            discontinuity.frame_results
        )
    )

    profile = discontinuity.aggregate_profile
    profile_csv = build_aggregate_edge_profile_csv(profile)
    if profile is not None and profile_csv is not None:
        artifacts[PROFILE_FILENAME] = profile_csv.encode(
            "utf-8"
        )
        artifacts[PROFILE_PLOT_FILENAME] = _figure_png_bytes(
            plot_edge_probability_profile(profile)
        )

    if representative is not None:
        if representative.label_map is None:
            raise ValueError(
                "representative analysis is missing its label map"
            )
        artifacts[LABEL_MAP_FILENAME] = _npy_bytes(
            representative.label_map
        )
        representative_display = depth_to_edge_display(
            source.dataset.depth[
                representative.result.frame_index
            ]
        )
        artifacts[LABEL_OVERLAY_FILENAME] = _figure_png_bytes(
            plot_edge_label_map(
                representative_display,
                representative.label_map,
                source.config,
            )
        )

    return artifacts


def save_edge_analysis(
    output_dir: Path,
    result: EdgeAnalysisResult,
) -> Path:
    """Save available artifacts without overwrite or partial output."""
    resolved_output_dir = Path(output_dir).expanduser()
    artifacts = build_edge_artifacts(result)
    output_paths = {
        filename: resolved_output_dir / filename
        for filename in artifacts
    }
    existing_paths = [
        path
        for path in output_paths.values()
        if path.exists()
    ]
    if existing_paths:
        existing = ", ".join(
            str(path)
            for path in existing_paths
        )
        raise FileExistsError(
            f"Edge analysis output already exists: {existing}"
        )

    resolved_output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )
    created_paths: list[Path] = []
    try:
        for filename, payload in artifacts.items():
            output_path = output_paths[filename]
            with output_path.open("xb") as stream:
                created_paths.append(output_path)
                stream.write(payload)
    except BaseException:
        for path in reversed(created_paths):
            _remove_created_output(path)
        raise
    return resolved_output_dir


def resolve_output_dir(
    experiment_name: str,
    output_dir: Path | None,
) -> Path:
    """Resolve the default Scene 04 artifact output directory."""
    if output_dir is not None:
        return Path(output_dir).expanduser()
    return (
        DEFAULT_RESULTS_ROOT
        / experiment_name
        / "edge_discontinuity"
    )


def print_completion(
    result: EdgeAnalysisResult,
    output_dir: Path,
) -> None:
    """Print one concise completed Scene 04 analysis report."""
    metadata = result.metadata
    discontinuity = result.metrics.discontinuity
    representative = result.metrics.representative_analysis
    valid_frames = [
        frame
        for frame in discontinuity.frame_results
        if frame.analysis_status == "ok"
    ]
    foreground = _finite_median_or_none(
        [
            frame.foreground_reference_mm
            for frame in valid_frames
        ]
    )
    background = _finite_median_or_none(
        [
            frame.background_reference_mm
            for frame in valid_frames
        ]
    )
    measured_gap = _finite_median_or_none(
        [
            (
                frame.background_reference_mm
                - frame.foreground_reference_mm
            )
            for frame in valid_frames
            if (
                np.isfinite(frame.foreground_reference_mm)
                and np.isfinite(
                    frame.background_reference_mm
                )
            )
        ]
    )

    print("Edge analysis complete.")
    print()
    print("Dataset:")
    print(f"  {_summary_path(result.source.dataset_path)}")
    print()
    print("Setup:")
    print(f"  orientation: {metadata.orientation}")
    print(
        "  nominal foreground/background/gap: "
        f"{metadata.nominal_foreground_distance_mm} / "
        f"{metadata.nominal_background_distance_mm} / "
        f"{metadata.nominal_gap_mm} mm"
    )
    print(
        "  measured foreground/background/gap: "
        f"{_format_mm(foreground)} / "
        f"{_format_mm(background)} / "
        f"{_format_mm(measured_gap)}"
    )
    print()
    print("Frames:")
    print(f"  valid: {discontinuity.valid_frames}")
    print(f"  rejected: {discontinuity.rejected_frames}")
    print(
        "  representative target/selected: "
        f"{result.metrics.representative_target_frame_index} / "
        f"{'none' if representative is None else representative.result.frame_index}"
    )
    print()
    print("Edge quality:")
    print(
        "  foreground/background bleeding median: "
        f"{_format_ratio(discontinuity.median_foreground_bleeding_ratio)} / "
        f"{_format_ratio(discontinuity.median_background_bleeding_ratio)}"
    )
    print(
        "  mixed/outlier/invalid median: "
        f"{_format_ratio(discontinuity.median_mixed_ratio)} / "
        f"{_format_ratio(discontinuity.median_outlier_ratio)} / "
        f"{_format_ratio(discontinuity.median_invalid_ratio)}"
    )
    print(
        "  transition width median: "
        f"{_format_px(discontinuity.median_transition_width_px)}"
    )
    print()
    print("Saved:")
    print(f"  {_summary_path(Path(output_dir).expanduser())}")
    if discontinuity.aggregate_profile is None:
        print("  aggregate profile artifacts omitted")
    if representative is None:
        print("  representative label artifacts omitted")


def main(
    argv: Sequence[str] | None = None,
) -> int:
    """Run one Scene 04 edge analysis from the command line."""
    args = parse_args(argv)
    result = analyze_edge(
        dataset_dir=args.dataset_dir,
        roi_root=args.roi_root,
        representative_frame_index=args.frame_index,
    )
    output_dir = resolve_output_dir(
        experiment_name=result.source.experiment_name,
        output_dir=args.output_dir,
    )
    saved_dir = save_edge_analysis(
        output_dir=output_dir,
        result=result,
    )
    print_completion(result, saved_dir)
    return 0


def _validate_raw_depth(raw_depth: np.ndarray) -> None:
    """Validate the raw array contract used by edge analysis."""
    if not isinstance(raw_depth, np.ndarray):
        raise TypeError(
            "raw_depth must be a numpy array"
        )
    if raw_depth.ndim != 3:
        raise ValueError(
            "raw_depth must have shape (N, H, W)"
        )
    if raw_depth.dtype != np.uint16:
        raise ValueError(
            "raw_depth must have dtype uint16"
        )
    if raw_depth.shape[0] == 0:
        raise ValueError(
            "raw_depth must contain at least one frame"
        )


def _resolve_representative_target(
    *,
    num_frames: int,
    requested_frame_index: int | None,
) -> int:
    """Return a validated preferred representative frame index."""
    if requested_frame_index is None:
        return num_frames // 2
    if (
        not isinstance(requested_frame_index, Integral)
        or isinstance(
            requested_frame_index,
            (bool, np.bool_),
        )
    ):
        raise TypeError(
            "representative_frame_index must be an integer or None"
        )

    normalized_index = int(requested_frame_index)
    if (
        normalized_index < 0
        or normalized_index >= num_frames
    ):
        raise ValueError(
            "representative_frame_index must satisfy "
            f"0 <= index < {num_frames}; got {normalized_index}"
        )
    return normalized_index


def _validate_analysis_result(
    result: EdgeAnalysisResult,
) -> None:
    """Validate the top-level result type used by serialization."""
    if not isinstance(result, EdgeAnalysisResult):
        raise TypeError(
            "result must be an EdgeAnalysisResult"
        )


def _finite_or_none(
    value: float,
) -> float | None:
    """Return one finite Python float or None."""
    if not np.isfinite(value):
        return None
    return float(value)


def _finite_median_or_none(
    values: Sequence[float],
) -> float | None:
    """Return the median of finite values or None."""
    array = np.asarray(values, dtype=np.float64)
    finite = array[np.isfinite(array)]
    if finite.size == 0:
        return None
    return float(np.median(finite))


def _csv_float_or_blank(
    value: float,
) -> float | str:
    """Return a finite CSV value or an empty field."""
    if not np.isfinite(value):
        return ""
    return float(value)


def _summary_path(path: Path) -> str:
    """Return a project-relative path when possible."""
    resolved = Path(path).expanduser().resolve()
    try:
        return str(resolved.relative_to(PROJECT_ROOT))
    except ValueError:
        return str(resolved)


def _rgb_png_bytes(image: np.ndarray) -> bytes:
    """Encode one uint8 RGB image as PNG bytes."""
    if (
        not isinstance(image, np.ndarray)
        or image.ndim != 3
        or image.shape[2] != 3
        or image.dtype != np.uint8
    ):
        raise ValueError(
            "PNG image must have shape (H, W, 3) and dtype uint8"
        )
    bgr = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    success, encoded = cv2.imencode(".png", bgr)
    if not success:
        raise ValueError("Failed to encode RGB image as PNG")
    return encoded.tobytes()


def _figure_png_bytes(figure: object) -> bytes:
    """Encode one Matplotlib figure as PNG bytes."""
    from matplotlib.figure import Figure

    if not isinstance(figure, Figure):
        raise TypeError("figure must be a Matplotlib Figure")
    stream = BytesIO()
    try:
        figure.savefig(
            stream,
            format="png",
            dpi=150,
        )
        return stream.getvalue()
    finally:
        figure.clear()


def _npy_bytes(array: np.ndarray) -> bytes:
    """Encode one NumPy array without pickle support."""
    if not isinstance(array, np.ndarray):
        raise TypeError("array must be a numpy array")
    stream = BytesIO()
    np.save(
        stream,
        array,
        allow_pickle=False,
    )
    return stream.getvalue()


def _remove_created_output(path: Path) -> None:
    """Best-effort cleanup for an artifact created by this call."""
    try:
        path.unlink(missing_ok=True)
    except OSError:
        pass


def _format_mm(value: float | None) -> str:
    """Format one optional millimetre value for completion output."""
    if value is None:
        return "undefined"
    return f"{value:.3f} mm"


def _format_ratio(value: float) -> str:
    """Format one optional ratio for completion output."""
    if not np.isfinite(value):
        return "undefined"
    return f"{float(value):.6f}"


def _format_px(value: float) -> str:
    """Format one optional pixel value for completion output."""
    if not np.isfinite(value):
        return "undefined"
    return f"{float(value):.3f} px"


if __name__ == "__main__":
    raise SystemExit(main())
