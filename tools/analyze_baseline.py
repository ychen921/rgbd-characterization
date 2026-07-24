"""Analyze one extracted baseline depth dataset."""

import argparse
import csv
from dataclasses import dataclass
import sys
from pathlib import Path

import numpy as np
import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_ROI_ROOT = PROJECT_ROOT / "config" / "roi"
DEFAULT_RESULTS_ROOT = PROJECT_ROOT / "results"
DEFAULT_DEPTH_CAMERA_INFO_PATH = (
    PROJECT_ROOT
    / "config"
    / "calib"
    / "depth_camera_info.yaml"
)

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.io.dataset import DepthDataset
from src.metrics.depth_quality import (
    DepthQualityResult,
    compute_depth_quality,
)
from src.metrics.measured_depth import (
    MeasuredDepthResult,
    compute_measured_depth,
)
from src.metrics.temporal import (
    DEFAULT_MIN_VALID_RATIO,
    TemporalNoiseResult,
    compute_temporal_noise,
)
from src.preprocessing.depth import prepare_depth
from src.preprocessing.roi import (
    RectROI,
    derive_roi_key,
    get_roi_path,
    load_roi,
)
from src.geometry.camera import (
    CameraIntrinsics,
    load_camera_intrinsics,
    validate_depth_resolution,
)
from src.metrics.planarity import (
    DEFAULT_INLIER_THRESHOLD_MM,
    DEFAULT_MIN_VALID_POINTS,
    PlanarityResult,
    compute_planarity,
)


@dataclass(frozen=True)
class BaselineInput:
    """Store one loaded dataset and its selected raw ROI."""

    experiment_name: str
    dataset_path: Path
    roi_key: str
    roi_path: Path
    dataset: DepthDataset
    roi: RectROI
    raw_roi: np.ndarray


@dataclass(frozen=True)
class BaselineMetricResults:
    """Store all baseline metric results for a single dataset."""

    depth_quality: DepthQualityResult
    temporal_noise: TemporalNoiseResult
    measured_depth: MeasuredDepthResult
    planarity: PlanarityResult


@dataclass(frozen=True)
class BaselineAnalysisResult:
    """Store loaded input metadata and computed baseline metrics."""

    source: BaselineInput
    metrics: BaselineMetricResults
    min_valid_ratio: float

    depth_camera_info_path: Path
    intrinsics: CameraIntrinsics
    plane_inlier_threshold_mm: float
    plane_min_valid_points: int


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments for baseline analysis."""

    parser = argparse.ArgumentParser(
        description=(
            "Analyze one extracted baseline depth dataset."
        )
    )

    # Dataset directory
    parser.add_argument(
        "dataset_dir",
        type=Path,
        help="Experiment directory containing depth.npz.",
    )

    # ROI root directory
    parser.add_argument(
        "--roi-root",
        type=Path,
        default=DEFAULT_ROI_ROOT,
        help="ROI configuration directory (default: config/roi).",
    )

    # Output directory
    parser.add_argument(
        "--output-dir",
        type=Path,
        help=(
            "Artifact output directory. Defaults to "
            "results/<experiment>/baseline."
        ),
    )

    # Temporal threshold
    parser.add_argument(
        "--min-valid-ratio",
        type=float,
        default=DEFAULT_MIN_VALID_RATIO,
        help="Minimum valid-frame ratio for temporal noise.",
    )

    # Depth intrinsic info
    parser.add_argument(
        "--depth-camera-info",
        type=Path,
        default=DEFAULT_DEPTH_CAMERA_INFO_PATH,
        help=(
            "Depth CameraInfo YAML path "
            "(default: config/calib/depth_camera_info.yaml)."
        ),
    )

    # Plane residual counted as inliers
    parser.add_argument(
        "--plane-inlier-threshold-mm",
        type=float,
        default=DEFAULT_INLIER_THRESHOLD_MM,
        help=(
            "Maximum absolute plane residual counted as an inlier, "
            "in millimetres. Defaults to 5.0."
        ),
    )

    # Minimum depth points for plane fitting
    parser.add_argument(
        "--plane-min-valid-points",
        type=int,
        default=DEFAULT_MIN_VALID_POINTS,
        help=(
            "Minimum valid depth points required for each plane fit. "
            "Defaults to 100."
        ),
    )

    return parser.parse_args()


def print_completion(
    result: BaselineAnalysisResult,
    output_dir: Path,
) -> None:
    """Print a concise completed-analysis report."""

    source = result.source
    roi = source.roi
    median_depth = result.metrics.measured_depth.median_depth
    planarity = result.metrics.planarity

    median_text = (
        "undefined"
        if not np.isfinite(median_depth)
        else f"{median_depth:.3f} mm"
    )

    distance_text = (
        "undefined"
        if not np.isfinite(
            planarity.median_distance_m
        )
        else (
            f"{planarity.median_distance_m:.6f} m"
        )
    )

    rmse_text = (
        "undefined"
        if not np.isfinite(
            planarity.median_rmse_mm
        )
        else f"{planarity.median_rmse_mm:.3f} mm"
    )

    print("Baseline analysis complete.")
    print()
    print("Dataset:")
    print(f"  {_summary_path(source.dataset_path)}")
    print()
    print("ROI:")
    print(f"  {_summary_path(source.roi_path)}")
    print(
        "  rectangle: "
        f"x={roi.x}, y={roi.y}, "
        f"width={roi.width}, height={roi.height}"
    )
    print(f"  pixels: {roi.pixel_count}")
    print()
    print("Measured depth:")
    print(f"  median: {median_text}")
    print()
    print("Planarity:")
    print(
        "  successful frames: "
        f"{planarity.successful_frames}"
    )
    print(
        "  failed frames: "
        f"{planarity.failed_frames}"
    )
    print(
        "  median plane distance: "
        f"{distance_text}"
    )
    print(
        "  median residual RMSE: "
        f"{rmse_text}"
    )
    print()
    print("Saved:")
    print(f"  {_summary_path(Path(output_dir).expanduser())}")


def compute_baseline_metrics(
    raw_roi: np.ndarray,
    *,
    intrinsics: CameraIntrinsics,
    roi: RectROI,
    min_valid_ratio: float = DEFAULT_MIN_VALID_RATIO,
    plane_inlier_threshold_mm: float = DEFAULT_INLIER_THRESHOLD_MM,
    plane_min_valid_points: int = DEFAULT_MIN_VALID_POINTS,
) -> BaselineMetricResults:
    """Compute all baseline metrics for a single dataset."""

    # Compute special-value occurrence statistics for raw ROI frames
    dq_result = compute_depth_quality(raw_roi)

    # Convert the raw ROI from uint16 to float32
    prepared_roi = prepare_depth(
        depth=raw_roi
    )
    # Compute temporal noise and measured depth metrics
    tn_result = compute_temporal_noise(
        depth=prepared_roi,
        min_valid_ratio=min_valid_ratio,
    )
    md_result = compute_measured_depth(
        depth=prepared_roi
    )

    # Planarity requires the ROI's original full-image offset so that
    # cropped pixels are back-projected with the correct camera rays.
    planarity_result = compute_planarity(
        prepared_roi,
        intrinsics=intrinsics,
        roi_x=roi.x,
        roi_y=roi.y,
        inlier_threshold_mm=plane_inlier_threshold_mm,
        min_valid_points=plane_min_valid_points,
    )

    return BaselineMetricResults(
        depth_quality=dq_result,
        temporal_noise=tn_result,
        measured_depth=md_result,
        planarity=planarity_result,
    )


def load_baseline_input(
    dataset_dir: Path,
    roi_root: Path = DEFAULT_ROI_ROOT,
) -> BaselineInput:
    """Load one depth dataset and crop its configured raw ROI."""

    # Ensure the dataset directory and ROI root exist
    dataset_dir = Path(dataset_dir).expanduser()
    roi_root = Path(roi_root).expanduser()
    experiment_name = dataset_dir.name
    if experiment_name == "":
        raise ValueError(
            f"Cannot derive experiment name from {dataset_dir}"
        )

    # Ensure depth dataset exists
    dataset_path = dataset_dir / "depth.npz"
    if not dataset_path.is_file():
        raise FileNotFoundError(
            f"Cannot find dataset file {dataset_path}"
        )

    # Ensure ROI file exists
    roi_key = derive_roi_key(
        experiment_name=experiment_name
    )
    roi_path = get_roi_path(
        roi_root=roi_root,
        experiment_name=experiment_name,
    )
    if not roi_path.is_file():
        raise FileNotFoundError(
            f"ROI configuration not found: {roi_path}\n\n"
            "Run:\n"
            f"python3 tools/select_roi.py {dataset_dir}"
        )

    # Load the ROI configuration and depth dataset
    roi = load_roi(path=roi_path)
    dataset = DepthDataset.load(path=dataset_path)
    if dataset.num_frames == 0:
        raise ValueError(
            f"Dataset {dataset_path} contains no depth frames"
        )

    # Crop the dataset's depth frames to the selected ROI
    raw_roi = roi.crop(dataset.depth)

    return BaselineInput(
        experiment_name=experiment_name,
        dataset_path=dataset_path,
        roi_key=roi_key,
        roi_path=roi_path,
        dataset=dataset,
        roi=roi,
        raw_roi=raw_roi,
    )


def _finite_or_none(value: float) -> float | None:
    """Return a finite Python float, or None for an undefined metric."""
    if not np.isfinite(value):
        return None
    return float(value)


def _summary_path(path: Path) -> str:
    """Return a project-relative path when possible."""
    resolved = path.resolve()
    try:
        return str(resolved.relative_to(PROJECT_ROOT))
    except ValueError:
        return str(resolved)


def build_summary(
    result: BaselineAnalysisResult,
) -> dict[str, object]:
    """Build a YAML-safe summary for one baseline analysis."""
    source = result.source
    dataset = source.dataset
    roi = source.roi

    quality = result.metrics.depth_quality
    temporal = result.metrics.temporal_noise
    measured = result.metrics.measured_depth

    planarity = result.metrics.planarity
    intrinsics = result.intrinsics

    return {
        "dataset": {
            "experiment": source.experiment_name,
            "path": _summary_path(source.dataset_path),
            "num_frames": int(dataset.num_frames),
            "width": int(dataset.width),
            "height": int(dataset.height),
        },
        "roi": {
            "key": source.roi_key,
            "config": _summary_path(source.roi_path),
            "x": roi.x,
            "y": roi.y,
            "width": roi.width,
            "height": roi.height,
            "pixel_count": roi.pixel_count,
        },
        "depth_camera": {
            "config": _summary_path(
                result.depth_camera_info_path
            ),
            "frame_id": intrinsics.frame_id,
            "width": intrinsics.width,
            "height": intrinsics.height,
            "fx": intrinsics.fx,
            "fy": intrinsics.fy,
            "cx": intrinsics.cx,
            "cy": intrinsics.cy,
        },
        "depth_preprocessing": {
            "excluded_raw_values": [
                0,
                int(np.iinfo(np.uint16).max),
            ],
            "depth_scale_to_mm": 1.0,
        },
        "depth_quality": {
            "zero_ratio": _finite_or_none(quality.zero_ratio),
            "max_uint16": {
                "ratio": _finite_or_none(quality.max_uint16_ratio),
                "affected_frames": quality.max_uint16_affected_frames,
                "max_pixels_per_frame": (
                    quality.max_uint16_max_pixels_per_frame
                ),
            },
        },
        "temporal_noise": {
            "min_valid_ratio": float(result.min_valid_ratio),
            "median_std_mm": _finite_or_none(temporal.median_std),
            "mean_std_mm": _finite_or_none(temporal.mean_std),
            "p95_std_mm": _finite_or_none(temporal.p95_std),
        },
        "measured_depth": {
            "median_mm": _finite_or_none(measured.median_depth),
            "mean_mm": _finite_or_none(measured.mean_depth),
            "std_mm": _finite_or_none(measured.std_depth),
            "p05_mm": _finite_or_none(measured.p05_depth),
            "p95_mm": _finite_or_none(measured.p95_depth),
        },
        "planarity": {
            "fitting_method": "svd",
            "inlier_threshold_mm": float(
                result.plane_inlier_threshold_mm
            ),
            "min_valid_points": int(
                result.plane_min_valid_points
            ),
            "successful_frames": (
                planarity.successful_frames
            ),
            "failed_frames": planarity.failed_frames,
            "plane_distance": {
                "median_m": _finite_or_none(
                    planarity.median_distance_m
                ),
                "std_mm": _finite_or_none(
                    planarity.distance_std_mm
                ),
            },
            "tilt": {
                "median_deg": _finite_or_none(
                    planarity.median_tilt_deg
                ),
                "std_deg": _finite_or_none(
                    planarity.tilt_std_deg
                ),
            },
            "residual": {
                "median_rmse_mm": _finite_or_none(
                    planarity.median_rmse_mm
                ),
                "p95_rmse_mm": _finite_or_none(
                    planarity.p95_rmse_mm
                ),
                "median_p95_abs_mm": _finite_or_none(
                    planarity.median_p95_abs_mm
                ),
            },
            "inlier_ratio": {
                "median": _finite_or_none(
                    planarity.median_inlier_ratio
                ),
            },
        },
    }


def _csv_float_or_blank(
    value: float,
) -> float | str:
    """Return a finite CSV number or an empty field."""
    if not np.isfinite(value):
        return ""
    return float(value)


def _validate_frame_plane_metrics(
    result: BaselineAnalysisResult,
) -> None:
    """Validate timestamp-aligned planarity arrays before writing."""
    timestamps_ns = result.source.dataset.timestamps_ns
    planarity = result.metrics.planarity
    num_frames = result.source.dataset.num_frames

    if timestamps_ns.shape != (num_frames,):
        raise ValueError(
            "timestamps_ns must have shape "
            f"({num_frames},); got shape {timestamps_ns.shape}"
        )

    if planarity.frame_normal.shape != (
        num_frames,
        3,
    ):
        raise ValueError(
            "frame_normal must have shape "
            f"({num_frames}, 3); "
            f"got shape {planarity.frame_normal.shape}"
        )

    one_dimensional_arrays = {
        "frame_distance_m": (
            planarity.frame_distance_m
        ),
        "frame_tilt_deg": (
            planarity.frame_tilt_deg
        ),
        "frame_rmse_mm": (
            planarity.frame_rmse_mm
        ),
        "frame_residual_std_mm": (
            planarity.frame_residual_std_mm
        ),
        "frame_p95_abs_mm": (
            planarity.frame_p95_abs_mm
        ),
        "frame_inlier_ratio": (
            planarity.frame_inlier_ratio
        ),
        "frame_valid_points": (
            planarity.frame_valid_points
        ),
        "frame_fit_succeeded": (
            planarity.frame_fit_succeeded
        ),
    }

    for field_name, values in one_dimensional_arrays.items():
        if values.shape != (num_frames,):
            raise ValueError(
                f"{field_name} must have shape "
                f"({num_frames},); got shape {values.shape}"
            )


def save_frame_plane_metrics_csv(
    csv_path: Path,
    result: BaselineAnalysisResult,
) -> None:
    """Save timestamp-aligned per-frame plane metrics."""
    _validate_frame_plane_metrics(result)

    csv_path = Path(csv_path).expanduser()
    csv_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    timestamps_ns = result.source.dataset.timestamps_ns
    planarity = result.metrics.planarity
    num_frames = result.source.dataset.num_frames

    with csv_path.open(
        "x",
        encoding="utf-8",
        newline="",
    ) as stream:
        writer = csv.writer(stream)

        writer.writerow(
            [
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
            ]
        )

        for frame_index in range(num_frames):
            normal = planarity.frame_normal[
                frame_index
            ]

            writer.writerow(
                [
                    frame_index,
                    int(timestamps_ns[frame_index]),
                    str(
                        bool(
                            planarity.frame_fit_succeeded[
                                frame_index
                            ]
                        )
                    ).lower(),
                    int(
                        planarity.frame_valid_points[
                            frame_index
                        ]
                    ),
                    _csv_float_or_blank(normal[0]),
                    _csv_float_or_blank(normal[1]),
                    _csv_float_or_blank(normal[2]),
                    _csv_float_or_blank(
                        planarity.frame_distance_m[
                            frame_index
                        ]
                    ),
                    _csv_float_or_blank(
                        planarity.frame_tilt_deg[
                            frame_index
                        ]
                    ),
                    _csv_float_or_blank(
                        planarity.frame_rmse_mm[
                            frame_index
                        ]
                    ),
                    _csv_float_or_blank(
                        planarity.frame_residual_std_mm[
                            frame_index
                        ]
                    ),
                    _csv_float_or_blank(
                        planarity.frame_p95_abs_mm[
                            frame_index
                        ]
                    ),
                    _csv_float_or_blank(
                        planarity.frame_inlier_ratio[
                            frame_index
                        ]
                    ),
                ]
            )


def save_summary(
    summary_path: Path,
    summary: dict[str, object],
) -> None:
    """Save one baseline summary as YAML without overwriting."""

    summary_path = Path(summary_path).expanduser()
    summary_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    # Write the YAML file without overwriting an existing file.
    with summary_path.open(
        "x",
        encoding="utf-8",
    ) as stream:
        yaml.safe_dump(
            summary,
            stream,
            sort_keys=False,
            allow_unicode=True,
        )


def save_frame_median_csv(
    csv_path: Path,
    result: BaselineAnalysisResult,
) -> None:
    """Save timestamp-aligned per-frame median depth values."""

    csv_path = Path(csv_path).expanduser()

    timestamps_ns = result.source.dataset.timestamps_ns
    frame_medians = result.metrics.measured_depth.frame_median

    # Validate shapes before creating the file.
    if timestamps_ns.ndim != 1:
        raise ValueError(
            f"timestamps_ns must have shape (N,); got shape "
            f"{timestamps_ns.shape}"
        )
    if frame_medians.ndim != 1:
        raise ValueError(
            f"frame_medians must have shape (N,); got shape "
            f"{frame_medians.shape}"
        )
    if timestamps_ns.shape != frame_medians.shape:
        raise ValueError(
            "Timestamp and frame-median counts do not match"
        )

    csv_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    # Write the CSV file without overwriting an existing file.
    with csv_path.open(
        "x",
        encoding="utf-8",
        newline="",
    ) as stream:
        writer = csv.writer(stream)

        # Write the header row
        writer.writerow(
            [
                "frame_index",
                "timestamp_ns",
                "median_depth_mm",
            ]
        )

        # Write one row per frame, leaving undefined median depth blank.
        for frame_index, (timestamp_ns, median_depth) in enumerate(
            zip(timestamps_ns, frame_medians, strict=True),
        ):
            writer.writerow(
                [
                    frame_index,
                    int(timestamp_ns),
                    (
                        ""
                        if np.isnan(median_depth)
                        else float(median_depth)
                    ),
                ]
            )


def save_metric_maps(
    output_dir: Path,
    result: BaselineAnalysisResult,
) -> None:
    """Save baseline per-pixel metric maps without overwriting."""

    # Extract metric maps from the analysis result
    temporal_std = result.metrics.temporal_noise.std_map
    zero_ratio = result.metrics.depth_quality.zero_ratio_map
    max_uint16_ratio = (
        result.metrics.depth_quality.max_uint16_ratio_map
    )

    # Validate shapes before creating the files
    expected_shape = (
        result.source.roi.height,
        result.source.roi.width,
    )
    maps = {
        "temporal_std.npy": temporal_std,
        "zero_ratio_map.npy": zero_ratio,
        "max_uint16_ratio_map.npy": max_uint16_ratio,
    }
    for filename, array in maps.items():
        if array.shape != expected_shape:
            raise ValueError(
                f"{filename} must have shape {expected_shape}; "
                f"got shape {array.shape}"
            )

    # Check for conflicting outputs before writing any map.
    output_dir = Path(output_dir).expanduser()

    output_paths = {
        filename: output_dir / filename
        for filename in maps
    }

    existing_paths = [
        path
        for path in output_paths.values()
        if path.exists()
    ]

    # Raise an error if any of the output files already exist
    if existing_paths:
        existing = ", ".join(
            str(path)
            for path in existing_paths
        )
        raise FileExistsError(
            f"Metric map output already exists: {existing}"
        )

    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    for filename, array in maps.items():
        output_path = output_paths[filename]

        with output_path.open(
            "xb",
        ) as stream:
            np.save(
                stream,
                array,
                allow_pickle=False,
            )


def save_baseline_analysis(
    output_dir: Path,
    result: BaselineAnalysisResult,
) -> Path:
    """Save all artifacts for one completed baseline analysis."""

    output_dir = Path(output_dir).expanduser()
    artifact_paths = [
        output_dir / "summary.yaml",
        output_dir / "frame_median_depth.csv",
        output_dir / "frame_plane_metrics.csv",
        output_dir / "temporal_std.npy",
        output_dir / "zero_ratio_map.npy",
        output_dir / "max_uint16_ratio_map.npy",
    ]

    # Add existing paths to a list
    existing_paths = [
        path
        for path in artifact_paths
        if path.exists()
    ]

    # Raise an error if any of the output files already exist
    if existing_paths:
        existing = ", ".join(
            str(path)
            for path in existing_paths
        )
        raise FileExistsError(
            f"Baseline output already exists: {existing}"
        )

    # Save all artifacts for the baseline analysis
    summary = build_summary(result=result)

    _validate_frame_plane_metrics(result)

    save_frame_median_csv(
        csv_path=output_dir / "frame_median_depth.csv",
        result=result,
    )

    save_frame_plane_metrics_csv(
        csv_path=output_dir / "frame_plane_metrics.csv",
        result=result,
    )

    save_metric_maps(
        output_dir=output_dir,
        result=result,
    )

    save_summary(
        summary_path=output_dir / "summary.yaml",
        summary=summary,
    )

    return output_dir


def analyze_baseline(
    dataset_dir: Path,
    roi_root: Path = DEFAULT_ROI_ROOT,
    min_valid_ratio: float = DEFAULT_MIN_VALID_RATIO,
    depth_camera_info_path: Path = DEFAULT_DEPTH_CAMERA_INFO_PATH,
    plane_inlier_threshold_mm: float = DEFAULT_INLIER_THRESHOLD_MM,
    plane_min_valid_points: int = DEFAULT_MIN_VALID_POINTS,
) -> BaselineAnalysisResult:
    """Load one baseline dataset and compute all ROI metrics."""

    # Load the dataset and selected ROI
    baseline_input = load_baseline_input(
        dataset_dir=dataset_dir,
        roi_root=roi_root,
    )

    # Load depth camera intrinsic
    camera_info_path = Path(depth_camera_info_path).expanduser()
    intrinsics = load_camera_intrinsics(camera_info_path)

    # CameraInfo describes the full uncropped depth image, so validate
    # it against the dataset rather than the cropped ROI.
    validate_depth_resolution(
        baseline_input.dataset.depth,
        intrinsics,
    )

    # Compute all baseline metrics
    metrics_results = compute_baseline_metrics(
        raw_roi=baseline_input.raw_roi,
        min_valid_ratio=min_valid_ratio,
        roi=baseline_input.roi,
        intrinsics=intrinsics,
        plane_inlier_threshold_mm=plane_inlier_threshold_mm,
        plane_min_valid_points=plane_min_valid_points,
    )

    return BaselineAnalysisResult(
        source=baseline_input,
        metrics=metrics_results,
        min_valid_ratio=float(min_valid_ratio),
        depth_camera_info_path=camera_info_path,
        intrinsics=intrinsics,
        plane_inlier_threshold_mm=float(plane_inlier_threshold_mm),
        plane_min_valid_points=int(plane_min_valid_points),
    )


def resolve_output_dir(
    experiment_name: str,
    output_dir: Path | None,
) -> Path:
    """Resolve the baseline artifact output directory."""

    if output_dir is not None:
        return Path(output_dir).expanduser()

    return (
        DEFAULT_RESULTS_ROOT
        / experiment_name
        / "baseline"
    )


def main() -> int:
    """Run one baseline analysis from the command line."""

    args = parse_args()

    result = analyze_baseline(
        dataset_dir=args.dataset_dir,
        roi_root=args.roi_root,
        min_valid_ratio=args.min_valid_ratio,
        depth_camera_info_path=args.depth_camera_info,
        plane_inlier_threshold_mm=args.plane_inlier_threshold_mm,
        plane_min_valid_points=args.plane_min_valid_points,
    )

    output_dir = resolve_output_dir(
        experiment_name=result.source.experiment_name,
        output_dir=args.output_dir,
    )

    save_dir = save_baseline_analysis(
        output_dir=output_dir,
        result=result,
    )

    print_completion(
        result=result,
        output_dir=save_dir,
    )

    return 0

if __name__ == "__main__":
    raise SystemExit(main())
