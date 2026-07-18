"""Analyze one extracted baseline depth dataset."""

import csv
from dataclasses import dataclass
import sys
from pathlib import Path

import numpy as np
import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_ROI_ROOT = PROJECT_ROOT / "config" / "roi"

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


@dataclass(frozen=True)
class BaselineAnalysisResult:
    """Store loaded input metadata and computed baseline metrics."""

    source: BaselineInput
    metrics: BaselineMetricResults
    min_valid_ratio: float


def compute_baseline_metrics(
    raw_roi: np.ndarray,
    min_valid_ratio: float = DEFAULT_MIN_VALID_RATIO,
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

    return BaselineMetricResults(
        depth_quality=dq_result,
        temporal_noise=tn_result,
        measured_depth=md_result,
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
    }


def save_summary(
    path: Path,
    summary: dict[str, object],
) -> None:
    """Save one baseline summary as YAML without overwriting."""
    
    summary_path = Path(path).expanduser()
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



def analyze_baseline(
    dataset_dir: Path,
    roi_root: Path = DEFAULT_ROI_ROOT,
    min_valid_ratio: float = DEFAULT_MIN_VALID_RATIO,
) -> BaselineAnalysisResult:
    """Load one baseline dataset and compute all ROI metrics."""

    # Load the dataset and selected ROI
    baseline_input = load_baseline_input(
        dataset_dir=dataset_dir,
        roi_root=roi_root,
    )

    # Compute all baseline metrics
    metrics_results = compute_baseline_metrics(
        raw_roi=baseline_input.raw_roi,
        min_valid_ratio=min_valid_ratio,
    )

    return BaselineAnalysisResult(
        source=baseline_input,
        metrics=metrics_results,
        min_valid_ratio=float(min_valid_ratio),
    )
