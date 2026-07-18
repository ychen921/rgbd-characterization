"""Analyze one extracted baseline depth dataset."""

from dataclasses import dataclass
import sys
from pathlib import Path

import numpy as np

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
    )
