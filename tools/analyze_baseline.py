"""Analyze one extracted baseline depth dataset."""

from dataclasses import dataclass
import sys
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

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


@dataclass(frozen=True)
class BaselineMetricResults:
    """Store all baseline metric results for a single dataset."""
    depth_quality: DepthQualityResult
    temporal_noise: TemporalNoiseResult
    measured_depth: MeasuredDepthResult

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