"""Compute Scene 04 depth-discontinuity metrics for one dataset.

This module currently provides the input-loading and numerical orchestration
layers. CLI parsing, artifact persistence, and experiment metadata belong to
the next integration stage.
"""

from dataclasses import dataclass, replace
from numbers import Integral
from pathlib import Path
import sys

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_ROI_ROOT = PROJECT_ROOT / "config" / "roi"

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.geometry.edge_geometry import (  # noqa: E402
    compute_signed_distance_map,
)
from src.io.dataset import DepthDataset  # noqa: E402
from src.metrics.edge_discontinuity import (  # noqa: E402
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
class EdgeAnalysisResult:
    """Store loaded input metadata and computed Scene 04 metrics."""

    source: EdgeAnalysisInput
    metrics: EdgeMetricResults


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
    metrics = compute_edge_metrics(
        raw_depth=source.dataset.depth,
        config=source.config,
        representative_frame_index=representative_frame_index,
    )
    return EdgeAnalysisResult(
        source=source,
        metrics=metrics,
    )


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
