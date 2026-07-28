"""Pure configuration logic for interactive Scene 04 edge ROI selection.

The OpenCV interaction and persistence workflow are intentionally added in
later implementation slices.  This module currently defines the reproducible
CLI options, output paths, foreground-side inference, and configuration
validation used by that workflow.
"""

import argparse
from dataclasses import dataclass
import math
from pathlib import Path
import sys
from typing import Sequence

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.geometry.edge_geometry import compute_signed_distance_map
from src.preprocessing.edge_roi import (
    EdgeBleedingConfig,
    EdgeInvalidConfig,
    EdgeReferenceConfig,
    EdgeROIConfig,
    EdgeTransitionConfig,
    Line2D,
    validate_edge_roi_config,
)
from src.preprocessing.roi import RectROI, derive_roi_key, get_roi_path


DEFAULT_ROI_ROOT = PROJECT_ROOT / "config" / "roi"
DEFAULT_PREVIEW_ROOT = PROJECT_ROOT / "results" / "roi_preview"


@dataclass(frozen=True)
class EdgeAnalysisOptions:
    """Store analysis values supplied by the future selection CLI."""

    distance_bin_px: float = 2.0
    max_edge_distance_px: float = 30.0
    minimum_tolerance_mm: float = 10.0
    mad_scale: float = 3.0
    minimum_valid_ratio: float = 0.9
    minimum_valid_count: int = 100
    bleeding_probability_threshold: float = 0.05
    invalid_ratio_threshold: float = 0.5
    transition_high_probability: float = 0.9
    transition_low_probability: float = 0.1


DEFAULT_ANALYSIS_OPTIONS = EdgeAnalysisOptions()


@dataclass(frozen=True)
class EdgeSelectionPaths:
    """Store paths and names derived from one dataset directory."""

    dataset_dir: Path
    dataset_path: Path
    experiment_name: str
    roi_key: str
    roi_path: Path
    preview_path: Path


@dataclass(frozen=True)
class EdgeSelectionBuildResult:
    """Return a validated configuration and non-fatal selection warnings."""

    config: EdgeROIConfig
    warnings: tuple[str, ...]


def parse_args(
    argv: Sequence[str] | None = None,
) -> argparse.Namespace:
    """Parse reproducible inputs for the future interactive selector."""
    defaults = DEFAULT_ANALYSIS_OPTIONS
    parser = argparse.ArgumentParser(
        description=(
            "Select foreground, background, and edge ROIs plus a nominal "
            "edge line from an extracted Scene 04 depth dataset."
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
        help="Edge ROI configuration directory (default: config/roi).",
    )
    parser.add_argument(
        "--preview-root",
        type=Path,
        default=DEFAULT_PREVIEW_ROOT,
        help=(
            "ROI overlay preview directory "
            "(default: results/roi_preview)."
        ),
    )
    parser.add_argument(
        "--frame-index",
        type=int,
        help="Representative frame index (default: middle frame).",
    )
    parser.add_argument(
        "--distance-bin-px",
        type=float,
        default=defaults.distance_bin_px,
    )
    parser.add_argument(
        "--max-edge-distance-px",
        type=float,
        default=defaults.max_edge_distance_px,
    )
    parser.add_argument(
        "--minimum-tolerance-mm",
        type=float,
        default=defaults.minimum_tolerance_mm,
    )
    parser.add_argument(
        "--mad-scale",
        type=float,
        default=defaults.mad_scale,
    )
    parser.add_argument(
        "--minimum-valid-ratio",
        type=float,
        default=defaults.minimum_valid_ratio,
    )
    parser.add_argument(
        "--minimum-valid-count",
        type=int,
        default=defaults.minimum_valid_count,
    )
    parser.add_argument(
        "--bleeding-threshold",
        type=float,
        default=defaults.bleeding_probability_threshold,
    )
    parser.add_argument(
        "--invalid-threshold",
        type=float,
        default=defaults.invalid_ratio_threshold,
    )
    parser.add_argument(
        "--transition-high",
        type=float,
        default=defaults.transition_high_probability,
    )
    parser.add_argument(
        "--transition-low",
        type=float,
        default=defaults.transition_low_probability,
    )
    return parser.parse_args(argv)


def analysis_options_from_args(
    args: argparse.Namespace,
) -> EdgeAnalysisOptions:
    """Build typed analysis options from parsed command-line arguments."""
    return EdgeAnalysisOptions(
        distance_bin_px=args.distance_bin_px,
        max_edge_distance_px=args.max_edge_distance_px,
        minimum_tolerance_mm=args.minimum_tolerance_mm,
        mad_scale=args.mad_scale,
        minimum_valid_ratio=args.minimum_valid_ratio,
        minimum_valid_count=args.minimum_valid_count,
        bleeding_probability_threshold=args.bleeding_threshold,
        invalid_ratio_threshold=args.invalid_threshold,
        transition_high_probability=args.transition_high,
        transition_low_probability=args.transition_low,
    )


def resolve_selection_paths(
    dataset_dir: Path,
    roi_root: Path = DEFAULT_ROI_ROOT,
    preview_root: Path = DEFAULT_PREVIEW_ROOT,
) -> EdgeSelectionPaths:
    """Derive the shared ROI key and output paths for one experiment."""
    resolved_dataset_dir = Path(dataset_dir).expanduser()
    experiment_name = resolved_dataset_dir.name
    if not experiment_name:
        raise ValueError(
            f"Cannot derive experiment name from {resolved_dataset_dir}"
        )

    roi_key = derive_roi_key(experiment_name)
    roi_path = get_roi_path(roi_root, experiment_name)
    preview_path = (
        Path(preview_root).expanduser() / f"{roi_key}.png"
    )
    return EdgeSelectionPaths(
        dataset_dir=resolved_dataset_dir,
        dataset_path=resolved_dataset_dir / "depth.npz",
        experiment_name=experiment_name,
        roi_key=roi_key,
        roi_path=roi_path,
        preview_path=preview_path,
    )


def infer_foreground_side(
    line: Line2D,
    foreground_roi: RectROI,
    background_roi: RectROI,
) -> str:
    """Infer a stable foreground-side name from the two reference centers."""
    if not isinstance(line, Line2D):
        raise TypeError(
            f"line must be a Line2D; got {type(line).__name__}"
        )
    for field_name, roi in (
        ("foreground_roi", foreground_roi),
        ("background_roi", background_roi),
    ):
        if not isinstance(roi, RectROI):
            raise TypeError(
                f"{field_name} must be a RectROI; "
                f"got {type(roi).__name__}"
            )

    foreground_raw = _raw_line_side(
        line,
        _roi_center(foreground_roi),
    )
    background_raw = _raw_line_side(
        line,
        _roi_center(background_roi),
    )
    if math.isclose(foreground_raw, 0.0, abs_tol=1e-12):
        raise ValueError(
            "foreground ROI center lies on the nominal edge"
        )
    if math.isclose(background_raw, 0.0, abs_tol=1e-12):
        raise ValueError(
            "background ROI center lies on the nominal edge"
        )
    if foreground_raw * background_raw >= 0.0:
        raise ValueError(
            "foreground and background ROI centers must lie on "
            "opposite sides of the nominal edge"
        )

    dy = line.p2[1] - line.p1[1]
    if math.isclose(dy, 0.0, abs_tol=1e-12):
        return "positive" if foreground_raw > 0.0 else "negative"

    left_raw_sign = 1.0 if dy > 0.0 else -1.0
    foreground_raw_sign = (
        1.0 if foreground_raw > 0.0 else -1.0
    )
    return (
        "left"
        if foreground_raw_sign == left_raw_sign
        else "right"
    )


def build_edge_roi_config(
    *,
    name: str,
    source_experiment: str,
    source_frame_index: int,
    foreground_roi: RectROI,
    background_roi: RectROI,
    edge_roi: RectROI,
    nominal_edge: Line2D,
    image_shape: tuple[int, int],
    options: EdgeAnalysisOptions | None = None,
) -> EdgeSelectionBuildResult:
    """Construct and validate a selection using inferred foreground geometry."""
    selected_options = (
        DEFAULT_ANALYSIS_OPTIONS
        if options is None
        else options
    )
    if not isinstance(selected_options, EdgeAnalysisOptions):
        raise TypeError(
            "options must be an EdgeAnalysisOptions; "
            f"got {type(selected_options).__name__}"
        )

    foreground_side = infer_foreground_side(
        nominal_edge,
        foreground_roi,
        background_roi,
    )
    config = EdgeROIConfig(
        name=name,
        source_experiment=source_experiment,
        source_frame_index=source_frame_index,
        foreground_roi=foreground_roi,
        background_roi=background_roi,
        edge_roi=edge_roi,
        nominal_edge=nominal_edge,
        foreground_side=foreground_side,
        distance_bin_px=selected_options.distance_bin_px,
        max_edge_distance_px=(
            selected_options.max_edge_distance_px
        ),
        reference=EdgeReferenceConfig(
            minimum_tolerance_mm=(
                selected_options.minimum_tolerance_mm
            ),
            mad_scale=selected_options.mad_scale,
            minimum_valid_ratio=(
                selected_options.minimum_valid_ratio
            ),
            minimum_valid_count=(
                selected_options.minimum_valid_count
            ),
        ),
        bleeding=EdgeBleedingConfig(
            probability_threshold=(
                selected_options.bleeding_probability_threshold
            ),
        ),
        invalid=EdgeInvalidConfig(
            ratio_threshold=(
                selected_options.invalid_ratio_threshold
            ),
        ),
        transition=EdgeTransitionConfig(
            high_probability=(
                selected_options.transition_high_probability
            ),
            low_probability=(
                selected_options.transition_low_probability
            ),
        ),
    )
    warnings = validate_selection_semantics(config, image_shape)
    return EdgeSelectionBuildResult(
        config=config,
        warnings=warnings,
    )


def validate_selection_semantics(
    config: EdgeROIConfig,
    image_shape: tuple[int, int],
) -> tuple[str, ...]:
    """Validate required geometry and report non-fatal selection concerns."""
    validate_edge_roi_config(config, image_shape)

    inferred_side = infer_foreground_side(
        config.nominal_edge,
        config.foreground_roi,
        config.background_roi,
    )
    if not _side_matches_inference(
        config.foreground_side,
        inferred_side,
        config.nominal_edge,
        config.foreground_roi,
    ):
        raise ValueError(
            "foreground_side does not match the foreground ROI side"
        )

    warnings: list[str] = []
    roi_pairs = (
        (
            "foreground_roi",
            config.foreground_roi,
            "background_roi",
            config.background_roi,
        ),
        (
            "foreground_roi",
            config.foreground_roi,
            "edge_roi",
            config.edge_roi,
        ),
        (
            "background_roi",
            config.background_roi,
            "edge_roi",
            config.edge_roi,
        ),
    )
    for first_name, first, second_name, second in roi_pairs:
        if _rectangles_overlap(first, second):
            warnings.append(
                f"{first_name} overlaps {second_name}"
            )

    minimum_count = config.reference.minimum_valid_count
    for field_name in ("foreground_roi", "background_roi"):
        roi = getattr(config, field_name)
        if roi.pixel_count < minimum_count:
            warnings.append(
                f"{field_name} has {roi.pixel_count} pixels, fewer "
                f"than minimum_valid_count {minimum_count}"
            )

    distance = compute_signed_distance_map(
        image_shape,
        config.nominal_edge,
        config.foreground_side,
    )
    edge_distance = distance[
        config.edge_roi.y:config.edge_roi.y + config.edge_roi.height,
        config.edge_roi.x:config.edge_roi.x + config.edge_roi.width,
    ]
    max_distance = config.max_edge_distance_px
    if float(np.min(edge_distance)) > -max_distance:
        warnings.append(
            "edge_roi does not cover max_edge_distance_px on "
            "the foreground side"
        )
    if float(np.max(edge_distance)) < max_distance:
        warnings.append(
            "edge_roi does not cover max_edge_distance_px on "
            "the background side"
        )

    return tuple(warnings)


def _roi_center(roi: RectROI) -> tuple[float, float]:
    """Return the continuous geometric center of one rectangle."""
    return (
        roi.x + roi.width / 2.0,
        roi.y + roi.height / 2.0,
    )


def _raw_line_side(
    line: Line2D,
    point: tuple[float, float],
) -> float:
    """Return the unnormalized directed cross product for one point."""
    x1, y1 = line.p1
    x2, y2 = line.p2
    x, y = point
    return (
        (x2 - x1) * (y - y1)
        - (y2 - y1) * (x - x1)
    )


def _side_matches_inference(
    configured_side: str,
    inferred_side: str,
    line: Line2D,
    foreground_roi: RectROI,
) -> bool:
    """Allow equivalent raw-sign and left/right foreground descriptions."""
    if configured_side in {"left", "right"}:
        return configured_side == inferred_side

    foreground_raw = _raw_line_side(
        line,
        _roi_center(foreground_roi),
    )
    if configured_side == "positive":
        return foreground_raw > 0.0
    if configured_side == "negative":
        return foreground_raw < 0.0
    return False


def _rectangles_overlap(first: RectROI, second: RectROI) -> bool:
    """Return whether two half-open image rectangles share positive area."""
    return (
        first.x < second.x + second.width
        and second.x < first.x + first.width
        and first.y < second.y + second.height
        and second.y < first.y + first.height
    )
