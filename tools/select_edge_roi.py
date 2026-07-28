"""Interactive selection primitives for Scene 04 edge ROI configuration.

Persistence and final confirmation are intentionally added in a later
implementation slice.  This module defines reproducible CLI options, pure
configuration logic, and the OpenCV interactions that collect three
rectangles plus a two-point nominal edge.
"""

import argparse
from dataclasses import dataclass, field
from enum import Enum
import math
from pathlib import Path
import sys
from typing import Sequence

import cv2
import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.geometry.edge_geometry import compute_signed_distance_map
from src.io.dataset import DepthDataset
from src.preprocessing.edge_roi import (
    EdgeBleedingConfig,
    EdgeInvalidConfig,
    EdgeReferenceConfig,
    EdgeROIConfig,
    EdgeTransitionConfig,
    Line2D,
    load_edge_roi_config,
    save_edge_roi_config,
    validate_edge_roi_config,
)
from src.preprocessing.roi import RectROI, derive_roi_key, get_roi_path


DEFAULT_ROI_ROOT = PROJECT_ROOT / "config" / "roi"
DEFAULT_PREVIEW_ROOT = PROJECT_ROOT / "results" / "roi_preview"
DISPLAY_PERCENTILE_LOW = 1.0
DISPLAY_PERCENTILE_HIGH = 99.0

FOREGROUND_COLOR = (0, 255, 0)
BACKGROUND_COLOR = (255, 0, 0)
EDGE_COLOR = (0, 255, 255)
LINE_COLOR = (0, 0, 255)
INVALID_COLOR = (255, 0, 255)

LINE_WINDOW_NAME = "Select nominal edge"
CONFIRM_WINDOW_NAME = "Review edge selection"
ENTER_KEYS = frozenset({10, 13})
ESCAPE_KEY = 27
RESET_KEYS = frozenset({ord("r"), ord("R")})


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


class ConfirmationAction(Enum):
    """Represent one final edge-selection review decision."""

    ACCEPT = "accept"
    RETRY = "retry"
    CANCEL = "cancel"


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


@dataclass(frozen=True)
class EdgeSelectionGeometry:
    """Store the four annotations collected by the OpenCV workflow."""

    foreground_roi: RectROI
    background_roi: RectROI
    edge_roi: RectROI
    nominal_edge: Line2D


@dataclass(frozen=True)
class EdgeSelectionFrame:
    """Store the representative frame index and display-only image."""

    frame_index: int
    display_image: np.ndarray


@dataclass(frozen=True)
class DatasetEdgeSelection:
    """Store one representative frame and its collected edge geometry."""

    frame: EdgeSelectionFrame
    geometry: EdgeSelectionGeometry


@dataclass(frozen=True)
class ConfirmedEdgeSelection:
    """Retain the accepted annotations and their validated config."""

    selection: DatasetEdgeSelection
    build_result: EdgeSelectionBuildResult


@dataclass(frozen=True)
class EdgeSelectionOutputPaths:
    """Store the two files created for one accepted selection."""

    roi_path: Path
    preview_path: Path


@dataclass
class LineSelectionState:
    """Track mutable mouse state while the user selects two endpoints."""

    points: list[tuple[int, int]] = field(default_factory=list)
    cursor: tuple[int, int] | None = None


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


def depth_to_edge_display(depth: np.ndarray) -> np.ndarray:
    """Convert one raw depth frame to an annotated-selection BGR image."""
    if not isinstance(depth, np.ndarray):
        raise TypeError(
            f"depth must be a numpy.ndarray; got {type(depth).__name__}"
        )
    if depth.ndim != 2:
        raise ValueError(
            f"depth must have shape (H, W); got shape {depth.shape}"
        )
    if depth.dtype != np.uint16:
        raise ValueError(
            f"depth must have dtype uint16; got {depth.dtype}"
        )

    max_uint16 = np.iinfo(np.uint16).max
    valid = (depth > 0) & (depth < max_uint16)
    if not np.any(valid):
        raise ValueError("Frame contains no displayable depth")

    values = depth[valid]
    lower = float(np.percentile(values, DISPLAY_PERCENTILE_LOW))
    upper = float(np.percentile(values, DISPLAY_PERCENTILE_HIGH))
    if upper <= lower:
        raise ValueError("Invalid display depth range")

    clipped = np.clip(depth.astype(np.float32), lower, upper)
    normalized = (clipped - lower) / (upper - lower) * 255.0
    gray = normalized.astype(np.uint8)
    display = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
    display[~valid] = INVALID_COLOR
    return display


def choose_rectangle(
    display_image: np.ndarray,
    label: str,
) -> RectROI:
    """Select one named rectangle or raise without returning partial data."""
    _validate_display_image(display_image)
    if not isinstance(label, str) or not label:
        raise ValueError("label must be a non-empty string")

    window_name = f"Select {label} ROI"
    try:
        x, y, width, height = cv2.selectROI(
            window_name,
            display_image,
            showCrosshair=True,
            fromCenter=False,
        )
    finally:
        cv2.destroyAllWindows()

    if width <= 0 or height <= 0:
        raise ValueError(
            f"{label} ROI selection was cancelled"
        )
    return RectROI(
        x=int(x),
        y=int(y),
        width=int(width),
        height=int(height),
    )


def choose_nominal_edge(
    display_image: np.ndarray,
) -> Line2D:
    """Collect two line endpoints with reset, accept, and cancel controls."""
    _validate_display_image(display_image)
    base_image = display_image.copy()
    state = LineSelectionState()

    try:
        cv2.namedWindow(LINE_WINDOW_NAME)
        cv2.setMouseCallback(
            LINE_WINDOW_NAME,
            _line_mouse_callback,
            state,
        )
        while True:
            cv2.imshow(
                LINE_WINDOW_NAME,
                render_line_selection(base_image, state),
            )
            key = cv2.waitKey(20)
            if key < 0:
                continue
            key &= 0xFF

            if key == ESCAPE_KEY:
                raise ValueError(
                    "nominal edge selection was cancelled"
                )
            if key in RESET_KEYS:
                state.points.clear()
                state.cursor = None
                continue
            if key in ENTER_KEYS and len(state.points) == 2:
                return Line2D(
                    p1=state.points[0],
                    p2=state.points[1],
                )
    finally:
        cv2.destroyAllWindows()


def render_line_selection(
    base_image: np.ndarray,
    state: LineSelectionState,
) -> np.ndarray:
    """Draw accepted points, a live segment, and keyboard instructions."""
    _validate_display_image(base_image)
    if not isinstance(state, LineSelectionState):
        raise TypeError(
            "state must be a LineSelectionState; "
            f"got {type(state).__name__}"
        )

    rendered = base_image.copy()
    if state.points:
        first = state.points[0]
        cv2.circle(rendered, first, 5, LINE_COLOR, -1)
        cv2.putText(
            rendered,
            "p1",
            (first[0] + 6, first[1] - 6),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            LINE_COLOR,
            1,
            cv2.LINE_AA,
        )

        line_end = (
            state.points[1]
            if len(state.points) == 2
            else state.cursor
        )
        if line_end is not None:
            cv2.line(
                rendered,
                first,
                line_end,
                LINE_COLOR,
                2,
                cv2.LINE_AA,
            )

    if len(state.points) == 2:
        second = state.points[1]
        cv2.circle(rendered, second, 5, LINE_COLOR, -1)
        cv2.putText(
            rendered,
            "p2",
            (second[0] + 6, second[1] - 6),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            LINE_COLOR,
            1,
            cv2.LINE_AA,
        )

    cv2.putText(
        rendered,
        "Click p1/p2 | Enter: accept | R: reset | Esc: cancel",
        (10, 24),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.5,
        LINE_COLOR,
        1,
        cv2.LINE_AA,
    )
    return rendered


def collect_edge_selection(
    display_image: np.ndarray,
) -> EdgeSelectionGeometry:
    """Run the three-rectangle and two-endpoint annotation sequence."""
    _validate_display_image(display_image)

    foreground_roi = choose_rectangle(
        display_image,
        "FOREGROUND reference",
    )
    with_foreground = draw_selection_rectangle(
        display_image,
        foreground_roi,
        "F",
        FOREGROUND_COLOR,
    )

    background_roi = choose_rectangle(
        with_foreground,
        "BACKGROUND reference",
    )
    with_references = draw_selection_rectangle(
        with_foreground,
        background_roi,
        "B",
        BACKGROUND_COLOR,
    )

    edge_roi = choose_rectangle(
        with_references,
        "EDGE analysis",
    )
    with_edge = draw_selection_rectangle(
        with_references,
        edge_roi,
        "EDGE",
        EDGE_COLOR,
    )
    nominal_edge = choose_nominal_edge(with_edge)

    # Fail before configuration construction if the two references cannot
    # define opposite foreground/background half-planes.
    infer_foreground_side(
        nominal_edge,
        foreground_roi,
        background_roi,
    )
    return EdgeSelectionGeometry(
        foreground_roi=foreground_roi,
        background_roi=background_roi,
        edge_roi=edge_roi,
        nominal_edge=nominal_edge,
    )


def draw_selection_rectangle(
    image: np.ndarray,
    roi: RectROI,
    label: str,
    color: tuple[int, int, int],
) -> np.ndarray:
    """Return a copy with one labeled rectangle for staged GUI context."""
    _validate_display_image(image)
    if not isinstance(roi, RectROI):
        raise TypeError(
            f"roi must be a RectROI; got {type(roi).__name__}"
        )
    if not isinstance(label, str) or not label:
        raise ValueError("label must be a non-empty string")
    if (
        not isinstance(color, tuple)
        or len(color) != 3
        or any(
            not isinstance(channel, int)
            or isinstance(channel, bool)
            or not 0 <= channel <= 255
            for channel in color
        )
    ):
        raise ValueError(
            "color must contain three integer BGR channels"
        )

    height, width = image.shape[:2]
    if roi.x + roi.width > width:
        raise ValueError("ROI exceeds image width")
    if roi.y + roi.height > height:
        raise ValueError("ROI exceeds image height")

    rendered = image.copy()
    top_left = (roi.x, roi.y)
    bottom_right = (
        roi.x + roi.width - 1,
        roi.y + roi.height - 1,
    )
    cv2.rectangle(
        rendered,
        top_left,
        bottom_right,
        color,
        2,
    )
    cv2.putText(
        rendered,
        label,
        (roi.x, max(12, roi.y - 5)),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.5,
        color,
        1,
        cv2.LINE_AA,
    )
    return rendered


def render_edge_preview(
    selection: DatasetEdgeSelection,
    build_result: EdgeSelectionBuildResult,
) -> np.ndarray:
    """Render one clean, reproducible edge-selection preview."""
    if not isinstance(selection, DatasetEdgeSelection):
        raise TypeError(
            "selection must be a DatasetEdgeSelection; "
            f"got {type(selection).__name__}"
        )
    if not isinstance(build_result, EdgeSelectionBuildResult):
        raise TypeError(
            "build_result must be an EdgeSelectionBuildResult; "
            f"got {type(build_result).__name__}"
        )

    _validate_display_image(selection.frame.display_image)
    config = build_result.config
    if not isinstance(config, EdgeROIConfig):
        raise TypeError(
            "build_result.config must be an EdgeROIConfig; "
            f"got {type(config).__name__}"
        )
    _validate_preview_consistency(selection, config)
    validate_edge_roi_config(
        config,
        selection.frame.display_image.shape[:2],
    )

    rendered = draw_selection_rectangle(
        selection.frame.display_image,
        config.foreground_roi,
        "F",
        FOREGROUND_COLOR,
    )
    rendered = draw_selection_rectangle(
        rendered,
        config.background_roi,
        "B",
        BACKGROUND_COLOR,
    )
    rendered = draw_selection_rectangle(
        rendered,
        config.edge_roi,
        "EDGE",
        EDGE_COLOR,
    )

    p1 = _rounded_image_point(config.nominal_edge.p1)
    p2 = _rounded_image_point(config.nominal_edge.p2)
    cv2.line(
        rendered,
        p1,
        p2,
        LINE_COLOR,
        2,
        cv2.LINE_AA,
    )
    for label, point in (("p1", p1), ("p2", p2)):
        cv2.circle(rendered, point, 5, LINE_COLOR, -1)
        cv2.putText(
            rendered,
            label,
            (point[0] + 6, point[1] - 6),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            LINE_COLOR,
            1,
            cv2.LINE_AA,
        )

    height = rendered.shape[0]
    status_y = max(18, height - 30)
    cv2.putText(
        rendered,
        (
            f"frame={selection.frame.frame_index} | "
            f"foreground={config.foreground_side} | "
            f"warnings={len(build_result.warnings)}"
        ),
        (10, status_y),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.45,
        LINE_COLOR,
        1,
        cv2.LINE_AA,
    )
    return rendered


def render_edge_selection_overlay(
    selection: DatasetEdgeSelection,
    build_result: EdgeSelectionBuildResult,
) -> np.ndarray:
    """Add interactive review controls to the clean preview."""
    rendered = render_edge_preview(
        selection,
        build_result,
    )
    controls_y = max(18, rendered.shape[0] - 10)
    cv2.putText(
        rendered,
        "Enter: accept | R: reselect | Esc: cancel",
        (10, controls_y),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.45,
        LINE_COLOR,
        1,
        cv2.LINE_AA,
    )
    return rendered


def confirm_edge_selection(
    selection: DatasetEdgeSelection,
    build_result: EdgeSelectionBuildResult,
) -> ConfirmationAction:
    """Show the final review window and return the requested action."""
    review_image = render_edge_selection_overlay(
        selection,
        build_result,
    )
    try:
        cv2.namedWindow(CONFIRM_WINDOW_NAME)
        while True:
            cv2.imshow(CONFIRM_WINDOW_NAME, review_image)
            if (
                cv2.getWindowProperty(
                    CONFIRM_WINDOW_NAME,
                    cv2.WND_PROP_VISIBLE,
                )
                < 1
            ):
                return ConfirmationAction.CANCEL

            key = cv2.waitKey(20)
            if key < 0:
                continue
            key &= 0xFF
            if key in ENTER_KEYS:
                return ConfirmationAction.ACCEPT
            if key in RESET_KEYS:
                return ConfirmationAction.RETRY
            if key == ESCAPE_KEY:
                return ConfirmationAction.CANCEL
    finally:
        cv2.destroyAllWindows()


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


def load_edge_dataset(
    dataset_path: Path,
) -> DepthDataset:
    """Load one non-empty extracted depth dataset for edge selection."""
    resolved_dataset_path = Path(dataset_path).expanduser()
    if not resolved_dataset_path.is_file():
        raise FileNotFoundError(
            f"Cannot find dataset file {resolved_dataset_path}"
        )

    dataset = DepthDataset.load(resolved_dataset_path)
    if dataset.num_frames == 0:
        raise ValueError(
            "Dataset contains no depth frames: "
            f"{resolved_dataset_path}"
        )
    return dataset


def prepare_selection_frame(
    dataset: DepthDataset,
    frame_index: int | None = None,
) -> EdgeSelectionFrame:
    """Select one representative frame and build its display image."""
    if not isinstance(dataset, DepthDataset):
        raise TypeError(
            "dataset must be a DepthDataset; "
            f"got {type(dataset).__name__}"
        )
    if dataset.num_frames == 0:
        raise ValueError("Dataset contains no depth frames")

    if frame_index is None:
        selected_index = dataset.num_frames // 2
    else:
        if not isinstance(frame_index, int) or isinstance(
            frame_index,
            bool,
        ):
            raise TypeError(
                "frame_index must be an integer or None; "
                f"got {type(frame_index).__name__}"
            )
        selected_index = frame_index

    if selected_index < 0 or selected_index >= dataset.num_frames:
        raise ValueError(
            "frame_index must satisfy "
            f"0 <= frame_index < {dataset.num_frames}; "
            f"got {selected_index}"
        )

    return EdgeSelectionFrame(
        frame_index=selected_index,
        display_image=depth_to_edge_display(
            dataset.depth[selected_index]
        ),
    )


def collect_dataset_edge_selection(
    dataset: DepthDataset,
    frame_index: int | None = None,
) -> DatasetEdgeSelection:
    """Prepare one frame and collect all Scene 04 edge annotations."""
    selection_frame = prepare_selection_frame(
        dataset,
        frame_index=frame_index,
    )
    geometry = collect_edge_selection(
        selection_frame.display_image,
    )
    return DatasetEdgeSelection(
        frame=selection_frame,
        geometry=geometry,
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


def build_dataset_edge_config(
    *,
    paths: EdgeSelectionPaths,
    selection: DatasetEdgeSelection,
    options: EdgeAnalysisOptions | None = None,
) -> EdgeSelectionBuildResult:
    """Build a validated config from one collected dataset selection."""
    if not isinstance(paths, EdgeSelectionPaths):
        raise TypeError(
            "paths must be an EdgeSelectionPaths; "
            f"got {type(paths).__name__}"
        )
    if not isinstance(selection, DatasetEdgeSelection):
        raise TypeError(
            "selection must be a DatasetEdgeSelection; "
            f"got {type(selection).__name__}"
        )
    if not isinstance(selection.frame, EdgeSelectionFrame):
        raise TypeError(
            "selection.frame must be an EdgeSelectionFrame; "
            f"got {type(selection.frame).__name__}"
        )
    if not isinstance(
        selection.geometry,
        EdgeSelectionGeometry,
    ):
        raise TypeError(
            "selection.geometry must be an EdgeSelectionGeometry; "
            f"got {type(selection.geometry).__name__}"
        )

    _validate_display_image(selection.frame.display_image)
    geometry = selection.geometry
    return build_edge_roi_config(
        name=paths.roi_key,
        source_experiment=paths.experiment_name,
        source_frame_index=selection.frame.frame_index,
        foreground_roi=geometry.foreground_roi,
        background_roi=geometry.background_roi,
        edge_roi=geometry.edge_roi,
        nominal_edge=geometry.nominal_edge,
        image_shape=selection.frame.display_image.shape[:2],
        options=options,
    )


def select_confirmed_edge_selection(
    *,
    dataset: DepthDataset,
    paths: EdgeSelectionPaths,
    options: EdgeAnalysisOptions | None = None,
    frame_index: int | None = None,
) -> ConfirmedEdgeSelection:
    """Collect, validate, and review selections until accepted or cancelled."""
    while True:
        selection = collect_dataset_edge_selection(
            dataset,
            frame_index=frame_index,
        )
        build_result = build_dataset_edge_config(
            paths=paths,
            selection=selection,
            options=options,
        )

        if build_result.warnings:
            print("Selection warnings:")
            for warning in build_result.warnings:
                print(f"  - {warning}")
            print()

        action = confirm_edge_selection(
            selection,
            build_result,
        )
        if action is ConfirmationAction.ACCEPT:
            return ConfirmedEdgeSelection(
                selection=selection,
                build_result=build_result,
            )
        if action is ConfirmationAction.RETRY:
            continue
        raise ValueError(
            "Edge selection was cancelled; "
            "no configuration saved"
        )


def select_confirmed_edge_config(
    *,
    dataset: DepthDataset,
    paths: EdgeSelectionPaths,
    options: EdgeAnalysisOptions | None = None,
    frame_index: int | None = None,
) -> EdgeSelectionBuildResult:
    """Return only the config result for backward-compatible callers."""
    confirmed = select_confirmed_edge_selection(
        dataset=dataset,
        paths=paths,
        options=options,
        frame_index=frame_index,
    )
    return confirmed.build_result


def save_confirmed_edge_selection(
    *,
    paths: EdgeSelectionPaths,
    confirmed: ConfirmedEdgeSelection,
) -> EdgeSelectionOutputPaths:
    """Save one accepted edge config and preview without overwriting."""
    if not isinstance(paths, EdgeSelectionPaths):
        raise TypeError(
            "paths must be an EdgeSelectionPaths; "
            f"got {type(paths).__name__}"
        )
    if not isinstance(confirmed, ConfirmedEdgeSelection):
        raise TypeError(
            "confirmed must be a ConfirmedEdgeSelection; "
            f"got {type(confirmed).__name__}"
        )
    if not isinstance(
        confirmed.selection,
        DatasetEdgeSelection,
    ):
        raise TypeError(
            "confirmed.selection must be a DatasetEdgeSelection; "
            f"got {type(confirmed.selection).__name__}"
        )
    if not isinstance(
        confirmed.build_result,
        EdgeSelectionBuildResult,
    ):
        raise TypeError(
            "confirmed.build_result must be an "
            "EdgeSelectionBuildResult; "
            f"got {type(confirmed.build_result).__name__}"
        )

    config = confirmed.build_result.config
    if not isinstance(config, EdgeROIConfig):
        raise TypeError(
            "confirmed.build_result.config must be an EdgeROIConfig; "
            f"got {type(config).__name__}"
        )
    if config.name != paths.roi_key:
        raise ValueError(
            "config name does not match the output ROI key"
        )
    if config.source_experiment != paths.experiment_name:
        raise ValueError(
            "config source experiment does not match "
            "the output experiment"
        )

    for output_path in (paths.roi_path, paths.preview_path):
        if output_path.exists():
            raise FileExistsError(
                f"Edge selection output already exists: {output_path}"
            )

    preview = render_edge_preview(
        confirmed.selection,
        confirmed.build_result,
    )
    encode_success, encoded_preview = cv2.imencode(
        ".png",
        preview,
    )
    if not encode_success or encoded_preview is None:
        raise ValueError("Failed to encode edge selection preview")

    preview_created = False
    roi_created = False
    try:
        paths.preview_path.parent.mkdir(
            parents=True,
            exist_ok=True,
        )
        with paths.preview_path.open("xb") as stream:
            preview_created = True
            stream.write(encoded_preview.tobytes())

        save_edge_roi_config(
            paths.roi_path,
            config,
        )
        roi_created = True

        loaded_config = load_edge_roi_config(paths.roi_path)
        if loaded_config != config:
            raise ValueError(
                "Saved edge ROI configuration failed "
                "round-trip verification"
            )

        loaded_preview = cv2.imread(
            str(paths.preview_path),
            cv2.IMREAD_COLOR,
        )
        if (
            loaded_preview is None
            or loaded_preview.shape != preview.shape
            or loaded_preview.dtype != preview.dtype
            or not np.array_equal(loaded_preview, preview)
        ):
            raise ValueError(
                "Saved edge selection preview failed verification"
            )
    except Exception:
        if roi_created:
            _remove_created_output(paths.roi_path)
        if preview_created:
            _remove_created_output(paths.preview_path)
        raise

    return EdgeSelectionOutputPaths(
        roi_path=paths.roi_path,
        preview_path=paths.preview_path,
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


def _rounded_image_point(
    point: tuple[float, float],
) -> tuple[int, int]:
    """Round one floating-point annotation point for OpenCV drawing."""
    return (
        int(round(point[0])),
        int(round(point[1])),
    )


def _validate_preview_consistency(
    selection: DatasetEdgeSelection,
    config: EdgeROIConfig,
) -> None:
    """Reject previews whose image annotations do not match the config."""
    if selection.frame.frame_index != config.source_frame_index:
        raise ValueError(
            "selection frame index does not match "
            "config source frame index"
        )

    for field_name in (
        "foreground_roi",
        "background_roi",
        "edge_roi",
        "nominal_edge",
    ):
        selected_value = getattr(
            selection.geometry,
            field_name,
        )
        configured_value = getattr(config, field_name)
        if selected_value != configured_value:
            raise ValueError(
                f"selection {field_name} does not match "
                f"config {field_name}"
            )


def _remove_created_output(path: Path) -> None:
    """Best-effort cleanup for an output created by the current call."""
    try:
        path.unlink(missing_ok=True)
    except OSError:
        pass


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


def _line_mouse_callback(
    event: int,
    x: int,
    y: int,
    flags: int,
    state: LineSelectionState,
) -> None:
    """Update the two-point selection state from OpenCV mouse events."""
    del flags
    if not isinstance(state, LineSelectionState):
        raise TypeError(
            "mouse callback state must be a LineSelectionState"
        )

    if event == cv2.EVENT_MOUSEMOVE:
        state.cursor = (int(x), int(y))
    elif event == cv2.EVENT_LBUTTONDOWN:
        state.cursor = (int(x), int(y))
        if len(state.points) < 2:
            state.points.append((int(x), int(y)))


def _validate_display_image(image: np.ndarray) -> None:
    """Validate one uint8 BGR image used by the OpenCV selection GUI."""
    if not isinstance(image, np.ndarray):
        raise TypeError(
            f"display image must be a numpy.ndarray; "
            f"got {type(image).__name__}"
        )
    if image.ndim != 3 or image.shape[2] != 3:
        raise ValueError(
            "display image must have shape (H, W, 3)"
        )
    if image.dtype != np.uint8:
        raise ValueError(
            "display image must have dtype uint8"
        )
    if image.shape[0] == 0 or image.shape[1] == 0:
        raise ValueError(
            "display image dimensions must be positive"
        )
