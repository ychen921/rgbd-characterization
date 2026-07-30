"""Diagnostic visualizations for Scene 04 edge analysis."""

from collections.abc import Sequence

import cv2
from matplotlib.figure import Figure
from matplotlib.patches import Patch, Rectangle
import numpy as np

from src.metrics.edge_discontinuity import (
    DistanceProfileResult,
    EdgePixelLabel,
    FrameEdgeResult,
)
from src.preprocessing.edge_roi import (
    EdgeROIConfig,
    validate_edge_roi_config,
)
from src.preprocessing.roi import RectROI


FOREGROUND_COLOR_RGB = (0, 180, 0)
BACKGROUND_COLOR_RGB = (0, 105, 255)
EDGE_COLOR_RGB = (255, 210, 0)
NOMINAL_EDGE_COLOR_RGB = (255, 0, 255)

LABEL_COLORS_RGB = {
    EdgePixelLabel.OUTSIDE: (255, 255, 255),
    EdgePixelLabel.INVALID: (0, 0, 0),
    EdgePixelLabel.FOREGROUND: FOREGROUND_COLOR_RGB,
    EdgePixelLabel.BACKGROUND: BACKGROUND_COLOR_RGB,
    EdgePixelLabel.MIXED: (255, 140, 0),
    EdgePixelLabel.OUTLIER: (190, 0, 190),
}
LABEL_ALPHA = 0.62
DISPLAY_PERCENTILE_LOW = 1.0
DISPLAY_PERCENTILE_HIGH = 99.0
INVALID_DISPLAY_COLOR_RGB = (255, 0, 255)

_PROFILE_SERIES = (
    (
        "foreground_ratio",
        "Foreground",
        FOREGROUND_COLOR_RGB,
        "-",
    ),
    (
        "background_ratio",
        "Background",
        BACKGROUND_COLOR_RGB,
        "-",
    ),
    (
        "mixed_ratio",
        "Mixed",
        LABEL_COLORS_RGB[EdgePixelLabel.MIXED],
        "-",
    ),
    (
        "outlier_ratio",
        "Outlier",
        LABEL_COLORS_RGB[EdgePixelLabel.OUTLIER],
        "-",
    ),
    (
        "invalid_ratio",
        "Invalid",
        LABEL_COLORS_RGB[EdgePixelLabel.INVALID],
        "--",
    ),
)


def depth_to_edge_display(
    raw_depth_frame: np.ndarray,
) -> np.ndarray:
    """Convert one raw uint16 depth frame to a diagnostic RGB image."""
    if not isinstance(raw_depth_frame, np.ndarray):
        raise TypeError(
            "raw_depth_frame must be a numpy array"
        )
    if raw_depth_frame.ndim != 2:
        raise ValueError(
            "raw_depth_frame must have shape (H, W)"
        )
    if raw_depth_frame.dtype != np.uint16:
        raise ValueError(
            "raw_depth_frame must have dtype uint16"
        )

    max_uint16 = np.iinfo(np.uint16).max
    valid = (
        (raw_depth_frame > 0)
        & (raw_depth_frame < max_uint16)
    )
    gray = np.zeros(
        raw_depth_frame.shape,
        dtype=np.uint8,
    )
    if np.any(valid):
        values = raw_depth_frame[valid]
        lower = float(
            np.percentile(values, DISPLAY_PERCENTILE_LOW)
        )
        upper = float(
            np.percentile(values, DISPLAY_PERCENTILE_HIGH)
        )
        if upper > lower:
            clipped = np.clip(
                raw_depth_frame.astype(np.float32),
                lower,
                upper,
            )
            normalized = (
                (clipped - lower)
                / (upper - lower)
                * 255.0
            )
            gray = normalized.astype(np.uint8)
        else:
            gray[valid] = 127

    display = np.repeat(
        gray[:, :, np.newaxis],
        3,
        axis=2,
    )
    display[~valid] = INVALID_DISPLAY_COLOR_RGB
    return display


def draw_edge_roi_overlay(
    display_image: np.ndarray,
    config: EdgeROIConfig,
) -> np.ndarray:
    """Return an RGB image annotated with Scene 04 ROI geometry."""
    rendered = _normalize_display_image(display_image)
    validate_edge_roi_config(config, rendered.shape[:2])

    annotations = (
        (
            config.foreground_roi,
            "F",
            FOREGROUND_COLOR_RGB,
        ),
        (
            config.background_roi,
            "B",
            BACKGROUND_COLOR_RGB,
        ),
        (
            config.edge_roi,
            "EDGE",
            EDGE_COLOR_RGB,
        ),
    )
    for roi, label, color in annotations:
        _draw_rectangle(rendered, roi, label, color)

    p1 = _rounded_point(config.nominal_edge.p1)
    p2 = _rounded_point(config.nominal_edge.p2)
    cv2.line(
        rendered,
        p1,
        p2,
        NOMINAL_EDGE_COLOR_RGB,
        2,
        cv2.LINE_AA,
    )
    for label, point in (("p1", p1), ("p2", p2)):
        cv2.circle(
            rendered,
            point,
            4,
            NOMINAL_EDGE_COLOR_RGB,
            -1,
        )
        cv2.putText(
            rendered,
            label,
            (point[0] + 5, max(10, point[1] - 5)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.4,
            NOMINAL_EDGE_COLOR_RGB,
            1,
            cv2.LINE_AA,
        )

    cv2.putText(
        rendered,
        f"foreground side: {config.foreground_side}",
        (8, max(14, rendered.shape[0] - 8)),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.45,
        NOMINAL_EDGE_COLOR_RGB,
        1,
        cv2.LINE_AA,
    )
    return rendered


def plot_edge_label_map(
    display_image: np.ndarray,
    label_map: np.ndarray,
    config: EdgeROIConfig,
) -> Figure:
    """Plot a representative edge-label overlay with a fixed legend."""
    base_image = _normalize_display_image(display_image)
    validate_edge_roi_config(config, base_image.shape[:2])
    labels = _validate_label_map(
        label_map,
        image_shape=base_image.shape[:2],
    )

    overlay = base_image.astype(np.float64)
    for label in EdgePixelLabel:
        if label == EdgePixelLabel.OUTSIDE:
            continue
        mask = labels == int(label)
        if not np.any(mask):
            continue
        color = np.asarray(
            LABEL_COLORS_RGB[label],
            dtype=np.float64,
        )
        overlay[mask] = (
            (1.0 - LABEL_ALPHA) * overlay[mask]
            + LABEL_ALPHA * color
        )
    overlay = np.clip(
        np.rint(overlay),
        0,
        255,
    ).astype(np.uint8)

    figure = Figure(figsize=_image_figure_size(overlay.shape[:2]))
    axis = figure.subplots()
    axis.imshow(overlay, origin="upper")

    edge_roi = config.edge_roi
    axis.add_patch(
        Rectangle(
            (edge_roi.x - 0.5, edge_roi.y - 0.5),
            edge_roi.width,
            edge_roi.height,
            fill=False,
            edgecolor=_matplotlib_color(EDGE_COLOR_RGB),
            linewidth=1.5,
            label="Edge ROI",
        )
    )
    p1 = config.nominal_edge.p1
    p2 = config.nominal_edge.p2
    axis.plot(
        (p1[0], p2[0]),
        (p1[1], p2[1]),
        color=_matplotlib_color(NOMINAL_EDGE_COLOR_RGB),
        linewidth=1.5,
        label="Nominal edge",
    )

    legend_handles = [
        Patch(
            facecolor="none",
            edgecolor="0.5",
            label="Outside",
        ),
        *[
            Patch(
                facecolor=_matplotlib_color(
                    LABEL_COLORS_RGB[label]
                ),
                alpha=LABEL_ALPHA,
                label=label.name.title(),
            )
            for label in EdgePixelLabel
            if label != EdgePixelLabel.OUTSIDE
        ],
    ]
    axis.legend(
        handles=legend_handles,
        loc="upper left",
        bbox_to_anchor=(1.02, 1.0),
        borderaxespad=0.0,
    )
    axis.set_title(
        "Edge pixel classification "
        f"(foreground side: {config.foreground_side})"
    )
    axis.set_xlabel("Image x (px)")
    axis.set_ylabel("Image y (px)")
    figure.tight_layout()
    return figure


def plot_edge_probability_profile(
    profile: DistanceProfileResult,
) -> Figure:
    """Plot edge-class probabilities against signed distance."""
    _validate_profile_for_plot(profile)

    figure = Figure(figsize=(9.0, 5.5))
    axis = figure.subplots()
    distance = profile.distance_center_px
    for field_name, label, color, linestyle in _PROFILE_SERIES:
        axis.plot(
            distance,
            getattr(profile, field_name),
            label=label,
            color=_matplotlib_color(color),
            linestyle=linestyle,
            linewidth=1.8,
        )

    axis.axvline(
        0.0,
        color=_matplotlib_color(NOMINAL_EDGE_COLOR_RGB),
        linestyle=":",
        linewidth=1.5,
        label="Nominal edge",
    )
    axis.set_ylim(0.0, 1.0)
    axis.set_xlabel(
        "Signed distance from nominal edge (px)\n"
        "negative: foreground side; positive: background side"
    )
    axis.set_ylabel("Ratio")
    axis.set_title("Aggregate edge probability profile")
    axis.grid(True, alpha=0.25)
    axis.legend(loc="best")
    axis.text(
        0.01,
        -0.27,
        (
            "Foreground/background/mixed/outlier ratios use valid "
            "pixels; invalid ratio uses all pixels."
        ),
        transform=axis.transAxes,
        fontsize=8,
        va="top",
    )
    figure.subplots_adjust(bottom=0.28)
    return figure


def plot_edge_temporal_metrics(
    frame_results: Sequence[FrameEdgeResult],
) -> Figure:
    """Plot frame-level reference depths and edge-quality metrics."""
    ordered_results = _validate_frame_results(frame_results)
    frame_index = np.asarray(
        [result.frame_index for result in ordered_results],
        dtype=np.int64,
    )

    figure = Figure(figsize=(11.0, 12.0))
    axes = figure.subplots(
        4,
        1,
        sharex=True,
    )

    _plot_result_fields(
        axes[0],
        frame_index,
        ordered_results,
        (
            (
                "foreground_reference_mm",
                "Foreground reference",
                FOREGROUND_COLOR_RGB,
                "-",
            ),
            (
                "background_reference_mm",
                "Background reference",
                BACKGROUND_COLOR_RGB,
                "-",
            ),
        ),
    )
    axes[0].set_ylabel("Depth (mm)")
    axes[0].set_title("Reference depth")

    _plot_result_fields(
        axes[1],
        frame_index,
        ordered_results,
        (
            (
                "foreground_bleeding_ratio",
                "Foreground bleeding",
                FOREGROUND_COLOR_RGB,
                "-",
            ),
            (
                "background_bleeding_ratio",
                "Background bleeding",
                BACKGROUND_COLOR_RGB,
                "-",
            ),
            (
                "mixed_ratio",
                "Mixed",
                LABEL_COLORS_RGB[EdgePixelLabel.MIXED],
                "-",
            ),
            (
                "outlier_ratio",
                "Outlier",
                LABEL_COLORS_RGB[EdgePixelLabel.OUTLIER],
                "-",
            ),
            (
                "invalid_ratio",
                "Invalid",
                LABEL_COLORS_RGB[EdgePixelLabel.INVALID],
                "--",
            ),
        ),
    )
    axes[1].set_ylim(0.0, 1.0)
    axes[1].set_ylabel("Ratio")
    axes[1].set_title("Edge classification ratios")

    _plot_result_fields(
        axes[2],
        frame_index,
        ordered_results,
        (
            (
                "transition_width_px",
                "Transition width",
                EDGE_COLOR_RGB,
                "-",
            ),
            (
                "invalid_band_width_px",
                "Invalid-band width",
                LABEL_COLORS_RGB[EdgePixelLabel.INVALID],
                "--",
            ),
        ),
    )
    axes[2].set_ylabel("Width (px)")
    axes[2].set_title("Edge width metrics")

    _plot_result_fields(
        axes[3],
        frame_index,
        ordered_results,
        (
            (
                "nominal_edge_offset_px",
                "Nominal edge offset",
                NOMINAL_EDGE_COLOR_RGB,
                "-",
            ),
        ),
    )
    axes[3].axhline(
        0.0,
        color="0.4",
        linewidth=1.0,
        linestyle=":",
    )
    axes[3].set_ylabel("Offset (px)")
    axes[3].set_title("Offset from nominal edge")
    axes[3].set_xlabel("Frame index")

    rejected = [
        result.frame_index
        for result in ordered_results
        if result.analysis_status != "ok"
    ]
    failed_transition = [
        result.frame_index
        for result in ordered_results
        if (
            result.analysis_status == "ok"
            and result.transition_status != "ok"
        )
    ]
    _mark_frame_statuses(
        axes,
        rejected=rejected,
        failed_transition=failed_transition,
    )

    for axis in axes:
        axis.grid(True, alpha=0.25)
        axis.legend(loc="best")

    figure.tight_layout()
    return figure


def _normalize_display_image(
    display_image: np.ndarray,
) -> np.ndarray:
    """Validate and copy one uint8 grayscale or RGB display image."""
    if not isinstance(display_image, np.ndarray):
        raise TypeError("display_image must be a numpy array")
    if display_image.dtype != np.uint8:
        raise ValueError("display_image must have dtype uint8")
    if display_image.ndim == 2:
        return np.repeat(
            display_image[:, :, np.newaxis],
            3,
            axis=2,
        )
    if (
        display_image.ndim == 3
        and display_image.shape[2] == 3
    ):
        return display_image.copy()
    raise ValueError(
        "display_image must have shape (H, W) or (H, W, 3)"
    )


def _draw_rectangle(
    image: np.ndarray,
    roi: RectROI,
    label: str,
    color: tuple[int, int, int],
) -> None:
    """Draw one ROI rectangle and label in place."""
    top_left = (roi.x, roi.y)
    bottom_right = (
        roi.x + roi.width - 1,
        roi.y + roi.height - 1,
    )
    cv2.rectangle(
        image,
        top_left,
        bottom_right,
        color,
        2,
    )
    cv2.putText(
        image,
        label,
        (roi.x, max(12, roi.y - 4)),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.45,
        color,
        1,
        cv2.LINE_AA,
    )


def _rounded_point(
    point: tuple[float, float],
) -> tuple[int, int]:
    """Round an annotation point for OpenCV drawing."""
    return (
        int(round(point[0])),
        int(round(point[1])),
    )


def _validate_label_map(
    label_map: np.ndarray,
    image_shape: tuple[int, int],
) -> np.ndarray:
    """Validate one full-image edge label map."""
    if not isinstance(label_map, np.ndarray):
        raise TypeError("label_map must be a numpy array")
    if label_map.shape != image_shape:
        raise ValueError(
            "label_map must match display_image height and width"
        )
    if (
        not np.issubdtype(label_map.dtype, np.integer)
        or np.issubdtype(label_map.dtype, np.bool_)
    ):
        raise ValueError("label_map must have an integer dtype")

    valid_values = np.asarray(
        [int(label) for label in EdgePixelLabel],
        dtype=np.int64,
    )
    if not np.all(np.isin(label_map, valid_values)):
        raise ValueError("label_map contains an unknown edge label")
    return label_map


def _validate_profile_for_plot(
    profile: DistanceProfileResult,
) -> None:
    """Validate the structure and plot-safe values of a profile."""
    if not isinstance(profile, DistanceProfileResult):
        raise TypeError(
            "profile must be a DistanceProfileResult"
        )

    arrays: dict[str, np.ndarray] = {}
    expected_length: int | None = None
    for field_name in DistanceProfileResult.__dataclass_fields__:
        value = getattr(profile, field_name)
        if not isinstance(value, np.ndarray) or value.ndim != 1:
            raise ValueError(
                f"profile {field_name} must be a one-dimensional "
                "numpy array"
            )
        if expected_length is None:
            expected_length = len(value)
        elif len(value) != expected_length:
            raise ValueError(
                "all profile arrays must have the same length"
            )
        arrays[field_name] = value

    if expected_length == 0:
        raise ValueError("profile must contain at least one bin")

    distance = arrays["distance_center_px"]
    if not np.all(np.isfinite(distance)):
        raise ValueError(
            "profile distance_center_px must be finite"
        )
    if distance.size > 1 and np.any(np.diff(distance) <= 0):
        raise ValueError(
            "profile distance_center_px must be strictly increasing"
        )

    for field_name, _, _, _ in _PROFILE_SERIES:
        ratio = np.asarray(arrays[field_name], dtype=np.float64)
        finite = np.isfinite(ratio)
        if np.any(np.isinf(ratio)) or np.any(
            (ratio[finite] < 0.0)
            | (ratio[finite] > 1.0)
        ):
            raise ValueError(
                f"profile {field_name} must contain ratios in [0, 1] "
                "or NaN"
            )


def _validate_frame_results(
    frame_results: Sequence[FrameEdgeResult],
) -> tuple[FrameEdgeResult, ...]:
    """Validate and order frame results by frame index."""
    if isinstance(frame_results, (str, bytes)):
        raise TypeError(
            "frame_results must be a sequence of FrameEdgeResult"
        )
    try:
        results = tuple(frame_results)
    except TypeError as error:
        raise TypeError(
            "frame_results must be a sequence of FrameEdgeResult"
        ) from error
    if not results:
        raise ValueError("frame_results must not be empty")
    for result in results:
        if not isinstance(result, FrameEdgeResult):
            raise TypeError(
                "frame_results must contain FrameEdgeResult values"
            )

    ordered = tuple(
        sorted(results, key=lambda result: result.frame_index)
    )
    frame_indices = [
        result.frame_index
        for result in ordered
    ]
    if len(set(frame_indices)) != len(frame_indices):
        raise ValueError(
            "frame_results must have unique frame indices"
        )
    return ordered


def _plot_result_fields(
    axis: object,
    frame_index: np.ndarray,
    results: tuple[FrameEdgeResult, ...],
    series: tuple[
        tuple[
            str,
            str,
            tuple[int, int, int],
            str,
        ],
        ...,
    ],
) -> None:
    """Plot selected FrameEdgeResult fields on one axis."""
    for field_name, label, color, linestyle in series:
        values = np.asarray(
            [
                getattr(result, field_name)
                for result in results
            ],
            dtype=np.float64,
        )
        axis.plot(
            frame_index,
            values,
            label=label,
            color=_matplotlib_color(color),
            linestyle=linestyle,
            linewidth=1.4,
        )


def _mark_frame_statuses(
    axes: np.ndarray,
    rejected: list[int],
    failed_transition: list[int],
) -> None:
    """Mark rejected analyses and failed transition estimates."""
    for axis_index, axis in enumerate(axes):
        for marker_index, frame_index in enumerate(rejected):
            axis.axvline(
                frame_index,
                color=(0.8, 0.1, 0.1),
                linestyle=":",
                linewidth=0.9,
                alpha=0.45,
                label=(
                    "Rejected frame"
                    if axis_index == 0 and marker_index == 0
                    else None
                ),
            )

    for axis_index in (2, 3):
        axis = axes[axis_index]
        for marker_index, frame_index in enumerate(
            failed_transition
        ):
            axis.axvline(
                frame_index,
                color=_matplotlib_color(
                    LABEL_COLORS_RGB[EdgePixelLabel.MIXED]
                ),
                linestyle=":",
                linewidth=0.9,
                alpha=0.55,
                label=(
                    "Failed transition"
                    if marker_index == 0
                    else None
                ),
            )


def _matplotlib_color(
    color: tuple[int, int, int],
) -> tuple[float, float, float]:
    """Convert one 8-bit RGB color for Matplotlib."""
    return tuple(channel / 255.0 for channel in color)


def _image_figure_size(
    image_shape: tuple[int, int],
) -> tuple[float, float]:
    """Choose a readable figure size while preserving image aspect."""
    height, width = image_shape
    figure_width = 8.0
    figure_height = min(
        10.0,
        max(4.0, figure_width * height / width),
    )
    return (figure_width, figure_height)
