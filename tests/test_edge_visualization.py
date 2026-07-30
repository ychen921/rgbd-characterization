"""Tests for Scene 04 diagnostic visualizations."""

from dataclasses import replace
from io import BytesIO

import numpy as np
import pytest

from src.metrics.edge_discontinuity import (
    DistanceProfileResult,
    EdgePixelLabel,
    FrameEdgeResult,
)
from src.preprocessing.edge_roi import (
    EdgeReferenceConfig,
    EdgeROIConfig,
    EdgeTransitionConfig,
    Line2D,
)
from src.preprocessing.roi import RectROI
from src.visualization.edge import (
    INVALID_DISPLAY_COLOR_RGB,
    LABEL_ALPHA,
    LABEL_COLORS_RGB,
    depth_to_edge_display,
    draw_edge_roi_overlay,
    plot_edge_label_map,
    plot_edge_probability_profile,
    plot_edge_temporal_metrics,
)


def _config() -> EdgeROIConfig:
    return EdgeROIConfig(
        name="scene04_gap030_vertical_white_d100",
        source_experiment=(
            "scene04_gap030_vertical_white_d100_r01"
        ),
        source_frame_index=5,
        foreground_roi=RectROI(
            x=2,
            y=5,
            width=10,
            height=20,
        ),
        background_roi=RectROI(
            x=45,
            y=5,
            width=10,
            height=20,
        ),
        edge_roi=RectROI(
            x=24,
            y=5,
            width=12,
            height=20,
        ),
        nominal_edge=Line2D(
            p1=(30.0, 5.0),
            p2=(30.0, 25.0),
        ),
        foreground_side="left",
        distance_bin_px=1.0,
        max_edge_distance_px=5.0,
        reference=EdgeReferenceConfig(
            minimum_tolerance_mm=10.0,
            mad_scale=3.0,
            minimum_valid_count=1,
        ),
        transition=EdgeTransitionConfig(
            high_probability=0.9,
            low_probability=0.1,
        ),
    )


def _profile() -> DistanceProfileResult:
    distance = np.asarray(
        [-2.0, -1.0, 0.0, 1.0, 2.0],
        dtype=np.float64,
    )
    pixel_count = np.full(5, 10, dtype=np.int64)
    valid_count = np.asarray(
        [10, 10, 9, 10, 10],
        dtype=np.int64,
    )
    foreground_count = np.asarray(
        [10, 9, 4, 1, 0],
        dtype=np.int64,
    )
    background_count = np.asarray(
        [0, 1, 4, 9, 10],
        dtype=np.int64,
    )
    mixed_count = np.asarray(
        [0, 0, 1, 0, 0],
        dtype=np.int64,
    )
    outlier_count = np.zeros(5, dtype=np.int64)
    invalid_count = pixel_count - valid_count

    def ratio(
        numerator: np.ndarray,
        denominator: np.ndarray,
    ) -> np.ndarray:
        values = np.full(
            denominator.shape,
            np.nan,
            dtype=np.float64,
        )
        np.divide(
            numerator,
            denominator,
            out=values,
            where=denominator > 0,
        )
        return values

    return DistanceProfileResult(
        distance_min_px=distance - 0.5,
        distance_max_px=distance + 0.5,
        distance_center_px=distance,
        pixel_count=pixel_count,
        valid_count=valid_count,
        foreground_count=foreground_count,
        background_count=background_count,
        mixed_count=mixed_count,
        outlier_count=outlier_count,
        invalid_count=invalid_count,
        foreground_ratio=ratio(
            foreground_count,
            valid_count,
        ),
        background_ratio=ratio(
            background_count,
            valid_count,
        ),
        mixed_ratio=ratio(mixed_count, valid_count),
        outlier_ratio=ratio(outlier_count, valid_count),
        invalid_ratio=ratio(invalid_count, pixel_count),
    )


def _frame_result(
    frame_index: int,
    *,
    analysis_status: str = "ok",
    transition_status: str = "ok",
) -> FrameEdgeResult:
    value = (
        float(frame_index) / 100.0
        if analysis_status == "ok"
        else float("nan")
    )
    transition_value = (
        float(frame_index)
        if (
            analysis_status == "ok"
            and transition_status == "ok"
        )
        else float("nan")
    )
    return FrameEdgeResult(
        frame_index=frame_index,
        foreground_reference_mm=1000.0 + frame_index,
        background_reference_mm=1300.0 + frame_index,
        foreground_bleeding_ratio=value,
        foreground_bleeding_max_distance_px=value,
        background_bleeding_ratio=value,
        background_bleeding_max_distance_px=value,
        mixed_ratio=value,
        peak_mixed_ratio=value,
        peak_mixed_distance_px=value,
        outlier_ratio=value,
        invalid_ratio=value,
        invalid_band_width_px=transition_value,
        transition_width_px=transition_value,
        nominal_edge_offset_px=transition_value,
        analysis_status=analysis_status,
        transition_status=transition_status,
    )


def _assert_png_serializable(figure: object) -> None:
    output = BytesIO()
    figure.savefig(output, format="png")
    assert output.getbuffer().nbytes > 0


def test_depth_to_edge_display_normalizes_valid_depth() -> None:
    depth = np.asarray(
        [
            [1000, 1100, 1200],
            [1000, 1100, 1200],
        ],
        dtype=np.uint16,
    )

    display = depth_to_edge_display(depth)

    assert display.shape == (2, 3, 3)
    assert display.dtype == np.uint8
    np.testing.assert_array_equal(
        display[:, :, 0],
        display[:, :, 1],
    )
    assert int(display[0, 0, 0]) < int(display[0, 2, 0])


def test_depth_to_edge_display_marks_special_values() -> None:
    depth = np.asarray(
        [[0, 1000, 1200, np.iinfo(np.uint16).max]],
        dtype=np.uint16,
    )

    display = depth_to_edge_display(depth)

    np.testing.assert_array_equal(
        display[0, 0],
        INVALID_DISPLAY_COLOR_RGB,
    )
    np.testing.assert_array_equal(
        display[0, 3],
        INVALID_DISPLAY_COLOR_RGB,
    )


def test_depth_to_edge_display_handles_constant_and_all_invalid() -> None:
    constant = np.full((2, 3), 1000, dtype=np.uint16)
    all_invalid = np.zeros((2, 3), dtype=np.uint16)

    constant_display = depth_to_edge_display(constant)
    invalid_display = depth_to_edge_display(all_invalid)

    assert np.all(constant_display == 127)
    assert np.all(
        invalid_display
        == np.asarray(INVALID_DISPLAY_COLOR_RGB)
    )


@pytest.mark.parametrize(
    ("depth", "error_type", "message"),
    [
        (
            [[1000]],
            TypeError,
            "numpy array",
        ),
        (
            np.zeros((1, 2, 3), dtype=np.uint16),
            ValueError,
            "shape",
        ),
        (
            np.zeros((2, 3), dtype=np.float32),
            ValueError,
            "dtype uint16",
        ),
    ],
)
def test_depth_to_edge_display_rejects_invalid_input(
    depth: object,
    error_type: type[Exception],
    message: str,
) -> None:
    with pytest.raises(error_type, match=message):
        depth_to_edge_display(depth)


def test_draw_edge_roi_overlay_returns_rgb_copy() -> None:
    image = np.full((40, 60), 120, dtype=np.uint8)
    original = image.copy()

    rendered = draw_edge_roi_overlay(image, _config())

    assert rendered.shape == (40, 60, 3)
    assert rendered.dtype == np.uint8
    np.testing.assert_array_equal(image, original)
    assert not np.array_equal(
        rendered[5, 2],
        np.asarray([120, 120, 120]),
    )
    assert not np.array_equal(
        rendered[15, 30],
        np.asarray([120, 120, 120]),
    )


def test_draw_edge_roi_overlay_copies_rgb_input() -> None:
    image = np.full((40, 60, 3), 90, dtype=np.uint8)
    original = image.copy()

    rendered = draw_edge_roi_overlay(image, _config())

    assert not np.shares_memory(rendered, image)
    np.testing.assert_array_equal(image, original)


@pytest.mark.parametrize(
    ("image", "error_type", "message"),
    [
        (
            [[0]],
            TypeError,
            "numpy array",
        ),
        (
            np.zeros((40, 60), dtype=np.float32),
            ValueError,
            "dtype uint8",
        ),
        (
            np.zeros((40, 60, 4), dtype=np.uint8),
            ValueError,
            "shape",
        ),
    ],
)
def test_draw_edge_roi_overlay_rejects_invalid_images(
    image: object,
    error_type: type[Exception],
    message: str,
) -> None:
    with pytest.raises(error_type, match=message):
        draw_edge_roi_overlay(image, _config())


def test_plot_edge_label_map_blends_fixed_colors_and_keeps_outside() -> None:
    image = np.full((40, 60), 120, dtype=np.uint8)
    labels = np.full(
        image.shape,
        int(EdgePixelLabel.OUTSIDE),
        dtype=np.uint8,
    )
    positions = {
        EdgePixelLabel.INVALID: (10, 25),
        EdgePixelLabel.FOREGROUND: (10, 26),
        EdgePixelLabel.BACKGROUND: (10, 27),
        EdgePixelLabel.MIXED: (10, 28),
        EdgePixelLabel.OUTLIER: (10, 29),
    }
    for label, position in positions.items():
        labels[position] = int(label)

    figure = plot_edge_label_map(
        image,
        labels,
        _config(),
    )
    plotted = np.asarray(figure.axes[0].images[0].get_array())

    np.testing.assert_array_equal(
        plotted[0, 0],
        np.asarray([120, 120, 120]),
    )
    for label, position in positions.items():
        expected = np.rint(
            (1.0 - LABEL_ALPHA) * 120.0
            + LABEL_ALPHA * np.asarray(
                LABEL_COLORS_RGB[label],
                dtype=np.float64,
            )
        ).astype(np.uint8)
        np.testing.assert_array_equal(
            plotted[position],
            expected,
        )

    legend_labels = {
        text.get_text()
        for text in figure.axes[0].get_legend().get_texts()
    }
    assert legend_labels == {
        "Outside",
        "Invalid",
        "Foreground",
        "Background",
        "Mixed",
        "Outlier",
    }
    _assert_png_serializable(figure)


@pytest.mark.parametrize(
    ("label_map", "message"),
    [
        (
            np.zeros((39, 60), dtype=np.uint8),
            "match display_image",
        ),
        (
            np.zeros((40, 60), dtype=np.float64),
            "integer dtype",
        ),
        (
            np.full((40, 60), 99, dtype=np.int16),
            "unknown edge label",
        ),
    ],
)
def test_plot_edge_label_map_rejects_invalid_labels(
    label_map: np.ndarray,
    message: str,
) -> None:
    image = np.zeros((40, 60), dtype=np.uint8)

    with pytest.raises(ValueError, match=message):
        plot_edge_label_map(
            image,
            label_map,
            _config(),
        )


def test_plot_edge_probability_profile_draws_required_series() -> None:
    figure = plot_edge_probability_profile(_profile())
    axis = figure.axes[0]

    labels = {
        line.get_label()
        for line in axis.lines
    }
    assert labels == {
        "Foreground",
        "Background",
        "Mixed",
        "Outlier",
        "Invalid",
        "Nominal edge",
    }
    assert axis.get_ylim() == pytest.approx((0.0, 1.0))
    assert "Signed distance" in axis.get_xlabel()
    _assert_png_serializable(figure)


def test_plot_edge_probability_profile_accepts_nan_ratio() -> None:
    profile = _profile()
    foreground_ratio = profile.foreground_ratio.copy()
    foreground_ratio[2] = np.nan
    profile = replace(
        profile,
        foreground_ratio=foreground_ratio,
    )

    figure = plot_edge_probability_profile(profile)

    plotted = figure.axes[0].lines[0].get_ydata()
    assert np.isnan(plotted[2])
    _assert_png_serializable(figure)


@pytest.mark.parametrize(
    ("profile", "message"),
    [
        (
            replace(
                _profile(),
                mixed_ratio=np.zeros(4, dtype=np.float64),
            ),
            "same length",
        ),
        (
            replace(
                _profile(),
                distance_center_px=np.asarray(
                    [-2.0, -1.0, -1.0, 1.0, 2.0]
                ),
            ),
            "strictly increasing",
        ),
        (
            replace(
                _profile(),
                invalid_ratio=np.asarray(
                    [0.0, 0.0, np.inf, 0.0, 0.0]
                ),
            ),
            "ratios in",
        ),
    ],
)
def test_plot_edge_probability_profile_rejects_invalid_profile(
    profile: DistanceProfileResult,
    message: str,
) -> None:
    with pytest.raises(ValueError, match=message):
        plot_edge_probability_profile(profile)


def test_plot_edge_temporal_metrics_builds_four_panels() -> None:
    results = [
        _frame_result(2, transition_status="missing_crossing"),
        _frame_result(0),
        _frame_result(1, analysis_status="insufficient_reference"),
    ]

    figure = plot_edge_temporal_metrics(results)

    assert len(figure.axes) == 4
    assert [axis.get_title() for axis in figure.axes] == [
        "Reference depth",
        "Edge classification ratios",
        "Edge width metrics",
        "Offset from nominal edge",
    ]
    reference_x = figure.axes[0].lines[0].get_xdata()
    np.testing.assert_array_equal(
        reference_x,
        np.asarray([0, 1, 2]),
    )
    all_labels = {
        line.get_label()
        for axis in figure.axes
        for line in axis.lines
    }
    assert "Rejected frame" in all_labels
    assert "Failed transition" in all_labels
    _assert_png_serializable(figure)


def test_plot_edge_temporal_metrics_accepts_all_rejected_frames() -> None:
    results = [
        _frame_result(
            0,
            analysis_status="insufficient_foreground_reference",
            transition_status="not_analyzed",
        ),
        _frame_result(
            1,
            analysis_status="ambiguous_reference_overlap",
            transition_status="not_analyzed",
        ),
    ]

    figure = plot_edge_temporal_metrics(results)

    assert len(figure.axes) == 4
    _assert_png_serializable(figure)


@pytest.mark.parametrize(
    ("results", "error_type", "message"),
    [
        (
            [],
            ValueError,
            "must not be empty",
        ),
        (
            [object()],
            TypeError,
            "FrameEdgeResult",
        ),
        (
            [_frame_result(0), _frame_result(0)],
            ValueError,
            "unique frame indices",
        ),
    ],
)
def test_plot_edge_temporal_metrics_rejects_invalid_input(
    results: list[object],
    error_type: type[Exception],
    message: str,
) -> None:
    with pytest.raises(error_type, match=message):
        plot_edge_temporal_metrics(results)
