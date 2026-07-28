"""Tests for pure Scene 04 edge ROI selection logic."""

from dataclasses import replace
from pathlib import Path

import pytest

from src.preprocessing.edge_roi import Line2D
from src.preprocessing.roi import RectROI
from tools import select_edge_roi as select_edge_roi_tool


def _selection_geometry() -> dict[str, object]:
    return {
        "name": "scene04_edge_d050",
        "source_experiment": "scene04_edge_d050_r01",
        "source_frame_index": 4,
        "foreground_roi": RectROI(
            x=0,
            y=1,
            width=2,
            height=6,
        ),
        "background_roi": RectROI(
            x=10,
            y=1,
            width=2,
            height=6,
        ),
        "edge_roi": RectROI(
            x=3,
            y=1,
            width=7,
            height=6,
        ),
        "nominal_edge": Line2D(
            p1=(6.0, 1.0),
            p2=(6.0, 7.0),
        ),
        "image_shape": (10, 12),
    }


def _small_options(
    **changes: object,
) -> select_edge_roi_tool.EdgeAnalysisOptions:
    values = {
        "distance_bin_px": 1.0,
        "max_edge_distance_px": 2.0,
        "minimum_tolerance_mm": 10.0,
        "mad_scale": 3.0,
        "minimum_valid_ratio": 0.9,
        "minimum_valid_count": 1,
        "bleeding_probability_threshold": 0.05,
        "invalid_ratio_threshold": 0.5,
        "transition_high_probability": 0.9,
        "transition_low_probability": 0.1,
    }
    values.update(changes)
    return select_edge_roi_tool.EdgeAnalysisOptions(**values)


def test_parse_args_uses_reproducible_defaults() -> None:
    args = select_edge_roi_tool.parse_args(
        ["data/scene04_edge_d050_r01"]
    )
    options = select_edge_roi_tool.analysis_options_from_args(args)

    assert args.dataset_dir == Path(
        "data/scene04_edge_d050_r01"
    )
    assert args.frame_index is None
    assert (
        options
        == select_edge_roi_tool.DEFAULT_ANALYSIS_OPTIONS
    )


def test_parse_args_accepts_frame_and_analysis_overrides() -> None:
    args = select_edge_roi_tool.parse_args(
        [
            "data/scene04_edge_d050_r01",
            "--frame-index",
            "12",
            "--distance-bin-px",
            "1",
            "--max-edge-distance-px",
            "20",
            "--minimum-valid-count",
            "50",
            "--transition-high",
            "0.8",
        ]
    )
    options = select_edge_roi_tool.analysis_options_from_args(args)

    assert args.frame_index == 12
    assert options.distance_bin_px == 1.0
    assert options.max_edge_distance_px == 20.0
    assert options.minimum_valid_count == 50
    assert options.transition_high_probability == 0.8


def test_resolve_selection_paths_reuses_repeat_key(
    tmp_path: Path,
) -> None:
    paths = select_edge_roi_tool.resolve_selection_paths(
        tmp_path / "data" / "scene04_edge_d050_r03",
        tmp_path / "config" / "roi",
        tmp_path / "results" / "roi_preview",
    )

    assert paths.experiment_name == "scene04_edge_d050_r03"
    assert paths.roi_key == "scene04_edge_d050"
    assert paths.dataset_path.name == "depth.npz"
    assert paths.roi_path == (
        tmp_path / "config" / "roi" / "scene04_edge_d050.yaml"
    )
    assert paths.preview_path == (
        tmp_path
        / "results"
        / "roi_preview"
        / "scene04_edge_d050.png"
    )


@pytest.mark.parametrize(
    ("foreground_roi", "background_roi", "expected"),
    [
        (
            RectROI(x=0, y=0, width=2, height=4),
            RectROI(x=4, y=0, width=2, height=4),
            "left",
        ),
        (
            RectROI(x=4, y=0, width=2, height=4),
            RectROI(x=0, y=0, width=2, height=4),
            "right",
        ),
    ],
)
def test_infer_foreground_side_for_vertical_line(
    foreground_roi: RectROI,
    background_roi: RectROI,
    expected: str,
) -> None:
    line = Line2D(p1=(3.0, 0.0), p2=(3.0, 4.0))

    side = select_edge_roi_tool.infer_foreground_side(
        line,
        foreground_roi,
        background_roi,
    )

    assert side == expected


def test_left_right_inference_is_independent_of_endpoint_order() -> None:
    foreground = RectROI(x=0, y=0, width=2, height=4)
    background = RectROI(x=4, y=0, width=2, height=4)

    forward = select_edge_roi_tool.infer_foreground_side(
        Line2D(p1=(3.0, 0.0), p2=(3.0, 4.0)),
        foreground,
        background,
    )
    reverse = select_edge_roi_tool.infer_foreground_side(
        Line2D(p1=(3.0, 4.0), p2=(3.0, 0.0)),
        foreground,
        background,
    )

    assert forward == reverse == "left"


@pytest.mark.parametrize(
    ("line", "expected"),
    [
        (
            Line2D(p1=(0.0, 3.0), p2=(6.0, 3.0)),
            "negative",
        ),
        (
            Line2D(p1=(6.0, 3.0), p2=(0.0, 3.0)),
            "positive",
        ),
    ],
)
def test_horizontal_inference_uses_raw_side(
    line: Line2D,
    expected: str,
) -> None:
    foreground = RectROI(x=0, y=0, width=2, height=2)
    background = RectROI(x=0, y=4, width=2, height=2)

    assert (
        select_edge_roi_tool.infer_foreground_side(
            line,
            foreground,
            background,
        )
        == expected
    )


def test_inference_rejects_reference_centers_on_same_side() -> None:
    line = Line2D(p1=(6.0, 0.0), p2=(6.0, 8.0))

    with pytest.raises(ValueError, match="opposite sides"):
        select_edge_roi_tool.infer_foreground_side(
            line,
            RectROI(x=0, y=0, width=2, height=2),
            RectROI(x=3, y=0, width=2, height=2),
        )


def test_inference_rejects_reference_center_on_line() -> None:
    line = Line2D(p1=(6.0, 0.0), p2=(6.0, 8.0))

    with pytest.raises(ValueError, match="foreground ROI center"):
        select_edge_roi_tool.infer_foreground_side(
            line,
            RectROI(x=5, y=0, width=2, height=2),
            RectROI(x=8, y=0, width=2, height=2),
        )


def test_build_edge_roi_config_constructs_valid_nested_config() -> None:
    result = select_edge_roi_tool.build_edge_roi_config(
        **_selection_geometry(),
        options=_small_options(),
    )

    assert result.config.foreground_side == "left"
    assert result.config.distance_bin_px == 1.0
    assert result.config.max_edge_distance_px == 2.0
    assert result.config.reference.minimum_valid_count == 1
    assert result.config.bleeding.probability_threshold == 0.05
    assert result.config.invalid.ratio_threshold == 0.5
    assert result.config.transition.high_probability == 0.9
    assert result.warnings == ()


def test_build_rejects_nominal_line_missing_edge_roi() -> None:
    geometry = _selection_geometry()
    geometry["nominal_edge"] = Line2D(
        p1=(2.0, 0.0),
        p2=(2.0, 8.0),
    )

    with pytest.raises(ValueError, match="does not intersect"):
        select_edge_roi_tool.build_edge_roi_config(
            **geometry,
            options=_small_options(),
        )


def test_semantic_validation_rejects_incorrect_configured_side() -> None:
    result = select_edge_roi_tool.build_edge_roi_config(
        **_selection_geometry(),
        options=_small_options(),
    )
    incorrect = replace(result.config, foreground_side="right")

    with pytest.raises(ValueError, match="does not match"):
        select_edge_roi_tool.validate_selection_semantics(
            incorrect,
            (10, 12),
        )


def test_semantic_validation_reports_roi_overlap() -> None:
    result = select_edge_roi_tool.build_edge_roi_config(
        **_selection_geometry(),
        options=_small_options(),
    )
    overlapping = replace(
        result.config,
        foreground_roi=RectROI(
            x=2,
            y=1,
            width=2,
            height=6,
        ),
    )

    warnings = select_edge_roi_tool.validate_selection_semantics(
        overlapping,
        (10, 12),
    )

    assert "foreground_roi overlaps edge_roi" in warnings


def test_semantic_validation_warns_about_impossible_valid_count() -> None:
    result = select_edge_roi_tool.build_edge_roi_config(
        **_selection_geometry(),
        options=_small_options(minimum_valid_count=20),
    )

    assert result.warnings == (
        "foreground_roi has 12 pixels, fewer than "
        "minimum_valid_count 20",
        "background_roi has 12 pixels, fewer than "
        "minimum_valid_count 20",
    )


def test_semantic_validation_warns_about_edge_band_coverage() -> None:
    geometry = _selection_geometry()
    geometry["edge_roi"] = RectROI(
        x=5,
        y=1,
        width=3,
        height=6,
    )

    result = select_edge_roi_tool.build_edge_roi_config(
        **geometry,
        options=_small_options(),
    )

    assert result.warnings == (
        "edge_roi does not cover max_edge_distance_px on "
        "the foreground side",
        "edge_roi does not cover max_edge_distance_px on "
        "the background side",
    )
