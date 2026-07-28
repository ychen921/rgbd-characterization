"""Tests for pure Scene 04 edge ROI selection logic."""

from dataclasses import replace
from pathlib import Path

import cv2
import numpy as np
import pytest

from src.io.dataset import DepthDataset
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


def test_load_edge_dataset_returns_valid_dataset(
    tmp_path: Path,
) -> None:
    paths = select_edge_roi_tool.resolve_selection_paths(
        tmp_path / "data" / "scene04_edge_d050_r01",
    )
    paths.dataset_dir.mkdir(parents=True)
    depth = (
        np.arange(3 * 4 * 5, dtype=np.uint16)
        .reshape(3, 4, 5)
        + 1
    )
    timestamps_ns = np.array(
        [100, 200, 300],
        dtype=np.int64,
    )
    DepthDataset(
        depth=depth,
        timestamps_ns=timestamps_ns,
    ).save(paths.dataset_path)

    loaded = select_edge_roi_tool.load_edge_dataset(
        paths.dataset_path
    )

    np.testing.assert_array_equal(loaded.depth, depth)
    np.testing.assert_array_equal(
        loaded.timestamps_ns,
        timestamps_ns,
    )


def test_load_edge_dataset_rejects_missing_archive(
    tmp_path: Path,
) -> None:
    dataset_path = tmp_path / "missing" / "depth.npz"

    with pytest.raises(
        FileNotFoundError,
        match=r"Cannot find dataset file .*depth\.npz",
    ):
        select_edge_roi_tool.load_edge_dataset(dataset_path)


def test_load_edge_dataset_rejects_zero_frames(
    tmp_path: Path,
) -> None:
    dataset_path = tmp_path / "depth.npz"
    DepthDataset(
        depth=np.empty((0, 4, 5), dtype=np.uint16),
        timestamps_ns=np.empty((0,), dtype=np.int64),
    ).save(dataset_path)

    with pytest.raises(
        ValueError,
        match=r"Dataset contains no depth frames: .*depth\.npz",
    ):
        select_edge_roi_tool.load_edge_dataset(dataset_path)


def _displayable_dataset(num_frames: int) -> DepthDataset:
    depth = (
        np.arange(num_frames * 4 * 5, dtype=np.uint16)
        .reshape(num_frames, 4, 5)
        + 1
    )
    timestamps_ns = np.arange(
        num_frames,
        dtype=np.int64,
    )
    return DepthDataset(
        depth=depth,
        timestamps_ns=timestamps_ns,
    )


@pytest.mark.parametrize(
    ("num_frames", "expected_index"),
    [
        (4, 2),
        (5, 2),
    ],
)
def test_prepare_selection_frame_uses_middle_frame_by_default(
    num_frames: int,
    expected_index: int,
) -> None:
    dataset = _displayable_dataset(num_frames)
    original_depth = dataset.depth.copy()

    result = select_edge_roi_tool.prepare_selection_frame(dataset)

    assert result.frame_index == expected_index
    assert result.display_image.shape == (4, 5, 3)
    assert result.display_image.dtype == np.uint8
    np.testing.assert_array_equal(dataset.depth, original_depth)


def test_prepare_selection_frame_accepts_explicit_index() -> None:
    dataset = _displayable_dataset(5)

    result = select_edge_roi_tool.prepare_selection_frame(
        dataset,
        frame_index=1,
    )

    assert result.frame_index == 1
    np.testing.assert_array_equal(
        result.display_image,
        select_edge_roi_tool.depth_to_edge_display(
            dataset.depth[1]
        ),
    )


@pytest.mark.parametrize("frame_index", [-1, 3])
def test_prepare_selection_frame_rejects_out_of_range_index(
    frame_index: int,
) -> None:
    dataset = _displayable_dataset(3)

    with pytest.raises(
        ValueError,
        match=r"0 <= frame_index < 3",
    ):
        select_edge_roi_tool.prepare_selection_frame(
            dataset,
            frame_index=frame_index,
        )


@pytest.mark.parametrize("frame_index", [True, 1.5])
def test_prepare_selection_frame_rejects_non_integer_index(
    frame_index: object,
) -> None:
    dataset = _displayable_dataset(3)

    with pytest.raises(
        TypeError,
        match="integer or None",
    ):
        select_edge_roi_tool.prepare_selection_frame(
            dataset,
            frame_index=frame_index,
        )


def test_prepare_selection_frame_rejects_empty_dataset() -> None:
    dataset = DepthDataset(
        depth=np.empty((0, 4, 5), dtype=np.uint16),
        timestamps_ns=np.empty((0,), dtype=np.int64),
    )

    with pytest.raises(
        ValueError,
        match="no depth frames",
    ):
        select_edge_roi_tool.prepare_selection_frame(dataset)


def test_prepare_selection_frame_rejects_non_dataset() -> None:
    with pytest.raises(
        TypeError,
        match="dataset must be a DepthDataset",
    ):
        select_edge_roi_tool.prepare_selection_frame(
            np.zeros((2, 4, 5), dtype=np.uint16)
        )


def test_prepare_selection_frame_rejects_unusable_middle_frame() -> None:
    dataset = DepthDataset(
        depth=np.array(
            [
                [
                    [100, 200],
                    [300, 400],
                ],
                [
                    [500, 500],
                    [500, 500],
                ],
            ],
            dtype=np.uint16,
        ),
        timestamps_ns=np.array([100, 200], dtype=np.int64),
    )

    with pytest.raises(
        ValueError,
        match="Invalid display depth range",
    ):
        select_edge_roi_tool.prepare_selection_frame(dataset)


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


def test_depth_to_edge_display_marks_invalid_pixels() -> None:
    depth = np.array(
        [
            [0, 100, 200],
            [300, 400, 65535],
        ],
        dtype=np.uint16,
    )

    display = select_edge_roi_tool.depth_to_edge_display(depth)

    assert display.shape == (2, 3, 3)
    assert display.dtype == np.uint8
    assert tuple(display[0, 0]) == (
        select_edge_roi_tool.INVALID_COLOR
    )
    assert tuple(display[1, 2]) == (
        select_edge_roi_tool.INVALID_COLOR
    )
    np.testing.assert_array_equal(
        display[0, 1, 0],
        display[0, 1, 1],
    )


@pytest.mark.parametrize(
    "depth",
    [
        np.array([[0, 65535]], dtype=np.uint16),
        np.full((2, 2), 500, dtype=np.uint16),
    ],
)
def test_depth_to_edge_display_rejects_unusable_frame(
    depth: np.ndarray,
) -> None:
    with pytest.raises(ValueError):
        select_edge_roi_tool.depth_to_edge_display(depth)


def test_choose_rectangle_returns_named_selection(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[tuple[str, bool, bool]] = []

    def fake_select_roi(
        window_name: str,
        image: np.ndarray,
        *,
        showCrosshair: bool,
        fromCenter: bool,
    ) -> tuple[int, int, int, int]:
        assert image.shape == (10, 12, 3)
        calls.append(
            (window_name, showCrosshair, fromCenter)
        )
        return (2, 3, 4, 5)

    monkeypatch.setattr(cv2, "selectROI", fake_select_roi)
    monkeypatch.setattr(cv2, "destroyAllWindows", lambda: None)

    roi = select_edge_roi_tool.choose_rectangle(
        np.zeros((10, 12, 3), dtype=np.uint8),
        "FOREGROUND reference",
    )

    assert roi == RectROI(x=2, y=3, width=4, height=5)
    assert calls == [
        ("Select FOREGROUND reference ROI", True, False)
    ]


def test_choose_rectangle_cancel_raises(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        cv2,
        "selectROI",
        lambda *args, **kwargs: (0, 0, 0, 0),
    )
    monkeypatch.setattr(cv2, "destroyAllWindows", lambda: None)

    with pytest.raises(ValueError, match="EDGE.*cancelled"):
        select_edge_roi_tool.choose_rectangle(
            np.zeros((10, 12, 3), dtype=np.uint8),
            "EDGE analysis",
        )


def _mock_line_windows(
    monkeypatch: pytest.MonkeyPatch,
) -> dict[str, object]:
    captured: dict[str, object] = {}
    monkeypatch.setattr(cv2, "namedWindow", lambda name: None)
    monkeypatch.setattr(cv2, "imshow", lambda name, image: None)
    monkeypatch.setattr(
        cv2,
        "destroyAllWindows",
        lambda: None,
    )

    def capture_callback(
        name: str,
        callback: object,
        state: object,
    ) -> None:
        captured["callback"] = callback
        captured["state"] = state

    monkeypatch.setattr(
        cv2,
        "setMouseCallback",
        capture_callback,
    )
    return captured


def test_choose_nominal_edge_accepts_two_clicked_points(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    captured = _mock_line_windows(monkeypatch)

    def fake_wait_key(delay: int) -> int:
        assert delay == 20
        callback = captured["callback"]
        state = captured["state"]
        callback(cv2.EVENT_LBUTTONDOWN, 3, 1, 0, state)
        callback(cv2.EVENT_LBUTTONDOWN, 3, 8, 0, state)
        return 13

    monkeypatch.setattr(cv2, "waitKey", fake_wait_key)

    line = select_edge_roi_tool.choose_nominal_edge(
        np.zeros((10, 12, 3), dtype=np.uint8)
    )

    assert line == Line2D(p1=(3.0, 1.0), p2=(3.0, 8.0))


def test_choose_nominal_edge_reset_discards_old_points(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    captured = _mock_line_windows(monkeypatch)
    call_count = 0

    def fake_wait_key(delay: int) -> int:
        nonlocal call_count
        del delay
        callback = captured["callback"]
        state = captured["state"]
        if call_count == 0:
            callback(cv2.EVENT_LBUTTONDOWN, 1, 1, 0, state)
            callback(cv2.EVENT_LBUTTONDOWN, 1, 8, 0, state)
            call_count += 1
            return ord("r")

        callback(cv2.EVENT_LBUTTONDOWN, 4, 1, 0, state)
        callback(cv2.EVENT_LBUTTONDOWN, 4, 8, 0, state)
        return 13

    monkeypatch.setattr(cv2, "waitKey", fake_wait_key)

    line = select_edge_roi_tool.choose_nominal_edge(
        np.zeros((10, 12, 3), dtype=np.uint8)
    )

    assert line == Line2D(p1=(4.0, 1.0), p2=(4.0, 8.0))


def test_choose_nominal_edge_cancel_raises(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _mock_line_windows(monkeypatch)
    monkeypatch.setattr(
        cv2,
        "waitKey",
        lambda delay: select_edge_roi_tool.ESCAPE_KEY,
    )

    with pytest.raises(ValueError, match="nominal edge.*cancelled"):
        select_edge_roi_tool.choose_nominal_edge(
            np.zeros((10, 12, 3), dtype=np.uint8)
        )


def test_collect_edge_selection_runs_staged_workflow(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    rectangles = iter(
        [
            RectROI(x=0, y=1, width=2, height=6),
            RectROI(x=10, y=1, width=2, height=6),
            RectROI(x=3, y=1, width=7, height=6),
        ]
    )
    calls: list[tuple[str, int]] = []

    def fake_choose_rectangle(
        image: np.ndarray,
        label: str,
    ) -> RectROI:
        calls.append((label, int(np.count_nonzero(image))))
        return next(rectangles)

    monkeypatch.setattr(
        select_edge_roi_tool,
        "choose_rectangle",
        fake_choose_rectangle,
    )
    monkeypatch.setattr(
        select_edge_roi_tool,
        "choose_nominal_edge",
        lambda image: Line2D(
            p1=(6.0, 1.0),
            p2=(6.0, 7.0),
        ),
    )

    selection = select_edge_roi_tool.collect_edge_selection(
        np.zeros((10, 12, 3), dtype=np.uint8)
    )

    assert selection.foreground_roi.x == 0
    assert selection.background_roi.x == 10
    assert selection.edge_roi.x == 3
    assert calls[0] == ("FOREGROUND reference", 0)
    assert calls[1][0] == "BACKGROUND reference"
    assert calls[1][1] > 0
    assert calls[2][0] == "EDGE analysis"
    assert calls[2][1] > calls[1][1]


def test_collect_edge_selection_rejects_same_side_references(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    rectangles = iter(
        [
            RectROI(x=0, y=1, width=2, height=6),
            RectROI(x=3, y=1, width=2, height=6),
            RectROI(x=5, y=1, width=3, height=6),
        ]
    )
    monkeypatch.setattr(
        select_edge_roi_tool,
        "choose_rectangle",
        lambda image, label: next(rectangles),
    )
    monkeypatch.setattr(
        select_edge_roi_tool,
        "choose_nominal_edge",
        lambda image: Line2D(
            p1=(6.0, 1.0),
            p2=(6.0, 7.0),
        ),
    )

    with pytest.raises(ValueError, match="opposite sides"):
        select_edge_roi_tool.collect_edge_selection(
            np.zeros((10, 12, 3), dtype=np.uint8)
        )
