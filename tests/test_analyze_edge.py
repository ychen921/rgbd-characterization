"""Tests for Scene 04 edge-analysis orchestration."""

import csv
from dataclasses import replace
from io import BytesIO, StringIO
from pathlib import Path

import cv2
import numpy as np
import pytest
import yaml

from src.io.dataset import DepthDataset
from src.metrics.edge_discontinuity import (
    EdgeFrameAnalysis,
    FrameEdgeResult,
    ReferenceDepthResult,
)
from src.preprocessing.edge_roi import (
    EdgeReferenceConfig,
    EdgeROIConfig,
    EdgeTransitionConfig,
    Line2D,
    save_edge_roi_config,
)
from src.preprocessing.roi import RectROI
import tools.analyze_edge as analyze_edge_module
from tools.analyze_edge import (
    FRAME_METRICS_FILENAME,
    LABEL_MAP_FILENAME,
    LABEL_OVERLAY_FILENAME,
    PROFILE_FILENAME,
    PROFILE_PLOT_FILENAME,
    ROI_OVERLAY_FILENAME,
    SUMMARY_FILENAME,
    TEMPORAL_PLOT_FILENAME,
    analyze_edge,
    build_aggregate_edge_profile_csv,
    build_edge_artifacts,
    build_frame_edge_metrics_csv,
    build_summary,
    compute_edge_metrics,
    load_edge_input,
    main,
    parse_args,
    parse_edge_experiment_name,
    resolve_output_dir,
    save_edge_analysis,
)


EXPERIMENT_NAME = (
    "scene04_gap030_vertical_white_d100_r01"
)
ROI_KEY = "scene04_gap030_vertical_white_d100"
IMAGE_SHAPE = (8, 12)


def _config(
    *,
    name: str = ROI_KEY,
    source_experiment: str = EXPERIMENT_NAME,
) -> EdgeROIConfig:
    return EdgeROIConfig(
        name=name,
        source_experiment=source_experiment,
        source_frame_index=2,
        foreground_roi=RectROI(
            x=0,
            y=0,
            width=3,
            height=8,
        ),
        background_roi=RectROI(
            x=9,
            y=0,
            width=3,
            height=8,
        ),
        edge_roi=RectROI(
            x=3,
            y=0,
            width=6,
            height=8,
        ),
        nominal_edge=Line2D(
            p1=(6.0, 0.0),
            p2=(6.0, 8.0),
        ),
        foreground_side="left",
        distance_bin_px=1.0,
        max_edge_distance_px=2.0,
        reference=EdgeReferenceConfig(
            minimum_tolerance_mm=10.0,
            mad_scale=3.0,
            minimum_valid_ratio=0.8,
            minimum_valid_count=8,
        ),
        transition=EdgeTransitionConfig(
            high_probability=0.9,
            low_probability=0.1,
        ),
    )


def _raw_depth(
    num_frames: int = 5,
) -> np.ndarray:
    depth = np.full(
        (num_frames, *IMAGE_SHAPE),
        1300,
        dtype=np.uint16,
    )
    depth[:, :, :6] = 1000
    return depth


def _write_dataset(
    dataset_dir: Path,
    depth: np.ndarray,
) -> Path:
    dataset_dir.mkdir(parents=True, exist_ok=True)
    dataset_path = dataset_dir / "depth.npz"
    DepthDataset(
        depth=depth,
        timestamps_ns=np.arange(
            depth.shape[0],
            dtype=np.int64,
        ),
    ).save(dataset_path)
    return dataset_path


def _write_config(
    roi_root: Path,
    config: EdgeROIConfig,
) -> Path:
    roi_path = roi_root / f"{ROI_KEY}.yaml"
    save_edge_roi_config(roi_path, config)
    return roi_path


def _reference(depth_mm: float) -> ReferenceDepthResult:
    return ReferenceDepthResult(
        median_mm=depth_mm,
        mad_mm=0.0,
        robust_sigma_mm=0.0,
        std_mm=0.0,
        valid_ratio=1.0,
        valid_count=24,
    )


def _fake_frame_analysis(
    frame_index: int,
    *,
    status: str = "ok",
) -> EdgeFrameAnalysis:
    valid = status == "ok"
    value = float(frame_index) if valid else float("nan")
    result = FrameEdgeResult(
        frame_index=frame_index,
        foreground_reference_mm=1000.0,
        background_reference_mm=1300.0,
        foreground_bleeding_ratio=value,
        foreground_bleeding_max_distance_px=value,
        background_bleeding_ratio=value,
        background_bleeding_max_distance_px=value,
        mixed_ratio=value,
        peak_mixed_ratio=value,
        peak_mixed_distance_px=value,
        outlier_ratio=value,
        invalid_ratio=value,
        invalid_band_width_px=value,
        transition_width_px=value,
        nominal_edge_offset_px=value,
        analysis_status=status,
        transition_status=(
            "ok"
            if valid
            else "not_analyzed"
        ),
    )
    return EdgeFrameAnalysis(
        result=result,
        foreground_reference=_reference(1000.0),
        background_reference=_reference(1300.0),
        profile=None,
        label_map=(
            np.full(
                IMAGE_SHAPE,
                2,
                dtype=np.uint8,
            )
            if valid
            else None
        ),
    )


def _analyze_in_tmp(
    tmp_path: Path,
    *,
    depth: np.ndarray | None = None,
):
    dataset_dir = tmp_path / EXPERIMENT_NAME
    _write_dataset(
        dataset_dir,
        _raw_depth(3) if depth is None else depth,
    )
    roi_root = tmp_path / "roi"
    _write_config(roi_root, _config())
    return analyze_edge(
        dataset_dir,
        roi_root=roi_root,
    )


def test_load_edge_input_loads_repeat_shared_config(
    tmp_path: Path,
) -> None:
    dataset_dir = (
        tmp_path
        / "scene04_gap030_vertical_white_d100_r02"
    )
    dataset_path = _write_dataset(
        dataset_dir,
        _raw_depth(),
    )
    roi_root = tmp_path / "roi"
    roi_path = _write_config(roi_root, _config())

    loaded = load_edge_input(
        dataset_dir,
        roi_root=roi_root,
    )

    assert loaded.experiment_name.endswith("_r02")
    assert loaded.dataset_dir == dataset_dir
    assert loaded.dataset_path == dataset_path
    assert loaded.roi_key == ROI_KEY
    assert loaded.roi_path == roi_path
    assert loaded.dataset.num_frames == 5
    assert loaded.config == _config()


def test_load_edge_input_rejects_missing_dataset(
    tmp_path: Path,
) -> None:
    dataset_dir = tmp_path / EXPERIMENT_NAME

    with pytest.raises(
        FileNotFoundError,
        match="Cannot find dataset file",
    ):
        load_edge_input(
            dataset_dir,
            roi_root=tmp_path / "roi",
        )


def test_load_edge_input_rejects_missing_config(
    tmp_path: Path,
) -> None:
    dataset_dir = tmp_path / EXPERIMENT_NAME
    _write_dataset(dataset_dir, _raw_depth())

    with pytest.raises(
        FileNotFoundError,
        match="tools/select_edge_roi.py",
    ):
        load_edge_input(
            dataset_dir,
            roi_root=tmp_path / "roi",
        )


def test_load_edge_input_rejects_empty_dataset(
    tmp_path: Path,
) -> None:
    dataset_dir = tmp_path / EXPERIMENT_NAME
    _write_dataset(
        dataset_dir,
        np.empty(
            (0, *IMAGE_SHAPE),
            dtype=np.uint16,
        ),
    )
    roi_root = tmp_path / "roi"
    _write_config(roi_root, _config())

    with pytest.raises(
        ValueError,
        match="contains no depth frames",
    ):
        load_edge_input(dataset_dir, roi_root=roi_root)


def test_load_edge_input_rejects_config_name_mismatch(
    tmp_path: Path,
) -> None:
    dataset_dir = tmp_path / EXPERIMENT_NAME
    _write_dataset(dataset_dir, _raw_depth())
    roi_root = tmp_path / "roi"
    _write_config(
        roi_root,
        _config(name="scene04_wrong_setup"),
    )

    with pytest.raises(
        ValueError,
        match="config name does not match",
    ):
        load_edge_input(dataset_dir, roi_root=roi_root)


def test_load_edge_input_rejects_source_key_mismatch(
    tmp_path: Path,
) -> None:
    dataset_dir = tmp_path / EXPERIMENT_NAME
    _write_dataset(dataset_dir, _raw_depth())
    roi_root = tmp_path / "roi"
    _write_config(
        roi_root,
        _config(
            source_experiment=(
                "scene04_gap030_vertical_white_d050_r01"
            )
        ),
    )

    with pytest.raises(
        ValueError,
        match="source experiment does not match",
    ):
        load_edge_input(dataset_dir, roi_root=roi_root)


def test_load_edge_input_validates_config_resolution(
    tmp_path: Path,
) -> None:
    dataset_dir = tmp_path / EXPERIMENT_NAME
    _write_dataset(dataset_dir, _raw_depth())
    roi_root = tmp_path / "roi"
    invalid_config = replace(
        _config(),
        background_roi=RectROI(
            x=11,
            y=0,
            width=3,
            height=8,
        ),
    )
    _write_config(roi_root, invalid_config)

    with pytest.raises(
        ValueError,
        match="background_roi exceeds image width",
    ):
        load_edge_input(dataset_dir, roi_root=roi_root)


def test_compute_edge_metrics_runs_real_metrics() -> None:
    raw_depth = _raw_depth()

    metrics = compute_edge_metrics(
        raw_depth,
        _config(),
    )

    result = metrics.discontinuity
    assert len(result.frame_results) == raw_depth.shape[0]
    assert result.valid_frames == raw_depth.shape[0]
    assert result.rejected_frames == 0
    assert result.aggregate_profile is not None
    assert metrics.representative_target_frame_index == 2
    representative = metrics.representative_analysis
    assert representative is not None
    assert representative.result.frame_index == 2
    assert representative.label_map is not None
    assert representative.label_map.dtype == np.uint8


def test_compute_edge_metrics_prepares_one_frame_at_a_time(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    raw_depth = _raw_depth(3)
    raw_depth[0, 0, 0] = 0
    raw_depth[1, 0, 0] = np.iinfo(np.uint16).max
    original = raw_depth.copy()
    prepared_shapes: list[tuple[int, ...]] = []
    special_values: list[float] = []
    real_prepare_depth = analyze_edge_module.prepare_depth

    def capture_prepare(depth: np.ndarray) -> np.ndarray:
        prepared_shapes.append(depth.shape)
        prepared = real_prepare_depth(depth)
        special_values.append(float(prepared[0, 0, 0]))
        return prepared

    monkeypatch.setattr(
        analyze_edge_module,
        "prepare_depth",
        capture_prepare,
    )

    compute_edge_metrics(raw_depth, _config())

    assert prepared_shapes == [
        (1, *IMAGE_SHAPE),
        (1, *IMAGE_SHAPE),
        (1, *IMAGE_SHAPE),
    ]
    assert np.isnan(special_values[0])
    assert np.isnan(special_values[1])
    np.testing.assert_array_equal(raw_depth, original)


def test_compute_edge_metrics_builds_distance_map_once(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    distance_calls = 0
    frame_calls = 0
    real_distance = (
        analyze_edge_module.compute_signed_distance_map
    )
    real_frame = analyze_edge_module.analyze_edge_frame

    def count_distance(*args: object, **kwargs: object) -> np.ndarray:
        nonlocal distance_calls
        distance_calls += 1
        return real_distance(*args, **kwargs)

    def count_frame(
        *args: object,
        **kwargs: object,
    ) -> EdgeFrameAnalysis:
        nonlocal frame_calls
        frame_calls += 1
        return real_frame(*args, **kwargs)

    monkeypatch.setattr(
        analyze_edge_module,
        "compute_signed_distance_map",
        count_distance,
    )
    monkeypatch.setattr(
        analyze_edge_module,
        "analyze_edge_frame",
        count_frame,
    )

    compute_edge_metrics(_raw_depth(4), _config())

    assert distance_calls == 1
    assert frame_calls == 4


def test_compute_edge_metrics_retains_only_representative_label_map(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    captured_analyses: list[EdgeFrameAnalysis] = []
    real_aggregate = analyze_edge_module.aggregate_edge_dataset

    def capture_aggregate(
        analyses: list[EdgeFrameAnalysis],
    ):
        captured_analyses.extend(analyses)
        return real_aggregate(analyses)

    monkeypatch.setattr(
        analyze_edge_module,
        "aggregate_edge_dataset",
        capture_aggregate,
    )

    metrics = compute_edge_metrics(
        _raw_depth(4),
        _config(),
    )

    assert captured_analyses
    assert all(
        analysis.label_map is None
        for analysis in captured_analyses
    )
    assert metrics.representative_analysis is not None
    assert (
        metrics.representative_analysis.label_map
        is not None
    )


def test_compute_edge_metrics_falls_back_to_nearest_lower_frame(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    statuses = {
        0: "insufficient_reference",
        1: "ok",
        2: "insufficient_reference",
        3: "ok",
        4: "insufficient_reference",
    }

    def fake_analyze(
        *,
        frame_index: int,
        **kwargs: object,
    ) -> EdgeFrameAnalysis:
        return _fake_frame_analysis(
            frame_index,
            status=statuses[frame_index],
        )

    monkeypatch.setattr(
        analyze_edge_module,
        "analyze_edge_frame",
        fake_analyze,
    )

    metrics = compute_edge_metrics(
        _raw_depth(),
        _config(),
    )

    assert metrics.representative_target_frame_index == 2
    assert metrics.representative_analysis is not None
    assert (
        metrics.representative_analysis.result.frame_index
        == 1
    )


def test_compute_edge_metrics_uses_requested_representative(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def fake_analyze(
        *,
        frame_index: int,
        **kwargs: object,
    ) -> EdgeFrameAnalysis:
        return _fake_frame_analysis(frame_index)

    monkeypatch.setattr(
        analyze_edge_module,
        "analyze_edge_frame",
        fake_analyze,
    )

    metrics = compute_edge_metrics(
        _raw_depth(),
        _config(),
        representative_frame_index=4,
    )

    assert metrics.representative_target_frame_index == 4
    assert metrics.representative_analysis is not None
    assert (
        metrics.representative_analysis.result.frame_index
        == 4
    )


def test_compute_edge_metrics_all_rejected_has_no_representative(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def fake_analyze(
        *,
        frame_index: int,
        **kwargs: object,
    ) -> EdgeFrameAnalysis:
        return _fake_frame_analysis(
            frame_index,
            status="insufficient_reference",
        )

    monkeypatch.setattr(
        analyze_edge_module,
        "analyze_edge_frame",
        fake_analyze,
    )

    metrics = compute_edge_metrics(
        _raw_depth(3),
        _config(),
    )

    assert metrics.representative_analysis is None
    assert metrics.discontinuity.valid_frames == 0
    assert metrics.discontinuity.rejected_frames == 3


@pytest.mark.parametrize(
    ("frame_index", "error_type", "message"),
    [
        (
            -1,
            ValueError,
            "0 <= index",
        ),
        (
            5,
            ValueError,
            "0 <= index",
        ),
        (
            1.5,
            TypeError,
            "integer or None",
        ),
        (
            True,
            TypeError,
            "integer or None",
        ),
    ],
)
def test_compute_edge_metrics_rejects_invalid_representative_index(
    frame_index: object,
    error_type: type[Exception],
    message: str,
) -> None:
    with pytest.raises(error_type, match=message):
        compute_edge_metrics(
            _raw_depth(),
            _config(),
            representative_frame_index=frame_index,
        )


@pytest.mark.parametrize(
    ("raw_depth", "error_type", "message"),
    [
        (
            [[[1000]]],
            TypeError,
            "numpy array",
        ),
        (
            np.zeros((2, 3), dtype=np.uint16),
            ValueError,
            "shape",
        ),
        (
            np.zeros((1, 2, 3), dtype=np.float32),
            ValueError,
            "dtype uint16",
        ),
        (
            np.empty((0, 2, 3), dtype=np.uint16),
            ValueError,
            "at least one frame",
        ),
    ],
)
def test_compute_edge_metrics_rejects_invalid_raw_depth(
    raw_depth: object,
    error_type: type[Exception],
    message: str,
) -> None:
    with pytest.raises(error_type, match=message):
        compute_edge_metrics(raw_depth, _config())


def test_analyze_edge_composes_loading_and_metrics(
    tmp_path: Path,
) -> None:
    dataset_dir = tmp_path / EXPERIMENT_NAME
    _write_dataset(dataset_dir, _raw_depth(3))
    roi_root = tmp_path / "roi"
    _write_config(roi_root, _config())

    result = analyze_edge(
        dataset_dir,
        roi_root=roi_root,
        representative_frame_index=1,
    )

    assert result.source.experiment_name == EXPERIMENT_NAME
    assert result.source.roi_key == ROI_KEY
    assert result.metrics.discontinuity.valid_frames == 3
    assert result.metrics.representative_analysis is not None
    assert (
        result.metrics.representative_analysis.result.frame_index
        == 1
    )


@pytest.mark.parametrize(
    (
        "experiment_name",
        "orientation",
        "foreground_mm",
        "background_mm",
    ),
    [
        (
            "scene04_gap030_horizon_white_d100_r01",
            "horizontal",
            1000,
            1300,
        ),
        (
            "scene04_gap030_vertical_white_d050_r01",
            "vertical",
            500,
            800,
        ),
        (
            "scene04_gap030_vertical_white_d100_r01",
            "vertical",
            1000,
            1300,
        ),
        (
            "scene04_gap030_vertical_white_d200_r01",
            "vertical",
            2000,
            2300,
        ),
    ],
)
def test_parse_edge_experiment_name(
    experiment_name: str,
    orientation: str,
    foreground_mm: int,
    background_mm: int,
) -> None:
    metadata = parse_edge_experiment_name(experiment_name)

    assert metadata.orientation == orientation
    assert metadata.target == "white"
    assert metadata.nominal_foreground_distance_mm == foreground_mm
    assert metadata.nominal_gap_mm == 300
    assert metadata.nominal_background_distance_mm == background_mm
    assert (
        metadata.distance_reference
        == "camera_optical_reference_plane"
    )
    assert metadata.repeat_index == 1


@pytest.mark.parametrize(
    ("experiment_name", "message"),
    [
        (
            "scene04_vertical_white_d100_r01",
            "must match",
        ),
        (
            "scene04_gap000_vertical_white_d100_r01",
            "gap must be positive",
        ),
        (
            "scene04_gap030_vertical_white_d000_r01",
            "distance must be positive",
        ),
        (
            "scene04_gap030_vertical_white_d100_r00",
            "repeat index must be positive",
        ),
    ],
)
def test_parse_edge_experiment_name_rejects_invalid_name(
    experiment_name: str,
    message: str,
) -> None:
    with pytest.raises(ValueError, match=message):
        parse_edge_experiment_name(experiment_name)


def test_build_summary_records_setup_and_measured_gap(
    tmp_path: Path,
) -> None:
    result = _analyze_in_tmp(tmp_path)

    summary = build_summary(result)
    serialized = yaml.safe_dump(summary)

    assert summary["setup"] == {
        "orientation_token": "vertical",
        "orientation": "vertical",
        "target": "white",
        "repeat_index": 1,
        "distance_reference": (
            "camera_optical_reference_plane"
        ),
        "nominal_foreground_distance_mm": 1000,
        "nominal_gap_mm": 300,
        "nominal_background_distance_mm": 1300,
    }
    reference = summary["reference_depth"]
    assert reference["foreground_median_mm"] == pytest.approx(1000.0)
    assert reference["background_median_mm"] == pytest.approx(1300.0)
    assert reference["measured_gap_median_mm"] == pytest.approx(300.0)
    assert reference["gap_error_mm"] == pytest.approx(0.0)
    assert ".nan" not in serialized.lower()


def test_build_frame_edge_metrics_csv_keeps_rejected_status(
    tmp_path: Path,
) -> None:
    depth = _raw_depth(3)
    depth[1] = 0
    result = _analyze_in_tmp(tmp_path, depth=depth)

    rows = list(
        csv.DictReader(
            StringIO(build_frame_edge_metrics_csv(result))
        )
    )

    assert len(rows) == 3
    assert rows[0]["timestamp_ns"] == "0"
    assert rows[0]["measured_gap_mm"] == "300.0"
    assert rows[1]["analysis_status"] != "ok"
    assert rows[1]["foreground_bleeding_ratio"] == ""
    assert rows[1]["transition_status"] == "not_analyzed"


def test_build_aggregate_edge_profile_csv_preserves_counts(
    tmp_path: Path,
) -> None:
    result = _analyze_in_tmp(tmp_path)
    profile = result.metrics.discontinuity.aggregate_profile

    csv_text = build_aggregate_edge_profile_csv(profile)

    assert csv_text is not None
    rows = list(csv.DictReader(StringIO(csv_text)))
    assert rows
    assert set(rows[0]) == {
        "distance_min_px",
        "distance_max_px",
        "distance_center_px",
        "pixel_count",
        "valid_count",
        "foreground_count",
        "background_count",
        "mixed_count",
        "outlier_count",
        "invalid_count",
        "foreground_ratio",
        "background_ratio",
        "mixed_ratio",
        "outlier_ratio",
        "invalid_ratio",
    }
    assert int(rows[0]["pixel_count"]) >= 0


def test_build_edge_artifacts_builds_normal_eight_file_set(
    tmp_path: Path,
) -> None:
    result = _analyze_in_tmp(tmp_path)

    artifacts = build_edge_artifacts(result)

    assert set(artifacts) == {
        SUMMARY_FILENAME,
        FRAME_METRICS_FILENAME,
        PROFILE_FILENAME,
        LABEL_MAP_FILENAME,
        ROI_OVERLAY_FILENAME,
        LABEL_OVERLAY_FILENAME,
        PROFILE_PLOT_FILENAME,
        TEMPORAL_PLOT_FILENAME,
    }
    label_map = np.load(
        BytesIO(artifacts[LABEL_MAP_FILENAME]),
        allow_pickle=False,
    )
    assert label_map.shape == IMAGE_SHAPE
    assert label_map.dtype == np.uint8
    for filename in (
        ROI_OVERLAY_FILENAME,
        LABEL_OVERLAY_FILENAME,
        PROFILE_PLOT_FILENAME,
        TEMPORAL_PLOT_FILENAME,
    ):
        decoded = cv2.imdecode(
            np.frombuffer(artifacts[filename], dtype=np.uint8),
            cv2.IMREAD_COLOR,
        )
        assert decoded is not None


def test_build_edge_artifacts_all_rejected_builds_diagnostics(
    tmp_path: Path,
) -> None:
    result = _analyze_in_tmp(
        tmp_path,
        depth=np.zeros(
            (3, *IMAGE_SHAPE),
            dtype=np.uint16,
        ),
    )

    artifacts = build_edge_artifacts(result)
    summary = yaml.safe_load(
        artifacts[SUMMARY_FILENAME].decode("utf-8")
    )

    assert set(artifacts) == {
        SUMMARY_FILENAME,
        FRAME_METRICS_FILENAME,
        ROI_OVERLAY_FILENAME,
        TEMPORAL_PLOT_FILENAME,
    }
    assert summary["frames"]["valid"] == 0
    assert summary["frames"]["rejected"] == 3
    assert (
        summary["diagnostics"]["aggregate_profile_available"]
        is False
    )
    assert (
        summary["diagnostics"][
            "representative_label_map_available"
        ]
        is False
    )


def test_save_edge_analysis_writes_readable_artifacts(
    tmp_path: Path,
) -> None:
    result = _analyze_in_tmp(tmp_path / "input")
    output_dir = tmp_path / "output"

    saved = save_edge_analysis(output_dir, result)

    assert saved == output_dir
    assert {
        path.name
        for path in output_dir.iterdir()
    } == {
        SUMMARY_FILENAME,
        FRAME_METRICS_FILENAME,
        PROFILE_FILENAME,
        LABEL_MAP_FILENAME,
        ROI_OVERLAY_FILENAME,
        LABEL_OVERLAY_FILENAME,
        PROFILE_PLOT_FILENAME,
        TEMPORAL_PLOT_FILENAME,
    }
    with (output_dir / SUMMARY_FILENAME).open(
        "r",
        encoding="utf-8",
    ) as stream:
        summary = yaml.safe_load(stream)
    assert summary["dataset"]["num_frames"] == 3


def test_save_edge_analysis_preflights_existing_artifact(
    tmp_path: Path,
) -> None:
    result = _analyze_in_tmp(tmp_path / "input")
    output_dir = tmp_path / "output"
    output_dir.mkdir()
    summary_path = output_dir / SUMMARY_FILENAME
    summary_path.write_text("sentinel", encoding="utf-8")

    with pytest.raises(FileExistsError, match="already exists"):
        save_edge_analysis(output_dir, result)

    assert summary_path.read_text(encoding="utf-8") == "sentinel"
    assert list(output_dir.iterdir()) == [summary_path]


def test_save_edge_analysis_rolls_back_created_files(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    result = _analyze_in_tmp(tmp_path / "input")
    output_dir = tmp_path / "output"
    real_open = Path.open
    exclusive_open_count = 0

    def fail_second_exclusive_open(
        path: Path,
        mode: str = "r",
        *args: object,
        **kwargs: object,
    ):
        nonlocal exclusive_open_count
        if mode == "xb" and path.parent == output_dir:
            exclusive_open_count += 1
            if exclusive_open_count == 2:
                raise OSError("injected write failure")
        return real_open(path, mode, *args, **kwargs)

    monkeypatch.setattr(Path, "open", fail_second_exclusive_open)

    with pytest.raises(OSError, match="injected write failure"):
        save_edge_analysis(output_dir, result)

    assert output_dir.is_dir()
    assert list(output_dir.iterdir()) == []


def test_parse_args_and_resolve_output_dir(
    tmp_path: Path,
) -> None:
    dataset_dir = tmp_path / EXPERIMENT_NAME
    roi_root = tmp_path / "roi"
    output_dir = tmp_path / "output"

    args = parse_args(
        [
            str(dataset_dir),
            "--roi-root",
            str(roi_root),
            "--output-dir",
            str(output_dir),
            "--frame-index",
            "2",
        ]
    )

    assert args.dataset_dir == dataset_dir
    assert args.roi_root == roi_root
    assert args.output_dir == output_dir
    assert args.frame_index == 2
    assert (
        resolve_output_dir(EXPERIMENT_NAME, None)
        == analyze_edge_module.DEFAULT_RESULTS_ROOT
        / EXPERIMENT_NAME
        / "edge_discontinuity"
    )
    assert (
        resolve_output_dir(EXPERIMENT_NAME, output_dir)
        == output_dir
    )


def test_parse_args_help() -> None:
    with pytest.raises(SystemExit) as error:
        parse_args(["--help"])

    assert error.value.code == 0


def test_main_runs_formal_cli_pipeline(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    dataset_dir = tmp_path / EXPERIMENT_NAME
    _write_dataset(dataset_dir, _raw_depth(3))
    roi_root = tmp_path / "roi"
    _write_config(roi_root, _config())
    output_dir = tmp_path / "output"

    exit_code = main(
        [
            str(dataset_dir),
            "--roi-root",
            str(roi_root),
            "--output-dir",
            str(output_dir),
            "--frame-index",
            "1",
        ]
    )

    assert exit_code == 0
    assert (output_dir / SUMMARY_FILENAME).is_file()
    output = capsys.readouterr().out
    assert "Edge analysis complete." in output
    assert "nominal foreground/background/gap" in output
    assert "representative target/selected: 1 / 1" in output
