"""Tests for Scene 04 edge-analysis orchestration."""

from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest

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
    analyze_edge,
    compute_edge_metrics,
    load_edge_input,
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
