"""Tests for Scene 04 depth-discontinuity metric computation."""

from dataclasses import replace

import numpy as np
import pytest

from src.geometry.edge_geometry import compute_signed_distance_map
from src.metrics.edge_discontinuity import (
    ROBUST_SIGMA_SCALE,
    AmbiguousReferenceError,
    DistanceProfileResult,
    EdgeFrameAnalysis,
    EdgePixelLabel,
    FrameEdgeResult,
    ReferenceDepthResult,
    aggregate_edge_dataset,
    aggregate_labels_by_distance,
    analyze_edge_frame,
    classify_edge_depth,
    compute_bleeding_metrics,
    compute_invalid_edge_metrics,
    compute_mixed_outlier_metrics,
    compute_reference_tolerance,
    compute_transition_width,
    estimate_reference_depth,
    estimate_transition_center,
)
from src.preprocessing.edge_roi import (
    EdgeBleedingConfig,
    EdgeInvalidConfig,
    EdgeReferenceConfig,
    EdgeROIConfig,
    EdgeTransitionConfig,
    Line2D,
)
from src.preprocessing.roi import RectROI


def _reference(
    median_mm: float,
    *,
    robust_sigma_mm: float = 0.0,
) -> ReferenceDepthResult:
    return ReferenceDepthResult(
        median_mm=median_mm,
        mad_mm=robust_sigma_mm / ROBUST_SIGMA_SCALE,
        robust_sigma_mm=robust_sigma_mm,
        std_mm=robust_sigma_mm,
        valid_ratio=1.0,
        valid_count=100,
    )


def _profile_from_counts(
    centers: list[float],
    *,
    pixel: list[int],
    foreground: list[int],
    background: list[int],
    mixed: list[int],
    outlier: list[int],
    invalid: list[int],
) -> DistanceProfileResult:
    center_array = np.asarray(centers, dtype=np.float64)
    if center_array.size > 1:
        half_width = float(
            np.min(np.diff(center_array))
            / 2.0
        )
    else:
        half_width = 0.5
    max_distance = float(np.max(np.abs(center_array)))
    distance_min = np.maximum(
        center_array - half_width,
        -max_distance,
    )
    distance_max = np.minimum(
        center_array + half_width,
        max_distance,
    )

    pixel_count = np.asarray(pixel, dtype=np.int64)
    foreground_count = np.asarray(foreground, dtype=np.int64)
    background_count = np.asarray(background, dtype=np.int64)
    mixed_count = np.asarray(mixed, dtype=np.int64)
    outlier_count = np.asarray(outlier, dtype=np.int64)
    invalid_count = np.asarray(invalid, dtype=np.int64)
    valid_count = (
        foreground_count
        + background_count
        + mixed_count
        + outlier_count
    )

    def ratio(
        numerator: np.ndarray,
        denominator: np.ndarray,
    ) -> np.ndarray:
        result = np.full(
            denominator.shape,
            np.nan,
            dtype=np.float64,
        )
        np.divide(
            numerator,
            denominator,
            out=result,
            where=denominator > 0,
        )
        return result

    return DistanceProfileResult(
        distance_min_px=distance_min,
        distance_max_px=distance_max,
        distance_center_px=center_array,
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


def _frame_config() -> EdgeROIConfig:
    return EdgeROIConfig(
        name="scene04_edge_d050",
        source_experiment="scene04_edge_d050_r01",
        source_frame_index=0,
        foreground_roi=RectROI(
            x=0,
            y=0,
            width=2,
            height=3,
        ),
        background_roi=RectROI(
            x=7,
            y=0,
            width=2,
            height=3,
        ),
        edge_roi=RectROI(
            x=3,
            y=0,
            width=3,
            height=3,
        ),
        nominal_edge=Line2D(
            p1=(4.0, 0.0),
            p2=(4.0, 3.0),
        ),
        foreground_side="left",
        distance_bin_px=1.0,
        max_edge_distance_px=1.0,
        reference=EdgeReferenceConfig(
            minimum_tolerance_mm=10.0,
            mad_scale=3.0,
            minimum_valid_ratio=1.0,
            minimum_valid_count=6,
        ),
        transition=EdgeTransitionConfig(
            high_probability=0.9,
            low_probability=0.1,
        ),
        bleeding=EdgeBleedingConfig(
            probability_threshold=0.05,
        ),
        invalid=EdgeInvalidConfig(
            ratio_threshold=0.5,
        ),
    )


def _ideal_frame(
    *,
    background_depth: float = 200.0,
) -> np.ndarray:
    frame = np.full((3, 9), 150.0, dtype=np.float32)
    frame[:, 0:2] = 100.0
    frame[:, 7:9] = background_depth
    frame[:, 3] = 100.0
    frame[:, 4] = 150.0
    frame[:, 5] = background_depth
    return frame


def test_estimate_reference_depth_returns_robust_statistics() -> None:
    frame = np.array(
        [
            [100.0, 102.0],
            [98.0, np.nan],
        ],
        dtype=np.float32,
    )

    result = estimate_reference_depth(
        frame,
        RectROI(x=0, y=0, width=2, height=2),
    )

    assert result.median_mm == pytest.approx(100.0)
    assert result.mad_mm == pytest.approx(2.0)
    assert result.robust_sigma_mm == pytest.approx(
        ROBUST_SIGMA_SCALE * 2.0
    )
    assert result.std_mm == pytest.approx(
        np.std([100.0, 102.0, 98.0])
    )
    assert result.valid_ratio == pytest.approx(0.75)
    assert result.valid_count == 3


def test_estimate_reference_depth_handles_all_invalid_roi() -> None:
    frame = np.full((2, 2), np.nan, dtype=np.float32)

    result = estimate_reference_depth(
        frame,
        RectROI(x=0, y=0, width=2, height=2),
    )

    assert np.isnan(result.median_mm)
    assert np.isnan(result.mad_mm)
    assert np.isnan(result.robust_sigma_mm)
    assert np.isnan(result.std_mm)
    assert result.valid_ratio == 0.0
    assert result.valid_count == 0


def test_compute_reference_tolerance_uses_fixed_minimum() -> None:
    result = compute_reference_tolerance(
        _reference(100.0, robust_sigma_mm=2.0),
        minimum_tolerance_mm=10.0,
        mad_scale=3.0,
    )

    assert result == pytest.approx(10.0)


def test_compute_reference_tolerance_uses_robust_scale() -> None:
    result = compute_reference_tolerance(
        _reference(100.0, robust_sigma_mm=5.0),
        minimum_tolerance_mm=10.0,
        mad_scale=3.0,
    )

    assert result == pytest.approx(15.0)


def test_classify_edge_depth_assigns_all_label_types() -> None:
    frame = np.array(
        [[np.nan, 100.0, 200.0, 150.0, 50.0, 250.0, 110.0]],
        dtype=np.float32,
    )

    labels = classify_edge_depth(
        frame,
        RectROI(x=0, y=0, width=7, height=1),
        _reference(100.0),
        _reference(200.0),
        minimum_tolerance_mm=10.0,
        mad_scale=3.0,
    )

    np.testing.assert_array_equal(
        labels,
        [[
            EdgePixelLabel.INVALID,
            EdgePixelLabel.FOREGROUND,
            EdgePixelLabel.BACKGROUND,
            EdgePixelLabel.MIXED,
            EdgePixelLabel.OUTLIER,
            EdgePixelLabel.OUTLIER,
            EdgePixelLabel.FOREGROUND,
        ]],
    )
    assert labels.dtype == np.uint8


def test_classification_handles_farther_foreground() -> None:
    frame = np.array(
        [[200.0, 100.0, 150.0]],
        dtype=np.float32,
    )

    labels = classify_edge_depth(
        frame,
        RectROI(x=0, y=0, width=3, height=1),
        _reference(200.0),
        _reference(100.0),
        minimum_tolerance_mm=10.0,
        mad_scale=3.0,
    )

    np.testing.assert_array_equal(
        labels,
        [[
            EdgePixelLabel.FOREGROUND,
            EdgePixelLabel.BACKGROUND,
            EdgePixelLabel.MIXED,
        ]],
    )


def test_classification_rejects_touching_reference_ranges() -> None:
    frame = np.full((1, 1), 110.0, dtype=np.float32)

    with pytest.raises(
        AmbiguousReferenceError,
        match="overlap",
    ):
        classify_edge_depth(
            frame,
            RectROI(x=0, y=0, width=1, height=1),
            _reference(100.0),
            _reference(120.0),
            minimum_tolerance_mm=10.0,
            mad_scale=3.0,
        )


def test_distance_profile_uses_explicit_denominators() -> None:
    labels = np.array(
        [[
            EdgePixelLabel.FOREGROUND,
            EdgePixelLabel.FOREGROUND,
            EdgePixelLabel.INVALID,
            EdgePixelLabel.MIXED,
            EdgePixelLabel.BACKGROUND,
            EdgePixelLabel.OUTLIER,
            EdgePixelLabel.BACKGROUND,
        ]],
        dtype=np.uint8,
    )
    distance = np.arange(-3, 4, dtype=np.float64)[None, :]

    profile = aggregate_labels_by_distance(
        labels,
        distance,
        RectROI(x=0, y=0, width=7, height=1),
        distance_bin_px=2.0,
        max_edge_distance_px=2.0,
    )

    np.testing.assert_allclose(
        profile.distance_center_px,
        [-2.0, 0.0, 2.0],
    )
    np.testing.assert_array_equal(
        profile.pixel_count,
        [1, 2, 2],
    )
    np.testing.assert_array_equal(
        profile.valid_count,
        [1, 1, 2],
    )
    np.testing.assert_allclose(
        profile.invalid_ratio,
        [0.0, 0.5, 0.0],
    )
    np.testing.assert_allclose(
        profile.mixed_ratio,
        [0.0, 1.0, 0.0],
    )
    np.testing.assert_allclose(
        profile.background_ratio,
        [0.0, 0.0, 0.5],
    )
    np.testing.assert_allclose(
        profile.outlier_ratio,
        [0.0, 0.0, 0.5],
    )


def test_distance_profile_returns_nan_class_ratios_for_invalid_bin() -> None:
    labels = np.array(
        [[EdgePixelLabel.INVALID]],
        dtype=np.uint8,
    )
    distance = np.zeros((1, 1), dtype=np.float64)

    profile = aggregate_labels_by_distance(
        labels,
        distance,
        RectROI(x=0, y=0, width=1, height=1),
        distance_bin_px=1.0,
        max_edge_distance_px=1.0,
    )

    center = 1
    assert profile.invalid_ratio[center] == 1.0
    assert np.isnan(profile.foreground_ratio[center])
    assert np.isnan(profile.background_ratio[center])
    assert np.isnan(profile.mixed_ratio[center])
    assert np.isnan(profile.outlier_ratio[center])


def test_compute_bleeding_metrics_uses_opposite_side_valid_pixels() -> None:
    labels = np.array(
        [[
            EdgePixelLabel.FOREGROUND,
            EdgePixelLabel.BACKGROUND,
            EdgePixelLabel.FOREGROUND,
            EdgePixelLabel.MIXED,
            EdgePixelLabel.FOREGROUND,
            EdgePixelLabel.BACKGROUND,
            EdgePixelLabel.FOREGROUND,
        ]],
        dtype=np.uint8,
    )
    distance = np.arange(-3, 4, dtype=np.float64)[None, :]
    roi = RectROI(x=0, y=0, width=7, height=1)
    profile = aggregate_labels_by_distance(
        labels,
        distance,
        roi,
        distance_bin_px=1.0,
        max_edge_distance_px=3.0,
    )

    result = compute_bleeding_metrics(
        labels,
        distance,
        roi,
        profile,
        max_edge_distance_px=3.0,
        probability_threshold=0.5,
    )

    assert result.foreground_bleeding_ratio == pytest.approx(2 / 3)
    assert result.background_bleeding_ratio == pytest.approx(1 / 3)
    assert (
        result.foreground_bleeding_max_distance_px
        == pytest.approx(3.0)
    )
    assert (
        result.background_bleeding_max_distance_px
        == pytest.approx(2.0)
    )


def test_compute_mixed_outlier_metrics_reports_nearest_tied_peak() -> None:
    profile = _profile_from_counts(
        [-2.0, 0.0, 2.0],
        pixel=[10, 10, 10],
        foreground=[5, 2, 0],
        background=[0, 2, 5],
        mixed=[5, 5, 5],
        outlier=[0, 1, 0],
        invalid=[0, 0, 0],
    )

    result = compute_mixed_outlier_metrics(profile)

    assert result.mixed_ratio == pytest.approx(0.5)
    assert result.peak_mixed_ratio == pytest.approx(0.5)
    assert result.peak_mixed_distance_px == pytest.approx(0.0)
    assert result.outlier_ratio == pytest.approx(1 / 30)


def test_compute_invalid_edge_metrics_uses_contiguous_central_bins() -> None:
    profile = _profile_from_counts(
        [-2.0, 0.0, 2.0],
        pixel=[10, 10, 10],
        foreground=[4, 2, 3],
        background=[0, 0, 0],
        mixed=[0, 0, 0],
        outlier=[0, 0, 0],
        invalid=[6, 8, 7],
    )

    result = compute_invalid_edge_metrics(
        profile,
        invalid_ratio_threshold=0.5,
    )

    assert result.invalid_ratio == pytest.approx(21 / 30)
    assert result.invalid_band_width_px == pytest.approx(4.0)


def test_invalid_band_width_is_zero_when_center_is_below_threshold() -> None:
    profile = _profile_from_counts(
        [-2.0, 0.0, 2.0],
        pixel=[10, 10, 10],
        foreground=[4, 9, 3],
        background=[0, 0, 0],
        mixed=[0, 0, 0],
        outlier=[0, 0, 0],
        invalid=[6, 1, 7],
    )

    result = compute_invalid_edge_metrics(
        profile,
        invalid_ratio_threshold=0.5,
    )

    assert result.invalid_band_width_px == 0.0


def test_transition_width_uses_linear_interpolation() -> None:
    profile = _profile_from_counts(
        [-2.0, -1.0, 0.0, 1.0, 2.0],
        pixel=[100] * 5,
        foreground=[100, 90, 50, 10, 0],
        background=[0, 10, 50, 90, 100],
        mixed=[0] * 5,
        outlier=[0] * 5,
        invalid=[0] * 5,
    )

    result = compute_transition_width(
        profile,
        high_probability=0.9,
        low_probability=0.1,
    )
    center = estimate_transition_center(profile)

    assert result.status == "ok"
    assert result.high_crossing_px == pytest.approx(-1.0)
    assert result.center_crossing_px == pytest.approx(0.0)
    assert result.low_crossing_px == pytest.approx(1.0)
    assert result.transition_width_px == pytest.approx(2.0)
    assert result.nominal_edge_offset_px == pytest.approx(0.0)
    assert center.status == "ok"
    assert center.distance_px == pytest.approx(0.0)


def test_transition_reports_missing_crossing() -> None:
    profile = _profile_from_counts(
        [-1.0, 0.0, 1.0],
        pixel=[10, 10, 10],
        foreground=[10, 10, 10],
        background=[0, 0, 0],
        mixed=[0, 0, 0],
        outlier=[0, 0, 0],
        invalid=[0, 0, 0],
    )

    result = compute_transition_width(
        profile,
        high_probability=0.9,
        low_probability=0.1,
    )

    assert result.status == "missing_crossing"
    assert np.isnan(result.transition_width_px)


def test_transition_reports_multiple_descending_crossings() -> None:
    profile = _profile_from_counts(
        [-2.0, -1.0, 0.0, 1.0, 2.0],
        pixel=[10] * 5,
        foreground=[10, 0, 10, 0, 0],
        background=[0, 10, 0, 10, 10],
        mixed=[0] * 5,
        outlier=[0] * 5,
        invalid=[0] * 5,
    )

    result = compute_transition_width(
        profile,
        high_probability=0.9,
        low_probability=0.1,
    )

    assert result.status == "nonmonotonic_crossing"
    assert np.isnan(result.nominal_edge_offset_px)


def test_analyze_edge_frame_returns_metrics_for_valid_frame() -> None:
    config = _frame_config()
    frame = _ideal_frame()
    distance = compute_signed_distance_map(
        frame.shape,
        config.nominal_edge,
        config.foreground_side,
    )

    analysis = analyze_edge_frame(
        0,
        frame,
        config,
        distance,
    )

    assert analysis.result.analysis_status == "ok"
    assert analysis.result.transition_status == "ok"
    assert analysis.result.foreground_reference_mm == 100.0
    assert analysis.result.background_reference_mm == 200.0
    assert analysis.result.foreground_bleeding_ratio == 0.0
    assert analysis.result.background_bleeding_ratio == 0.0
    assert analysis.result.mixed_ratio == pytest.approx(1 / 3)
    assert analysis.profile is not None
    assert analysis.label_map is not None


def test_analyze_edge_frame_rejects_insufficient_reference() -> None:
    config = replace(
        _frame_config(),
        reference=EdgeReferenceConfig(
            minimum_tolerance_mm=10.0,
            mad_scale=3.0,
            minimum_valid_ratio=1.0,
            minimum_valid_count=1,
        ),
    )
    frame = _ideal_frame()
    frame[0, 0] = np.nan
    distance = compute_signed_distance_map(
        frame.shape,
        config.nominal_edge,
        config.foreground_side,
    )

    analysis = analyze_edge_frame(
        0,
        frame,
        config,
        distance,
    )

    assert (
        analysis.result.analysis_status
        == "insufficient_foreground_reference"
    )
    assert analysis.result.transition_status == "not_analyzed"
    assert analysis.profile is None
    assert analysis.label_map is None


def test_analyze_edge_frame_rejects_ambiguous_references() -> None:
    config = _frame_config()
    frame = _ideal_frame(background_depth=115.0)
    distance = compute_signed_distance_map(
        frame.shape,
        config.nominal_edge,
        config.foreground_side,
    )

    analysis = analyze_edge_frame(
        0,
        frame,
        config,
        distance,
    )

    assert (
        analysis.result.analysis_status
        == "ambiguous_reference_overlap"
    )
    assert analysis.profile is None


def _dataset_analysis(
    frame_index: int,
    profile: DistanceProfileResult | None,
    *,
    analysis_status: str,
    transition_status: str,
    scalar: float,
) -> EdgeFrameAnalysis:
    reference = _reference(100.0)
    metric_value = (
        scalar
        if analysis_status == "ok"
        else float("nan")
    )
    result = FrameEdgeResult(
        frame_index=frame_index,
        foreground_reference_mm=100.0,
        background_reference_mm=200.0,
        foreground_bleeding_ratio=metric_value,
        foreground_bleeding_max_distance_px=metric_value,
        background_bleeding_ratio=metric_value,
        background_bleeding_max_distance_px=metric_value,
        mixed_ratio=metric_value,
        peak_mixed_ratio=metric_value,
        peak_mixed_distance_px=metric_value,
        outlier_ratio=metric_value,
        invalid_ratio=metric_value,
        invalid_band_width_px=metric_value,
        transition_width_px=metric_value,
        nominal_edge_offset_px=metric_value,
        analysis_status=analysis_status,
        transition_status=transition_status,
    )
    return EdgeFrameAnalysis(
        result=result,
        foreground_reference=reference,
        background_reference=_reference(200.0),
        profile=profile,
        label_map=None,
    )


def test_aggregate_edge_dataset_sums_counts_before_ratios() -> None:
    first_profile = _profile_from_counts(
        [0.0],
        pixel=[1],
        foreground=[1],
        background=[0],
        mixed=[0],
        outlier=[0],
        invalid=[0],
    )
    second_profile = _profile_from_counts(
        [0.0],
        pixel=[9],
        foreground=[0],
        background=[9],
        mixed=[0],
        outlier=[0],
        invalid=[0],
    )
    analyses = [
        _dataset_analysis(
            0,
            first_profile,
            analysis_status="ok",
            transition_status="ok",
            scalar=1.0,
        ),
        _dataset_analysis(
            1,
            second_profile,
            analysis_status="ok",
            transition_status="missing_crossing",
            scalar=0.0,
        ),
        _dataset_analysis(
            2,
            None,
            analysis_status="insufficient_foreground_reference",
            transition_status="not_analyzed",
            scalar=0.0,
        ),
    ]

    result = aggregate_edge_dataset(analyses)

    assert result.aggregate_profile is not None
    assert result.aggregate_profile.foreground_ratio[0] == pytest.approx(0.1)
    assert result.median_foreground_bleeding_ratio == pytest.approx(0.5)
    assert result.valid_frames == 2
    assert result.rejected_frames == 1
    assert result.valid_transition_frames == 1
    assert result.failed_transition_frames == 1


def test_aggregate_edge_dataset_handles_empty_input() -> None:
    result = aggregate_edge_dataset([])

    assert result.frame_results == ()
    assert result.aggregate_profile is None
    assert result.valid_frames == 0
    assert result.rejected_frames == 0
    assert np.isnan(result.median_mixed_ratio)
