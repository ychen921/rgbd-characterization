"""Depth-discontinuity metrics for Scene 04 edge datasets."""

from dataclasses import dataclass
from enum import IntEnum
import math
from numbers import Integral, Real
from typing import Sequence

import numpy as np

from src.preprocessing.edge_roi import (
    EdgeROIConfig,
    validate_edge_roi_config,
)
from src.preprocessing.roi import RectROI


ROBUST_SIGMA_SCALE = 1.4826


class AmbiguousReferenceError(ValueError):
    """Raised when foreground and background accepted ranges overlap."""


class EdgePixelLabel(IntEnum):
    """Classify one pixel inside or outside the analyzed edge ROI."""

    OUTSIDE = 0
    INVALID = 1
    FOREGROUND = 2
    BACKGROUND = 3
    MIXED = 4
    OUTLIER = 5


@dataclass(frozen=True)
class ReferenceDepthResult:
    """Store robust depth statistics for one reference ROI."""

    median_mm: float
    mad_mm: float
    robust_sigma_mm: float
    std_mm: float
    valid_ratio: float
    valid_count: int


@dataclass(frozen=True)
class DistanceProfileResult:
    """Store distance-bin counts and explicitly denominated ratios."""

    distance_min_px: np.ndarray
    distance_max_px: np.ndarray
    distance_center_px: np.ndarray

    pixel_count: np.ndarray
    valid_count: np.ndarray

    foreground_count: np.ndarray
    background_count: np.ndarray
    mixed_count: np.ndarray
    outlier_count: np.ndarray
    invalid_count: np.ndarray

    foreground_ratio: np.ndarray
    background_ratio: np.ndarray
    mixed_ratio: np.ndarray
    outlier_ratio: np.ndarray
    invalid_ratio: np.ndarray


@dataclass(frozen=True)
class BleedingMetrics:
    """Store wrong-side foreground/background classification metrics."""

    foreground_bleeding_ratio: float
    foreground_bleeding_max_distance_px: float
    background_bleeding_ratio: float
    background_bleeding_max_distance_px: float


@dataclass(frozen=True)
class MixedOutlierMetrics:
    """Store mixed-depth and outlier summaries within the edge band."""

    mixed_ratio: float
    peak_mixed_ratio: float
    peak_mixed_distance_px: float
    outlier_ratio: float


@dataclass(frozen=True)
class InvalidEdgeMetrics:
    """Store invalid occurrence and the central invalid-band width."""

    invalid_ratio: float
    invalid_band_width_px: float


@dataclass(frozen=True)
class ProbabilityCrossingResult:
    """Store one interpolated profile crossing and its status."""

    distance_px: float
    status: str


@dataclass(frozen=True)
class TransitionResult:
    """Store foreground-probability transition geometry."""

    high_crossing_px: float
    center_crossing_px: float
    low_crossing_px: float
    transition_width_px: float
    nominal_edge_offset_px: float
    status: str


@dataclass(frozen=True)
class FrameEdgeResult:
    """Store scalar Scene 04 metrics for one input frame."""

    frame_index: int
    foreground_reference_mm: float
    background_reference_mm: float

    foreground_bleeding_ratio: float
    foreground_bleeding_max_distance_px: float
    background_bleeding_ratio: float
    background_bleeding_max_distance_px: float

    mixed_ratio: float
    peak_mixed_ratio: float
    peak_mixed_distance_px: float
    outlier_ratio: float

    invalid_ratio: float
    invalid_band_width_px: float

    transition_width_px: float
    nominal_edge_offset_px: float

    analysis_status: str
    transition_status: str


@dataclass(frozen=True)
class EdgeFrameAnalysis:
    """Keep a frame's scalar result and optional diagnostic arrays."""

    result: FrameEdgeResult
    foreground_reference: ReferenceDepthResult
    background_reference: ReferenceDepthResult
    profile: DistanceProfileResult | None
    label_map: np.ndarray | None


@dataclass(frozen=True)
class EdgeDiscontinuityResult:
    """Store dataset-level Scene 04 aggregation."""

    frame_results: tuple[FrameEdgeResult, ...]
    aggregate_profile: DistanceProfileResult | None

    median_foreground_bleeding_ratio: float
    median_background_bleeding_ratio: float
    median_mixed_ratio: float
    median_outlier_ratio: float
    median_invalid_ratio: float
    median_transition_width_px: float
    transition_width_p95_px: float
    median_nominal_edge_offset_px: float
    nominal_edge_offset_std_px: float

    valid_frames: int
    rejected_frames: int
    valid_transition_frames: int
    failed_transition_frames: int


def estimate_reference_depth(
    prepared_depth_frame: np.ndarray,
    roi: RectROI,
) -> ReferenceDepthResult:
    """Estimate robust depth statistics inside one reference rectangle."""
    _validate_prepared_frame(prepared_depth_frame)
    _validate_roi_for_frame(roi, prepared_depth_frame.shape)

    cropped = prepared_depth_frame[
        roi.y:roi.y + roi.height,
        roi.x:roi.x + roi.width,
    ]
    valid_values = cropped[np.isfinite(cropped)].astype(
        np.float64,
        copy=False,
    )
    valid_count = int(valid_values.size)
    valid_ratio = valid_count / roi.pixel_count

    if valid_count == 0:
        return ReferenceDepthResult(
            median_mm=float("nan"),
            mad_mm=float("nan"),
            robust_sigma_mm=float("nan"),
            std_mm=float("nan"),
            valid_ratio=0.0,
            valid_count=0,
        )

    median_mm = float(np.median(valid_values))
    mad_mm = float(
        np.median(np.abs(valid_values - median_mm))
    )

    return ReferenceDepthResult(
        median_mm=median_mm,
        mad_mm=mad_mm,
        robust_sigma_mm=ROBUST_SIGMA_SCALE * mad_mm,
        std_mm=float(np.std(valid_values, ddof=0)),
        valid_ratio=float(valid_ratio),
        valid_count=valid_count,
    )


def compute_reference_tolerance(
    reference: ReferenceDepthResult,
    minimum_tolerance_mm: float,
    mad_scale: float,
) -> float:
    """Return the larger of the fixed and robust reference tolerances."""
    _validate_reference_result(reference)
    minimum = _normalize_positive_real(
        minimum_tolerance_mm,
        "minimum_tolerance_mm",
    )
    scale = _normalize_positive_real(mad_scale, "mad_scale")

    if not np.isfinite(reference.robust_sigma_mm):
        raise ValueError(
            "reference robust_sigma_mm must be finite"
        )

    return float(
        max(
            minimum,
            scale * reference.robust_sigma_mm,
        )
    )


def classify_edge_depth(
    prepared_depth_frame: np.ndarray,
    edge_roi: RectROI,
    foreground_reference: ReferenceDepthResult,
    background_reference: ReferenceDepthResult,
    minimum_tolerance_mm: float,
    mad_scale: float,
) -> np.ndarray:
    """Classify full-image edge pixels without assuming depth ordering."""
    _validate_prepared_frame(prepared_depth_frame)
    _validate_roi_for_frame(edge_roi, prepared_depth_frame.shape)

    foreground_tolerance = compute_reference_tolerance(
        foreground_reference,
        minimum_tolerance_mm,
        mad_scale,
    )
    background_tolerance = compute_reference_tolerance(
        background_reference,
        minimum_tolerance_mm,
        mad_scale,
    )

    foreground_low = (
        foreground_reference.median_mm
        - foreground_tolerance
    )
    foreground_high = (
        foreground_reference.median_mm
        + foreground_tolerance
    )
    background_low = (
        background_reference.median_mm
        - background_tolerance
    )
    background_high = (
        background_reference.median_mm
        + background_tolerance
    )

    # Touching accepted ranges are ambiguous too: a boundary sample could
    # otherwise receive either label depending on classification order.
    overlap_low = max(foreground_low, background_low)
    overlap_high = min(foreground_high, background_high)
    if overlap_low <= overlap_high:
        raise AmbiguousReferenceError(
            "foreground and background accepted ranges overlap"
        )

    labels = np.full(
        prepared_depth_frame.shape,
        int(EdgePixelLabel.OUTSIDE),
        dtype=np.uint8,
    )
    depth = prepared_depth_frame[
        edge_roi.y:edge_roi.y + edge_roi.height,
        edge_roi.x:edge_roi.x + edge_roi.width,
    ]
    roi_labels = labels[
        edge_roi.y:edge_roi.y + edge_roi.height,
        edge_roi.x:edge_roi.x + edge_roi.width,
    ]

    invalid = np.isnan(depth)
    valid = ~invalid
    foreground = valid & (
        (depth >= foreground_low)
        & (depth <= foreground_high)
    )
    background = valid & (
        (depth >= background_low)
        & (depth <= background_high)
    )

    # Mixed depths lie strictly in the gap between accepted ranges. This
    # ordering-independent form also handles a farther foreground surface.
    if foreground_high < background_low:
        mixed_low = foreground_high
        mixed_high = background_low
    else:
        mixed_low = background_high
        mixed_high = foreground_low

    mixed = valid & (
        (depth > mixed_low)
        & (depth < mixed_high)
    )
    outlier = valid & ~(
        foreground | background | mixed
    )

    roi_labels[invalid] = int(EdgePixelLabel.INVALID)
    roi_labels[foreground] = int(EdgePixelLabel.FOREGROUND)
    roi_labels[background] = int(EdgePixelLabel.BACKGROUND)
    roi_labels[mixed] = int(EdgePixelLabel.MIXED)
    roi_labels[outlier] = int(EdgePixelLabel.OUTLIER)
    return labels


def aggregate_labels_by_distance(
    label_map: np.ndarray,
    signed_distance_map: np.ndarray,
    edge_roi: RectROI,
    distance_bin_px: float,
    max_edge_distance_px: float,
) -> DistanceProfileResult:
    """Aggregate edge labels into symmetric signed-distance bins."""
    _validate_label_and_distance_maps(
        label_map,
        signed_distance_map,
    )
    _validate_roi_for_frame(edge_roi, label_map.shape)
    bin_width, max_distance, bins_per_side = (
        _validate_distance_parameters(
            distance_bin_px,
            max_edge_distance_px,
        )
    )

    roi_labels = label_map[
        edge_roi.y:edge_roi.y + edge_roi.height,
        edge_roi.x:edge_roi.x + edge_roi.width,
    ]
    roi_distance = signed_distance_map[
        edge_roi.y:edge_roi.y + edge_roi.height,
        edge_roi.x:edge_roi.x + edge_roi.width,
    ]

    if np.any(roi_labels == int(EdgePixelLabel.OUTSIDE)):
        raise ValueError(
            "label_map contains OUTSIDE labels inside edge_roi"
        )

    band = np.abs(roi_distance) <= max_distance
    band_labels = roi_labels[band]
    band_distance = roi_distance[band]
    num_bins = bins_per_side * 2 + 1

    # Round each signed distance to the nearest bin centered on an integer
    # multiple of bin_width. This guarantees one central bin at distance zero.
    relative_index = np.floor(
        band_distance / bin_width + 0.5
    ).astype(np.int64)
    bin_index = np.clip(
        relative_index + bins_per_side,
        0,
        num_bins - 1,
    )

    centers = (
        np.arange(
            -bins_per_side,
            bins_per_side + 1,
            dtype=np.float64,
        )
        * bin_width
    )
    distance_min = np.maximum(
        centers - bin_width / 2.0,
        -max_distance,
    )
    distance_max = np.minimum(
        centers + bin_width / 2.0,
        max_distance,
    )

    pixel_count = np.bincount(
        bin_index,
        minlength=num_bins,
    ).astype(np.int64)

    def label_counts(label: EdgePixelLabel) -> np.ndarray:
        selected_bins = bin_index[
            band_labels == int(label)
        ]
        return np.bincount(
            selected_bins,
            minlength=num_bins,
        ).astype(np.int64)

    invalid_count = label_counts(EdgePixelLabel.INVALID)
    foreground_count = label_counts(
        EdgePixelLabel.FOREGROUND
    )
    background_count = label_counts(
        EdgePixelLabel.BACKGROUND
    )
    mixed_count = label_counts(EdgePixelLabel.MIXED)
    outlier_count = label_counts(EdgePixelLabel.OUTLIER)
    valid_count = (
        foreground_count
        + background_count
        + mixed_count
        + outlier_count
    )

    return _build_distance_profile(
        distance_min_px=distance_min,
        distance_max_px=distance_max,
        distance_center_px=centers,
        pixel_count=pixel_count,
        valid_count=valid_count,
        foreground_count=foreground_count,
        background_count=background_count,
        mixed_count=mixed_count,
        outlier_count=outlier_count,
        invalid_count=invalid_count,
    )


def compute_bleeding_metrics(
    label_map: np.ndarray,
    signed_distance_map: np.ndarray,
    edge_roi: RectROI,
    profile: DistanceProfileResult,
    max_edge_distance_px: float,
    probability_threshold: float,
) -> BleedingMetrics:
    """Compute wrong-side classification ratios and profile extents."""
    _validate_label_and_distance_maps(
        label_map,
        signed_distance_map,
    )
    _validate_roi_for_frame(edge_roi, label_map.shape)
    _validate_profile(profile)
    max_distance = _normalize_positive_real(
        max_edge_distance_px,
        "max_edge_distance_px",
    )
    threshold = _normalize_unit_interval(
        probability_threshold,
        "probability_threshold",
    )

    labels = label_map[
        edge_roi.y:edge_roi.y + edge_roi.height,
        edge_roi.x:edge_roi.x + edge_roi.width,
    ]
    distance = signed_distance_map[
        edge_roi.y:edge_roi.y + edge_roi.height,
        edge_roi.x:edge_roi.x + edge_roi.width,
    ]
    valid = _is_valid_label(labels)
    band = np.abs(distance) <= max_distance

    background_side = band & (distance > 0.0)
    foreground_side = band & (distance < 0.0)

    foreground_bleeding_ratio = _safe_scalar_ratio(
        np.count_nonzero(
            background_side
            & (labels == int(EdgePixelLabel.FOREGROUND))
        ),
        np.count_nonzero(background_side & valid),
    )
    background_bleeding_ratio = _safe_scalar_ratio(
        np.count_nonzero(
            foreground_side
            & (labels == int(EdgePixelLabel.BACKGROUND))
        ),
        np.count_nonzero(foreground_side & valid),
    )

    positive_bins = (
        (profile.distance_center_px > 0.0)
        & np.isfinite(profile.foreground_ratio)
        & (profile.foreground_ratio >= threshold)
    )
    negative_bins = (
        (profile.distance_center_px < 0.0)
        & np.isfinite(profile.background_ratio)
        & (profile.background_ratio >= threshold)
    )

    foreground_extent = (
        float(np.max(profile.distance_center_px[positive_bins]))
        if np.any(positive_bins)
        else float("nan")
    )
    background_extent = (
        float(
            np.max(
                np.abs(
                    profile.distance_center_px[negative_bins]
                )
            )
        )
        if np.any(negative_bins)
        else float("nan")
    )

    return BleedingMetrics(
        foreground_bleeding_ratio=foreground_bleeding_ratio,
        foreground_bleeding_max_distance_px=foreground_extent,
        background_bleeding_ratio=background_bleeding_ratio,
        background_bleeding_max_distance_px=background_extent,
    )


def compute_mixed_outlier_metrics(
    profile: DistanceProfileResult,
) -> MixedOutlierMetrics:
    """Summarize mixed and outlier labels over the full analyzed band."""
    _validate_profile(profile)
    total_valid = int(np.sum(profile.valid_count))
    mixed_ratio = _safe_scalar_ratio(
        int(np.sum(profile.mixed_count)),
        total_valid,
    )
    outlier_ratio = _safe_scalar_ratio(
        int(np.sum(profile.outlier_count)),
        total_valid,
    )

    finite_mixed = np.isfinite(profile.mixed_ratio)
    if not np.any(finite_mixed):
        peak_ratio = float("nan")
        peak_distance = float("nan")
    else:
        peak_ratio = float(
            np.max(profile.mixed_ratio[finite_mixed])
        )
        candidates = finite_mixed & np.isclose(
            profile.mixed_ratio,
            peak_ratio,
            rtol=1e-12,
            atol=1e-15,
        )
        # Ties are resolved toward the nominal edge for a stable, meaningful
        # representative peak location.
        candidate_distance = profile.distance_center_px[
            candidates
        ]
        peak_distance = float(
            candidate_distance[
                np.argmin(np.abs(candidate_distance))
            ]
        )

    return MixedOutlierMetrics(
        mixed_ratio=mixed_ratio,
        peak_mixed_ratio=peak_ratio,
        peak_mixed_distance_px=peak_distance,
        outlier_ratio=outlier_ratio,
    )


def compute_invalid_edge_metrics(
    profile: DistanceProfileResult,
    invalid_ratio_threshold: float,
) -> InvalidEdgeMetrics:
    """Compute overall invalid ratio and central contiguous band width."""
    _validate_profile(profile)
    threshold = _normalize_unit_interval(
        invalid_ratio_threshold,
        "invalid_ratio_threshold",
    )
    invalid_ratio = _safe_scalar_ratio(
        int(np.sum(profile.invalid_count)),
        int(np.sum(profile.pixel_count)),
    )

    center_candidates = np.flatnonzero(
        np.isclose(
            profile.distance_center_px,
            0.0,
            rtol=0.0,
            atol=1e-12,
        )
    )
    if center_candidates.size != 1:
        raise ValueError(
            "profile must contain exactly one zero-distance bin"
        )

    center = int(center_candidates[0])
    qualifies = (
        np.isfinite(profile.invalid_ratio)
        & (profile.invalid_ratio >= threshold)
    )
    if not qualifies[center]:
        width = 0.0
    else:
        left = center
        right = center
        while left > 0 and qualifies[left - 1]:
            left -= 1
        while right + 1 < qualifies.size and qualifies[right + 1]:
            right += 1
        width = float(
            profile.distance_max_px[right]
            - profile.distance_min_px[left]
        )

    return InvalidEdgeMetrics(
        invalid_ratio=invalid_ratio,
        invalid_band_width_px=width,
    )


def estimate_transition_center(
    profile: DistanceProfileResult,
    probability: float = 0.5,
) -> ProbabilityCrossingResult:
    """Estimate one descending foreground-probability crossing."""
    _validate_profile(profile)
    threshold = _normalize_probability(
        probability,
        "probability",
    )
    return _find_descending_crossing(
        profile.distance_center_px,
        profile.foreground_ratio,
        threshold,
    )


def compute_transition_width(
    profile: DistanceProfileResult,
    high_probability: float,
    low_probability: float,
) -> TransitionResult:
    """Estimate 90/50/10-style foreground-probability transition metrics."""
    _validate_profile(profile)
    high = _normalize_probability(
        high_probability,
        "high_probability",
    )
    low = _normalize_probability(
        low_probability,
        "low_probability",
    )
    if low >= high:
        raise ValueError(
            "low_probability must be less than high_probability"
        )

    high_crossing = _find_descending_crossing(
        profile.distance_center_px,
        profile.foreground_ratio,
        high,
    )
    center_crossing = estimate_transition_center(
        profile,
        probability=0.5,
    )
    low_crossing = _find_descending_crossing(
        profile.distance_center_px,
        profile.foreground_ratio,
        low,
    )

    statuses = (
        high_crossing.status,
        center_crossing.status,
        low_crossing.status,
    )
    if "nonmonotonic_crossing" in statuses:
        status = "nonmonotonic_crossing"
    elif "missing_crossing" in statuses:
        status = "missing_crossing"
    elif "insufficient_profile" in statuses:
        status = "insufficient_profile"
    else:
        status = "ok"

    if status != "ok":
        return _undefined_transition(status)

    if not (
        high_crossing.distance_px
        <= center_crossing.distance_px
        <= low_crossing.distance_px
    ):
        return _undefined_transition(
            "nonmonotonic_crossing"
        )

    return TransitionResult(
        high_crossing_px=high_crossing.distance_px,
        center_crossing_px=center_crossing.distance_px,
        low_crossing_px=low_crossing.distance_px,
        transition_width_px=(
            low_crossing.distance_px
            - high_crossing.distance_px
        ),
        nominal_edge_offset_px=center_crossing.distance_px,
        status="ok",
    )


def analyze_edge_frame(
    frame_index: int,
    prepared_depth_frame: np.ndarray,
    config: EdgeROIConfig,
    signed_distance_map: np.ndarray,
) -> EdgeFrameAnalysis:
    """Run all Batch 2 metrics for one prepared Scene 04 frame."""
    if (
        not isinstance(frame_index, Integral)
        or isinstance(frame_index, (bool, np.bool_))
        or int(frame_index) < 0
    ):
        raise ValueError("frame_index must be a non-negative integer")
    _validate_prepared_frame(prepared_depth_frame)
    if not isinstance(config, EdgeROIConfig):
        raise TypeError(
            "config must be an EdgeROIConfig; "
            f"got {type(config).__name__}"
        )
    validate_edge_roi_config(
        config,
        image_shape=prepared_depth_frame.shape,
    )
    if (
        not isinstance(signed_distance_map, np.ndarray)
        or signed_distance_map.shape
        != prepared_depth_frame.shape
    ):
        raise ValueError(
            "signed_distance_map must match prepared_depth_frame shape"
        )
    if not np.all(np.isfinite(signed_distance_map)):
        raise ValueError(
            "signed_distance_map must contain only finite values"
        )

    foreground_reference = estimate_reference_depth(
        prepared_depth_frame,
        config.foreground_roi,
    )
    background_reference = estimate_reference_depth(
        prepared_depth_frame,
        config.background_roi,
    )

    foreground_ok = _reference_is_sufficient(
        foreground_reference,
        config,
    )
    background_ok = _reference_is_sufficient(
        background_reference,
        config,
    )
    if not foreground_ok:
        return _rejected_frame_analysis(
            frame_index=int(frame_index),
            foreground_reference=foreground_reference,
            background_reference=background_reference,
            status="insufficient_foreground_reference",
        )
    if not background_ok:
        return _rejected_frame_analysis(
            frame_index=int(frame_index),
            foreground_reference=foreground_reference,
            background_reference=background_reference,
            status="insufficient_background_reference",
        )

    try:
        label_map = classify_edge_depth(
            prepared_depth_frame,
            config.edge_roi,
            foreground_reference,
            background_reference,
            config.reference.minimum_tolerance_mm,
            config.reference.mad_scale,
        )
    except AmbiguousReferenceError:
        return _rejected_frame_analysis(
            frame_index=int(frame_index),
            foreground_reference=foreground_reference,
            background_reference=background_reference,
            status="ambiguous_reference_overlap",
        )

    profile = aggregate_labels_by_distance(
        label_map,
        signed_distance_map,
        config.edge_roi,
        config.distance_bin_px,
        config.max_edge_distance_px,
    )
    bleeding = compute_bleeding_metrics(
        label_map,
        signed_distance_map,
        config.edge_roi,
        profile,
        config.max_edge_distance_px,
        config.bleeding.probability_threshold,
    )
    mixed_outlier = compute_mixed_outlier_metrics(profile)
    invalid = compute_invalid_edge_metrics(
        profile,
        config.invalid.ratio_threshold,
    )
    transition = compute_transition_width(
        profile,
        config.transition.high_probability,
        config.transition.low_probability,
    )

    result = FrameEdgeResult(
        frame_index=int(frame_index),
        foreground_reference_mm=(
            foreground_reference.median_mm
        ),
        background_reference_mm=(
            background_reference.median_mm
        ),
        foreground_bleeding_ratio=(
            bleeding.foreground_bleeding_ratio
        ),
        foreground_bleeding_max_distance_px=(
            bleeding.foreground_bleeding_max_distance_px
        ),
        background_bleeding_ratio=(
            bleeding.background_bleeding_ratio
        ),
        background_bleeding_max_distance_px=(
            bleeding.background_bleeding_max_distance_px
        ),
        mixed_ratio=mixed_outlier.mixed_ratio,
        peak_mixed_ratio=mixed_outlier.peak_mixed_ratio,
        peak_mixed_distance_px=(
            mixed_outlier.peak_mixed_distance_px
        ),
        outlier_ratio=mixed_outlier.outlier_ratio,
        invalid_ratio=invalid.invalid_ratio,
        invalid_band_width_px=(
            invalid.invalid_band_width_px
        ),
        transition_width_px=(
            transition.transition_width_px
        ),
        nominal_edge_offset_px=(
            transition.nominal_edge_offset_px
        ),
        analysis_status="ok",
        transition_status=transition.status,
    )
    return EdgeFrameAnalysis(
        result=result,
        foreground_reference=foreground_reference,
        background_reference=background_reference,
        profile=profile,
        label_map=label_map,
    )


def aggregate_edge_dataset(
    frame_analyses: Sequence[EdgeFrameAnalysis],
) -> EdgeDiscontinuityResult:
    """Aggregate valid frame profiles by counts and scalar finite values."""
    analyses = tuple(frame_analyses)
    for analysis in analyses:
        if not isinstance(analysis, EdgeFrameAnalysis):
            raise TypeError(
                "frame_analyses must contain EdgeFrameAnalysis values"
            )

    frame_results = tuple(
        analysis.result
        for analysis in analyses
    )
    valid_analyses = tuple(
        analysis
        for analysis in analyses
        if analysis.result.analysis_status == "ok"
    )
    profiles = tuple(
        analysis.profile
        for analysis in valid_analyses
        if analysis.profile is not None
    )
    aggregate_profile = (
        _combine_profiles(profiles)
        if profiles
        else None
    )

    valid_frames = len(valid_analyses)
    rejected_frames = len(analyses) - valid_frames
    valid_transition_frames = sum(
        analysis.result.transition_status == "ok"
        for analysis in valid_analyses
    )
    failed_transition_frames = (
        valid_frames - valid_transition_frames
    )

    return EdgeDiscontinuityResult(
        frame_results=frame_results,
        aggregate_profile=aggregate_profile,
        median_foreground_bleeding_ratio=_finite_median(
            [
                result.foreground_bleeding_ratio
                for result in frame_results
                if result.analysis_status == "ok"
            ]
        ),
        median_background_bleeding_ratio=_finite_median(
            [
                result.background_bleeding_ratio
                for result in frame_results
                if result.analysis_status == "ok"
            ]
        ),
        median_mixed_ratio=_finite_median(
            [
                result.mixed_ratio
                for result in frame_results
                if result.analysis_status == "ok"
            ]
        ),
        median_outlier_ratio=_finite_median(
            [
                result.outlier_ratio
                for result in frame_results
                if result.analysis_status == "ok"
            ]
        ),
        median_invalid_ratio=_finite_median(
            [
                result.invalid_ratio
                for result in frame_results
                if result.analysis_status == "ok"
            ]
        ),
        median_transition_width_px=_finite_median(
            [
                result.transition_width_px
                for result in frame_results
                if result.transition_status == "ok"
            ]
        ),
        transition_width_p95_px=_finite_percentile(
            [
                result.transition_width_px
                for result in frame_results
                if result.transition_status == "ok"
            ],
            95,
        ),
        median_nominal_edge_offset_px=_finite_median(
            [
                result.nominal_edge_offset_px
                for result in frame_results
                if result.transition_status == "ok"
            ]
        ),
        nominal_edge_offset_std_px=_finite_std(
            [
                result.nominal_edge_offset_px
                for result in frame_results
                if result.transition_status == "ok"
            ]
        ),
        valid_frames=valid_frames,
        rejected_frames=rejected_frames,
        valid_transition_frames=valid_transition_frames,
        failed_transition_frames=failed_transition_frames,
    )


def _reference_is_sufficient(
    reference: ReferenceDepthResult,
    config: EdgeROIConfig,
) -> bool:
    """Apply configured per-frame reference count and ratio gates."""
    return (
        reference.valid_count
        >= config.reference.minimum_valid_count
        and reference.valid_ratio
        >= config.reference.minimum_valid_ratio
        and np.isfinite(reference.median_mm)
    )


def _rejected_frame_analysis(
    *,
    frame_index: int,
    foreground_reference: ReferenceDepthResult,
    background_reference: ReferenceDepthResult,
    status: str,
) -> EdgeFrameAnalysis:
    """Build one rejected frame while preserving available references."""
    nan = float("nan")
    result = FrameEdgeResult(
        frame_index=frame_index,
        foreground_reference_mm=(
            foreground_reference.median_mm
        ),
        background_reference_mm=(
            background_reference.median_mm
        ),
        foreground_bleeding_ratio=nan,
        foreground_bleeding_max_distance_px=nan,
        background_bleeding_ratio=nan,
        background_bleeding_max_distance_px=nan,
        mixed_ratio=nan,
        peak_mixed_ratio=nan,
        peak_mixed_distance_px=nan,
        outlier_ratio=nan,
        invalid_ratio=nan,
        invalid_band_width_px=nan,
        transition_width_px=nan,
        nominal_edge_offset_px=nan,
        analysis_status=status,
        transition_status="not_analyzed",
    )
    return EdgeFrameAnalysis(
        result=result,
        foreground_reference=foreground_reference,
        background_reference=background_reference,
        profile=None,
        label_map=None,
    )


def _build_distance_profile(
    *,
    distance_min_px: np.ndarray,
    distance_max_px: np.ndarray,
    distance_center_px: np.ndarray,
    pixel_count: np.ndarray,
    valid_count: np.ndarray,
    foreground_count: np.ndarray,
    background_count: np.ndarray,
    mixed_count: np.ndarray,
    outlier_count: np.ndarray,
    invalid_count: np.ndarray,
) -> DistanceProfileResult:
    """Build ratios from counts using their documented denominators."""
    return DistanceProfileResult(
        distance_min_px=distance_min_px.astype(
            np.float64,
            copy=False,
        ),
        distance_max_px=distance_max_px.astype(
            np.float64,
            copy=False,
        ),
        distance_center_px=distance_center_px.astype(
            np.float64,
            copy=False,
        ),
        pixel_count=pixel_count.astype(np.int64, copy=False),
        valid_count=valid_count.astype(np.int64, copy=False),
        foreground_count=foreground_count.astype(
            np.int64,
            copy=False,
        ),
        background_count=background_count.astype(
            np.int64,
            copy=False,
        ),
        mixed_count=mixed_count.astype(np.int64, copy=False),
        outlier_count=outlier_count.astype(
            np.int64,
            copy=False,
        ),
        invalid_count=invalid_count.astype(
            np.int64,
            copy=False,
        ),
        foreground_ratio=_safe_array_ratio(
            foreground_count,
            valid_count,
        ),
        background_ratio=_safe_array_ratio(
            background_count,
            valid_count,
        ),
        mixed_ratio=_safe_array_ratio(
            mixed_count,
            valid_count,
        ),
        outlier_ratio=_safe_array_ratio(
            outlier_count,
            valid_count,
        ),
        invalid_ratio=_safe_array_ratio(
            invalid_count,
            pixel_count,
        ),
    )


def _combine_profiles(
    profiles: Sequence[DistanceProfileResult],
) -> DistanceProfileResult:
    """Sum per-frame counts before recomputing aggregate ratios."""
    first = profiles[0]
    _validate_profile(first)
    count_fields = (
        "pixel_count",
        "valid_count",
        "foreground_count",
        "background_count",
        "mixed_count",
        "outlier_count",
        "invalid_count",
    )
    totals = {
        field_name: np.zeros_like(
            getattr(first, field_name),
            dtype=np.int64,
        )
        for field_name in count_fields
    }

    for profile in profiles:
        _validate_profile(profile)
        for coordinate_name in (
            "distance_min_px",
            "distance_max_px",
            "distance_center_px",
        ):
            if not np.array_equal(
                getattr(profile, coordinate_name),
                getattr(first, coordinate_name),
            ):
                raise ValueError(
                    "all frame profiles must use identical distance bins"
                )
        for field_name in count_fields:
            totals[field_name] += getattr(profile, field_name)

    return _build_distance_profile(
        distance_min_px=first.distance_min_px.copy(),
        distance_max_px=first.distance_max_px.copy(),
        distance_center_px=first.distance_center_px.copy(),
        **totals,
    )


def _find_descending_crossing(
    distance: np.ndarray,
    probability: np.ndarray,
    threshold: float,
) -> ProbabilityCrossingResult:
    """Find exactly one descending threshold crossing by interpolation."""
    finite = np.isfinite(distance) & np.isfinite(probability)
    if np.count_nonzero(finite) < 2:
        return ProbabilityCrossingResult(
            distance_px=float("nan"),
            status="insufficient_profile",
        )

    candidates: list[float] = []
    for index in range(distance.size - 1):
        if not finite[index] or not finite[index + 1]:
            continue

        x1 = float(distance[index])
        x2 = float(distance[index + 1])
        y1 = float(probability[index])
        y2 = float(probability[index + 1])

        if y1 == y2 == threshold:
            return ProbabilityCrossingResult(
                distance_px=float("nan"),
                status="nonmonotonic_crossing",
            )
        if y1 >= threshold >= y2 and y1 != y2:
            crossing = x1 + (
                (threshold - y1)
                * (x2 - x1)
                / (y2 - y1)
            )
            if not any(
                math.isclose(
                    crossing,
                    existing,
                    rel_tol=1e-9,
                    abs_tol=1e-12,
                )
                for existing in candidates
            ):
                candidates.append(float(crossing))

    if not candidates:
        return ProbabilityCrossingResult(
            distance_px=float("nan"),
            status="missing_crossing",
        )
    if len(candidates) > 1:
        return ProbabilityCrossingResult(
            distance_px=float("nan"),
            status="nonmonotonic_crossing",
        )
    return ProbabilityCrossingResult(
        distance_px=candidates[0],
        status="ok",
    )


def _undefined_transition(status: str) -> TransitionResult:
    """Return undefined transition values with an explicit status."""
    nan = float("nan")
    return TransitionResult(
        high_crossing_px=nan,
        center_crossing_px=nan,
        low_crossing_px=nan,
        transition_width_px=nan,
        nominal_edge_offset_px=nan,
        status=status,
    )


def _safe_array_ratio(
    numerator: np.ndarray,
    denominator: np.ndarray,
) -> np.ndarray:
    """Divide count arrays and leave zero-denominator entries as NaN."""
    ratio = np.full(
        denominator.shape,
        np.nan,
        dtype=np.float64,
    )
    np.divide(
        numerator,
        denominator,
        out=ratio,
        where=denominator > 0,
    )
    return ratio


def _safe_scalar_ratio(
    numerator: int,
    denominator: int,
) -> float:
    """Divide scalar counts or return NaN for an empty denominator."""
    if denominator == 0:
        return float("nan")
    return float(numerator / denominator)


def _is_valid_label(labels: np.ndarray) -> np.ndarray:
    """Return the valid-class mask used by every class-ratio denominator."""
    return (
        (labels == int(EdgePixelLabel.FOREGROUND))
        | (labels == int(EdgePixelLabel.BACKGROUND))
        | (labels == int(EdgePixelLabel.MIXED))
        | (labels == int(EdgePixelLabel.OUTLIER))
    )


def _validate_prepared_frame(frame: np.ndarray) -> None:
    """Validate one prepared full-image depth frame."""
    if not isinstance(frame, np.ndarray):
        raise TypeError(
            "prepared_depth_frame must be a numpy.ndarray"
        )
    if frame.ndim != 2:
        raise ValueError(
            "prepared_depth_frame must have shape (H, W)"
        )
    if frame.dtype != np.float32:
        raise ValueError(
            "prepared_depth_frame must have dtype float32"
        )
    if frame.shape[0] == 0 or frame.shape[1] == 0:
        raise ValueError(
            "prepared_depth_frame dimensions must be positive"
        )
    if np.any(np.isinf(frame)):
        raise ValueError(
            "prepared_depth_frame must contain finite values or NaN"
        )
    finite = np.isfinite(frame)
    if np.any(frame[finite] <= 0.0):
        raise ValueError(
            "finite prepared depth values must be positive"
        )


def _validate_roi_for_frame(
    roi: RectROI,
    image_shape: tuple[int, int],
) -> None:
    """Validate one rectangle against a 2D frame."""
    if not isinstance(roi, RectROI):
        raise TypeError(
            f"roi must be a RectROI; got {type(roi).__name__}"
        )
    height, width = image_shape
    if roi.x + roi.width > width:
        raise ValueError("ROI exceeds image width")
    if roi.y + roi.height > height:
        raise ValueError("ROI exceeds image height")


def _validate_reference_result(
    reference: ReferenceDepthResult,
) -> None:
    """Validate a reference-result object used for classification."""
    if not isinstance(reference, ReferenceDepthResult):
        raise TypeError(
            "reference must be a ReferenceDepthResult"
        )
    if not np.isfinite(reference.median_mm):
        raise ValueError("reference median_mm must be finite")
    if reference.valid_count <= 0:
        raise ValueError("reference valid_count must be positive")


def _validate_label_and_distance_maps(
    label_map: np.ndarray,
    signed_distance_map: np.ndarray,
) -> None:
    """Validate aligned full-image label and signed-distance maps."""
    if not isinstance(label_map, np.ndarray):
        raise TypeError("label_map must be a numpy.ndarray")
    if label_map.ndim != 2:
        raise ValueError("label_map must have shape (H, W)")
    if not np.issubdtype(label_map.dtype, np.integer):
        raise ValueError("label_map must contain integer labels")
    if not isinstance(signed_distance_map, np.ndarray):
        raise TypeError(
            "signed_distance_map must be a numpy.ndarray"
        )
    if signed_distance_map.shape != label_map.shape:
        raise ValueError(
            "signed_distance_map must match label_map shape"
        )
    if not np.all(np.isfinite(signed_distance_map)):
        raise ValueError(
            "signed_distance_map must contain finite values"
        )
    if np.any(label_map < int(EdgePixelLabel.OUTSIDE)) or np.any(
        label_map > int(EdgePixelLabel.OUTLIER)
    ):
        raise ValueError("label_map contains an unknown label")


def _validate_distance_parameters(
    distance_bin_px: float,
    max_edge_distance_px: float,
) -> tuple[float, float, int]:
    """Validate symmetric distance-bin configuration."""
    bin_width = _normalize_positive_real(
        distance_bin_px,
        "distance_bin_px",
    )
    max_distance = _normalize_positive_real(
        max_edge_distance_px,
        "max_edge_distance_px",
    )
    bins_per_side_float = max_distance / bin_width
    bins_per_side = int(round(bins_per_side_float))
    if not math.isclose(
        bins_per_side_float,
        bins_per_side,
        rel_tol=1e-9,
        abs_tol=1e-12,
    ):
        raise ValueError(
            "max_edge_distance_px must be an integer multiple "
            "of distance_bin_px"
        )
    return bin_width, max_distance, bins_per_side


def _validate_profile(profile: DistanceProfileResult) -> None:
    """Validate the aligned one-dimensional arrays of a distance profile."""
    if not isinstance(profile, DistanceProfileResult):
        raise TypeError(
            "profile must be a DistanceProfileResult"
        )
    expected_shape = profile.distance_center_px.shape
    if (
        len(expected_shape) != 1
        or expected_shape[0] == 0
    ):
        raise ValueError(
            "profile arrays must be non-empty and one-dimensional"
        )
    for field_name in DistanceProfileResult.__dataclass_fields__:
        value = getattr(profile, field_name)
        if (
            not isinstance(value, np.ndarray)
            or value.shape != expected_shape
        ):
            raise ValueError(
                f"{field_name} must have shape {expected_shape}"
            )


def _normalize_finite_real(
    value: object,
    field_name: str,
) -> float:
    """Return a finite real number as a Python float."""
    if (
        isinstance(value, (bool, np.bool_))
        or not isinstance(value, Real)
    ):
        raise TypeError(f"{field_name} must be a real number")
    normalized = float(value)
    if not math.isfinite(normalized):
        raise ValueError(f"{field_name} must be finite")
    return normalized


def _normalize_positive_real(
    value: object,
    field_name: str,
) -> float:
    """Return a finite strictly positive real number."""
    normalized = _normalize_finite_real(value, field_name)
    if normalized <= 0.0:
        raise ValueError(f"{field_name} must be positive")
    return normalized


def _normalize_unit_interval(
    value: object,
    field_name: str,
) -> float:
    """Return a finite ratio between zero and one."""
    normalized = _normalize_finite_real(value, field_name)
    if not 0.0 <= normalized <= 1.0:
        raise ValueError(
            f"{field_name} must be between 0 and 1"
        )
    return normalized


def _normalize_probability(
    value: object,
    field_name: str,
) -> float:
    """Return a finite probability strictly between zero and one."""
    normalized = _normalize_finite_real(value, field_name)
    if not 0.0 < normalized < 1.0:
        raise ValueError(
            f"{field_name} must be between 0 and 1"
        )
    return normalized


def _finite_values(values: Sequence[float]) -> np.ndarray:
    """Return only finite scalar values as float64."""
    array = np.asarray(values, dtype=np.float64)
    return array[np.isfinite(array)]


def _finite_median(values: Sequence[float]) -> float:
    finite = _finite_values(values)
    if finite.size == 0:
        return float("nan")
    return float(np.median(finite))


def _finite_percentile(
    values: Sequence[float],
    percentile: float,
) -> float:
    finite = _finite_values(values)
    if finite.size == 0:
        return float("nan")
    return float(np.percentile(finite, percentile))


def _finite_std(values: Sequence[float]) -> float:
    finite = _finite_values(values)
    if finite.size == 0:
        return float("nan")
    return float(np.std(finite, ddof=0))
