"""Measure per-pixel temporal noise inside a prepared depth ROI."""

from dataclasses import dataclass
from numbers import Real

import numpy as np


DEFAULT_MIN_VALID_RATIO = 0.9


@dataclass(frozen=True)
class TemporalNoiseResult:
    """Store per-pixel and aggregate temporal noise statistics."""

    std_map: np.ndarray
    median_std: float
    mean_std: float
    p95_std: float


def compute_temporal_noise(
    depth: np.ndarray,
    min_valid_ratio: float = DEFAULT_MIN_VALID_RATIO,
) -> TemporalNoiseResult:
    """Compute temporal depth standard deviation for each ROI pixel.

    ``depth`` must contain prepared float32 frames with shape ``(N, H, W)``.
    Excluded samples must be represented as NaN. Pixels with fewer valid
    samples than ``min_valid_ratio`` are excluded from the result map.

    Standard deviations use the population definition (``ddof=0``). For an
    empty frame sequence, or when no pixel has enough valid samples, result
    statistics are undefined and are returned as NaN.
    """
    _validate_depth(depth)
    valid_ratio_threshold = _validate_min_valid_ratio(min_valid_ratio)

    num_frames, height, width = depth.shape
    if num_frames == 0:
        return _empty_result(height, width)

    # Count each pixel's valid samples across frames.
    valid_count = np.count_nonzero(~np.isnan(depth), axis=0)
    valid_ratio = valid_count / num_frames

    # Keep pixels that have enough valid temporal samples.
    eligible = (valid_count > 0) & (
        valid_ratio >= valid_ratio_threshold
    )

    std_map = np.full((height, width), np.nan, dtype=np.float64)
    if not np.any(eligible):
        return TemporalNoiseResult(
            std_map=std_map,
            median_std=float("nan"),
            mean_std=float("nan"),
            p95_std=float("nan"),
        )

    # Compute per-pixel standard deviation along the frame axis.
    std_map[eligible] = np.nanstd(
        depth[:, eligible],
        axis=0,
        dtype=np.float64,
        ddof=0,
    )

    # Summarize temporal noise across eligible ROI pixels.
    valid_std = std_map[eligible]

    return TemporalNoiseResult(
        std_map=std_map,
        median_std=float(np.median(valid_std)),
        mean_std=float(np.mean(valid_std)),
        p95_std=float(np.percentile(valid_std, 95)),
    )


def _empty_result(height: int, width: int) -> TemporalNoiseResult:
    """Return an undefined result with the requested spatial shape."""
    return TemporalNoiseResult(
        std_map=np.full((height, width), np.nan, dtype=np.float64),
        median_std=float("nan"),
        mean_std=float("nan"),
        p95_std=float("nan"),
    )


def _validate_depth(depth: np.ndarray) -> None:
    """Validate prepared ROI frames for temporal-noise computation."""
    if not isinstance(depth, np.ndarray):
        raise TypeError(
            f"depth must be a numpy.ndarray; got {type(depth).__name__}"
        )

    if depth.ndim != 3:
        raise ValueError(
            f"depth must have shape (N, H, W); got shape {depth.shape}"
        )

    if depth.dtype != np.float32:
        raise ValueError(f"depth must have dtype float32; got {depth.dtype}")

    if depth.shape[1] == 0 or depth.shape[2] == 0:
        raise ValueError(
            "depth spatial dimensions must be positive; got shape "
            f"{depth.shape}"
        )

    if np.any(np.isinf(depth)):
        raise ValueError("depth must contain only finite values or NaN")


def _validate_min_valid_ratio(min_valid_ratio: float) -> float:
    """Validate and normalize the minimum per-pixel valid-sample ratio."""
    if isinstance(min_valid_ratio, (bool, np.bool_)) or not isinstance(
        min_valid_ratio,
        Real,
    ):
        raise TypeError("min_valid_ratio must be a real number")

    normalized = float(min_valid_ratio)
    if not np.isfinite(normalized) or not 0.0 <= normalized <= 1.0:
        raise ValueError("min_valid_ratio must be between 0 and 1")

    return normalized
