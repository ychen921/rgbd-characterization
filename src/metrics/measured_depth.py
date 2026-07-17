"""Summarize measured depth inside a prepared depth ROI."""

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class MeasuredDepthResult:
    """Store per-frame and aggregate measured-depth statistics."""

    frame_median: np.ndarray
    median_depth: float
    mean_depth: float
    std_depth: float
    p05_depth: float
    p95_depth: float


def compute_measured_depth(depth: np.ndarray) -> MeasuredDepthResult:
    """Compute per-frame ROI median depth and cross-frame statistics.

    ``depth`` must contain prepared float32 frames with shape ``(N, H, W)``.
    Excluded samples must be represented as NaN. Each frame contributes at
    most one value to the aggregate statistics, so frames with different
    valid-pixel counts retain equal weight.

    An all-invalid frame has a NaN frame median and is excluded from aggregate
    statistics. If the frame sequence is empty, or if every frame is invalid,
    all aggregate statistics are returned as NaN. Standard deviation uses the
    population definition (``ddof=0``).
    """
    _validate_depth(depth)

    num_frames = depth.shape[0]
    # Preallocate one output slot per input frame to preserve frame alignment.
    frame_median = np.full(num_frames, np.nan, dtype=np.float64)
    if num_frames == 0:
        return _undefined_result(frame_median)

    # Select frames before calling nanmedian so all-NaN slices do not emit a
    # RuntimeWarning. Keeping NaN entries preserves the input frame indices.
    valid_frame_mask = np.any(~np.isnan(depth), axis=(1, 2))
    if not np.any(valid_frame_mask):
        return _undefined_result(frame_median)

    frame_median[valid_frame_mask] = np.nanmedian(
        depth[valid_frame_mask],
        axis=(1, 2),
    )

    # Give every valid frame equal weight in the cross-frame summary, regardless
    # of how many valid ROI pixels contributed to that frame's median.
    valid_frame_median = frame_median[valid_frame_mask]

    return MeasuredDepthResult(
        frame_median=frame_median,
        median_depth=float(np.median(valid_frame_median)),
        mean_depth=float(np.mean(valid_frame_median)),
        std_depth=float(np.std(valid_frame_median, ddof=0)),
        p05_depth=float(np.percentile(valid_frame_median, 5)),
        p95_depth=float(np.percentile(valid_frame_median, 95)),
    )


def _undefined_result(frame_median: np.ndarray) -> MeasuredDepthResult:
    """Return a result whose aggregate statistics are undefined."""
    return MeasuredDepthResult(
        frame_median=frame_median,
        median_depth=float("nan"),
        mean_depth=float("nan"),
        std_depth=float("nan"),
        p05_depth=float("nan"),
        p95_depth=float("nan"),
    )


def _validate_depth(depth: np.ndarray) -> None:
    """Validate prepared ROI frames for measured-depth computation."""
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
