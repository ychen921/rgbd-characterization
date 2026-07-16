"""Measure special raw depth-value occurrence inside an ROI."""

from dataclasses import dataclass

import numpy as np

from src.preprocessing.depth import MAX_UINT16


@dataclass(frozen=True)
class DepthQualityResult:
    """Store aggregate and per-pixel raw depth quality statistics."""

    zero_ratio: float
    zero_ratio_map: np.ndarray

    max_uint16_ratio: float
    max_uint16_ratio_map: np.ndarray

    max_uint16_affected_frames: int
    max_uint16_max_pixels_per_frame: int


def compute_depth_quality(depth: np.ndarray) -> DepthQualityResult:
    """Compute special-value occurrence statistics for raw ROI frames.

    ``depth`` must contain unprepared uint16 frames with shape ``(N, H, W)``.
    Ratios are computed only from the samples passed to this function, so the
    caller should crop the frames to the selected ROI before calling it.

    For an empty frame sequence, ratios are undefined and are returned as NaN;
    frame-count statistics are returned as zero.
    """
    _validate_depth(depth)

    num_frames, height, width = depth.shape
    if num_frames == 0:
        nan_map = np.full((height, width), np.nan, dtype=np.float64)
        return DepthQualityResult(
            zero_ratio=float("nan"),
            zero_ratio_map=nan_map.copy(),
            max_uint16_ratio=float("nan"),
            max_uint16_ratio_map=nan_map,
            max_uint16_affected_frames=0,
            max_uint16_max_pixels_per_frame=0,
        )

    zero_mask = depth == 0
    max_uint16_mask = depth == MAX_UINT16

    zero_ratio = float(np.count_nonzero(zero_mask) / depth.size)
    zero_ratio_map = np.mean(zero_mask, axis=0)

    max_uint16_ratio = float(
        np.count_nonzero(max_uint16_mask) / depth.size
    )
    max_uint16_ratio_map = np.mean(max_uint16_mask, axis=0)

    max_uint16_pixels_per_frame = np.count_nonzero(
        max_uint16_mask,
        axis=(1, 2),
    )
    max_uint16_affected_frames = int(
        np.count_nonzero(max_uint16_pixels_per_frame)
    )
    max_uint16_max_pixels_per_frame = int(
        np.max(max_uint16_pixels_per_frame)
    )

    return DepthQualityResult(
        zero_ratio=zero_ratio,
        zero_ratio_map=zero_ratio_map,
        max_uint16_ratio=max_uint16_ratio,
        max_uint16_ratio_map=max_uint16_ratio_map,
        max_uint16_affected_frames=max_uint16_affected_frames,
        max_uint16_max_pixels_per_frame=max_uint16_max_pixels_per_frame,
    )


def _validate_depth(depth: np.ndarray) -> None:
    """Validate raw ROI frames for depth-quality computation."""
    if not isinstance(depth, np.ndarray):
        raise TypeError(
            f"depth must be a numpy.ndarray; got {type(depth).__name__}"
        )

    if depth.ndim != 3:
        raise ValueError(
            f"depth must have shape (N, H, W); got shape {depth.shape}"
        )

    if depth.dtype != np.uint16:
        raise ValueError(f"depth must have dtype uint16; got {depth.dtype}")

    if depth.shape[1] == 0 or depth.shape[2] == 0:
        raise ValueError(
            "depth spatial dimensions must be positive; got shape "
            f"{depth.shape}"
        )
