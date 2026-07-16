"""Prepare raw depth frames for numerical analysis."""

import numpy as np


MAX_UINT16 = np.iinfo(np.uint16).max


def prepare_depth(depth: np.ndarray) -> np.ndarray:
    """Return a float32 copy with excluded raw samples represented as NaN.

    Zero represents a no-depth sample. The maximum uint16 value is an
    observed special value and is also excluded from numerical analysis.
    """
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

    # Convert raw special values to NaN for downstream nan-aware metrics.
    prepared = depth.astype(np.float32, copy=True)
    excluded = (depth == 0) | (depth == MAX_UINT16)
    prepared[excluded] = np.nan
    return prepared
