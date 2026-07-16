"""Tests for raw depth preprocessing."""

import numpy as np
import pytest

from src.preprocessing.depth import prepare_depth


def test_prepare_depth_converts_values_and_excludes_special_samples() -> None:
    depth = np.array(
        [
            [[0, 1, 514], [65534, 65535, 1000]],
            [[200, 300, 400], [500, 600, 700]],
        ],
        dtype=np.uint16,
    )

    prepared = prepare_depth(depth)

    assert prepared.shape == depth.shape
    assert prepared.dtype == np.float32
    assert np.isnan(prepared[0, 0, 0])
    assert np.isnan(prepared[0, 1, 1])
    assert prepared[0, 0, 1] == np.float32(1)
    assert prepared[0, 0, 2] == np.float32(514)
    assert prepared[0, 1, 0] == np.float32(65534)
    assert prepared[1, 1, 2] == np.float32(700)


def test_prepare_depth_does_not_modify_or_share_memory_with_input() -> None:
    depth = np.array([[[0, 514, 65535]]], dtype=np.uint16)
    original = depth.copy()

    prepared = prepare_depth(depth)

    assert np.array_equal(depth, original)
    assert not np.shares_memory(prepared, depth)


def test_prepare_depth_handles_all_excluded_samples() -> None:
    depth = np.array([[[0, 65535]]], dtype=np.uint16)

    prepared = prepare_depth(depth)

    assert np.all(np.isnan(prepared))


def test_prepare_depth_accepts_empty_frame_array() -> None:
    depth = np.empty((0, 2, 3), dtype=np.uint16)

    prepared = prepare_depth(depth)

    assert prepared.shape == (0, 2, 3)
    assert prepared.dtype == np.float32


def test_prepare_depth_rejects_non_array_input() -> None:
    with pytest.raises(TypeError, match="depth must be a numpy.ndarray"):
        prepare_depth([[[514]]])


@pytest.mark.parametrize(
    "depth",
    [
        np.zeros((2, 3), dtype=np.uint16),
        np.zeros((1, 2, 3, 4), dtype=np.uint16),
    ],
)
def test_prepare_depth_rejects_non_frame_array(depth: np.ndarray) -> None:
    with pytest.raises(ValueError, match=r"shape \(N, H, W\)"):
        prepare_depth(depth)


@pytest.mark.parametrize(
    "dtype",
    [np.int32, np.float32],
)
def test_prepare_depth_rejects_non_uint16_dtype(
    dtype: np.dtype,
) -> None:
    depth = np.zeros((1, 2, 3), dtype=dtype)

    with pytest.raises(ValueError, match="dtype uint16"):
        prepare_depth(depth)
