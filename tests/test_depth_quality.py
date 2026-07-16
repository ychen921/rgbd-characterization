"""Tests for raw ROI depth quality metrics."""

import warnings

import numpy as np
import pytest

from src.metrics.depth_quality import compute_depth_quality


def test_compute_depth_quality_returns_expected_statistics() -> None:
    depth = np.array(
        [
            [[0, 1], [65535, 2]],
            [[3, 0], [65535, 65535]],
            [[0, 4], [5, 6]],
        ],
        dtype=np.uint16,
    )

    result = compute_depth_quality(depth)

    assert result.zero_ratio == pytest.approx(3 / 12)
    np.testing.assert_allclose(
        result.zero_ratio_map,
        np.array([[2 / 3, 1 / 3], [0, 0]]),
    )
    assert result.max_uint16_ratio == pytest.approx(3 / 12)
    np.testing.assert_allclose(
        result.max_uint16_ratio_map,
        np.array([[0, 0], [2 / 3, 1 / 3]]),
    )
    assert result.max_uint16_affected_frames == 2
    assert result.max_uint16_max_pixels_per_frame == 2


def test_compute_depth_quality_handles_no_special_values() -> None:
    depth = np.arange(1, 13, dtype=np.uint16).reshape(3, 2, 2)

    result = compute_depth_quality(depth)

    assert result.zero_ratio == 0.0
    assert np.array_equal(result.zero_ratio_map, np.zeros((2, 2)))
    assert result.max_uint16_ratio == 0.0
    assert np.array_equal(result.max_uint16_ratio_map, np.zeros((2, 2)))
    assert result.max_uint16_affected_frames == 0
    assert result.max_uint16_max_pixels_per_frame == 0


def test_compute_depth_quality_handles_all_zero_values() -> None:
    depth = np.zeros((3, 2, 2), dtype=np.uint16)

    result = compute_depth_quality(depth)

    assert result.zero_ratio == 1.0
    assert np.array_equal(result.zero_ratio_map, np.ones((2, 2)))
    assert result.max_uint16_ratio == 0.0
    assert result.max_uint16_affected_frames == 0
    assert result.max_uint16_max_pixels_per_frame == 0


def test_compute_depth_quality_handles_all_max_uint16_values() -> None:
    depth = np.full((3, 2, 2), 65535, dtype=np.uint16)

    result = compute_depth_quality(depth)

    assert result.zero_ratio == 0.0
    assert result.max_uint16_ratio == 1.0
    assert np.array_equal(result.max_uint16_ratio_map, np.ones((2, 2)))
    assert result.max_uint16_affected_frames == 3
    assert result.max_uint16_max_pixels_per_frame == 4


def test_compute_depth_quality_does_not_modify_input() -> None:
    depth = np.array([[[0, 514, 65535]]], dtype=np.uint16)
    original = depth.copy()

    compute_depth_quality(depth)

    assert np.array_equal(depth, original)


def test_compute_depth_quality_handles_empty_frame_sequence() -> None:
    depth = np.empty((0, 2, 3), dtype=np.uint16)

    with warnings.catch_warnings(record=True) as warning_record:
        warnings.simplefilter("always")
        result = compute_depth_quality(depth)

    assert len(warning_record) == 0
    assert np.isnan(result.zero_ratio)
    assert result.zero_ratio_map.shape == (2, 3)
    assert np.all(np.isnan(result.zero_ratio_map))
    assert np.isnan(result.max_uint16_ratio)
    assert result.max_uint16_ratio_map.shape == (2, 3)
    assert np.all(np.isnan(result.max_uint16_ratio_map))
    assert result.max_uint16_affected_frames == 0
    assert result.max_uint16_max_pixels_per_frame == 0


def test_compute_depth_quality_rejects_non_array_input() -> None:
    with pytest.raises(TypeError, match="depth must be a numpy.ndarray"):
        compute_depth_quality([[[514]]])


@pytest.mark.parametrize(
    "depth",
    [
        np.zeros((2, 3), dtype=np.uint16),
        np.zeros((1, 2, 3, 4), dtype=np.uint16),
    ],
)
def test_compute_depth_quality_rejects_non_frame_array(
    depth: np.ndarray,
) -> None:
    with pytest.raises(ValueError, match=r"shape \(N, H, W\)"):
        compute_depth_quality(depth)


@pytest.mark.parametrize("dtype", [np.int32, np.float32])
def test_compute_depth_quality_rejects_non_uint16_dtype(
    dtype: np.dtype,
) -> None:
    depth = np.zeros((1, 2, 3), dtype=dtype)

    with pytest.raises(ValueError, match="dtype uint16"):
        compute_depth_quality(depth)


@pytest.mark.parametrize(
    "shape",
    [
        (1, 0, 3),
        (1, 2, 0),
    ],
)
def test_compute_depth_quality_rejects_empty_spatial_dimension(
    shape: tuple[int, int, int],
) -> None:
    depth = np.empty(shape, dtype=np.uint16)

    with pytest.raises(ValueError, match="spatial dimensions must be positive"):
        compute_depth_quality(depth)
