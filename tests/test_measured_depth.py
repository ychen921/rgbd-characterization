"""Tests for prepared ROI measured-depth metrics."""

import warnings

import numpy as np
import pytest

from src.metrics.measured_depth import compute_measured_depth


def test_compute_measured_depth_returns_expected_frame_medians() -> None:
    depth = np.array(
        [
            [[1, 2], [3, 4]],
            [[10, 20], [30, 40]],
        ],
        dtype=np.float32,
    )

    result = compute_measured_depth(depth)

    np.testing.assert_allclose(result.frame_median, [2.5, 25.0])
    assert result.frame_median.shape == (2,)
    assert result.frame_median.dtype == np.float64


def test_compute_measured_depth_ignores_nan_samples() -> None:
    depth = np.array(
        [
            [[1, np.nan], [3, 5]],
            [[np.nan, 10], [20, np.nan]],
        ],
        dtype=np.float32,
    )

    result = compute_measured_depth(depth)

    np.testing.assert_allclose(result.frame_median, [3.0, 15.0])


def test_compute_measured_depth_returns_expected_aggregates() -> None:
    depth = np.array(
        [
            [[10]],
            [[20]],
            [[30]],
            [[40]],
        ],
        dtype=np.float32,
    )

    result = compute_measured_depth(depth)

    assert result.median_depth == pytest.approx(25.0)
    assert result.mean_depth == pytest.approx(25.0)
    assert result.std_depth == pytest.approx(np.sqrt(125.0))
    assert result.p05_depth == pytest.approx(11.5)
    assert result.p95_depth == pytest.approx(38.5)


def test_compute_measured_depth_excludes_all_invalid_frame() -> None:
    depth = np.array(
        [
            [[10, 12]],
            [[np.nan, np.nan]],
            [[20, 24]],
        ],
        dtype=np.float32,
    )

    result = compute_measured_depth(depth)

    np.testing.assert_allclose(result.frame_median[[0, 2]], [11.0, 22.0])
    assert np.isnan(result.frame_median[1])
    assert result.median_depth == pytest.approx(16.5)
    assert result.mean_depth == pytest.approx(16.5)
    assert result.std_depth == pytest.approx(5.5)


def test_compute_measured_depth_accepts_single_valid_sample() -> None:
    depth = np.array(
        [[[np.nan, 514, np.nan]]],
        dtype=np.float32,
    )

    result = compute_measured_depth(depth)

    np.testing.assert_allclose(result.frame_median, [514.0])
    assert result.median_depth == pytest.approx(514.0)
    assert result.mean_depth == pytest.approx(514.0)
    assert result.std_depth == pytest.approx(0.0)
    assert result.p05_depth == pytest.approx(514.0)
    assert result.p95_depth == pytest.approx(514.0)


def test_compute_measured_depth_handles_all_invalid_frames() -> None:
    depth = np.full((3, 2, 2), np.nan, dtype=np.float32)

    with warnings.catch_warnings(record=True) as warning_record:
        warnings.simplefilter("always")
        result = compute_measured_depth(depth)

    assert len(warning_record) == 0
    assert result.frame_median.shape == (3,)
    assert result.frame_median.dtype == np.float64
    assert np.all(np.isnan(result.frame_median))
    assert np.isnan(result.median_depth)
    assert np.isnan(result.mean_depth)
    assert np.isnan(result.std_depth)
    assert np.isnan(result.p05_depth)
    assert np.isnan(result.p95_depth)


def test_compute_measured_depth_handles_empty_frame_sequence() -> None:
    depth = np.empty((0, 2, 3), dtype=np.float32)

    with warnings.catch_warnings(record=True) as warning_record:
        warnings.simplefilter("always")
        result = compute_measured_depth(depth)

    assert len(warning_record) == 0
    assert result.frame_median.shape == (0,)
    assert result.frame_median.dtype == np.float64
    assert np.isnan(result.median_depth)
    assert np.isnan(result.mean_depth)
    assert np.isnan(result.std_depth)
    assert np.isnan(result.p05_depth)
    assert np.isnan(result.p95_depth)


def test_compute_measured_depth_does_not_modify_input() -> None:
    depth = np.array([[[514, np.nan]], [[515, 516]]], dtype=np.float32)
    original = depth.copy()

    compute_measured_depth(depth)

    np.testing.assert_equal(depth, original)


def test_compute_measured_depth_rejects_non_array_input() -> None:
    with pytest.raises(TypeError, match="depth must be a numpy.ndarray"):
        compute_measured_depth([[[514.0]]])


@pytest.mark.parametrize(
    "depth",
    [
        np.zeros((2, 3), dtype=np.float32),
        np.zeros((1, 2, 3, 4), dtype=np.float32),
    ],
)
def test_compute_measured_depth_rejects_non_frame_array(
    depth: np.ndarray,
) -> None:
    with pytest.raises(ValueError, match=r"shape \(N, H, W\)"):
        compute_measured_depth(depth)


@pytest.mark.parametrize("dtype", [np.uint16, np.int32, np.float64])
def test_compute_measured_depth_rejects_non_float32_dtype(
    dtype: np.dtype,
) -> None:
    depth = np.zeros((1, 2, 3), dtype=dtype)

    with pytest.raises(ValueError, match="dtype float32"):
        compute_measured_depth(depth)


@pytest.mark.parametrize(
    "shape",
    [
        (1, 0, 3),
        (1, 2, 0),
    ],
)
def test_compute_measured_depth_rejects_empty_spatial_dimension(
    shape: tuple[int, int, int],
) -> None:
    depth = np.empty(shape, dtype=np.float32)

    with pytest.raises(ValueError, match="spatial dimensions must be positive"):
        compute_measured_depth(depth)


@pytest.mark.parametrize("value", [np.inf, -np.inf])
def test_compute_measured_depth_rejects_infinite_depth(value: float) -> None:
    depth = np.array([[[value]]], dtype=np.float32)

    with pytest.raises(ValueError, match="finite values or NaN"):
        compute_measured_depth(depth)
