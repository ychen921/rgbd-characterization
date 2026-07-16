"""Tests for prepared ROI temporal noise metrics."""

import warnings

import numpy as np
import pytest

from src.metrics.temporal import compute_temporal_noise


def test_compute_temporal_noise_returns_expected_std_map() -> None:
    depth = np.array(
        [
            [[514, 1], [10, 20]],
            [[514, 2], [12, 22]],
            [[514, 3], [14, 24]],
        ],
        dtype=np.float32,
    )

    result = compute_temporal_noise(depth)

    expected = np.array(
        [
            [0.0, np.sqrt(2 / 3)],
            [np.sqrt(8 / 3), np.sqrt(8 / 3)],
        ]
    )
    np.testing.assert_allclose(result.std_map, expected)
    assert result.std_map.dtype == np.float64


def test_compute_temporal_noise_ignores_nan_samples() -> None:
    depth = np.array([[[1]], [[np.nan]], [[3]]], dtype=np.float32)

    result = compute_temporal_noise(depth, min_valid_ratio=2 / 3)

    assert result.std_map[0, 0] == pytest.approx(1.0)


def test_compute_temporal_noise_excludes_insufficient_valid_ratio() -> None:
    depth = np.full((10, 1, 2), np.nan, dtype=np.float32)
    depth[:9, 0, 0] = np.arange(9, dtype=np.float32)
    depth[:8, 0, 1] = np.arange(8, dtype=np.float32)

    result = compute_temporal_noise(depth, min_valid_ratio=0.9)

    assert result.std_map[0, 0] == pytest.approx(np.std(np.arange(9)))
    assert np.isnan(result.std_map[0, 1])


def test_compute_temporal_noise_returns_expected_aggregates() -> None:
    depth = np.array(
        [
            [[10, 10, 10, 10]],
            [[10, 12, 14, 16]],
        ],
        dtype=np.float32,
    )

    result = compute_temporal_noise(depth)

    np.testing.assert_allclose(result.std_map, [[0.0, 1.0, 2.0, 3.0]])
    assert result.median_std == pytest.approx(1.5)
    assert result.mean_std == pytest.approx(1.5)
    assert result.p95_std == pytest.approx(2.85)


def test_compute_temporal_noise_aggregates_only_eligible_pixels() -> None:
    depth = np.array(
        [
            [[10, 1]],
            [[12, np.nan]],
        ],
        dtype=np.float32,
    )

    result = compute_temporal_noise(depth)

    np.testing.assert_allclose(result.std_map[0, 0], 1.0)
    assert np.isnan(result.std_map[0, 1])
    assert result.median_std == pytest.approx(1.0)
    assert result.mean_std == pytest.approx(1.0)
    assert result.p95_std == pytest.approx(1.0)


def test_compute_temporal_noise_accepts_single_valid_sample() -> None:
    depth = np.array([[[np.nan]], [[514]], [[np.nan]]], dtype=np.float32)

    result = compute_temporal_noise(depth, min_valid_ratio=0.0)

    assert result.std_map[0, 0] == 0.0


def test_compute_temporal_noise_handles_no_eligible_pixels() -> None:
    depth = np.full((3, 2, 2), np.nan, dtype=np.float32)

    with warnings.catch_warnings(record=True) as warning_record:
        warnings.simplefilter("always")
        result = compute_temporal_noise(depth)

    assert len(warning_record) == 0
    assert np.all(np.isnan(result.std_map))
    assert np.isnan(result.median_std)
    assert np.isnan(result.mean_std)
    assert np.isnan(result.p95_std)


def test_compute_temporal_noise_handles_empty_frame_sequence() -> None:
    depth = np.empty((0, 2, 3), dtype=np.float32)

    with warnings.catch_warnings(record=True) as warning_record:
        warnings.simplefilter("always")
        result = compute_temporal_noise(depth)

    assert len(warning_record) == 0
    assert result.std_map.shape == (2, 3)
    assert result.std_map.dtype == np.float64
    assert np.all(np.isnan(result.std_map))
    assert np.isnan(result.median_std)
    assert np.isnan(result.mean_std)
    assert np.isnan(result.p95_std)


def test_compute_temporal_noise_does_not_modify_input() -> None:
    depth = np.array([[[514, np.nan]], [[515, 516]]], dtype=np.float32)
    original = depth.copy()

    compute_temporal_noise(depth, min_valid_ratio=0.5)

    np.testing.assert_equal(depth, original)


def test_compute_temporal_noise_rejects_non_array_input() -> None:
    with pytest.raises(TypeError, match="depth must be a numpy.ndarray"):
        compute_temporal_noise([[[514.0]]])


@pytest.mark.parametrize(
    "depth",
    [
        np.zeros((2, 3), dtype=np.float32),
        np.zeros((1, 2, 3, 4), dtype=np.float32),
    ],
)
def test_compute_temporal_noise_rejects_non_frame_array(
    depth: np.ndarray,
) -> None:
    with pytest.raises(ValueError, match=r"shape \(N, H, W\)"):
        compute_temporal_noise(depth)


@pytest.mark.parametrize("dtype", [np.uint16, np.int32, np.float64])
def test_compute_temporal_noise_rejects_non_float32_dtype(
    dtype: np.dtype,
) -> None:
    depth = np.zeros((1, 2, 3), dtype=dtype)

    with pytest.raises(ValueError, match="dtype float32"):
        compute_temporal_noise(depth)


@pytest.mark.parametrize(
    "shape",
    [
        (1, 0, 3),
        (1, 2, 0),
    ],
)
def test_compute_temporal_noise_rejects_empty_spatial_dimension(
    shape: tuple[int, int, int],
) -> None:
    depth = np.empty(shape, dtype=np.float32)

    with pytest.raises(ValueError, match="spatial dimensions must be positive"):
        compute_temporal_noise(depth)


@pytest.mark.parametrize("value", [np.inf, -np.inf])
def test_compute_temporal_noise_rejects_infinite_depth(value: float) -> None:
    depth = np.array([[[value]]], dtype=np.float32)

    with pytest.raises(ValueError, match="finite values or NaN"):
        compute_temporal_noise(depth)


@pytest.mark.parametrize("min_valid_ratio", [-0.1, 1.1, np.nan, np.inf])
def test_compute_temporal_noise_rejects_invalid_valid_ratio(
    min_valid_ratio: float,
) -> None:
    depth = np.zeros((1, 1, 1), dtype=np.float32)

    with pytest.raises(ValueError, match="between 0 and 1"):
        compute_temporal_noise(depth, min_valid_ratio=min_valid_ratio)


@pytest.mark.parametrize("min_valid_ratio", ["0.9", True, None])
def test_compute_temporal_noise_rejects_non_numeric_valid_ratio(
    min_valid_ratio: object,
) -> None:
    depth = np.zeros((1, 1, 1), dtype=np.float32)

    with pytest.raises(TypeError, match="must be a real number"):
        compute_temporal_noise(depth, min_valid_ratio=min_valid_ratio)
