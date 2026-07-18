"""Tests for baseline metric orchestration."""

import numpy as np
import pytest

from tools.analyze_baseline import compute_baseline_metrics


def test_compute_baseline_metrics_combines_all_metrics() -> None:
    raw_roi = np.array(
        [
            [[0, 10, 65535]],
            [[20, 14, 30]],
        ],
        dtype=np.uint16,
    )

    result = compute_baseline_metrics(raw_roi)

    assert result.depth_quality.zero_ratio == pytest.approx(1 / 6)
    assert result.depth_quality.max_uint16_ratio == pytest.approx(1 / 6)

    np.testing.assert_allclose(
        result.measured_depth.frame_median,
        [10.0, 20.0],
    )
    assert result.measured_depth.median_depth == pytest.approx(15.0)

    np.testing.assert_allclose(
        result.temporal_noise.std_map,
        [[np.nan, 2.0, np.nan]],
        equal_nan=True,
    )
    assert result.temporal_noise.median_std == pytest.approx(2.0)


def test_compute_baseline_metrics_forwards_min_valid_ratio() -> None:
    raw_roi = np.array(
        [
            [[0, 10, 65535]],
            [[20, 14, 30]],
        ],
        dtype=np.uint16,
    )

    result = compute_baseline_metrics(
        raw_roi,
        min_valid_ratio=0.5,
    )

    np.testing.assert_allclose(
        result.temporal_noise.std_map,
        [[0.0, 2.0, 0.0]],
    )
