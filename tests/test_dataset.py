"""Tests for the internal raw depth dataset format."""

from pathlib import Path

import numpy as np
import pytest

from src.io.dataset import DepthDataset


@pytest.fixture
def depth() -> np.ndarray:
    return np.arange(5 * 4 * 3, dtype=np.uint16).reshape(5, 4, 3)


@pytest.fixture
def timestamps_ns() -> np.ndarray:
    return np.array([100, 200, 300, 400, 500], dtype=np.int64)


def test_dataset_properties(
    depth: np.ndarray,
    timestamps_ns: np.ndarray,
) -> None:
    dataset = DepthDataset(depth=depth, timestamps_ns=timestamps_ns)

    assert dataset.num_frames == 5
    assert dataset.height == 4
    assert dataset.width == 3


def test_save_load_round_trip(
    tmp_path: Path,
    depth: np.ndarray,
    timestamps_ns: np.ndarray,
) -> None:
    dataset = DepthDataset(depth=depth, timestamps_ns=timestamps_ns)
    output_path = tmp_path / "depth.npz"

    dataset.save(output_path)
    loaded = DepthDataset.load(output_path)

    assert np.array_equal(loaded.depth, dataset.depth)
    assert np.array_equal(loaded.timestamps_ns, dataset.timestamps_ns)
    assert loaded.depth.dtype == np.uint16
    assert loaded.timestamps_ns.dtype == np.int64


@pytest.mark.parametrize(
    ("invalid_depth", "expected_message"),
    [
        (np.zeros((4, 3), dtype=np.uint16), r"shape \(N, H, W\)"),
        (np.zeros((5, 4, 3), dtype=np.float32), "dtype uint16"),
    ],
)
def test_rejects_invalid_depth_array(
    invalid_depth: np.ndarray,
    expected_message: str,
    timestamps_ns: np.ndarray,
) -> None:
    with pytest.raises(ValueError, match=expected_message):
        DepthDataset(depth=invalid_depth, timestamps_ns=timestamps_ns)


@pytest.mark.parametrize(
    ("invalid_timestamps", "expected_message"),
    [
        (np.zeros((5, 1), dtype=np.int64), r"shape \(N,\)"),
        (np.zeros(5, dtype=np.uint64), "dtype int64"),
    ],
)
def test_rejects_invalid_timestamp_array(
    invalid_timestamps: np.ndarray,
    expected_message: str,
    depth: np.ndarray,
) -> None:
    with pytest.raises(ValueError, match=expected_message):
        DepthDataset(depth=depth, timestamps_ns=invalid_timestamps)


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("depth", [[[1]]]),
        ("timestamps_ns", [100]),
    ],
)
def test_rejects_non_array_fields(
    field: str,
    value: object,
    depth: np.ndarray,
    timestamps_ns: np.ndarray,
) -> None:
    arguments = {
        "depth": depth,
        "timestamps_ns": timestamps_ns,
    }
    arguments[field] = value

    with pytest.raises(TypeError, match=f"{field} must be a numpy.ndarray"):
        DepthDataset(**arguments)


def test_rejects_mismatched_frame_and_timestamp_counts(
    depth: np.ndarray,
    timestamps_ns: np.ndarray,
) -> None:
    with pytest.raises(ValueError, match="frame count does not match"):
        DepthDataset(depth=depth, timestamps_ns=timestamps_ns[:-1])


def test_load_rejects_archive_missing_required_array(
    tmp_path: Path,
    depth: np.ndarray,
) -> None:
    input_path = tmp_path / "missing_timestamps.npz"
    np.savez(input_path, depth=depth)

    with pytest.raises(ValueError, match="timestamps_ns"):
        DepthDataset.load(input_path)
