"""Tests for the unpaired RGB and aligned-depth dataset contract."""

from pathlib import Path

import numpy as np
import pytest

from src.io.alignment_dataset import AlignmentDataset


@pytest.fixture
def valid_arrays() -> dict[str, np.ndarray]:
    rgb = np.arange(3 * 2 * 4 * 3, dtype=np.uint8).reshape(3, 2, 4, 3)
    aligned_depth = np.array(
        [
            [[0, 100, 200, 300], [400, 500, 600, 65535]],
            [[10, 110, 210, 310], [410, 510, 610, 710]],
            [[20, 120, 220, 320], [420, 520, 620, 720]],
        ],
        dtype=np.uint16,
    )
    return {
        "rgb": rgb,
        "aligned_depth": aligned_depth,
        "rgb_timestamp_ns": np.array([100, 200, 300], dtype=np.int64),
        "depth_timestamp_ns": np.array([105, 205, 305], dtype=np.int64),
        "rgb_recorded_timestamp_ns": np.array(
            [130, 230, 330], dtype=np.int64
        ),
        "depth_recorded_timestamp_ns": np.array(
            [135, 235, 335], dtype=np.int64
        ),
    }


def make_dataset(
    valid_arrays: dict[str, np.ndarray],
    **overrides: object,
) -> AlignmentDataset:
    arguments: dict[str, object] = dict(valid_arrays)
    arguments.update(overrides)
    return AlignmentDataset(**arguments)


def test_dataset_properties_and_contract_constants(
    valid_arrays: dict[str, np.ndarray],
) -> None:
    dataset = make_dataset(valid_arrays)

    assert dataset.num_rgb_frames == 3
    assert dataset.num_depth_frames == 3
    assert dataset.height == 2
    assert dataset.width == 4
    assert dataset.rgb_height == dataset.depth_height == 2
    assert dataset.rgb_width == dataset.depth_width == 4
    assert dataset.COLOR_ENCODING == "rgb8"
    assert dataset.COLOR_CHANNEL_ORDER == "RGB"
    assert dataset.DEPTH_ENCODING == "16UC1"
    assert dataset.DEPTH_PRECISION == "1mm"
    assert dataset.DEPTH_UNIT == "mm"
    assert dataset.DEPTH_INVALID_VALUES == (0, 65535)
    assert dataset.PRIMARY_TIMESTAMP_SOURCE == "message_header"
    assert dataset.RECORDED_TIMESTAMP_SOURCE == "rosbag_storage"


def test_allows_unpaired_streams_with_different_frame_counts(
    valid_arrays: dict[str, np.ndarray],
) -> None:
    dataset = make_dataset(
        valid_arrays,
        aligned_depth=valid_arrays["aligned_depth"][:2],
        depth_timestamp_ns=valid_arrays["depth_timestamp_ns"][:2],
        depth_recorded_timestamp_ns=valid_arrays[
            "depth_recorded_timestamp_ns"
        ][:2],
    )

    assert dataset.num_rgb_frames == 3
    assert dataset.num_depth_frames == 2


def test_preserves_invalid_depth_values(
    valid_arrays: dict[str, np.ndarray],
) -> None:
    dataset = make_dataset(valid_arrays)

    assert dataset.aligned_depth[0, 0, 0] == 0
    assert dataset.aligned_depth[0, 1, 3] == 65535


def test_save_load_round_trip(
    tmp_path: Path,
    valid_arrays: dict[str, np.ndarray],
) -> None:
    dataset = make_dataset(valid_arrays)
    output_dir = tmp_path / "data" / "scene05_alignment_test"

    dataset.save(output_dir)
    loaded = AlignmentDataset.load(output_dir)

    assert {path.name for path in output_dir.iterdir()} == {
        "rgb.npz",
        "aligned_depth.npz",
        "timestamps.npz",
    }
    for field_name in valid_arrays:
        assert np.array_equal(
            getattr(loaded, field_name),
            getattr(dataset, field_name),
        )

    with np.load(output_dir / "rgb.npz", allow_pickle=False) as archive:
        assert set(archive.files) == {"schema_version", "rgb"}
        assert archive["schema_version"].item() == 1
    with np.load(output_dir / "aligned_depth.npz", allow_pickle=False) as archive:
        assert set(archive.files) == {"schema_version", "aligned_depth"}
    with np.load(output_dir / "timestamps.npz", allow_pickle=False) as archive:
        assert set(archive.files) == {
            "schema_version",
            "rgb_timestamp_ns",
            "depth_timestamp_ns",
            "rgb_recorded_timestamp_ns",
            "depth_recorded_timestamp_ns",
        }


def test_save_refuses_to_overwrite_any_dataset_artifact(
    tmp_path: Path,
    valid_arrays: dict[str, np.ndarray],
) -> None:
    dataset = make_dataset(valid_arrays)
    output_dir = tmp_path / "dataset"
    output_dir.mkdir()
    existing_path = output_dir / "aligned_depth.npz"
    existing_path.write_bytes(b"existing")

    with pytest.raises(FileExistsError, match="already exist"):
        dataset.save(output_dir)

    assert existing_path.read_bytes() == b"existing"
    assert not (output_dir / "rgb.npz").exists()
    assert not (output_dir / "timestamps.npz").exists()


@pytest.mark.parametrize(
    ("field_name", "invalid_value", "expected_message"),
    [
        ("rgb", [[[[]]]], "rgb must be a numpy.ndarray"),
        ("aligned_depth", [[[1]]], "aligned_depth must be a numpy.ndarray"),
        ("rgb_timestamp_ns", [100], "rgb_timestamp_ns must be a numpy.ndarray"),
        ("depth_timestamp_ns", [100], "depth_timestamp_ns must be a numpy.ndarray"),
        (
            "rgb_recorded_timestamp_ns",
            [100],
            "rgb_recorded_timestamp_ns must be a numpy.ndarray",
        ),
        (
            "depth_recorded_timestamp_ns",
            [100],
            "depth_recorded_timestamp_ns must be a numpy.ndarray",
        ),
    ],
)
def test_rejects_non_array_fields(
    field_name: str,
    invalid_value: object,
    expected_message: str,
    valid_arrays: dict[str, np.ndarray],
) -> None:
    with pytest.raises(TypeError, match=expected_message):
        make_dataset(valid_arrays, **{field_name: invalid_value})


@pytest.mark.parametrize(
    ("invalid_rgb", "expected_message"),
    [
        (np.zeros((2, 3, 4), dtype=np.uint8), r"shape \(N, H, W, 3\)"),
        (np.zeros((2, 3, 4, 4), dtype=np.uint8), r"shape \(N, H, W, 3\)"),
        (np.zeros((2, 3, 4, 3), dtype=np.float32), "dtype uint8"),
        (np.zeros((0, 3, 4, 3), dtype=np.uint8), "at least one frame"),
        (np.zeros((2, 0, 4, 3), dtype=np.uint8), "must be positive"),
    ],
)
def test_rejects_invalid_rgb(
    invalid_rgb: np.ndarray,
    expected_message: str,
    valid_arrays: dict[str, np.ndarray],
) -> None:
    with pytest.raises(ValueError, match=expected_message):
        make_dataset(valid_arrays, rgb=invalid_rgb)


@pytest.mark.parametrize(
    ("invalid_depth", "expected_message"),
    [
        (np.zeros((2, 3), dtype=np.uint16), r"shape \(N, H, W\)"),
        (np.zeros((2, 3, 4), dtype=np.float32), "dtype uint16"),
        (np.zeros((0, 3, 4), dtype=np.uint16), "at least one frame"),
        (np.zeros((2, 3, 0), dtype=np.uint16), "must be positive"),
    ],
)
def test_rejects_invalid_aligned_depth(
    invalid_depth: np.ndarray,
    expected_message: str,
    valid_arrays: dict[str, np.ndarray],
) -> None:
    with pytest.raises(ValueError, match=expected_message):
        make_dataset(valid_arrays, aligned_depth=invalid_depth)


def test_rejects_incompatible_rgb_and_depth_pixel_grids(
    valid_arrays: dict[str, np.ndarray],
) -> None:
    with pytest.raises(ValueError, match="same pixel grid"):
        make_dataset(
            valid_arrays,
            aligned_depth=np.zeros((3, 2, 5), dtype=np.uint16),
        )


@pytest.mark.parametrize(
    ("field_name", "invalid_value", "expected_message"),
    [
        (
            "rgb_timestamp_ns",
            np.zeros((3, 1), dtype=np.int64),
            r"shape \(N,\)",
        ),
        (
            "depth_timestamp_ns",
            np.array([105, 205, 305], dtype=np.uint64),
            "dtype int64",
        ),
        (
            "rgb_recorded_timestamp_ns",
            np.array([130, 230], dtype=np.int64),
            "count does not match",
        ),
        (
            "depth_recorded_timestamp_ns",
            np.array([135, 0, 335], dtype=np.int64),
            "must be positive",
        ),
        (
            "rgb_timestamp_ns",
            np.array([100, 100, 300], dtype=np.int64),
            "strictly increasing",
        ),
        (
            "depth_timestamp_ns",
            np.array([105, 305, 205], dtype=np.int64),
            "strictly increasing",
        ),
    ],
)
def test_rejects_invalid_timestamps(
    field_name: str,
    invalid_value: np.ndarray,
    expected_message: str,
    valid_arrays: dict[str, np.ndarray],
) -> None:
    with pytest.raises(ValueError, match=expected_message):
        make_dataset(valid_arrays, **{field_name: invalid_value})


def test_load_rejects_missing_archive(
    tmp_path: Path,
    valid_arrays: dict[str, np.ndarray],
) -> None:
    dataset = make_dataset(valid_arrays)
    dataset.save(tmp_path)
    (tmp_path / "timestamps.npz").unlink()

    with pytest.raises(FileNotFoundError, match="timestamps.npz"):
        AlignmentDataset.load(tmp_path)


def test_load_rejects_missing_required_array(
    tmp_path: Path,
    valid_arrays: dict[str, np.ndarray],
) -> None:
    dataset = make_dataset(valid_arrays)
    dataset.save(tmp_path)
    np.savez(
        tmp_path / "rgb.npz",
        schema_version=np.asarray(1, dtype=np.int64),
    )

    with pytest.raises(ValueError, match="rgb"):
        AlignmentDataset.load(tmp_path)


def test_load_rejects_unsupported_schema_version(
    tmp_path: Path,
    valid_arrays: dict[str, np.ndarray],
) -> None:
    dataset = make_dataset(valid_arrays)
    dataset.save(tmp_path)
    np.savez(
        tmp_path / "rgb.npz",
        schema_version=np.asarray(2, dtype=np.int64),
        rgb=valid_arrays["rgb"],
    )

    with pytest.raises(ValueError, match="Unsupported.*schema version 2"):
        AlignmentDataset.load(tmp_path)
