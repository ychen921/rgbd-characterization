"""Tests for the atomic RGB/aligned-depth extraction workflow."""

from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path
from typing import Any

import numpy as np
import pytest
import yaml

from src.io.alignment_bag_reader import (
    AlignmentBagContract,
    CameraInfoSnapshot,
    ImageFrame,
    ImageStreamContract,
)
from src.io.alignment_dataset import AlignmentDataset
from tools.extract_alignment_dataset import (
    CAMERA_PARAMS_FILENAME,
    COLOR_CAMERA_INFO_FILENAME,
    DEPTH_CAMERA_INFO_FILENAME,
    EXPERIMENT_FILENAME,
    POST_RECORDING_FILENAME,
    PREFLIGHT_FILENAME,
    SUMMARY_FILENAME,
    extract_alignment_dataset,
    main,
)


COLOR_TOPIC = "/camera/color/image_raw"
DEPTH_TOPIC = "/camera/depth/image_raw"
COLOR_INFO_TOPIC = "/camera/color/camera_info"
DEPTH_INFO_TOPIC = "/camera/depth/camera_info"
FRAME_ID = "camera_color_optical_frame"
WIDTH = 1280
HEIGHT = 720


def _camera_info(*, depth: bool) -> CameraInfoSnapshot:
    return CameraInfoSnapshot(
        frame_id=FRAME_ID,
        width=WIDTH,
        height=HEIGHT,
        distortion_model="rational_polynomial",
        d=() if depth else (0.1, 0.2, 0.0, 0.0, 0.0),
        k=(100.0, 0.0, 640.0, 0.0, 100.0, 360.0, 0.0, 0.0, 1.0),
        r=(1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0),
        p=(
            100.0,
            0.0,
            640.0,
            0.0,
            0.0,
            100.0,
            360.0,
            0.0,
            0.0,
            0.0,
            1.0,
            0.0,
        ),
        binning_x=0,
        binning_y=0,
        roi=(0, 0, 0, 0, depth),
    )


def _bag_contract() -> AlignmentBagContract:
    return AlignmentBagContract(
        color=ImageStreamContract(
            topic=COLOR_TOPIC,
            message_count=2,
            width=WIDTH,
            height=HEIGHT,
            encoding="rgb8",
            frame_id=FRAME_ID,
        ),
        aligned_depth=ImageStreamContract(
            topic=DEPTH_TOPIC,
            message_count=1,
            width=WIDTH,
            height=HEIGHT,
            encoding="16UC1",
            frame_id=FRAME_ID,
        ),
        color_camera_info=_camera_info(depth=False),
        depth_camera_info=_camera_info(depth=True),
    )


class FakeReader:
    color_info_topic = COLOR_INFO_TOPIC
    depth_info_topic = DEPTH_INFO_TOPIC

    def __init__(
        self,
        bag_path: Path,
        *,
        color_topic: str,
        aligned_depth_topic: str,
    ) -> None:
        assert (bag_path / "metadata.yaml").is_file()
        assert color_topic == COLOR_TOPIC
        assert aligned_depth_topic == DEPTH_TOPIC
        self.contract = _bag_contract()

    def inspect(self) -> AlignmentBagContract:
        return self.contract

    def read_color_frames(self) -> Iterator[ImageFrame]:
        for index, timestamp_ns in enumerate((1_000_000_000, 1_100_000_000), 1):
            yield ImageFrame(
                stream="color",
                image=np.full((HEIGHT, WIDTH, 3), index, dtype=np.uint8),
                header_timestamp_ns=timestamp_ns,
                recorded_timestamp_ns=timestamp_ns + 10_000_000,
                frame_id=FRAME_ID,
            )

    def read_aligned_depth_frames(self) -> Iterator[ImageFrame]:
        timestamp_ns = 1_000_001_000
        yield ImageFrame(
            stream="aligned_depth",
            image=np.full((HEIGHT, WIDTH), 1000, dtype=np.uint16),
            header_timestamp_ns=timestamp_ns,
            recorded_timestamp_ns=timestamp_ns + 12_000_000,
            frame_id=FRAME_ID,
        )


class MissingDepthReader(FakeReader):
    def read_aligned_depth_frames(self) -> Iterator[ImageFrame]:
        return iter(())


def _documents(experiment_name: str) -> dict[str, dict[str, Any]]:
    return {
        EXPERIMENT_FILENAME: {
            "schema_version": 1,
            "experiment": {
                "name": experiment_name,
                "type": "rgb_depth_alignment",
                "scene": 5,
                "repeat": 1,
            },
            "registration": {
                "enabled": True,
                "aligned_depth_topic": DEPTH_TOPIC,
                "mode": "sdk",
                "wrapper_align_mode": "SW",
                "aligned_to": "color",
                "coordinate_convention": "color_pixel_grid",
            },
            "color": {
                "topic": COLOR_TOPIC,
                "width": WIDTH,
                "height": HEIGHT,
                "encoding": "rgb8",
            },
            "depth": {
                "topic": DEPTH_TOPIC,
                "width": WIDTH,
                "height": HEIGHT,
                "encoding": "16UC1",
                "precision": "1mm",
                "unit": "mm",
                "invalid_values": [0, 65535],
            },
            "timestamps": {
                "source": "message_header",
                "unit": "ns",
                "clock_domain": "global",
            },
        },
        CAMERA_PARAMS_FILENAME: {
            "/camera/camera": {
                "ros__parameters": {
                    "depth_registration": True,
                    "align_mode": "SW",
                    "align_target_stream": "COLOR",
                    "depth_precision": "1mm",
                }
            }
        },
        PREFLIGHT_FILENAME: {
            "schema_version": 1,
            "experiment": experiment_name,
            "overall_result": "warn",
            "checks": [],
            "warnings": ["optional topic unavailable"],
            "errors": [],
        },
        POST_RECORDING_FILENAME: {
            "schema_version": 1,
            "experiment": experiment_name,
            "status": "success",
            "recorded_topics": [
                COLOR_TOPIC,
                COLOR_INFO_TOPIC,
                DEPTH_TOPIC,
                DEPTH_INFO_TOPIC,
            ],
        },
    }


def _write_experiment(tmp_path: Path) -> tuple[Path, dict[str, str]]:
    source = tmp_path / "scene05_alignment_d100_center_yaw00_r01"
    source.mkdir()
    bag = source / "rosbag"
    bag.mkdir()
    (bag / "metadata.yaml").write_text("metadata", encoding="utf-8")

    serialized: dict[str, str] = {}
    for filename, document in _documents(source.name).items():
        content = yaml.safe_dump(document, sort_keys=False)
        (source / filename).write_text(content, encoding="utf-8")
        serialized[filename] = content
    return source, serialized


def _read_yaml(path: Path) -> dict[str, Any]:
    document = yaml.safe_load(path.read_text(encoding="utf-8"))
    assert isinstance(document, dict)
    return document


def test_extracts_unpaired_streams_and_traceability_artifacts(
    tmp_path: Path,
) -> None:
    source, serialized = _write_experiment(tmp_path)
    output = tmp_path / "data" / source.name

    result = extract_alignment_dataset(
        source,
        output,
        reader_factory=FakeReader,
    )

    assert result.rgb_frames == 2
    assert result.depth_frames == 1
    assert result.width == WIDTH
    assert result.height == HEIGHT
    dataset = AlignmentDataset.load(output)
    assert dataset.rgb.shape == (2, HEIGHT, WIDTH, 3)
    assert dataset.aligned_depth.shape == (1, HEIGHT, WIDTH)
    assert dataset.rgb[:, 0, 0, 0].tolist() == [1, 2]
    assert int(dataset.aligned_depth[0, 0, 0]) == 1000
    assert dataset.rgb_timestamp_ns.tolist() == [1_000_000_000, 1_100_000_000]
    assert dataset.depth_timestamp_ns.tolist() == [1_000_001_000]
    assert dataset.rgb_recorded_timestamp_ns.tolist() == [
        1_010_000_000,
        1_110_000_000,
    ]
    assert dataset.depth_recorded_timestamp_ns.tolist() == [1_012_001_000]

    for filename, original_content in serialized.items():
        assert (output / filename).read_text(encoding="utf-8") == original_content

    color_info = _read_yaml(output / COLOR_CAMERA_INFO_FILENAME)
    depth_info = _read_yaml(output / DEPTH_CAMERA_INFO_FILENAME)
    assert color_info["topic"] == COLOR_INFO_TOPIC
    assert color_info["d"] == [0.1, 0.2, 0.0, 0.0, 0.0]
    assert color_info["roi"]["do_rectify"] is False
    assert depth_info["topic"] == DEPTH_INFO_TOPIC
    assert depth_info["d"] == []
    assert depth_info["roi"]["do_rectify"] is True

    summary = _read_yaml(output / SUMMARY_FILENAME)
    assert summary["status"] == "success"
    assert summary["streams"]["color"]["frames"] == 2
    assert summary["streams"]["aligned_depth"]["frames"] == 1
    assert summary["streams"]["color"]["shape"] == [2, HEIGHT, WIDTH, 3]
    assert summary["streams"]["color"]["header_timestamp_ns"] == {
        "first": 1_000_000_000,
        "last": 1_100_000_000,
        "span": 100_000_000,
        "nominal_fps": 10.0,
    }
    assert "pair_rgb_index" not in yaml.safe_dump(summary)


@pytest.mark.parametrize(
    ("document_name", "path", "invalid_value", "expected_message"),
    [
        (
            EXPERIMENT_FILENAME,
            ("registration", "enabled"),
            False,
            "enabled",
        ),
        (
            EXPERIMENT_FILENAME,
            ("registration", "coordinate_convention"),
            "depth_pixel_grid",
            "coordinate_convention",
        ),
        (
            EXPERIMENT_FILENAME,
            ("registration", "wrapper_align_mode"),
            "HW",
            "align_mode",
        ),
        (
            EXPERIMENT_FILENAME,
            ("depth", "precision"),
            "0.1mm",
            "precision",
        ),
        (
            EXPERIMENT_FILENAME,
            ("color", "width"),
            640,
            "width",
        ),
        (PREFLIGHT_FILENAME, ("overall_result",), "fail", "overall_result"),
        (POST_RECORDING_FILENAME, ("status",), "failed", "status"),
        (
            POST_RECORDING_FILENAME,
            ("recorded_topics",),
            [COLOR_TOPIC, DEPTH_TOPIC],
            "missing required topic",
        ),
        (
            CAMERA_PARAMS_FILENAME,
            ("/camera/camera", "ros__parameters", "depth_registration"),
            False,
            "depth_registration",
        ),
        (
            CAMERA_PARAMS_FILENAME,
            ("/camera/camera", "ros__parameters", "depth_precision"),
            "0.1mm",
            "depth_precision",
        ),
    ],
)
def test_rejects_invalid_source_contract_before_opening_bag(
    tmp_path: Path,
    document_name: str,
    path: tuple[str, ...],
    invalid_value: Any,
    expected_message: str,
) -> None:
    source, _ = _write_experiment(tmp_path)
    document = _read_yaml(source / document_name)
    target: dict[str, Any] = document
    for key in path[:-1]:
        target = target[key]
    target[path[-1]] = invalid_value
    (source / document_name).write_text(
        yaml.safe_dump(document, sort_keys=False),
        encoding="utf-8",
    )
    reader_called = False

    def fail_if_called(**kwargs):
        nonlocal reader_called
        reader_called = True
        raise AssertionError(kwargs)

    with pytest.raises(ValueError, match=expected_message):
        extract_alignment_dataset(
            source,
            tmp_path / "data",
            reader_factory=fail_if_called,
        )
    assert reader_called is False


def test_rejects_experiment_directory_name_mismatch(tmp_path: Path) -> None:
    source, _ = _write_experiment(tmp_path)
    document = _read_yaml(source / EXPERIMENT_FILENAME)
    document["experiment"]["name"] = "different_experiment"
    (source / EXPERIMENT_FILENAME).write_text(
        yaml.safe_dump(document, sort_keys=False),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="does not match directory"):
        extract_alignment_dataset(
            source,
            tmp_path / "data",
            reader_factory=FakeReader,
        )


def test_rejects_missing_required_source_metadata(tmp_path: Path) -> None:
    source, _ = _write_experiment(tmp_path)
    (source / PREFLIGHT_FILENAME).unlink()

    with pytest.raises(FileNotFoundError, match="preflight.yaml"):
        extract_alignment_dataset(
            source,
            tmp_path / "data",
            reader_factory=FakeReader,
        )


def test_rejects_existing_output_without_modifying_it(tmp_path: Path) -> None:
    source, _ = _write_experiment(tmp_path)
    output = tmp_path / "data"
    output.mkdir()
    marker = output / "keep.txt"
    marker.write_text("keep", encoding="utf-8")

    with pytest.raises(FileExistsError, match="already exists"):
        extract_alignment_dataset(source, output, reader_factory=FakeReader)

    assert marker.read_text(encoding="utf-8") == "keep"


def test_removes_staging_directory_when_extraction_fails(tmp_path: Path) -> None:
    source, _ = _write_experiment(tmp_path)
    output_parent = tmp_path / "data"
    output = output_parent / source.name

    with pytest.raises(ValueError, match="returned 0 aligned-depth frames"):
        extract_alignment_dataset(
            source,
            output,
            reader_factory=MissingDepthReader,
        )

    assert not output.exists()
    assert list(output_parent.iterdir()) == []


def test_cli_reports_extraction_failure(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    missing = tmp_path / "missing"

    assert main([str(missing), str(tmp_path / "output")]) == 1
    assert "Experiment directory does not exist" in capsys.readouterr().err
