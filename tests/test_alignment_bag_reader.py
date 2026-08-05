"""Tests for alignment bag validation using a ROS-independent fake backend."""

from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from src.io.alignment_bag_reader import (
    CAMERA_INFO_MESSAGE_TYPE,
    IMAGE_MESSAGE_TYPE,
    AlignmentBagReader,
    BagMessage,
)


COLOR = "/camera/color/image_raw"
DEPTH = "/camera/depth/image_raw"
COLOR_INFO = "/camera/color/camera_info"
DEPTH_INFO = "/camera/depth/camera_info"
COLOR_FRAME = "camera_color_optical_frame"


class FakeBackend:
    def __init__(
        self,
        messages: list[BagMessage],
        *,
        topic_types: dict[str, str] | None = None,
        message_counts: dict[str, int] | None = None,
    ) -> None:
        self.messages = messages
        self._topic_types = topic_types or {
            COLOR: IMAGE_MESSAGE_TYPE,
            DEPTH: IMAGE_MESSAGE_TYPE,
            COLOR_INFO: CAMERA_INFO_MESSAGE_TYPE,
            DEPTH_INFO: CAMERA_INFO_MESSAGE_TYPE,
        }
        if message_counts is None:
            message_counts = {
                topic: sum(message.topic == topic for message in messages)
                for topic in self._topic_types
            }
        self._message_counts = message_counts

    def topic_types(self) -> dict[str, str]:
        return dict(self._topic_types)

    def message_counts(self) -> dict[str, int]:
        return dict(self._message_counts)

    def iter_messages(self, topics) -> object:
        selected = set(topics)
        return iter(message for message in self.messages if message.topic in selected)


def make_header(timestamp_ns: int, message_frame_id: str = COLOR_FRAME):
    return SimpleNamespace(
        stamp=SimpleNamespace(
            sec=timestamp_ns // 1_000_000_000,
            nanosec=timestamp_ns % 1_000_000_000,
        ),
        frame_id=message_frame_id,
    )


def make_rgb(timestamp_ns: int, *, width: int = 2, frame: str = COLOR_FRAME):
    height = 1
    values = np.arange(height * width * 3, dtype=np.uint8)
    return SimpleNamespace(
        header=make_header(timestamp_ns, frame),
        height=height,
        width=width,
        encoding="rgb8",
        is_bigendian=0,
        step=width * 3,
        data=values.tobytes(),
    )


def make_depth(timestamp_ns: int, *, width: int = 2, frame: str = COLOR_FRAME):
    height = 1
    values = np.arange(1, height * width + 1, dtype=np.uint16)
    return SimpleNamespace(
        header=make_header(timestamp_ns, frame),
        height=height,
        width=width,
        encoding="16UC1",
        is_bigendian=0,
        step=width * 2,
        data=values.astype("<u2").tobytes(),
    )


def make_camera_info(
    timestamp_ns: int,
    *,
    width: int = 2,
    frame: str = COLOR_FRAME,
    fx: float = 100.0,
):
    return SimpleNamespace(
        header=make_header(timestamp_ns, frame),
        width=width,
        height=1,
        distortion_model="rational_polynomial",
        d=[0.0] * 8,
        k=[fx, 0.0, 1.0, 0.0, fx, 0.5, 0.0, 0.0, 1.0],
        r=[1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0],
        p=[fx, 0.0, 1.0, 0.0, 0.0, fx, 0.5, 0.0, 0.0, 0.0, 1.0, 0.0],
        binning_x=0,
        binning_y=0,
        roi=SimpleNamespace(
            x_offset=0,
            y_offset=0,
            width=0,
            height=0,
            do_rectify=False,
        ),
    )


def default_messages() -> list[BagMessage]:
    messages: list[BagMessage] = []
    for index, timestamp_ns in enumerate((1_000_000_000, 1_100_000_000)):
        base_recorded = timestamp_ns + 10_000_000
        messages.extend(
            [
                BagMessage(COLOR, make_rgb(timestamp_ns), base_recorded),
                BagMessage(
                    DEPTH, make_depth(timestamp_ns + 1_000), base_recorded + 1_000
                ),
                BagMessage(
                    COLOR_INFO,
                    make_camera_info(timestamp_ns),
                    base_recorded + 2_000,
                ),
                BagMessage(
                    DEPTH_INFO,
                    make_camera_info(timestamp_ns + 1_000),
                    base_recorded + 3_000,
                ),
            ]
        )
    return messages


@pytest.fixture
def bag_path(tmp_path: Path) -> Path:
    path = tmp_path / "rosbag"
    path.mkdir()
    (path / "metadata.yaml").write_text("metadata", encoding="utf-8")
    return path


def make_reader(
    bag_path: Path,
    messages: list[BagMessage] | None = None,
    **backend_options,
) -> AlignmentBagReader:
    backend = FakeBackend(messages or default_messages(), **backend_options)
    return AlignmentBagReader(bag_path, backend=backend)


def test_inspects_and_streams_valid_alignment_bag(bag_path: Path) -> None:
    reader = make_reader(bag_path)

    contract = reader.inspect()
    color_frames = list(reader.read_color_frames())
    depth_frames = list(reader.read_aligned_depth_frames())

    assert contract.color.message_count == 2
    assert contract.aligned_depth.message_count == 2
    assert contract.color.width == contract.aligned_depth.width == 2
    assert contract.color.height == contract.aligned_depth.height == 1
    assert contract.color.frame_id == contract.aligned_depth.frame_id == COLOR_FRAME
    assert contract.color_camera_info == contract.depth_camera_info
    assert [frame.header_timestamp_ns for frame in color_frames] == [
        1_000_000_000,
        1_100_000_000,
    ]
    assert [frame.recorded_timestamp_ns for frame in depth_frames] == [
        1_010_001_000,
        1_110_001_000,
    ]
    assert color_frames[0].stream == "color"
    assert color_frames[0].image.shape == (1, 2, 3)
    assert depth_frames[0].stream == "aligned_depth"
    assert depth_frames[0].image.shape == (1, 2)


def test_allows_different_color_and_depth_frame_counts(bag_path: Path) -> None:
    messages = [
        message
        for message in default_messages()
        if not (
            message.topic == DEPTH
            and message.message.header.stamp.nanosec == 100_001_000
        )
    ]
    reader = make_reader(bag_path, messages)

    assert len(list(reader.read_color_frames())) == 2
    assert len(list(reader.read_aligned_depth_frames())) == 1


@pytest.mark.parametrize(
    ("topic", "actual_type", "expected_message"),
    [
        (COLOR, None, "Required topic"),
        (DEPTH, "std_msgs/msg/String", "has type"),
    ],
)
def test_rejects_missing_or_wrong_topic_type(
    bag_path: Path,
    topic: str,
    actual_type: str | None,
    expected_message: str,
) -> None:
    topic_types = FakeBackend(default_messages()).topic_types()
    if actual_type is None:
        del topic_types[topic]
    else:
        topic_types[topic] = actual_type
    reader = make_reader(bag_path, topic_types=topic_types)

    with pytest.raises(ValueError, match=expected_message):
        reader.inspect()


def test_rejects_zero_message_required_topic(bag_path: Path) -> None:
    counts = FakeBackend(default_messages()).message_counts()
    counts[DEPTH] = 0
    reader = make_reader(bag_path, message_counts=counts)

    with pytest.raises(ValueError, match="at least one message"):
        reader.inspect()


@pytest.mark.parametrize(
    ("changed_topic", "changed_message", "expected_message"),
    [
        (COLOR, make_rgb(1_100_000_000, width=3), "contract changed"),
        (
            DEPTH,
            make_depth(1_100_001_000, frame="camera_depth_optical_frame"),
            "contract changed",
        ),
    ],
)
def test_rejects_image_contract_change_within_stream(
    bag_path: Path,
    changed_topic: str,
    changed_message,
    expected_message: str,
) -> None:
    messages = default_messages()
    matching_indices = [
        index
        for index, message in enumerate(messages)
        if message.topic == changed_topic
    ]
    target_index = matching_indices[1]
    messages[target_index] = replace(
        messages[target_index],
        message=changed_message,
    )
    reader = make_reader(bag_path, messages)

    with pytest.raises(ValueError, match=expected_message):
        if changed_topic == COLOR:
            list(reader.read_color_frames())
        else:
            list(reader.read_aligned_depth_frames())


def test_rejects_incompatible_color_and_depth_pixel_grids(bag_path: Path) -> None:
    messages = [
        (
            replace(message, message=make_depth(1_000_001_000, width=3))
            if message.topic == DEPTH
            else message
        )
        for message in default_messages()
    ]
    reader = make_reader(bag_path, messages)

    with pytest.raises(ValueError, match="same pixel grid"):
        reader.inspect()


def test_rejects_non_increasing_header_timestamps(bag_path: Path) -> None:
    messages = default_messages()
    color_indices = [
        index for index, message in enumerate(messages) if message.topic == COLOR
    ]
    messages[color_indices[1]] = replace(
        messages[color_indices[1]],
        message=make_rgb(1_000_000_000),
    )
    reader = make_reader(bag_path, messages)

    with pytest.raises(ValueError, match="Header timestamps.*strictly increasing"):
        list(reader.read_color_frames())


def test_rejects_non_increasing_recorded_timestamps(bag_path: Path) -> None:
    messages = default_messages()
    depth_indices = [
        index for index, message in enumerate(messages) if message.topic == DEPTH
    ]
    messages[depth_indices[1]] = replace(
        messages[depth_indices[1]],
        recorded_timestamp_ns=messages[depth_indices[0]].recorded_timestamp_ns,
    )
    reader = make_reader(bag_path, messages)

    with pytest.raises(ValueError, match="Recorded timestamps.*strictly increasing"):
        list(reader.read_aligned_depth_frames())


def test_rejects_camera_info_change_within_topic(bag_path: Path) -> None:
    messages = default_messages()
    color_info_indices = [
        index for index, message in enumerate(messages) if message.topic == COLOR_INFO
    ]
    messages[color_info_indices[1]] = replace(
        messages[color_info_indices[1]],
        message=make_camera_info(1_100_000_000, fx=101.0),
    )
    reader = make_reader(bag_path, messages)

    with pytest.raises(ValueError, match="CameraInfo payload changed"):
        reader.inspect()


def test_rejects_color_and_depth_camera_info_mismatch(bag_path: Path) -> None:
    messages = [
        (
            replace(
                message,
                message=make_camera_info(
                    message.message.header.stamp.sec * 1_000_000_000
                    + message.message.header.stamp.nanosec,
                    fx=102.0,
                ),
            )
            if message.topic == DEPTH_INFO
            else message
        )
        for message in default_messages()
    ]
    reader = make_reader(bag_path, messages)

    with pytest.raises(ValueError, match="projection contracts do not match"):
        reader.inspect()


def test_allows_modality_specific_distortion_and_rectification(
    bag_path: Path,
) -> None:
    messages = []
    for message in default_messages():
        if message.topic not in (COLOR_INFO, DEPTH_INFO):
            messages.append(message)
            continue
        updated = make_camera_info(
            message.message.header.stamp.sec * 1_000_000_000
            + message.message.header.stamp.nanosec
        )
        if message.topic == COLOR_INFO:
            updated.d = [0.1] + [0.0] * 7
        else:
            updated.roi.do_rectify = True
        messages.append(replace(message, message=updated))

    contract = make_reader(bag_path, messages).inspect()

    assert contract.color_camera_info.d[0] == 0.1
    assert contract.color_camera_info.d != contract.depth_camera_info.d
    assert contract.depth_camera_info.roi[-1] is True


def test_rejects_metadata_and_stream_count_mismatch(bag_path: Path) -> None:
    counts = FakeBackend(default_messages()).message_counts()
    counts[COLOR] = 3
    reader = make_reader(bag_path, message_counts=counts)

    with pytest.raises(ValueError, match="message count mismatch"):
        list(reader.read_color_frames())


def test_rejects_duplicate_or_relative_configured_topics(bag_path: Path) -> None:
    with pytest.raises(ValueError, match="must be unique"):
        AlignmentBagReader(
            bag_path,
            color_topic=COLOR,
            aligned_depth_topic=COLOR,
            backend=FakeBackend(default_messages()),
        )

    with pytest.raises(ValueError, match="absolute ROS topic"):
        AlignmentBagReader(
            bag_path,
            color_topic="camera/color/image_raw",
            backend=FakeBackend(default_messages()),
        )
