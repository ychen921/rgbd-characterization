"""Read validated RGB and aligned-depth streams from a ROS 2 bag."""

from __future__ import annotations

from collections.abc import Iterator, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol

import numpy as np
import yaml

from src.io.ros_image import decode_16uc1, decode_rgb8, frame_id, header_timestamp_ns


IMAGE_MESSAGE_TYPE = "sensor_msgs/msg/Image"
CAMERA_INFO_MESSAGE_TYPE = "sensor_msgs/msg/CameraInfo"


@dataclass(frozen=True)
class BagMessage:
    """One deserialized bag message and its rosbag storage timestamp."""

    topic: str
    message: Any
    recorded_timestamp_ns: int


class BagBackend(Protocol):
    """Minimal bag access interface used by ``AlignmentBagReader``."""

    def topic_types(self) -> Mapping[str, str]: ...

    def message_counts(self) -> Mapping[str, int]: ...

    def iter_messages(self, topics: Sequence[str]) -> Iterator[BagMessage]: ...


@dataclass(frozen=True)
class ImageFrame:
    """One decoded image with both timestamp sources preserved."""

    stream: str
    image: np.ndarray
    header_timestamp_ns: int
    recorded_timestamp_ns: int
    frame_id: str


@dataclass(frozen=True)
class ImageStreamContract:
    """Stable image properties expected throughout one bag stream."""

    topic: str
    message_count: int
    width: int
    height: int
    encoding: str
    frame_id: str


@dataclass(frozen=True)
class CameraInfoSnapshot:
    """Timestamp-independent CameraInfo calibration payload."""

    frame_id: str
    width: int
    height: int
    distortion_model: str
    d: tuple[float, ...]
    k: tuple[float, ...]
    r: tuple[float, ...]
    p: tuple[float, ...]
    binning_x: int
    binning_y: int
    roi: tuple[int, int, int, int, bool]


@dataclass(frozen=True)
class AlignmentBagContract:
    """Validated RGB, aligned-depth, and CameraInfo bag contract."""

    color: ImageStreamContract
    aligned_depth: ImageStreamContract
    color_camera_info: CameraInfoSnapshot
    depth_camera_info: CameraInfoSnapshot


class AlignmentBagReader:
    """Validate and stream unpaired RGB and SDK-aligned depth frames."""

    DEFAULT_COLOR_TOPIC = "/camera/color/image_raw"
    DEFAULT_ALIGNED_DEPTH_TOPIC = "/camera/depth/image_raw"
    DEFAULT_COLOR_INFO_TOPIC = "/camera/color/camera_info"
    DEFAULT_DEPTH_INFO_TOPIC = "/camera/depth/camera_info"

    def __init__(
        self,
        bag_path: Path,
        *,
        color_topic: str = DEFAULT_COLOR_TOPIC,
        aligned_depth_topic: str = DEFAULT_ALIGNED_DEPTH_TOPIC,
        color_info_topic: str = DEFAULT_COLOR_INFO_TOPIC,
        depth_info_topic: str = DEFAULT_DEPTH_INFO_TOPIC,
        backend: BagBackend | None = None,
    ) -> None:
        self.bag_path = Path(bag_path).expanduser()
        self.color_topic = color_topic
        self.aligned_depth_topic = aligned_depth_topic
        self.color_info_topic = color_info_topic
        self.depth_info_topic = depth_info_topic
        self._validate_configuration()

        self._backend: BagBackend = backend or _Rosbag2Backend(self.bag_path)
        self._topic_types: dict[str, str] | None = None
        self._message_counts: dict[str, int] | None = None
        self._camera_info: tuple[CameraInfoSnapshot, CameraInfoSnapshot] | None = None
        self._contract: AlignmentBagContract | None = None

    def inspect(self) -> AlignmentBagContract:
        """Inspect and cache the complete color-pixel-grid bag contract."""
        if self._contract is not None:
            return self._contract

        self._validate_topic_inventory()
        color_contract, depth_contract = self._read_first_image_contracts()
        color_info, depth_info = self.read_camera_info()

        self._validate_image_pair(color_contract, depth_contract)
        self._validate_image_camera_info(color_contract, color_info, "color")
        self._validate_image_camera_info(
            depth_contract,
            depth_info,
            "aligned depth",
        )
        self._validate_camera_info_pair(color_info, depth_info)

        self._contract = AlignmentBagContract(
            color=color_contract,
            aligned_depth=depth_contract,
            color_camera_info=color_info,
            depth_camera_info=depth_info,
        )
        return self._contract

    def read_color_frames(self) -> Iterator[ImageFrame]:
        """Yield validated RGB frames without pairing them to depth."""
        contract = self.inspect().color
        yield from self._iter_frames(
            stream="color",
            contract=contract,
            decoder=decode_rgb8,
        )

    def read_aligned_depth_frames(self) -> Iterator[ImageFrame]:
        """Yield validated aligned-depth frames without pairing them to RGB."""
        contract = self.inspect().aligned_depth
        yield from self._iter_frames(
            stream="aligned_depth",
            contract=contract,
            decoder=decode_16uc1,
        )

    def read_camera_info(
        self,
    ) -> tuple[CameraInfoSnapshot, CameraInfoSnapshot]:
        """Return stable color and aligned-depth CameraInfo snapshots."""
        if self._camera_info is not None:
            return self._camera_info

        self._validate_topic_inventory()
        topics = (self.color_info_topic, self.depth_info_topic)
        first_snapshots: dict[str, CameraInfoSnapshot] = {}
        counts = {topic: 0 for topic in topics}
        previous_header_ns: dict[str, int | None] = {topic: None for topic in topics}
        previous_recorded_ns: dict[str, int | None] = {topic: None for topic in topics}

        for bag_message in self._backend.iter_messages(topics):
            topic = bag_message.topic
            if topic not in counts:
                continue
            message = bag_message.message
            timestamp_ns = header_timestamp_ns(message)
            recorded_ns = self._validate_recorded_timestamp(
                bag_message.recorded_timestamp_ns
            )
            self._validate_increasing_timestamp(
                topic=topic,
                source="header",
                current=timestamp_ns,
                previous=previous_header_ns[topic],
            )
            self._validate_increasing_timestamp(
                topic=topic,
                source="recorded",
                current=recorded_ns,
                previous=previous_recorded_ns[topic],
            )
            previous_header_ns[topic] = timestamp_ns
            previous_recorded_ns[topic] = recorded_ns

            snapshot = self._camera_info_snapshot(message)
            if topic not in first_snapshots:
                first_snapshots[topic] = snapshot
            elif snapshot != first_snapshots[topic]:
                raise ValueError(f"CameraInfo payload changed within topic {topic!r}")
            counts[topic] += 1

        for topic in topics:
            expected_count = self._required_message_count(topic)
            if counts[topic] != expected_count:
                raise ValueError(
                    f"CameraInfo message count mismatch for {topic!r}; "
                    f"read {counts[topic]}, metadata declares {expected_count}"
                )
            if topic not in first_snapshots:
                raise ValueError(f"No CameraInfo messages were read from {topic!r}")

        self._camera_info = (
            first_snapshots[self.color_info_topic],
            first_snapshots[self.depth_info_topic],
        )
        return self._camera_info

    def _iter_frames(
        self,
        *,
        stream: str,
        contract: ImageStreamContract,
        decoder: Any,
    ) -> Iterator[ImageFrame]:
        count = 0
        previous_header_ns: int | None = None
        previous_recorded_ns: int | None = None

        for bag_message in self._backend.iter_messages((contract.topic,)):
            if bag_message.topic != contract.topic:
                continue
            message = bag_message.message
            image = decoder(message)
            current_frame_id = frame_id(message)
            self._validate_frame_contract(message, current_frame_id, contract)

            timestamp_ns = header_timestamp_ns(message)
            recorded_ns = self._validate_recorded_timestamp(
                bag_message.recorded_timestamp_ns
            )
            self._validate_increasing_timestamp(
                topic=contract.topic,
                source="header",
                current=timestamp_ns,
                previous=previous_header_ns,
            )
            self._validate_increasing_timestamp(
                topic=contract.topic,
                source="recorded",
                current=recorded_ns,
                previous=previous_recorded_ns,
            )
            previous_header_ns = timestamp_ns
            previous_recorded_ns = recorded_ns
            count += 1

            yield ImageFrame(
                stream=stream,
                image=image,
                header_timestamp_ns=timestamp_ns,
                recorded_timestamp_ns=recorded_ns,
                frame_id=current_frame_id,
            )

        if count != contract.message_count:
            raise ValueError(
                f"Image message count mismatch for {contract.topic!r}; "
                f"read {count}, metadata declares {contract.message_count}"
            )

    def _read_first_image_contracts(
        self,
    ) -> tuple[ImageStreamContract, ImageStreamContract]:
        topics = (self.color_topic, self.aligned_depth_topic)
        contracts: dict[str, ImageStreamContract] = {}
        for bag_message in self._backend.iter_messages(topics):
            topic = bag_message.topic
            if topic in contracts or topic not in topics:
                continue
            message = bag_message.message
            if topic == self.color_topic:
                decode_rgb8(message)
            else:
                decode_16uc1(message)
            timestamp_ns = header_timestamp_ns(message)
            self._validate_recorded_timestamp(bag_message.recorded_timestamp_ns)
            if timestamp_ns <= 0:
                raise ValueError(f"Invalid header timestamp on {topic!r}")

            contracts[topic] = ImageStreamContract(
                topic=topic,
                message_count=self._required_message_count(topic),
                width=self._require_positive_integer(message.width, "Image width"),
                height=self._require_positive_integer(message.height, "Image height"),
                encoding=message.encoding,
                frame_id=frame_id(message),
            )
            if len(contracts) == 2:
                break

        for topic in topics:
            if topic not in contracts:
                raise ValueError(f"No image messages were read from {topic!r}")
        return contracts[self.color_topic], contracts[self.aligned_depth_topic]

    def _validate_topic_inventory(self) -> None:
        if self._topic_types is not None and self._message_counts is not None:
            return

        try:
            topic_types = dict(self._backend.topic_types())
            message_counts = dict(self._backend.message_counts())
        except Exception as exc:
            raise RuntimeError(
                f"Failed to inspect ROS 2 bag metadata: {self.bag_path}"
            ) from exc

        expected_types = {
            self.color_topic: IMAGE_MESSAGE_TYPE,
            self.aligned_depth_topic: IMAGE_MESSAGE_TYPE,
            self.color_info_topic: CAMERA_INFO_MESSAGE_TYPE,
            self.depth_info_topic: CAMERA_INFO_MESSAGE_TYPE,
        }
        for topic, expected_type in expected_types.items():
            if topic not in topic_types:
                available = ", ".join(sorted(topic_types)) or "<none>"
                raise ValueError(
                    f"Required topic {topic!r} was not found in {self.bag_path}. "
                    f"Available topics: {available}"
                )
            actual_type = topic_types[topic]
            if actual_type != expected_type:
                raise ValueError(
                    f"Topic {topic!r} has type {actual_type!r}; expected "
                    f"{expected_type!r}"
                )
            count = message_counts.get(topic, 0)
            if not isinstance(count, int) or isinstance(count, bool) or count <= 0:
                raise ValueError(
                    f"Required topic {topic!r} must contain at least one message; "
                    f"metadata count is {count!r}"
                )

        self._topic_types = topic_types
        self._message_counts = message_counts

    def _required_message_count(self, topic: str) -> int:
        self._validate_topic_inventory()
        assert self._message_counts is not None
        return self._message_counts[topic]

    @staticmethod
    def _validate_frame_contract(
        message: Any,
        current_frame_id: str,
        contract: ImageStreamContract,
    ) -> None:
        current = (
            message.width,
            message.height,
            message.encoding,
            current_frame_id,
        )
        expected = (
            contract.width,
            contract.height,
            contract.encoding,
            contract.frame_id,
        )
        if current != expected:
            raise ValueError(
                f"Image contract changed within topic {contract.topic!r}; "
                f"got {current}, expected {expected}"
            )

    @staticmethod
    def _validate_image_pair(
        color: ImageStreamContract,
        depth: ImageStreamContract,
    ) -> None:
        if color.width != depth.width or color.height != depth.height:
            raise ValueError(
                "RGB and aligned depth do not share the same pixel grid; got "
                f"RGB {color.width}x{color.height} and aligned depth "
                f"{depth.width}x{depth.height}"
            )
        if color.frame_id != depth.frame_id:
            raise ValueError(
                "RGB and aligned depth frame IDs do not match; got "
                f"{color.frame_id!r} and {depth.frame_id!r}"
            )

    @staticmethod
    def _validate_image_camera_info(
        image: ImageStreamContract,
        info: CameraInfoSnapshot,
        label: str,
    ) -> None:
        if image.width != info.width or image.height != info.height:
            raise ValueError(
                f"{label} CameraInfo dimensions do not match its image stream"
            )
        if image.frame_id != info.frame_id:
            raise ValueError(
                f"{label} CameraInfo frame ID does not match its image stream"
            )

    @staticmethod
    def _validate_camera_info_pair(
        color: CameraInfoSnapshot,
        depth: CameraInfoSnapshot,
    ) -> None:
        color_projection = (
            color.width,
            color.height,
            color.frame_id,
            color.k,
            color.r,
            color.p,
            color.binning_x,
            color.binning_y,
        )
        depth_projection = (
            depth.width,
            depth.height,
            depth.frame_id,
            depth.k,
            depth.r,
            depth.p,
            depth.binning_x,
            depth.binning_y,
        )
        if color_projection != depth_projection:
            raise ValueError(
                "Color and aligned-depth CameraInfo projection contracts do "
                "not match"
            )

    @classmethod
    def _camera_info_snapshot(cls, message: Any) -> CameraInfoSnapshot:
        width = cls._require_positive_integer(message.width, "CameraInfo width")
        height = cls._require_positive_integer(message.height, "CameraInfo height")
        distortion_model = message.distortion_model
        if not isinstance(distortion_model, str) or not distortion_model:
            raise ValueError("CameraInfo distortion_model must be a non-empty string")

        k = cls._float_tuple(message.k, "CameraInfo K", expected_length=9)
        r = cls._float_tuple(message.r, "CameraInfo R", expected_length=9)
        p = cls._float_tuple(message.p, "CameraInfo P", expected_length=12)
        d = cls._float_tuple(message.d, "CameraInfo D")

        binning_x = cls._require_non_negative_integer(
            message.binning_x,
            "CameraInfo binning_x",
        )
        binning_y = cls._require_non_negative_integer(
            message.binning_y,
            "CameraInfo binning_y",
        )
        try:
            roi = message.roi
            roi_tuple = (
                cls._require_non_negative_integer(roi.x_offset, "ROI x_offset"),
                cls._require_non_negative_integer(roi.y_offset, "ROI y_offset"),
                cls._require_non_negative_integer(roi.width, "ROI width"),
                cls._require_non_negative_integer(roi.height, "ROI height"),
                roi.do_rectify,
            )
        except AttributeError as exc:
            raise ValueError("CameraInfo is missing roi fields") from exc
        if not isinstance(roi_tuple[4], bool):
            raise ValueError("ROI do_rectify must be a boolean")

        return CameraInfoSnapshot(
            frame_id=frame_id(message),
            width=width,
            height=height,
            distortion_model=distortion_model,
            d=d,
            k=k,
            r=r,
            p=p,
            binning_x=binning_x,
            binning_y=binning_y,
            roi=roi_tuple,
        )

    @staticmethod
    def _float_tuple(
        values: Any,
        field_name: str,
        *,
        expected_length: int | None = None,
    ) -> tuple[float, ...]:
        try:
            result = tuple(float(value) for value in values)
        except (TypeError, ValueError) as exc:
            raise ValueError(f"{field_name} must be a numeric sequence") from exc
        if expected_length is not None and len(result) != expected_length:
            raise ValueError(
                f"{field_name} must contain {expected_length} values; got "
                f"{len(result)}"
            )
        return result

    @staticmethod
    def _validate_recorded_timestamp(value: Any) -> int:
        if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
            raise ValueError("Rosbag recorded timestamp must be a positive integer")
        return value

    @staticmethod
    def _validate_increasing_timestamp(
        *,
        topic: str,
        source: str,
        current: int,
        previous: int | None,
    ) -> None:
        if previous is not None and current <= previous:
            raise ValueError(
                f"{source.capitalize()} timestamps must be strictly increasing "
                f"within topic {topic!r}; got {current} after {previous}"
            )

    @staticmethod
    def _require_positive_integer(value: Any, field_name: str) -> int:
        if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
            raise ValueError(f"{field_name} must be a positive integer")
        return value

    @staticmethod
    def _require_non_negative_integer(value: Any, field_name: str) -> int:
        if not isinstance(value, int) or isinstance(value, bool) or value < 0:
            raise ValueError(f"{field_name} must be a non-negative integer")
        return value

    def _validate_configuration(self) -> None:
        if not self.bag_path.exists():
            raise FileNotFoundError(f"Bag path does not exist: {self.bag_path}")
        if not self.bag_path.is_dir():
            raise NotADirectoryError(f"Bag path is not a directory: {self.bag_path}")
        if not (self.bag_path / "metadata.yaml").is_file():
            raise ValueError(
                f"Bag path does not contain metadata.yaml: {self.bag_path}"
            )

        topics = (
            self.color_topic,
            self.aligned_depth_topic,
            self.color_info_topic,
            self.depth_info_topic,
        )
        for topic in topics:
            if not isinstance(topic, str) or not topic.startswith("/"):
                raise ValueError(
                    f"Configured topics must be absolute ROS topic names: {topic!r}"
                )
        if len(set(topics)) != len(topics):
            raise ValueError("Configured alignment topics must be unique")


class _Rosbag2Backend:
    """Lazy ROS 2 backend so importing this module does not load ROS binaries."""

    def __init__(self, bag_path: Path) -> None:
        self.bag_path = bag_path
        self._metadata: Mapping[str, Any] | None = None
        self._topic_types: dict[str, str] | None = None

    def topic_types(self) -> Mapping[str, str]:
        if self._topic_types is not None:
            return self._topic_types
        reader, _ = self._open_reader()
        self._topic_types = {
            topic.name: topic.type for topic in reader.get_all_topics_and_types()
        }
        return self._topic_types

    def message_counts(self) -> Mapping[str, int]:
        info = self._bag_information()
        counts: dict[str, int] = {}
        entries = info.get("topics_with_message_count")
        if not isinstance(entries, list):
            raise ValueError("Rosbag metadata is missing topics_with_message_count")
        for entry in entries:
            if not isinstance(entry, Mapping):
                raise ValueError("Invalid topic count entry in rosbag metadata")
            topic_metadata = entry.get("topic_metadata")
            if not isinstance(topic_metadata, Mapping):
                raise ValueError("Topic count entry is missing topic_metadata")
            name = topic_metadata.get("name")
            count = entry.get("message_count")
            if not isinstance(name, str) or not isinstance(count, int):
                raise ValueError("Invalid topic name or count in rosbag metadata")
            counts[name] = count
        return counts

    def iter_messages(self, topics: Sequence[str]) -> Iterator[BagMessage]:
        topic_types = self.topic_types()
        reader, rosbag2_py = self._open_reader()
        deserialize_message, message_classes = self._message_dependencies()
        reader.set_filter(rosbag2_py.StorageFilter(topics=list(topics)))

        while reader.has_next():
            topic, serialized_data, recorded_timestamp_ns = reader.read_next()
            if topic not in topics:
                continue
            message_type = topic_types.get(topic)
            message_class = message_classes.get(message_type)
            if message_class is None:
                raise ValueError(
                    f"Unsupported message type {message_type!r} on topic {topic!r}"
                )
            try:
                message = deserialize_message(serialized_data, message_class)
            except Exception as exc:
                raise RuntimeError(
                    f"Failed to deserialize message from {topic!r} at rosbag "
                    f"timestamp {recorded_timestamp_ns}"
                ) from exc
            yield BagMessage(
                topic=topic,
                message=message,
                recorded_timestamp_ns=int(recorded_timestamp_ns),
            )

    def _open_reader(self) -> tuple[Any, Any]:
        try:
            import rosbag2_py
        except (ImportError, ModuleNotFoundError) as exc:
            raise RuntimeError(
                "rosbag2_py could not be loaded. Run alignment bag access with "
                "the ROS Humble Python environment (for example /usr/bin/python3)."
            ) from exc

        info = self._bag_information()
        storage_id = info.get("storage_identifier")
        if not isinstance(storage_id, str) or not storage_id:
            raise ValueError("Rosbag metadata has no storage_identifier")

        reader = rosbag2_py.SequentialReader()
        storage_options = rosbag2_py.StorageOptions(
            uri=str(self.bag_path),
            storage_id=storage_id,
        )
        converter_options = rosbag2_py.ConverterOptions(
            input_serialization_format="cdr",
            output_serialization_format="cdr",
        )
        try:
            reader.open(storage_options, converter_options)
        except Exception as exc:
            raise RuntimeError(f"Failed to open ROS 2 bag: {self.bag_path}") from exc
        return reader, rosbag2_py

    @staticmethod
    def _message_dependencies() -> tuple[Any, Mapping[str, Any]]:
        try:
            from rclpy.serialization import deserialize_message
            from sensor_msgs.msg import CameraInfo, Image
        except (ImportError, ModuleNotFoundError) as exc:
            raise RuntimeError(
                "ROS message modules could not be loaded in this Python environment"
            ) from exc
        return deserialize_message, {
            IMAGE_MESSAGE_TYPE: Image,
            CAMERA_INFO_MESSAGE_TYPE: CameraInfo,
        }

    def _bag_information(self) -> Mapping[str, Any]:
        if self._metadata is not None:
            return self._metadata
        metadata_path = self.bag_path / "metadata.yaml"
        try:
            with metadata_path.open("r", encoding="utf-8") as stream:
                document = yaml.safe_load(stream)
        except yaml.YAMLError as exc:
            raise ValueError(f"Invalid rosbag metadata YAML: {metadata_path}") from exc
        if not isinstance(document, Mapping):
            raise ValueError(f"Rosbag metadata must be a mapping: {metadata_path}")
        info = document.get("rosbag2_bagfile_information")
        if not isinstance(info, Mapping):
            raise ValueError(
                f"Rosbag metadata is missing rosbag2_bagfile_information: "
                f"{metadata_path}"
            )
        self._metadata = info
        return info
