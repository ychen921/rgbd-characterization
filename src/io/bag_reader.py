"""ROS 2 bag access for raw depth images."""

from collections.abc import Iterator
from pathlib import Path

import numpy as np
import rosbag2_py
from rclpy.serialization import deserialize_message
from sensor_msgs.msg import Image


class DepthBagReader:
    """Read raw uint16 depth frames from a ROS 2 bag."""

    _DEPTH_MESSAGE_TYPE = "sensor_msgs/msg/Image"
    _DEPTH_ENCODING = "16UC1"
    _BYTES_PER_PIXEL = 2
    _STORAGE_ID = "sqlite3"

    def __init__(self, bag_path: Path, depth_topic: str) -> None:
        self.bag_path = Path(bag_path).expanduser()
        self.depth_topic = depth_topic

        self._validate_configuration()

    def read_frames(self) -> Iterator[tuple[int, np.ndarray]]:
        """Yield recorded timestamps and decoded uint16 depth frames."""
        for timestamp_ns, message in self._read_depth_messages():
            try:
                depth_frame = self._decode_depth_image(message)
            except ValueError as exc:
                raise ValueError(
                    "Invalid depth Image at recorded timestamp "
                    f"{timestamp_ns}: {exc}"
                ) from exc

            yield timestamp_ns, depth_frame

    def _read_depth_messages(self) -> Iterator[tuple[int, Image]]:
        """Yield recorded timestamps and deserialized depth Image messages."""
        reader = self._open_reader()
        self._validate_depth_topic(reader)
        reader.set_filter(
            rosbag2_py.StorageFilter(
                topics=[self.depth_topic],
            )
        )

        while reader.has_next():
            topic, serialized_data, timestamp_ns = reader.read_next()

            # Keep this guard even though the storage filter should only return
            # the configured topic.
            if topic != self.depth_topic:
                continue

            try:
                message = deserialize_message(serialized_data, Image)
            except Exception as exc:
                raise RuntimeError(
                    "Failed to deserialize depth message from topic "
                    f"{topic!r} at timestamp {timestamp_ns}"
                ) from exc

            yield int(timestamp_ns), message

    @staticmethod
    def _decode_depth_image(message: Image) -> np.ndarray:
        """Decode a 16UC1 ROS Image into an owned native uint16 array."""
        if message.encoding != DepthBagReader._DEPTH_ENCODING:
            raise ValueError(
                f"Unsupported depth encoding {message.encoding!r}; "
                f"expected {DepthBagReader._DEPTH_ENCODING!r}"
            )

        height = int(message.height)
        width = int(message.width)
        step = int(message.step)
        is_bigendian = int(message.is_bigendian)

        if height <= 0 or width <= 0:
            raise ValueError(
                f"Depth dimensions must be positive; got height={height}, "
                f"width={width}"
            )

        minimum_step = width * DepthBagReader._BYTES_PER_PIXEL
        if step < minimum_step:
            raise ValueError(
                f"Depth row step is too small; got {step} bytes, "
                f"expected at least {minimum_step}"
            )

        if is_bigendian not in (0, 1):
            raise ValueError(
                f"is_bigendian must be 0 or 1; got {is_bigendian}"
            )

        try:
            data_buffer = memoryview(message.data)
        except TypeError as exc:
            raise ValueError("Depth image data does not expose a byte buffer") from exc

        if not data_buffer.c_contiguous:
            raise ValueError("Depth image data buffer must be C-contiguous")

        expected_data_size = height * step
        if data_buffer.nbytes != expected_data_size:
            raise ValueError(
                f"Depth data size mismatch; got {data_buffer.nbytes} bytes, "
                f"expected {expected_data_size}"
            )

        byte_order = ">" if is_bigendian else "<"
        source_dtype = np.dtype(f"{byte_order}u2")
        frame_view = np.ndarray(
            shape=(height, width),
            dtype=source_dtype,
            buffer=data_buffer,
            strides=(step, DepthBagReader._BYTES_PER_PIXEL),
        )

        return np.array(
            frame_view,
            dtype=np.uint16,
            order="C",
            copy=True,
        )

    def _validate_configuration(self) -> None:
        if not self.depth_topic:
            raise ValueError("depth_topic must not be empty")

        if not self.depth_topic.startswith("/"):
            raise ValueError(
                f"depth_topic must be an absolute ROS topic name: "
                f"{self.depth_topic!r}"
            )

        if not self.bag_path.exists():
            raise FileNotFoundError(f"Bag path does not exist: {self.bag_path}")

        if not self.bag_path.is_dir():
            raise NotADirectoryError(f"Bag path is not a directory: {self.bag_path}")

        metadata_path = self.bag_path / "metadata.yaml"
        if not metadata_path.is_file():
            raise ValueError(
                "Bag path does not contain metadata.yaml: "
                f"{self.bag_path}"
            )

    def _open_reader(self) -> rosbag2_py.SequentialReader:
        reader = rosbag2_py.SequentialReader()
        storage_options = rosbag2_py.StorageOptions(
            uri=str(self.bag_path),
            storage_id=self._STORAGE_ID,
        )
        converter_options = rosbag2_py.ConverterOptions(
            input_serialization_format="cdr",
            output_serialization_format="cdr",
        )

        try:
            reader.open(storage_options, converter_options)
        except Exception as exc:
            raise RuntimeError(f"Failed to open ROS 2 bag: {self.bag_path}") from exc

        return reader

    def _validate_depth_topic(self, reader: rosbag2_py.SequentialReader) -> None:
        topic_types = {
            topic.name: topic.type
            for topic in reader.get_all_topics_and_types()
        }

        if self.depth_topic not in topic_types:
            available_topics = ", ".join(sorted(topic_types)) or "<none>"
            raise ValueError(
                f"Depth topic {self.depth_topic!r} was not found in "
                f"{self.bag_path}. Available topics: {available_topics}"
            )

        actual_type = topic_types[self.depth_topic]
        if actual_type != self._DEPTH_MESSAGE_TYPE:
            raise ValueError(
                f"Depth topic {self.depth_topic!r} has type {actual_type!r}; "
                f"expected {self._DEPTH_MESSAGE_TYPE!r}"
            )
