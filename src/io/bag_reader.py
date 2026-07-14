"""ROS 2 bag access for raw depth images."""

from collections.abc import Iterator
from pathlib import Path

import numpy as np
import rosbag2_py


class DepthBagReader:
    """Read raw depth frames from a ROS 2 bag.

    Frame deserialization is intentionally left for the next implementation
    step. Iterating :meth:`read_frames` currently opens the bag and validates
    the configured depth topic before raising ``NotImplementedError``.
    """

    _DEPTH_MESSAGE_TYPE = "sensor_msgs/msg/Image"
    _STORAGE_ID = "sqlite3"

    def __init__(self, bag_path: Path, depth_topic: str) -> None:
        self.bag_path = Path(bag_path).expanduser()
        self.depth_topic = depth_topic

        self._validate_configuration()

    def read_frames(self) -> Iterator[tuple[int, np.ndarray]]:
        """Validate the bag and depth topic, then iterate depth frames.

        The iterator will yield ``(recorded_timestamp_ns, depth_frame)`` once
        frame deserialization is implemented.
        """
        reader = self._open_reader()
        self._validate_depth_topic(reader)

        raise NotImplementedError(
            "Depth frame deserialization has not been implemented yet"
        )
        yield  # pragma: no cover - keeps this method an iterator until step 3

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
