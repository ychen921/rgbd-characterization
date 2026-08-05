"""NumPy dataset contract for unpaired RGB and aligned-depth streams."""

from dataclasses import dataclass
from pathlib import Path
from typing import ClassVar

import numpy as np


@dataclass
class AlignmentDataset:
    """Store full-resolution RGB and aligned-depth streams before pairing."""

    rgb: np.ndarray
    aligned_depth: np.ndarray
    rgb_timestamp_ns: np.ndarray
    depth_timestamp_ns: np.ndarray
    rgb_recorded_timestamp_ns: np.ndarray
    depth_recorded_timestamp_ns: np.ndarray

    SCHEMA_VERSION: ClassVar[int] = 1
    RGB_FILENAME: ClassVar[str] = "rgb.npz"
    DEPTH_FILENAME: ClassVar[str] = "aligned_depth.npz"
    TIMESTAMPS_FILENAME: ClassVar[str] = "timestamps.npz"

    COLOR_ENCODING: ClassVar[str] = "rgb8"
    COLOR_CHANNEL_ORDER: ClassVar[str] = "RGB"
    DEPTH_ENCODING: ClassVar[str] = "16UC1"
    DEPTH_PRECISION: ClassVar[str] = "1mm"
    DEPTH_UNIT: ClassVar[str] = "mm"
    DEPTH_INVALID_VALUES: ClassVar[tuple[int, int]] = (0, 65535)
    PRIMARY_TIMESTAMP_SOURCE: ClassVar[str] = "message_header"
    RECORDED_TIMESTAMP_SOURCE: ClassVar[str] = "rosbag_storage"
    TIMESTAMP_UNIT: ClassVar[str] = "ns"

    def __post_init__(self) -> None:
        """Validate array types, dimensions, dtypes, and stream contracts."""
        self._validate_rgb()
        self._validate_aligned_depth()

        self._validate_timestamps(
            name="rgb_timestamp_ns",
            timestamps=self.rgb_timestamp_ns,
            expected_count=self.num_rgb_frames,
        )
        self._validate_timestamps(
            name="depth_timestamp_ns",
            timestamps=self.depth_timestamp_ns,
            expected_count=self.num_depth_frames,
        )
        self._validate_timestamps(
            name="rgb_recorded_timestamp_ns",
            timestamps=self.rgb_recorded_timestamp_ns,
            expected_count=self.num_rgb_frames,
        )
        self._validate_timestamps(
            name="depth_recorded_timestamp_ns",
            timestamps=self.depth_recorded_timestamp_ns,
            expected_count=self.num_depth_frames,
        )

        if self.rgb_height != self.depth_height or self.rgb_width != self.depth_width:
            raise ValueError(
                "RGB and aligned depth must use the same pixel grid; got "
                f"RGB {self.rgb_width}x{self.rgb_height} and aligned depth "
                f"{self.depth_width}x{self.depth_height}"
            )

    @property
    def num_rgb_frames(self) -> int:
        """Return the number of RGB frames."""
        return self.rgb.shape[0]

    @property
    def num_depth_frames(self) -> int:
        """Return the number of aligned-depth frames."""
        return self.aligned_depth.shape[0]

    @property
    def rgb_height(self) -> int:
        """Return the RGB frame height."""
        return self.rgb.shape[1]

    @property
    def rgb_width(self) -> int:
        """Return the RGB frame width."""
        return self.rgb.shape[2]

    @property
    def depth_height(self) -> int:
        """Return the aligned-depth frame height."""
        return self.aligned_depth.shape[1]

    @property
    def depth_width(self) -> int:
        """Return the aligned-depth frame width."""
        return self.aligned_depth.shape[2]

    @property
    def height(self) -> int:
        """Return the common color-pixel-grid height."""
        return self.rgb_height

    @property
    def width(self) -> int:
        """Return the common color-pixel-grid width."""
        return self.rgb_width

    def save(self, directory: Path) -> None:
        """Save the dataset as three versioned NPZ archives."""
        output_dir = Path(directory).expanduser()
        if output_dir.exists() and not output_dir.is_dir():
            raise NotADirectoryError(
                f"Alignment dataset output is not a directory: {output_dir}"
            )

        output_paths = (
            output_dir / self.RGB_FILENAME,
            output_dir / self.DEPTH_FILENAME,
            output_dir / self.TIMESTAMPS_FILENAME,
        )
        existing_paths = [path for path in output_paths if path.exists()]
        if existing_paths:
            existing = ", ".join(str(path) for path in existing_paths)
            raise FileExistsError(
                f"Alignment dataset artifact(s) already exist: {existing}"
            )

        output_dir.mkdir(parents=True, exist_ok=True)
        schema_version = np.asarray(self.SCHEMA_VERSION, dtype=np.int64)

        np.savez(
            output_paths[0],
            schema_version=schema_version,
            rgb=self.rgb,
        )
        np.savez(
            output_paths[1],
            schema_version=schema_version,
            aligned_depth=self.aligned_depth,
        )
        np.savez(
            output_paths[2],
            schema_version=schema_version,
            rgb_timestamp_ns=self.rgb_timestamp_ns,
            depth_timestamp_ns=self.depth_timestamp_ns,
            rgb_recorded_timestamp_ns=self.rgb_recorded_timestamp_ns,
            depth_recorded_timestamp_ns=self.depth_recorded_timestamp_ns,
        )

    @classmethod
    def load(cls, directory: Path) -> "AlignmentDataset":
        """Load and validate an alignment dataset directory."""
        input_dir = Path(directory).expanduser()
        if not input_dir.exists():
            raise FileNotFoundError(
                f"Alignment dataset directory does not exist: {input_dir}"
            )
        if not input_dir.is_dir():
            raise NotADirectoryError(
                f"Alignment dataset path is not a directory: {input_dir}"
            )

        rgb_archive = cls._load_archive(
            input_dir / cls.RGB_FILENAME,
            required_arrays=("rgb",),
        )
        depth_archive = cls._load_archive(
            input_dir / cls.DEPTH_FILENAME,
            required_arrays=("aligned_depth",),
        )
        timestamp_archive = cls._load_archive(
            input_dir / cls.TIMESTAMPS_FILENAME,
            required_arrays=(
                "rgb_timestamp_ns",
                "depth_timestamp_ns",
                "rgb_recorded_timestamp_ns",
                "depth_recorded_timestamp_ns",
            ),
        )

        return cls(
            rgb=rgb_archive["rgb"],
            aligned_depth=depth_archive["aligned_depth"],
            rgb_timestamp_ns=timestamp_archive["rgb_timestamp_ns"],
            depth_timestamp_ns=timestamp_archive["depth_timestamp_ns"],
            rgb_recorded_timestamp_ns=timestamp_archive[
                "rgb_recorded_timestamp_ns"
            ],
            depth_recorded_timestamp_ns=timestamp_archive[
                "depth_recorded_timestamp_ns"
            ],
        )

    def _validate_rgb(self) -> None:
        self._require_array("rgb", self.rgb)
        if self.rgb.ndim != 4 or self.rgb.shape[-1] != 3:
            raise ValueError(
                "rgb must have shape (N, H, W, 3); got shape "
                f"{self.rgb.shape}"
            )
        if self.rgb.dtype != np.uint8:
            raise ValueError(f"rgb must have dtype uint8; got {self.rgb.dtype}")
        if self.rgb.shape[0] == 0:
            raise ValueError("rgb must contain at least one frame")
        if self.rgb.shape[1] == 0 or self.rgb.shape[2] == 0:
            raise ValueError("rgb height and width must be positive")

    def _validate_aligned_depth(self) -> None:
        self._require_array("aligned_depth", self.aligned_depth)
        if self.aligned_depth.ndim != 3:
            raise ValueError(
                "aligned_depth must have shape (N, H, W); got shape "
                f"{self.aligned_depth.shape}"
            )
        if self.aligned_depth.dtype != np.uint16:
            raise ValueError(
                "aligned_depth must have dtype uint16; got "
                f"{self.aligned_depth.dtype}"
            )
        if self.aligned_depth.shape[0] == 0:
            raise ValueError("aligned_depth must contain at least one frame")
        if self.aligned_depth.shape[1] == 0 or self.aligned_depth.shape[2] == 0:
            raise ValueError("aligned_depth height and width must be positive")

    @staticmethod
    def _validate_timestamps(
        *,
        name: str,
        timestamps: np.ndarray,
        expected_count: int,
    ) -> None:
        AlignmentDataset._require_array(name, timestamps)
        if timestamps.ndim != 1:
            raise ValueError(
                f"{name} must have shape (N,); got shape {timestamps.shape}"
            )
        if timestamps.dtype != np.int64:
            raise ValueError(
                f"{name} must have dtype int64; got {timestamps.dtype}"
            )
        if timestamps.shape[0] != expected_count:
            raise ValueError(
                f"{name} count does not match its stream frame count; got "
                f"{timestamps.shape[0]} timestamps and {expected_count} frames"
            )
        if np.any(timestamps <= 0):
            raise ValueError(f"{name} values must be positive")
        if timestamps.size > 1 and np.any(np.diff(timestamps) <= 0):
            raise ValueError(f"{name} values must be strictly increasing")

    @staticmethod
    def _require_array(name: str, value: object) -> None:
        if not isinstance(value, np.ndarray):
            raise TypeError(
                f"{name} must be a numpy.ndarray; got {type(value).__name__}"
            )

    @classmethod
    def _load_archive(
        cls,
        path: Path,
        *,
        required_arrays: tuple[str, ...],
    ) -> dict[str, np.ndarray]:
        if not path.is_file():
            raise FileNotFoundError(
                f"Alignment dataset archive does not exist: {path}"
            )

        with np.load(path, allow_pickle=False) as archive:
            required_keys = {"schema_version", *required_arrays}
            missing_keys = required_keys.difference(archive.files)
            if missing_keys:
                missing = ", ".join(sorted(missing_keys))
                raise ValueError(
                    f"Alignment dataset archive {path} is missing required "
                    f"array(s): {missing}"
                )

            schema_version = archive["schema_version"]
            if schema_version.shape != () or schema_version.dtype != np.int64:
                raise ValueError(
                    f"Alignment dataset archive {path} has an invalid "
                    "schema_version"
                )
            if int(schema_version) != cls.SCHEMA_VERSION:
                raise ValueError(
                    f"Unsupported alignment dataset schema version "
                    f"{int(schema_version)} in {path}; expected "
                    f"{cls.SCHEMA_VERSION}"
                )

            return {name: archive[name] for name in required_arrays}
