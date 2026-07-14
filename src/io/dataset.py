"""Internal NumPy dataset format for raw depth frames."""

from dataclasses import dataclass
from pathlib import Path

import numpy as np


@dataclass
class DepthDataset:
    """Store a sequence of raw uint16 depth frames and their timestamps."""

    depth: np.ndarray
    timestamps_ns: np.ndarray

    def __post_init__(self) -> None:
        """Validate the dataset's array types, shapes, and dtypes."""
        if not isinstance(self.depth, np.ndarray):
            raise TypeError(
                f"depth must be a numpy.ndarray; got {type(self.depth).__name__}"
            )

        if not isinstance(self.timestamps_ns, np.ndarray):
            raise TypeError(
                "timestamps_ns must be a numpy.ndarray; got "
                f"{type(self.timestamps_ns).__name__}"
            )

        if self.depth.ndim != 3:
            raise ValueError(
                f"depth must have shape (N, H, W); got shape {self.depth.shape}"
            )

        if self.timestamps_ns.ndim != 1:
            raise ValueError(
                "timestamps_ns must have shape (N,); got shape "
                f"{self.timestamps_ns.shape}"
            )

        if self.depth.dtype != np.uint16:
            raise ValueError(
                f"depth must have dtype uint16; got {self.depth.dtype}"
            )

        if self.timestamps_ns.dtype != np.int64:
            raise ValueError(
                "timestamps_ns must have dtype int64; got "
                f"{self.timestamps_ns.dtype}"
            )

        if self.depth.shape[0] != self.timestamps_ns.shape[0]:
            raise ValueError(
                "Depth frame count does not match timestamp count; got "
                f"{self.depth.shape[0]} frames and "
                f"{self.timestamps_ns.shape[0]} timestamps"
            )

    @property
    def num_frames(self) -> int:
        """Return the number of depth frames."""
        return self.depth.shape[0]

    @property
    def height(self) -> int:
        """Return the depth frame height in pixels."""
        return self.depth.shape[1]

    @property
    def width(self) -> int:
        """Return the depth frame width in pixels."""
        return self.depth.shape[2]

    def save(self, path: Path) -> None:
        """Save the dataset to a NumPy NPZ archive."""
        output_path = Path(path).expanduser()
        np.savez(
            output_path,
            depth=self.depth,
            timestamps_ns=self.timestamps_ns,
        )

    @classmethod
    def load(cls, path: Path) -> "DepthDataset":
        """Load and validate a dataset from a NumPy NPZ archive."""
        input_path = Path(path).expanduser()

        with np.load(input_path, allow_pickle=False) as archive:
            missing_keys = {
                "depth",
                "timestamps_ns",
            }.difference(archive.files)
            if missing_keys:
                missing = ", ".join(sorted(missing_keys))
                raise ValueError(
                    f"Dataset archive is missing required array(s): {missing}"
                )

            depth = archive["depth"]
            timestamps_ns = archive["timestamps_ns"]

        return cls(
            depth=depth,
            timestamps_ns=timestamps_ns,
        )
