"""Depth-camera calibration loading and image-resolution validation."""

from dataclasses import dataclass
import math
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import yaml


@dataclass(frozen=True)
class CameraIntrinsics:
    """Pinhole intrinsics associated with a calibrated image resolution."""

    width: int
    height: int
    fx: float
    fy: float
    cx: float
    cy: float
    frame_id: str

    def __post_init__(self) -> None:
        """Reject incomplete or physically invalid calibration values."""
        for field_name in ("width", "height"):
            value = getattr(self, field_name)
            if (
                not isinstance(value, int)
                or isinstance(value, bool)
                or value <= 0
            ):
                raise ValueError(f"{field_name} must be a positive integer")

        for field_name in ("fx", "fy", "cx", "cy"):
            value = getattr(self, field_name)
            if (
                not isinstance(value, (int, float))
                or isinstance(value, bool)
                or not math.isfinite(value)
            ):
                raise ValueError(f"{field_name} must be a finite number")

        if self.fx <= 0 or self.fy <= 0:
            raise ValueError("fx and fy must be positive")

        if not isinstance(self.frame_id, str) or not self.frame_id:
            raise ValueError("frame_id must be a non-empty string")


def load_camera_intrinsics(path: Path) -> CameraIntrinsics:
    """Load pinhole intrinsics from a ROS CameraInfo YAML document."""
    input_path = Path(path).expanduser()

    try:
        with input_path.open("r", encoding="utf-8") as stream:
            documents = list(yaml.safe_load_all(stream))
    except yaml.YAMLError as error:
        raise ValueError(
            f"Invalid camera-info YAML in {input_path}: {error}"
        ) from error

    non_empty_documents = [document for document in documents if document is not None]
    if len(non_empty_documents) != 1:
        raise ValueError(
            "camera-info YAML must contain exactly one non-empty document"
        )

    root = _require_mapping(non_empty_documents[0], "camera-info configuration")
    header = _require_mapping(root.get("header"), "header")
    frame_id = header.get("frame_id")
    if not isinstance(frame_id, str) or not frame_id:
        raise ValueError("frame_id must be a non-empty string")

    matrix = root.get("k")
    if not isinstance(matrix, list) or len(matrix) != 9:
        raise ValueError("k must be a list containing 9 values")

    return CameraIntrinsics(
        width=_require_integer(root, "width"),
        height=_require_integer(root, "height"),
        fx=_require_number(matrix[0], "k[0] (fx)"),
        fy=_require_number(matrix[4], "k[4] (fy)"),
        cx=_require_number(matrix[2], "k[2] (cx)"),
        cy=_require_number(matrix[5], "k[5] (cy)"),
        frame_id=frame_id,
    )


def validate_depth_resolution(
    depth: np.ndarray,
    intrinsics: CameraIntrinsics,
) -> None:
    """Ensure a depth image or frame sequence matches the calibration."""
    if not isinstance(depth, np.ndarray):
        raise TypeError(
            f"depth must be a numpy.ndarray; got {type(depth).__name__}"
        )
    if not isinstance(intrinsics, CameraIntrinsics):
        raise TypeError(
            "intrinsics must be CameraIntrinsics; got "
            f"{type(intrinsics).__name__}"
        )
    if depth.ndim not in (2, 3):
        raise ValueError(
            "depth must have shape (H, W) or (N, H, W); "
            f"got shape {depth.shape}"
        )

    actual_height, actual_width = depth.shape[-2:]
    if (actual_width, actual_height) != (
        intrinsics.width,
        intrinsics.height,
    ):
        raise ValueError(
            "Depth resolution does not match camera intrinsics: "
            f"depth={actual_width}x{actual_height}, "
            f"intrinsics={intrinsics.width}x{intrinsics.height}"
        )


def _require_mapping(value: Any, field_name: str) -> Mapping[str, Any]:
    """Return a required configuration mapping."""
    if not isinstance(value, Mapping):
        raise ValueError(f"{field_name} must be a mapping")
    return value


def _require_integer(mapping: Mapping[str, Any], field_name: str) -> int:
    """Return a required integer configuration value."""
    value = mapping.get(field_name)
    if not isinstance(value, int) or isinstance(value, bool):
        raise ValueError(f"{field_name} must be an integer")
    return value


def _require_number(value: Any, field_name: str) -> float:
    """Return a required numeric configuration value as a float."""
    if not isinstance(value, (int, float)) or isinstance(value, bool):
        raise ValueError(f"{field_name} must be a number")
    return float(value)
