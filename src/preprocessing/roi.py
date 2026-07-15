"""Rectangle ROI handling and configuration persistence."""

from dataclasses import dataclass
from pathlib import Path
import re
from typing import Any, Mapping

import numpy as np
import yaml


@dataclass(frozen=True)
class RectROI:
    """Describe an axis-aligned rectangular image region."""

    x: int
    y: int
    width: int
    height: int

    def __post_init__(self) -> None:
        """Validate coordinate types and geometry independent of an image."""
        for field_name in ("x", "y", "width", "height"):
            value = getattr(self, field_name)
            if not isinstance(value, int) or isinstance(value, bool):
                raise ValueError(f"ROI {field_name} must be an integer")

        if self.x < 0 or self.y < 0:
            raise ValueError("ROI x and y must be non-negative")

        if self.width <= 0 or self.height <= 0:
            raise ValueError("ROI width and height must be positive")

    @property
    def pixel_count(self) -> int:
        """Return the number of pixels inside the ROI."""
        return self.width * self.height

    def crop(self, frames: np.ndarray) -> np.ndarray:
        """Crop an ``(N, H, W)`` frame array without clipping the ROI."""
        if not isinstance(frames, np.ndarray):
            raise TypeError(
                f"frames must be a numpy.ndarray; got {type(frames).__name__}"
            )

        if frames.ndim != 3:
            raise ValueError(
                f"frames must have shape (N, H, W); got shape {frames.shape}"
            )

        _, image_height, image_width = frames.shape

        if self.x + self.width > image_width:
            raise ValueError("ROI exceeds image width")

        if self.y + self.height > image_height:
            raise ValueError("ROI exceeds image height")

        return frames[
            :,
            self.y:self.y + self.height,
            self.x:self.x + self.width,
        ]


def derive_roi_key(experiment_name: str) -> str:
    """Remove a trailing repeat suffix from an experiment name."""
    if not isinstance(experiment_name, str):
        raise TypeError(
            "experiment_name must be a string; got "
            f"{type(experiment_name).__name__}"
        )

    return re.sub(r"_r\d+$", "", experiment_name)


def get_roi_path(roi_root: Path, experiment_name: str) -> Path:
    """Return the distance-group ROI configuration path for an experiment."""
    roi_key = derive_roi_key(experiment_name)
    return Path(roi_root).expanduser() / f"{roi_key}.yaml"


def save_roi(
    path: Path,
    roi: RectROI,
    *,
    name: str,
    source_experiment: str,
    source_frame_index: int,
) -> None:
    """Save an ROI and its selection provenance as YAML."""
    if not isinstance(roi, RectROI):
        raise TypeError(f"roi must be a RectROI; got {type(roi).__name__}")

    _validate_non_empty_string(name, "name")
    _validate_non_empty_string(source_experiment, "source experiment")

    if (
        not isinstance(source_frame_index, int)
        or isinstance(source_frame_index, bool)
        or source_frame_index < 0
    ):
        raise ValueError("source frame_index must be a non-negative integer")

    output_path = Path(path).expanduser()
    output_path.parent.mkdir(parents=True, exist_ok=True)

    document = {
        "name": name,
        "source": {
            "experiment": source_experiment,
            "frame_index": source_frame_index,
        },
        "roi": {
            "type": "rectangle",
            "x": roi.x,
            "y": roi.y,
            "width": roi.width,
            "height": roi.height,
        },
    }

    with output_path.open("w", encoding="utf-8") as stream:
        yaml.safe_dump(document, stream, sort_keys=False)


def load_roi(path: Path) -> RectROI:
    """Load and validate a rectangular ROI from a YAML configuration."""
    input_path = Path(path).expanduser()

    try:
        with input_path.open("r", encoding="utf-8") as stream:
            document = yaml.safe_load(stream)
    except yaml.YAMLError as error:
        message = f"Invalid ROI YAML in {input_path}: {error}"
        raise ValueError(message) from error

    root = _require_mapping(document, "ROI configuration")
    _validate_non_empty_string(root.get("name"), "name")

    source = _require_mapping(root.get("source"), "source")
    _validate_non_empty_string(
        source.get("experiment"),
        "source experiment",
    )
    frame_index = source.get("frame_index")
    if (
        not isinstance(frame_index, int)
        or isinstance(frame_index, bool)
        or frame_index < 0
    ):
        raise ValueError("source frame_index must be a non-negative integer")

    roi_data = _require_mapping(root.get("roi"), "roi")
    if roi_data.get("type") != "rectangle":
        raise ValueError("roi type must be 'rectangle'")

    return RectROI(
        x=_require_integer(roi_data, "x"),
        y=_require_integer(roi_data, "y"),
        width=_require_integer(roi_data, "width"),
        height=_require_integer(roi_data, "height"),
    )


def _require_mapping(value: Any, field_name: str) -> Mapping[str, Any]:
    """Return a configuration mapping or raise a descriptive error."""
    if not isinstance(value, Mapping):
        raise ValueError(f"{field_name} must be a mapping")
    return value


def _require_integer(mapping: Mapping[str, Any], field_name: str) -> int:
    """Return a required integer mapping value."""
    value = mapping.get(field_name)
    if not isinstance(value, int) or isinstance(value, bool):
        raise ValueError(f"ROI {field_name} must be an integer")
    return value


def _validate_non_empty_string(value: Any, field_name: str) -> None:
    """Validate a required, non-empty configuration string."""
    if not isinstance(value, str) or not value:
        raise ValueError(f"{field_name} must be a non-empty string")
