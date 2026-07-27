"""Scene 04 edge ROI configuration models and YAML persistence."""

from dataclasses import dataclass, field
import math
from numbers import Integral, Real
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import yaml

from src.preprocessing.roi import RectROI


EDGE_SCENE_TYPE = "edge_discontinuity"
VALID_FOREGROUND_SIDES = frozenset(
    {"left", "right", "positive", "negative"}
)


@dataclass(frozen=True)
class Line2D:
    """Represent a finite directed line segment in full-image coordinates."""

    p1: tuple[float, float]
    p2: tuple[float, float]

    def __post_init__(self) -> None:
        """Normalize endpoint coordinates and reject a zero-length line."""
        normalized_p1 = _normalize_point(self.p1, "p1")
        normalized_p2 = _normalize_point(self.p2, "p2")

        if normalized_p1 == normalized_p2:
            raise ValueError("nominal edge line must have non-zero length")

        object.__setattr__(self, "p1", normalized_p1)
        object.__setattr__(self, "p2", normalized_p2)


@dataclass(frozen=True)
class EdgeReferenceConfig:
    """Configure robust foreground/background reference tolerances."""

    minimum_tolerance_mm: float
    mad_scale: float
    minimum_valid_ratio: float = 0.9
    minimum_valid_count: int = 100

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "minimum_tolerance_mm",
            _normalize_positive_real(
                self.minimum_tolerance_mm,
                "minimum_tolerance_mm",
            ),
        )
        object.__setattr__(
            self,
            "mad_scale",
            _normalize_positive_real(self.mad_scale, "mad_scale"),
        )
        object.__setattr__(
            self,
            "minimum_valid_ratio",
            _normalize_unit_interval(
                self.minimum_valid_ratio,
                "minimum_valid_ratio",
            ),
        )
        if (
            not isinstance(self.minimum_valid_count, Integral)
            or isinstance(self.minimum_valid_count, (bool, np.bool_))
            or int(self.minimum_valid_count) <= 0
        ):
            raise ValueError(
                "minimum_valid_count must be a positive integer"
            )
        object.__setattr__(
            self,
            "minimum_valid_count",
            int(self.minimum_valid_count),
        )


@dataclass(frozen=True)
class EdgeBleedingConfig:
    """Configure the profile probability used for bleeding extent."""

    probability_threshold: float = 0.05

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "probability_threshold",
            _normalize_unit_interval(
                self.probability_threshold,
                "bleeding probability_threshold",
            ),
        )


@dataclass(frozen=True)
class EdgeInvalidConfig:
    """Configure the invalid-ratio threshold for invalid-band width."""

    ratio_threshold: float = 0.5

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "ratio_threshold",
            _normalize_unit_interval(
                self.ratio_threshold,
                "invalid ratio_threshold",
            ),
        )


@dataclass(frozen=True)
class EdgeTransitionConfig:
    """Configure probability crossings used by later transition metrics."""

    high_probability: float
    low_probability: float

    def __post_init__(self) -> None:
        high = _normalize_probability(
            self.high_probability,
            "high_probability",
        )
        low = _normalize_probability(
            self.low_probability,
            "low_probability",
        )

        if low >= high:
            raise ValueError(
                "low_probability must be less than high_probability"
            )

        object.__setattr__(self, "high_probability", high)
        object.__setattr__(self, "low_probability", low)


@dataclass(frozen=True)
class EdgeROIConfig:
    """Store one reusable Scene 04 multi-ROI annotation."""

    name: str
    source_experiment: str
    source_frame_index: int

    foreground_roi: RectROI
    background_roi: RectROI
    edge_roi: RectROI

    nominal_edge: Line2D
    foreground_side: str

    distance_bin_px: float
    max_edge_distance_px: float

    reference: EdgeReferenceConfig
    transition: EdgeTransitionConfig
    bleeding: EdgeBleedingConfig = field(
        default_factory=EdgeBleedingConfig
    )
    invalid: EdgeInvalidConfig = field(
        default_factory=EdgeInvalidConfig
    )

    def __post_init__(self) -> None:
        _validate_non_empty_string(self.name, "name")
        _validate_non_empty_string(
            self.source_experiment,
            "source experiment",
        )

        if (
            not isinstance(self.source_frame_index, Integral)
            or isinstance(self.source_frame_index, (bool, np.bool_))
            or int(self.source_frame_index) < 0
        ):
            raise ValueError(
                "source frame_index must be a non-negative integer"
            )

        for field_name in (
            "foreground_roi",
            "background_roi",
            "edge_roi",
        ):
            value = getattr(self, field_name)
            if not isinstance(value, RectROI):
                raise TypeError(
                    f"{field_name} must be a RectROI; "
                    f"got {type(value).__name__}"
                )

        if not isinstance(self.nominal_edge, Line2D):
            raise TypeError(
                "nominal_edge must be a Line2D; "
                f"got {type(self.nominal_edge).__name__}"
            )

        if self.foreground_side not in VALID_FOREGROUND_SIDES:
            allowed = ", ".join(sorted(VALID_FOREGROUND_SIDES))
            raise ValueError(
                f"foreground_side must be one of: {allowed}"
            )

        # Left/right describes horizontal image position. A horizontal line
        # has no distinct left or right half-plane, so its config must use the
        # endpoint-direction-aware positive/negative convention instead.
        line_dy = self.nominal_edge.p2[1] - self.nominal_edge.p1[1]
        if (
            self.foreground_side in {"left", "right"}
            and math.isclose(line_dy, 0.0, abs_tol=1e-12)
        ):
            raise ValueError(
                "horizontal nominal edge must use foreground_side "
                "'positive' or 'negative'"
            )

        object.__setattr__(
            self,
            "source_frame_index",
            int(self.source_frame_index),
        )
        object.__setattr__(
            self,
            "distance_bin_px",
            _normalize_positive_real(
                self.distance_bin_px,
                "distance_bin_px",
            ),
        )
        object.__setattr__(
            self,
            "max_edge_distance_px",
            _normalize_positive_real(
                self.max_edge_distance_px,
                "max_edge_distance_px",
            ),
        )

        # A whole number of bins on each side keeps the profile symmetric and
        # guarantees that one bin is centered exactly on the nominal edge.
        bin_count_per_side = (
            self.max_edge_distance_px
            / self.distance_bin_px
        )
        if not math.isclose(
            bin_count_per_side,
            round(bin_count_per_side),
            rel_tol=1e-9,
            abs_tol=1e-12,
        ):
            raise ValueError(
                "max_edge_distance_px must be an integer multiple "
                "of distance_bin_px"
            )

        if not isinstance(self.reference, EdgeReferenceConfig):
            raise TypeError(
                "reference must be an EdgeReferenceConfig; "
                f"got {type(self.reference).__name__}"
            )
        if not isinstance(self.transition, EdgeTransitionConfig):
            raise TypeError(
                "transition must be an EdgeTransitionConfig; "
                f"got {type(self.transition).__name__}"
            )
        if not isinstance(self.bleeding, EdgeBleedingConfig):
            raise TypeError(
                "bleeding must be an EdgeBleedingConfig; "
                f"got {type(self.bleeding).__name__}"
            )
        if not isinstance(self.invalid, EdgeInvalidConfig):
            raise TypeError(
                "invalid must be an EdgeInvalidConfig; "
                f"got {type(self.invalid).__name__}"
            )


def load_edge_roi_config(path: Path) -> EdgeROIConfig:
    """Load and validate a Scene 04 edge ROI YAML document."""
    input_path = Path(path).expanduser()

    try:
        with input_path.open("r", encoding="utf-8") as stream:
            documents = list(yaml.safe_load_all(stream))
    except yaml.YAMLError as error:
        raise ValueError(
            f"Invalid edge ROI YAML in {input_path}: {error}"
        ) from error

    non_empty_documents = [
        document
        for document in documents
        if document is not None
    ]
    if len(non_empty_documents) != 1:
        raise ValueError(
            "edge ROI YAML must contain exactly one non-empty document"
        )

    # Parse every nested section explicitly so malformed configuration fails
    # at load time instead of surfacing later inside frame analysis.
    root = _require_mapping(
        non_empty_documents[0],
        "edge ROI configuration",
    )
    if root.get("scene_type") != EDGE_SCENE_TYPE:
        raise ValueError(
            f"scene_type must be '{EDGE_SCENE_TYPE}'"
        )

    source = _require_mapping(root.get("source"), "source")
    nominal_edge_data = _require_mapping(
        root.get("nominal_edge"),
        "nominal_edge",
    )
    if nominal_edge_data.get("type") != "line":
        raise ValueError("nominal_edge type must be 'line'")

    analysis = _require_mapping(root.get("analysis"), "analysis")
    reference = _require_mapping(
        analysis.get("reference"),
        "analysis.reference",
    )
    transition = _require_mapping(
        analysis.get("transition"),
        "analysis.transition",
    )
    bleeding_value = analysis.get("bleeding")
    bleeding = (
        {}
        if bleeding_value is None
        else _require_mapping(
            bleeding_value,
            "analysis.bleeding",
        )
    )
    invalid_value = analysis.get("invalid")
    invalid = (
        {}
        if invalid_value is None
        else _require_mapping(
            invalid_value,
            "analysis.invalid",
        )
    )

    return EdgeROIConfig(
        name=_require_non_empty_string(root, "name"),
        source_experiment=_require_non_empty_string(
            source,
            "experiment",
        ),
        source_frame_index=_require_integer(
            source,
            "frame_index",
            prefix="source ",
        ),
        foreground_roi=_load_rectangle(
            root.get("foreground_roi"),
            "foreground_roi",
        ),
        background_roi=_load_rectangle(
            root.get("background_roi"),
            "background_roi",
        ),
        edge_roi=_load_rectangle(
            root.get("edge_roi"),
            "edge_roi",
        ),
        nominal_edge=Line2D(
            p1=_require_point(nominal_edge_data, "p1"),
            p2=_require_point(nominal_edge_data, "p2"),
        ),
        foreground_side=_require_non_empty_string(
            nominal_edge_data,
            "foreground_side",
        ),
        distance_bin_px=_require_real(
            analysis,
            "distance_bin_px",
        ),
        max_edge_distance_px=_require_real(
            analysis,
            "max_edge_distance_px",
        ),
        reference=EdgeReferenceConfig(
            minimum_tolerance_mm=_require_real(
                reference,
                "minimum_tolerance_mm",
            ),
            mad_scale=_require_real(reference, "mad_scale"),
            minimum_valid_ratio=_optional_real(
                reference,
                "minimum_valid_ratio",
                0.9,
            ),
            minimum_valid_count=_optional_integer(
                reference,
                "minimum_valid_count",
                100,
            ),
        ),
        transition=EdgeTransitionConfig(
            high_probability=_require_real(
                transition,
                "high_probability",
            ),
            low_probability=_require_real(
                transition,
                "low_probability",
            ),
        ),
        bleeding=EdgeBleedingConfig(
            probability_threshold=_optional_real(
                bleeding,
                "probability_threshold",
                0.05,
            ),
        ),
        invalid=EdgeInvalidConfig(
            ratio_threshold=_optional_real(
                invalid,
                "ratio_threshold",
                0.5,
            ),
        ),
    )


def save_edge_roi_config(
    path: Path,
    config: EdgeROIConfig,
) -> None:
    """Save one edge ROI configuration without overwriting existing data."""
    if not isinstance(config, EdgeROIConfig):
        raise TypeError(
            "config must be an EdgeROIConfig; "
            f"got {type(config).__name__}"
        )

    output_path = Path(path).expanduser()
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Keep the persisted topology stable because the later selection and
    # analysis tools will share this exact schema.
    document = {
        "name": config.name,
        "scene_type": EDGE_SCENE_TYPE,
        "source": {
            "experiment": config.source_experiment,
            "frame_index": config.source_frame_index,
        },
        "foreground_roi": _rectangle_document(
            config.foreground_roi
        ),
        "background_roi": _rectangle_document(
            config.background_roi
        ),
        "edge_roi": _rectangle_document(config.edge_roi),
        "nominal_edge": {
            "type": "line",
            "p1": list(config.nominal_edge.p1),
            "p2": list(config.nominal_edge.p2),
            "foreground_side": config.foreground_side,
        },
        "analysis": {
            "distance_bin_px": config.distance_bin_px,
            "max_edge_distance_px": config.max_edge_distance_px,
            "reference": {
                "minimum_tolerance_mm": (
                    config.reference.minimum_tolerance_mm
                ),
                "mad_scale": config.reference.mad_scale,
                "minimum_valid_ratio": (
                    config.reference.minimum_valid_ratio
                ),
                "minimum_valid_count": (
                    config.reference.minimum_valid_count
                ),
            },
            "bleeding": {
                "probability_threshold": (
                    config.bleeding.probability_threshold
                ),
            },
            "invalid": {
                "ratio_threshold": (
                    config.invalid.ratio_threshold
                ),
            },
            "transition": {
                "high_probability": (
                    config.transition.high_probability
                ),
                "low_probability": (
                    config.transition.low_probability
                ),
            },
        },
    }

    with output_path.open("x", encoding="utf-8") as stream:
        yaml.safe_dump(
            document,
            stream,
            sort_keys=False,
            allow_unicode=True,
        )


def validate_edge_roi_config(
    config: EdgeROIConfig,
    image_shape: tuple[int, int],
) -> None:
    """Validate all Scene 04 rectangles against one image resolution."""
    if not isinstance(config, EdgeROIConfig):
        raise TypeError(
            "config must be an EdgeROIConfig; "
            f"got {type(config).__name__}"
        )

    height, width = _validate_image_shape(image_shape)

    for field_name in (
        "foreground_roi",
        "background_roi",
        "edge_roi",
    ):
        roi = getattr(config, field_name)
        if roi.x + roi.width > width:
            raise ValueError(
                f"{field_name} exceeds image width"
            )
        if roi.y + roi.height > height:
            raise ValueError(
                f"{field_name} exceeds image height"
            )

    # Import locally to keep the data-model module and geometry module free
    # from an import-time cycle while still centralizing intersection logic.
    from src.geometry.edge_geometry import validate_edge_intersection

    validate_edge_intersection(
        config.edge_roi,
        config.nominal_edge,
    )


def build_roi_mask(
    image_shape: tuple[int, int],
    roi: RectROI,
) -> np.ndarray:
    """Return a full-image boolean mask for one rectangular ROI."""
    height, width = _validate_image_shape(image_shape)
    if not isinstance(roi, RectROI):
        raise TypeError(
            f"roi must be a RectROI; got {type(roi).__name__}"
        )
    if roi.x + roi.width > width:
        raise ValueError("ROI exceeds image width")
    if roi.y + roi.height > height:
        raise ValueError("ROI exceeds image height")

    mask = np.zeros((height, width), dtype=bool)
    mask[
        roi.y:roi.y + roi.height,
        roi.x:roi.x + roi.width,
    ] = True
    return mask


def _load_rectangle(value: Any, field_name: str) -> RectROI:
    """Load one required rectangle mapping."""
    mapping = _require_mapping(value, field_name)
    if mapping.get("type") != "rectangle":
        raise ValueError(
            f"{field_name} type must be 'rectangle'"
        )
    return RectROI(
        x=_require_integer(mapping, "x", prefix=f"{field_name} "),
        y=_require_integer(mapping, "y", prefix=f"{field_name} "),
        width=_require_integer(
            mapping,
            "width",
            prefix=f"{field_name} ",
        ),
        height=_require_integer(
            mapping,
            "height",
            prefix=f"{field_name} ",
        ),
    )


def _rectangle_document(roi: RectROI) -> dict[str, object]:
    """Return one YAML-safe rectangle mapping."""
    return {
        "type": "rectangle",
        "x": roi.x,
        "y": roi.y,
        "width": roi.width,
        "height": roi.height,
    }


def _normalize_point(
    value: object,
    field_name: str,
) -> tuple[float, float]:
    """Normalize a two-coordinate point to finite Python floats."""
    if not isinstance(value, (tuple, list)) or len(value) != 2:
        raise ValueError(
            f"{field_name} must contain exactly two coordinates"
        )
    return (
        _normalize_finite_real(value[0], f"{field_name}[0]"),
        _normalize_finite_real(value[1], f"{field_name}[1]"),
    )


def _normalize_finite_real(value: object, field_name: str) -> float:
    """Return a finite real number as a Python float."""
    if (
        isinstance(value, (bool, np.bool_))
        or not isinstance(value, Real)
    ):
        raise ValueError(f"{field_name} must be a finite number")
    normalized = float(value)
    if not math.isfinite(normalized):
        raise ValueError(f"{field_name} must be a finite number")
    return normalized


def _normalize_positive_real(value: object, field_name: str) -> float:
    """Return a finite strictly positive number."""
    normalized = _normalize_finite_real(value, field_name)
    if normalized <= 0.0:
        raise ValueError(f"{field_name} must be positive")
    return normalized


def _normalize_probability(value: object, field_name: str) -> float:
    """Return a probability strictly between zero and one."""
    normalized = _normalize_finite_real(value, field_name)
    if not 0.0 < normalized < 1.0:
        raise ValueError(
            f"{field_name} must be between 0 and 1"
        )
    return normalized


def _normalize_unit_interval(value: object, field_name: str) -> float:
    """Return a finite ratio or probability between zero and one."""
    normalized = _normalize_finite_real(value, field_name)
    if not 0.0 <= normalized <= 1.0:
        raise ValueError(
            f"{field_name} must be between 0 and 1"
        )
    return normalized


def _validate_non_empty_string(value: object, field_name: str) -> None:
    """Validate a required non-empty string."""
    if not isinstance(value, str) or not value:
        raise ValueError(f"{field_name} must be a non-empty string")


def _validate_image_shape(
    image_shape: tuple[int, int],
) -> tuple[int, int]:
    """Validate and normalize an image ``(height, width)`` pair."""
    if (
        not isinstance(image_shape, (tuple, list))
        or len(image_shape) != 2
    ):
        raise ValueError(
            "image_shape must contain (height, width)"
        )

    normalized: list[int] = []
    for field_name, value in zip(
        ("height", "width"),
        image_shape,
        strict=True,
    ):
        if (
            not isinstance(value, Integral)
            or isinstance(value, (bool, np.bool_))
            or int(value) <= 0
        ):
            raise ValueError(
                f"image {field_name} must be a positive integer"
            )
        normalized.append(int(value))

    return normalized[0], normalized[1]


def _require_mapping(value: Any, field_name: str) -> Mapping[str, Any]:
    """Return a required configuration mapping."""
    if not isinstance(value, Mapping):
        raise ValueError(f"{field_name} must be a mapping")
    return value


def _require_non_empty_string(
    mapping: Mapping[str, Any],
    field_name: str,
) -> str:
    """Return a required non-empty string field."""
    value = mapping.get(field_name)
    _validate_non_empty_string(value, field_name)
    return value


def _require_integer(
    mapping: Mapping[str, Any],
    field_name: str,
    *,
    prefix: str = "",
) -> int:
    """Return a required integer field."""
    value = mapping.get(field_name)
    if (
        not isinstance(value, Integral)
        or isinstance(value, (bool, np.bool_))
    ):
        raise ValueError(f"{prefix}{field_name} must be an integer")
    return int(value)


def _require_real(
    mapping: Mapping[str, Any],
    field_name: str,
) -> float:
    """Return a required finite numeric field."""
    return _normalize_finite_real(mapping.get(field_name), field_name)


def _optional_real(
    mapping: Mapping[str, Any],
    field_name: str,
    default: float,
) -> float:
    """Return an optional finite numeric field or its default."""
    if field_name not in mapping:
        return default
    return _normalize_finite_real(mapping[field_name], field_name)


def _optional_integer(
    mapping: Mapping[str, Any],
    field_name: str,
    default: int,
) -> int:
    """Return an optional integer field or its default."""
    if field_name not in mapping:
        return default
    return _require_integer(mapping, field_name)


def _require_point(
    mapping: Mapping[str, Any],
    field_name: str,
) -> tuple[float, float]:
    """Return a required two-coordinate point."""
    return _normalize_point(mapping.get(field_name), field_name)
