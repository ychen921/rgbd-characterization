"""Signed-distance and intersection geometry for Scene 04 edges."""

import math
from numbers import Integral

import numpy as np

from src.preprocessing.edge_roi import (
    Line2D,
    VALID_FOREGROUND_SIDES,
)
from src.preprocessing.roi import RectROI


def compute_signed_distance_map(
    image_shape: tuple[int, int],
    line: Line2D,
    foreground_side: str,
) -> np.ndarray:
    """Return signed perpendicular pixel distance for every image pixel.

    The returned convention is fixed for all callers:
    negative is foreground, positive is background, and zero is the nominal
    edge. ``positive`` and ``negative`` refer to the raw cross-product side of
    the directed segment p1 -> p2; ``left`` and ``right`` refer to horizontal
    image position and are independent of endpoint order.
    """
    height, width = _validate_image_shape(image_shape)
    _validate_line_and_side(line, foreground_side)

    x1, y1 = line.p1
    x2, y2 = line.p2
    dx = x2 - x1
    dy = y2 - y1
    line_length = math.hypot(dx, dy)

    # Pixel coordinates follow the repository's existing camera convention:
    # integer (u, v) coordinates are used directly, without a +0.5 offset.
    v, u = np.indices((height, width), dtype=np.float64)
    raw_distance = (
        dx * (v - y1)
        - dy * (u - x1)
    ) / line_length

    # Convert every accepted foreground-side description into one multiplier
    # that maps that physical side to a negative returned distance.
    multiplier = _foreground_multiplier(
        line=line,
        foreground_side=foreground_side,
    )
    return raw_distance * multiplier


def validate_edge_intersection(
    edge_roi: RectROI,
    line: Line2D,
) -> None:
    """Require the finite nominal-edge segment to intersect the edge ROI."""
    if not isinstance(edge_roi, RectROI):
        raise TypeError(
            "edge_roi must be a RectROI; "
            f"got {type(edge_roi).__name__}"
        )
    if not isinstance(line, Line2D):
        raise TypeError(
            f"line must be a Line2D; got {type(line).__name__}"
        )

    # RectROI uses exclusive array bounds for cropping. For continuous line
    # geometry, both outer rectangle boundaries are included so an annotated
    # endpoint at x + width or y + height counts as touching the ROI.
    x_min = float(edge_roi.x)
    x_max = float(edge_roi.x + edge_roi.width)
    y_min = float(edge_roi.y)
    y_max = float(edge_roi.y + edge_roi.height)

    if not _segment_intersects_rectangle(
        line=line,
        x_min=x_min,
        x_max=x_max,
        y_min=y_min,
        y_max=y_max,
    ):
        raise ValueError(
            "nominal edge line segment does not intersect edge ROI"
        )


def _foreground_multiplier(
    *,
    line: Line2D,
    foreground_side: str,
) -> float:
    """Return the multiplier that makes the foreground side negative."""
    if foreground_side == "positive":
        return -1.0
    if foreground_side == "negative":
        return 1.0

    dy = line.p2[1] - line.p1[1]
    if math.isclose(dy, 0.0, abs_tol=1e-12):
        raise ValueError(
            "horizontal nominal edge must use foreground_side "
            "'positive' or 'negative'"
        )

    # At the segment midpoint, shifting one pixel left produces a raw
    # cross-product sign equal to sign(dy). This makes left/right semantics
    # independent of whether endpoints were selected top-to-bottom or in the
    # reverse order.
    left_raw_sign = 1.0 if dy > 0.0 else -1.0
    if foreground_side == "left":
        return -left_raw_sign
    return left_raw_sign


def _segment_intersects_rectangle(
    *,
    line: Line2D,
    x_min: float,
    x_max: float,
    y_min: float,
    y_max: float,
) -> bool:
    """Test a finite segment against an inclusive rectangle.

    Liang-Barsky clipping tracks the parameter interval ``t in [0, 1]`` for
    which p1 + t * (p2 - p1) lies inside every rectangle half-plane. A
    non-empty interval means the segment crosses, enters, or touches the ROI.
    """
    x1, y1 = line.p1
    x2, y2 = line.p2
    dx = x2 - x1
    dy = y2 - y1

    t_enter = 0.0
    t_exit = 1.0

    boundaries = (
        (-dx, x1 - x_min),
        (dx, x_max - x1),
        (-dy, y1 - y_min),
        (dy, y_max - y1),
    )

    for direction, offset in boundaries:
        if math.isclose(direction, 0.0, abs_tol=1e-12):
            if offset < 0.0:
                return False
            continue

        ratio = offset / direction
        if direction < 0.0:
            t_enter = max(t_enter, ratio)
        else:
            t_exit = min(t_exit, ratio)

        if t_enter > t_exit:
            return False

    return True


def _validate_line_and_side(
    line: Line2D,
    foreground_side: str,
) -> None:
    """Validate public signed-distance inputs."""
    if not isinstance(line, Line2D):
        raise TypeError(
            f"line must be a Line2D; got {type(line).__name__}"
        )
    if foreground_side not in VALID_FOREGROUND_SIDES:
        allowed = ", ".join(sorted(VALID_FOREGROUND_SIDES))
        raise ValueError(
            f"foreground_side must be one of: {allowed}"
        )


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
