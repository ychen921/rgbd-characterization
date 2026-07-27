"""Tests for Scene 04 signed-distance and intersection geometry."""

import numpy as np
import pytest

from src.geometry.edge_geometry import (
    compute_signed_distance_map,
    validate_edge_intersection,
)
from src.preprocessing.edge_roi import Line2D
from src.preprocessing.roi import RectROI


def test_vertical_line_with_left_foreground() -> None:
    line = Line2D(p1=(2.0, 0.0), p2=(2.0, 4.0))

    distance = compute_signed_distance_map(
        (3, 5),
        line,
        foreground_side="left",
    )

    expected_row = np.array([-2.0, -1.0, 0.0, 1.0, 2.0])
    np.testing.assert_allclose(
        distance,
        np.tile(expected_row, (3, 1)),
    )
    assert distance.dtype == np.float64


def test_vertical_line_with_right_foreground() -> None:
    line = Line2D(p1=(2.0, 0.0), p2=(2.0, 4.0))

    distance = compute_signed_distance_map(
        (1, 5),
        line,
        foreground_side="right",
    )

    np.testing.assert_allclose(
        distance,
        [[2.0, 1.0, 0.0, -1.0, -2.0]],
    )


def test_left_right_semantics_do_not_depend_on_endpoint_order() -> None:
    forward = Line2D(p1=(2.0, 0.0), p2=(2.0, 4.0))
    reverse = Line2D(p1=(2.0, 4.0), p2=(2.0, 0.0))

    forward_distance = compute_signed_distance_map(
        (3, 5),
        forward,
        foreground_side="left",
    )
    reverse_distance = compute_signed_distance_map(
        (3, 5),
        reverse,
        foreground_side="left",
    )

    np.testing.assert_allclose(reverse_distance, forward_distance)


def test_horizontal_line_with_positive_foreground() -> None:
    line = Line2D(p1=(0.0, 1.0), p2=(4.0, 1.0))

    distance = compute_signed_distance_map(
        (3, 2),
        line,
        foreground_side="positive",
    )

    # The raw positive side is below the directed horizontal line, so it is
    # normalized to negative foreground distance.
    np.testing.assert_allclose(
        distance,
        [[1.0, 1.0], [0.0, 0.0], [-1.0, -1.0]],
    )


def test_horizontal_line_with_negative_foreground() -> None:
    line = Line2D(p1=(0.0, 1.0), p2=(4.0, 1.0))

    distance = compute_signed_distance_map(
        (3, 2),
        line,
        foreground_side="negative",
    )

    np.testing.assert_allclose(
        distance,
        [[-1.0, -1.0], [0.0, 0.0], [1.0, 1.0]],
    )


def test_slanted_line_returns_perpendicular_distance() -> None:
    line = Line2D(p1=(0.0, 0.0), p2=(2.0, 2.0))

    distance = compute_signed_distance_map(
        (3, 3),
        line,
        foreground_side="negative",
    )

    expected = np.array(
        [
            [0.0, -1 / np.sqrt(2), -np.sqrt(2)],
            [1 / np.sqrt(2), 0.0, -1 / np.sqrt(2)],
            [np.sqrt(2), 1 / np.sqrt(2), 0.0],
        ]
    )
    np.testing.assert_allclose(distance, expected)


def test_horizontal_line_rejects_left_right_foreground() -> None:
    line = Line2D(p1=(0.0, 1.0), p2=(4.0, 1.0))

    with pytest.raises(ValueError, match="horizontal"):
        compute_signed_distance_map(
            (3, 3),
            line,
            foreground_side="left",
        )


def test_signed_distance_rejects_unknown_foreground_side() -> None:
    line = Line2D(p1=(1.0, 0.0), p2=(1.0, 2.0))

    with pytest.raises(ValueError, match="foreground_side"):
        compute_signed_distance_map(
            (3, 3),
            line,
            foreground_side="inside",
        )


@pytest.mark.parametrize(
    ("image_shape", "expected_message"),
    [
        ((3,), "height, width"),
        ((0, 3), "height"),
        ((3, 0), "width"),
        ((3.5, 3), "height"),
    ],
)
def test_signed_distance_rejects_invalid_image_shape(
    image_shape: object,
    expected_message: str,
) -> None:
    line = Line2D(p1=(1.0, 0.0), p2=(1.0, 2.0))

    with pytest.raises(ValueError, match=expected_message):
        compute_signed_distance_map(
            image_shape,
            line,
            foreground_side="left",
        )


def test_line2d_rejects_zero_length_segment() -> None:
    with pytest.raises(ValueError, match="non-zero length"):
        Line2D(p1=(1.0, 2.0), p2=(1.0, 2.0))


@pytest.mark.parametrize(
    "line",
    [
        Line2D(p1=(-1.0, 2.0), p2=(6.0, 2.0)),
        Line2D(p1=(2.0, 2.0), p2=(3.0, 3.0)),
        Line2D(p1=(-1.0, 1.0), p2=(1.0, 1.0)),
        Line2D(p1=(4.0, 4.0), p2=(6.0, 6.0)),
    ],
)
def test_validate_edge_intersection_accepts_crossing_inside_or_touching(
    line: Line2D,
) -> None:
    roi = RectROI(x=1, y=1, width=3, height=3)

    validate_edge_intersection(roi, line)


@pytest.mark.parametrize(
    "line",
    [
        Line2D(p1=(-1.0, 0.0), p2=(6.0, 0.0)),
        Line2D(p1=(-3.0, 2.0), p2=(-1.0, 2.0)),
        Line2D(p1=(5.0, 1.0), p2=(5.0, 4.0)),
    ],
)
def test_validate_edge_intersection_rejects_non_intersecting_segment(
    line: Line2D,
) -> None:
    roi = RectROI(x=1, y=1, width=3, height=3)

    with pytest.raises(ValueError, match="does not intersect"):
        validate_edge_intersection(roi, line)


def test_signed_distance_is_deterministic() -> None:
    line = Line2D(p1=(1.0, 0.0), p2=(2.0, 3.0))

    first = compute_signed_distance_map(
        (4, 4),
        line,
        foreground_side="left",
    )
    second = compute_signed_distance_map(
        (4, 4),
        line,
        foreground_side="left",
    )

    np.testing.assert_array_equal(first, second)
