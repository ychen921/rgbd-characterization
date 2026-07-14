"""Tests for decoding ROS depth Image messages."""

from array import array

import numpy as np
import pytest
from sensor_msgs.msg import Image

from src.io.bag_reader import DepthBagReader


def make_depth_image(
    data: bytes,
    *,
    height: int,
    width: int,
    step: int,
    is_bigendian: int = 0,
    encoding: str = "16UC1",
) -> Image:
    message = Image()
    message.height = height
    message.width = width
    message.encoding = encoding
    message.is_bigendian = is_bigendian
    message.step = step
    message.data = array("B", data)
    return message


def test_decodes_little_endian_depth_image_to_owned_native_array() -> None:
    expected = np.array(
        [[1, 256, 65535], [42, 1000, 4096]],
        dtype=np.uint16,
    )
    message = make_depth_image(
        expected.astype("<u2").tobytes(),
        height=2,
        width=3,
        step=6,
    )

    actual = DepthBagReader._decode_depth_image(message)

    assert np.array_equal(actual, expected)
    assert actual.dtype == np.uint16
    assert actual.flags.c_contiguous
    assert actual.flags.owndata


def test_decodes_big_endian_depth_image() -> None:
    expected = np.array([[1, 256], [4096, 65535]], dtype=np.uint16)
    message = make_depth_image(
        expected.astype(">u2").tobytes(),
        height=2,
        width=2,
        step=4,
        is_bigendian=1,
    )

    actual = DepthBagReader._decode_depth_image(message)

    assert np.array_equal(actual, expected)
    assert actual.dtype == np.uint16


def test_decodes_rows_with_padding() -> None:
    expected = np.array([[10, 20], [30, 40]], dtype=np.uint16)
    padding = b"\xaa\xbb"
    data = b"".join(
        row.astype("<u2").tobytes() + padding
        for row in expected
    )
    message = make_depth_image(
        data,
        height=2,
        width=2,
        step=6,
    )

    actual = DepthBagReader._decode_depth_image(message)

    assert np.array_equal(actual, expected)


def test_rejects_unsupported_encoding() -> None:
    message = make_depth_image(
        b"\x00\x00",
        height=1,
        width=1,
        step=2,
        encoding="mono16",
    )

    with pytest.raises(ValueError, match="Unsupported depth encoding"):
        DepthBagReader._decode_depth_image(message)


@pytest.mark.parametrize(("height", "width"), [(0, 1), (1, 0)])
def test_rejects_non_positive_dimensions(height: int, width: int) -> None:
    message = make_depth_image(
        b"",
        height=height,
        width=width,
        step=width * 2,
    )

    with pytest.raises(ValueError, match="dimensions must be positive"):
        DepthBagReader._decode_depth_image(message)


def test_rejects_row_step_smaller_than_pixel_data() -> None:
    message = make_depth_image(
        b"\x00\x00\x00",
        height=1,
        width=2,
        step=3,
    )

    with pytest.raises(ValueError, match="row step is too small"):
        DepthBagReader._decode_depth_image(message)


def test_rejects_invalid_endian_flag() -> None:
    message = make_depth_image(
        b"\x00\x00",
        height=1,
        width=1,
        step=2,
        is_bigendian=2,
    )

    with pytest.raises(ValueError, match="is_bigendian must be 0 or 1"):
        DepthBagReader._decode_depth_image(message)


def test_rejects_data_size_mismatch() -> None:
    message = make_depth_image(
        b"\x00\x00",
        height=2,
        width=1,
        step=2,
    )

    with pytest.raises(ValueError, match="Depth data size mismatch"):
        DepthBagReader._decode_depth_image(message)
