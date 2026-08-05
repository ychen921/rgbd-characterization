"""Tests for ROS Image-like message decoding without ROS dependencies."""

from types import SimpleNamespace

import numpy as np
import pytest

from src.io.ros_image import decode_16uc1, decode_rgb8, frame_id, header_timestamp_ns


def make_image(
    data: bytes,
    *,
    height: int,
    width: int,
    step: int,
    encoding: str,
    is_bigendian: int = 0,
    sec: int = 1,
    nanosec: int = 2,
    image_frame_id: str = "camera_color_optical_frame",
) -> SimpleNamespace:
    return SimpleNamespace(
        height=height,
        width=width,
        step=step,
        encoding=encoding,
        is_bigendian=is_bigendian,
        data=data,
        header=SimpleNamespace(
            stamp=SimpleNamespace(sec=sec, nanosec=nanosec),
            frame_id=image_frame_id,
        ),
    )


def test_decodes_rgb8_without_changing_channel_order() -> None:
    expected = np.array(
        [[[255, 0, 0], [0, 255, 0]], [[0, 0, 255], [1, 2, 3]]],
        dtype=np.uint8,
    )
    message = make_image(
        expected.tobytes(),
        height=2,
        width=2,
        step=6,
        encoding="rgb8",
    )

    actual = decode_rgb8(message)

    assert np.array_equal(actual, expected)
    assert actual.dtype == np.uint8
    assert actual.flags.c_contiguous
    assert actual.flags.owndata


def test_decodes_rgb_rows_with_padding() -> None:
    expected = np.array([[[1, 2, 3]], [[4, 5, 6]]], dtype=np.uint8)
    data = b"\x01\x02\x03\xaa\xbb\x04\x05\x06\xcc\xdd"
    message = make_image(
        data,
        height=2,
        width=1,
        step=5,
        encoding="rgb8",
    )

    assert np.array_equal(decode_rgb8(message), expected)


@pytest.mark.parametrize(
    ("byte_order", "is_bigendian"),
    [("<", 0), (">", 1)],
)
def test_decodes_depth_endianness(byte_order: str, is_bigendian: int) -> None:
    expected = np.array([[0, 256], [1000, 65535]], dtype=np.uint16)
    message = make_image(
        expected.astype(f"{byte_order}u2").tobytes(),
        height=2,
        width=2,
        step=4,
        encoding="16UC1",
        is_bigendian=is_bigendian,
    )

    actual = decode_16uc1(message)

    assert np.array_equal(actual, expected)
    assert actual.dtype == np.uint16
    assert actual.flags.c_contiguous
    assert actual.flags.owndata


def test_decodes_depth_rows_with_padding() -> None:
    expected = np.array([[10, 20], [30, 40]], dtype=np.uint16)
    data = b"".join(row.astype("<u2").tobytes() + b"\xaa\xbb" for row in expected)
    message = make_image(
        data,
        height=2,
        width=2,
        step=6,
        encoding="16UC1",
    )

    assert np.array_equal(decode_16uc1(message), expected)


@pytest.mark.parametrize(
    ("decoder", "encoding", "expected"),
    [
        (decode_rgb8, "bgr8", "Unsupported rgb encoding"),
        (decode_16uc1, "mono16", "Unsupported depth encoding"),
    ],
)
def test_rejects_unsupported_encoding(decoder, encoding: str, expected: str) -> None:
    message = make_image(
        b"\x00\x00\x00",
        height=1,
        width=1,
        step=3,
        encoding=encoding,
    )

    with pytest.raises(ValueError, match=expected):
        decoder(message)


@pytest.mark.parametrize("height,width", [(0, 1), (1, 0)])
def test_rejects_non_positive_rgb_dimensions(height: int, width: int) -> None:
    message = make_image(
        b"",
        height=height,
        width=width,
        step=width * 3,
        encoding="rgb8",
    )

    with pytest.raises(ValueError, match="dimensions must be positive"):
        decode_rgb8(message)


def test_rejects_small_row_step() -> None:
    message = make_image(
        b"\x00" * 5,
        height=1,
        width=2,
        step=5,
        encoding="rgb8",
    )

    with pytest.raises(ValueError, match="row step is too small"):
        decode_rgb8(message)


def test_rejects_data_size_mismatch() -> None:
    message = make_image(
        b"\x00\x00",
        height=2,
        width=1,
        step=2,
        encoding="16UC1",
    )

    with pytest.raises(ValueError, match="data size mismatch"):
        decode_16uc1(message)


def test_rejects_invalid_endian_flag() -> None:
    message = make_image(
        b"\x00\x00",
        height=1,
        width=1,
        step=2,
        encoding="16UC1",
        is_bigendian=2,
    )

    with pytest.raises(ValueError, match="is_bigendian must be 0 or 1"):
        decode_16uc1(message)


def test_converts_header_timestamp_to_nanoseconds() -> None:
    message = make_image(
        b"\x00\x00\x00",
        height=1,
        width=1,
        step=3,
        encoding="rgb8",
        sec=12,
        nanosec=345,
    )

    assert header_timestamp_ns(message) == 12_000_000_345
    assert frame_id(message) == "camera_color_optical_frame"


@pytest.mark.parametrize(
    ("sec", "nanosec", "expected"),
    [
        (0, 0, "must be positive"),
        (-1, 0, "sec must be non-negative"),
        (1, -1, "nanosec must be in the range"),
        (1, 1_000_000_000, "nanosec must be in the range"),
    ],
)
def test_rejects_invalid_header_timestamp(
    sec: int,
    nanosec: int,
    expected: str,
) -> None:
    message = make_image(
        b"\x00\x00\x00",
        height=1,
        width=1,
        step=3,
        encoding="rgb8",
        sec=sec,
        nanosec=nanosec,
    )

    with pytest.raises(ValueError, match=expected):
        header_timestamp_ns(message)


def test_rejects_empty_frame_id() -> None:
    message = make_image(
        b"\x00\x00\x00",
        height=1,
        width=1,
        step=3,
        encoding="rgb8",
        image_frame_id="",
    )

    with pytest.raises(ValueError, match="non-empty"):
        frame_id(message)
