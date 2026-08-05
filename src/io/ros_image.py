"""Pure NumPy decoding helpers for ROS Image-like messages."""

from typing import Any

import numpy as np


def decode_rgb8(message: Any) -> np.ndarray:
    """Decode an ``rgb8`` image into an owned ``(H, W, 3)`` array."""
    height, width, step, data_buffer = _validate_image_layout(
        message,
        expected_encoding="rgb8",
        bytes_per_pixel=3,
        label="RGB",
    )

    frame_view = np.ndarray(
        shape=(height, width, 3),
        dtype=np.uint8,
        buffer=data_buffer,
        strides=(step, 3, 1),
    )
    return np.array(frame_view, dtype=np.uint8, order="C", copy=True)


def decode_16uc1(message: Any) -> np.ndarray:
    """Decode a ``16UC1`` image into an owned native-endian array."""
    height, width, step, data_buffer = _validate_image_layout(
        message,
        expected_encoding="16UC1",
        bytes_per_pixel=2,
        label="Depth",
    )

    is_bigendian = _require_integer_field(message, "is_bigendian", "Image")
    byte_order = ">" if is_bigendian else "<"
    source_dtype = np.dtype(f"{byte_order}u2")
    frame_view = np.ndarray(
        shape=(height, width),
        dtype=source_dtype,
        buffer=data_buffer,
        strides=(step, 2),
    )
    return np.array(frame_view, dtype=np.uint16, order="C", copy=True)


def header_timestamp_ns(message: Any) -> int:
    """Return a positive nanosecond timestamp from ``message.header.stamp``."""
    try:
        stamp = message.header.stamp
    except AttributeError as exc:
        raise ValueError("Message is missing header.stamp") from exc

    sec = _require_integer_value(getattr(stamp, "sec", None), "header.stamp.sec")
    nanosec = _require_integer_value(
        getattr(stamp, "nanosec", None),
        "header.stamp.nanosec",
    )
    if sec < 0:
        raise ValueError("header.stamp.sec must be non-negative")
    if nanosec < 0 or nanosec >= 1_000_000_000:
        raise ValueError("header.stamp.nanosec must be in the range [0, 1000000000)")

    timestamp_ns = sec * 1_000_000_000 + nanosec
    if timestamp_ns <= 0:
        raise ValueError("Message header timestamp must be positive")
    return timestamp_ns


def frame_id(message: Any) -> str:
    """Return a non-empty frame ID from an Image or CameraInfo message."""
    try:
        value = message.header.frame_id
    except AttributeError as exc:
        raise ValueError("Message is missing header.frame_id") from exc
    if not isinstance(value, str) or not value:
        raise ValueError("Message header.frame_id must be a non-empty string")
    return value


def _validate_image_layout(
    message: Any,
    *,
    expected_encoding: str,
    bytes_per_pixel: int,
    label: str,
) -> tuple[int, int, int, memoryview]:
    encoding = getattr(message, "encoding", None)
    if encoding != expected_encoding:
        raise ValueError(
            f"Unsupported {label.lower()} encoding {encoding!r}; "
            f"expected {expected_encoding!r}"
        )

    height = _require_integer_field(message, "height", label)
    width = _require_integer_field(message, "width", label)
    step = _require_integer_field(message, "step", label)
    is_bigendian = _require_integer_field(message, "is_bigendian", label)

    if height <= 0 or width <= 0:
        raise ValueError(
            f"{label} dimensions must be positive; got "
            f"height={height}, width={width}"
        )
    minimum_step = width * bytes_per_pixel
    if step < minimum_step:
        raise ValueError(
            f"{label} row step is too small; got {step} bytes, "
            f"expected at least {minimum_step}"
        )
    if is_bigendian not in (0, 1):
        raise ValueError(f"{label} is_bigendian must be 0 or 1; got {is_bigendian}")

    try:
        data_buffer = memoryview(message.data)
    except (AttributeError, TypeError) as exc:
        raise ValueError(f"{label} image data does not expose a byte buffer") from exc
    if not data_buffer.c_contiguous:
        raise ValueError(f"{label} image data buffer must be C-contiguous")

    expected_data_size = height * step
    if data_buffer.nbytes != expected_data_size:
        raise ValueError(
            f"{label} data size mismatch; got {data_buffer.nbytes} bytes, "
            f"expected {expected_data_size}"
        )
    return height, width, step, data_buffer


def _require_integer_field(message: Any, field_name: str, label: str) -> int:
    value = getattr(message, field_name, None)
    return _require_integer_value(value, f"{label} {field_name}")


def _require_integer_value(value: Any, field_name: str) -> int:
    if not isinstance(value, int) or isinstance(value, bool):
        raise ValueError(f"{field_name} must be an integer")
    return value
