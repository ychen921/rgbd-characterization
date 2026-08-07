"""Contracts for one-to-one RGB and aligned-depth timestamp pairing."""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from numbers import Integral, Real
from typing import ClassVar

import numpy as np


NANOSECONDS_PER_MILLISECOND = 1_000_000


@dataclass(frozen=True)
class FramePairingConfig:
    """Configure the fixed schema-v1 timestamp-pairing contract."""

    max_abs_delta_ms: float = 20.0
    _max_abs_delta_ns: int = field(init=False, repr=False)

    SCHEMA_VERSION: ClassVar[int] = 1
    METHOD: ClassVar[str] = "nearest_timestamp"
    TIMESTAMP_SOURCE: ClassVar[str] = "message_header"
    DELTA_DEFINITION: ClassVar[str] = "depth_minus_rgb"
    THRESHOLD_INCLUSIVE: ClassVar[bool] = True
    CARDINALITY: ClassVar[str] = "one_to_one"
    PRESERVE_ORDER: ClassVar[bool] = True
    TIE_BREAKER: ClassVar[str] = "earlier_depth"

    def __post_init__(self) -> None:
        """Normalize the threshold and require nanosecond precision."""
        value = self.max_abs_delta_ms
        if (
            not isinstance(value, Real)
            or isinstance(value, (bool, np.bool_))
            or not math.isfinite(float(value))
        ):
            raise ValueError("max_abs_delta_ms must be a finite real number")

        threshold_ms = float(value)
        if threshold_ms < 0.0:
            raise ValueError("max_abs_delta_ms must be non-negative")

        threshold_ns_float = threshold_ms * NANOSECONDS_PER_MILLISECOND
        if (
            not math.isfinite(threshold_ns_float)
            or threshold_ns_float > np.iinfo(np.int64).max
        ):
            raise ValueError("max_abs_delta_ms exceeds the int64 timestamp range")
        threshold_ns = round(threshold_ns_float)
        if not math.isclose(
            threshold_ns_float,
            threshold_ns,
            rel_tol=0.0,
            abs_tol=1e-9,
        ):
            raise ValueError(
                "max_abs_delta_ms must be representable as an integer "
                "number of nanoseconds"
            )
        object.__setattr__(
            self,
            "max_abs_delta_ms",
            threshold_ns / NANOSECONDS_PER_MILLISECOND,
        )
        object.__setattr__(self, "_max_abs_delta_ns", threshold_ns)

    @property
    def max_abs_delta_ns(self) -> int:
        """Return the inclusive pairing threshold in nanoseconds."""
        return self._max_abs_delta_ns


@dataclass(frozen=True)
class FramePair:
    """Describe one accepted RGB/aligned-depth timestamp pair."""

    rgb_index: int
    depth_index: int
    rgb_timestamp_ns: int
    depth_timestamp_ns: int

    def __post_init__(self) -> None:
        """Normalize integer fields and reject invalid indices or timestamps."""
        object.__setattr__(
            self,
            "rgb_index",
            _normalize_non_negative_integer(self.rgb_index, "rgb_index"),
        )
        object.__setattr__(
            self,
            "depth_index",
            _normalize_non_negative_integer(self.depth_index, "depth_index"),
        )
        object.__setattr__(
            self,
            "rgb_timestamp_ns",
            _normalize_positive_timestamp(
                self.rgb_timestamp_ns,
                "rgb_timestamp_ns",
            ),
        )
        object.__setattr__(
            self,
            "depth_timestamp_ns",
            _normalize_positive_timestamp(
                self.depth_timestamp_ns,
                "depth_timestamp_ns",
            ),
        )

    @property
    def delta_ns(self) -> int:
        """Return signed ``depth - RGB`` timestamp delta in nanoseconds."""
        return self.depth_timestamp_ns - self.rgb_timestamp_ns

    @property
    def delta_ms(self) -> float:
        """Return signed ``depth - RGB`` timestamp delta in milliseconds."""
        return self.delta_ns / NANOSECONDS_PER_MILLISECOND


@dataclass(frozen=True)
class FramePairingResult:
    """Store accepted pairs and derive all unmatched frame indices."""

    config: FramePairingConfig
    rgb_frame_count: int
    depth_frame_count: int
    pairs: tuple[FramePair, ...]

    def __post_init__(self) -> None:
        """Validate pair ordering, bounds, uniqueness, and threshold."""
        if not isinstance(self.config, FramePairingConfig):
            raise TypeError(
                "config must be a FramePairingConfig; got "
                f"{type(self.config).__name__}"
            )
        object.__setattr__(
            self,
            "rgb_frame_count",
            _normalize_positive_integer(
                self.rgb_frame_count,
                "rgb_frame_count",
            ),
        )
        object.__setattr__(
            self,
            "depth_frame_count",
            _normalize_positive_integer(
                self.depth_frame_count,
                "depth_frame_count",
            ),
        )
        if not isinstance(self.pairs, tuple):
            raise TypeError(f"pairs must be a tuple; got {type(self.pairs).__name__}")

        previous_rgb_index = -1
        previous_depth_index = -1
        previous_rgb_timestamp_ns = 0
        previous_depth_timestamp_ns = 0
        for pair in self.pairs:
            if not isinstance(pair, FramePair):
                raise TypeError(
                    "pairs must contain only FramePair values; got "
                    f"{type(pair).__name__}"
                )
            if pair.rgb_index >= self.rgb_frame_count:
                raise ValueError(
                    f"rgb_index {pair.rgb_index} is outside rgb_frame_count "
                    f"{self.rgb_frame_count}"
                )
            if pair.depth_index >= self.depth_frame_count:
                raise ValueError(
                    f"depth_index {pair.depth_index} is outside "
                    f"depth_frame_count {self.depth_frame_count}"
                )
            if pair.rgb_index <= previous_rgb_index:
                raise ValueError("paired RGB indices must be strictly increasing")
            if pair.depth_index <= previous_depth_index:
                raise ValueError("paired depth indices must be strictly increasing")
            if pair.rgb_timestamp_ns <= previous_rgb_timestamp_ns:
                raise ValueError("paired RGB timestamps must be strictly increasing")
            if pair.depth_timestamp_ns <= previous_depth_timestamp_ns:
                raise ValueError("paired depth timestamps must be strictly increasing")
            if abs(pair.delta_ns) > self.config.max_abs_delta_ns:
                raise ValueError(
                    f"Pair delta {pair.delta_ns} ns exceeds inclusive threshold "
                    f"{self.config.max_abs_delta_ns} ns"
                )
            previous_rgb_index = pair.rgb_index
            previous_depth_index = pair.depth_index
            previous_rgb_timestamp_ns = pair.rgb_timestamp_ns
            previous_depth_timestamp_ns = pair.depth_timestamp_ns

    @property
    def accepted_pair_count(self) -> int:
        """Return the number of accepted pairs."""
        return len(self.pairs)

    @property
    def paired_rgb_indices(self) -> tuple[int, ...]:
        """Return accepted RGB indices in pairing order."""
        return tuple(pair.rgb_index for pair in self.pairs)

    @property
    def paired_depth_indices(self) -> tuple[int, ...]:
        """Return accepted depth indices in pairing order."""
        return tuple(pair.depth_index for pair in self.pairs)

    @property
    def rejected_rgb_indices(self) -> tuple[int, ...]:
        """Return RGB indices not present in an accepted pair."""
        paired = set(self.paired_rgb_indices)
        return tuple(
            index for index in range(self.rgb_frame_count) if index not in paired
        )

    @property
    def unmatched_depth_indices(self) -> tuple[int, ...]:
        """Return depth indices not present in an accepted pair."""
        paired = set(self.paired_depth_indices)
        return tuple(
            index for index in range(self.depth_frame_count) if index not in paired
        )

    @property
    def rejected_rgb_count(self) -> int:
        """Return the number of RGB frames without an accepted pair."""
        return self.rgb_frame_count - self.accepted_pair_count

    @property
    def unmatched_depth_count(self) -> int:
        """Return the number of depth frames without an accepted pair."""
        return self.depth_frame_count - self.accepted_pair_count


def pair_frames_by_timestamp(
    rgb_timestamp_ns: np.ndarray,
    depth_timestamp_ns: np.ndarray,
    config: FramePairingConfig | None = None,
) -> FramePairingResult:
    """Validate pairing inputs; nearest matching is implemented in Step 2.2."""
    _validate_timestamp_array("rgb_timestamp_ns", rgb_timestamp_ns)
    _validate_timestamp_array("depth_timestamp_ns", depth_timestamp_ns)
    if config is not None and not isinstance(config, FramePairingConfig):
        raise TypeError(
            "config must be a FramePairingConfig or None; got "
            f"{type(config).__name__}"
        )
    raise NotImplementedError(
        "Nearest timestamp matching belongs to alignment Phase 2 Step 2.2"
    )


def _validate_timestamp_array(name: str, timestamps: np.ndarray) -> None:
    if not isinstance(timestamps, np.ndarray):
        raise TypeError(
            f"{name} must be a numpy.ndarray; got {type(timestamps).__name__}"
        )
    if timestamps.ndim != 1:
        raise ValueError(f"{name} must have shape (N,); got shape {timestamps.shape}")
    if timestamps.dtype != np.int64:
        raise ValueError(f"{name} must have dtype int64; got {timestamps.dtype}")
    if timestamps.size == 0:
        raise ValueError(f"{name} must contain at least one timestamp")
    if np.any(timestamps <= 0):
        raise ValueError(f"{name} values must be positive")
    if timestamps.size > 1 and np.any(np.diff(timestamps) <= 0):
        raise ValueError(f"{name} values must be strictly increasing")


def _normalize_non_negative_integer(value: object, name: str) -> int:
    if (
        not isinstance(value, Integral)
        or isinstance(value, (bool, np.bool_))
        or int(value) < 0
    ):
        raise ValueError(f"{name} must be a non-negative integer")
    return int(value)


def _normalize_positive_integer(value: object, name: str) -> int:
    if (
        not isinstance(value, Integral)
        or isinstance(value, (bool, np.bool_))
        or int(value) <= 0
    ):
        raise ValueError(f"{name} must be a positive integer")
    return int(value)


def _normalize_positive_timestamp(value: object, name: str) -> int:
    timestamp = _normalize_positive_integer(value, name)
    if timestamp > np.iinfo(np.int64).max:
        raise ValueError(f"{name} exceeds the int64 timestamp range")
    return timestamp
