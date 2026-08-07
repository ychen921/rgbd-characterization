"""Contract tests for RGB/aligned-depth timestamp pairing."""

from dataclasses import FrozenInstanceError

import numpy as np
import pytest

from src.preprocessing.frame_pairing import (
    FramePair,
    FramePairingConfig,
    FramePairingResult,
    pair_frames_by_timestamp,
)


def make_pair(
    rgb_index: int,
    depth_index: int,
    *,
    rgb_timestamp_ns: int | None = None,
    delta_ns: int = 1_000_000,
) -> FramePair:
    if rgb_timestamp_ns is None:
        rgb_timestamp_ns = 1_000_000_000 + rgb_index * 100_000_000
    return FramePair(
        rgb_index=rgb_index,
        depth_index=depth_index,
        rgb_timestamp_ns=rgb_timestamp_ns,
        depth_timestamp_ns=rgb_timestamp_ns + delta_ns,
    )


def test_default_config_fixes_schema_v1_pairing_policy() -> None:
    config = FramePairingConfig()

    assert config.SCHEMA_VERSION == 1
    assert config.METHOD == "nearest_timestamp"
    assert config.TIMESTAMP_SOURCE == "message_header"
    assert config.DELTA_DEFINITION == "depth_minus_rgb"
    assert config.THRESHOLD_INCLUSIVE is True
    assert config.CARDINALITY == "one_to_one"
    assert config.PRESERVE_ORDER is True
    assert config.TIE_BREAKER == "earlier_depth"
    assert config.max_abs_delta_ms == 20.0
    assert config.max_abs_delta_ns == 20_000_000


@pytest.mark.parametrize(
    ("threshold_ms", "expected_ns"),
    [
        (0.0, 0),
        (0.000001, 1),
        (4.1, 4_100_000),
        (np.float64(20.0), 20_000_000),
    ],
)
def test_config_normalizes_threshold_to_integer_nanoseconds(
    threshold_ms: float,
    expected_ns: int,
) -> None:
    config = FramePairingConfig(max_abs_delta_ms=threshold_ms)

    assert config.max_abs_delta_ns == expected_ns
    assert config.max_abs_delta_ms == expected_ns / 1_000_000


@pytest.mark.parametrize(
    ("threshold", "expected_message"),
    [
        (True, "finite real"),
        (np.bool_(False), "finite real"),
        ("20", "finite real"),
        (float("nan"), "finite real"),
        (float("inf"), "finite real"),
        (-0.1, "non-negative"),
        (0.0000001, "integer number of nanoseconds"),
        (1.0e20, "int64 timestamp range"),
        (1.0e308, "int64 timestamp range"),
    ],
)
def test_config_rejects_invalid_threshold(
    threshold: object,
    expected_message: str,
) -> None:
    with pytest.raises(ValueError, match=expected_message):
        FramePairingConfig(max_abs_delta_ms=threshold)


def test_contract_models_are_frozen() -> None:
    config = FramePairingConfig()
    pair = make_pair(0, 0)
    result = FramePairingResult(config, 1, 1, (pair,))

    with pytest.raises(FrozenInstanceError):
        config.max_abs_delta_ms = 1.0
    with pytest.raises(FrozenInstanceError):
        pair.rgb_index = 1
    with pytest.raises(FrozenInstanceError):
        result.pairs = ()


@pytest.mark.parametrize(
    ("delta_ns", "expected_delta_ms"),
    [
        (2_500_000, 2.5),
        (-1_250_000, -1.25),
        (0, 0.0),
    ],
)
def test_frame_pair_derives_signed_depth_minus_rgb_delta(
    delta_ns: int,
    expected_delta_ms: float,
) -> None:
    pair = make_pair(0, 0, delta_ns=delta_ns)

    assert pair.delta_ns == delta_ns
    assert pair.delta_ms == expected_delta_ms


def test_frame_pair_normalizes_numpy_integers() -> None:
    pair = FramePair(
        rgb_index=np.int64(0),
        depth_index=np.int32(1),
        rgb_timestamp_ns=np.int64(1_000_000_000),
        depth_timestamp_ns=np.int64(1_000_000_001),
    )

    assert type(pair.rgb_index) is int
    assert type(pair.depth_index) is int
    assert type(pair.rgb_timestamp_ns) is int
    assert type(pair.depth_timestamp_ns) is int


@pytest.mark.parametrize(
    ("field", "value", "expected_message"),
    [
        ("rgb_index", -1, "rgb_index"),
        ("depth_index", 0.5, "depth_index"),
        ("rgb_index", True, "rgb_index"),
        ("depth_index", np.bool_(False), "depth_index"),
        ("rgb_timestamp_ns", 0, "rgb_timestamp_ns"),
        ("depth_timestamp_ns", -1, "depth_timestamp_ns"),
        ("rgb_timestamp_ns", 1.5, "rgb_timestamp_ns"),
        ("depth_timestamp_ns", True, "depth_timestamp_ns"),
        (
            "depth_timestamp_ns",
            int(np.iinfo(np.int64).max) + 1,
            "int64 timestamp range",
        ),
    ],
)
def test_frame_pair_rejects_invalid_fields(
    field: str,
    value: object,
    expected_message: str,
) -> None:
    arguments: dict[str, object] = {
        "rgb_index": 0,
        "depth_index": 0,
        "rgb_timestamp_ns": 1_000_000_000,
        "depth_timestamp_ns": 1_000_000_001,
    }
    arguments[field] = value

    with pytest.raises(ValueError, match=expected_message):
        FramePair(**arguments)


def test_result_derives_accepted_rejected_and_unmatched_indices() -> None:
    pairs = (make_pair(0, 0), make_pair(2, 1), make_pair(3, 3))
    result = FramePairingResult(
        config=FramePairingConfig(),
        rgb_frame_count=np.int64(5),
        depth_frame_count=np.int64(6),
        pairs=pairs,
    )

    assert result.accepted_pair_count == 3
    assert result.paired_rgb_indices == (0, 2, 3)
    assert result.paired_depth_indices == (0, 1, 3)
    assert result.rejected_rgb_indices == (1, 4)
    assert result.unmatched_depth_indices == (2, 4, 5)
    assert result.rejected_rgb_count == 2
    assert result.unmatched_depth_count == 3
    assert type(result.rgb_frame_count) is int
    assert type(result.depth_frame_count) is int


def test_result_allows_no_accepted_pairs() -> None:
    result = FramePairingResult(FramePairingConfig(), 2, 3, ())

    assert result.accepted_pair_count == 0
    assert result.rejected_rgb_indices == (0, 1)
    assert result.unmatched_depth_indices == (0, 1, 2)


def test_result_threshold_is_inclusive() -> None:
    result = FramePairingResult(
        FramePairingConfig(max_abs_delta_ms=2.0),
        1,
        1,
        (make_pair(0, 0, delta_ns=-2_000_000),),
    )

    assert result.accepted_pair_count == 1


@pytest.mark.parametrize(
    ("arguments", "exception", "expected_message"),
    [
        (
            {"config": None, "rgb_frame_count": 1, "depth_frame_count": 1},
            TypeError,
            "FramePairingConfig",
        ),
        (
            {
                "config": FramePairingConfig(),
                "rgb_frame_count": 0,
                "depth_frame_count": 1,
            },
            ValueError,
            "rgb_frame_count",
        ),
        (
            {
                "config": FramePairingConfig(),
                "rgb_frame_count": 1,
                "depth_frame_count": True,
            },
            ValueError,
            "depth_frame_count",
        ),
    ],
)
def test_result_rejects_invalid_configuration_or_counts(
    arguments: dict[str, object],
    exception: type[Exception],
    expected_message: str,
) -> None:
    with pytest.raises(exception, match=expected_message):
        FramePairingResult(pairs=(), **arguments)


def test_result_requires_immutable_pair_tuple() -> None:
    with pytest.raises(TypeError, match="pairs must be a tuple"):
        FramePairingResult(FramePairingConfig(), 1, 1, [make_pair(0, 0)])


def test_result_rejects_non_pair_member() -> None:
    with pytest.raises(TypeError, match="FramePair"):
        FramePairingResult(FramePairingConfig(), 1, 1, ("pair",))


@pytest.mark.parametrize(
    ("pairs", "rgb_count", "depth_count", "expected_message"),
    [
        ((make_pair(1, 0),), 1, 1, "outside rgb_frame_count"),
        ((make_pair(0, 1),), 1, 1, "outside depth_frame_count"),
        ((make_pair(1, 0), make_pair(0, 1)), 2, 2, "RGB indices"),
        ((make_pair(0, 1), make_pair(1, 0)), 2, 2, "depth indices"),
    ],
)
def test_result_rejects_out_of_bounds_or_non_monotonic_indices(
    pairs: tuple[FramePair, ...],
    rgb_count: int,
    depth_count: int,
    expected_message: str,
) -> None:
    with pytest.raises(ValueError, match=expected_message):
        FramePairingResult(
            FramePairingConfig(),
            rgb_count,
            depth_count,
            pairs,
        )


@pytest.mark.parametrize(
    ("pairs", "expected_message"),
    [
        (
            (
                make_pair(0, 0, rgb_timestamp_ns=2_000_000_000),
                make_pair(1, 1, rgb_timestamp_ns=1_000_000_000),
            ),
            "RGB timestamps",
        ),
        (
            (
                make_pair(
                    0,
                    0,
                    rgb_timestamp_ns=1_000_000_000,
                    delta_ns=1_000_000_000,
                ),
                make_pair(
                    1,
                    1,
                    rgb_timestamp_ns=1_100_000_000,
                    delta_ns=1_000_000,
                ),
            ),
            "depth timestamps",
        ),
    ],
)
def test_result_rejects_non_monotonic_pair_timestamps(
    pairs: tuple[FramePair, ...],
    expected_message: str,
) -> None:
    with pytest.raises(ValueError, match=expected_message):
        FramePairingResult(
            FramePairingConfig(max_abs_delta_ms=2_000.0),
            2,
            2,
            pairs,
        )


def test_result_rejects_pair_outside_threshold() -> None:
    with pytest.raises(ValueError, match="exceeds inclusive threshold"):
        FramePairingResult(
            FramePairingConfig(max_abs_delta_ms=2.0),
            1,
            1,
            (make_pair(0, 0, delta_ns=2_000_001),),
        )


def valid_timestamps() -> tuple[np.ndarray, np.ndarray]:
    return (
        np.array([1_000_000_000, 1_100_000_000], dtype=np.int64),
        np.array([1_000_001_000, 1_100_001_000], dtype=np.int64),
    )


@pytest.mark.parametrize(
    ("stream", "invalid", "exception", "expected_message"),
    [
        ("rgb", [1], TypeError, "numpy.ndarray"),
        ("depth", np.ones((1, 1), dtype=np.int64), ValueError, "shape"),
        ("rgb", np.array([1], dtype=np.int32), ValueError, "dtype int64"),
        ("depth", np.array([], dtype=np.int64), ValueError, "at least one"),
        ("rgb", np.array([0], dtype=np.int64), ValueError, "positive"),
        (
            "depth",
            np.array([2, 1], dtype=np.int64),
            ValueError,
            "strictly increasing",
        ),
        (
            "rgb",
            np.array([1, 1], dtype=np.int64),
            ValueError,
            "strictly increasing",
        ),
    ],
)
def test_pairing_entrypoint_rejects_invalid_timestamp_arrays(
    stream: str,
    invalid: object,
    exception: type[Exception],
    expected_message: str,
) -> None:
    rgb, depth = valid_timestamps()
    if stream == "rgb":
        rgb = invalid
    else:
        depth = invalid

    with pytest.raises(exception, match=expected_message):
        pair_frames_by_timestamp(rgb, depth)


def test_pairing_entrypoint_rejects_invalid_config() -> None:
    rgb, depth = valid_timestamps()

    with pytest.raises(TypeError, match="FramePairingConfig or None"):
        pair_frames_by_timestamp(rgb, depth, config={})


def test_pairing_entrypoint_marks_step_2_2_algorithm_boundary() -> None:
    rgb, depth = valid_timestamps()
    original_rgb = rgb.copy()
    original_depth = depth.copy()

    with pytest.raises(NotImplementedError, match="Step 2.2"):
        pair_frames_by_timestamp(rgb, depth, FramePairingConfig())

    assert np.array_equal(rgb, original_rgb)
    assert np.array_equal(depth, original_depth)
