"""Extract unpaired RGB and aligned depth from one Phase 0 experiment bag."""

from __future__ import annotations

import argparse
import shutil
import sys
import tempfile
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import yaml


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.io.alignment_bag_reader import (
    AlignmentBagReader,
    AlignmentBagContract,
    CameraInfoSnapshot,
)
from src.io.alignment_dataset import AlignmentDataset


EXPERIMENT_FILENAME = "experiment.yaml"
CAMERA_PARAMS_FILENAME = "camera_params.yaml"
PREFLIGHT_FILENAME = "preflight.yaml"
POST_RECORDING_FILENAME = "post_recording.yaml"
COLOR_CAMERA_INFO_FILENAME = "color_camera_info.yaml"
DEPTH_CAMERA_INFO_FILENAME = "depth_camera_info.yaml"
SUMMARY_FILENAME = "extraction_summary.yaml"
ROSBAG_DIRECTORY = "rosbag"

REQUIRED_SOURCE_FILENAMES = (
    EXPERIMENT_FILENAME,
    CAMERA_PARAMS_FILENAME,
    PREFLIGHT_FILENAME,
    POST_RECORDING_FILENAME,
)


@dataclass(frozen=True)
class ExperimentContract:
    """Metadata values needed to open and validate an alignment bag."""

    name: str
    color_topic: str
    aligned_depth_topic: str
    width: int
    height: int


@dataclass(frozen=True)
class ExtractionSummary:
    """High-level result returned after an atomic extraction."""

    experiment: str
    output_directory: Path
    rgb_frames: int
    depth_frames: int
    width: int
    height: int


ReaderFactory = Callable[..., AlignmentBagReader]


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description=(
            "Extract unpaired RGB and SDK-aligned depth from one Phase 0 "
            "experiment directory."
        )
    )
    parser.add_argument(
        "experiment_dir",
        type=Path,
        help="Experiment directory containing metadata YAML files and rosbag/.",
    )
    parser.add_argument(
        "output_dir",
        type=Path,
        help="New output directory for the alignment dataset.",
    )
    return parser.parse_args(argv)


def extract_alignment_dataset(
    experiment_dir: Path,
    output_dir: Path,
    *,
    reader_factory: ReaderFactory = AlignmentBagReader,
) -> ExtractionSummary:
    """Validate, extract, reload-check, and atomically publish one dataset."""
    source_dir = Path(experiment_dir).expanduser()
    destination = Path(output_dir).expanduser()
    documents = _load_and_validate_source_metadata(source_dir)
    experiment = _validate_experiment_contract(source_dir, documents)
    bag_path = source_dir / ROSBAG_DIRECTORY

    if destination.exists():
        raise FileExistsError(f"Alignment dataset output already exists: {destination}")
    destination.parent.mkdir(parents=True, exist_ok=True)

    reader = reader_factory(
        bag_path=bag_path,
        color_topic=experiment.color_topic,
        aligned_depth_topic=experiment.aligned_depth_topic,
    )
    bag_contract = reader.inspect()
    _validate_bag_against_experiment(bag_contract, experiment)

    staging_path = Path(
        tempfile.mkdtemp(
            prefix=f".{destination.name}.staging-",
            dir=destination.parent,
        )
    )
    try:
        dataset = _read_dataset(reader, bag_contract)
        summary_document = _summary_document(
            source_dir=source_dir,
            destination=destination,
            experiment=experiment,
            bag_contract=bag_contract,
            dataset=dataset,
            preflight=documents[PREFLIGHT_FILENAME],
            post_recording=documents[POST_RECORDING_FILENAME],
        )

        dataset.save(staging_path)
        _write_camera_info(
            staging_path / COLOR_CAMERA_INFO_FILENAME,
            topic=reader.color_info_topic,
            stream="color",
            snapshot=bag_contract.color_camera_info,
        )
        _write_camera_info(
            staging_path / DEPTH_CAMERA_INFO_FILENAME,
            topic=reader.depth_info_topic,
            stream="aligned_depth",
            snapshot=bag_contract.depth_camera_info,
        )
        for filename in REQUIRED_SOURCE_FILENAMES:
            shutil.copy2(source_dir / filename, staging_path / filename)
        _write_yaml(staging_path / SUMMARY_FILENAME, summary_document)

        result = ExtractionSummary(
            experiment=experiment.name,
            output_directory=destination,
            rgb_frames=dataset.num_rgb_frames,
            depth_frames=dataset.num_depth_frames,
            width=dataset.width,
            height=dataset.height,
        )

        # Drop the extraction arrays before the reload check to avoid holding two
        # complete 1280x720 datasets at once.
        del dataset
        reloaded = AlignmentDataset.load(staging_path)
        _validate_reloaded_dataset(reloaded, result)
        del reloaded

        staging_path.replace(destination)
        return result
    except BaseException:
        shutil.rmtree(staging_path, ignore_errors=True)
        raise


def _load_and_validate_source_metadata(
    source_dir: Path,
) -> dict[str, Mapping[str, Any]]:
    if not source_dir.exists():
        raise FileNotFoundError(f"Experiment directory does not exist: {source_dir}")
    if not source_dir.is_dir():
        raise NotADirectoryError(f"Experiment path is not a directory: {source_dir}")

    bag_path = source_dir / ROSBAG_DIRECTORY
    if not bag_path.is_dir() or not (bag_path / "metadata.yaml").is_file():
        raise ValueError(
            f"Experiment does not contain rosbag/metadata.yaml: {source_dir}"
        )

    documents: dict[str, Mapping[str, Any]] = {}
    for filename in REQUIRED_SOURCE_FILENAMES:
        path = source_dir / filename
        if not path.is_file():
            raise FileNotFoundError(f"Required experiment metadata is missing: {path}")
        try:
            with path.open("r", encoding="utf-8") as stream:
                document = yaml.safe_load(stream)
        except yaml.YAMLError as exc:
            raise ValueError(f"Invalid YAML metadata: {path}") from exc
        if not isinstance(document, Mapping):
            raise ValueError(f"Metadata document must be a mapping: {path}")
        documents[filename] = document

    _validate_preflight(documents[PREFLIGHT_FILENAME])
    _validate_post_recording(documents[POST_RECORDING_FILENAME])
    _validate_runtime_camera_parameters(documents[CAMERA_PARAMS_FILENAME])
    return documents


def _validate_experiment_contract(
    source_dir: Path,
    documents: Mapping[str, Mapping[str, Any]],
) -> ExperimentContract:
    document = documents[EXPERIMENT_FILENAME]
    _require_equal(document, ("schema_version",), 1)
    experiment = _require_mapping(document, ("experiment",))
    name = _require_string(experiment, ("name",))
    if name != source_dir.name:
        raise ValueError(
            f"Experiment name {name!r} does not match directory {source_dir.name!r}"
        )
    _require_equal(experiment, ("type",), "rgb_depth_alignment")

    registration = _require_mapping(document, ("registration",))
    _require_equal(registration, ("enabled",), True)
    _require_equal(registration, ("mode",), "sdk")
    _require_equal(registration, ("aligned_to",), "color")
    _require_equal(
        registration,
        ("coordinate_convention",),
        "color_pixel_grid",
    )
    wrapper_align_mode = _value_at(registration, ("wrapper_align_mode",))
    if wrapper_align_mode not in {"SW", "HW"}:
        raise ValueError(
            "Metadata field wrapper_align_mode must be 'SW' or 'HW'; "
            f"got {wrapper_align_mode!r}"
        )
    runtime_parameters = _camera_parameter_mapping(documents[CAMERA_PARAMS_FILENAME])
    _require_equal(runtime_parameters, ("align_mode",), wrapper_align_mode)

    color = _require_mapping(document, ("color",))
    depth = _require_mapping(document, ("depth",))
    color_topic = _require_absolute_topic(color, "topic")
    depth_topic = _require_absolute_topic(depth, "topic")
    _require_equal(registration, ("aligned_depth_topic",), depth_topic)
    _require_equal(color, ("width",), 1280)
    _require_equal(color, ("height",), 720)
    _require_equal(color, ("encoding",), AlignmentDataset.COLOR_ENCODING)
    _require_equal(depth, ("width",), 1280)
    _require_equal(depth, ("height",), 720)
    _require_equal(depth, ("encoding",), AlignmentDataset.DEPTH_ENCODING)
    _require_equal(depth, ("precision",), AlignmentDataset.DEPTH_PRECISION)
    _require_equal(depth, ("unit",), AlignmentDataset.DEPTH_UNIT)
    invalid_values = depth.get("invalid_values")
    if invalid_values != list(AlignmentDataset.DEPTH_INVALID_VALUES):
        raise ValueError(
            "Metadata field depth.invalid_values must equal "
            f"{list(AlignmentDataset.DEPTH_INVALID_VALUES)!r}; got "
            f"{invalid_values!r}"
        )

    timestamps = _require_mapping(document, ("timestamps",))
    _require_equal(
        timestamps,
        ("source",),
        AlignmentDataset.PRIMARY_TIMESTAMP_SOURCE,
    )
    _require_equal(timestamps, ("unit",), AlignmentDataset.TIMESTAMP_UNIT)
    _require_equal(timestamps, ("clock_domain",), "global")

    preflight = documents[PREFLIGHT_FILENAME]
    post_recording = documents[POST_RECORDING_FILENAME]
    _require_equal(preflight, ("experiment",), name)
    _require_equal(post_recording, ("experiment",), name)
    recorded_topics = post_recording.get("recorded_topics")
    if not isinstance(recorded_topics, list) or not all(
        isinstance(topic, str) for topic in recorded_topics
    ):
        raise ValueError("post_recording recorded_topics must be a list of strings")
    required_topics = {
        color_topic,
        depth_topic,
        AlignmentBagReader.DEFAULT_COLOR_INFO_TOPIC,
        AlignmentBagReader.DEFAULT_DEPTH_INFO_TOPIC,
    }
    missing_topics = required_topics.difference(recorded_topics)
    if missing_topics:
        raise ValueError(
            "post_recording recorded_topics is missing required topic(s): "
            + ", ".join(sorted(missing_topics))
        )

    return ExperimentContract(
        name=name,
        color_topic=color_topic,
        aligned_depth_topic=depth_topic,
        width=1280,
        height=720,
    )


def _validate_preflight(document: Mapping[str, Any]) -> None:
    _require_equal(document, ("schema_version",), 1)
    result = document.get("overall_result")
    if result not in {"pass", "warn"}:
        raise ValueError(
            "Metadata field overall_result must be 'pass' or 'warn'; " f"got {result!r}"
        )
    errors = document.get("errors")
    if not isinstance(errors, list) or errors:
        raise ValueError("Preflight errors must be an empty list")
    checks = document.get("checks")
    if not isinstance(checks, list):
        raise ValueError("Preflight checks must be a list")
    for check in checks:
        if not isinstance(check, Mapping):
            raise ValueError("Each preflight check must be a mapping")
        if check.get("status") == "fail":
            raise ValueError(
                f"Preflight check {check.get('name', '<unnamed>')!r} failed"
            )


def _validate_post_recording(document: Mapping[str, Any]) -> None:
    _require_equal(document, ("schema_version",), 1)
    _require_equal(document, ("status",), "success")


def _validate_runtime_camera_parameters(document: Mapping[str, Any]) -> None:
    parameters = _camera_parameter_mapping(document)
    _require_equal(parameters, ("depth_registration",), True)
    _require_equal(parameters, ("align_target_stream",), "COLOR")
    _require_equal(parameters, ("depth_precision",), "1mm")
    align_mode = parameters.get("align_mode")
    if align_mode not in {"SW", "HW"}:
        raise ValueError(
            "Metadata field align_mode must be 'SW' or 'HW'; " f"got {align_mode!r}"
        )


def _camera_parameter_mapping(document: Mapping[str, Any]) -> Mapping[str, Any]:
    candidates: list[Mapping[str, Any]] = []
    for node in document.values():
        if not isinstance(node, Mapping):
            continue
        parameters = node.get("ros__parameters")
        if isinstance(parameters, Mapping) and "depth_registration" in parameters:
            candidates.append(parameters)
    if len(candidates) != 1:
        raise ValueError(
            "camera_params.yaml must contain exactly one ROS parameter mapping "
            "with depth_registration"
        )
    return candidates[0]


def _validate_bag_against_experiment(
    bag: AlignmentBagContract,
    experiment: ExperimentContract,
) -> None:
    if bag.color.topic != experiment.color_topic:
        raise ValueError("Bag color topic does not match experiment metadata")
    if bag.aligned_depth.topic != experiment.aligned_depth_topic:
        raise ValueError("Bag aligned-depth topic does not match experiment metadata")
    for label, stream in (("color", bag.color), ("aligned depth", bag.aligned_depth)):
        if stream.width != experiment.width or stream.height != experiment.height:
            raise ValueError(
                f"Bag {label} dimensions are {stream.width}x{stream.height}; "
                f"expected {experiment.width}x{experiment.height}"
            )
    if bag.color.encoding != AlignmentDataset.COLOR_ENCODING:
        raise ValueError("Bag color encoding does not match the dataset contract")
    if bag.aligned_depth.encoding != AlignmentDataset.DEPTH_ENCODING:
        raise ValueError(
            "Bag aligned-depth encoding does not match the dataset contract"
        )


def _read_dataset(
    reader: AlignmentBagReader,
    contract: AlignmentBagContract,
) -> AlignmentDataset:
    color = contract.color
    depth = contract.aligned_depth
    rgb = np.empty(
        (color.message_count, color.height, color.width, 3),
        dtype=np.uint8,
    )
    aligned_depth = np.empty(
        (depth.message_count, depth.height, depth.width),
        dtype=np.uint16,
    )
    rgb_timestamp_ns = np.empty(color.message_count, dtype=np.int64)
    depth_timestamp_ns = np.empty(depth.message_count, dtype=np.int64)
    rgb_recorded_timestamp_ns = np.empty(color.message_count, dtype=np.int64)
    depth_recorded_timestamp_ns = np.empty(depth.message_count, dtype=np.int64)

    color_count = 0
    for color_count, frame in enumerate(reader.read_color_frames(), start=1):
        if color_count > color.message_count:
            raise ValueError("Reader returned more color frames than declared")
        index = color_count - 1
        rgb[index] = frame.image
        rgb_timestamp_ns[index] = frame.header_timestamp_ns
        rgb_recorded_timestamp_ns[index] = frame.recorded_timestamp_ns
    if color_count != color.message_count:
        raise ValueError(
            f"Reader returned {color_count} color frames; expected "
            f"{color.message_count}"
        )

    depth_count = 0
    for depth_count, frame in enumerate(
        reader.read_aligned_depth_frames(),
        start=1,
    ):
        if depth_count > depth.message_count:
            raise ValueError("Reader returned more aligned-depth frames than declared")
        index = depth_count - 1
        aligned_depth[index] = frame.image
        depth_timestamp_ns[index] = frame.header_timestamp_ns
        depth_recorded_timestamp_ns[index] = frame.recorded_timestamp_ns
    if depth_count != depth.message_count:
        raise ValueError(
            f"Reader returned {depth_count} aligned-depth frames; expected "
            f"{depth.message_count}"
        )

    return AlignmentDataset(
        rgb=rgb,
        aligned_depth=aligned_depth,
        rgb_timestamp_ns=rgb_timestamp_ns,
        depth_timestamp_ns=depth_timestamp_ns,
        rgb_recorded_timestamp_ns=rgb_recorded_timestamp_ns,
        depth_recorded_timestamp_ns=depth_recorded_timestamp_ns,
    )


def _summary_document(
    *,
    source_dir: Path,
    destination: Path,
    experiment: ExperimentContract,
    bag_contract: AlignmentBagContract,
    dataset: AlignmentDataset,
    preflight: Mapping[str, Any],
    post_recording: Mapping[str, Any],
) -> dict[str, Any]:
    return {
        "schema_version": 1,
        "status": "success",
        "experiment": experiment.name,
        "source": {
            "experiment_directory": str(source_dir.resolve()),
            "bag_directory": str((source_dir / ROSBAG_DIRECTORY).resolve()),
        },
        "output_directory": str(destination.resolve()),
        "dataset_schema_version": AlignmentDataset.SCHEMA_VERSION,
        "streams": {
            "color": _stream_summary(
                topic=bag_contract.color.topic,
                frame_id=bag_contract.color.frame_id,
                encoding=bag_contract.color.encoding,
                frames=dataset.num_rgb_frames,
                shape=list(dataset.rgb.shape),
                dtype=str(dataset.rgb.dtype),
                header=dataset.rgb_timestamp_ns,
                recorded=dataset.rgb_recorded_timestamp_ns,
            ),
            "aligned_depth": _stream_summary(
                topic=bag_contract.aligned_depth.topic,
                frame_id=bag_contract.aligned_depth.frame_id,
                encoding=bag_contract.aligned_depth.encoding,
                frames=dataset.num_depth_frames,
                shape=list(dataset.aligned_depth.shape),
                dtype=str(dataset.aligned_depth.dtype),
                header=dataset.depth_timestamp_ns,
                recorded=dataset.depth_recorded_timestamp_ns,
            ),
        },
        "source_validation": {
            "preflight_result": preflight["overall_result"],
            "post_recording_status": post_recording["status"],
        },
        "artifacts": {
            "rgb": AlignmentDataset.RGB_FILENAME,
            "aligned_depth": AlignmentDataset.DEPTH_FILENAME,
            "timestamps": AlignmentDataset.TIMESTAMPS_FILENAME,
            "color_camera_info": COLOR_CAMERA_INFO_FILENAME,
            "depth_camera_info": DEPTH_CAMERA_INFO_FILENAME,
            "source_metadata": list(REQUIRED_SOURCE_FILENAMES),
        },
    }


def _stream_summary(
    *,
    topic: str,
    frame_id: str,
    encoding: str,
    frames: int,
    shape: list[int],
    dtype: str,
    header: np.ndarray,
    recorded: np.ndarray,
) -> dict[str, Any]:
    latency = recorded - header
    return {
        "topic": topic,
        "frame_id": frame_id,
        "encoding": encoding,
        "frames": frames,
        "shape": shape,
        "dtype": dtype,
        "header_timestamp_ns": _timestamp_summary(header),
        "recorded_timestamp_ns": _timestamp_summary(recorded),
        "recorded_minus_header_ns": {
            "median": float(np.median(latency)),
            "p95": float(np.percentile(latency, 95)),
        },
    }


def _timestamp_summary(timestamps: np.ndarray) -> dict[str, Any]:
    first = int(timestamps[0])
    last = int(timestamps[-1])
    span = last - first
    fps = None
    if timestamps.size > 1 and span > 0:
        fps = float((timestamps.size - 1) * 1_000_000_000 / span)
    return {
        "first": first,
        "last": last,
        "span": span,
        "nominal_fps": fps,
    }


def _write_camera_info(
    path: Path,
    *,
    topic: str,
    stream: str,
    snapshot: CameraInfoSnapshot,
) -> None:
    roi_x, roi_y, roi_width, roi_height, do_rectify = snapshot.roi
    document = {
        "schema_version": 1,
        "stream": stream,
        "topic": topic,
        "frame_id": snapshot.frame_id,
        "width": snapshot.width,
        "height": snapshot.height,
        "distortion_model": snapshot.distortion_model,
        "d": list(snapshot.d),
        "k": list(snapshot.k),
        "r": list(snapshot.r),
        "p": list(snapshot.p),
        "binning_x": snapshot.binning_x,
        "binning_y": snapshot.binning_y,
        "roi": {
            "x_offset": roi_x,
            "y_offset": roi_y,
            "width": roi_width,
            "height": roi_height,
            "do_rectify": do_rectify,
        },
    }
    _write_yaml(path, document)


def _write_yaml(path: Path, document: Mapping[str, Any]) -> None:
    with path.open("x", encoding="utf-8") as stream:
        yaml.safe_dump(dict(document), stream, sort_keys=False)


def _validate_reloaded_dataset(
    dataset: AlignmentDataset,
    summary: ExtractionSummary,
) -> None:
    actual = (
        dataset.num_rgb_frames,
        dataset.num_depth_frames,
        dataset.width,
        dataset.height,
    )
    expected = (
        summary.rgb_frames,
        summary.depth_frames,
        summary.width,
        summary.height,
    )
    if actual != expected:
        raise ValueError(
            f"Reloaded alignment dataset contract {actual} does not match {expected}"
        )


def _require_mapping(
    document: Mapping[str, Any],
    path: tuple[str, ...],
) -> Mapping[str, Any]:
    value: Any = document
    for key in path:
        if not isinstance(value, Mapping) or key not in value:
            raise ValueError(f"Missing metadata field {'.'.join(path)}")
        value = value[key]
    if not isinstance(value, Mapping):
        raise ValueError(f"Metadata field {'.'.join(path)} must be a mapping")
    return value


def _require_string(document: Mapping[str, Any], path: tuple[str, ...]) -> str:
    value = _value_at(document, path)
    if not isinstance(value, str) or not value:
        raise ValueError(f"Metadata field {'.'.join(path)} must be a non-empty string")
    return value


def _require_absolute_topic(document: Mapping[str, Any], key: str) -> str:
    topic = _require_string(document, (key,))
    if not topic.startswith("/"):
        raise ValueError(f"Metadata field {key} must be an absolute ROS topic")
    return topic


def _require_equal(
    document: Mapping[str, Any],
    path: tuple[str, ...],
    expected: Any,
) -> None:
    actual = _value_at(document, path)
    if actual != expected or type(actual) is not type(expected):
        raise ValueError(
            f"Metadata field {'.'.join(path)} must equal {expected!r}; "
            f"got {actual!r}"
        )


def _value_at(document: Mapping[str, Any], path: tuple[str, ...]) -> Any:
    value: Any = document
    for key in path:
        if not isinstance(value, Mapping) or key not in value:
            raise ValueError(f"Missing metadata field {'.'.join(path)}")
        value = value[key]
    return value


def print_summary(summary: ExtractionSummary) -> None:
    """Print a concise extraction result."""
    print(f"Experiment: {summary.experiment}")
    print(f"RGB frames: {summary.rgb_frames}")
    print(f"Aligned-depth frames: {summary.depth_frames}")
    print(f"Pixel grid: {summary.width}x{summary.height}")
    print(f"Output: {summary.output_directory}")


def main(argv: list[str] | None = None) -> int:
    """Run the alignment extraction CLI."""
    args = parse_args(argv)
    try:
        summary = extract_alignment_dataset(
            experiment_dir=args.experiment_dir,
            output_dir=args.output_dir,
        )
    except (OSError, RuntimeError, ValueError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    print_summary(summary)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
