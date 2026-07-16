"""Extract raw depth frames from a ROS 2 bag into an NPZ dataset."""

import argparse
import sys
from pathlib import Path

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.io.bag_reader import DepthBagReader
from src.io.dataset import DepthDataset


DEFAULT_DEPTH_TOPIC = "/camera/depth/image_raw"


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments for dataset extraction."""
    parser = argparse.ArgumentParser(
        description=(
            "Extract raw depth frames from a ROS 2 bag into an NPZ dataset."
        )
    )
    parser.add_argument(
        "bag_path",
        type=Path,
        help="Path to the ROS 2 bag directory.",
    )
    parser.add_argument(
        "output_path",
        type=Path,
        help="Path of the output depth NPZ dataset.",
    )
    return parser.parse_args()


def extract_dataset(
    bag_path: Path,
    output_path: Path,
    depth_topic: str = DEFAULT_DEPTH_TOPIC,
) -> DepthDataset:
    """Extract all depth frames from a bag and save them as a dataset."""
    bag_path = Path(bag_path).expanduser()
    output_path = Path(output_path).expanduser()

    reader = DepthBagReader(
        bag_path=bag_path,
        depth_topic=depth_topic,
    )

    frames: list[np.ndarray] = []
    timestamps: list[int] = []

    # Keep each frame aligned with its recorded timestamp.
    for timestamp_ns, depth_frame in reader.read_frames():
        timestamps.append(timestamp_ns)
        frames.append(depth_frame)

    if not frames:
        raise RuntimeError(
            f"No depth frames found on topic {depth_topic!r} in bag {bag_path}"
        )

    # Build the dataset arrays with shapes (N, H, W) and (N,).
    depth = np.stack(frames, axis=0)
    timestamps_ns = np.asarray(timestamps, dtype=np.int64)

    dataset = DepthDataset(
        depth=depth,
        timestamps_ns=timestamps_ns,
    )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    dataset.save(output_path)

    return dataset


def print_summary(
    bag_path: Path,
    output_path: Path,
    depth_topic: str,
    dataset: DepthDataset,
) -> None:
    """Print a summary of a completed dataset extraction."""
    print(f"Bag: {bag_path}")
    print(f"Depth topic: {depth_topic}")
    print(f"Frames extracted: {dataset.num_frames}")
    print(f"Depth shape: {dataset.depth.shape}")
    print(f"Depth dtype: {dataset.depth.dtype}")
    print(f"Output: {output_path}")


def main() -> int:
    """Run the command-line extraction pipeline."""
    args = parse_args()

    dataset = extract_dataset(
        bag_path=args.bag_path,
        output_path=args.output_path,
        depth_topic=DEFAULT_DEPTH_TOPIC,
    )

    print_summary(
        bag_path=args.bag_path,
        output_path=args.output_path,
        depth_topic=DEFAULT_DEPTH_TOPIC,
        dataset=dataset,
    )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
