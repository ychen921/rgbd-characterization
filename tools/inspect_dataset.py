"""Inspect an extracted depth dataset and generate QA images."""

import argparse
import math
import sys
from pathlib import Path

import matplotlib
import numpy as np


matplotlib.use("Agg")
import matplotlib.pyplot as plt


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.io.dataset import DepthDataset


UINT16_VALUE_COUNT = 1 << 16
DISPLAY_PERCENTILE_LOW = 1.0
DISPLAY_PERCENTILE_HIGH = 99.0


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments for dataset inspection."""
    parser = argparse.ArgumentParser(
        description="Inspect an extracted depth NPZ dataset."
    )
    parser.add_argument(
        "dataset_path",
        type=Path,
        help="Path to the depth NPZ dataset.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        help=(
            "Directory for inspection images. Defaults to "
            "results/<experiment>/inspection."
        ),
    )
    return parser.parse_args()


def resolve_output_dir(
    dataset_path: Path,
    output_dir: Path | None,
) -> Path:
    """Resolve the inspection image output directory."""
    if output_dir is not None:
        return output_dir.expanduser()

    experiment_name = dataset_path.parent.name
    return PROJECT_ROOT / "results" / experiment_name / "inspection"


def compute_depth_histogram(depth: np.ndarray) -> np.ndarray:
    """Count all uint16 values without flattening the full dataset."""
    histogram = np.zeros(UINT16_VALUE_COUNT, dtype=np.uint64)

    for frame in depth:
        frame_histogram = np.bincount(
            frame.ravel(),
            minlength=UINT16_VALUE_COUNT,
        )
        histogram += frame_histogram.astype(np.uint64, copy=False)

    return histogram


def histogram_percentile(
    histogram: np.ndarray,
    percentile: float,
) -> float:
    """Return a linearly interpolated percentile from a value histogram."""
    if not 0.0 <= percentile <= 100.0:
        raise ValueError(f"Percentile must be in [0, 100]; got {percentile}")

    total = int(histogram.sum())
    if total == 0:
        raise ValueError("Cannot compute a percentile from an empty histogram")

    rank = percentile / 100.0 * (total - 1)
    lower_rank = math.floor(rank)
    upper_rank = math.ceil(rank)
    cumulative = np.cumsum(histogram)

    lower_value = int(np.searchsorted(cumulative, lower_rank + 1))
    upper_value = int(np.searchsorted(cumulative, upper_rank + 1))
    fraction = rank - lower_rank

    return lower_value + (upper_value - lower_value) * fraction


def print_report(
    dataset_path: Path,
    dataset: DepthDataset,
    histogram: np.ndarray,
    image_paths: list[Path],
) -> None:
    """Print dataset, timestamp, and raw depth statistics."""
    timestamps_ns = dataset.timestamps_ns
    intervals_ns = np.diff(timestamps_ns)
    populated_values = np.flatnonzero(histogram)

    raw_min = int(populated_values[0])
    raw_max = int(populated_values[-1])
    median = histogram_percentile(histogram, 50.0)
    p01 = histogram_percentile(histogram, 1.0)
    p05 = histogram_percentile(histogram, 5.0)
    p95 = histogram_percentile(histogram, 95.0)
    p99 = histogram_percentile(histogram, 99.0)

    print("Dataset:")
    print(f"  path: {display_path(dataset_path)}")
    print()
    print("Depth:")
    print(f"  shape: {dataset.depth.shape}")
    print(f"  dtype: {dataset.depth.dtype}")
    print(f"  width: {dataset.width}")
    print(f"  height: {dataset.height}")
    print()
    print("Frames:")
    print(f"  count: {dataset.num_frames}")
    print()
    print("Timestamp:")
    print(f"  first: {int(timestamps_ns[0])} ns")
    print(f"  last: {int(timestamps_ns[-1])} ns")

    if intervals_ns.size:
        duration_sec = (int(timestamps_ns[-1]) - int(timestamps_ns[0])) / 1e9
        intervals_ms = intervals_ns / 1e6
        estimated_fps = (
            (dataset.num_frames - 1) / duration_sec
            if duration_sec > 0.0
            else None
        )

        print(f"  duration: {duration_sec:.6f} sec")
        print(f"  mean interval: {np.mean(intervals_ms):.3f} ms")
        print(f"  median interval: {np.median(intervals_ms):.3f} ms")
        print(f"  min interval: {np.min(intervals_ms):.3f} ms")
        print(f"  max interval: {np.max(intervals_ms):.3f} ms")
        if estimated_fps is None:
            print("  estimated FPS: N/A")
        else:
            print(f"  estimated FPS: {estimated_fps:.3f} Hz")
        print(
            "  strictly increasing: "
            f"{'yes' if np.all(intervals_ns > 0) else 'no'}"
        )
    else:
        print("  duration: 0.000000 sec")
        print("  interval statistics: N/A")
        print("  estimated FPS: N/A")
        print("  strictly increasing: N/A")

    print()
    print("Raw depth:")
    print(f"  min: {raw_min}")
    print(f"  max: {raw_max}")
    print(f"  median: {median:.2f}")
    print(f"  p01: {p01:.2f}")
    print(f"  p05: {p05:.2f}")
    print(f"  p95: {p95:.2f}")
    print(f"  p99: {p99:.2f}")
    print()
    print("Inspection images:")
    for image_path in image_paths:
        print(f"  {display_path(image_path)}")


def save_inspection_images(
    dataset: DepthDataset,
    output_dir: Path,
    histogram: np.ndarray,
) -> list[Path]:
    """Save first, middle, and last depth frames with a shared scale."""
    output_dir.mkdir(parents=True, exist_ok=True)

    populated_values = np.flatnonzero(histogram)
    raw_min = float(populated_values[0])
    raw_max = float(populated_values[-1])
    display_min = histogram_percentile(histogram, DISPLAY_PERCENTILE_LOW)
    display_max = histogram_percentile(histogram, DISPLAY_PERCENTILE_HIGH)

    if display_min >= display_max:
        display_min = raw_min
        display_max = raw_max
    if display_min >= display_max:
        display_max = display_min + 1.0

    frames = (
        ("first", 0),
        ("middle", dataset.num_frames // 2),
        ("last", dataset.num_frames - 1),
    )
    image_paths: list[Path] = []

    for label, index in frames:
        image_path = output_dir / f"frame_{label}.png"
        figure, axes = plt.subplots(figsize=(10, 6))
        image = axes.imshow(
            dataset.depth[index],
            cmap="viridis",
            vmin=display_min,
            vmax=display_max,
        )
        axes.set_title(
            f"{label.capitalize()} frame {index} "
            f"({int(dataset.timestamps_ns[index])} ns)"
        )
        axes.set_xlabel("x [pixel]")
        axes.set_ylabel("y [pixel]")
        figure.colorbar(image, ax=axes, label="Raw depth value")
        figure.savefig(image_path, dpi=150, bbox_inches="tight")
        plt.close(figure)
        image_paths.append(image_path)

    return image_paths


def display_path(path: Path) -> Path:
    """Return a project-relative path when possible."""
    try:
        return path.resolve().relative_to(PROJECT_ROOT)
    except ValueError:
        return path


def inspect_dataset(dataset_path: Path, output_dir: Path | None = None) -> None:
    """Inspect a dataset, print statistics, and save QA images."""
    dataset_path = dataset_path.expanduser()
    dataset = DepthDataset.load(dataset_path)

    if dataset.num_frames == 0:
        raise ValueError(f"Dataset contains no depth frames: {dataset_path}")

    resolved_output_dir = resolve_output_dir(dataset_path, output_dir)
    histogram = compute_depth_histogram(dataset.depth)
    image_paths = save_inspection_images(
        dataset=dataset,
        output_dir=resolved_output_dir,
        histogram=histogram,
    )
    print_report(
        dataset_path=dataset_path,
        dataset=dataset,
        histogram=histogram,
        image_paths=image_paths,
    )


def main() -> int:
    """Run the command-line dataset inspection."""
    args = parse_args()
    inspect_dataset(
        dataset_path=args.dataset_path,
        output_dir=args.output_dir,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
