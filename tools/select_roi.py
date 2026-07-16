"""Interactively select and save a rectangular ROI."""

import argparse
import sys
from pathlib import Path

import cv2
import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.io.dataset import DepthDataset
from src.preprocessing.roi import RectROI, derive_roi_key, get_roi_path, save_roi


DEFAULT_ROI_ROOT = PROJECT_ROOT / "config" / "roi"
WINDOW_NAME = "Select ROI"
DISPLAY_PERCENTILE_LOW = 1.0
DISPLAY_PERCENTILE_HIGH = 99.0


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments for interactive ROI selection."""
    parser = argparse.ArgumentParser(
        description="Select an ROI from an extracted depth dataset."
    )
    parser.add_argument(
        "dataset_dir",
        type=Path,
        help="Experiment directory containing depth.npz.",
    )
    parser.add_argument(
        "--roi-root",
        type=Path,
        default=DEFAULT_ROI_ROOT,
        help="ROI configuration directory (default: config/roi).",
    )
    return parser.parse_args()


def depth_to_display(depth: np.ndarray) -> np.ndarray:
    """Convert one raw depth frame to a display-only 8-bit BGR image."""
    if not isinstance(depth, np.ndarray):
        raise TypeError(
            f"depth must be a numpy.ndarray; got {type(depth).__name__}"
        )
    if depth.ndim != 2:
        raise ValueError(f"depth must have shape (H, W); got shape {depth.shape}")

    max_uint16 = np.iinfo(np.uint16).max
    valid = (depth > 0) & (depth < max_uint16)
    if not np.any(valid):
        raise ValueError("Frame contains no displayable depth")

    values = depth[valid]
    lower = float(np.percentile(values, DISPLAY_PERCENTILE_LOW))
    upper = float(np.percentile(values, DISPLAY_PERCENTILE_HIGH))
    if upper <= lower:
        raise ValueError("Invalid display depth range")

    clipped = np.clip(depth.astype(np.float32), lower, upper)
    normalized = (clipped - lower) / (upper - lower) * 255.0
    image = normalized.astype(np.uint8)
    return cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)


def choose_rectangle(display_image: np.ndarray) -> RectROI:
    """Open the ROI selector and return the selected rectangle."""
    try:
        x, y, width, height = cv2.selectROI(
            WINDOW_NAME,
            display_image,
            showCrosshair=True,
            fromCenter=False,
        )
    finally:
        cv2.destroyAllWindows()

    if width <= 0 or height <= 0:
        raise ValueError("ROI selection was cancelled; no configuration saved")

    return RectROI(
        x=int(x),
        y=int(y),
        width=int(width),
        height=int(height),
    )


def select_roi(dataset_dir: Path, roi_root: Path = DEFAULT_ROI_ROOT) -> Path:
    """Select an ROI for an experiment unless its shared YAML already exists."""
    dataset_dir = Path(dataset_dir).expanduser()
    experiment_name = dataset_dir.name
    if not experiment_name:
        raise ValueError(f"Cannot derive experiment name from {dataset_dir}")

    roi_key = derive_roi_key(experiment_name)
    roi_path = get_roi_path(roi_root, experiment_name)

    print("Dataset:")
    print(f"  {dataset_dir}")
    print()
    print("ROI key:")
    print(f"  {roi_key}")
    print()

    if roi_path.exists():
        print("ROI already exists:")
        print(f"  {roi_path}")
        print()
        print("Skipping ROI selection.")
        return roi_path

    dataset_path = dataset_dir / "depth.npz"
    dataset = DepthDataset.load(dataset_path)
    if dataset.num_frames == 0:
        raise ValueError(f"Dataset contains no depth frames: {dataset_path}")

    frame_index = dataset.num_frames // 2
    display_image = depth_to_display(dataset.depth[frame_index])

    print("Selecting ROI...")
    roi = choose_rectangle(display_image)
    roi.crop(dataset.depth)

    save_roi(
        roi_path,
        roi,
        name=roi_key,
        source_experiment=experiment_name,
        source_frame_index=frame_index,
    )

    print()
    print("Saved:")
    print(f"  {roi_path}")
    print(f"  rectangle: x={roi.x}, y={roi.y}, width={roi.width}, height={roi.height}")
    print(f"  pixels: {roi.pixel_count}")
    return roi_path


def main() -> int:
    """Run interactive ROI selection."""
    args = parse_args()
    select_roi(dataset_dir=args.dataset_dir, roi_root=args.roi_root)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
