# Tools

## `extract_dataset.py`

Extracts raw depth frames from a ROS 2 bag into an NPZ dataset.

### Usage

Load the ROS 2 Humble environment and go to the project root:

```bash
source /opt/ros/humble/setup.bash
cd ~/dev/rgbd-characterization
```

Run the extraction:

```bash
python3 tools/extract_dataset.py \
    bags/scene01_white_d050_r01/rosbag \
    data/scene01_white_d050_r01/depth.npz
```

Arguments:

```text
python3 tools/extract_dataset.py BAG_PATH OUTPUT_PATH
```

- `BAG_PATH` must be the rosbag directory that directly contains `metadata.yaml`.
- The parent directory of `OUTPUT_PATH` is created automatically.
- The depth topic is fixed to `/camera/depth/image_raw`.

The output NPZ contains:

| Array | Shape | Dtype |
|---|---|---|
| `depth` | `(N, H, W)` | `uint16` |
| `timestamps_ns` | `(N,)` | `int64` |

Show the CLI help:

```bash
python3 tools/extract_dataset.py --help
```

## `inspect_dataset.py`

Prints dataset statistics and saves first, middle, and last frame images.

```bash
python3 tools/inspect_dataset.py \
    data/scene01_white_d050_r01/depth.npz
```

Images are written to:

```text
results/scene01_white_d050_r01/inspection/
```

Use `--output-dir PATH` to select a different image directory.

## `select_roi.py`

Displays the middle depth frame and saves a rectangular white-board ROI shared
by repeats at the same scene, target, and distance.

```bash
python3 tools/select_roi.py \
    data/scene01_white_d050_r01
```

The experiment directory must contain `depth.npz`. The generated configuration
is saved as:

```text
config/roi/scene01_white_d050.yaml
```

If that ROI file already exists, selection is skipped without overwriting it.
Use `--roi-root PATH` to select a different configuration directory.
