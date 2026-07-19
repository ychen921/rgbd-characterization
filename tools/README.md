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

## `analyze_baseline.py`

Analyzes one extracted white-board baseline dataset inside its configured ROI.
The tool loads existing files only and never opens the ROI selection GUI.

Before running the analysis, the experiment directory must contain:

```text
data/scene01_white_d050_r01/
└── depth.npz
```

The shared ROI configuration must also exist:

```text
config/roi/scene01_white_d050.yaml
```

Run from the workspace root:

```bash
python3 tools/analyze_baseline.py \
    data/scene01_white_d050_r01
```

The analysis pipeline is:

```text
depth.npz
↓
load shared distance-group ROI
↓
crop raw uint16 depth frames
├── zero and maximum-uint16 occurrence metrics
└── convert 0 and 65535 to NaN
        ├── per-pixel temporal noise
        └── per-frame measured-depth median
↓
save baseline artifacts
```

By default, artifacts are written to:

```text
results/scene01_white_d050_r01/baseline/
├── summary.yaml
├── frame_median_depth.csv
├── temporal_std.npy
├── zero_ratio_map.npy
└── max_uint16_ratio_map.npy
```

`frame_median_depth.csv` contains one timestamp-aligned row per input frame:

```text
frame_index,timestamp_ns,median_depth_mm
```

An all-invalid frame keeps its row and uses an empty median-depth field. The
three NPY files contain ROI-sized `float64` maps and preserve NaN values.

Options:

```text
--roi-root PATH
    ROI configuration directory. Defaults to config/roi.

--output-dir PATH
    Artifact output directory. Defaults to
    results/<experiment>/baseline.

--min-valid-ratio FLOAT
    Minimum valid-frame ratio for each temporal-noise pixel. Defaults to 0.9.
```

Example with explicit paths:

```bash
python3 tools/analyze_baseline.py \
    data/scene01_white_d050_r01 \
    --roi-root config/roi \
    --output-dir results/scene01_white_d050_r01/baseline \
    --min-valid-ratio 0.9
```

The analysis is non-overwriting. If any planned artifact already exists, the
tool fails before writing new output files. If the ROI configuration is
missing, run `select_roi.py` first.

Show the CLI help:

```bash
python3 tools/analyze_baseline.py --help
```
