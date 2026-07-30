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

## `select_edge_roi.py`

Selects a foreground reference ROI, background reference ROI, edge analysis
ROI, and two-point nominal edge from a representative depth frame.

```bash
python3 tools/select_edge_roi.py \
    data/scene04_edge_d050_r01
```

The middle frame is used by default. Use `--frame-index INDEX` to select a
different representative frame.

The interaction order is:

```text
foreground reference ROI
↓
background reference ROI
↓
edge analysis ROI
↓
nominal edge p1 and p2
↓
final review
```

During nominal-edge and final review:

```text
Enter → accept
R     → reselect
Esc   → cancel
```

Accepted selections produce one shared configuration and clean preview:

```text
config/roi/scene04_edge_d050.yaml
results/roi_preview/scene04_edge_d050.png
```

Repeats such as `r01`, `r02`, and `r03` share the same ROI key. When both
outputs already exist and are readable, the tool skips selection. If only one
output exists, or either existing output is invalid, the command fails without
opening the GUI. Existing files are never overwritten.

Use `--roi-root PATH` and `--preview-root PATH` to change the output
directories. Show all analysis-threshold options with:

```bash
python3 tools/select_edge_roi.py --help
```

The default signed-distance profile uses 2 px bins and covers 20 px on each
side of the nominal edge. Use `--distance-bin-px` and
`--max-edge-distance-px` to override both values consistently across all
datasets in one comparison.

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

The depth CameraInfo calibration must match the extracted depth-frame
resolution:

```text
config/calib/depth_camera_info.yaml
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
load depth CameraInfo and validate the full-frame resolution
↓
crop raw uint16 depth frames
├── zero and maximum-uint16 occurrence metrics
└── convert 0 and 65535 to NaN
        ├── per-pixel temporal noise
        ├── per-frame measured-depth median
        └── restore full-image ROI coordinates
                ↓
            back-project valid depth to camera-space points
                ↓
            fit one deterministic SVD plane per frame
                ↓
            distance, tilt, residual, and inlier metrics
↓
save baseline artifacts
```

By default, artifacts are written to:

```text
results/scene01_white_d050_r01/baseline/
├── summary.yaml
├── frame_median_depth.csv
├── frame_plane_metrics.csv
├── temporal_std.npy
├── zero_ratio_map.npy
└── max_uint16_ratio_map.npy
```

`frame_median_depth.csv` contains one timestamp-aligned row per input frame:

```text
frame_index,timestamp_ns,median_depth_mm
```

`frame_plane_metrics.csv` also keeps one timestamp-aligned row per input
frame:

```text
frame_index,timestamp_ns,fit_succeeded,valid_points,normal_x,normal_y,normal_z,plane_distance_m,tilt_deg,residual_rmse_mm,residual_std_mm,residual_p95_abs_mm,inlier_ratio
```

An all-invalid or insufficient-point frame keeps its rows and uses empty
floating-point fields. Its valid-point count and fit status remain available.
The three NPY files contain ROI-sized `float64` maps and preserve NaN values.
`summary.yaml` records the CameraInfo source and intrinsics, plane-fitting
parameters, successful and failed frame counts, and aggregate planarity
statistics.

Options:

```text
--roi-root PATH
    ROI configuration directory. Defaults to config/roi.

--output-dir PATH
    Artifact output directory. Defaults to
    results/<experiment>/baseline.

--min-valid-ratio FLOAT
    Minimum valid-frame ratio for each temporal-noise pixel. Defaults to 0.9.

--depth-camera-info PATH
    Depth CameraInfo YAML. Defaults to
    config/calib/depth_camera_info.yaml.

--plane-inlier-threshold-mm FLOAT
    Maximum absolute plane residual counted as an inlier, in millimetres.
    Defaults to 5.0.

--plane-min-valid-points INT
    Minimum valid depth points required for each frame's plane fit.
    Defaults to 100.
```

Example with explicit paths:

```bash
python3 tools/analyze_baseline.py \
    data/scene01_white_d050_r01 \
    --roi-root config/roi \
    --output-dir results/scene01_white_d050_r01/baseline \
    --min-valid-ratio 0.9 \
    --depth-camera-info config/calib/depth_camera_info.yaml \
    --plane-inlier-threshold-mm 5.0 \
    --plane-min-valid-points 100
```

The analysis is non-overwriting. If any planned artifact already exists, the
tool fails before writing new output files. If the ROI configuration is
missing, run `select_roi.py` first. A missing CameraInfo file or a calibration
resolution mismatch also stops the analysis before metric computation.

Show the CLI help:

```bash
python3 tools/analyze_baseline.py --help
```

## `analyze_edge.py`

Analyzes one Scene 04 depth-discontinuity dataset using an existing
foreground reference ROI, background reference ROI, edge ROI, and nominal
edge annotation. The command is non-interactive and never opens the ROI
selection GUI.

Current experiment names must follow:

```text
scene04_gap<cm>_<horizon|horizontal|vertical>_<target>_d<foreground-cm>_r<repeat>
```

For example:

```text
scene04_gap030_vertical_white_d100_r01
```

means:

```text
camera optical reference plane → foreground: 1000 mm
foreground → background Z-depth gap: 300 mm
camera optical reference plane → background: 1300 mm
```

The existing `horizon` token is retained in the experiment name and
normalized to `horizontal` in result metadata.

Before analysis, create the shared edge ROI configuration:

```bash
python3 tools/select_edge_roi.py \
    data/scene04_gap030_vertical_white_d100_r01
```

Then run:

```bash
python3 tools/analyze_edge.py \
    data/scene04_gap030_vertical_white_d100_r01
```

Use a specific representative target frame when needed:

```bash
python3 tools/analyze_edge.py \
    data/scene04_gap030_vertical_white_d100_r01 \
    --frame-index 100
```

If the target frame is rejected, the nearest valid frame is used. A tie uses
the smaller frame index.

By default, a normal analysis writes:

```text
results/<experiment>/edge_discontinuity/
├── summary.yaml
├── frame_edge_metrics.csv
├── aggregate_edge_profile.csv
├── representative_label_map.npy
├── roi_overlay.png
├── label_overlay.png
├── edge_probability_profile.png
└── temporal_edge_metrics.png
```

`frame_edge_metrics.csv` contains one timestamp-aligned row per input frame.
Rejected frames remain in the table with explicit analysis and transition
status values; undefined floating-point fields are blank.

In `aggregate_edge_profile.csv`, foreground, background, mixed, and outlier
ratios use valid pixels as their denominator. Invalid ratio uses all pixels
in the signed-distance bin.

When all frames are rejected, the command still writes:

```text
summary.yaml
frame_edge_metrics.csv
roi_overlay.png
temporal_edge_metrics.png
```

Profile and representative-label artifacts are omitted, and their
availability is recorded in `summary.yaml`.

Options:

```text
--roi-root PATH
    Edge ROI configuration directory. Defaults to config/roi.

--output-dir PATH
    Artifact output directory. Defaults to
    results/<experiment>/edge_discontinuity.

--frame-index INDEX
    Preferred representative frame. Defaults to the middle frame.
```

All available artifacts are serialized before any output file is created.
Existing artifacts are never overwritten. If a write fails, files created by
that invocation are removed.

Show the CLI help:

```bash
python3 tools/analyze_edge.py --help
```
