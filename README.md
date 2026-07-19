# [rgbd-characterization](https://github.com/ychen921/rgbd-characterization)

This workspace supports RGB-D camera operation and reproducible depth-sensor
characterization. The characterization pipeline extracts depth frames from
recordings, applies a reusable white-board ROI, computes baseline metrics, and
saves inspectable artifacts for later comparison.

Current workspace path:

```bash
~/dev/rgbd-characterization
```

---

## Workspace Structure

The structure below focuses on the baseline characterization workflow. Folder
purposes are included directly in the tree to keep the overview compact.

```text
rgbd-characterization/
├── bags/                    # Raw experiment recordings
├── config/
│   └── roi/                 # Shared ROI YAML grouped by scene/target/distance
├── data/                    # Extracted depth NPZ datasets
├── docs/                    # Analysis plans and experiment protocols
├── results/                 # Inspection reports and baseline artifacts
├── src/
│   ├── io/                  # Dataset loading and persistence
│   ├── metrics/             # Depth quality, temporal, and measured-depth metrics
│   └── preprocessing/       # ROI handling and depth preparation
├── tests/                   # Unit and integration tests
└── tools/                   # Extraction, inspection, ROI, and analysis CLIs
```

---

## Baseline Depth Characterization

The single-dataset workflow is:

```text
experiment recording
↓
full-frame depth.npz
↓
shared scene/target/distance ROI
↓
depth-quality, temporal-noise, and measured-depth metrics
↓
summary, frame-aligned CSV, and ROI-sized NPY maps
```

Run each command from the workspace root.

### 1. Extract Depth Frames

```bash
python3 tools/extract_dataset.py \
    bags/scene01_white_d050_r01/rosbag \
    data/scene01_white_d050_r01/depth.npz
```

### 2. Inspect the Dataset

```bash
python3 tools/inspect_dataset.py \
    data/scene01_white_d050_r01/depth.npz
```

### 3. Select the Shared ROI

```bash
python3 tools/select_roi.py \
    data/scene01_white_d050_r01
```

Repeats at the same scene, target, and distance reuse one ROI configuration:

```text
scene01_white_d050_r01
scene01_white_d050_r02  →  config/roi/scene01_white_d050.yaml
scene01_white_d050_r03
```

### 4. Analyze the Baseline

```bash
python3 tools/analyze_baseline.py \
    data/scene01_white_d050_r01
```

Default output:

```text
results/scene01_white_d050_r01/baseline/
├── summary.yaml
├── frame_median_depth.csv
├── temporal_std.npy
├── zero_ratio_map.npy
└── max_uint16_ratio_map.npy
```

The analysis is non-overwriting. Existing planned artifacts cause the command
to fail before new outputs are written. See [tools/README.md](tools/README.md)
for complete CLI options and artifact details, and
[docs/baseline_analysis_plan.md](docs/baseline_analysis_plan.md) for the
analysis rationale.

---

## Testing

Run the Python unit and integration tests from the workspace root:

```bash
PYTHONPATH=. PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 pytest -q
```

Disabling external pytest plugin autoload keeps system-installed ROS testing
plugins from affecting the local NumPy-based test suite.

---

## Current Characterization Milestone

Completed:

- depth extraction into a validated NPZ dataset
- dataset inspection and QA images
- reusable rectangular ROI selection
- raw-depth preprocessing for `0` and `65535`
- ROI depth-quality metrics
- per-pixel temporal-noise metrics
- per-frame measured-depth metrics
- single-dataset baseline CLI and reproducible artifacts

Next:

- validate `scene01_white_d050_r01` using the real dataset
- verify ROI reuse across the remaining 50 cm repeats
- analyze additional distances
- aggregate repeats and implement cross-distance reports and plots

---

## Runtime Environment

Before using ROS2 commands on the host machine, source the workspace environment script:

```bash
cd ~/dev/rgbd-characterization
source scripts/host_orbbec_env.sh
```

The script is the source of truth for the ROS and DDS environment. Update the
script itself instead of copying its configuration into this README.

### Verify Environment Variables

After sourcing the script:

```bash
echo $ROS_DOMAIN_ID
echo $RMW_IMPLEMENTATION
echo $ROS_LOCALHOST_ONLY
echo $FASTRTPS_DEFAULT_PROFILES_FILE
echo $FASTDDS_DEFAULT_PROFILES_FILE
```

---

## Build

From host Ubuntu or from the Docker container:

```bash
cd ~/dev/rgbd-characterization
source /opt/ros/humble/setup.bash
colcon build
```

After build:

```bash
source scripts/host_orbbec_env.sh
```

Check that Orbbec ROS2 packages are visible:

```bash
ros2 pkg list | grep -i orbbec
ros2 pkg prefix orbbec_camera
```

---

## Launch Orbbec Camera

After sourcing the environment:

```bash
cd ~/dev/rgbd-characterization
source scripts/host_orbbec_env.sh
ros2 launch orbbec_camera gemini_330_series.launch.py
```

---

## Check ROS2 Topics

List topics:

```bash
ros2 topic list
```

Common Orbbec topics:

```bash
/camera/color/image_raw
/camera/color/image_raw/compressed
/camera/color/camera_info
/camera/depth/image_raw
/camera/depth/image_raw/compressedDepth
/camera/depth/image_unaligned
/camera/depth/camera_info
/camera/depth/points
/camera/left_ir/image_raw
/camera/left_ir/camera_info
/camera/right_ir/image_raw
/camera/right_ir/camera_info
```

---

## Docker

Start the Ubuntu container:

```bash
cd ~/dev/rgbd-characterization
./scripts/run_ubuntu.sh
```

---

## DDS / Fast DDS Configuration

The workspace uses:

```bash
config/fastdds_udp_only.xml
```

Environment variables:

```bash
export FASTRTPS_DEFAULT_PROFILES_FILE=$HOME/dev/rgbd-characterization/config/fastdds_udp_only.xml
export FASTDDS_DEFAULT_PROFILES_FILE=$HOME/dev/rgbd-characterization/config/fastdds_udp_only.xml
```

This is used to keep ROS2 communication consistent between host and container.

---

## Known Assumptions and Limitations

- Raw depth currently uses the working interpretation `1 raw unit = 1 mm`;
  the baseline analyzer does not apply an additional scale conversion.
- `0` is treated as a no-depth value and excluded from numerical depth
  metrics.
- `65535` is excluded as an observed maximum-uint16 special value. It is not
  documented as an official device invalid-depth sentinel.
- Measured depth summarizes the observed ROI depth. It is not yet labeled as
  sensor accuracy or bias because setup reference-plane effects remain.
- Batch repeat aggregation and cross-distance comparison are not implemented
  yet.
