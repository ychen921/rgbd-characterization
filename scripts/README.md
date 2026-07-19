# rgbd-characterization scripts

This document describes the shell scripts under `scripts/` and how to use them.

## Overview

The repository includes these helper scripts:

- `run_ubuntu.sh` - Launches a Docker container for the ROS 2 Humble workspace.
- `host_orbbec_env.sh` - Exports ROS 2 environment variables for Orbbec experiment setup.
- `record_experiment.sh` - Creates experiment metadata and records ROS bag data for sensor characterization.

---

## `run_ubuntu.sh`

### Purpose

Starts a Docker container named `rgbd-characterization-ubuntu` with the
repository mounted and device access enabled.

### Behavior

- Uses Docker image `ros2-humble:latest`
- Resolves the repository root from the script location by default
- Mounts the repository into `/workspaces/rgbd-characterization`
- Mounts `/bags`, `/data`, and `/results` directories from the workspace
- Shares host network, IPC, and device access
- Sets ROS environment variables inside the container
- Starts a Bash shell in the container

### Usage

From the repository root:

```bash
./scripts/run_ubuntu.sh
```

### Notes

- `RGBD_CHARACTERIZATION_ROOT` can override the repository root when needed.
- It expects Docker to be installed and the image `ros2-humble:latest` available locally.

---

## `host_orbbec_env.sh`

### Purpose

Configures the current shell environment for Orbbec ROS 2 experiments on the host machine.

### Behavior

- Sources `/opt/ros/humble/setup.bash`
- Exports:
  - `ROS_DOMAIN_ID=30`
  - `RMW_IMPLEMENTATION=rmw_fastrtps_cpp`
  - `ROS_LOCALHOST_ONLY=0`
  - `FASTRTPS_DEFAULT_PROFILES_FILE`
  - `FASTDDS_DEFAULT_PROFILES_FILE`

### Usage

Source it into your shell so the exported variables persist in the current session:

```bash
source scripts/host_orbbec_env.sh
```

### Notes

- `RGBD_CHARACTERIZATION_ROOT` defaults to the repository root resolved from
  the script location.
- The Fast DDS profile paths are derived from
  `${RGBD_CHARACTERIZATION_ROOT}/config/fastdds_udp_only.xml`.

---

## `record_experiment.sh`

### Purpose

Creates an experiment directory with YAML metadata and records a ROS bag of Orbbec camera topics.

### Environment

- `RGBD_CHARACTERIZATION_ROOT` can be overridden externally. It defaults to
  the repository root resolved from the script location.
- `BAG_ROOT` can be overridden externally. It defaults to
  `${RGBD_CHARACTERIZATION_ROOT}/bags`.
- Requires the `ros2` command to be available in the environment.

### Required arguments

- `--scene NAME`
- `--material NAME`

### Optional arguments

- `--distance M` (meters)
- `--angle DEG`
- `--camera-height M`
- `--target-width M`
- `--target-height M`
- `--duration SEC` (default: `30`)
- `--repeat ID` (default: `1`)
- `--notes TEXT`

### Example

```bash
./scripts/record_experiment.sh \
  --scene scene01_distance \
  --material white \
  --distance 1.0 \
  --angle 0 \
  --camera-height 0.85 \
  --target-width 0.30 \
  --target-height 0.50 \
  --duration 30 \
  --repeat 1
```

### Behavior

- Validates required options and numeric values
- Computes experiment ID from `scene`, `material`, optional distance, and repeat number
- Creates `${BAG_ROOT}/${EXPERIMENT_ID}` and `${EXPERIMENT_DIR}/experiment.yaml`
- Writes metadata and starts `ros2 bag record`

### Recorded topics

- `/camera/depth/image_raw`
- `/camera/depth/camera_info`
- `/camera/depth/metadata`
- `/camera/depth/points`
- `/camera/color/image_raw`
- `/camera/color/camera_info`
- `/camera/device_status`
- `/camera/depth_filter_status`
- `/camera/depth_filters/status`
- `/diagnostics`

### Notes

- If the experiment directory already exists, the script exits with an error.
- `--distance` is converted to centimeters for the experiment ID.
- `--duration` must be a positive integer.
- `--repeat` must be an integer.

---

## General Recommendations

- Run `run_ubuntu.sh` when you need a consistent Ubuntu/ROS 2 Humble container environment.
- Use `source scripts/host_orbbec_env.sh` if you are working directly on the host and need Orbbec ROS 2 settings.
- Use `record_experiment.sh` to create a reproducible experiment record with metadata and rosbag capture.
