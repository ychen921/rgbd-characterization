# RGB–Depth Alignment Dataset Plan

## 1. Purpose

This document defines the formal dataset collection plan for RGB–Depth spatial alignment using the sensor or SDK aligned-depth output.

The goal is to measure whether the same physical foreground boundary appears at the same pixel location in:

```text
RGB image
aligned depth image
```

This dataset is for physical alignment characterization, not for validating a custom depth-to-color transformation implementation.

Related documents:

```text
docs/baseline_analysis_plan.md
docs/rgb_depth_alignment_plan.md
docs/d2c_transformation_validation_plan.md
```

---

## 2. Scene Assignment

```text
scene01 → white-board depth baseline / distance
scene02 → depth incidence-angle characterization
scene03 → depth material characterization
scene04 → depth edge / depth-discontinuity characterization
scene05 → RGB–Depth alignment characterization
```

Formal alignment bags must therefore use:

```text
scene05_alignment_...
```

Do not use `scene04` for RGB–Depth alignment.

---

## 3. Alignment Target

Recommended setup:

```text
RGB-D camera
↓
dark matte rectangular foreground target
↓ 200–400 mm separation
light matte flat background board
```

The foreground target should provide clear:

```text
left edge
right edge
top edge
bottom edge
four corners
```

Avoid transparent, reflective, glossy, flexible, or irregular targets.

---

## 4. Camera Configuration

The formal dataset should be recorded with depth-to-color registration enabled.

```text
depth_registration = true
depth_precision = 1mm
```

The aligned depth image must have a documented mapping to the RGB image coordinate system.

Do not infer alignment solely from:

```text
topic name
resolution
frame_id
```

Confirm:

```text
registration parameter
topic semantics
visual RGB/depth overlay
```

---

## 5. Required ROS Topics

Record at least:

```text
/camera/color/image_raw
/camera/color/camera_info
/camera/depth/image_raw
/camera/depth/camera_info
/tf_static
```

If available, also record:

```text
/camera/depth/image_unaligned
/camera/depth_to_color
```

The actual names depend on the Orbbec ROS2 wrapper version.

Before recording:

```bash
ros2 topic list | grep -E "color|depth|extrinsic|tf"
```

---

## 6. Resolution Requirements

Native resolutions may differ, for example:

```text
RGB:          1280 × 720
native depth:  848 × 480
```

This is normal.

For pixel-domain alignment metrics, the aligned-depth coordinate convention must be known.

Preferred:

```text
aligned depth width  == RGB width
aligned depth height == RGB height
```

If not, document:

```text
aligned output resolution
crop or scaling behavior
camera_info association
pixel-coordinate conversion
```

Do not use `cv2.resize()` on raw depth as a substitute for registration.

---

## 7. Dataset Phases

### 7.1 Phase A — Sanity Check

Record first:

```text
scene05_alignment_d100_center_yaw00_r01
scene05_alignment_d100_center_yaw00_r02
scene05_alignment_d100_center_yaw00_r03
```

Configuration:

```yaml
distance_mm: 1000
position: center
yaw_deg: 0
repeats: 3
duration_sec: 5
```

Validate:

```text
aligned topic semantics
frame pairing
RGB/depth segmentation
edge overlay
repeatability
```

Do not record the full matrix before these bags pass analysis.

### 7.2 Phase B — Core Static Dataset

```yaml
distances_mm:
  - 500
  - 1000
  - 2000

positions:
  - center
  - top_left
  - top_right
  - bottom_left
  - bottom_right

yaw_deg:
  - 0

repeats: 3
duration_sec: 5
```

Total:

```text
3 distances × 5 positions × 3 repeats = 45 bags
```

This is the primary alignment dataset.

### 7.3 Phase C — Angle Extension

Add only after Phase B:

```yaml
distances_mm:
  - 1000
  - 2000

positions:
  - center

additional_yaw_deg:
  - -30
  - 30

repeats: 3
```

Additional total:

```text
2 distances × 2 angles × 3 repeats = 12 bags
```

The `yaw00` cases already exist and should not be repeated.

### 7.4 Optional Phase D — Dynamic Synchronization

Keep this separate from static statistics.

Suggested family:

```text
scene06_alignment_dynamic_...
```

Dynamic tests combine:

```text
spatial registration error
temporal synchronization error
```

---

## 8. Naming Convention

```text
scene05_alignment_d{distance}_{position}_yaw{angle}_r{repeat}
```

Examples:

```text
scene05_alignment_d050_center_yaw00_r01
scene05_alignment_d100_top_left_yaw00_r02
scene05_alignment_d200_bottom_right_yaw00_r03
scene05_alignment_d100_center_yawm30_r01
scene05_alignment_d100_center_yawp30_r01
```

Use:

```text
yaw00
yawm30
yawp30
```

Avoid `+` and `-` in directory names.

---

## 9. Per-Bag Metadata

Save `experiment.yaml` beside each bag.

```yaml
experiment:
  name: scene05_alignment_d100_center_yaw00_r01
  type: rgb_depth_alignment
  scene: 5
  repeat: 1

target:
  foreground_material: matte_black_board
  background_material: matte_white_board
  foreground_background_separation_mm: 300

geometry:
  camera_to_foreground_mm: 1000
  image_position: center
  yaw_deg: 0

recording:
  duration_sec: 5
  nominal_fps: 30

registration:
  enabled: true
  aligned_depth_topic: /camera/depth/image_raw
  unaligned_depth_topic: /camera/depth/image_unaligned
  mode: unknown

color:
  topic: /camera/color/image_raw
  width: 1280
  height: 720
  encoding: rgb8

depth:
  topic: /camera/depth/image_raw
  width: 1280
  height: 720
  encoding: 16UC1
```

Replace `unknown` after confirming hardware or software registration mode.

---

## 10. Recording Procedure

```text
1. Fix the camera on a tripod
2. Fix the background board
3. Place the foreground target at the required distance
4. Set the foreground/background separation
5. Move the target to the required image position
6. Set target yaw
7. Verify all four target edges are visible
8. Verify RGB and aligned-depth streams
9. Record for 5 seconds
10. Stop and validate the bag
11. Save metadata
12. Record the next repeat
```

Document whether the target is repositioned between repeats.

---

## 11. Pre-Recording Validation

```bash
ros2 topic hz /camera/color/image_raw
ros2 topic hz /camera/depth/image_raw

ros2 topic echo --once /camera/color/image_raw
ros2 topic echo --once /camera/depth/image_raw

ros2 topic echo --once /camera/color/camera_info
ros2 topic echo --once /camera/depth/camera_info

ros2 node list
ros2 param dump <camera_node_name> > camera_params.yaml
```

Confirm:

```text
registration enabled
RGB resolution
aligned-depth resolution
encoding
frame_id semantics
nominal frame rate
```

---

## 12. Post-Recording Validation

Record for every bag:

```text
duration
RGB frame count
depth frame count
RGB resolution
depth resolution
RGB encoding
depth encoding
RGB frame_id
depth frame_id
first timestamp
last timestamp
```

Reject or flag bags with:

```text
missing streams
target clipping
unknown registration state
unexpected resolution changes
abnormally low frame count
failed timestamp pairing
failed foreground segmentation
```

---

## 13. Alignment ROI

Repeats share one alignment ROI:

```text
scene05_alignment_d100_center_yaw00_r01
scene05_alignment_d100_center_yaw00_r02
scene05_alignment_d100_center_yaw00_r03
↓
config/roi/alignment/scene05_alignment_d100_center_yaw00.yaml
```

The ROI must include:

```text
foreground interior
all four foreground edges
surrounding background
```

Example:

```yaml
name: scene05_alignment_d100_center_yaw00

source:
  experiment: scene05_alignment_d100_center_yaw00_r01
  rgb_frame_index: 120
  depth_frame_index: 119

roi:
  type: rectangle
  x: 180
  y: 100
  width: 280
  height: 260
```

---

## 14. Extracted Dataset Structure

```text
data/
└── scene05_alignment_d100_center_yaw00_r01/
    ├── rgb.npz
    ├── aligned_depth.npz
    ├── timestamps.npz
    ├── color_camera_info.yaml
    ├── depth_camera_info.yaml
    ├── camera_params.yaml
    ├── experiment.yaml
    ├── preflight.yaml
    ├── post_recording.yaml
    ├── extraction_summary.yaml
    └── pairing/
        ├── frame_pairing.csv
        └── summary.yaml
```

Recommended arrays:

```text
rgb
rgb_timestamp_ns
aligned_depth
depth_timestamp_ns
rgb_recorded_timestamp_ns
depth_recorded_timestamp_ns
```

Optional paired indices:

```text
pair_rgb_index
pair_depth_index
pair_delta_ms
```

Use `tools/extract_alignment_dataset.py` to produce this directory. Extraction
uses a non-overwriting staging directory, reload-validates all three NPZ files,
and only then publishes the final output directory. `extraction_summary.yaml`
records source topics, stream contracts, frame counts, timestamp ranges, and
storage-versus-header latency. It does not contain paired indices or pairing
deltas; those belong to the frame-pairing phase.

Pairing artifacts are derived without modifying the extracted NPZ files.
`frame_pairing.csv` contains accepted one-to-one pairs and preserves exact
signed nanosecond deltas; `pairing/summary.yaml` stores the pairing contract,
source stream counts, rejected/unmatched counts, and delta statistics. The
directory is reload-validated, atomically published, and never overwritten.
Zero accepted pairs are represented by an empty CSV plus `null` delta
statistics, not a zero-error measurement.

---

## 15. Dataset Inventory

```text
Phase A: 3 bags
Phase B: 45 bags total, including Phase A conditions
Phase C: 12 additional bags
```

Full static dataset with angle extension:

```text
45 + 12 = 57 bags
```

Current target:

```text
Phase A only
```

---

## 16. Collection Order

```text
1. Record d100 center × 3 repeats
2. Validate extraction and edge overlay
3. Complete d050/d100/d200 center × 3 repeats
4. Add four corner positions
5. Compare center versus corners
6. Add yaw ±30° only if useful
7. Consider dynamic synchronization separately
```

---

## 17. Completion Criteria

Phase A is complete when:

```text
required topics are present
aligned-depth semantics are confirmed
frames can be paired
target masks can be extracted
edge overlay is correct
three repeats are stable
```

Phase B is complete when:

```text
all 45 bags are recorded
all bags have metadata
all bags pass basic validation
all conditions have ROI files
center and corner results can be aggregated
```
