# Depth-to-Color Transformation Validation Plan

## 1. Purpose

This document defines how to use non-aligned RGB and depth data to validate a custom depth-to-color transformation pipeline.

Input configuration:

```text
depth registration disabled
```

Goal:

```text
raw depth
↓
depth back-projection
↓
depth-to-color extrinsic transformation
↓
color-camera projection
↓
z-buffer rasterization
↓
custom aligned depth
```

This is an implementation-validation dataset, not the formal sensor alignment dataset.

Related document:

```text
docs/rgb_depth_alignment_dataset_plan.md
```

---

## 2. Validation Targets

Verify:

```text
depth intrinsics
color intrinsics
depth unit
extrinsic direction
extrinsic translation unit
optical-frame convention
projection equations
z-buffer handling
output image coordinates
```

Compare:

```text
custom aligned depth
vs SDK aligned depth
vs RGB target boundary
```

---

## 3. Recording Modes

### 3.1 Registration Off

```text
depth_registration = false
```

Record:

```text
/camera/color/image_raw
/camera/color/camera_info
/camera/depth/image_raw
/camera/depth/camera_info
/tf_static
```

If available:

```text
/camera/depth_to_color
```

This is the source dataset for custom transformation.

### 3.2 Registration On

```text
depth_registration = true
```

Record the same static setup to obtain the SDK-aligned reference.

If aligned and unaligned depth can be published simultaneously, record both in one bag. Otherwise record separate off/on bags without moving the camera or target.

---

## 4. Dataset Location and Naming

This is diagnostic data and should not consume a formal scene number.

Recommended:

```text
diagnostics/d2c_transform/
```

Names:

```text
d2c_d100_center_registration_off_r01
d2c_d100_center_registration_on_r01
```

Repeat for `r02` and `r03` only after the first off/on pair passes validation.

---

## 5. Initial Validation Scene

Use:

```text
dark matte rectangular foreground
light matte background
foreground/background separation: 200–400 mm
```

Initial condition:

```yaml
distance_mm: 1000
position: center
yaw_deg: 0
duration_sec: 5
```

Do not begin with multiple distances and image positions.

Optional later extension:

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
```

---

## 6. Required Calibration Data

### Depth camera

```text
fx_d
fy_d
cx_d
cy_d
distortion model
distortion coefficients
image width
image height
```

### Color camera

```text
fx_c
fy_c
cx_c
cy_c
distortion model
distortion coefficients
image width
image height
```

### Depth-to-color extrinsic

```text
R_cd
t_cd
```

Required convention:

```text
p_color = R_cd × p_depth + t_cd
```

Also record:

```text
depth scale to meters
extrinsic translation unit
extrinsic source
```

Do not assume `/tf_static` is identical to the calibration extrinsic used by SDK alignment.

---

## 7. Camera Info Validation

Check:

```text
camera_info width/height matches image width/height
K matrix matches the active profile
P matrix convention
frame_id
image rectification state
distortion model
```

Do not reuse camera info from another stream profile.

Save:

```text
color_camera_info.yaml
depth_camera_info.yaml
camera_params.yaml
extrinsic.yaml
```

---

## 8. Mathematical Pipeline

For raw depth pixel `(u_d, v_d)` and depth `Z_d`:

```text
X_d = (u_d - cx_d) × Z_d / fx_d
Y_d = (v_d - cy_d) × Z_d / fy_d
```

Depth-frame point:

```text
p_d = [X_d, Y_d, Z_d]^T
```

Transform:

```text
p_c = R_cd × p_d + t_cd
```

Project:

```text
u_c = fx_c × X_c / Z_c + cx_c
v_c = fy_c × Y_c / Z_c + cy_c
```

Accept only:

```text
valid input depth
Z_c > 0
projected pixel inside color image
```

---

## 9. Distortion Policy

Explicitly choose one:

```text
rectified coordinate pipeline
raw distorted coordinate pipeline
```

Preferred first implementation:

```text
rectified coordinate pipeline
```

Do not mix raw distorted images with rectified intrinsics or projection matrices.

---

## 10. Z-Buffer Policy

Multiple depth points may project to one color pixel.

Use:

```text
nearest positive Z_c wins
```

Initial rasterization:

```text
nearest-neighbor pixel assignment
```

Defer:

```text
splatting
bilinear assignment
hole filling
```

until geometric correctness is established.

---

## 11. Dataset Structure

Source data:

```text
data/
└── diagnostics/
    └── d2c_transform/
        └── d2c_d100_center_registration_off_r01/
            ├── rgb.npz
            ├── raw_depth.npz
            ├── timestamps.npz
            ├── color_camera_info.yaml
            ├── depth_camera_info.yaml
            ├── extrinsic.yaml
            ├── camera_params.yaml
            └── experiment.yaml
```

Generated results:

```text
results/
└── diagnostics/
    └── d2c_transform/
        └── d2c_d100_center_registration_off_r01/
            ├── custom_aligned_depth.npz
            ├── projection_metrics.csv
            ├── summary.yaml
            └── overlay/
```

---

## 12. Recommended Modules

```text
src/geometry/camera.py
src/geometry/rigid_transform.py
src/geometry/depth_to_color.py
src/preprocessing/frame_pairing.py
src/visualization/alignment.py
```

Suggested functions:

```text
back_project_depth()
transform_points()
project_points()
rasterize_depth_zbuffer()
generate_aligned_depth()
```

Do not add this logic to `analyze_baseline.py`.

---

## 13. Synthetic Tests

### Identity transform

```text
same intrinsics
R = identity
t = zero
```

Expected:

```text
output pixel equals input pixel
output depth equals input depth
```

### Principal point

```text
u = cx_d
v = cy_d
Z = 1.0 m
```

Expected:

```text
[0, 0, 1]
```

### Known translation

Verify the expected projected shift.

### Transform direction

Test depth-to-color and its inverse separately.

### Z-buffer

Project two points to one output pixel and retain the nearer point.

### Invalid data

Reject:

```text
0
65535
NaN
negative depth
out-of-bounds projection
```

---

## 14. Real-Data Validation Layers

### Layer 1 — Projection sanity

Inspect:

```text
output dimensions
target location
orientation
mirroring
large holes
foreground/background ordering
```

Common failure causes:

```text
wrong transform direction
translation-unit mismatch
depth-unit mismatch
row/column swap
wrong optical-frame convention
wrong camera matrix
wrong stream profile
```

### Layer 2 — SDK comparison

Compare custom aligned depth with SDK aligned depth using:

```text
valid-pixel overlap
depth difference on common valid pixels
foreground-mask IoU
boundary-distance error
left/right/top/bottom offsets
```

Exact image equality is not required because rasterization and post-processing may differ.

### Layer 3 — RGB boundary comparison

Compare both custom and SDK aligned-depth boundaries against the RGB target boundary.

Interpretation:

```text
custom wrong, SDK correct
→ likely custom implementation error

custom and SDK similar, both shifted
→ likely calibration, occlusion, or synchronization behavior

custom and SDK differ mainly in holes
→ likely rasterization or filtering difference
```

---

## 15. Pairing Separate Off/On Bags

If registration-off and registration-on data are in separate bags:

```text
keep camera fixed
keep target fixed
keep lighting stable
use a static target
```

Exact cross-bag frame correspondence is unnecessary for basic static geometry comparison.

Do not compare separately recorded dynamic scenes.

---

## 16. Initial Dataset Inventory

Preferred initial recordings:

```text
d2c_d100_center_registration_off_r01
d2c_d100_center_registration_on_r01
```

After the first pair passes:

```text
d2c_d100_center_registration_off_r02
d2c_d100_center_registration_on_r02
d2c_d100_center_registration_off_r03
d2c_d100_center_registration_on_r03
```

Total if separate modes are required:

```text
6 bags
```

If raw and aligned depth are available simultaneously:

```text
3 bags
```

---

## 17. Per-Bag Metadata

```yaml
experiment:
  name: d2c_d100_center_registration_off_r01
  type: d2c_transformation_validation
  repeat: 1

registration:
  enabled: false

geometry:
  camera_to_foreground_mm: 1000
  position: center
  yaw_deg: 0
  foreground_background_separation_mm: 300

color:
  topic: /camera/color/image_raw
  width: 1280
  height: 720

depth:
  topic: /camera/depth/image_raw
  width: 848
  height: 480
  depth_scale_m_per_unit: 0.001

extrinsic:
  source: orbbec_extrinsic_topic
  convention: depth_to_color
```

---

## 18. Validation Criteria

The implementation is considered validated when:

```text
synthetic tests pass
projection orientation is correct
transform direction and units are confirmed
custom output structurally matches SDK output
boundary location is consistent with SDK output
results repeat across recordings
```

Do not require exact pixel equality.

---

## 19. Immediate Next Tasks

```text
1. Confirm registration-off raw topics
2. Confirm registration-on aligned topics
3. Check whether raw and aligned depth can publish simultaneously
4. Record camera_info for both streams
5. Locate the exact depth-to-color extrinsic source
6. Confirm extrinsic direction and translation unit
7. Record one registration-off bag
8. Record one registration-on bag without moving the setup
9. Implement synthetic tests
10. Implement back-projection
11. Implement rigid transformation
12. Implement color projection
13. Implement z-buffer rasterization
14. Generate custom aligned depth
15. Compare against SDK aligned depth
16. Expand to repeats only after the first pair passes
```

---

## 20. Current Milestones

First milestone:

> Generate a color-resolution custom aligned-depth image from one registration-off bag and verify target location and orientation.

Second milestone:

> Compare the custom aligned-depth result against one SDK-aligned recording of the same static scene.
