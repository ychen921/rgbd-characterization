# RGB–Depth Alignment Analysis Plan

## 1. Purpose

This document defines the RGB–Depth alignment characterization pipeline.

The goal is to measure whether the same physical boundary is projected to the same pixel location in:

```text
RGB image
aligned depth image
```

This is separate from depth baseline characterization.

The depth baseline pipeline evaluates:

```text
depth quality
measured depth
temporal noise
plane fitting
planarity
```

The alignment pipeline evaluates:

```text
RGB/depth spatial registration
pixel-domain edge displacement
image-center versus image-corner behavior
optional dynamic synchronization behavior
```

Related document:

```text
docs/baseline_analysis_plan.md
```

---

## 2. Scope

### 2.1 Included

```text
static RGB–Depth spatial alignment
RGB/depth frame pairing
alignment ROI management
RGB foreground extraction
depth foreground extraction
edge correspondence
pixel-offset metrics
center and corner comparison
distance comparison
alignment visualizations
```

### 2.2 Deferred

```text
dynamic target synchronization
hardware-versus-software registration comparison
automatic calibration refinement
extrinsic re-calibration
multi-camera alignment
```

Dynamic synchronization may be added after the static pipeline is validated.

---

## 3. Experimental Principle

Use a foreground target with a clear geometric boundary and a background at a different depth.

Recommended setup:

```text
RGB-D camera
↓
dark rectangular foreground target
↓ 20–40 cm depth separation
light background board
```

The foreground target should provide:

```text
left edge
right edge
top edge
bottom edge
four corners
```

The target should have strong RGB contrast against the background and a clear depth discontinuity.

A uniform white board used for depth baseline characterization is insufficient because it does not provide a reliable RGB/depth boundary correspondence.

---

## 4. Recommended Test Matrix

Initial validation:

```yaml
distances_m:
  - 0.5
  - 1.0
  - 2.0

positions:
  - center

yaw_deg:
  - 0

repeats: 3
duration_sec: 5
```

Extended static validation:

```yaml
distances_m:
  - 0.5
  - 1.0
  - 2.0

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

Optional angle extension:

```yaml
yaw_deg:
  - -30
  - 0
  - 30
```

Do not begin with the full distance × position × angle matrix. Validate the center-position pipeline first.

---

## 5. Experiment Naming

Recommended convention:

```text
scene04_alignment_d050_center_yaw00_r01
scene04_alignment_d100_center_yaw00_r01
scene04_alignment_d100_top_left_yaw00_r01
scene04_alignment_d100_center_yaw30_r01
```

Recommended parsing fields:

```text
scene
experiment_type
distance
position
yaw
repeat
```

Example experiment metadata:

```yaml
experiment:
  name: scene04_alignment_d100_center_yaw00_r01
  type: rgb_depth_alignment
  distance_mm: 1000
  position: center
  yaw_deg: 0
  repeat: 1
```

---

## 6. Required ROS Data

Record or extract:

```text
color image
aligned depth image
unaligned depth image, optional but recommended
color camera_info
depth camera_info
timestamps
```

Example topics:

```text
/camera/color/image_raw
/camera/color/camera_info
/camera/depth/image_raw
/camera/depth/image_unaligned
/camera/depth/camera_info
```

The semantic meaning of `/camera/depth/image_raw` must be verified from launch configuration and image dimensions.

Do not infer alignment state only from the topic name.

Store alignment configuration in experiment metadata:

```yaml
alignment:
  registration_enabled: true
  aligned_depth_topic: /camera/depth/image_raw
  unaligned_depth_topic: /camera/depth/image_unaligned
  registration_mode: unknown
```

Replace `unknown` after confirming whether registration is performed by the device, SDK, or ROS node.

---

## 7. Extracted Dataset Format

Recommended directory:

```text
data/
└── scene04_alignment_d100_center_yaw00_r01/
    ├── rgb.npz
    ├── aligned_depth.npz
    ├── timestamps.npz
    ├── color_camera_info.yaml
    ├── depth_camera_info.yaml
    └── experiment.yaml
```

Recommended arrays:

```text
rgb:
shape (N_rgb, H_rgb, W_rgb, 3)
dtype uint8

aligned_depth:
shape (N_depth, H_depth, W_depth)
dtype uint16

rgb_timestamp_ns:
shape (N_rgb,)

depth_timestamp_ns:
shape (N_depth,)
```

If RGB and aligned depth are already paired during extraction, also store:

```text
pair_rgb_index
pair_depth_index
pair_delta_ms
```

Full-resolution source arrays should remain unchanged.

---

## 8. Project Structure

```text
rgbd-characterization/
├── config/
│   └── roi/
│       ├── baseline/
│       └── alignment/
│           ├── scene04_alignment_d050_center_yaw00.yaml
│           └── ...
│
├── tools/
│   ├── extract_data.py
│   ├── select_roi.py
│   ├── analyze_baseline.py
│   ├── analyze_alignment.py
│   └── summarize_alignment.py
│
└── src/
    ├── io/
    │   ├── dataset.py
    │   └── synchronized_dataset.py
    │
    ├── preprocessing/
    │   ├── roi.py
    │   ├── depth.py
    │   ├── rgb.py
    │   └── frame_pairing.py
    │
    ├── segmentation/
    │   ├── rgb_foreground.py
    │   └── depth_foreground.py
    │
    ├── metrics/
    │   └── alignment.py
    │
    └── visualization/
        └── alignment.py
```

Do not add RGB-specific logic to `tools/analyze_baseline.py`.

---

## 9. Frame Pairing

Implement:

```text
src/preprocessing/frame_pairing.py
```

Initial strategy:

```text
for each RGB frame
↓
find nearest depth timestamp
↓
accept pair if |Δt| <= max_pair_delta_ms
```

Recommended configuration:

```yaml
frame_pairing:
  method: nearest_timestamp
  max_pair_delta_ms: 20.0
```

Record for every accepted pair:

```text
rgb_index
depth_index
rgb_timestamp_ns
depth_timestamp_ns
delta_ms
```

Reject pairs exceeding the threshold.

Do not silently pair frames with excessive timestamp differences.

Recommended result model:

```python
@dataclass(frozen=True)
class FramePair:
    rgb_index: int
    depth_index: int
    rgb_timestamp_ns: int
    depth_timestamp_ns: int
    delta_ms: float
```

Initial summary:

```text
number of RGB frames
number of depth frames
accepted pairs
rejected RGB frames
median absolute timestamp delta
p95 absolute timestamp delta
maximum absolute timestamp delta
```

Static alignment should still report timestamp deltas even when the target is stationary.

---

## 10. Alignment ROI

Alignment ROI must include the foreground target boundary.

This differs from baseline ROI:

```text
baseline ROI
→ exclude board edges
→ preserve planar interior

alignment ROI
→ include target edges
→ preserve foreground/background transition
```

Store alignment ROI files under:

```text
config/roi/alignment/
```

Recommended key:

```text
experiment name without repeat suffix
```

Example:

```text
scene04_alignment_d100_center_yaw00_r01
↓
scene04_alignment_d100_center_yaw00
↓
config/roi/alignment/scene04_alignment_d100_center_yaw00.yaml
```

Recommended YAML:

```yaml
name: scene04_alignment_d100_center_yaw00

source:
  experiment: scene04_alignment_d100_center_yaw00_r01
  rgb_frame_index: 120
  depth_frame_index: 119

roi:
  type: rectangle
  x: 180
  y: 100
  width: 280
  height: 260
```

The selected ROI must use the common aligned image coordinate system.

---

## 11. RGB Foreground Extraction

Implement:

```text
src/segmentation/rgb_foreground.py
```

Recommended initial target:

```text
dark rectangle
light background
```

Initial method:

```text
RGB or grayscale conversion
↓
intensity threshold
↓
morphological cleanup
↓
largest valid contour
↓
foreground mask
↓
rectangle or polygon fit
```

Do not begin with a general semantic segmentation model.

Recommended output:

```python
@dataclass(frozen=True)
class RGBForegroundResult:
    mask: np.ndarray
    contour: np.ndarray
    edges: np.ndarray
    bbox_xyxy: tuple[int, int, int, int]
    area_px: int
```

Validation conditions:

```text
exactly one dominant target
target area above minimum
target not clipped by ROI
contour approximately rectangular
```

---

## 12. Depth Foreground Extraction

Implement:

```text
src/segmentation/depth_foreground.py
```

Preferred first method:

```text
foreground/background depth threshold
```

For a known foreground and background separation:

```text
foreground depth < threshold < background depth
```

The threshold may be estimated from the depth histogram inside the ROI.

Alternative fallback:

```text
depth gradient
Sobel or finite difference
depth-jump threshold
```

Recommended initial method:

```text
valid depth filtering
↓
two-cluster or histogram separation
↓
foreground mask
↓
largest connected component
↓
depth boundary
```

Do not classify:

```text
0
65535
```

as valid foreground samples.

Recommended output:

```python
@dataclass(frozen=True)
class DepthForegroundResult:
    mask: np.ndarray
    edges: np.ndarray
    bbox_xyxy: tuple[int, int, int, int]
    valid_ratio: float
    threshold_mm: float
```

---

## 13. Occlusion and Edge Validity

RGB and depth sensors observe the scene from different physical viewpoints.

Therefore some boundary disagreement is expected near occlusion edges, especially at short distance.

Do not treat every unmatched edge pixel as calibration error.

Recommended handling:

```text
compute metrics for all four sides separately
identify consistently invalid or occluded sides
retain both full-boundary and selected-side summaries
```

A side may be marked unusable when:

```text
depth invalid ratio exceeds threshold
foreground or background is missing
edge is clipped by ROI
severe flying-pixel contamination is present
```

Recommended output:

```text
left_edge_valid
right_edge_valid
top_edge_valid
bottom_edge_valid
```

---

## 14. Alignment Metrics

Implement:

```text
src/metrics/alignment.py
```

### 14.1 Bounding-Edge Offset

For RGB and depth bounding boxes:

```text
RGB:
left_rgb
right_rgb
top_rgb
bottom_rgb

Depth:
left_depth
right_depth
top_depth
bottom_depth
```

Calculate:

```text
left_offset_px = left_depth - left_rgb
right_offset_px = right_depth - right_rgb
top_offset_px = top_depth - top_rgb
bottom_offset_px = bottom_depth - bottom_rgb
```

Interpretation:

```text
positive horizontal offset
→ depth edge lies to the right of RGB edge

positive vertical offset
→ depth edge lies below RGB edge
```

This should be the first metric implemented because it is simple and interpretable.

### 14.2 Boundary Distance

For each valid depth-boundary pixel:

```text
distance to nearest RGB-boundary pixel
```

Recommended summaries:

```text
median boundary error in pixels
mean boundary error in pixels
p95 boundary error in pixels
```

Use a distance transform for efficient computation.

Do not use maximum boundary error as the primary metric.

### 14.3 Mask Overlap

Calculate:

```text
intersection over union
```

IoU is a secondary metric.

A large target can retain high IoU despite a visible 2–3 pixel displacement.

### 14.4 Per-Side Metrics

Record separate statistics for:

```text
left
right
top
bottom
```

This helps distinguish:

```text
global translation
scale mismatch
distortion behavior
occlusion behavior
```

---

## 15. Per-Frame Result

Recommended model:

```python
@dataclass(frozen=True)
class FrameAlignmentResult:
    rgb_index: int
    depth_index: int
    timestamp_delta_ms: float

    left_offset_px: float
    right_offset_px: float
    top_offset_px: float
    bottom_offset_px: float

    median_boundary_error_px: float
    p95_boundary_error_px: float
    mask_iou: float

    depth_valid_ratio: float

    left_edge_valid: bool
    right_edge_valid: bool
    top_edge_valid: bool
    bottom_edge_valid: bool
```

Frames failing segmentation or validity checks must be marked invalid rather than assigned fabricated metrics.

---

## 16. Experiment Summary

Recommended summary:

```yaml
dataset:
  experiment: scene04_alignment_d100_center_yaw00_r01
  experiment_type: rgb_depth_alignment

frame_pairing:
  accepted_pairs: 148
  rejected_rgb_frames: 2
  median_abs_delta_ms: 1.2
  p95_abs_delta_ms: 2.8
  max_abs_delta_ms: 4.1

roi:
  key: scene04_alignment_d100_center_yaw00
  config: config/roi/alignment/scene04_alignment_d100_center_yaw00.yaml
  x: 180
  y: 100
  width: 280
  height: 260

valid_frames:
  total_pairs: 148
  valid_alignment_frames: 142
  valid_ratio: 0.959

alignment:
  left_offset_px:
    median: 1.2
    p05: 0.8
    p95: 1.9

  right_offset_px:
    median: 1.5
    p05: 0.9
    p95: 2.3

  top_offset_px:
    median: -0.4
    p05: -1.0
    p95: 0.2

  bottom_offset_px:
    median: -0.2
    p05: -0.8
    p95: 0.5

  boundary_error_px:
    median: 1.3
    p95: 2.8

  mask_iou:
    median: 0.975

depth:
  median_valid_ratio: 0.993
```

These values are examples only and must not be used as acceptance thresholds.

---

## 17. Visualization Outputs

For selected frames save:

```text
rgb.png
aligned_depth_colormap.png
rgb_mask.png
depth_mask.png
edge_overlay.png
mask_overlay.png
```

Recommended edge overlay:

```text
RGB boundary
depth boundary
bounding boxes
per-side offset labels
timestamp delta
```

Also save:

```text
frame_alignment_metrics.csv
pairing_metrics.csv
summary.yaml
```

Recommended result structure:

```text
results/
└── scene04_alignment_d100_center_yaw00_r01/
    └── alignment/
        ├── summary.yaml
        ├── frame_alignment_metrics.csv
        ├── frame_pairing.csv
        └── visualizations/
            ├── frame_000120_edge_overlay.png
            └── ...
```

---

## 18. Alignment Analysis Tool

Implement:

```text
tools/analyze_alignment.py
```

Workflow:

```text
dataset directory
↓
load experiment metadata
↓
verify experiment_type == rgb_depth_alignment
↓
load RGB and aligned depth
↓
pair frames by timestamp
↓
load alignment ROI
↓
for each accepted pair:
    crop RGB and aligned depth ROI
    extract RGB foreground
    extract depth foreground
    validate edges
    compute alignment metrics
    save selected visualizations
↓
aggregate experiment summary
↓
save CSV and YAML
```

The analyzer must fail clearly when:

```text
aligned depth is missing
RGB is missing
timestamps are missing
ROI is missing
image coordinate systems are incompatible
```

Do not automatically resize an unaligned depth image to RGB resolution and call it aligned.

---

## 19. Synthetic and Controlled Tests

### 19.1 Identical Masks

Expected:

```text
all side offsets = 0
boundary median = 0
boundary p95 = 0
IoU = 1
```

### 19.2 Known Translation

Shift depth mask:

```text
dx = +3 px
dy = -2 px
```

Expected:

```text
left/right offsets ≈ +3 px
top/bottom offsets ≈ -2 px
```

### 19.3 Scale Difference

Enlarge depth rectangle while preserving center.

Expected:

```text
left and top offsets negative
right and bottom offsets positive
```

This validates the ability to distinguish translation from scale mismatch.

### 19.4 Invalid Depth Edge

Inject invalid depth on one side.

Expected:

```text
affected side marked invalid
remaining sides still reported
```

### 19.5 Timestamp Pairing

Create timestamp sequences with known nearest neighbors and threshold failures.

Expected:

```text
correct pair indices
correct delta_ms
out-of-threshold pairs rejected
```

---

## 20. First Validation Workflow

Use:

```text
scene04_alignment_d100_center_yaw00_r01
```

first.

Recommended order:

```text
1. Confirm RGB topic, aligned-depth topic, and resolutions
2. Confirm timestamps are extracted
3. Implement frame pairing
4. Add alignment ROI selection
5. Manually verify RGB and depth crops
6. Implement RGB foreground extraction
7. Implement depth foreground extraction
8. Implement bounding-edge offsets
9. Generate edge overlay
10. Validate one frame manually
11. Validate multiple frames
12. Add boundary-distance and IoU metrics
13. Save experiment summary
```

Do not begin with all distances or image-corner positions.

---

## 21. Multi-Repeat and Multi-Position Workflow

After the first dataset passes:

```text
analyze all d100 center repeats
↓
verify repeatability
↓
analyze d050 and d200 center
↓
compare distance behavior
↓
add four image-corner positions
↓
compare center versus corners
```

Recommended aggregation fields:

```text
distance_mm
position
yaw_deg
repeat
valid_frame_ratio
median timestamp delta
median left/right/top/bottom offset
median boundary error
p95 boundary error
median IoU
median depth valid ratio
```

Recommended plots:

```text
distance vs boundary error
distance vs horizontal offset
distance vs vertical offset
image position vs boundary error
image position vs side offsets
repeat vs alignment stability
```

---

## 22. Static Versus Dynamic Interpretation

Static alignment isolates spatial registration behavior.

Dynamic alignment includes both:

```text
spatial registration
temporal synchronization
```

Interpretation:

```text
static aligned, dynamic misaligned
→ likely timestamp or synchronization issue

static and dynamic both misaligned
→ likely registration, calibration, scaling, or image-coordinate issue
```

Dynamic testing should use a later extension document or an added section after static validation is complete.

---

## 23. Acceptance Threshold Policy

Do not define a universal pass/fail threshold before collecting baseline data.

First compare:

```text
center versus corners
near versus far distance
repeat-to-repeat stability
horizontal versus vertical offsets
valid versus occluded sides
```

Initial results should be descriptive.

Acceptance thresholds may later be defined from:

```text
application tolerance
target size in pixels
working distance
downstream projection requirements
repeatability distribution
```

Avoid using:

```text
maximum error
```

as the primary acceptance metric.

Prefer:

```text
median
p95
valid-frame ratio
per-side offset stability
```

---

## 24. Immediate Next Tasks

```text
1. Confirm the aligned depth topic and image dimensions

2. Extend extraction to save:
   - RGB frames
   - aligned depth frames
   - RGB timestamps
   - depth timestamps
   - camera info
   - experiment metadata

3. Implement src/preprocessing/frame_pairing.py

4. Test timestamp pairing with synthetic data

5. Create config/roi/alignment/

6. Extend select_roi.py to support:
   --roi-type alignment

7. Select ROI for:
   scene04_alignment_d100_center_yaw00_r01

8. Implement src/segmentation/rgb_foreground.py

9. Implement src/segmentation/depth_foreground.py

10. Implement initial bounding-edge offsets

11. Implement edge-overlay visualization

12. Validate one paired frame manually

13. Implement boundary-distance metrics

14. Implement mask IoU

15. Implement tools/analyze_alignment.py

16. Validate all d100 center repeats

17. Add d050 and d200 center tests

18. Add four image-corner positions

19. Implement tools/summarize_alignment.py
```

---

## 25. Current Milestone

The first alignment milestone is:

> Successfully pair RGB and aligned-depth frames for `scene04_alignment_d100_center_yaw00_r01`, extract one valid foreground mask from each modality, and produce an edge overlay with left/right/top/bottom pixel offsets.

Required pipeline:

```text
rgb.npz + aligned_depth.npz + timestamps
↓
nearest-timestamp frame pairing
↓
load alignment ROI
↓
RGB foreground mask
↓
depth foreground mask
↓
bounding-edge offsets
↓
edge overlay
↓
frame_alignment_metrics.csv
```

Only after this milestone passes should the pipeline add:

```text
boundary-distance metrics
multi-repeat analysis
multi-distance analysis
image-corner tests
dynamic synchronization tests
```
