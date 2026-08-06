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
docs/rgb_depth_alignment_dataset_plan.md
docs/d2c_transformation_validation_plan.md
```

Document responsibilities:

```text
rgb_depth_alignment_dataset_plan.md
→ formal scene05 dataset collection using sensor or SDK aligned depth

rgb_depth_alignment_plan.md
→ analysis of the formal aligned-depth dataset

d2c_transformation_validation_plan.md
→ diagnostic validation of a custom raw-depth-to-color transformation
```

The formal alignment analyzer consumes depth that is already mapped to the
documented color-image coordinate system. It does not implement raw-depth
back-projection, extrinsic transformation, color-camera projection, or
z-buffer rasterization.

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

Custom depth-to-color transformation validation is handled separately under:

```text
data/diagnostics/d2c_transform/
docs/d2c_transformation_validation_plan.md
```

---

## 3. Experimental Principle

Use a foreground target with a clear geometric boundary and a background at a different depth.

Recommended setup:

```text
RGB-D camera
↓
dark rectangular foreground target
↓ 200–400 mm depth separation
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
distances_mm:
  - 1000

positions:
  - center

yaw_deg:
  - 0

repeats: 3
duration_sec: 5
```

Extended static validation:

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

Optional angle extension:

```yaml
yaw_deg:
  - -30
  - 30
```

Do not begin with the full distance × position × angle matrix. Validate the center-position pipeline first.

The formal collection phases and bag counts are defined in:

```text
docs/rgb_depth_alignment_dataset_plan.md
```

---

## 5. Experiment Naming

Recommended convention:

```text
scene05_alignment_d050_center_yaw00_r01
scene05_alignment_d100_center_yaw00_r01
scene05_alignment_d100_top_left_yaw00_r01
scene05_alignment_d100_center_yawp30_r01
scene05_alignment_d100_center_yawm30_r01
```

Yaw tokens:

```text
yaw00  →   0°
yawp30 → +30°
yawm30 → -30°
```

Do not use `+` or `-` in directory names.

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
  name: scene05_alignment_d100_center_yaw00_r01
  type: rgb_depth_alignment
  scene: 5
  repeat: 1

geometry:
  camera_to_foreground_mm: 1000
  image_position: center
  yaw_deg: 0

registration:
  enabled: true
  aligned_depth_topic: /camera/depth/image_raw
  mode: sdk
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
/tf_static
camera node parameter dump
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
Do not infer alignment state only from resolution or `frame_id`.

Store registration configuration in experiment metadata:

```yaml
registration:
  enabled: true
  aligned_depth_topic: /camera/depth/image_raw
  unaligned_depth_topic: /camera/depth/image_unaligned
  mode: sdk
  aligned_to: color
  coordinate_convention: color_pixel_grid
```

During collection, `registration.mode` may temporarily be `unknown`. Formal
analysis must require it to be resolved to `device`, `sdk`, or `ros_node`,
unless an explicit diagnostic override is used and recorded in the summary.

Before analysis, confirm:

```text
registration is enabled
aligned-depth topic semantics are documented
aligned-depth coordinate convention is documented
image dimensions match the declared coordinate mapping
color and depth timestamps share a documented clock domain
```

---

## 7. Extracted Dataset Format

Recommended directory:

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
    └── extraction_summary.yaml
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

rgb_recorded_timestamp_ns:
shape (N_rgb,)

depth_recorded_timestamp_ns:
shape (N_depth,)
```

If RGB and aligned depth are already paired during extraction, also store:

```text
pair_rgb_index
pair_depth_index
pair_delta_ms
```

Full-resolution source arrays should remain unchanged.

Recommended dataset contract:

```yaml
schema_version: 1

color:
  array_key: rgb
  timestamp_key: rgb_timestamp_ns
  recorded_timestamp_key: rgb_recorded_timestamp_ns
  encoding: rgb8
  channel_order: RGB

depth:
  array_key: aligned_depth
  timestamp_key: depth_timestamp_ns
  recorded_timestamp_key: depth_recorded_timestamp_ns
  encoding: 16UC1
  precision: 1mm
  unit: mm
  invalid_values: [0, 65535]

timestamps:
  unit: ns
  primary_source: message_header
  recorded_source: rosbag_storage
  header_clock_domain: global
```

Replace timestamp fields with the actual capture semantics when they differ.
Do not silently assume that bag receive time and message-header time are
equivalent.

The alignment bag reader must preserve both timestamp sources and must not pair
frames during extraction. It must also validate that color and aligned depth
share image dimensions, frame ID, and CameraInfo projection fields (`K`, `R`,
and `P`). Modality-specific distortion coefficients and ROI rectification flags
may differ: the current Orbbec SW-aligned depth reports the color projection
with zero distortion coefficients and `do_rectify: true`, while the raw color
CameraInfo retains its distortion coefficients.

`tools/extract_alignment_dataset.py` preallocates the declared stream arrays,
writes the dataset and traceability metadata into a same-filesystem staging
directory, reload-validates the three NPZ archives, and atomically publishes a
new output directory. Existing output directories are never overwritten.

---

## 8. Project Structure

```text
rgbd-characterization/
├── config/
│   └── roi/
│       ├── baseline/
│       └── alignment/
│           ├── scene05_alignment_d050_center_yaw00.yaml
│           └── ...
│
├── scripts/
│   ├── record_experiment.sh
│   └── record_alignment_experiment.sh
│
├── tools/
│   ├── extract_dataset.py
│   ├── extract_alignment_dataset.py
│   ├── select_roi.py
│   ├── analyze_baseline.py
│   ├── analyze_alignment.py
│   └── summarize_alignment.py
│
└── src/
    ├── io/
    │   ├── dataset.py
    │   ├── alignment_dataset.py
    │   ├── ros_image.py
    │   └── alignment_bag_reader.py
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

Recording-script responsibilities:

```text
scripts/record_experiment.sh
→ preserve the existing depth-baseline and legacy recording workflow

scripts/record_alignment_experiment.sh
→ perform alignment Phase 0 preflight and record formal scene05 bags

future scripts/record_d2c_validation.sh
→ record registration-off/on diagnostic data under diagnostics/d2c_transform
```

Do not add custom D2C transformation recording modes to the formal alignment
recorder. Shared shell helpers may be extracted later if recorder duplication
becomes significant, but the first alignment implementation should not require
refactoring the existing recorder.

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
accept pair if |Δt| <= max_abs_delta_ms
```

Recommended configuration:

```yaml
frame_pairing:
  method: nearest_timestamp
  delta_definition: depth_minus_rgb
  max_abs_delta_ms: 20.0
  threshold_inclusive: true
  cardinality: one_to_one
  preserve_order: true
  tie_breaker: earlier_depth
```

Define:

```text
delta_ms = (depth_timestamp_ns - rgb_timestamp_ns) / 1e6
```

The acceptance threshold is applied to `abs(delta_ms)`. Store the signed
delta in per-pair output so that a systematic stream delay remains visible.
The initial implementation must not reuse a depth frame for multiple RGB
frames.

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
scene05_alignment_d100_center_yaw00_r01
↓
scene05_alignment_d100_center_yaw00
↓
config/roi/alignment/scene05_alignment_d100_center_yaw00.yaml
```

Recommended YAML:

```yaml
schema_version: 1
name: scene05_alignment_d100_center_yaw00

source:
  experiment: scene05_alignment_d100_center_yaw00_r01
  rgb_frame_index: 120
  depth_frame_index: 119

coordinate_system:
  pixel_grid: color
  origin: top_left
  indexing: zero_based
  interval: half_open

roi:
  type: rectangle
  x: 180
  y: 100
  width: 280
  height: 260
```

The selected ROI must use the common aligned image coordinate system.
Rectangle bounds use `[x, x + width)` and `[y, y + height)`.

Preferred input:

```text
aligned depth width  == RGB width
aligned depth height == RGB height
```

If the dimensions differ, analysis is allowed only when the dataset documents
a validated conversion into one canonical pixel grid. The ROI and all metrics
must use that grid. The initial analyzer may reject such datasets clearly
rather than attempt an inferred scale conversion.

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

Every accepted pair must produce one CSV row, including failed analyses.
Recommended status fields:

```text
pairing_status
analysis_status
failure_reason
```

Recommended `analysis_status` values:

```text
valid
rgb_segmentation_failed
depth_segmentation_failed
target_clipped
insufficient_valid_depth
no_valid_edges
incompatible_dimensions
```

Invalid metrics must be empty or `NaN`, not zero. Zero is a valid measurement
that means no observed offset.

---

## 16. Experiment Summary

Recommended summary:

```yaml
dataset:
  experiment: scene05_alignment_d100_center_yaw00_r01
  experiment_type: rgb_depth_alignment

frame_pairing:
  accepted_pairs: 148
  rejected_rgb_frames: 2
  median_abs_delta_ms: 1.2
  p95_abs_delta_ms: 2.8
  max_abs_delta_ms: 4.1

roi:
  key: scene05_alignment_d100_center_yaw00
  config: config/roi/alignment/scene05_alignment_d100_center_yaw00.yaml
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
└── scene05_alignment_d100_center_yaw00_r01/
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
registration is disabled or unresolved
aligned-depth coordinate convention is undocumented
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
scene05_alignment_d100_center_yaw00_r01
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

## 24. Development Phases

Use a gated development workflow. Complete and review the exit conditions of
each phase before expanding the implementation or dataset.

### Phase 0 — Data and Device Contract

Implement a standalone alignment recorder:

```text
scripts/record_alignment_experiment.sh
```

The recorder owns both the Phase 0 runtime preflight and formal scene05 bag
recording. It must not change the existing behavior of
`scripts/record_experiment.sh`.

Recommended invocation:

```bash
scripts/record_alignment_experiment.sh \
  --distance-mm 1000 \
  --position center \
  --yaw-deg 0 \
  --repeat 1 \
  --duration 5 \
  --camera-node /camera/camera \
  --registration-param depth_registration \
  --registration-mode sdk
```

The actual camera node and registration parameter must be verified for the
installed wrapper and must not be accepted as undocumented assumptions.

For the Gemini 330 series, start the camera with an explicit depth-unit
contract before running the recorder:

```bash
ros2 launch orbbec_camera gemini_330_series.launch.py \
  depth_registration:=true \
  depth_precision:=1mm
```

Required experiment arguments:

```text
distance_mm
position: center | top_left | top_right | bottom_left | bottom_right
yaw_deg
repeat
```

The recorder derives the formal experiment name, including:

```text
0°   → yaw00
+30° → yawp30
-30° → yawm30
```

Before creating a formal bag, run the following preflight:

```text
RGB and aligned-depth topics
registration enabled and mode resolved
aligned depth mapped to the documented color pixel grid
image dimensions and encodings
depth unit and invalid values
timestamp source and clock domain
camera info, camera parameters, and /tf_static retained
experiment metadata schema
```

Required topics:

```text
/camera/color/image_raw
/camera/color/camera_info
/camera/depth/image_raw
/camera/depth/camera_info
/tf_static
```

Optional topics:

```text
/camera/depth/image_unaligned
/camera/depth_to_color
/diagnostics
/camera/device_status
```

A missing required topic is a failure. A missing optional topic is a warning
and the unavailable topic must not be passed to `ros2 bag record`.

Preflight policy:

```text
PASS → continue to recording
WARN → record the warning and continue
FAIL → stop before creating a formal bag
```

Run preflight in a temporary staging directory. For `PASS` or `WARN`, move the
report and captured parameters into the new experiment directory before
recording. For `FAIL`, do not create the formal experiment directory; preserve
the failure report separately under:

```text
results/preflight/
└── scene05_alignment_d100_center_yaw00_r01_preflight.yaml
```

The first recorder implementation must check:

```text
required commands and arguments
experiment directory does not already exist
required and optional topic availability
camera parameter dump succeeds
the configured registration parameter is enabled
registration mode is not unknown
RGB and depth streams publish messages
RGB and aligned-depth dimensions match
encodings and camera-info dimensions are consistent
depth unit and invalid-value policy are documented
timestamp source, unit, and clock domain are documented
```

The same script then records the bag using only verified topics and stops
`ros2 bag record` gracefully with `SIGINT`.

Recommended output:

```text
bags/
└── scene05_alignment_d100_center_yaw00_r01/
    ├── rosbag/
    ├── experiment.yaml
    ├── camera_params.yaml
    ├── preflight.yaml
    └── post_recording.yaml
```

`preflight.yaml` must contain an overall result plus individual checks,
warnings, errors, and the evidence used to confirm registration. A formal
experiment directory may contain only a `pass` or `warn` preflight result; a
`fail` result is stored only under `results/preflight/`.
`post_recording.yaml` must initially record the requested duration, recorder
exit status, bag path, and recorded topics. Frame counts, actual duration, and
timestamp ranges may be added after the first recorder version is validated.

Exit condition:

```text
one dataset can be loaded without inferred registration, coordinate, depth-unit,
or timestamp semantics
the standalone recorder produces preflight.yaml, experiment.yaml,
camera_params.yaml, post_recording.yaml, and a gracefully closed rosbag
the existing scripts/record_experiment.sh workflow remains unchanged
```

### Phase 1 — Phase A Recording and Extraction

Record and extract:

```text
scene05_alignment_d100_center_yaw00_r01
scene05_alignment_d100_center_yaw00_r02
scene05_alignment_d100_center_yaw00_r03
```

Use `scripts/record_alignment_experiment.sh` for all three repeats. Do not use
the legacy recorder for formal scene05 data.

Exit condition:

```text
all required files and metadata are present
RGB and aligned depth decode correctly
frame counts and resolutions are plausible
the complete target boundary is visible
```

### Phase 2 — Frame Pairing

Implement one-to-one, order-preserving nearest-timestamp pairing and synthetic
tests.

Exit condition:

```text
known timestamp sequences produce the expected pairs and signed delta_ms
threshold failures are rejected
depth frames are not reused
pairing summary can be reproduced from frame_pairing.csv
```

### Phase 3 — Alignment ROI

Add alignment ROI selection and select the shared d100-center ROI in the color
pixel grid.

Exit condition:

```text
the ROI contains target interior, all four edges, and surrounding background
the same ROI is valid for all three repeats
paired RGB and depth crops pass manual review
```

### Phase 4 — Single-Frame RGB Segmentation

Implement dark-foreground extraction, cleanup, dominant-contour selection, and
target validation.

Exit condition:

```text
one selected RGB frame produces a complete target mask and four plausible edges
failure cases return an explicit status and reason
```

### Phase 5 — Single-Frame Depth Segmentation

Implement invalid-depth filtering, foreground/background separation, connected
component selection, and depth-edge validity.

Exit condition:

```text
the paired depth frame produces a complete foreground mask
invalid depth is not classified as foreground
failure cases return an explicit status and reason
```

### Phase 6 — Initial Alignment Metrics

Implement:

```text
left/right/top/bottom bounding-edge offsets
per-side validity
depth valid ratio
edge overlay
```

Validate identical masks, known translation, scale difference, and an invalid
depth edge.

Exit condition:

```text
synthetic results match their expected signs and values
one real paired frame passes manual edge-overlay review
```

This is the first alignment milestone.

### Phase 7 — Single-Repeat Analyzer

Integrate loading, validation, pairing, ROI, segmentation, metrics,
visualization, and summary generation in `tools/analyze_alignment.py`.

Exit condition:

```text
every accepted pair produces a CSV row
invalid frames retain status and failure reason
summary values can be recomputed from the CSV files
repeated analysis is deterministic
```

### Phase 8 — Phase A Repeatability

Analyze all three d100-center repeats.

Exit condition:

```text
all repeats complete analysis
valid-frame ratio and per-side stability are reported
segmentation parameters do not depend on manual per-frame tuning
repeat differences are quantified or explained
```

Completion of this phase completes formal dataset Phase A.

### Phase 9 — Boundary Metrics

Add:

```text
boundary-distance median, mean, and p95
optional symmetric RGB-to-depth and depth-to-RGB boundary distance
mask IoU
per-side boundary statistics
```

Exit condition:

```text
identical and translated synthetic masks produce expected results
metrics are added to per-frame and experiment summaries
```

### Phase 10 — Center Distance Comparison

Analyze:

```text
d050 center × 3 repeats
d100 center × 3 repeats
d200 center × 3 repeats
```

Exit condition:

```text
all distances use the same analysis pipeline
distance behavior and near-range occlusion effects are reported
```

### Phase 11 — Center and Corner Comparison

Add all four image-corner positions and complete the Phase B matrix.

Exit condition:

```text
all 45 Phase B bags have metadata and ROI coverage
all conditions pass basic validation or have an explicit rejection reason
center-versus-corner behavior can be aggregated
```

### Phase 12 — Cross-Experiment Summary

Implement `tools/summarize_alignment.py` to aggregate distance, position, yaw,
repeat, validity, pairing quality, per-side offsets, boundary error, and IoU.

Exit condition:

```text
aggregate values remain traceable to source experiments
missing and failed conditions are visible
tables, plots, CSV, and YAML summaries agree
```

### Phase 13 — Angle Extension

After Phase B is stable, add the `yawm30` and `yawp30` Phase C conditions.
Re-evaluate whether bounding-box edges remain meaningful for perspective-shaped
targets.

Exit condition:

```text
angled-target segmentation is stable
metric interpretation is documented
positive and negative yaw behavior can be compared
```

### Phase 14 — Acceptance Criteria

Define application-specific pass, warning, and invalid-test criteria using the
collected distributions.

Prefer:

```text
median boundary error
p95 boundary error
valid-frame ratio
per-side median offset
repeat-to-repeat stability
```

Do not use maximum error as the primary criterion.

### Phase 15 — Dynamic Synchronization Extension

Create a separate dynamic-test plan and dataset after static acceptance criteria
are established. Keep `scene06_alignment_dynamic_...` results separate from
static statistics.

Exit condition:

```text
spatial registration error and temporal synchronization error can be interpreted
separately
```

---

## 25. Immediate Next Tasks

```text
1. Implement scripts/record_alignment_experiment.sh with:
   - alignment argument validation and scene05 naming
   - required and optional topic discovery
   - registration runtime validation
   - stream and camera-info dimension checks
   - camera parameter capture
   - preflight.yaml and experiment.yaml output
   - graceful rosbag recording
   - post_recording.yaml output

2. Validate the standalone recorder using one d100-center test recording

3. Confirm the aligned depth topic and image dimensions

4. Extend extraction to save:
   - RGB frames
   - aligned depth frames
   - RGB timestamps
   - depth timestamps
   - camera info
   - experiment metadata

5. Implement src/preprocessing/frame_pairing.py

6. Test timestamp pairing with synthetic data

7. Create config/roi/alignment/

8. Extend select_roi.py to support:
   --roi-type alignment

9. Select ROI for:
   scene05_alignment_d100_center_yaw00_r01

10. Implement src/segmentation/rgb_foreground.py

11. Implement src/segmentation/depth_foreground.py

12. Implement initial bounding-edge offsets

13. Implement edge-overlay visualization

14. Validate one paired frame manually

15. Implement boundary-distance metrics

16. Implement mask IoU

17. Implement tools/analyze_alignment.py

18. Validate all d100 center repeats

19. Add d050 and d200 center tests

20. Add four image-corner positions

21. Implement tools/summarize_alignment.py
```

---

## 26. Current Milestone

The first alignment milestone is:

> Successfully pair RGB and aligned-depth frames for `scene05_alignment_d100_center_yaw00_r01`, extract one valid foreground mask from each modality, and produce an edge overlay with left/right/top/bottom pixel offsets.

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
