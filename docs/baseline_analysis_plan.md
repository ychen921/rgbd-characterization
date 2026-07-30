# Sensor Characterization Baseline Analysis Plan

## 0. Scope

This document covers the depth-only characterization pipeline:

- depth quality
- measured depth
- temporal noise
- camera back-projection
- per-frame plane fitting
- planarity
- repeat and cross-distance comparison
- Scene 04 edge / depth-discontinuity quality

Scene 04 is included as a separate depth-only analyzer. It evaluates:

- invalid-depth concentration near a depth edge
- foreground/background depth bleeding
- mixed and outlier pixels
- edge-transition width
- temporal edge stability

The following topics remain outside the scope of this document:

- RGB–Depth spatial alignment
- absolute RGB/depth edge correspondence
- color/depth frame pairing
- dynamic RGB–Depth synchronization
- absolute physical edge-localization accuracy without an independent reference

Those topics are specified separately in:

```text
docs/rgb_depth_alignment_plan.md
```

The Scene 04 depth-only analyzer may use a manually annotated **nominal edge line** for signed-distance binning. That line must not be described as independent ground truth unless it is obtained from a separately calibrated reference.

The planar and Scene 04 pipelines may share extraction, dataset loading, invalid-depth handling, configuration parsing, and result-writing infrastructure, but they must use separate ROI schemas, analyzers, and metrics.

---

## 1. Current Status

The extraction pipeline has been completed and tested:

```text
rosbag
↓
extract_dataset.py
↓
depth.npz
↓
inspect_dataset.py
↓
NPZ files can be loaded and inspected successfully
```

The planar implementation is complete through the dataset-level analysis
entry point. The following components are implemented and covered by tests:

```text
src/preprocessing/roi.py
tools/select_roi.py
src/preprocessing/depth.py
src/metrics/depth_quality.py
src/metrics/measured_depth.py
src/geometry/camera.py
src/geometry/plane_fitting.py
src/metrics/planarity.py
src/metrics/temporal.py
tools/analyze_baseline.py
```

The remaining planar work is validation and multi-dataset reporting:

```text
- select a planar ROI for scene01_white_d050_r01
- run and inspect the real baseline artifacts
- validate repeat reuse when repeat datasets are available
- analyze additional distances when datasets are available
- implement tools/summarize_baseline.py
```

The Scene 04 core configuration, selection, geometry, and metric layers are
also implemented and covered by synthetic tests:

```text
src/preprocessing/edge_roi.py
tools/select_edge_roi.py
src/geometry/edge_geometry.py
src/metrics/edge_discontinuity.py
```

`tools/select_edge_roi.py` now provides the complete interactive CLI,
including three-ROI selection, nominal-edge annotation, foreground-side
inference, final confirmation/reselection, clean preview generation, YAML/PNG
persistence, repeat-key reuse, and no-overwrite protection. Its interaction
and persistence workflow has also been manually exercised using
`scene01_white_d050_r01`. That dataset is suitable for workflow validation but
is not a controlled Scene 04 distance-discontinuity reference.

The remaining Scene 04 work is:

```text
- implement src/visualization/edge.py
- implement tools/analyze_edge.py
- validate classification and profiles on a formal Scene 04 dataset
- analyze repeats when datasets are available
- implement tools/summarize_edge.py
```

A focused implementation audit passed all 398 directly related planar and
Scene 04 tests. This test result confirms implementation behavior; it does not
replace real-dataset metric validation.

The next integration milestone remains:

> Produce and inspect a real planar baseline result for
> `scene01_white_d050_r01`, then complete the Scene 04 visualization and
> analysis entry point without modifying the semantics of the planar metrics.

---

## 2. Current Technical Decisions

### 2.1 Keep Full Depth Frames in NPZ

ROI is not applied during rosbag extraction.

```text
rosbag
↓
full depth frame extraction
↓
depth.npz
↓
ROI selection
↓
metrics
```

Reasons:

- ROI strategy may change.
- Board pixel footprint changes with distance.
- Modifying ROI must not require re-reading rosbag.
- Full-frame NPZ remains a reproducible derived dataset.

### 2.2 ROI Scope

Each `scene + target + distance` combination has one ROI YAML.

Repeats at the same scene and distance share the same ROI.

```text
scene01_white_d050_r01
scene01_white_d050_r02
scene01_white_d050_r03
             │
             ▼
config/roi/scene01_white_d050.yaml
```

Another distance uses another ROI:

```text
scene01_white_d100_r01
scene01_white_d100_r02
             │
             ▼
config/roi/scene01_white_d100.yaml
```

Therefore:

```text
ROI key = experiment name without repeat suffix
```

Example:

```text
scene01_white_d050_r01
↓
scene01_white_d050
```

This allows:

- different ROI sizes at different distances
- the same ROI to be reused across repeats
- no repeated ROI selection for `r02`, `r03`, etc.
- stable repeat comparison within one distance

Cross-distance reports must record ROI dimensions and pixel count because ROI area may differ by distance.

### 2.3 Interactive ROI Selection

ROI selection should behave like a simplified LabelImg workflow:

```text
load depth.npz
↓
show representative depth frame
↓
drag rectangle with mouse
↓
confirm ROI
↓
save ROI YAML
```

Recommended tool:

```text
tools/select_roi.py
```

Use OpenCV `cv2.selectROI()` for the first version.

Do not build a full annotation application.

### 2.4 Separate ROI Selection from Analysis

Do not let `analyze_baseline.py` unexpectedly open a GUI.

Recommended behavior:

```text
select_roi.py
↓
create ROI YAML

analyze_baseline.py
↓
load existing ROI YAML
↓
run analysis
```

If an ROI YAML is missing, `analyze_baseline.py` should fail clearly.

Example:

```text
ROI configuration not found:
config/roi/scene01_white_d050.yaml

Run:
python3 tools/select_roi.py data/scene01_white_d050_r01
```

This keeps batch analysis deterministic.

---


### 2.5 Separate Planar and Edge Analyzers

The existing baseline assumes that one ROI contains one approximately planar surface.

That assumption is valid for Scene 01 and other planar target scenes, but it is not valid for Scene 04 because an edge ROI contains at least two depth populations.

Use separate analysis entry points:

```text
tools/analyze_baseline.py
→ single-surface planar scenes

tools/analyze_edge.py
→ Scene 04 edge / depth-discontinuity scenes
```

Shared components:

```text
DepthDataset
prepare_depth()
depth-quality masks
camera intrinsics
result serialization
common rectangle ROI utilities
```

Scene-specific components:

```text
planar scenes
→ one RectROI
→ measured depth / temporal noise / planarity

Scene 04
→ foreground ROI + background ROI + edge ROI + nominal edge line
→ signed-distance profile / bleeding / mixed pixels / transition width
```

Do not run whole-edge-ROI mean, standard deviation, or single-plane fitting as Scene 04 quality metrics. Those values are dominated by the real foreground/background depth difference and do not represent sensor noise or planarity.

---

## 3. Updated Project Structure

```text
rgbd-characterization/
├── bags/
├── data/
├── results/
│
├── config/
│   └── roi/
│       ├── scene01_white_d050.yaml
│       ├── scene01_white_d100.yaml
│       ├── scene04_edge_d050.yaml
│       ├── scene04_edge_d100.yaml
│       └── ...
│
├── tools/
│   ├── extract_dataset.py
│   ├── inspect_dataset.py
│   ├── select_roi.py
│   ├── select_edge_roi.py
│   ├── analyze_baseline.py
│   ├── analyze_edge.py
│   ├── summarize_baseline.py
│   ├── summarize_edge.py
│   └── analyze_alignment.py
│
└── src/
    ├── io/
    │   ├── bag_reader.py
    │   ├── dataset.py
    │   └── synchronized_dataset.py
    │
    ├── preprocessing/
    │   ├── roi.py
    │   ├── edge_roi.py
    │   ├── depth.py
    │   ├── rgb.py
    │   └── frame_pairing.py
    │
    ├── geometry/
    │   ├── camera.py
    │   ├── plane_fitting.py
    │   └── edge_geometry.py
    │
    ├── metrics/
    │   ├── depth_quality.py
    │   ├── temporal.py
    │   ├── measured_depth.py
    │   ├── planarity.py
    │   ├── edge_discontinuity.py
    │   └── alignment.py
    │
    └── visualization/
        └── edge.py
```

Recommended implementation order:

```text
Planar baseline
1. Complete depth data semantic inspection                  [COMPLETED]
2. Implement src/preprocessing/roi.py                       [COMPLETED]
3. Implement tools/select_roi.py                            [COMPLETED]
4. Select ROI for one 50 cm distance group                  [PENDING]
5. Validate ROI reuse across repeats                        [DATA UNAVAILABLE]
6. Implement src/preprocessing/depth.py                     [COMPLETED]
7. Implement src/metrics/depth_quality.py                   [COMPLETED]
8. Implement src/metrics/measured_depth.py                  [COMPLETED]
9. Implement src/geometry/camera.py                         [COMPLETED]
10. Implement src/geometry/plane_fitting.py                 [COMPLETED]
11. Implement src/metrics/planarity.py                      [COMPLETED]
12. Implement src/metrics/temporal.py                       [COMPLETED]
13. Implement tools/analyze_baseline.py                     [COMPLETED]
14. Validate scene01_white_d050_r01                         [PENDING]
15. Analyze remaining repeats and distances                 [DATA UNAVAILABLE]
16. Implement cross-distance summary                        [PENDING]

Scene 04 extension
17. Implement src/preprocessing/edge_roi.py                 [COMPLETED]
18. Implement tools/select_edge_roi.py                      [COMPLETED]
19. Implement src/geometry/edge_geometry.py                 [COMPLETED]
20. Implement src/metrics/edge_discontinuity.py             [COMPLETED]
21. Implement src/visualization/edge.py                      [PENDING]
22. Implement tools/analyze_edge.py                          [PENDING]
23. Validate one Scene 04 dataset                            [DATA UNAVAILABLE]
24. Validate repeat reuse and temporal stability             [DATA UNAVAILABLE]
25. Implement tools/summarize_edge.py                        [PENDING]
```

---

## 4. Current Depth Data Observations

For the nominal 50 cm experiment, center-pixel samples were near:

```text
514–515
```

Observed center median:

```text
514
```

Current working interpretation:

```text
raw depth unit ≈ millimeter
```

Therefore:

```text
514 raw units ≈ 514 mm
```

The current observation should be described as:

```text
observed center depth median = 514 mm
nominal setup distance = 500 mm
observed offset from nominal = +14 mm
```

Do not yet call this:

```text
sensor bias = +14 mm
```

Possible setup effects include:

- camera reference plane definition
- measuring tape reference point
- white-board orientation
- setup tolerance
- use of one center pixel instead of a plane estimate

### 4.1 Zero Depth Observation

Full-frame statistics:

```text
zero ratio ≈ 19.46%
```

This is not a white-board target metric.

The zero-depth ratio must be recomputed inside the selected ROI.

### 4.2 Maximum uint16 Observation

Observed value:

```text
65535
```

Full-frame observations:

```text
total count: 7,376
ratio: approximately 0.00203%
affected frames: 48
maximum count in one affected frame: 434 pixels
```

Affected frames typically contain tens to hundreds of `65535` pixels.

Current interpretation:

> `65535` is an observed intermittent maximum-uint16 special value or burst-like artifact.

Do not document:

```text
Orbbec invalid depth sentinel = 65535
```

unless confirmed by official sensor documentation.

However, `65535` must not be included in normal measured-depth or temporal-noise statistics.

---

## 5. ROI Module

Implement:

```text
src/preprocessing/roi.py
```

Main responsibilities:

```text
ROI data model
ROI crop
ROI key derivation
ROI path derivation
ROI YAML load
ROI YAML save
```

GUI logic belongs in:

```text
tools/select_roi.py
```

### 5.1 RectROI

```python
from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class RectROI:
    x: int
    y: int
    width: int
    height: int

    @property
    def pixel_count(self) -> int:
        return self.width * self.height

    def crop(
        self,
        frames: np.ndarray,
    ) -> np.ndarray:
        if frames.ndim != 3:
            raise ValueError(
                "frames must have shape (N, H, W)"
            )

        _, image_height, image_width = frames.shape

        if self.x < 0 or self.y < 0:
            raise ValueError(
                "ROI x and y must be non-negative"
            )

        if self.width <= 0 or self.height <= 0:
            raise ValueError(
                "ROI width and height must be positive"
            )

        if self.x + self.width > image_width:
            raise ValueError(
                "ROI exceeds image width"
            )

        if self.y + self.height > image_height:
            raise ValueError(
                "ROI exceeds image height"
            )

        return frames[
            :,
            self.y:self.y + self.height,
            self.x:self.x + self.width,
        ]
```

### 5.2 ROI Key Derivation

```python
import re


def derive_roi_key(
    experiment_name: str,
) -> str:
    return re.sub(
        r"_r\d+$",
        "",
        experiment_name,
    )
```

Tests:

```python
assert derive_roi_key(
    "scene01_white_d050_r01"
) == "scene01_white_d050"

assert derive_roi_key(
    "scene01_white_d050_r12"
) == "scene01_white_d050"

assert derive_roi_key(
    "scene01_white_d050"
) == "scene01_white_d050"
```

### 5.3 ROI Path Derivation

```python
from pathlib import Path


def get_roi_path(
    roi_root: Path,
    experiment_name: str,
) -> Path:
    # roi_root should be config/roi for this pipeline
    roi_key = derive_roi_key(
        experiment_name
    )

    return roi_root / f"{roi_key}.yaml"
```

Example:

```text
scene01_white_d050_r02
↓
scene01_white_d050
↓
config/roi/scene01_white_d050.yaml
```

### 5.4 ROI YAML Format

```yaml
name: scene01_white_d050

source:
  experiment: scene01_white_d050_r01
  frame_index: 421

roi:
  type: rectangle
  x: 280
  y: 210
  width: 80
  height: 60
```

Store:

```text
source experiment
source frame index
```

for traceability.

### 5.5 ROI Module Tests

Test:

```text
RectROI crop
ROI bounds
ROI key derivation
ROI YAML save/load round trip
```

An invalid ROI must raise `ValueError`.

Do not silently clip coordinates.

---

## 6. Interactive ROI Selection Tool

Implement:

```text
tools/select_roi.py
```

Workflow:

```text
parse dataset directory
↓
derive experiment name
↓
derive ROI key
↓
derive ROI YAML path
↓
ROI YAML exists?
├── yes → print path and skip
└── no
    ↓
    load DepthDataset
    ↓
    select representative frame
    ↓
    convert depth to display image
    ↓
    cv2.selectROI()
    ↓
    validate rectangle
    ↓
    save ROI YAML
```

### 6.1 CLI

```bash
python3 tools/select_roi.py \
    data/scene01_white_d050_r01
```

First run:

```text
Dataset:
data/scene01_white_d050_r01

ROI key:
scene01_white_d050

Selecting ROI...

Saved:
config/roi/scene01_white_d050.yaml
```

Repeat:

```bash
python3 tools/select_roi.py \
    data/scene01_white_d050_r02
```

Expected:

```text
ROI already exists:
config/roi/scene01_white_d050.yaml

Skipping ROI selection.
```

Do not overwrite existing ROI by default.

A future `--force` option may be added only if needed.

### 6.2 Representative Frame

Use:

```python
frame_index = dataset.num_frames // 2
```

Store `frame_index` in the ROI YAML.

### 6.3 Depth Display Conversion

Do not display raw `uint16` depth directly.

Use display-only percentile normalization.

```python
import cv2
import numpy as np


def depth_to_display(
    depth: np.ndarray,
) -> np.ndarray:
    max_uint16 = np.iinfo(
        np.uint16
    ).max

    valid = (
        (depth > 0)
        & (depth < max_uint16)
    )

    if not np.any(valid):
        raise ValueError(
            "Frame contains no displayable depth"
        )

    values = depth[valid]

    lower = np.percentile(
        values,
        1,
    )

    upper = np.percentile(
        values,
        99,
    )

    if upper <= lower:
        raise ValueError(
            "Invalid display depth range"
        )

    clipped = np.clip(
        depth.astype(np.float32),
        lower,
        upper,
    )

    normalized = (
        (clipped - lower)
        / (upper - lower)
        * 255.0
    )

    image = normalized.astype(
        np.uint8
    )

    return cv2.cvtColor(
        image,
        cv2.COLOR_GRAY2BGR,
    )
```

This conversion is only for the ROI GUI.

The raw depth data must remain unchanged.

### 6.4 Rectangle Selection

```python
x, y, width, height = cv2.selectROI(
    "Select White Board ROI",
    display_image,
    showCrosshair=True,
    fromCenter=False,
)
```

If:

```python
width <= 0 or height <= 0
```

fail without saving YAML.

Convert to:

```python
roi = RectROI(
    x=int(x),
    y=int(y),
    width=int(width),
    height=int(height),
)
```

Validate the ROI against the original frame size before saving.

---

## 7. ROI Reuse Across Repeats

Example:

```text
data/
├── scene01_white_d050_r01/
│   └── depth.npz
├── scene01_white_d050_r02/
│   └── depth.npz
└── scene01_white_d050_r03/
    └── depth.npz
```

Only one ROI file exists:

```text
config/roi/scene01_white_d050.yaml
```

Then:

```text
r01 → load scene01_white_d050.yaml
r02 → load scene01_white_d050.yaml
r03 → load scene01_white_d050.yaml
```

No additional GUI selection is required.

Assumption:

> Camera and target placement remain sufficiently consistent across repeats at the same scene and distance.

Before batch analysis, visually verify that the reused ROI still lies entirely inside the white board for all repeats.

---

## 8. Depth Preprocessing

Implement:

```text
src/preprocessing/depth.py
```

Raw data remains:

```text
uint16
```

Analysis representation:

```text
float32
NaN = excluded sample
```

Current exclusion rules:

```text
0
65535
```

The semantics differ:

```text
0
→ zero/no-depth value

65535
→ observed maximum-uint16 special value
```

Recommended:

```python
import numpy as np


def prepare_depth(
    depth: np.ndarray,
) -> np.ndarray:
    prepared = depth.astype(
        np.float32,
        copy=True,
    )

    zero_mask = depth == 0

    max_uint16_mask = (
        depth
        == np.iinfo(np.uint16).max
    )

    prepared[
        zero_mask | max_uint16_mask
    ] = np.nan

    return prepared
```

Do not initially filter by nominal distance range.

A filter such as:

```text
500 ± 100 mm
```

could remove genuine gross sensor errors and bias the characterization result.

---

## 9. Depth Quality Metrics

Use:

```text
src/metrics/depth_quality.py
```

instead of:

```text
src/metrics/invalid.py
```

Reason:

The current data contains two relevant observations:

```text
zero-depth occurrence
maximum-uint16 occurrence
```

Recommended result:

```python
from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class DepthQualityResult:
    zero_ratio: float
    zero_ratio_map: np.ndarray

    max_uint16_ratio: float
    max_uint16_ratio_map: np.ndarray

    max_uint16_affected_frames: int
    max_uint16_max_pixels_per_frame: int
```

Input:

```text
raw ROI uint16
```

Compute:

```text
zero_mask
depth == 0

max_uint16_mask
depth == 65535
```

Record:

```text
overall zero ratio
per-pixel zero ratio map

overall max-uint16 ratio
per-pixel max-uint16 ratio map

number of affected frames
maximum max-uint16 pixels in one frame
```

These metrics must be computed inside the selected ROI.

---

## 10. Temporal Noise Metric

Implement:

```text
src/metrics/temporal.py
```

Input:

```text
prepared ROI depth
dtype: float32
NaN = excluded sample
shape: (N, H, W)
```

Per-pixel temporal standard deviation:

```python
std_map = np.nanstd(
    depth,
    axis=0,
)
```

Use:

```text
min_valid_ratio = 0.9
```

Example:

```python
valid_ratio = np.mean(
    ~np.isnan(depth),
    axis=0,
)

std_map = np.nanstd(
    depth,
    axis=0,
)

std_map[
    valid_ratio < min_valid_ratio
] = np.nan
```

Recommended result:

```python
@dataclass(frozen=True)
class TemporalNoiseResult:
    std_map: np.ndarray
    median_std: float
    mean_std: float
    p95_std: float
```

Synthetic tests must verify:

```text
constant pixel → std 0
varying pixel → expected non-zero std
NaN is ignored
insufficient valid ratio → NaN
```

---

## 11. Measured Depth Metric

Implement:

```text
src/metrics/measured_depth.py
```

Do not call it `accuracy.py` yet.

For each frame:

```python
frame_median = np.nanmedian(
    depth,
    axis=(1, 2),
)
```

Flow:

```text
frame 0 ROI → median
frame 1 ROI → median
frame 2 ROI → median
...
↓
frame_median
↓
aggregate statistics
```

Recommended result:

```python
@dataclass(frozen=True)
class MeasuredDepthResult:
    frame_median: np.ndarray
    median_depth: float
    mean_depth: float
    std_depth: float
    p05_depth: float
    p95_depth: float
```

An all-invalid frame should produce:

```text
frame median = NaN
```

NaN-aware aggregate statistics should exclude that frame.

---


## 12. Camera Model and Depth Back-Projection

Implement:

```text
src/geometry/camera.py
```

Plane fitting must operate in camera 3D coordinates rather than directly in image coordinates.

A depth image stores:

```text
z(u, v)
```

A physical plane is modeled as:

```text
ax + by + cz + d = 0
```

Therefore valid depth pixels must be back-projected using depth-camera intrinsics.

### 12.1 Camera Intrinsics

```python
from dataclasses import dataclass


@dataclass(frozen=True)
class CameraIntrinsics:
    fx: float
    fy: float
    cx: float
    cy: float
```

The values must come from the depth camera calibration data.

### 12.2 ROI Depth to 3D Points

For each valid pixel:

```text
Z = depth
X = (u - cx) × Z / fx
Y = (v - cy) × Z / fy
```

Recommended function:

```python
import numpy as np


def depth_roi_to_points(
    depth_mm: np.ndarray,
    intrinsics: CameraIntrinsics,
    roi_x: int,
    roi_y: int,
) -> np.ndarray:
    if depth_mm.ndim != 2:
        raise ValueError(
            "depth_mm must have shape (H, W)"
        )

    height, width = depth_mm.shape

    v_local, u_local = np.indices(
        (height, width)
    )

    u = u_local + roi_x
    v = v_local + roi_y

    z = depth_mm.astype(
        np.float64
    ) / 1000.0

    valid = np.isfinite(z)

    x = (
        (u - intrinsics.cx)
        * z
        / intrinsics.fx
    )

    y = (
        (v - intrinsics.cy)
        * z
        / intrinsics.fy
    )

    points = np.stack(
        [x, y, z],
        axis=-1,
    )

    return points[valid]
```

The ROI offset must be included. Do not treat the cropped ROI's local coordinate `(0, 0)` as the original image origin.

---

## 13. Plane Fitting Geometry

Implement:

```text
src/geometry/plane_fitting.py
```

Responsibilities:

```text
PlaneModel
SVD/PCA plane fitting
point-to-plane residuals
normal-direction normalization
```

### 13.1 Plane Model

```python
from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class PlaneModel:
    normal: np.ndarray
    d: float
    centroid: np.ndarray
```

Plane equation:

```text
normal · point + d = 0
```

The plane normal must satisfy:

```text
||normal|| = 1
```

### 13.2 Initial Fitting Method

Use deterministic SVD/PCA fitting for the first baseline version.

```python
def fit_plane_svd(
    points: np.ndarray,
) -> PlaneModel:
    if points.ndim != 2 or points.shape[1] != 3:
        raise ValueError(
            "points must have shape (N, 3)"
        )

    if points.shape[0] < 3:
        raise ValueError(
            "At least three points are required"
        )

    centroid = np.mean(
        points,
        axis=0,
    )

    centered = points - centroid

    _, _, vh = np.linalg.svd(
        centered,
        full_matrices=False,
    )

    normal = vh[-1]
    normal = normal / np.linalg.norm(normal)

    if normal[2] < 0:
        normal = -normal

    d = -float(
        np.dot(
            normal,
            centroid,
        )
    )

    return PlaneModel(
        normal=normal,
        d=d,
        centroid=centroid,
    )
```

Do not begin with RANSAC because the ROI is manually selected inside the white board and most valid samples should belong to one plane. RANSAC may be added later for edge-heavy or contaminated scenes.

### 13.3 Point-to-Plane Residuals

```python
def point_to_plane_distances(
    points: np.ndarray,
    plane: PlaneModel,
) -> np.ndarray:
    return (
        points @ plane.normal
        + plane.d
    )
```

Because the normal is unit length, the result is signed distance in meters.

For reporting:

```python
residual_mm = residual_m * 1000.0
```

---

## 14. Planarity Metric

Status:

```text
COMPLETED
```

Implemented module:

```text
src/metrics/planarity.py
```

Main responsibility:

> Perform per-frame plane fitting and summarize geometric stability and surface residuals.

Plane fitting should initially be performed independently for each frame:

```text
frame 0 → plane 0
frame 1 → plane 1
...
```

This provides:

```text
plane distance over time
plane normal over time
tilt over time
residual RMSE over time
inlier ratio over time
```

Do not merge all frames into one large point cloud as the primary metric because that mixes temporal and spatial variation.

### 14.1 Per-Frame Metrics

Record:

```text
normal_x
normal_y
normal_z
plane_distance_m
tilt_deg
residual_rmse_mm
residual_std_mm
residual_p95_abs_mm
inlier_ratio
valid_points
```

### 14.2 Plane Distance

For a unit-normal plane:

```text
normal · point + d = 0
```

the perpendicular distance from the camera origin is:

```text
abs(d)
```

This is not necessarily equal to the center-pixel Z depth when the board is tilted.

Retain both:

```text
frame median depth
plane perpendicular distance
```

### 14.3 Plane Tilt

The camera optical axis is:

```text
[0, 0, 1]
```

After enforcing:

```text
normal_z >= 0
```

calculate:

```python
tilt_deg = np.degrees(
    np.arccos(
        np.clip(
            plane.normal[2],
            -1.0,
            1.0,
        )
    )
)
```

### 14.4 Residual Metrics

For each frame calculate:

```text
RMSE
residual standard deviation
p95 absolute residual
```

Recommended:

```python
abs_residual_mm = np.abs(
    residual_mm
)

rmse_mm = np.sqrt(
    np.mean(
        residual_mm ** 2
    )
)

residual_std_mm = np.std(
    residual_mm
)

residual_p95_abs_mm = np.percentile(
    abs_residual_mm,
    95,
)
```

Do not rely on maximum residual as the primary summary because it is too sensitive to isolated outliers.

### 14.5 Inlier Ratio

Use configurable analysis parameters:

```yaml
plane:
  inlier_threshold_mm: 5.0
  min_valid_points: 100
```

Calculation:

```python
inlier_ratio = np.mean(
    np.abs(residual_mm)
    <= inlier_threshold_mm
)
```

### 14.6 Result Models

```python
@dataclass(frozen=True)
class FramePlaneResult:
    normal: np.ndarray
    distance_m: float
    tilt_deg: float
    rmse_mm: float
    residual_std_mm: float
    residual_p95_abs_mm: float
    inlier_ratio: float
    valid_points: int
```

```python
@dataclass(frozen=True)
class PlanarityResult:
    frame_distance_m: np.ndarray
    frame_tilt_deg: np.ndarray
    frame_rmse_mm: np.ndarray
    frame_p95_abs_mm: np.ndarray
    frame_inlier_ratio: np.ndarray

    median_distance_m: float
    distance_std_mm: float
    median_tilt_deg: float
    tilt_std_deg: float
    median_rmse_mm: float
    p95_rmse_mm: float
    median_p95_abs_mm: float
    median_inlier_ratio: float
```

---

## 15. Plane Fitting Tests

Plane fitting must pass synthetic tests before integration.

### 15.1 Perfect Plane

Generate points on:

```text
z = 1.0 m
```

Expected:

```text
normal ≈ [0, 0, 1]
distance ≈ 1.0 m
tilt ≈ 0°
RMSE ≈ 0 mm
```

### 15.2 Tilted Plane

Generate a known tilted plane, for example:

```text
z = 1.0 + 0.1x
```

Expected:

```text
fitted normal matches theoretical normal
tilt matches theoretical angle
```

### 15.3 Noisy Plane

Add Gaussian noise:

```text
sigma = 2 mm
```

Expected:

```text
residual RMSE approximately 2 mm
```

Use a tolerance rather than exact equality.

### 15.4 Insufficient Points

Fewer than three points must raise `ValueError`.

The real pipeline should additionally enforce:

```text
min_valid_points >= 100
```

### 15.5 Normal Direction

The returned model must always satisfy:

```text
normal_z >= 0
```

### 15.6 Back-Projection

At:

```text
u = cx
v = cy
depth = 1000 mm
```

expected point:

```text
[0, 0, 1]
```

Also test ROI offsets explicitly.

---

## 16. Scene 04 Edge / Depth-Discontinuity Extension

Scene 04 must not reuse the planar-scene metric assumptions unchanged.

The Scene 04 analyzer must treat the image as three regions:

```text
foreground reference ROI
edge analysis ROI
background reference ROI
```

The edge analysis ROI crosses the foreground/background boundary. The reference ROIs must remain away from the boundary and are used to estimate the stable foreground and background depth populations.

### 16.1 Scene 04 Analysis Boundary

The first Scene 04 implementation is depth-only.

It may report:

```text
foreground/background reference depth
reference-region noise and invalid ratio
foreground bleeding into the background side
background bleeding into the foreground side
mixed-depth ratio
outlier ratio
invalid ratio near the edge
transition width
frame-to-frame edge-profile stability
```

It must not claim independent absolute edge-localization accuracy when the nominal edge line is selected from the same depth data.

Use these terms:

```text
nominal edge line
estimated depth-transition center
offset from nominal edge
```

Do not use:

```text
true edge
ground-truth edge error
absolute localization bias
```

unless an independent calibrated reference is available.

### 16.2 Scene 04 ROI Scope and Reuse

Each `scene + edge setup + distance` combination has one edge ROI YAML.

Repeats at the same setup and distance reuse the same file:

```text
scene04_edge_d050_r01
scene04_edge_d050_r02
scene04_edge_d050_r03
             │
             ▼
config/roi/scene04_edge_d050.yaml
```

Reuse is valid only when:

```text
camera pose is unchanged
foreground target pose is unchanged
background plane is unchanged
the annotated edge remains inside the edge ROI
```

Before batch analysis, generate an overlay for every repeat and verify that the edge displacement is acceptably small.

If the setup moved materially, create a separate ROI key rather than forcing reuse.

### 16.3 Scene 04 ROI Data Model

Implement:

```text
src/preprocessing/edge_roi.py
```

Recommended models:

```python
from dataclasses import dataclass


@dataclass(frozen=True)
class Line2D:
    p1: tuple[float, float]
    p2: tuple[float, float]


@dataclass(frozen=True)
class EdgeROIConfig:
    foreground_roi: RectROI
    background_roi: RectROI
    edge_roi: RectROI
    nominal_edge: Line2D
    foreground_side: str
    distance_bin_px: float
    max_edge_distance_px: float
```

Required validation:

```text
all rectangles are inside the image
all rectangles have positive width and height
the nominal edge intersects the edge ROI
foreground_side is one of:
- left
- right
- positive
- negative

distance_bin_px > 0
max_edge_distance_px > 0
```

The internal signed-distance convention must be normalized to:

```text
distance < 0 → foreground side
distance > 0 → background side
distance = 0 → nominal edge line
```

### 16.4 Scene 04 ROI YAML Format

Recommended format:

```yaml
name: scene04_edge_d050
scene_type: edge_discontinuity

source:
  experiment: scene04_edge_d050_r01
  frame_index: 421

foreground_roi:
  type: rectangle
  x: 210
  y: 150
  width: 70
  height: 180

background_roi:
  type: rectangle
  x: 360
  y: 150
  width: 70
  height: 180

edge_roi:
  type: rectangle
  x: 285
  y: 150
  width: 70
  height: 180

nominal_edge:
  type: line
  p1: [320.0, 150.0]
  p2: [320.0, 330.0]
  foreground_side: left

analysis:
  distance_bin_px: 2.0
  max_edge_distance_px: 20.0

  reference:
    minimum_tolerance_mm: 10.0
    mad_scale: 3.0

  transition:
    high_probability: 0.9
    low_probability: 0.1
```

The foreground and background reference ROIs should:

```text
use approximately the same vertical span
have similar pixel counts when practical
avoid target corners, supports, tape, and clamps
remain sufficiently far from the depth boundary
```

The edge ROI should:

```text
cross only one depth boundary
avoid the top and bottom target edges
avoid corners
cover both sides of the nominal edge
```

### 16.5 Interactive Edge ROI Selection

Implement:

```text
tools/select_edge_roi.py
```

Do not force the existing single-rectangle `cv2.selectROI()` workflow to manage all Scene 04 annotations implicitly.

Recommended workflow:

```text
load DepthDataset
↓
select representative frame
↓
create display image
↓
select foreground rectangle
↓
select background rectangle
↓
select edge rectangle
↓
select two nominal-edge endpoints
↓
select/confirm foreground side
↓
validate configuration
↓
save Scene 04 YAML
↓
save overlay preview
```

For the first implementation:

```text
rectangles
→ cv2.selectROI()

line endpoints
→ OpenCV mouse callback
```

Required output:

```text
config/roi/scene04_edge_d050.yaml
results/roi_preview/scene04_edge_d050.png
```

The selection tool must not overwrite an existing YAML by default.

### 16.6 Signed-Distance Geometry

Implement:

```text
src/geometry/edge_geometry.py
```

Required function:

```python
def compute_signed_distance_map(
    image_shape: tuple[int, int],
    line: Line2D,
    foreground_side: str,
) -> np.ndarray:
    """
    Return one signed perpendicular distance in pixels
    for every image pixel.

    Negative: foreground side
    Positive: background side
    """
```

For:

```text
p1 = (x1, y1)
p2 = (x2, y2)
p  = (x, y)
```

use the normalized 2D cross-product distance:

```text
d(p) =
((x2 - x1)(y - y1) - (y2 - y1)(x - x1))
/
sqrt((x2 - x1)^2 + (y2 - y1)^2)
```

Then normalize the sign using `foreground_side`.

Also implement:

```python
def validate_edge_intersection(
    edge_roi: RectROI,
    line: Line2D,
) -> None:
    ...
```

Synthetic tests must cover:

```text
vertical line
horizontal line
slanted line
sign convention
zero-length line rejection
```

### 16.7 Foreground and Background Reference Estimation

Initial implementation should use robust depth-domain reference statistics.

Implement in:

```text
src/metrics/edge_discontinuity.py
```

Required result model:

```python
@dataclass(frozen=True)
class ReferenceDepthResult:
    median_mm: float
    mad_mm: float
    robust_sigma_mm: float
    std_mm: float
    valid_ratio: float
    valid_count: int
```

Required function:

```python
def estimate_reference_depth(
    prepared_depth_frame: np.ndarray,
    roi: RectROI,
) -> ReferenceDepthResult:
    ...
```

Use:

```text
robust_sigma_mm = 1.4826 × MAD
```

Classification tolerance:

```text
tolerance_mm =
max(
    minimum_tolerance_mm,
    mad_scale × robust_sigma_mm
)
```

Estimate the foreground and background references independently for each frame.

Dataset-level aggregation may summarize the per-frame reference values, but it must not replace per-frame estimation in the initial implementation.

Optional later extension:

```text
fit foreground reference plane
fit background reference plane
classify edge points using dual-plane residuals
```

Do not block the first Scene 04 milestone on dual-plane fitting.

### 16.8 Edge Pixel Classification

Implement:

```python
class EdgePixelLabel(IntEnum):
    OUTSIDE = 0
    INVALID = 1
    FOREGROUND = 2
    BACKGROUND = 3
    MIXED = 4
    OUTLIER = 5
```

Required function:

```python
def classify_edge_depth(
    prepared_depth_frame: np.ndarray,
    edge_roi: RectROI,
    foreground_reference: ReferenceDepthResult,
    background_reference: ReferenceDepthResult,
    minimum_tolerance_mm: float,
    mad_scale: float,
) -> np.ndarray:
    ...
```

For the common case:

```text
foreground depth < background depth
```

classify:

```text
invalid
→ NaN sample

foreground
→ within foreground tolerance

background
→ within background tolerance

mixed
→ valid depth strictly between the two accepted reference ranges

outlier
→ any remaining valid depth
```

The implementation must also handle:

```text
foreground depth > background depth
```

Do not hard-code the foreground as the nearer surface.

If the foreground and background accepted ranges overlap, the frame must be rejected or marked ambiguous rather than assigning arbitrary labels.

### 16.9 Distance-Binned Edge Profile

Implement:

```python
def aggregate_labels_by_distance(
    label_map: np.ndarray,
    signed_distance_map: np.ndarray,
    edge_roi: RectROI,
    distance_bin_px: float,
    max_edge_distance_px: float,
) -> "pandas.DataFrame":
    ...
```

Required columns:

```text
distance_min_px
distance_max_px
distance_center_px
pixel_count
valid_count
foreground_ratio
background_ratio
mixed_ratio
outlier_ratio
invalid_ratio
```

The denominator must be explicit:

```text
invalid_ratio
→ all pixels in the bin

foreground/background/mixed/outlier ratios
→ valid pixels in the bin
```

Do not mix these denominators silently.

### 16.10 Scene 04 Metrics

Implement the following required metrics.

#### Foreground Bleeding Ratio

Foreground-classified valid pixels on the background side:

```text
signed distance > 0
```

Report:

```text
overall foreground bleeding ratio
foreground bleeding ratio by distance bin
maximum distance where foreground ratio exceeds a configured threshold
```

#### Background Bleeding Ratio

Background-classified valid pixels on the foreground side:

```text
signed distance < 0
```

Report the same summary fields.

#### Mixed-Pixel Ratio

Report:

```text
overall mixed ratio inside the analyzed edge band
mixed ratio by signed-distance bin
peak mixed ratio
distance of peak mixed ratio
```

#### Outlier Ratio

Keep outliers separate from mixed pixels.

This preserves the distinction between:

```text
intermediate depths between the two surfaces
and
depths outside both reference ranges
```

#### Invalid Ratio and Invalid-Band Width

Report invalid ratio by distance bin.

An optional invalid-band width may be computed using a fixed configured threshold:

```text
invalid_ratio >= invalid_ratio_threshold
```

The threshold must be fixed in configuration and must not be tuned independently for every dataset.

#### Edge Transition Width

Use the foreground-class probability profile.

Initial method:

```text
find the 90% and 10% foreground-ratio crossings
use linear interpolation between neighboring bins
transition_width_px = crossing_10 - crossing_90
```

If the profile is not monotonic enough to produce valid crossings, report:

```text
transition_width_px = NaN
transition_status = insufficient_or_nonmonotonic
```

Do not introduce logistic fitting in the first implementation.

#### Nominal Edge Offset

Estimate the foreground-probability 50% crossing:

```text
estimated_transition_center_px
```

Then report:

```text
nominal_edge_offset_px =
estimated_transition_center_px - 0
```

This is an offset from the manually annotated nominal line, not an absolute physical localization error.

#### Temporal Stability

For each frame record:

```text
foreground_reference_mm
background_reference_mm
foreground_bleeding_ratio
background_bleeding_ratio
mixed_ratio
outlier_ratio
invalid_ratio
transition_width_px
nominal_edge_offset_px
analysis_status
```

Dataset-level aggregation should include:

```text
median
mean
standard deviation
p05
p95
valid frame count
rejected frame count
```

### 16.11 Required Functions

Minimum implementation set:

```text
src/preprocessing/edge_roi.py
- load_edge_roi_config()
- save_edge_roi_config()
- validate_edge_roi_config()
- build_roi_mask()

src/geometry/edge_geometry.py
- compute_signed_distance_map()
- validate_edge_intersection()

src/metrics/edge_discontinuity.py
- estimate_reference_depth()
- compute_reference_tolerance()
- classify_edge_depth()
- aggregate_labels_by_distance()
- compute_bleeding_metrics()
- compute_mixed_outlier_metrics()
- compute_invalid_edge_metrics()
- compute_transition_width()
- estimate_transition_center()
- analyze_edge_frame()
- aggregate_edge_dataset()

src/visualization/edge.py
- draw_edge_roi_overlay()
- plot_edge_label_map()
- plot_edge_probability_profile()
- plot_edge_temporal_metrics()

tools/
- select_edge_roi.py
- analyze_edge.py
- summarize_edge.py
```

### 16.12 Scene 04 Result Models

Recommended frame result:

```python
@dataclass(frozen=True)
class FrameEdgeResult:
    frame_index: int
    foreground_reference_mm: float
    background_reference_mm: float
    foreground_bleeding_ratio: float
    background_bleeding_ratio: float
    mixed_ratio: float
    outlier_ratio: float
    invalid_ratio: float
    transition_width_px: float
    nominal_edge_offset_px: float
    status: str
```

Recommended dataset result:

```python
@dataclass(frozen=True)
class EdgeDiscontinuityResult:
    frame_metrics: "pandas.DataFrame"
    aggregate_profile: "pandas.DataFrame"

    median_foreground_bleeding_ratio: float
    median_background_bleeding_ratio: float
    median_mixed_ratio: float
    median_outlier_ratio: float
    median_invalid_ratio: float
    median_transition_width_px: float
    transition_width_p95_px: float
    median_nominal_edge_offset_px: float
    nominal_edge_offset_std_px: float

    valid_frames: int
    rejected_frames: int
```

### 16.13 Scene 04 Synthetic Tests

Required tests:

```text
1. Ideal hard edge
   - no invalid pixels
   - no bleeding
   - no mixed pixels
   - transition width near one bin or lower

2. Foreground bleeding
   - foreground labels intentionally extend into positive distance
   - foreground bleeding ratio matches expected value

3. Background bleeding
   - background labels intentionally extend into negative distance

4. Mixed-depth band
   - intermediate depth values occupy known edge width
   - mixed ratio and peak distance match expectation

5. Invalid edge band
   - NaN/invalid values occupy known edge width

6. Slanted nominal edge
   - signed-distance bins remain correct

7. Reversed depth ordering
   - foreground is farther than background
   - classification remains correct

8. Overlapping reference tolerance
   - frame is rejected as ambiguous

9. Missing crossing
   - transition width returns NaN with explicit status

10. ROI/config validation
    - out-of-bounds rectangles fail
    - zero-length line fails
    - line missing the edge ROI fails
```

### 16.14 Scene 04 Analysis Tool

Implement:

```text
tools/analyze_edge.py
```

Workflow:

```text
dataset directory
↓
derive experiment name and ROI key
↓
load Scene 04 ROI YAML
↓
load DepthDataset
↓
validate all ROIs and nominal edge
↓
compute signed-distance map once
↓
for each frame:
    compute raw quality masks
    prepare depth
    estimate foreground reference
    estimate background reference
    validate reference separation
    classify edge pixels
    aggregate distance-bin profile
    compute frame metrics
↓
aggregate frame metrics and profiles
↓
save tables, arrays, overlays, and summary
```

Do not route Scene 04 through `tools/analyze_baseline.py`.

Recommended CLI:

```bash
python3 tools/analyze_edge.py \
    data/scene04_edge_d050_r01
```

### 16.15 Scene 04 Result Structure

```text
results/
└── scene04_edge_d050_r01/
    └── edge_discontinuity/
        ├── summary.yaml
        ├── frame_edge_metrics.csv
        ├── aggregate_edge_profile.csv
        ├── representative_label_map.npy
        ├── roi_overlay.png
        ├── label_overlay.png
        ├── edge_probability_profile.png
        └── temporal_edge_metrics.png
```

Recommended summary fields:

```yaml
dataset:
  experiment: scene04_edge_d050_r01
  num_frames: 842

roi:
  key: scene04_edge_d050
  config: config/roi/scene04_edge_d050.yaml
  foreground_pixels: 12600
  background_pixels: 12600
  edge_pixels: 12600

edge_geometry:
  nominal_edge_p1: [320.0, 150.0]
  nominal_edge_p2: [320.0, 330.0]
  foreground_side: left
  distance_bin_px: 2.0
  max_edge_distance_px: 20.0

reference_depth:
  foreground_median_mm: 514.0
  background_median_mm: 812.0

edge_quality:
  foreground_bleeding_ratio_median: 0.03
  background_bleeding_ratio_median: 0.01
  mixed_ratio_median: 0.06
  outlier_ratio_median: 0.01
  invalid_ratio_median: 0.04

transition:
  width_median_px: 5.2
  width_p95_px: 8.0
  nominal_offset_median_px: 1.1
  nominal_offset_std_px: 0.7

frames:
  valid: 820
  rejected: 22
```

The numeric values above are examples only.

### 16.16 First Scene 04 Validation Workflow

Use one clean dataset first:

```text
scene04_edge_d050_r01
```

Validation sequence:

```text
1. Select three rectangles and nominal edge line
2. Inspect saved overlay
3. Confirm foreground/background side convention
4. Inspect foreground and background reference histograms
5. Verify reference tolerance does not overlap
6. Inspect one representative classification map
7. Inspect signed-distance probability profile
8. Verify bleeding direction manually
9. Inspect frame metrics for unstable or rejected frames
10. Run synthetic tests before processing remaining repeats
```

Acceptance conditions for the first milestone:

```text
ROI overlay is correct
classification labels are visually plausible
all metric denominators are documented
transition failures return explicit status
one dataset produces deterministic summary and CSV outputs
no planar metric is computed over the combined edge ROI
```

### 16.17 Optional Dual-Plane Extension

After the depth-domain version is validated, optionally add:

```text
foreground ROI → foreground plane
background ROI → background plane
edge pixels → residual to both planes
```

Potential functions:

```python
def fit_reference_planes(...):
    ...

def compute_dual_plane_residuals(...):
    ...

def classify_edge_points_by_planes(...):
    ...
```

Use dual-plane classification when:

```text
foreground or background planes are significantly tilted
the edge ROI spans a large image region
fixed Z-depth tolerance produces position-dependent classification
```

Keep the depth-domain classification as the first reference implementation so the dual-plane version can be compared against a simpler baseline.

---

## 17. Baseline Analysis Tool

This tool is for single-surface planar scenes only.

Scene 04 must use:

```text
tools/analyze_edge.py
```

Implement:

```text
tools/analyze_baseline.py
```

Workflow:

```text
dataset directory
↓
derive experiment name
↓
derive ROI key
↓
load ROI YAML
↓
load DepthDataset
↓
crop raw ROI
│
├── compute depth quality
│
└── prepare_depth()
        ↓
    compute temporal noise
        ↓
    compute measured depth
        ↓
    back-project each ROI frame to 3D
        ↓
    fit one plane per frame
        ↓
    compute planarity metrics
↓
save results
```

Conceptual code:

```python
experiment_name = dataset_dir.name

roi_path = get_roi_path(
    roi_root,
    experiment_name,
)

if not roi_path.exists():
    raise FileNotFoundError(
        f"ROI configuration not found: "
        f"{roi_path}"
    )

roi = load_roi(
    roi_path
)

dataset = DepthDataset.load(
    dataset_path
)

raw_roi = roi.crop(
    dataset.depth
)

quality_result = compute_depth_quality(
    raw_roi
)

prepared_roi = prepare_depth(
    raw_roi
)

temporal_result = compute_temporal_noise(
    prepared_roi
)

measured_result = compute_measured_depth(
    prepared_roi
)

planarity_result = compute_planarity(
    prepared_roi,
    intrinsics=intrinsics,
    roi=roi,
    inlier_threshold_mm=5.0,
    min_valid_points=100,
)
```

---

## 18. Result Structure

```text
results/
└── scene01_white_d050_r01/
    └── baseline/
        ├── summary.yaml
        ├── frame_median_depth.csv
        ├── temporal_std.npy
        ├── zero_ratio_map.npy
        ├── max_uint16_ratio_map.npy
        └── frame_plane_metrics.csv
```

Recommended summary:

```yaml
dataset:
  experiment: scene01_white_d050_r01
  num_frames: 842
  width: 640
  height: 480

roi:
  key: scene01_white_d050
  config: config/roi/scene01_white_d050.yaml
  x: 280
  y: 210
  width: 80
  height: 60
  pixel_count: 4800

depth_quality:
  zero_ratio: 0.0001

  max_uint16:
    ratio: 0.0
    affected_frames: 0
    max_pixels_per_frame: 0

temporal_noise:
  min_valid_ratio: 0.9
  median_std_mm: 1.82
  mean_std_mm: 2.13
  p95_std_mm: 3.71

measured_depth:
  median_mm: 514.0
  mean_mm: 514.2
  std_mm: 0.71
  p05_mm: 513.0
  p95_mm: 515.0

planarity:
  fitting_method: svd
  inlier_threshold_mm: 5.0
  min_valid_points: 100

  plane_distance:
    median_m: 0.514
    std_mm: 0.8

  tilt:
    median_deg: 0.7
    std_deg: 0.1

  residual:
    median_rmse_mm: 1.4
    p95_rmse_mm: 2.1
    median_p95_abs_mm: 2.8

  inlier_ratio:
    median: 0.996
```

ROI dimensions and `pixel_count` must be recorded because distance-specific ROIs may differ in area.

---

## 19. First Validation Workflow

Use:

```text
scene01_white_d050_r01
```

first.

### Step 1

Implement:

```text
src/preprocessing/roi.py
```

Test:

```text
crop
bounds
ROI key derivation
YAML save/load
```

### Step 2

Implement:

```text
tools/select_roi.py
```

Run:

```bash
python3 tools/select_roi.py \
    data/scene01_white_d050_r01
```

Expected:

```text
config/roi/scene01_white_d050.yaml
```

### Step 3

Test reuse:

```bash
python3 tools/select_roi.py \
    data/scene01_white_d050_r02
```

Expected:

```text
ROI already exists
Skipping ROI selection
```

No GUI should open.

### Step 4

Visually verify that the same ROI remains inside the board for:

```text
r01
r02
r03
```

### Step 5

Inside the ROI, compute:

```text
zero ratio
65535 ratio
```

This determines whether the full-frame `65535` artifact reaches the target ROI.

### Step 6

Implement:

```text
src/preprocessing/depth.py
src/metrics/depth_quality.py
src/metrics/measured_depth.py
src/geometry/camera.py
src/geometry/plane_fitting.py
src/metrics/planarity.py
src/metrics/temporal.py
```

Each module must pass synthetic tests before integration.

### Step 7

Implement:

```text
tools/analyze_baseline.py
```

Run only:

```text
scene01_white_d050_r01
```

Validate numerical results manually.

---

## 20. Multi-Repeat and Multi-Distance Workflow

After 50 cm `r01` passes:

```text
scene01_white_d050_r01
scene01_white_d050_r02
scene01_white_d050_r03
```

all use:

```text
config/roi/scene01_white_d050.yaml
```

For each new distance:

```text
scene01_white_d100 → scene01_white_d100.yaml
scene01_white_d150 → scene01_white_d150.yaml
scene01_white_d200 → scene01_white_d200.yaml
...
```

Recommended execution pattern:

```text
select ROI for d050
analyze all d050 repeats

select ROI for d100
analyze all d100 repeats

select ROI for d150
analyze all d150 repeats

...
```

This is easier to validate than selecting all ROIs first and analyzing all datasets later.

---

## 21. Cross-Distance Comparison

Once every repeat has a `summary.yaml`, create:

```text
tools/summarize_baseline.py
```

Recommended output:

```text
results/baseline_summary.csv
```

Suggested columns:

| experiment | distance_mm | repeat | roi_width | roi_height | roi_pixels | zero_ratio | max_uint16_ratio | temporal_median_std_mm | measured_median_mm | plane_distance_m | plane_distance_std_mm | tilt_deg | plane_rmse_mm | plane_p95_abs_mm | plane_inlier_ratio |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|

Then aggregate repeats by distance.

Possible distance-level metrics:

```text
mean measured median across repeats
repeat-to-repeat standard deviation
mean temporal noise
mean zero-depth ratio
mean max-uint16 ratio
mean plane distance
plane-distance repeatability
mean plane residual RMSE
mean plane p95 absolute residual
mean fitted tilt
```

Recommended plots:

```text
distance vs measured offset from nominal
distance vs temporal noise
distance vs zero-depth ratio
distance vs max-uint16 occurrence ratio
distance vs plane residual RMSE
distance vs plane-distance stability
distance vs fitted tilt
```

Always retain ROI dimensions and pixel count because ROI size differs by distance.

---

Scene 04 must use a separate summary:

```text
tools/summarize_edge.py
results/edge_summary.csv
```

Suggested Scene 04 columns:

| experiment | distance_mm | repeat | edge_roi_pixels | foreground_ref_mm | background_ref_mm | foreground_bleeding_ratio | background_bleeding_ratio | mixed_ratio | outlier_ratio | invalid_ratio | transition_width_px | nominal_edge_offset_px | valid_frames | rejected_frames |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|

Recommended Scene 04 plots:

```text
distance vs foreground bleeding ratio
distance vs background bleeding ratio
distance vs mixed ratio
distance vs invalid ratio
distance vs transition width
distance vs nominal-edge offset stability
```

Do not merge planar residual RMSE and edge transition width into one generic quality score. They measure different failure modes.


## 22. Immediate Next Tasks

Status was audited against the implementation and focused test suite. A checked
item means the implementation and synthetic/automated tests are complete. It
does not imply that real-dataset acceptance has been completed.

Proceed in this order:

```text
Planar baseline integration

1. [x] Confirm the current status of:
       - src/preprocessing/roi.py
       - tools/select_roi.py
       - src/preprocessing/depth.py
       - src/metrics/depth_quality.py
       - src/metrics/measured_depth.py
       - src/geometry/camera.py
       - src/geometry/plane_fitting.py
       - src/metrics/temporal.py

2. [x] Complete any missing planar baseline modules

3. [x] Implement or finish tools/analyze_baseline.py

4. [ ] Validate:
       scene01_white_d050_r01

5. [ ] Confirm:
       - zero/max-uint16 handling
       - measured-depth output
       - temporal-noise output
       - per-frame planarity output
       - summary serialization

6. [ ] Validate d050 ROI reuse across r01/r02/r03
       BLOCKED: repeat datasets are not currently available

7. [ ] Analyze remaining Scene 01 distances
       BLOCKED: additional distance datasets are not currently available

8. [ ] Implement tools/summarize_baseline.py


Scene 04 extension

9.  [x] Implement src/preprocessing/edge_roi.py
        - Line2D
        - EdgeROIConfig
        - YAML load/save
        - bounds and intersection validation

10. [x] Implement tools/select_edge_roi.py
        - foreground ROI
        - background ROI
        - edge ROI
        - nominal edge line
        - foreground-side inference and confirmation
        - overlay preview
        - final accept/reselect/cancel workflow
        - YAML and PNG persistence
        - repeat-key reuse and no-overwrite protection
        - formal CLI entry point

11. [x] Implement src/geometry/edge_geometry.py
        - signed-distance map
        - sign normalization
        - edge/ROI intersection validation

12. [x] Implement src/metrics/edge_discontinuity.py
        - reference depth estimation
        - robust tolerance
        - edge classification
        - distance-bin aggregation
        - bleeding metrics
        - mixed/outlier metrics
        - invalid-edge metrics
        - transition width
        - nominal transition-center offset
        - per-frame and dataset aggregation

13. [ ] Implement src/visualization/edge.py
        - ROI overlay
        - label overlay
        - probability profile
        - temporal metric plots

14. [x] Add Scene 04 synthetic tests

15. [ ] Implement tools/analyze_edge.py

16. [ ] Validate:
        scene04_edge_d050_r01
        BLOCKED: a formal Scene 04 dataset is not currently available

17. [ ] Inspect classification and probability profile manually
        BLOCKED: requires tools/analyze_edge.py and a formal Scene 04 dataset

18. [ ] Analyze Scene 04 repeats
        BLOCKED: Scene 04 repeat datasets are not currently available

19. [ ] Implement tools/summarize_edge.py

20. [ ] Consider dual-plane classification only after the
        depth-domain Scene 04 baseline is validated
```

---

## 22.1 Relationship to RGB–Depth Alignment

The baseline milestone must be completed before implementing the alignment analyzer.

Recommended order:

```text
1. Validate one baseline dataset
2. Validate all 50 cm repeats
3. Complete multi-distance baseline summary
4. Confirm aligned RGB and depth topics and camera-info semantics
5. Implement rgb_depth_alignment_plan.md
```

Do not add RGB segmentation, edge matching, or frame synchronization logic to `analyze_baseline.py`.

Use:

```text
tools/analyze_baseline.py
```

for depth-only characterization and:

```text
tools/analyze_alignment.py
```

for RGB–Depth alignment.

---

## 23. Current Milestone

The current work is divided into two sequential milestones.

### Milestone A: Complete Planar Baseline Integration

Implementation status:

```text
COMPLETED

src/preprocessing/roi.py
tools/select_roi.py
src/preprocessing/depth.py
src/metrics/depth_quality.py
src/metrics/measured_depth.py
src/geometry/camera.py
src/geometry/plane_fitting.py
src/metrics/planarity.py
src/metrics/temporal.py
tools/analyze_baseline.py
```

The implementation and automated integration tests are complete. Real-dataset
acceptance is still pending because the planar ROI and baseline result
artifacts are not currently present.

Available inputs:

```text
data/scene01_white_d050_r01/depth.npz
config/calib/depth_camera_info.yaml
```

Required next outputs:

```text
config/roi/scene01_white_d050.yaml
results/scene01_white_d050_r01/baseline/summary.yaml
results/scene01_white_d050_r01/baseline/frame_median_depth.csv
results/scene01_white_d050_r01/baseline/frame_plane_metrics.csv
results/scene01_white_d050_r01/baseline/temporal_std.npy
results/scene01_white_d050_r01/baseline/zero_ratio_map.npy
results/scene01_white_d050_r01/baseline/max_uint16_ratio_map.npy
```

Target:

> Successfully produce depth-quality, temporal-noise, measured-depth, and planarity outputs for `scene01_white_d050_r01`.

Required pipeline:

```text
depth.npz
↓
derive ROI key
↓
load distance-specific planar ROI YAML
↓
crop raw ROI
├── zero/max-uint16 quality metrics
└── prepare depth
        ↓
    temporal noise
    measured depth
        ↓
    per-frame plane fitting
        ↓
    planarity metrics
↓
summary.yaml
```

### Milestone B: Scene 04 First Valid Result

Completed foundation:

```text
src/preprocessing/edge_roi.py
tools/select_edge_roi.py
src/geometry/edge_geometry.py
src/metrics/edge_discontinuity.py
Scene 04 synthetic tests
```

The selection workflow has been manually exercised, including ROI/line
annotation, confirmation, preview generation, and YAML/PNG persistence. The
formal CLI entry point is covered by automated tests and a direct `--help`
smoke test. The available `scene01_white_d050_r01` dataset was used only to
validate workflow mechanics. Its foreground/background geometry is not a
controlled Scene 04 distance setup and does not satisfy the formal metric
acceptance milestone.

Remaining implementation:

```text
src/visualization/edge.py
tools/analyze_edge.py
tools/summarize_edge.py
```

Remaining validation:

```text
formal Scene 04 dataset
representative label-map inspection
aggregate probability-profile inspection
frame rejection and temporal stability inspection
repeat and cross-distance analysis
```

Target:

> Select and reuse a Scene 04 multi-ROI configuration, then produce a visually validated edge-classification profile and edge-quality summary for `scene04_edge_d050_r01`.

Required pipeline:

```text
depth.npz
↓
load Scene 04 ROI YAML
├── foreground reference ROI
├── background reference ROI
├── edge analysis ROI
└── nominal edge line
        ↓
signed-distance map
        ↓
per-frame reference estimation
        ↓
edge pixel classification
        ↓
distance-bin probability profile
        ↓
bleeding / mixed / outlier / invalid metrics
        ↓
transition width and nominal-line offset
        ↓
summary.yaml + CSV + diagnostic plots
```

Scene 04 batch analysis should begin only after:

```text
synthetic tests pass
one representative label map is manually verified
one aggregate edge profile is manually verified
metric denominator definitions are confirmed
```

---

## 24. Documentation Split Decision

Keep the planar baseline and depth-only Scene 04 implementation in this document because the following topics share the same depth dataset and preprocessing foundation:

```text
ROI workflow
depth preprocessing
depth-quality metrics
measured-depth metrics
camera back-projection
plane fitting
planarity
Scene 04 depth-edge classification
edge-discontinuity metrics
baseline analysis orchestration
```

RGB–Depth alignment is a separate experiment family and must be maintained in:

```text
docs/rgb_depth_alignment_plan.md
```

The alignment document owns:

```text
alignment target setup
RGB/depth extraction requirements
frame pairing
alignment ROI
foreground and edge extraction
pixel-domain alignment metrics
static alignment validation
dynamic synchronization extension
```

Shared infrastructure may be referenced from both documents, but metric definitions and milestones must remain separate.

Reconsider further splitting this baseline document only when:

```text
tools/select_roi.py becomes a larger annotation application
```

or:

```text
baseline metric definitions require extensive mathematical methodology and validation notes
```

Possible future documents:

```text
docs/roi_workflow.md
docs/baseline_metrics.md
```
