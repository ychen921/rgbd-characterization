# Sensor Characterization Baseline Analysis Plan

## 0. Scope

This document covers the depth-only baseline characterization pipeline:

- depth quality
- measured depth
- temporal noise
- camera back-projection
- per-frame plane fitting
- planarity
- repeat and cross-distance comparison

The following topics are outside the scope of this document:

- RGB–Depth spatial alignment
- RGB/depth edge correspondence
- color/depth frame pairing
- dynamic RGB–Depth synchronization

Those topics are specified separately in:

```text
docs/rgb_depth_alignment_plan.md
```

The two pipelines may share extraction, dataset loading, ROI utilities, configuration parsing, and result-writing infrastructure, but they must use separate analyzers and metrics.


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

The next stage is:

> ROI-assisted white-board baseline characterization.

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
│       ├── scene01_white_d150.yaml
│       └── ...
│
├── tools/
│   ├── extract_dataset.py
│   ├── inspect_dataset.py
│   ├── select_roi.py
│   ├── analyze_baseline.py
│   ├── summarize_baseline.py
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
    │   ├── depth.py
    │   ├── rgb.py
    │   └── frame_pairing.py
    │
    ├── geometry/
    │   ├── camera.py
    │   └── plane_fitting.py
    │
    └── metrics/
        ├── depth_quality.py
        ├── temporal.py
        ├── measured_depth.py
        ├── planarity.py
        └── alignment.py
```

Recommended implementation order:

```text
1. Complete depth data semantic inspection
2. Implement src/preprocessing/roi.py
3. Implement tools/select_roi.py
4. Select ROI for one 50 cm distance group
5. Validate ROI reuse across repeats
6. Implement src/preprocessing/depth.py
7. Implement src/metrics/depth_quality.py
8. Implement src/metrics/measured_depth.py
9. Implement src/geometry/camera.py
10. Implement src/geometry/plane_fitting.py
11. Implement src/metrics/planarity.py
12. Implement src/metrics/temporal.py
13. Implement tools/analyze_baseline.py
14. Validate scene01_white_d050_r01
15. Analyze remaining repeats and distances
16. Implement cross-distance summary
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

Implement:

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

## 16. Baseline Analysis Tool

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

## 17. Result Structure

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

## 18. First Validation Workflow

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

## 19. Multi-Repeat and Multi-Distance Workflow

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

## 20. Cross-Distance Comparison

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

## 21. Immediate Next Tasks

```text
1. Implement src/preprocessing/roi.py

2. Test:
   - RectROI crop
   - invalid bounds
   - derive_roi_key()
   - ROI YAML round trip

3. Implement tools/select_roi.py

4. Use cv2.selectROI() on:
   scene01_white_d050_r01

5. Save:
   config/roi/scene01_white_d050.yaml

6. Run select_roi.py on d050 r02/r03
   - confirm ROI exists
   - skip GUI

7. Visually verify ROI reuse across d050 repeats

8. Compute d050 ROI:
   - zero ratio
   - 65535 ratio

9. Implement src/preprocessing/depth.py
   - 0 → NaN
   - 65535 → NaN

10. Implement src/metrics/depth_quality.py

11. Implement src/metrics/measured_depth.py

12. Implement src/geometry/camera.py
    - camera intrinsics
    - ROI depth back-projection
    - ROI offset handling

13. Implement src/geometry/plane_fitting.py
    - SVD plane fitting
    - normal normalization
    - point-to-plane residuals

14. Implement src/metrics/planarity.py
    - per-frame plane fitting
    - distance / tilt / residual metrics
    - inlier ratio

15. Implement src/metrics/temporal.py

16. Implement tools/analyze_baseline.py

17. Validate scene01_white_d050_r01

18. Analyze remaining d050 repeats

19. Repeat ROI selection and analysis for each new distance

20. Implement cross-distance summary
```

---

## 21.1 Relationship to RGB–Depth Alignment

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

## 22. Current Milestone

The next milestone is:

> Interactively select one ROI for `scene01_white_d050`, reuse it across all 50 cm repeats, and successfully produce depth-quality, temporal-noise, and measured-depth metrics for `scene01_white_d050_r01`.

Required pipeline:

```text
depth.npz
↓
derive ROI key
↓
load distance-specific ROI YAML
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

Batch analysis and cross-distance plotting should begin only after this milestone passes.

---

## 23. Documentation Split Decision

Keep the depth baseline implementation in this document because the following topics form one continuous pipeline:

```text
ROI workflow
depth preprocessing
depth-quality metrics
measured-depth metrics
camera back-projection
plane fitting
planarity
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
