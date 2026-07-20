# Scene01 White Board Depth Characterization Summary

## 1. Scope

This document summarizes the Scene01 white-board depth characterization results at the following nominal distances:

- d050: approximately 500 mm
- d100: approximately 1000 mm
- d150: approximately 1500 mm
- d200: approximately 2000 mm
- d300: approximately 3000 mm

Each distance contains three repeated recordings.

The analysis focuses on:

- measured ROI depth
- temporal noise
- repeatability
- invalid-depth ratio
- distance-dependent degradation
- limitations of the current manual ground-truth setup

---

## 2. Important Limitation: Distance Is Only an Approximate Nominal Reference

The board distance was measured manually from approximately the middle of the camera body. The reference point was not strictly aligned with the depth optical center, and manual placement error was not quantified.

Therefore:

- `measured_depth - nominal_distance` must **not** be interpreted as absolute sensor accuracy error
- the difference may contain camera-body reference offset, board placement error, tape-measure error, and board tilt
- the current dataset is still valid for temporal precision, repeatability, completeness, and distance-dependent trend analysis

Recommended terminology:

- use `measured-to-nominal difference`
- use `nominal-distance discrepancy`
- avoid `absolute accuracy error`
- avoid `sensor bias` unless a controlled ground-truth reference is available

---

## 3. Dataset Overview

| Distance | Repeats | Frames per repeat | ROI size | ROI pixels |
|---|---:|---:|---:|---:|
| d050 | 3 | 892 / 448 / 862 | 279 × 164 | 45,756 |
| d100 | 3 | 890 / 861 / 860 | 237 × 191 | 45,267 |
| d150 | 3 | 892 / 860 / 861 | 151 × 125 | 18,875 |
| d200 | 3 | 862 / 891 / 892 | 109 × 94 | 10,246 |
| d300 | 3 | 892 / 891 / 891 | 78 × 72 | 5,616 |

Each distance uses one shared ROI across its three repeats.

The ROI becomes progressively smaller at longer distances because the white board occupies fewer image pixels.

---

## 4. Per-Distance Results

## 4.1 d050 — Approximately 500 mm

### Summary

| Metric | r01 | r02 | r03 | Aggregate |
|---|---:|---:|---:|---:|
| Measured median | 512.0 mm | 512.0 mm | 512.0 mm | 512.0 mm |
| Measured mean | 512.0 mm | 512.0 mm | 512.0 mm | 512.0 mm |
| Frame-median std | 0.000 mm | 0.000 mm | 0.000 mm | 0.000 mm |
| Temporal median std | 0.440 mm | 0.444 mm | 0.441 mm | 0.442 mm |
| Temporal mean std | 0.418 mm | 0.422 mm | 0.420 mm | 0.420 mm |
| Temporal P95 std | 0.610 mm | 0.619 mm | 0.619 mm | 0.616 mm |
| Zero ratio | 0% | 0% | 0% | 0% |
| 65535 ratio | 0% | 0% | 0% | 0% |

### Interpretation

- Temporal noise is very low.
- All three repeats produce the same ROI median.
- No invalid zero values or 65535 values are observed.
- The ROI-level median is fully stable across frames, although individual pixels still have about 0.44 mm temporal standard deviation.

### Nominal-distance discrepancy

- Nominal distance: approximately 500 mm
- Measured median: 512 mm
- Difference: +12 mm

This +12 mm must not be treated as calibrated sensor bias because the physical reference point was not controlled.

---

## 4.2 d100 — Approximately 1000 mm

### Summary

| Metric | r01 | r02 | r03 | Aggregate |
|---|---:|---:|---:|---:|
| Measured median | 983.0 mm | 983.0 mm | 983.0 mm | 983.0 mm |
| Measured mean | 982.630 mm | 982.945 mm | 982.636 mm | 982.737 mm |
| Frame-median std | 0.483 mm | 0.227 mm | 0.481 mm | 0.397 mm |
| Temporal median std | 1.448 mm | 1.418 mm | 1.464 mm | 1.443 mm |
| Temporal mean std | 1.534 mm | 1.508 mm | 1.549 mm | 1.530 mm |
| Temporal P95 std | 2.463 mm | 2.413 mm | 2.450 mm | 2.442 mm |
| Zero ratio | 0% | 0% | 0% | 0% |
| 65535 ratio | 0% | 0% | 0% | 0% |

### Interpretation

- Temporal median noise increases to about 1.44 mm.
- This is approximately 3.3 times the d050 value.
- All three repeats still produce the same measured median.
- Depth completeness remains 100%.

### Nominal-distance discrepancy

- Nominal distance: approximately 1000 mm
- Measured median: 983 mm
- Difference: -17 mm

This value is only an exploratory nominal-distance difference.

---

## 4.3 d150 — Approximately 1500 mm

### Summary

| Metric | r01 | r02 | r03 | Aggregate |
|---|---:|---:|---:|---:|
| Measured median | 1486.0 mm | 1486.0 mm | 1486.0 mm | 1486.0 mm |
| Measured mean | 1486.101 mm | 1486.323 mm | 1486.046 mm | 1486.157 mm |
| Frame-median std | 0.438 mm | 0.736 mm | 0.301 mm | 0.492 mm |
| Temporal median std | 2.792 mm | 2.789 mm | 2.775 mm | 2.786 mm |
| Temporal mean std | 2.855 mm | 2.841 mm | 2.829 mm | 2.841 mm |
| Temporal P95 std | 3.827 mm | 3.806 mm | 3.774 mm | 3.802 mm |
| Zero ratio | 0% | 0% | 0% | 0% |
| 65535 ratio | 0% | 0% | 0% | 0% |

### Interpretation

- Temporal median noise rises to about 2.79 mm.
- Noise is approximately 1.9 times d100 and 6.3 times d050.
- Repeat-level measured median remains fully consistent.
- Depth completeness remains 100%.
- r02 has slightly larger frame-level variation, but no repeat-level drift is observed.

### Nominal-distance discrepancy

- Nominal distance: approximately 1500 mm
- Measured median: 1486 mm
- Difference: -14 mm

This must not be interpreted as a validated absolute depth error.

---

## 4.4 d200 — Approximately 2000 mm

### Summary

| Metric | r01 | r02 | r03 | Aggregate |
|---|---:|---:|---:|---:|
| Measured median | 1940.0 mm | 1940.0 mm | 1940.0 mm | 1940.0 mm |
| Measured mean | 1939.997 mm | 1939.980 mm | 1940.015 mm | 1939.997 mm |
| Frame-median std | 0.213 mm | 0.306 mm | 0.214 mm | 0.244 mm |
| Temporal median std | 4.769 mm | 4.640 mm | 4.622 mm | 4.677 mm |
| Temporal mean std | 4.906 mm | 4.785 mm | 4.760 mm | 4.817 mm |
| Temporal P95 std | 6.969 mm | 6.854 mm | 6.779 mm | 6.867 mm |
| Zero ratio | 0% | 0% | 0% | 0% |
| 65535 ratio | 0% | 0% | 0% | 0% |

### Interpretation

- Temporal median noise reaches about 4.68 mm.
- This is approximately 10.6 times the d050 value.
- Individual pixels fluctuate more strongly, but the large-area ROI median remains very stable.
- All three repeats produce the same measured median.
- No invalid depths are observed.

### Nominal-distance discrepancy

- Nominal distance: approximately 2000 mm
- Measured median: 1940 mm
- Difference: -60 mm

The larger discrepancy may reflect manual placement or reference-point inconsistency. It must not be assigned directly to sensor scale error.

---

## 4.5 d300 — Approximately 3000 mm

### Summary

| Metric | r01 | r02 | r03 | Aggregate |
|---|---:|---:|---:|---:|
| Measured median | 2920 mm | 2920 mm | 2914 mm | 2918 mm |
| Measured mean | 2918.174 mm | 2918.314 mm | 2916.827 mm | 2917.772 mm |
| Frame-median std | 2.759 mm | 2.734 mm | 3.061 mm | 2.851 mm |
| Temporal median std | 11.360 mm | 11.710 mm | 11.554 mm | 11.541 mm |
| Temporal mean std | 11.816 mm | 12.098 mm | 11.993 mm | 11.969 mm |
| Temporal P95 std | 15.750 mm | 16.340 mm | 16.450 mm | 16.180 mm |
| Zero ratio | 0.0232% | 0.0196% | 0.0162% | 0.0197% |
| 65535 ratio | 0% | 0% | 0% | 0% |

### Interpretation

- d300 is the first distance with clear degradation across multiple metrics.
- Temporal median noise increases to about 11.54 mm.
- This is about 2.47 times d200 and about 26 times d050.
- Repeat medians are no longer identical: the repeat range is 6 mm.
- Frame-level ROI median variation increases substantially.
- Small but measurable zero-depth dropout appears.
- Valid-depth completeness is still approximately 99.98%.

### Nominal-distance discrepancy

- Nominal distance: approximately 3000 mm
- Measured result: approximately 2914–2920 mm
- Difference: approximately -80 to -86 mm

This remains an exploratory difference because the physical distance reference is not controlled.

---

## 5. Cross-Distance Comparison

| Metric | d050 | d100 | d150 | d200 | d300 |
|---|---:|---:|---:|---:|---:|
| Aggregate measured median | 512 mm | 983 mm | 1486 mm | 1940 mm | 2918 mm |
| Temporal median std | 0.442 mm | 1.443 mm | 2.786 mm | 4.677 mm | 11.541 mm |
| Temporal mean std | 0.420 mm | 1.530 mm | 2.841 mm | 4.817 mm | 11.969 mm |
| Temporal P95 std | 0.616 mm | 2.442 mm | 3.802 mm | 6.867 mm | 16.180 mm |
| Mean frame-median std | 0.000 mm | 0.397 mm | 0.492 mm | 0.244 mm | 2.851 mm |
| Repeat median range | 0 mm | 0 mm | 0 mm | 0 mm | 6 mm |
| Mean zero ratio | 0% | 0% | 0% | 0% | 0.0197% |
| 65535 ratio | 0% | 0% | 0% | 0% | 0% |
| ROI pixels | 45,756 | 45,267 | 18,875 | 10,246 | 5,616 |

---

## 6. Main Findings

### 6.1 Temporal precision degrades strongly with distance

Temporal median standard deviation:

- 0.5 m: 0.442 mm
- 1.0 m: 1.443 mm
- 1.5 m: 2.786 mm
- 2.0 m: 4.677 mm
- 3.0 m: 11.541 mm

The growth is clearly nonlinear over the tested distances.

Increment by interval:

- 0.5 to 1.0 m: +1.00 mm
- 1.0 to 1.5 m: +1.34 mm
- 1.5 to 2.0 m: +1.89 mm
- 2.0 to 3.0 m: +6.86 mm

The most significant degradation occurs between approximately 2 m and 3 m.

### 6.2 Repeatability is strong up to approximately 2 m

For d050, d100, d150, and d200:

- all three repeats produce exactly the same measured median
- repeat-level temporal-noise values are also very close

At d300:

- measured median range becomes 6 mm
- frame-level aggregate depth variation increases
- repeatability is still usable, but no longer as strong as the nearer distances

### 6.3 Large-area ROI median is much more stable than individual pixels

At d200:

- pixel-level temporal median std is about 4.68 mm
- frame-level ROI median std is only about 0.24 mm

This means spatial aggregation over a large planar ROI suppresses much of the random pixel noise.

At d300, this benefit remains, but the aggregate ROI median itself starts fluctuating more strongly.

### 6.4 Completeness remains high

- d050 to d200: zero ratio is 0%
- d300: average zero ratio is approximately 0.0197%
- all distances: 65535 ratio is 0%

The sensor maintains high valid-depth coverage over the tested white-board ROI.

### 6.5 Approximately 3 m is the first clear degradation point

At d300, three degradation indicators appear together:

- temporal noise rises sharply
- repeat-level median is no longer identical
- zero-depth pixels begin to appear

This makes d300 the first clearly weaker operating point in Scene01.

---

## 7. ROI Size Considerations

The ROI pixel count decreases with distance:

- d050: 45,756
- d100: 45,267
- d150: 18,875
- d200: 10,246
- d300: 5,616

This does not directly create per-pixel temporal noise, but it affects:

- spatial representativeness
- sensitivity to local surface variation
- boundary contamination risk
- robustness of aggregate spatial statistics

The d300 ROI still contains more than 5,000 pixels, but its coverage is only about 12.3% of the d050 ROI pixel count.

For further analysis, inspect:

- `temporal_std.npy`
- invalid-depth maps
- ROI overlays
- board-edge distance
- spatial noise hot spots

---

## 8. Conclusions

### Supported conclusions

1. Temporal depth noise increases substantially with distance.
2. The increase is nonlinear across the tested range.
3. Repeatability is strong from approximately 0.5 m to 2.0 m.
4. At approximately 3.0 m, temporal precision, repeatability, and completeness all begin to degrade.
5. The central white-board ROI maintains nearly complete valid-depth coverage across all distances.
6. Large-area ROI median depth is substantially more stable than individual-pixel depth.

### Unsupported conclusions

The current dataset must not be used to claim:

- calibrated absolute depth accuracy
- exact sensor bias at each distance
- verified scale error
- absolute error percentage relative to the optical center

The current manual distance reference is insufficient for those claims.

---

## 9. Recommended Follow-up

### For the current dataset

- preserve the current results as temporal precision and completeness characterization
- retain nominal-distance discrepancy only as an exploratory field
- visualize temporal-noise maps for d200 and d300
- inspect whether d300 invalid pixels are randomly distributed or spatially clustered
- compare different materials and lighting conditions using the same metrics

### For future absolute-accuracy validation

Use a controlled setup with:

- clearly defined depth optical-center reference
- rigid camera and target mounting
- perpendicular planar target
- laser distance meter, linear stage, or calibrated mechanical fixture
- repeated ground-truth measurements
- recorded ground-truth uncertainty

Suggested metadata:

```yaml
ground_truth:
  type: manual_approximate
  reference_point: not_strictly_defined
  uncertainty_mm: unknown
```

For a future controlled setup:

```yaml
ground_truth:
  type: calibrated
  reference_point: depth_optical_center
  uncertainty_mm: <measured uncertainty>
```

---

## 10. Final Assessment

Scene01 demonstrates that the tested RGB-D sensor provides:

- very low temporal noise at approximately 0.5 m
- stable and repeatable planar-depth output through approximately 2.0 m
- increasingly degraded temporal precision with distance
- a clear degradation transition near approximately 3.0 m
- high depth completeness across the entire tested range

The strongest and most defensible conclusion is the distance-dependent temporal-noise trend, not the measured-to-nominal distance difference.
