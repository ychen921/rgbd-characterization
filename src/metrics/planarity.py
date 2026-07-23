"""Compute geometric planarity metrics from prepared depth ROIs."""

from dataclasses import dataclass
from numbers import Integral, Real

import numpy as np

from src.geometry.camera import (
    CameraIntrinsics,
    depth_roi_to_points,
)
from src.geometry.plane_fitting import (
    fit_plane_svd,
    point_to_plane_distances,
)


DEFAULT_INLIER_THRESHOLD_MM = 5.0
DEFAULT_MIN_VALID_POINTS = 100


@dataclass(frozen=True)
class FramePlaneResult:
    """Store plane geometry and residual metrics for one depth frame."""

    normal: np.ndarray
    distance_m: float
    tilt_deg: float
    rmse_mm: float
    residual_std_mm: float
    residual_p95_abs_mm: float
    inlier_ratio: float
    valid_points: int


def compute_frame_planarity(
    depth_mm: np.ndarray,
    *,
    intrinsics: CameraIntrinsics,
    roi_x: int,
    roi_y: int,
    inlier_threshold_mm: float = DEFAULT_INLIER_THRESHOLD_MM,
    min_valid_points: int = DEFAULT_MIN_VALID_POINTS,
) -> FramePlaneResult:
    """Fit one ROI depth frame and calculate geometric planarity metrics.

    ``depth_mm`` must be a prepared float32 ROI with shape ``(H, W)``.
    Excluded samples must be represented as NaN. Depth values are expressed
    in millimetres, while the fitted camera-space plane uses metres.
    """

    _validate_depth_frame(depth_mm)

    threshold = _validate_inlier_threshold(
        inlier_threshold_mm=inlier_threshold_mm
    )
    minimum_points = _validate_min_valid_points(
        min_valid_points=min_valid_points
    )

    # Restore full-image pixel coordinates and back-project valid depth
    # samples into the depth camera's three-dimensional coordinate system.
    points = depth_roi_to_points(
        depth_mm=depth_mm,
        intrinsics=intrinsics,
        roi_x=roi_x,
        roi_y=roi_y,
    )

    valid_points = points.shape[0]

    if valid_points < minimum_points:
        raise ValueError(
            "Insufficient valid depth points for plane fitting: "
            f"got {valid_points}, require at least {minimum_points}"
        )

    # Fit one independent plane to this frame so temporal movement is not
    # mixed with the frame's spatial surface residuals
    plane = fit_plane_svd(points=points)

    # Plane fitting operates in metres. Convert residuals to millimetres
    # before calculating user-facing characterization metrics.
    residual_mm = point_to_plane_distances(
        points=points,
        plane=plane,
    ) * 1000.0

    abs_residual_mm = np.abs(residual_mm)

    # For a unit-normal plane, abs(d) is the perpendicular distance from
    # the camera origin to the fitted plane.
    distance_m = abs(plane.d)

    # Compare the fitted normal with the camera optical axis [0, 0, 1].
    # Clipping protects arccos from small floating-point excursions.
    tilt_deg = np.degrees(
        np.arccos(
            np.clip(plane.normal[2], -1.0, 1.0)
        )
    )

    rmse_mm = np.sqrt(np.mean(residual_mm ** 2))
    residual_std_mm = np.std(residual_mm, ddof=0)
    residual_p95_abs_mm = np.percentile(abs_residual_mm, 95)
    inlier_ratio = np.mean(abs_residual_mm <= threshold)

    result = FramePlaneResult(
        normal=plane.normal,
        distance_m=float(distance_m),
        tilt_deg=float(tilt_deg),
        rmse_mm=float(rmse_mm),
        residual_std_mm=float(residual_std_mm),
        residual_p95_abs_mm=float(
            residual_p95_abs_mm
        ),
        inlier_ratio=float(inlier_ratio),
        valid_points=valid_points,
    )

    return result


def _validate_depth_frame(
    depth_mm: np.ndarray,
) -> None:
    """Validate one prepared ROI depth frame."""

    if not isinstance(depth_mm, np.ndarray):
        raise TypeError(
            "depth_mm must be a numpy.ndarray; "
            f"got {type(depth_mm).__name__}"
        )

    if depth_mm.ndim != 2:
        raise ValueError(
            "depth_mm must have shape (H, W); "
            f"got shape {depth_mm.shape}"
        )

    if depth_mm.dtype != np.float32:
        raise ValueError(
            "depth_mm must have dtype float32; "
            f"got {depth_mm.dtype}"
        )

    if depth_mm.shape[0] == 0 or depth_mm.shape[1] == 0:
        raise ValueError(
            "depth_mm spatial dimensions must be positive; "
            f"got shape {depth_mm.shape}"
        )

    if np.any(np.isinf(depth_mm)):
        raise ValueError(
            "depth_mm must contain only finite values or NaN"
        )


def _validate_inlier_threshold(
    inlier_threshold_mm: float,
) -> float:
    """Validate and normalize the residual inlier threshold."""

    if (
        isinstance(inlier_threshold_mm, (bool, np.bool_))
        or not isinstance(inlier_threshold_mm, Real)
    ):
        raise TypeError(
            "inlier_threshold_mm must be a real number"
        )

    normalized = float(inlier_threshold_mm)

    if not np.isfinite(normalized) or normalized <= 0.0:
        raise ValueError(
            "inlier_threshold_mm must be finite and positive"
        )

    return normalized


def _validate_min_valid_points(
    min_valid_points: int,
) -> int:
    """Validate and normalize the minimum point count."""

    if (
        isinstance(min_valid_points, (bool, np.bool_))
        or not isinstance(min_valid_points, Integral)
    ):
        raise TypeError(
            "min_valid_points must be an integer"
        )

    normalized = int(min_valid_points)

    if normalized < 3:
        raise ValueError(
            "min_valid_points must be at least 3"
        )

    return normalized
