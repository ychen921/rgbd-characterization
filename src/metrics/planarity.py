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


@dataclass(frozen=True)
class PlanarityResult:
    """Store frame-aligned and aggregate planarity metrics."""

    frame_normal: np.ndarray
    frame_distance_m: np.ndarray
    frame_tilt_deg: np.ndarray
    frame_rmse_mm: np.ndarray
    frame_residual_std_mm: np.ndarray
    frame_p95_abs_mm: np.ndarray
    frame_inlier_ratio: np.ndarray
    frame_valid_points: np.ndarray
    frame_fit_succeeded: np.ndarray

    median_distance_m: float
    distance_std_mm: float
    median_tilt_deg: float
    tilt_std_deg: float
    median_rmse_mm: float
    p95_rmse_mm: float
    median_p95_abs_mm: float
    median_inlier_ratio: float

    successful_frames: int
    failed_frames: int


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


def compute_planarity(
    depth: np.ndarray,
    *,
    intrinsics: CameraIntrinsics,
    roi_x: int,
    roi_y: int,
    inlier_threshold_mm: float = DEFAULT_INLIER_THRESHOLD_MM,
    min_valid_points: int = DEFAULT_MIN_VALID_POINTS,
) -> PlanarityResult:
    """Compute frame-aligned and aggregate planarity metrics.

    ``depth`` must contain prepared float32 depth frames with shape
    ``(N, H, W)``. Excluded samples must be represented as NaN.

    Frames with fewer than ``min_valid_points`` valid samples remain aligned
    with the input sequence, but their geometric metrics are recorded as NaN.
    Other geometry or configuration errors are not silently ignored.
    """

    _validate_depth_sequence(depth)

    threshold = _validate_inlier_threshold(
        inlier_threshold_mm
    )
    minimum_points = _validate_min_valid_points(
        min_valid_points
    )

    _validate_roi_context(
        depth,
        intrinsics=intrinsics,
        roi_x=roi_x,
        roi_y=roi_y,
    )

    num_frames = depth.shape[0]

    # Preallocate one slot per input frame so skipped fits do not change
    # the original frame indices.
    frame_normal = np.full(
        (num_frames, 3),
        np.nan,
        dtype=np.float64,
    )
    frame_distance_m = np.full(
        num_frames,
        np.nan,
        dtype=np.float64,
    )
    frame_tilt_deg = np.full(
        num_frames,
        np.nan,
        dtype=np.float64,
    )
    frame_rmse_mm = np.full(
        num_frames,
        np.nan,
        dtype=np.float64,
    )
    frame_residual_std_mm = np.full(
        num_frames,
        np.nan,
        dtype=np.float64,
    )
    frame_p95_abs_mm = np.full(
        num_frames,
        np.nan,
        dtype=np.float64,
    )
    frame_inlier_ratio = np.full(
        num_frames,
        np.nan,
        dtype=np.float64,
    )
    frame_valid_points = np.zeros(
        num_frames,
        dtype=np.int64,
    )
    frame_fit_succeeded = np.zeros(
        num_frames,
        dtype=bool,
    )

    for frame_index, frame in enumerate(depth):
        # Match depth_roi_to_points(): only finite, positive depth values
        # contribute camera-space points.
        valid_points = int(
            np.count_nonzero(
                np.isfinite(frame) & (frame > 0.0)
            )
        )

        frame_valid_points[frame_index] = valid_points

        # Insufficient frames remain present in every result array. Their
        # floating-point metrics retain NaN and fit_succeeded remains false.
        if valid_points < minimum_points:
            continue

        frame_result = compute_frame_planarity(
            frame,
            intrinsics=intrinsics,
            roi_x=roi_x,
            roi_y=roi_y,
            inlier_threshold_mm=threshold,
            min_valid_points=minimum_points,
        )

        # Store every successful fit at its original frame index.
        frame_normal[frame_index] = (
            frame_result.normal
        )
        frame_distance_m[frame_index] = (
            frame_result.distance_m
        )
        frame_tilt_deg[frame_index] = (
            frame_result.tilt_deg
        )
        frame_rmse_mm[frame_index] = (
            frame_result.rmse_mm
        )
        frame_residual_std_mm[frame_index] = (
            frame_result.residual_std_mm
        )
        frame_p95_abs_mm[frame_index] = (
            frame_result.residual_p95_abs_mm
        )
        frame_inlier_ratio[frame_index] = (
            frame_result.inlier_ratio
        )

        frame_fit_succeeded[frame_index] = True

    successful_frames = int(
        np.count_nonzero(frame_fit_succeeded)
    )

    failed_frames = num_frames - successful_frames

    if successful_frames == 0:
        return _undefined_planarity_result(
            frame_normal=frame_normal,
            frame_distance_m=frame_distance_m,
            frame_tilt_deg=frame_tilt_deg,
            frame_rmse_mm=frame_rmse_mm,
            frame_residual_std_mm=(
                frame_residual_std_mm
            ),
            frame_p95_abs_mm=frame_p95_abs_mm,
            frame_inlier_ratio=frame_inlier_ratio,
            frame_valid_points=frame_valid_points,
            frame_fit_succeeded=(
                frame_fit_succeeded
            ),
            failed_frames=failed_frames,
        )

    # Aggregate only successfully fitted frames. Failed frame slots remain
    # available in the frame-aligned arrays for later CSV output.
    successful_distance_m = frame_distance_m[frame_fit_succeeded]
    successful_tilt_deg = frame_tilt_deg[frame_fit_succeeded]
    successful_rmse_mm = frame_rmse_mm[frame_fit_succeeded]
    successful_p95_abs_mm = frame_p95_abs_mm[frame_fit_succeeded]
    successful_inlier_ratio = frame_inlier_ratio[frame_fit_succeeded]

    results = PlanarityResult(
        frame_normal=frame_normal,
        frame_distance_m=frame_distance_m,
        frame_tilt_deg=frame_tilt_deg,
        frame_rmse_mm=frame_rmse_mm,
        frame_residual_std_mm=(
            frame_residual_std_mm
        ),
        frame_p95_abs_mm=frame_p95_abs_mm,
        frame_inlier_ratio=frame_inlier_ratio,
        frame_valid_points=frame_valid_points,
        frame_fit_succeeded=frame_fit_succeeded,
        median_distance_m=float(
            np.median(successful_distance_m)
        ),
        # Frame distance is stored in metres, but its temporal spread is
        # reported in millimetres for consistency with residual metrics.
        distance_std_mm=float(
            np.std(
                successful_distance_m,
                ddof=0,
            )
            * 1000.0
        ),
        median_tilt_deg=float(
            np.median(successful_tilt_deg)
        ),
        tilt_std_deg=float(
            np.std(
                successful_tilt_deg,
                ddof=0,
            )
        ),
        median_rmse_mm=float(
            np.median(successful_rmse_mm)
        ),
        p95_rmse_mm=float(
            np.percentile(
                successful_rmse_mm,
                95,
            )
        ),
        median_p95_abs_mm=float(
            np.median(successful_p95_abs_mm)
        ),
        median_inlier_ratio=float(
            np.median(successful_inlier_ratio)
        ),
        successful_frames=successful_frames,
        failed_frames=failed_frames,
    )

    return results


def _undefined_planarity_result(
    *,
    frame_normal: np.ndarray,
    frame_distance_m: np.ndarray,
    frame_tilt_deg: np.ndarray,
    frame_rmse_mm: np.ndarray,
    frame_residual_std_mm: np.ndarray,
    frame_p95_abs_mm: np.ndarray,
    frame_inlier_ratio: np.ndarray,
    frame_valid_points: np.ndarray,
    frame_fit_succeeded: np.ndarray,
    failed_frames: int,
) -> PlanarityResult:
    """Return aligned frame arrays with undefined aggregate statistics."""
    return PlanarityResult(
        frame_normal=frame_normal,
        frame_distance_m=frame_distance_m,
        frame_tilt_deg=frame_tilt_deg,
        frame_rmse_mm=frame_rmse_mm,
        frame_residual_std_mm=(
            frame_residual_std_mm
        ),
        frame_p95_abs_mm=frame_p95_abs_mm,
        frame_inlier_ratio=frame_inlier_ratio,
        frame_valid_points=frame_valid_points,
        frame_fit_succeeded=frame_fit_succeeded,
        median_distance_m=float("nan"),
        distance_std_mm=float("nan"),
        median_tilt_deg=float("nan"),
        tilt_std_deg=float("nan"),
        median_rmse_mm=float("nan"),
        p95_rmse_mm=float("nan"),
        median_p95_abs_mm=float("nan"),
        median_inlier_ratio=float("nan"),
        successful_frames=0,
        failed_frames=failed_frames,
    )


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


def _validate_depth_sequence(
    depth: np.ndarray,
) -> None:
    """Validate prepared ROI frames for planarity computation."""

    if not isinstance(depth, np.ndarray):
        raise TypeError(
            "depth must be a numpy.ndarray; "
            f"got {type(depth).__name__}"
        )

    if depth.ndim != 3:
        raise ValueError(
            "depth must have shape (N, H, W); "
            f"got shape {depth.shape}"
        )

    if depth.dtype != np.float32:
        raise ValueError(
            "depth must have dtype float32; "
            f"got {depth.dtype}"
        )

    if depth.shape[1] == 0 or depth.shape[2] == 0:
        raise ValueError(
            "depth spatial dimensions must be positive; "
            f"got shape {depth.shape}"
        )

    if np.any(np.isinf(depth)):
        raise ValueError(
            "depth must contain only finite values or NaN"
        )


def _validate_roi_context(
    depth: np.ndarray,
    *,
    intrinsics: CameraIntrinsics,
    roi_x: int,
    roi_y: int,
) -> None:
    """Validate ROI placement against the calibrated image resolution."""

    if not isinstance(intrinsics, CameraIntrinsics):
        raise TypeError(
            "intrinsics must be CameraIntrinsics; "
            f"got {type(intrinsics).__name__}"
        )

    for field_name, value in (
        ("roi_x", roi_x),
        ("roi_y", roi_y),
    ):
        if not isinstance(value, int) or isinstance(value, bool):
            raise ValueError(
                f"{field_name} must be an integer"
            )

        if value < 0:
            raise ValueError(
                f"{field_name} must be non-negative"
            )

    _, roi_height, roi_width = depth.shape

    if roi_x + roi_width > intrinsics.width:
        raise ValueError(
            "Depth ROI exceeds calibrated image width"
        )

    if roi_y + roi_height > intrinsics.height:
        raise ValueError(
            "Depth ROI exceeds calibrated image height"
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
