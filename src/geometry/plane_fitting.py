"""Deterministic plane fitting and point-to-plane geometry."""

from dataclasses import dataclass
import math

import numpy as np


@dataclass(frozen=True)
class PlaneModel:
    """Represent a normalized plane in camera coordinates."""

    normal: np.ndarray
    d: float
    centroid: np.ndarray

    def __post_init__(self) -> None:
        """Validate the plane representation."""
        if not isinstance(self.normal, np.ndarray):
            raise TypeError(
                "normal must be a numpy.ndarray; "
                f"got {type(self.normal).__name__}"
            )

        if self.normal.shape != (3,):
            raise ValueError(
                "normal must have shape (3,); "
                f"got shape {self.normal.shape}"
            )

        if not isinstance(self.centroid, np.ndarray):
            raise TypeError(
                "centroid must be a numpy.ndarray; "
                f"got {type(self.centroid).__name__}"
            )

        if self.centroid.shape != (3,):
            raise ValueError(
                "centroid must have shape (3,); "
                f"got shape {self.centroid.shape}"
            )

        if not np.all(np.isfinite(self.normal)):
            raise ValueError("normal must contain only finite values")

        if not np.all(np.isfinite(self.centroid)):
            raise ValueError("centroid must contain only finite values")

        if (
            not isinstance(self.d, (int, float))
            or isinstance(self.d, bool)
            or not math.isfinite(self.d)
        ):
            raise ValueError("d must be a finite number")

        normal_length = np.linalg.norm(self.normal)

        if not np.isclose(
            normal_length,
            1.0,
            rtol=1e-9,
            atol=1e-12,
        ):
            raise ValueError("normal must have unit length")

        if self.normal[2] < 0:
            raise ValueError(
                "normal must point toward non-negative camera Z"
            )


def _validate_points(
    points: np.ndarray,
    *,
    min_points: int,
) -> np.ndarray:
    """Validate and convert an XYZ point array to float64."""
    if not isinstance(points, np.ndarray):
        raise TypeError(
            "points must be a numpy.ndarray; "
            f"got {type(points).__name__}"
        )

    if points.ndim != 2 or points.shape[1] != 3:
        raise ValueError(
            "points must have shape (N, 3); "
            f"got shape {points.shape}"
        )

    if points.shape[0] < min_points:
        raise ValueError(
            f"At least {min_points} points are required"
        )

    if not np.issubdtype(points.dtype, np.number):
        raise TypeError("points must contain numeric values")

    if np.issubdtype(points.dtype, np.complexfloating):
        raise TypeError("points must contain real-valued coordinates")

    converted = points.astype(
        np.float64,
        copy=False,
    )

    if not np.all(np.isfinite(converted)):
        raise ValueError(
            "points must contain only finite values"
        )

    return converted


def fit_plane_svd(points: np.ndarray) -> PlaneModel:
    """Fit a plane to camera-space XYZ points using deterministic SVD."""
    validated = _validate_points(
        points=points,
        min_points=3,
    )

    # Move the point cloud to its centroid so the SVD describes
    # surface orientation independently of its camera-space position.
    centroid = np.mean(
        validated,
        axis=0,
    )
    centered = validated - centroid

    # Identical or collinear points do not uniquely define a plane.
    if np.linalg.matrix_rank(centered) < 2:
        raise ValueError(
            "At least three non-collinear points are required"
        )

    # The right-singular vector associated with the smallest singular
    # value is perpendicular to the best-fit surface.
    _, _, vh = np.linalg.svd(
        centered,
        full_matrices=False,
    )

    normal = vh[-1]
    normal = normal / np.linalg.norm(normal)

    # SVD may return either sign for the same plane. Fixing the direction
    # makes normals and tilt values comparable across frames.
    if normal[2] < 0:
        normal = -normal

    # The fitted plane passes through the point-cloud centroid:
    # normal · centroid + d = 0.
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


def point_to_plane_distances(
    points: np.ndarray,
    plane: PlaneModel,
) -> np.ndarray:
    """Return signed point-to-plane distances in metres."""
    validated = _validate_points(
        points=points,
        min_points=0,
    )

    if not isinstance(plane, PlaneModel):
        raise TypeError(
            "plane must be a PlaneModel; "
            f"got {type(plane).__name__}"
        )

    # PlaneModel guarantees a unit normal, so the algebraic plane
    # equation directly produces signed metric distance in metres.
    signed_distances = validated @ plane.normal + plane.d

    return signed_distances
