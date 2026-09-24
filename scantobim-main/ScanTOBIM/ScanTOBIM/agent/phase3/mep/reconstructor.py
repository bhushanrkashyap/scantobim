"""MEP Geometric Reconstruction Engine for Scan-to-BIM.

Phase 3B Core Requirement (Section 18):
- Object-specific reconstruction from real source points:
    * PIPE: cylinder axis fitting, radius, centerline start/end, slope, residual check
    * DUCT: oriented rectangular cross-section, width, height, length, axis
    * CABLE TRAY: elongated profile, width, depth, length, axis
    * VALVE: compact connected MEP component, envelope, connection alignment

NEVER USE HARDCODED STATIC DIMENSIONS. ALL MEASUREMENTS ARE DERIVED FROM SCAN POINTS.
CPU ONLY — NUMERICALLY STABLE.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any
import numpy as np
import structlog

logger = structlog.get_logger(__name__)


@dataclass
class ReconstructedPipe:
    """Parametric cylinder representation of a mechanical pipe."""
    pipe_id: str
    start_point_m: tuple[float, float, float]
    end_point_m: tuple[float, float, float]
    diameter_m: float
    radius_m: float
    length_m: float
    slope: float
    axis_vector: tuple[float, float, float]
    source_point_indices: list[int]
    confidence: float
    residual_rmse_m: float

    def to_dict(self) -> dict[str, Any]:
        return {
            "pipe_id": self.pipe_id,
            "start_point_m": [round(float(c), 4) for c in self.start_point_m],
            "end_point_m": [round(float(c), 4) for c in self.end_point_m],
            "diameter_m": round(float(self.diameter_m), 4),
            "radius_m": round(float(self.radius_m), 4),
            "length_m": round(float(self.length_m), 4),
            "slope": round(float(self.slope), 4),
            "axis_vector": [round(float(c), 4) for c in self.axis_vector],
            "point_count": len(self.source_point_indices),
            "confidence": round(float(self.confidence), 4),
            "residual_rmse_m": round(float(self.residual_rmse_m), 4),
        }


@dataclass
class ReconstructedDuct:
    """Parametric rectangular cross-section representation of an HVAC duct."""
    duct_id: str
    start_point_m: tuple[float, float, float]
    end_point_m: tuple[float, float, float]
    width_m: float
    height_m: float
    length_m: float
    axis_vector: tuple[float, float, float]
    source_point_indices: list[int]
    confidence: float

    def to_dict(self) -> dict[str, Any]:
        return {
            "duct_id": self.duct_id,
            "start_point_m": [round(float(c), 4) for c in self.start_point_m],
            "end_point_m": [round(float(c), 4) for c in self.end_point_m],
            "width_m": round(float(self.width_m), 4),
            "height_m": round(float(self.height_m), 4),
            "length_m": round(float(self.length_m), 4),
            "axis_vector": [round(float(c), 4) for c in self.axis_vector],
            "point_count": len(self.source_point_indices),
            "confidence": round(float(self.confidence), 4),
        }


@dataclass
class ReconstructedCableTray:
    """Parametric channel representation of an electrical cable tray."""
    tray_id: str
    start_point_m: tuple[float, float, float]
    end_point_m: tuple[float, float, float]
    width_m: float
    depth_m: float
    length_m: float
    axis_vector: tuple[float, float, float]
    source_point_indices: list[int]
    confidence: float

    def to_dict(self) -> dict[str, Any]:
        return {
            "tray_id": self.tray_id,
            "start_point_m": [round(float(c), 4) for c in self.start_point_m],
            "end_point_m": [round(float(c), 4) for c in self.end_point_m],
            "width_m": round(float(self.width_m), 4),
            "depth_m": round(float(self.depth_m), 4),
            "length_m": round(float(self.length_m), 4),
            "axis_vector": [round(float(c), 4) for c in self.axis_vector],
            "point_count": len(self.source_point_indices),
            "confidence": round(float(self.confidence), 4),
        }


@dataclass
class ReconstructedValve:
    """Parametric representation of an inline valve or fitting."""
    valve_id: str
    center_m: tuple[float, float, float]
    length_m: float
    width_m: float
    height_m: float
    connected_pipe_id: str | None
    source_point_indices: list[int]
    confidence: float

    def to_dict(self) -> dict[str, Any]:
        return {
            "valve_id": self.valve_id,
            "center_m": [round(float(c), 4) for c in self.center_m],
            "dimensions_m": {
                "length": round(float(self.length_m), 4),
                "width": round(float(self.width_m), 4),
                "height": round(float(self.height_m), 4),
            },
            "connected_pipe_id": self.connected_pipe_id,
            "point_count": len(self.source_point_indices),
            "confidence": round(float(self.confidence), 4),
        }


def reconstruct_pipe_from_points(
    points: np.ndarray,
    source_indices: list[int] | np.ndarray,
    pipe_id: str = "pipe_001",
) -> ReconstructedPipe | None:
    """Fit a cylinder and centerline to point cloud points.

    Derives diameter, start point, end point, and slope directly from 3D points.
    """
    N = len(points)
    if N < 15:
        return None

    centroid = np.mean(points, axis=0)
    centered = points - centroid
    cov = np.dot(centered.T, centered) / float(N)
    eigvals, eigvecs = np.linalg.eigh(cov)

    # Primary axis of cylinder corresponds to largest eigenvector
    axis = eigvecs[:, 2]
    norm_axis = np.linalg.norm(axis)
    if norm_axis > 1e-9:
        axis /= norm_axis
    else:
        return None

    # Project points onto cylinder axis to get longitudinal extents
    projs = np.dot(centered, axis)
    p_min = float(np.min(projs))
    p_max = float(np.max(projs))
    length = float(p_max - p_min)

    if length < 0.10:  # Minimum 10cm pipe run
        return None

    # Project points orthogonal to axis to estimate radius
    ortho_projs = centered - np.outer(projs, axis)
    radial_dists = np.linalg.norm(ortho_projs, axis=1)
    radius = float(np.median(radial_dists))

    # Reject unrealistic pipe radius (> 0.6m or < 0.01m)
    if radius < 0.01 or radius > 0.60:
        return None

    diameter = radius * 2.0
    radial_rmse = float(np.sqrt(np.mean((radial_dists - radius) ** 2)))

    # Compute start and end points along centerline
    start_pt = centroid + p_min * axis
    end_pt = centroid + p_max * axis

    # Ensure canonical direction (increasing X or increasing Y or increasing Z)
    if start_pt[0] > end_pt[0] or (abs(start_pt[0] - end_pt[0]) < 1e-4 and start_pt[1] > end_pt[1]):
        start_pt, end_pt = end_pt, start_pt
        axis = -axis

    # Calculate slope: delta_Z / horizontal_length
    dx = end_pt[0] - start_pt[0]
    dy = end_pt[1] - start_pt[1]
    dz = end_pt[2] - start_pt[2]
    horiz_len = float(np.sqrt(dx * dx + dy * dy))
    slope = float(abs(dz) / max(horiz_len, 1e-4))

    # Confidence based on radial goodness-of-fit and point count
    fit_quality = max(0.0, 1.0 - (radial_rmse / max(radius, 1e-3)))
    pt_quality = min(1.0, N / 50.0)
    conf = float(np.clip(0.6 * fit_quality + 0.4 * pt_quality, 0.3, 0.98))

    return ReconstructedPipe(
        pipe_id=pipe_id,
        start_point_m=(float(start_pt[0]), float(start_pt[1]), float(start_pt[2])),
        end_point_m=(float(end_pt[0]), float(end_pt[1]), float(end_pt[2])),
        diameter_m=diameter,
        radius_m=radius,
        length_m=length,
        slope=slope,
        axis_vector=(float(axis[0]), float(axis[1]), float(axis[2])),
        source_point_indices=[int(i) for i in source_indices],
        confidence=conf,
        residual_rmse_m=radial_rmse,
    )


def reconstruct_duct_from_points(
    points: np.ndarray,
    source_indices: list[int] | np.ndarray,
    duct_id: str = "duct_001",
) -> ReconstructedDuct | None:
    """Fit an oriented rectangular duct to points."""
    N = len(points)
    if N < 20:
        return None

    centroid = np.mean(points, axis=0)
    centered = points - centroid
    cov = np.dot(centered.T, centered) / float(N)
    eigvals, eigvecs = np.linalg.eigh(cov)

    axis = eigvecs[:, 2]  # Long axis
    v_width = eigvecs[:, 1]
    v_height = eigvecs[:, 0]

    # Length along axis
    projs_l = np.dot(centered, axis)
    p_min, p_max = float(np.min(projs_l)), float(np.max(projs_l))
    length = float(p_max - p_min)

    # Cross section extents
    projs_w = np.dot(centered, v_width)
    width = float(np.max(projs_w) - np.min(projs_w))

    projs_h = np.dot(centered, v_height)
    height = float(np.max(projs_h) - np.min(projs_h))

    if length < 0.20 or width < 0.08 or height < 0.08:
        return None

    start_pt = centroid + p_min * axis
    end_pt = centroid + p_max * axis

    return ReconstructedDuct(
        duct_id=duct_id,
        start_point_m=(float(start_pt[0]), float(start_pt[1]), float(start_pt[2])),
        end_point_m=(float(end_pt[0]), float(end_pt[1]), float(end_pt[2])),
        width_m=width,
        height_m=height,
        length_m=length,
        axis_vector=(float(axis[0]), float(axis[1]), float(axis[2])),
        source_point_indices=[int(i) for i in source_indices],
        confidence=0.85,
    )


def reconstruct_cable_tray_from_points(
    points: np.ndarray,
    source_indices: list[int] | np.ndarray,
    tray_id: str = "tray_001",
) -> ReconstructedCableTray | None:
    """Fit a cable tray channel to points."""
    N = len(points)
    if N < 15:
        return None

    centroid = np.mean(points, axis=0)
    centered = points - centroid
    cov = np.dot(centered.T, centered) / float(N)
    eigvals, eigvecs = np.linalg.eigh(cov)

    axis = eigvecs[:, 2]
    v_width = eigvecs[:, 1]
    v_depth = eigvecs[:, 0]

    projs_l = np.dot(centered, axis)
    p_min, p_max = float(np.min(projs_l)), float(np.max(projs_l))
    length = float(p_max - p_min)

    width = float(np.ptp(np.dot(centered, v_width)))
    depth = float(np.ptp(np.dot(centered, v_depth)))

    if length < 0.20 or width < 0.05:
        return None

    start_pt = centroid + p_min * axis
    end_pt = centroid + p_max * axis

    return ReconstructedCableTray(
        tray_id=tray_id,
        start_point_m=(float(start_pt[0]), float(start_pt[1]), float(start_pt[2])),
        end_point_m=(float(end_pt[0]), float(end_pt[1]), float(end_pt[2])),
        width_m=width,
        depth_m=depth,
        length_m=length,
        axis_vector=(float(axis[0]), float(axis[1]), float(axis[2])),
        source_point_indices=[int(i) for i in source_indices],
        confidence=0.82,
    )


def reconstruct_valve_from_points(
    points: np.ndarray,
    source_indices: list[int] | np.ndarray,
    valve_id: str = "valve_001",
    connected_pipe_id: str | None = None,
) -> ReconstructedValve | None:
    """Reconstruct a compact valve or inline fitting from points."""
    N = len(points)
    if N < 10:
        return None

    centroid = np.mean(points, axis=0)
    centered = points - centroid
    cov = np.dot(centered.T, centered) / float(N)
    eigvals, eigvecs = np.linalg.eigh(cov)

    dims = [float(np.ptp(np.dot(centered, eigvecs[:, i]))) for i in range(3)]
    dims_sorted = sorted(dims, reverse=True)

    return ReconstructedValve(
        valve_id=valve_id,
        center_m=(float(centroid[0]), float(centroid[1]), float(centroid[2])),
        length_m=dims_sorted[0],
        width_m=dims_sorted[1],
        height_m=dims_sorted[2],
        connected_pipe_id=connected_pipe_id,
        source_point_indices=[int(i) for i in source_indices],
        confidence=0.80 if connected_pipe_id else 0.65,
    )
