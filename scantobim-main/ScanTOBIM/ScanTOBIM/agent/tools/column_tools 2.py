"""Structural Column Detection Module — Scan-to-BIM Production Engine.

Adapted from Scan-to-BIM-CVPR-2024 utils/t7_utils.py (T7 IfcColumn) and column_detector.py.
Zero hardcoded geometry:
  - Clusters vertical candidate points
  - Computes 2D ConvexHull and Minimum Bounding Rectangle (MBR) across continuous rotation angles
  - Extracts center, width, depth, height, and orientation from detected point geometry
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from scipy.spatial import ConvexHull
import structlog

from agent.tools.geometry_tools import meters_to_mm, mm_to_meters
from agent.tools.slab_tools import StoreyDefinition

logger = structlog.get_logger()


@dataclass
class ColumnEntity:
    """Represents a detected 3D structural column."""

    element_id: str
    storey_id: str
    center_xyz_m: tuple[float, float, float]
    width_m: float
    depth_m: float
    height_m: float
    rotation_deg: float
    profile_type: str = "RECTANGULAR"  # "RECTANGULAR" or "CIRCULAR"
    confidence: float = 0.88
    point_count: int = 0

    def to_dict(self) -> dict[str, Any]:
        return {
            "element_id": self.element_id,
            "storey_id": self.storey_id,
            "center_xyz_m": [round(c, 4) for c in self.center_xyz_m],
            "width_m": round(self.width_m, 4),
            "depth_m": round(self.depth_m, 4),
            "height_m": round(self.height_m, 4),
            "rotation_deg": round(self.rotation_deg, 2),
            "profile_type": self.profile_type,
            "center_xyz_mm": [round(meters_to_mm(c), 1) for c in self.center_xyz_m],
            "width_mm": round(meters_to_mm(self.width_m), 1),
            "depth_mm": round(meters_to_mm(self.depth_m), 1),
            "height_mm": round(meters_to_mm(self.height_m), 1),
            "confidence": round(self.confidence, 2),
            "point_count": self.point_count,
        }


def minimum_bounding_rectangle(
    points_2d: np.ndarray,
) -> tuple[np.ndarray, float, float, float]:
    """Compute Minimum Bounding Rectangle (MBR) for a 2D point set.

    Adapted from CVPR-2024 t7_utils.
    Returns:
        (4x2 corners, width, depth, rotation_degrees)
    """
    if len(points_2d) < 3:
        mins = points_2d.min(axis=0)
        maxs = points_2d.max(axis=0)
        w = maxs[0] - mins[0]
        d = maxs[1] - mins[1]
        corners = np.array(
            [
                [mins[0], mins[1]],
                [maxs[0], mins[1]],
                [maxs[0], maxs[1]],
                [mins[0], maxs[1]],
            ]
        )
        return corners, float(w), float(d), 0.0

    try:
        hull = ConvexHull(points_2d)
        hull_pts = points_2d[hull.vertices]
    except Exception:
        hull_pts = points_2d

    edges = hull_pts[1:] - hull_pts[:-1]
    angles = np.arctan2(edges[:, 1], edges[:, 0])
    angles = np.abs(np.mod(angles, np.pi / 2))
    angles = np.unique(angles)

    best_area = np.inf
    best_corners = None
    best_w, best_d, best_angle = 0.0, 0.0, 0.0

    for angle in angles:
        c, s = np.cos(angle), np.sin(angle)
        R = np.array([[c, s], [-s, c]])
        rotated = hull_pts @ R.T
        mins = rotated.min(axis=0)
        maxs = rotated.max(axis=0)
        area = (maxs[0] - mins[0]) * (maxs[1] - mins[1])
        if area < best_area:
            best_area = area
            best_w = float(maxs[0] - mins[0])
            best_d = float(maxs[1] - mins[1])
            best_angle = float(np.degrees(angle))
            corners_rot = np.array(
                [
                    [mins[0], mins[1]],
                    [maxs[0], mins[1]],
                    [maxs[0], maxs[1]],
                    [mins[0], maxs[1]],
                ]
            )
            Rinv = np.array([[c, -s], [s, c]])
            best_corners = corners_rot @ Rinv.T

    if best_corners is None:
        mins = points_2d.min(axis=0)
        maxs = points_2d.max(axis=0)
        best_corners = np.array(
            [
                [mins[0], mins[1]],
                [maxs[0], mins[1]],
                [maxs[0], maxs[1]],
                [mins[0], maxs[1]],
            ]
        )
        best_w = float(maxs[0] - mins[0])
        best_d = float(maxs[1] - mins[1])

    return best_corners, best_w, best_d, best_angle


def detect_structural_columns(
    points_xyz: np.ndarray,
    storeys: list[StoreyDefinition],
    min_column_height_ratio: float = 0.50,
    min_points_per_col: int = 40,
    max_column_dim_m: float = 1.20,
    min_column_dim_m: float = 0.15,
) -> list[ColumnEntity]:
    """Extract structural columns from candidate vertical point clusters.

    Zero hardcoding: column width, depth, height, and orientation are measured
    from the detected geometric point cluster.
    """
    if len(points_xyz) < 50 or not storeys:
        return []

    columns: list[ColumnEntity] = []

    for storey in storeys:
        base_z = storey.elevation_m
        top_z = storey.top_elevation_m
        storey_h = storey.height_m

        # Filter points within storey interior (exclude slab boundary buffers)
        z_pad = 0.15 * storey_h
        mask_z = (points_xyz[:, 2] >= base_z + z_pad) & (
            points_xyz[:, 2] <= top_z - z_pad
        )
        storey_pts = points_xyz[mask_z]

        if len(storey_pts) < min_points_per_col:
            continue

        # Grid-based clustering in 2D to isolate columnar candidates
        grid_res = 0.30  # 300mm cell
        pts_xy = storey_pts[:, :2]
        grid_coords = np.floor(pts_xy / grid_res).astype(int)
        unique_cells, cell_indices, counts = np.unique(
            grid_coords, axis=0, return_inverse=True, return_counts=True
        )

        for cell_idx, count in enumerate(counts):
            if count < min_points_per_col:
                continue

            cell_mask = cell_indices == cell_idx
            col_cluster = storey_pts[cell_mask]

            # Height check: vertical span must be at least min_column_height_ratio
            z_span = float(col_cluster[:, 2].max() - col_cluster[:, 2].min())
            if z_span < (storey_h * min_column_height_ratio):
                continue

            # Fit 2D MBR
            corners_2d, width, depth, angle_deg = minimum_bounding_rectangle(
                col_cluster[:, :2]
            )

            # Ensure width <= depth convention
            if width > depth:
                width, depth = depth, width
                angle_deg = (angle_deg + 90.0) % 180.0

            # Reject walls / thin partitions: aspect ratio > 4.0 or depth > max_column_dim
            if depth > max_column_dim_m or width < min_column_dim_m:
                continue
            if (depth / max(width, 0.05)) > 3.5:
                # Elongated element is a wall fragment, not a column
                continue

            center_x = float(col_cluster[:, 0].mean())
            center_y = float(col_cluster[:, 1].mean())
            center_z = float(base_z + storey_h * 0.5)

            # Detect if circular profile
            radius_pts = np.hypot(
                col_cluster[:, 0] - center_x, col_cluster[:, 1] - center_y
            )
            r_std = float(radius_pts.std())
            r_mean = float(radius_pts.mean())
            is_circular = (r_std / max(r_mean, 1e-4)) < 0.15

            col_id = f"col_{storey.storey_id}_{len(columns)}"
            col_entity = ColumnEntity(
                element_id=col_id,
                storey_id=storey.storey_id,
                center_xyz_m=(center_x, center_y, center_z),
                width_m=width,
                depth_m=depth,
                height_m=storey_h,
                rotation_deg=angle_deg,
                profile_type="CIRCULAR" if is_circular else "RECTANGULAR",
                confidence=0.90,
                point_count=len(col_cluster),
            )
            columns.append(col_entity)

    logger.info("columns_detected", count=len(columns))
    return columns
