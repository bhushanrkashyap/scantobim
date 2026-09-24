"""Wall Opening Detection Module (Doors & Windows) — Scan-to-BIM Production Engine.

Adapted from Cloud2BIM openings/opening_detector.py.
Zero hardcoded geometry:
  - Projects points in wall corridor to 2D elevation plane (u, z)
  - Identifies void regions bounded by structural wall inliers
  - Associates openings with parent host wall
  - Classifies doors (sill <= 300mm) vs windows (sill > 300mm)
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import structlog

from agent.tools.geometry_tools import meters_to_mm, mm_to_meters

logger = structlog.get_logger()


@dataclass
class OpeningEntity:
    """Represents a door or window opening hosted in a parent wall."""

    element_id: str
    host_wall_id: str
    storey_id: str
    type: str  # "DOOR" or "WINDOW"
    center_xyz_m: tuple[float, float, float]
    width_m: float
    height_m: float
    sill_z_m: float
    confidence: float = 0.85

    def to_dict(self) -> dict[str, Any]:
        return {
            "element_id": self.element_id,
            "host_wall_id": self.host_wall_id,
            "storey_id": self.storey_id,
            "type": self.type,
            "center_xyz_m": [round(c, 4) for c in self.center_xyz_m],
            "width_m": round(self.width_m, 4),
            "height_m": round(self.height_m, 4),
            "sill_z_m": round(self.sill_z_m, 4),
            "center_xyz_mm": [round(meters_to_mm(c), 1) for c in self.center_xyz_m],
            "width_mm": round(meters_to_mm(self.width_m), 1),
            "height_mm": round(meters_to_mm(self.height_m), 1),
            "sill_z_mm": round(meters_to_mm(self.sill_z_m), 1),
            "confidence": round(self.confidence, 2),
        }


def detect_wall_openings(
    wall_id: str,
    storey_id: str,
    start_pt: np.ndarray,
    end_pt: np.ndarray,
    base_z: float,
    wall_height: float,
    wall_thickness: float,
    points_xyz: np.ndarray,
    grid_res_m: float = 0.10,
    min_opening_width_m: float = 0.70,
    max_opening_width_m: float = 3.0,
    min_opening_height_m: float = 0.90,
) -> list[OpeningEntity]:
    """Detect hosted door/window openings along a physical wall axis."""
    if len(points_xyz) < 50:
        return []

    # Wall coordinate frame
    p0 = start_pt[:2]
    p1 = end_pt[:2]
    wall_vec = p1 - p0
    wall_len = float(np.linalg.norm(wall_vec))
    if wall_len < min_opening_width_m + 0.4:
        return []

    u_axis = wall_vec / wall_len
    v_normal = np.array([-u_axis[1], u_axis[0]])

    # Distance to line segment
    pts_xy = points_xyz[:, :2]
    rel_pts = pts_xy - p0
    proj_u = np.dot(rel_pts, u_axis)
    proj_v = np.dot(rel_pts, v_normal)

    # Filter points inside wall corridor
    corridor_half = max(0.25, wall_thickness * 0.8)
    mask_corridor = (
        (proj_u >= 0.0)
        & (proj_u <= wall_len)
        & (np.abs(proj_v) <= corridor_half)
        & (points_xyz[:, 2] >= base_z)
        & (points_xyz[:, 2] <= base_z + wall_height)
    )

    inlier_pts = points_xyz[mask_corridor]
    if len(inlier_pts) < 100:
        return []

    u_vals = proj_u[mask_corridor]
    z_vals = inlier_pts[:, 2] - base_z

    # 2D occupancy grid along (u, z)
    n_bins_u = max(5, int(np.ceil(wall_len / grid_res_m)))
    n_bins_z = max(5, int(np.ceil(wall_height / grid_res_m)))

    hist, u_edges, z_edges = np.histogram2d(
        u_vals, z_vals, bins=[n_bins_u, n_bins_z]
    )

    # Voids are low-density cells surrounded by high-density wall points
    occupancy = (hist > 2).astype(np.uint8)

    # Look for vertical columns of empty cells (void columns)
    col_density = np.sum(occupancy, axis=1)  # sum over z
    mean_density = np.mean(col_density)
    void_cols = np.where(col_density < 0.35 * mean_density)[0]

    if len(void_cols) < int(min_opening_width_m / grid_res_m):
        return []

    # Group contiguous void columns into opening spans
    spans: list[list[int]] = []
    current_span: list[int] = [void_cols[0]]

    for c in void_cols[1:]:
        if c == current_span[-1] + 1:
            current_span.append(c)
        else:
            if len(current_span) >= int(min_opening_width_m / grid_res_m):
                spans.append(current_span)
            current_span = [c]
    if len(current_span) >= int(min_opening_width_m / grid_res_m):
        spans.append(current_span)

    openings: list[OpeningEntity] = []

    for span_idx, span in enumerate(spans):
        u_start = float(u_edges[span[0]])
        u_end = float(u_edges[span[-1] + 1])
        w = u_end - u_start

        if w < min_opening_width_m or w > max_opening_width_m:
            continue
        if u_start < 0.20 or (wall_len - u_end) < 0.20:
            # Skip edges of wall
            continue

        # Check vertical extents inside the void span
        span_occupancy = occupancy[span[0] : span[-1] + 1, :]
        row_density = np.sum(span_occupancy, axis=0)
        void_rows = np.where(row_density < len(span) * 0.35)[0]

        if len(void_rows) < int(min_opening_height_m / grid_res_m):
            continue

        z_bottom = float(z_edges[void_rows[0]])
        z_top = float(z_edges[void_rows[-1] + 1])
        h = z_top - z_bottom

        if h < min_opening_height_m:
            continue

        sill_z = base_z + z_bottom
        is_door = z_bottom <= 0.30
        op_type = "DOOR" if is_door else "WINDOW"

        u_mid = 0.5 * (u_start + u_end)
        mid_2d = p0 + u_axis * u_mid
        center_xyz = (float(mid_2d[0]), float(mid_2d[1]), float(sill_z + 0.5 * h))

        op_id = f"op_{wall_id}_{span_idx}"
        opening = OpeningEntity(
            element_id=op_id,
            host_wall_id=wall_id,
            storey_id=storey_id,
            type=op_type,
            center_xyz_m=center_xyz,
            width_m=w,
            height_m=h,
            sill_z_m=sill_z,
            confidence=0.88,
        )
        openings.append(opening)

    return openings
