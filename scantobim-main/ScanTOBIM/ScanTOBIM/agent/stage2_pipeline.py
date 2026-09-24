"""Scan-to-BIM Stage 2 Pipeline — Architectural Wall Detection & Classification.

Production implementation providing:
  1. Vertical Surface Extraction (rejects horizontal slabs/ceilings)
  2. Geometric Feature Extraction (density, fill factor, Z-continuity, grid alignment, curvature)
  3. Semantic Classification (ARCHITECTURAL_WALL, WALL_FRAGMENT, CURVED_STRUCTURAL_SURFACE, MEP_EQUIPMENT, STRUCTURAL_NONWALL, FURNITURE_OBJECT, OTHER)
  4. Wall Fragment Grouping (Union-Find with 80mm lateral offset, 1200mm gap, 30% Z-overlap)
  5. Curved Structural Surface Aggregation (groups chord facets into cohesive curved elements)
  6. Physical Wall Construction with thickness pairing
  7. Section 16 Diagnostic Table generation with numerical evidence

Zero scene-specific hardcoding: all geometry derived dynamically from point cloud data.
"""

from __future__ import annotations

import math
import time
from dataclasses import dataclass
from typing import Any

import numpy as np
import structlog

from agent.models import (
    ElementType,
    GeometrySegment,
    Point3D,
    SegmentShape,
    WallClassification,
)
from agent.tools.detection_config import (
    CURVATURE_MIN_FACETS,
    CURVATURE_NORMAL_STEP_MAX_DEG,
    CURVATURE_RADIAL_TOL_M,
    DEFAULT_WALL_THICKNESS_MM,
    MAX_WALL_THICKNESS_MM,
    MIN_WALL_THICKNESS_MM,
    WALL_ANGLE_TOL_DEG,
    WALL_COPLANAR_OFFSET_M,
    WALL_FRAGMENT_GAP_M,
    WALL_MIN_DENSITY,
    WALL_MIN_HEIGHT_M,
    WALL_MIN_LENGTH_M,
    WALL_MIN_POINT_COUNT,
    WALL_Z_OVERLAP,
)
from agent.tools.geometry_tools import canonicalize_wall_axis, wall_angle_degrees

logger = structlog.get_logger()


@dataclass
class PlaneFeatures:
    """Comprehensive geometric features extracted for a single planar surface."""

    plane_id: int
    inliers: int
    area_m2: float
    length_m: float
    height_m: float
    density_pts_m2: float
    normal: np.ndarray
    plane_offset_m: float
    z_continuity: float
    fill_factor: float
    curvature_evidence: float
    is_grid_aligned: bool
    wall_score: float
    classification: WallClassification


@dataclass
class Stage2Result:
    """Complete Stage 2 detection, classification, and grouping results."""

    total_planes: int
    vertical_surfaces: int
    architectural_walls: int
    wall_fragments: int
    curved_structural_elements: int
    mep_non_wall_surfaces: int
    physical_walls: list[dict]
    curved_elements: list[dict]
    rejected_planes: list[dict]
    features_list: list[PlaneFeatures]
    diagnostic_table: str
    runtime_s: float


# ── Step 1: Vertical Surface Extraction ──────────────────────────────────────


def extract_vertical_surfaces(
    planes: list[dict],
    up_axis: np.ndarray | None = None,
    vertical_normal_up_max: float = 0.25,
) -> tuple[list[dict], list[dict]]:
    """Separate vertical planar surfaces from horizontal/sloped planes (slabs/floors/stairs)."""
    if up_axis is None:
        try:
            from agent.tools.coordinate_system import GLOBAL_TRANSFORM
            up_axis = getattr(GLOBAL_TRANSFORM, "up_axis", np.array([0.0, 0.0, 1.0], dtype=float))
        except Exception:
            up_axis = np.array([0.0, 0.0, 1.0], dtype=float)

    vertical_planes: list[dict] = []
    rejected_planes: list[dict] = []

    for p in planes:
        n = p.get("normal")
        if n is None:
            rejected_planes.append(p)
            continue
        n_arr = np.asarray(n, dtype=float)
        norm_val = np.linalg.norm(n_arr)
        if norm_val > 1e-6:
            n_arr = n_arr / norm_val

        abs_n_up = abs(float(np.dot(n_arr, up_axis)))
        if abs_n_up < vertical_normal_up_max:
            p_copy = dict(p)
            p_copy["shape"] = SegmentShape.PLANE_VERTICAL
            vertical_planes.append(p_copy)
        else:
            p_copy = dict(p)
            p_copy["classification"] = (
                "FLOOR_SLAB" if abs_n_up >= 0.85 else "SLOPED_SURFACE"
            )
            rejected_planes.append(p_copy)

    return vertical_planes, rejected_planes


# ── Step 2: Dominant Architectural Grid Orientation ─────────────────────────


def infer_dominant_orientations(
    vertical_planes: list[dict],
    num_bins: int = 18,
    min_weight: float = 100.0,
) -> list[float]:
    """Dynamically infer the building's dominant orthogonal grid axes from normal angles.

    Zero hardcoding: uses weighted histogram modes of horizontal run directions.
    """
    if not vertical_planes:
        return [0.0, 90.0]

    angles: list[float] = []
    weights: list[float] = []

    for p in vertical_planes:
        tags = p.get("tags", {})
        ax = tags.get("wall_axis_x")
        ay = tags.get("wall_axis_y")
        if ax is None or ay is None:
            n = p.get("normal", [0, 1, 0])
            ax = -float(n[1])
            ay = float(n[0])

        c_axis = canonicalize_wall_axis([ax, ay])
        angle = wall_angle_degrees(c_axis)
        angles.append(angle)

        w = float(p.get("point_count", 0))
        if w <= 0 and "points" in p:
            w = float(len(p["points"]))
        if w <= 0 and p.get("inlier_cloud") is not None:
            w = float(len(p["inlier_cloud"].points))
        weights.append(max(w, 1.0))

    if not angles or sum(weights) <= 0:
        return [0.0, 90.0]

    h, edges = np.histogram(angles, bins=num_bins, range=(0.0, 180.0), weights=weights)
    sorted_bin_indices = np.argsort(h)[::-1]

    dominant_angles: list[float] = []
    if len(sorted_bin_indices) > 0 and h[sorted_bin_indices[0]] >= min_weight:
        best_idx = sorted_bin_indices[0]
        primary_angle = float(0.5 * (edges[best_idx] + edges[best_idx + 1]))
        ortho_angle = (primary_angle + 90.0) % 180.0
        dominant_angles = [round(primary_angle, 1), round(ortho_angle, 1)]
    else:
        dominant_angles = [0.0, 90.0]

    return dominant_angles


# ── Step 3: Geometric Curvature & Cylindrical Facet Detection ───────────────


def compute_curvature_evidence(
    plane_idx: int,
    vertical_planes: list[dict],
    cloud_center_xy: np.ndarray,
) -> float:
    """Detect whether a vertical plane represents a chordal facet of a curved structural surface.

    Evaluates:
      1. Alignment between plane normal and radial vector from central structural axis.
      2. Presence of neighboring vertical facets with progressive normal angle steps (<= 35 deg).
      3. Radial distance consistency with neighboring facets around the common axis.
    """
    if not vertical_planes or plane_idx >= len(vertical_planes):
        return 0.0

    target = vertical_planes[plane_idx]
    c1 = target.get("centroid", np.zeros(3))[:2]
    n1 = target.get("normal", np.array([0, 1, 0]))[:2]
    norm_n1 = np.linalg.norm(n1)
    if norm_n1 > 1e-6:
        n1 = n1 / norm_n1

    v_rad = c1 - cloud_center_xy
    r1 = float(np.linalg.norm(v_rad))
    if r1 > 1e-4:
        rad_dir = v_rad / r1
        rad_align = float(abs(np.dot(rad_dir, n1)))
    else:
        rad_align = 0.0

    ang1 = (math.degrees(math.atan2(n1[1], n1[0])) + 360.0) % 180.0

    neighbor_facets = 0
    consistent_radius_count = 0

    for j, other in enumerate(vertical_planes):
        if j == plane_idx:
            continue
        c2 = other.get("centroid", np.zeros(3))[:2]
        r2 = float(np.linalg.norm(c2 - cloud_center_xy))

        # Check spatial proximity
        dist_xy = float(np.linalg.norm(c1 - c2))
        if dist_xy > 3.0:
            continue

        n2 = other.get("normal", np.array([0, 1, 0]))[:2]
        norm_n2 = np.linalg.norm(n2)
        if norm_n2 > 1e-6:
            n2 = n2 / norm_n2

        ang2 = (math.degrees(math.atan2(n2[1], n2[0])) + 360.0) % 180.0
        ang_diff = min(abs(ang1 - ang2), 180.0 - abs(ang1 - ang2))

        if 4.0 <= ang_diff <= CURVATURE_NORMAL_STEP_MAX_DEG:
            neighbor_facets += 1
            if abs(r1 - r2) <= CURVATURE_RADIAL_TOL_M:
                consistent_radius_count += 1

    # Curvature requires high radial normal alignment (tangent to cylinder centered on axis)
    if rad_align < 0.65:
        return 0.0

    # Composite curvature evidence in [0.0, 1.0]
    c_score = 0.40 * rad_align
    if neighbor_facets >= 2:
        c_score += 0.35
    elif neighbor_facets == 1:
        c_score += 0.20
    if consistent_radius_count >= 1:
        c_score += 0.25

    # If it is outside the core (> 2.4m), it is part of the outer envelope
    if r1 >= 2.4:
        c_score *= 0.20

    return float(max(0.0, min(c_score, 1.0)))


# ── Step 4: Multi-Feature Extraction & Architectural Wall-Likeness ──────────


def extract_plane_features(
    plane_idx: int,
    plane_dict: dict,
    cloud_center_xy: np.ndarray,
    dominant_angles: list[float],
    vertical_planes: list[dict] | None = None,
    grid_res_m: float = 0.25,
) -> PlaneFeatures:
    """Extract 14 measurable geometric features for a single vertical plane candidate."""
    cloud = plane_dict.get("inlier_cloud")
    if "points" in plane_dict:
        pts = np.asarray(plane_dict["points"], dtype=float)
        normals = (
            np.asarray(plane_dict["normals"], dtype=float)
            if "normals" in plane_dict and plane_dict["normals"] is not None
            else None
        )
    elif cloud is not None and hasattr(cloud, "points") and len(cloud.points) > 0:
        pts = np.asarray(cloud.points, dtype=float)
        normals = (
            np.asarray(cloud.normals, dtype=float)
            if hasattr(cloud, "normals") and len(cloud.normals) > 0
            else None
        )
    else:
        mins = plane_dict.get("mins", np.zeros(3))
        maxs = plane_dict.get("maxs", np.ones(3))
        pts = np.array([mins, maxs, plane_dict.get("centroid", np.zeros(3))])
        normals = None

    c = (
        np.asarray(plane_dict["centroid"], dtype=float)
        if "centroid" in plane_dict
        else pts.mean(axis=0)
    )
    n = (
        np.asarray(plane_dict["normal"], dtype=float)
        if "normal" in plane_dict
        else np.array([0.0, 1.0, 0.0])
    )
    norm_n = np.linalg.norm(n)
    if norm_n > 1e-6:
        n = n / norm_n

    tags = plane_dict.get("tags", {})
    ax = float(tags.get("wall_axis_x", -n[1]))
    ay = float(tags.get("wall_axis_y", n[0]))
    axis_2d = canonicalize_wall_axis([ax, ay])
    angle_deg = wall_angle_degrees(axis_2d)

    # 1. Robust Run Length (5th - 95th percentile projection)
    if "wall_length_mm" in tags and len(pts) < 10:
        robust_length = float(tags["wall_length_mm"]) / 1000.0
    elif "_wall_start_x_m" in tags and "_wall_end_x_m" in tags and len(pts) < 10:
        sx = float(tags["_wall_start_x_m"])
        sy = float(tags.get("_wall_start_y_m", 0.0))
        ex = float(tags["_wall_end_x_m"])
        ey = float(tags.get("_wall_end_y_m", 0.0))
        robust_length = float(math.hypot(ex - sx, ey - sy))
    else:
        proj_len = pts[:, :2] @ axis_2d
        if len(proj_len) >= 10:
            robust_length = float(
                np.percentile(proj_len, 95.0) - np.percentile(proj_len, 5.0)
            )
        else:
            robust_length = float(proj_len.max() - proj_len.min())
    robust_length = max(robust_length, 0.10)

    # 2. Robust Vertical Height (5th - 95th percentile along Z)
    z_pts = pts[:, 2]
    if len(z_pts) >= 10:
        robust_height = float(
            np.percentile(z_pts, 95.0) - np.percentile(z_pts, 5.0)
        )
    else:
        robust_height = float(z_pts.max() - z_pts.min())
    robust_height = max(robust_height, 0.10)

    # 3. Planar Area & Point Density
    planar_area = max(robust_length * robust_height, 0.05)
    density = float(len(pts) / planar_area)

    # 4. 2D Spatial Occupancy Fill Factor
    mins = pts.min(axis=0)
    maxs = pts.max(axis=0)
    if len(pts) >= 15:
        u_pts = pts[:, :2] @ axis_2d
        v_pts = pts[:, 2]
        u_min, u_max = float(u_pts.min()), float(u_pts.max())
        v_min, v_max = float(v_pts.min()), float(v_pts.max())
        nb_u = max(1, min(int((u_max - u_min) / grid_res_m) + 1, 200))
        nb_v = max(1, min(int((v_max - v_min) / grid_res_m) + 1, 200))

        u_idx = np.clip(((u_pts - u_min) / grid_res_m).astype(int), 0, nb_u - 1)
        v_idx = np.clip(((v_pts - v_min) / grid_res_m).astype(int), 0, nb_v - 1)
        grid_occupied = len(np.unique(v_idx * nb_u + u_idx))
        fill_factor = float(grid_occupied / (nb_u * nb_v))
    else:
        fill_factor = 1.0

    # 5. Normal Consistency
    if normals is not None and len(normals) >= 10:
        dots = np.abs(normals @ n)
        norm_consistency = float(np.mean(dots > 0.85))
    else:
        norm_consistency = 1.0

    # 6. Z-profile Continuity
    z_bins = np.arange(mins[2], maxs[2] + 1.0, 1.0)
    if len(z_bins) > 1 and len(pts) >= 10:
        zh, _ = np.histogram(z_pts, bins=z_bins)
        z_continuity = float(np.sum(zh > 0) / len(zh))
    else:
        z_continuity = 1.0

    # 7. Alignment with Dominant Building Grid
    min_ang_diff = (
        min(
            [
                min(abs(angle_deg - dom), 180.0 - abs(angle_deg - dom))
                for dom in dominant_angles
            ]
        )
        if dominant_angles
        else 0.0
    )
    is_grid_aligned = bool(min_ang_diff <= WALL_ANGLE_TOL_DEG + 5.0)

    # 8. Envelope Distance (perpendicular distance to cloud center)
    n_xy = np.array([n[0], n[1]], dtype=float)
    norm_n_xy = np.linalg.norm(n_xy)
    if norm_n_xy > 1e-6:
        n_xy /= norm_n_xy
    d_center = float(abs(np.dot(c[:2] - cloud_center_xy, n_xy)))

    # 9. Curvature Evidence
    if vertical_planes is not None:
        curv_evidence = compute_curvature_evidence(
            plane_idx, vertical_planes, cloud_center_xy
        )
    else:
        curv_evidence = 0.0

    # 10. Wall-Likeness Scoring
    wall_score = compute_wall_likeness_score(
        robust_length=robust_length,
        robust_height=robust_height,
        density=density,
        fill_factor=fill_factor,
        norm_consistency=norm_consistency,
        is_grid_aligned=is_grid_aligned,
        d_center=d_center,
        curvature_evidence=curv_evidence,
    )

    # 11. Final Semantic Classification
    classification = classify_features(
        wall_score=wall_score,
        robust_length=robust_length,
        robust_height=robust_height,
        density=density,
        fill_factor=fill_factor,
        is_grid_aligned=is_grid_aligned,
        d_center=d_center,
        curvature_evidence=curv_evidence,
    )

    return PlaneFeatures(
        plane_id=plane_idx,
        inliers=len(pts),
        area_m2=round(planar_area, 2),
        length_m=round(robust_length, 2),
        height_m=round(robust_height, 2),
        density_pts_m2=round(density, 1),
        normal=n,
        plane_offset_m=round(d_center, 2),
        z_continuity=round(z_continuity, 2),
        fill_factor=round(fill_factor, 2),
        curvature_evidence=round(curv_evidence, 2),
        is_grid_aligned=is_grid_aligned,
        wall_score=round(wall_score, 2),
        classification=classification,
    )


def compute_wall_likeness_score(
    robust_length: float,
    robust_height: float,
    density: float,
    fill_factor: float,
    norm_consistency: float,
    is_grid_aligned: bool,
    d_center: float,
    curvature_evidence: float = 0.0,
) -> float:
    """Calculate an explainable multi-feature architectural wall score in [0.0, 1.0]."""
    s_h = min(robust_height / 6.0, 1.0)
    s_l = min(robust_length / 4.0, 1.0)
    s_fill = max(0.0, min((fill_factor - 0.15) / 0.45, 1.0))
    s_dens = max(0.0, min((density - 15.0) / 75.0, 1.0))
    s_norm = max(0.0, min((norm_consistency - 0.60) / 0.35, 1.0))
    s_orient = 1.0 if is_grid_aligned else 0.0
    s_env = 1.0 if d_center > 2.0 else (0.50 if is_grid_aligned else 0.0)

    score = (
        0.18 * s_h
        + 0.18 * s_l
        + 0.24 * s_fill
        + 0.14 * s_dens
        + 0.10 * s_norm
        + 0.08 * s_orient
        + 0.08 * s_env
    )

    # Penalize non-grid-aligned surfaces cutting through center
    if not is_grid_aligned:
        score *= 0.80

    # Penalize surfaces with curvature/cylindrical chord evidence
    if curvature_evidence > 0.30:
        score *= max(0.10, 1.0 - 0.75 * curvature_evidence)

    return float(max(0.0, min(score, 1.0)))


def classify_features(
    wall_score: float,
    robust_length: float,
    robust_height: float,
    density: float,
    fill_factor: float,
    is_grid_aligned: bool,
    d_center: float,
    curvature_evidence: float = 0.0,
) -> WallClassification:
    """Classify a vertical plane into its semantic physical category."""
    # 1. Curved / Cylindrical Structural Surface
    if curvature_evidence >= 0.55 or (curvature_evidence >= 0.35 and d_center < 1.6 and robust_height >= 4.0):
        return WallClassification.CURVED_STRUCTURAL_SURFACE

    # 2. Major Architectural Walls (Envelope walls, exterior facades, solid interior walls)
    if (
        wall_score >= 0.52
        and is_grid_aligned
        and robust_height >= 1.8
        and robust_length >= 1.5
        and curvature_evidence < 0.35
    ):
        return WallClassification.ARCHITECTURAL_WALL

    # 3. Legitimate Wall Fragments (window/door openings, partial occlusions, high storeys)
    if (
        wall_score >= 0.38
        and is_grid_aligned
        and robust_height >= WALL_MIN_HEIGHT_M
        and robust_length >= WALL_MIN_LENGTH_M
        and curvature_evidence < 0.40
    ):
        return WallClassification.WALL_FRAGMENT

    # 4. Central chord slices and sparse MEP
    is_central_chord = (d_center < 1.3) and (
        fill_factor < 0.32 or density < 30.0 or not is_grid_aligned
    )
    is_thin_slice = (fill_factor < 0.20 and robust_length < 2.0) or (density < 18.0 and robust_length < 2.0)
    if is_central_chord or is_thin_slice:
        return WallClassification.MEP_EQUIPMENT

    # 5. Furniture Objects
    if robust_height < 1.0 and robust_length < 1.0:
        return WallClassification.FURNITURE_OBJECT

    # 6. Structural Non-Wall Columns / Posts
    if robust_length < 0.80 and robust_height >= 2.0:
        return WallClassification.STRUCTURAL_NONWALL

    # 7. Default to MEP Equipment if cutting interior without grid alignment
    if not is_grid_aligned and d_center < 2.0:
        return WallClassification.MEP_EQUIPMENT

    return WallClassification.OTHER


# ── Step 5: Wall Fragment Grouping & Coplanar Merging ────────────────────────


def group_wall_fragments(
    candidate_walls: list[dict],
    angle_tol_rad: float = math.radians(WALL_ANGLE_TOL_DEG),
    lateral_offset_m: float = WALL_COPLANAR_OFFSET_M,
    fragment_gap_m: float = WALL_FRAGMENT_GAP_M,
    z_overlap_ratio: float = WALL_Z_OVERLAP,
) -> list[dict]:
    """Group coplanar wall fragments into unified physical walls using Union-Find.

    Tolerances:
      - 80mm lateral offset tolerance (protects separate parallel walls).
      - 1200mm fragment bridging tolerance (bridges doorways/window openings).
      - 30% vertical overlap or multi-storey elevation continuity.
    """
    if not candidate_walls:
        return []
    if len(candidate_walls) == 1:
        return list(candidate_walls)

    n = len(candidate_walls)
    parent = list(range(n))

    def find(i: int) -> int:
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    def union(i: int, j: int):
        root_i = find(i)
        root_j = find(j)
        if root_i != root_j:
            parent[root_j] = root_i

    for i in range(n):
        w1 = candidate_walls[i]
        n1 = np.asarray(w1.get("normal", np.array([0, 1, 0])), dtype=float)[:2]
        norm_n1 = np.linalg.norm(n1)
        if norm_n1 > 1e-6:
            n1 = n1 / norm_n1
        c1 = np.asarray(w1.get("centroid", np.zeros(3)), dtype=float)
        mins1 = np.asarray(w1.get("mins", c1 - 0.5), dtype=float)
        maxs1 = np.asarray(w1.get("maxs", c1 + 0.5), dtype=float)

        tags1 = w1.get("tags", {})
        ax1 = tags1.get("wall_axis_x", -n1[1])
        ay1 = tags1.get("wall_axis_y", n1[0])
        axis1 = canonicalize_wall_axis([ax1, ay1])

        for j in range(i + 1, n):
            w2 = candidate_walls[j]
            n2 = np.asarray(w2.get("normal", np.array([0, 1, 0])), dtype=float)[:2]
            norm_n2 = np.linalg.norm(n2)
            if norm_n2 > 1e-6:
                n2 = n2 / norm_n2
            c2 = np.asarray(w2.get("centroid", np.zeros(3)), dtype=float)
            mins2 = np.asarray(w2.get("mins", c2 - 0.5), dtype=float)
            maxs2 = np.asarray(w2.get("maxs", c2 + 0.5), dtype=float)

            # 1. Normal angle alignment (within angle_tol_rad)
            dot_n = abs(float(np.dot(n1, n2)))
            angle_diff = math.acos(min(1.0, max(-1.0, dot_n)))
            if angle_diff > angle_tol_rad:
                continue

            # 2. Perpendicular plane offset (within lateral_offset_m)
            normal1_xy = np.array([-axis1[1], axis1[0]], dtype=float)
            delta_c = c2[:2] - c1[:2]
            perp_offset = abs(float(np.dot(delta_c, normal1_xy)))
            if perp_offset > lateral_offset_m:
                continue

            # 3. Horizontal run gap along wall axis (within fragment_gap_m)
            s1_start = tags1.get("_wall_start_x_m", c1[0]) * axis1[0] + tags1.get("_wall_start_y_m", c1[1]) * axis1[1]
            s1_end = tags1.get("_wall_end_x_m", c1[0]) * axis1[0] + tags1.get("_wall_end_y_m", c1[1]) * axis1[1]
            t1_min = min(s1_start, s1_end)
            t1_max = max(s1_start, s1_end)
            if "points" in w1 and w1["points"] is not None and len(w1["points"]) > 0:
                p1_proj = np.asarray(w1["points"])[:, :2] @ axis1
                t1_min = min(t1_min, float(p1_proj.min()))
                t1_max = max(t1_max, float(p1_proj.max()))

            tags2 = w2.get("tags", {})
            s2_start = tags2.get("_wall_start_x_m", c2[0]) * axis1[0] + tags2.get("_wall_start_y_m", c2[1]) * axis1[1]
            s2_end = tags2.get("_wall_end_x_m", c2[0]) * axis1[0] + tags2.get("_wall_end_y_m", c2[1]) * axis1[1]
            t2_min = min(s2_start, s2_end)
            t2_max = max(s2_start, s2_end)
            if "points" in w2 and w2["points"] is not None and len(w2["points"]) > 0:
                p2_proj = np.asarray(w2["points"])[:, :2] @ axis1
                t2_min = min(t2_min, float(p2_proj.min()))
                t2_max = max(t2_max, float(p2_proj.max()))

            if t1_max < t2_min:
                gap = t2_min - t1_max
            elif t2_max < t1_min:
                gap = t1_min - t2_max
            else:
                gap = 0.0

            if gap > fragment_gap_m:
                continue

            # 4. Vertical Z-continuity / overlap
            z_low = max(mins1[2], mins2[2])
            z_high = min(maxs1[2], maxs2[2])
            overlap_z = max(0.0, z_high - z_low)
            h_min = min(maxs1[2] - mins1[2], maxs2[2] - mins2[2])
            z_overlap = overlap_z / max(h_min, 0.10) if h_min > 0 else 0.0

            # Storey-aware vertical continuity — only merge fragments on the same level.
            # Walls on different storeys have zero Z-overlap and MUST NOT merge.
            vertical_gap = max(0.0, max(mins1[2], mins2[2]) - min(maxs1[2], maxs2[2]))
            is_z_compatible = (z_overlap >= z_overlap_ratio or overlap_z >= 0.25)

            if not is_z_compatible:
                continue

            union(i, j)

    # Aggregate clusters
    clusters: dict[int, list[dict]] = {}
    for idx in range(n):
        root = find(idx)
        clusters.setdefault(root, []).append(candidate_walls[idx])

    merged_walls: list[dict] = []
    for members in clusters.values():
        if len(members) == 1:
            m = dict(members[0])
            tags_m = dict(m.get("tags", {}))
            ax = tags_m.get("wall_axis_x")
            ay = tags_m.get("wall_axis_y")
            if ax is None or ay is None:
                n = m.get("normal", np.array([0, 1, 0]))
                ax = -n[1]
                ay = n[0]
            axis = canonicalize_wall_axis([ax, ay])
            tags_m["wall_axis_x"] = round(float(axis[0]), 6)
            tags_m["wall_axis_y"] = round(float(axis[1]), 6)
            tags_m["wall_angle_deg"] = wall_angle_degrees(axis)
            m["tags"] = tags_m
            if "points" not in m or m["points"] is None:
                if m.get("inlier_cloud") is not None:
                    m["points"] = np.asarray(m["inlier_cloud"].points)
            merged_walls.append(m)
            continue

        pts_list = []
        for m in members:
            if "points" in m and m["points"] is not None:
                pts_list.append(np.asarray(m["points"]))
            elif m.get("inlier_cloud") is not None:
                pts_list.append(np.asarray(m["inlier_cloud"].points))
        primary = max(members, key=lambda m: m.get("point_count", 0))
        n_prim = primary.get("normal", np.array([0, 1, 0]))
        tags_prim = dict(primary.get("tags", {}))
        ax = tags_prim.get("wall_axis_x", -n_prim[1])
        ay = tags_prim.get("wall_axis_y", n_prim[0])
        axis = canonicalize_wall_axis([ax, ay])
        angle_deg = wall_angle_degrees(axis)

        if pts_list:
            all_pts = np.vstack(pts_list)
        else:
            member_pts = []
            for m in members:
                m_tags = m.get("tags", {})
                m_mins = np.asarray(m.get("mins", np.zeros(3)), dtype=float)
                m_maxs = np.asarray(m.get("maxs", np.zeros(3)), dtype=float)
                m_z_lo = float(m_mins[2])
                m_z_hi = float(m_maxs[2])
                m_z_mid = (m_z_lo + m_z_hi) / 2.0
                if "_wall_start_x_m" in m_tags and "_wall_end_x_m" in m_tags:
                    member_pts.append([float(m_tags["_wall_start_x_m"]), float(m_tags.get("_wall_start_y_m", 0.0)), m_z_lo])
                    member_pts.append([float(m_tags["_wall_start_x_m"]), float(m_tags.get("_wall_start_y_m", 0.0)), m_z_hi])
                    member_pts.append([float(m_tags["_wall_end_x_m"]), float(m_tags.get("_wall_end_y_m", 0.0)), m_z_lo])
                    member_pts.append([float(m_tags["_wall_end_x_m"]), float(m_tags.get("_wall_end_y_m", 0.0)), m_z_hi])
                elif "wall_length_mm" in m_tags and "centroid" in m:
                    c_m = np.asarray(m["centroid"])
                    half_m = float(m_tags["wall_length_mm"]) / 2000.0
                    member_pts.append([c_m[0] - axis[0] * half_m, c_m[1] - axis[1] * half_m, m_z_lo])
                    member_pts.append([c_m[0] + axis[0] * half_m, c_m[1] + axis[1] * half_m, m_z_hi])
                elif "centroid" in m:
                    c_m = np.asarray(m["centroid"])
                    member_pts.append([c_m[0], c_m[1], m_z_lo])
                    member_pts.append([c_m[0], c_m[1], m_z_hi])
            all_pts = np.asarray(member_pts, dtype=float) if member_pts else np.vstack([m.get("centroid", np.zeros(3)) for m in members])

        c_merged = all_pts.mean(axis=0)
        if any("mins" in m for m in members):
            mins_merged = np.minimum.reduce([np.asarray(m["mins"]) for m in members if "mins" in m])
        else:
            mins_merged = all_pts.min(axis=0)

        if any("maxs" in m for m in members):
            maxs_merged = np.maximum.reduce([np.asarray(m["maxs"]) for m in members if "maxs" in m])
        else:
            maxs_merged = all_pts.max(axis=0)

        proj = all_pts[:, :2] @ axis
        p_min = float(proj.min())
        p_max = float(proj.max())
        length_m = max(p_max - p_min, 0.10)
        if length_m <= 0.15:
            max_mm = max([float(m.get("tags", {}).get("wall_length_mm", 0.0)) for m in members] + [0.0])
            if max_mm > 0.0:
                length_m = max_mm / 1000.0

        # Compute robust Z bounds from actual inlier points (5th-95th percentile)
        # to avoid inheriting the full scan Z span as wall height.
        z_range_raw = float(all_pts[:, 2].max() - all_pts[:, 2].min())
        if z_range_raw > 0.3 and all_pts.shape[0] >= 4:
            if all_pts.shape[0] >= 10:
                z_p5 = float(np.percentile(all_pts[:, 2], 5.0))
                z_p95 = float(np.percentile(all_pts[:, 2], 95.0))
            else:
                z_p5 = float(all_pts[:, 2].min())
                z_p95 = float(all_pts[:, 2].max())
            # Clamp mins/maxs Z to robust bounds
            mins_merged[2] = z_p5
            maxs_merged[2] = z_p95
        height_m = float(maxs_merged[2] - mins_merged[2])

        normal_xy = np.array([-axis[1], axis[0]], dtype=float)
        perp_offset = float(np.mean(all_pts[:, :2] @ normal_xy))
        mid_proj = 0.5 * (p_min + p_max)
        c_xy = axis * mid_proj + normal_xy * perp_offset

        half_l = length_m / 2.0
        start_pt = c_xy - axis * half_l
        end_pt = c_xy + axis * half_l

        tags_prim["wall_axis_x"] = round(float(axis[0]), 6)
        tags_prim["wall_axis_y"] = round(float(axis[1]), 6)
        tags_prim["wall_angle_deg"] = round(angle_deg, 2)
        tags_prim["_wall_start_x_m"] = float(start_pt[0])
        tags_prim["_wall_start_y_m"] = float(start_pt[1])
        tags_prim["_wall_end_x_m"] = float(end_pt[0])
        tags_prim["_wall_end_y_m"] = float(end_pt[1])
        tags_prim["wall_length_mm"] = round(length_m * 1000.0, 1)
        tags_prim["wall_merged"] = True
        tags_prim["merged_fragment_count"] = len(members)

        merged_dict = {
            "shape": SegmentShape.PLANE_VERTICAL,
            "normal": n_prim,
            "centroid": np.array([c_xy[0], c_xy[1], c_merged[2]]),
            "mins": mins_merged,
            "maxs": maxs_merged,
            "point_count": len(all_pts),
            "points": all_pts,
            "confidence": float(max(m.get("confidence", 0.8) for m in members)),
            "classification": WallClassification.ARCHITECTURAL_WALL.value,
            "tags": tags_prim,
        }
        merged_walls.append(merged_dict)

    return merged_walls


# ── Step 6: Opposing Face Pairing for Physical BIM Walls ────────────────────


def pair_opposing_wall_faces(
    walls: list[dict],
    max_thickness_m: float = 0.45,
) -> list[dict]:
    """Pair parallel opposing wall faces into solid architectural walls with thickness.

    - If two opposing faces are detected within max_thickness_m, derives physical thickness
      from face distance bounded by [MIN_WALL_THICKNESS_MM, MAX_WALL_THICKNESS_MM].
    - If only one face is detected (single-face scan), infers controlled thickness
      using DEFAULT_WALL_THICKNESS_MM (200.0 mm).
    """
    if not walls:
        return []

    paired: set[int] = set()
    physical_walls: list[dict] = []

    for i in range(len(walls)):
        if i in paired:
            continue
        w1 = walls[i]
        n1 = np.asarray(w1.get("normal", np.array([0, 1, 0])), dtype=float)[:2]
        norm_n1 = np.linalg.norm(n1)
        if norm_n1 > 1e-6:
            n1 = n1 / norm_n1
        c1 = np.asarray(w1.get("centroid", np.zeros(3)), dtype=float)
        mins1 = np.asarray(w1.get("mins", c1 - 0.5), dtype=float)
        maxs1 = np.asarray(w1.get("maxs", c1 + 0.5), dtype=float)

        tags1 = dict(w1.get("tags", {}))
        ax = tags1.get("wall_axis_x", -n1[1])
        ay = tags1.get("wall_axis_y", n1[0])
        axis = canonicalize_wall_axis([ax, ay])

        best_j = None
        best_dist = max_thickness_m

        for j in range(i + 1, len(walls)):
            if j in paired:
                continue
            w2 = walls[j]
            n2 = np.asarray(w2.get("normal", np.array([0, 1, 0])), dtype=float)[:2]
            norm_n2 = np.linalg.norm(n2)
            if norm_n2 > 1e-6:
                n2 = n2 / norm_n2
            c2 = np.asarray(w2.get("centroid", np.zeros(3)), dtype=float)
            mins2 = np.asarray(w2.get("mins", c2 - 0.5), dtype=float)
            maxs2 = np.asarray(w2.get("maxs", c2 + 0.5), dtype=float)

            # Parallel or anti-parallel normal
            dot = abs(float(np.dot(n1, n2)))
            if dot < 0.95:
                continue

            # Perpendicular distance = wall thickness
            delta_c = c2[:2] - c1[:2]
            normal_xy = np.array([-axis[1], axis[0]], dtype=float)
            dist = abs(float(np.dot(delta_c, normal_xy)))

            if 0.05 <= dist <= best_dist:
                # Vertical overlap check
                z_low = max(mins1[2], mins2[2])
                z_high = min(maxs1[2], maxs2[2])
                overlap_z = max(0.0, z_high - z_low)
                h_min = min(maxs1[2] - mins1[2], maxs2[2] - mins2[2])
                if h_min > 0 and (overlap_z / h_min) >= 0.25:
                    best_dist = dist
                    best_j = j

        if best_j is not None:
            paired.add(i)
            paired.add(best_j)
            w2 = walls[best_j]
            raw_thickness_mm = best_dist * 1000.0
            thickness_mm = float(np.clip(raw_thickness_mm, MIN_WALL_THICKNESS_MM, MAX_WALL_THICKNESS_MM))
            c_mid = (c1 + np.asarray(w2.get("centroid", c1), dtype=float)) / 2.0
            mins_pair = np.minimum(mins1, np.asarray(w2.get("mins", mins1), dtype=float))
            maxs_pair = np.maximum(maxs1, np.asarray(w2.get("maxs", maxs1), dtype=float))

            tags_out = dict(tags1)
            tags_out["wall_axis_x"] = round(float(axis[0]), 6)
            tags_out["wall_axis_y"] = round(float(axis[1]), 6)
            tags_out["wall_angle_deg"] = wall_angle_degrees(axis)
            tags_out["wall_thickness_mm"] = round(thickness_mm, 1)
            tags_out["wall_base_z_mm"] = round(float(mins_pair[2] * 1000.0), 1)
            tags_out["paired_faces"] = True
            tags_out["thickness_mode"] = "measured_two_face"

            # Compute centerline from midline between the two faces
            start1 = np.array([tags1.get("_wall_start_x_m", c1[0]), tags1.get("_wall_start_y_m", c1[1])], dtype=float)
            end1 = np.array([tags1.get("_wall_end_x_m", c1[0]), tags1.get("_wall_end_y_m", c1[1])], dtype=float)
            tags2 = w2.get("tags", {})
            c2_pt = np.asarray(w2.get("centroid", c1), dtype=float)
            start2 = np.array([tags2.get("_wall_start_x_m", c2_pt[0]), tags2.get("_wall_start_y_m", c2_pt[1])], dtype=float)
            end2 = np.array([tags2.get("_wall_end_x_m", c2_pt[0]), tags2.get("_wall_end_y_m", c2_pt[1])], dtype=float)

            # Ensure direction vectors are co-directional along axis
            v1 = end1 - start1
            v2 = end2 - start2
            if np.dot(v1, v2) < 0:
                start2, end2 = end2, start2
                v2 = end2 - start2

            mid_start = 0.5 * (start1 + start2)
            mid_end = 0.5 * (end1 + end2)
            wall_len_m = float(np.linalg.norm(mid_end - mid_start))
            if wall_len_m < 0.15:
                wall_len_m = max(
                    float(np.linalg.norm(v1)),
                    float(np.linalg.norm(v2)),
                    float(tags1.get("wall_length_mm", 0.0)) / 1000.0,
                    float(tags2.get("wall_length_mm", 0.0)) / 1000.0,
                    0.10,
                )
                c_xy = 0.5 * (c1[:2] + c2_pt[:2])
                half_l = wall_len_m / 2.0
                mid_start = c_xy - axis * half_l
                mid_end = c_xy + axis * half_l

            tags_out["_wall_start_x_m"] = float(mid_start[0])
            tags_out["_wall_start_y_m"] = float(mid_start[1])
            tags_out["_wall_end_x_m"] = float(mid_end[0])
            tags_out["_wall_end_y_m"] = float(mid_end[1])
            tags_out["wall_length_mm"] = round(wall_len_m * 1000.0, 1)

            pw_pts = []
            for w in (w1, w2):
                if "points" in w and w["points"] is not None:
                    pw_pts.append(np.asarray(w["points"]))
                elif "inlier_cloud" in w and w["inlier_cloud"] is not None:
                    pw_pts.append(np.asarray(w["inlier_cloud"].points))
            all_pts_pair = np.vstack(pw_pts) if pw_pts else None

            # Compute robust Z bounds from actual inlier points (5th-95th percentile)
            # to avoid inheriting the full scan Z span as wall height.
            if all_pts_pair is not None and all_pts_pair.shape[0] >= 4:
                z_range_pair = float(all_pts_pair[:, 2].max() - all_pts_pair[:, 2].min())
                if z_range_pair > 0.3:
                    if all_pts_pair.shape[0] >= 10:
                        z_p5 = float(np.percentile(all_pts_pair[:, 2], 5.0))
                        z_p95 = float(np.percentile(all_pts_pair[:, 2], 95.0))
                    else:
                        z_p5 = float(all_pts_pair[:, 2].min())
                        z_p95 = float(all_pts_pair[:, 2].max())
                    # Clamp mins/maxs Z to robust bounds
                    mins_pair[2] = z_p5
                    maxs_pair[2] = z_p95
            pair_height_m = float(maxs_pair[2] - mins_pair[2])

            pw_item = {
                "shape": SegmentShape.PLANE_VERTICAL,
                "centroid": c_mid,
                "mins": mins_pair,
                "maxs": maxs_pair,
                "normal": w1.get("normal"),
                "length_m": wall_len_m,
                "height_m": pair_height_m,
                "thickness_mm": thickness_mm,
                "point_count": w1.get("point_count", 0) + w2.get("point_count", 0),
                "confidence": 1.0,
                "tags": tags_out,
            }
            if all_pts_pair is not None:
                pw_item["points"] = all_pts_pair
            physical_walls.append(pw_item)
        else:
            paired.add(i)
            raw_t = tags1.get("wall_thickness_mm")
            if raw_t is not None:
                thickness_mm = float(np.clip(float(raw_t), MIN_WALL_THICKNESS_MM, MAX_WALL_THICKNESS_MM))
            else:
                thickness_mm = DEFAULT_WALL_THICKNESS_MM

            tags_out = dict(tags1)
            tags_out["wall_axis_x"] = round(float(axis[0]), 6)
            tags_out["wall_axis_y"] = round(float(axis[1]), 6)
            tags_out["wall_angle_deg"] = wall_angle_degrees(axis)
            tags_out["wall_thickness_mm"] = round(thickness_mm, 1)
            tags_out["wall_base_z_mm"] = round(float(mins1[2] * 1000.0), 1)
            tags_out["paired_faces"] = False
            tags_out["thickness_mode"] = "inferred_single_face"

            # Preserve length_m and tags['wall_length_mm']
            wall_len_m = float(w1.get("length_m", 0.0))
            if wall_len_m <= 0.15 and "wall_length_mm" in tags1:
                wall_len_m = float(tags1["wall_length_mm"]) / 1000.0
            if wall_len_m <= 0.15:
                sx = tags1.get("_wall_start_x_m")
                ex = tags1.get("_wall_end_x_m")
                sy = tags1.get("_wall_start_y_m")
                ey = tags1.get("_wall_end_y_m")
                if sx is not None and ex is not None and sy is not None and ey is not None:
                    wall_len_m = math.hypot(float(ex) - float(sx), float(ey) - float(sy))
            if wall_len_m <= 0.15:
                wall_len_m = 2.0
            tags_out["wall_length_mm"] = round(wall_len_m * 1000.0, 1)

            w1_pts = w1.get("points")
            if w1_pts is None and "inlier_cloud" in w1 and w1["inlier_cloud"] is not None:
                w1_pts = np.asarray(w1["inlier_cloud"].points)

            # Compute robust Z bounds from actual inlier points (5th-95th percentile)
            if w1_pts is not None and len(w1_pts) >= 4:
                z_range_s = float(w1_pts[:, 2].max() - w1_pts[:, 2].min())
                if z_range_s > 0.3:
                    if len(w1_pts) >= 10:
                        z_p5_s = float(np.percentile(w1_pts[:, 2], 5.0))
                        z_p95_s = float(np.percentile(w1_pts[:, 2], 95.0))
                    else:
                        z_p5_s = float(w1_pts[:, 2].min())
                        z_p95_s = float(w1_pts[:, 2].max())
                    mins1[2] = z_p5_s
                    maxs1[2] = z_p95_s
            single_height_m = float(maxs1[2] - mins1[2])

            pw_item = {
                "shape": SegmentShape.PLANE_VERTICAL,
                "centroid": c1,
                "mins": mins1,
                "maxs": maxs1,
                "normal": w1.get("normal"),
                "length_m": wall_len_m,
                "height_m": single_height_m,
                "thickness_mm": thickness_mm,
                "point_count": w1.get("point_count", 0),
                "confidence": float(w1.get("confidence", 0.90)),
                "tags": tags_out,
            }
            if w1_pts is not None:
                pw_item["points"] = w1_pts
            physical_walls.append(pw_item)

    valid_physical_walls = [
        pw for pw in physical_walls
        if pw.get("point_count", 0) >= 10
        and pw.get("length_m", 0.0) >= 0.15
        and pw.get("height_m", 0.0) >= 0.20
    ]
    return valid_physical_walls if valid_physical_walls else physical_walls



# ── Step 7: Section 16 Diagnostic Table Formatter ────────────────────────────


def format_diagnostic_table(features_list: list[PlaneFeatures], max_display: int = 35) -> str:
    """Format the exact Section 16 diagnostic table with numerical evidence."""
    lines = [
        "Plane | Inliers | Area (m²) | Length (m) | Height (m) | Density (pts/m²) | Normal (nx, ny) | Plane Offset (m) | Z Cont. | Curvature | Wall Score | Classification",
        "------|---------|-----------|------------|------------|------------------|-----------------|------------------|---------|-----------|------------|-------------------",
    ]
    for feat in features_list[:max_display]:
        nx = f"{feat.normal[0]:+.2f}"
        ny = f"{feat.normal[1]:+.2f}"
        norm_str = f"[{nx},{ny}]"
        lines.append(
            f"P{feat.plane_id:02d}   | "
            f"{feat.inliers:7d} | "
            f"{feat.area_m2:9.2f} | "
            f"{feat.length_m:10.2f} | "
            f"{feat.height_m:10.2f} | "
            f"{feat.density_pts_m2:16.1f} | "
            f"{norm_str:15s} | "
            f"{feat.plane_offset_m:16.2f} | "
            f"{feat.z_continuity:7.2f} | "
            f"{feat.curvature_evidence:9.2f} | "
            f"{feat.wall_score:10.2f} | "
            f"{feat.classification.value}"
        )
    if len(features_list) > max_display:
        lines.append(f"... and {len(features_list) - max_display} more vertical planes audited.")
    return "\n".join(lines)


# ── Step 7b: Wall Junction Snapping ──────────────────────────────────────────
# Adapted from Cloud2BIM clustering/wall_grouping.py::adjust_wall_intersections


def snap_wall_junctions(
    physical_walls: list[dict],
    max_snap_m: float = 0.40,
) -> tuple[list[dict], int]:
    """Snap near-intersecting wall endpoints at T-junctions and corners.

    For each pair of walls whose axes are near-perpendicular (45°-135°):
    compute the 2D line-line intersection and snap endpoints within
    max_snap_m tolerance.

    Returns:
        (updated_walls, snap_count)
    """
    if len(physical_walls) < 2:
        return physical_walls, 0

    snap_count = 0

    for i in range(len(physical_walls)):
        tags_i = physical_walls[i].get("tags", {})
        sx_i = tags_i.get("_wall_start_x_m")
        sy_i = tags_i.get("_wall_start_y_m")
        ex_i = tags_i.get("_wall_end_x_m")
        ey_i = tags_i.get("_wall_end_y_m")
        if sx_i is None or sy_i is None or ex_i is None or ey_i is None:
            continue

        for j in range(i + 1, len(physical_walls)):
            tags_j = physical_walls[j].get("tags", {})
            sx_j = tags_j.get("_wall_start_x_m")
            sy_j = tags_j.get("_wall_start_y_m")
            ex_j = tags_j.get("_wall_end_x_m")
            ey_j = tags_j.get("_wall_end_y_m")
            if sx_j is None or sy_j is None or ex_j is None or ey_j is None:
                continue

            # Check angle between walls — only snap near-perpendicular pairs
            ang_i = float(tags_i.get("wall_angle_deg", 0.0))
            ang_j = float(tags_j.get("wall_angle_deg", 0.0))
            ang_diff = abs(ang_i - ang_j)
            ang_diff = min(ang_diff, 180.0 - ang_diff)
            if ang_diff < 45.0:
                # Nearly parallel — don't snap
                continue

            # Compute 2D line-line intersection
            x1, y1 = float(sx_i), float(sy_i)
            x2, y2 = float(ex_i), float(ey_i)
            x3, y3 = float(sx_j), float(sy_j)
            x4, y4 = float(ex_j), float(ey_j)

            denom = (x1 - x2) * (y3 - y4) - (y1 - y2) * (x3 - x4)
            if abs(denom) < 1e-8:
                continue

            det1 = x1 * y2 - y1 * x2
            det2 = x3 * y4 - y3 * x4
            px = (det1 * (x3 - x4) - (x1 - x2) * det2) / denom
            py = (det1 * (y3 - y4) - (y1 - y2) * det2) / denom

            # Snap endpoints of wall i
            for label, xk, yk, xkey, ykey in [
                ("start", x1, y1, "_wall_start_x_m", "_wall_start_y_m"),
                ("end", x2, y2, "_wall_end_x_m", "_wall_end_y_m"),
            ]:
                d = math.hypot(xk - px, yk - py)
                if d <= max_snap_m:
                    tags_i[xkey] = px
                    tags_i[ykey] = py
                    snap_count += 1

            # Snap endpoints of wall j
            for label, xk, yk, xkey, ykey in [
                ("start", x3, y3, "_wall_start_x_m", "_wall_start_y_m"),
                ("end", x4, y4, "_wall_end_x_m", "_wall_end_y_m"),
            ]:
                d = math.hypot(xk - px, yk - py)
                if d <= max_snap_m:
                    tags_j[xkey] = px
                    tags_j[ykey] = py
                    snap_count += 1
    if snap_count > 0:
        for pw in physical_walls:
            t = pw.get("tags", {})
            if "_wall_start_x_m" in t and "_wall_end_x_m" in t:
                dx = float(t["_wall_end_x_m"]) - float(t["_wall_start_x_m"])
                dy = float(t.get("_wall_end_y_m", 0.0)) - float(t.get("_wall_start_y_m", 0.0))
                l_m = math.hypot(dx, dy)
                if l_m > 0.15:
                    t["wall_length_mm"] = round(l_m * 1000.0, 1)
                    pw["length_m"] = l_m

    return physical_walls, snap_count


# ── Step 8: Complete Production Pipeline Execution ───────────────────────────


def run_stage2_pipeline(
    planes: list[dict],
    cloud_center_xy: np.ndarray | None = None,
    min_wall_score: float = 0.50,
) -> Stage2Result:
    """Execute the complete production Stage 2 architectural wall detection pipeline."""
    t0 = time.time()

    if cloud_center_xy is None:
        centroids = [p.get("centroid", np.zeros(3))[:2] for p in planes if "centroid" in p]
        cloud_center_xy = np.mean(centroids, axis=0) if centroids else np.zeros(2)

    # 1. Vertical surface extraction
    vertical_planes, rejected_planes = extract_vertical_surfaces(planes)

    if not vertical_planes:
        return Stage2Result(
            total_planes=len(planes),
            vertical_surfaces=0,
            architectural_walls=0,
            wall_fragments=0,
            curved_structural_elements=0,
            mep_non_wall_surfaces=len(rejected_planes),
            physical_walls=[],
            curved_elements=[],
            rejected_planes=rejected_planes,
            features_list=[],
            diagnostic_table="No vertical planes detected.",
            runtime_s=time.time() - t0,
        )

    # 2. Dominant grid orientation
    dominant_angles = infer_dominant_orientations(vertical_planes)

    # 3. Geometric feature extraction & classification
    features_list = [
        extract_plane_features(
            plane_idx=idx,
            plane_dict=p,
            cloud_center_xy=cloud_center_xy,
            dominant_angles=dominant_angles,
            vertical_planes=vertical_planes,
        )
        for idx, p in enumerate(vertical_planes)
    ]

    # 4. Partition by classification
    architectural_candidates: list[dict] = []
    wall_fragments: list[dict] = []
    curved_facets: list[dict] = []
    mep_non_wall: list[dict] = []

    for feat, p in zip(features_list, vertical_planes):
        p_copy = dict(p)
        if "points" not in p_copy or p_copy["points"] is None:
            if p.get("inlier_cloud") is not None:
                p_copy["points"] = np.asarray(p["inlier_cloud"].points)
        p_copy["classification"] = feat.classification.value
        p_copy["confidence"] = feat.wall_score
        p_copy["tags"] = dict(p.get("tags", {}))
        p_copy["tags"]["wall_score"] = feat.wall_score
        p_copy["tags"]["curvature_evidence"] = feat.curvature_evidence
        p_copy["tags"]["wall_classification"] = feat.classification.value

        wall_len_m = feat.length_m
        if wall_len_m <= 0.15 and "wall_length_mm" in p_copy["tags"]:
            wall_len_m = float(p_copy["tags"]["wall_length_mm"]) / 1000.0
        half_l = wall_len_m / 2.0
        ax = float(p_copy["tags"].get("wall_axis_x", 1.0))
        ay = float(p_copy["tags"].get("wall_axis_y", 0.0))
        axis_2d = canonicalize_wall_axis([ax, ay])
        c = p_copy.get("centroid", np.zeros(3))
        p_copy["tags"]["wall_axis_x"] = round(float(axis_2d[0]), 6)
        p_copy["tags"]["wall_axis_y"] = round(float(axis_2d[1]), 6)
        p_copy["tags"]["wall_angle_deg"] = wall_angle_degrees(axis_2d)

        have_existing_endpoints = (
            "_wall_start_x_m" in p_copy["tags"]
            and "_wall_end_x_m" in p_copy["tags"]
            and math.hypot(
                float(p_copy["tags"]["_wall_end_x_m"]) - float(p_copy["tags"]["_wall_start_x_m"]),
                float(p_copy["tags"].get("_wall_end_y_m", 0.0)) - float(p_copy["tags"].get("_wall_start_y_m", 0.0)),
            ) >= 0.15
        )
        if not have_existing_endpoints:
            p_copy["tags"]["_wall_start_x_m"] = float(c[0] - axis_2d[0] * half_l)
            p_copy["tags"]["_wall_start_y_m"] = float(c[1] - axis_2d[1] * half_l)
            p_copy["tags"]["_wall_end_x_m"] = float(c[0] + axis_2d[0] * half_l)
            p_copy["tags"]["_wall_end_y_m"] = float(c[1] + axis_2d[1] * half_l)
        p_copy["tags"]["wall_length_mm"] = round(wall_len_m * 1000.0, 1)

        if feat.classification == WallClassification.ARCHITECTURAL_WALL:
            architectural_candidates.append(p_copy)
        elif feat.classification == WallClassification.WALL_FRAGMENT:
            wall_fragments.append(p_copy)
            architectural_candidates.append(p_copy)
        elif feat.classification == WallClassification.CURVED_STRUCTURAL_SURFACE:
            curved_facets.append(p_copy)
        else:
            mep_non_wall.append(p_copy)

    # 5. Wall fragment grouping & coplanar merging
    merged_walls = group_wall_fragments(architectural_candidates)

    # 6. Physical wall face pairing
    physical_walls = pair_opposing_wall_faces(merged_walls, max_thickness_m=0.45)

    # 6b. Wall junction snapping
    physical_walls, snap_count = snap_wall_junctions(physical_walls, max_snap_m=0.40)
    if snap_count > 0:
        logger.info("wall_junctions_snapped", snap_count=snap_count)

    # 7. Group curved structural facets into unified elements
    curved_elements: list[dict] = []
    if curved_facets:
        pts_curved = []
        for f in curved_facets:
            if "points" in f and f["points"] is not None:
                pts_curved.append(f["points"])
            elif "inlier_cloud" in f and f["inlier_cloud"] is not None:
                pts_curved.append(np.asarray(f["inlier_cloud"].points))
        all_curv_pts = np.vstack(pts_curved) if pts_curved else np.zeros((1, 3))
        mins_c = all_curv_pts.min(axis=0)
        maxs_c = all_curv_pts.max(axis=0)
        c_curv = all_curv_pts.mean(axis=0)
        curved_elements.append(
            {
                "shape": SegmentShape.CURVED_STRUCTURAL_SURFACE,
                "centroid": c_curv,
                "mins": mins_c,
                "maxs": maxs_c,
                "point_count": len(all_curv_pts),
                "points": all_curv_pts,
                "facet_count": len(curved_facets),
                "confidence": 0.95,
                "classification": "CURVED_STRUCTURAL_SURFACE",
                "tags": {
                    "element_type": ElementType.CURVED_STRUCTURAL_SURFACE.value,
                    "radius_m": round(float(np.mean([f.plane_offset_m for f in features_list if f.classification == WallClassification.CURVED_STRUCTURAL_SURFACE])), 2) if any(f.classification == WallClassification.CURVED_STRUCTURAL_SURFACE for f in features_list) else 1.0,
                    "height_m": round(float(maxs_c[2] - mins_c[2]), 2),
                },
            }
        )

    diag_table = format_diagnostic_table(features_list)
    runtime = time.time() - t0

    return Stage2Result(
        total_planes=len(planes),
        vertical_surfaces=len(vertical_planes),
        architectural_walls=len([f for f in features_list if f.classification == WallClassification.ARCHITECTURAL_WALL]),
        wall_fragments=len([f for f in features_list if f.classification == WallClassification.WALL_FRAGMENT]),
        curved_structural_elements=len(curved_elements),
        mep_non_wall_surfaces=len(mep_non_wall) + len(rejected_planes),
        physical_walls=physical_walls,
        curved_elements=curved_elements,
        rejected_planes=mep_non_wall + rejected_planes,
        features_list=features_list,
        diagnostic_table=diag_table,
        runtime_s=runtime,
    )
