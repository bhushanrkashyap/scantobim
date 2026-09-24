"""Architectural Wall-Likeness Classifier & Geometric Feature Extraction.

Distinguishes true physical architectural envelope walls and fragments from
MEP equipment, internal radial plates, pipe supports, and tangential cylinder facets.
Complies with anti-hardcoding mandates (zero scene coordinates, zero magic IDs).
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from enum import Enum
from typing import Any

import numpy as np


class WallClassification(str, Enum):
    ARCHITECTURAL_WALL = "ARCHITECTURAL_WALL"
    WALL_FRAGMENT = "WALL_FRAGMENT"
    CURVED_STRUCTURAL_SURFACE = "CURVED_STRUCTURAL_SURFACE"
    MEP_EQUIPMENT = "MEP_EQUIPMENT"
    STRUCTURAL_NONWALL = "STRUCTURAL_NONWALL"
    FURNITURE_OBJECT = "FURNITURE_OBJECT"
    OTHER = "OTHER"


@dataclass
class PlaneGeometricFeatures:
    plane_id: int
    point_count: int
    normal: np.ndarray
    centroid: np.ndarray
    angle_deg: float
    robust_length_m: float
    robust_height_m: float
    planar_area_m2: float
    point_density_pts_m2: float
    spatial_fill_factor: float
    normal_consistency: float
    z_continuity: float
    d_center_m: float
    is_grid_aligned: bool
    wall_score: float
    classification: WallClassification


def infer_dominant_orientations(
    vertical_planes: list[dict], num_bins: int = 18, min_weight: float = 100.0
) -> list[float]:
    """Infer the primary architectural grid orientations from point-weighted distribution.

    No hardcoding: extracts histogram modes directly from detected vertical plane normals.
    """
    if not vertical_planes:
        return [0.0, 90.0]

    angles = []
    weights = []
    for p in vertical_planes:
        tags = p.get("tags", {})
        ax = tags.get("wall_axis_x", 1.0)
        ay = tags.get("wall_axis_y", 0.0)
        angle = (math.degrees(math.atan2(ay, ax)) + 360.0) % 180.0
        angles.append(angle)
        weights.append(float(p.get("point_count", len(p.get("inlier_cloud", {}).points if p.get("inlier_cloud") else []))))

    if not angles or sum(weights) <= 0:
        return [0.0, 90.0]

    h, edges = np.histogram(angles, bins=num_bins, range=(0.0, 180.0), weights=weights)
    sorted_bin_indices = np.argsort(h)[::-1]

    dominant_angles: list[float] = []
    for bin_idx in sorted_bin_indices:
        if h[bin_idx] < min_weight:
            continue
        bin_center = 0.5 * (edges[bin_idx] + edges[bin_idx + 1])
        if all(abs(bin_center - da) > 25.0 and abs(bin_center - da) < 155.0 for da in dominant_angles):
            dominant_angles.append(float(round(bin_center, 1)))
            if len(dominant_angles) >= 4:
                break

    if not dominant_angles:
        dominant_angles = [0.0, 90.0]

    return dominant_angles


def extract_plane_features(
    plane_idx: int,
    plane_dict: dict,
    cloud_center_xy: np.ndarray,
    dominant_angles: list[float],
    grid_res_m: float = 0.25,
) -> PlaneGeometricFeatures:
    """Extract measurable geometric features for a single vertical plane candidate."""
    cloud = plane_dict.get("inlier_cloud")
    if "points" in plane_dict:
        pts = np.asarray(plane_dict["points"], dtype=float)
        normals = np.asarray(plane_dict["normals"], dtype=float) if "normals" in plane_dict and plane_dict["normals"] is not None else None
    elif cloud is not None and hasattr(cloud, "points") and len(cloud.points) > 0:
        pts = np.asarray(cloud.points, dtype=float)
        normals = np.asarray(cloud.normals, dtype=float) if hasattr(cloud, "has_normals") and cloud.has_normals() and len(cloud.normals) > 0 else None
    else:
        pts = np.array([plane_dict.get("centroid", np.zeros(3))], dtype=float)
        normals = None

    c = np.asarray(plane_dict.get("centroid", pts.mean(axis=0)), dtype=float)
    n = np.asarray(plane_dict.get("normal", np.array([1.0, 0.0, 0.0])), dtype=float)
    norm_n = np.linalg.norm(n)
    if norm_n > 1e-6:
        n = n / norm_n

    tags = plane_dict.get("tags", {})
    ax = float(tags.get("wall_axis_x", -n[1]))
    ay = float(tags.get("wall_axis_y", n[0]))
    axis_2d = np.array([ax, ay], dtype=float)
    norm_ax = np.linalg.norm(axis_2d)
    if norm_ax > 1e-6:
        axis_2d /= norm_ax
    else:
        axis_2d = np.array([1.0, 0.0], dtype=float)

    angle_deg = float((math.degrees(math.atan2(axis_2d[1], axis_2d[0])) + 360.0) % 180.0)

    # 1. Projected 1D axis coordinate (along wall run)
    s = pts[:, :2] @ axis_2d
    if len(s) >= 10:
        s_p05, s_p95 = np.percentile(s, [5, 95])
        robust_length = max(float((s_p95 - s_p05) / 0.90), 0.10)
    else:
        s_p05 = float(s.min())
        robust_length = max(float(s.max() - s.min()), 0.10)

    # 2. Vertical span along Z
    z_pts = pts[:, 2]
    if len(z_pts) >= 10:
        z_p05, z_p95 = np.percentile(z_pts, [5, 95])
        robust_height = max(float((z_p95 - z_p05) / 0.90), 0.10)
    else:
        z_p05 = float(z_pts.min())
        robust_height = max(float(z_pts.max() - z_pts.min()), 0.10)

    # 3. Planar Area & Density
    planar_area = max(robust_length * robust_height, 0.01)
    density = float(len(pts) / planar_area)

    # 4. 2D Spatial Occupancy Fill Factor
    grid_s = max(int(robust_length / grid_res_m), 1)
    grid_z = max(int(robust_height / grid_res_m), 1)
    if grid_s <= 200 and grid_z <= 200 and len(pts) >= 10:
        occ = np.zeros((grid_z, grid_s), dtype=bool)
        si = np.clip(((s - s_p05) / grid_res_m).astype(int), 0, grid_s - 1)
        zi = np.clip(((z_pts - z_p05) / grid_res_m).astype(int), 0, grid_z - 1)
        occ[zi, si] = True
        fill_factor = float(np.sum(occ) / (grid_s * grid_z))
    else:
        fill_factor = 0.50 if len(pts) >= 100 else 0.10

    # 5. Normal Consistency of Inliers
    if normals is not None and len(normals) >= 10:
        dot_n = np.abs(np.dot(normals, n))
        norm_consistency = float(np.mean(dot_n >= 0.80))
    else:
        norm_consistency = 1.0

    # 6. Z Continuity (fraction of 1-metre vertical bins containing points)
    mins = pts.min(axis=0)
    maxs = pts.max(axis=0)
    z_bins = np.arange(mins[2], maxs[2] + 1.0, 1.0)
    if len(z_bins) > 1 and len(pts) >= 10:
        zh, _ = np.histogram(z_pts, bins=z_bins)
        z_continuity = float(np.sum(zh > 0) / len(zh))
    else:
        z_continuity = 1.0

    # 7. Alignment with Dominant Building Grid
    min_ang_diff = min(
        [min(abs(angle_deg - dom), 180.0 - abs(angle_deg - dom)) for dom in dominant_angles]
    ) if dominant_angles else 0.0
    is_grid_aligned = bool(min_ang_diff <= 10.0)

    # 8. Envelope vs Central Chord Distance
    n_xy = np.array([n[0], n[1]], dtype=float)
    norm_n_xy = np.linalg.norm(n_xy)
    if norm_n_xy > 1e-6:
        n_xy /= norm_n_xy
    d_center = float(abs(np.dot(c[:2] - cloud_center_xy, n_xy)))

    # Compute Wall Score & Classification
    score = compute_wall_likeness_score(
        robust_length=robust_length,
        robust_height=robust_height,
        density=density,
        fill_factor=fill_factor,
        norm_consistency=norm_consistency,
        is_grid_aligned=is_grid_aligned,
        d_center=d_center,
    )

    classification = classify_features(
        wall_score=score,
        robust_length=robust_length,
        robust_height=robust_height,
        density=density,
        fill_factor=fill_factor,
        is_grid_aligned=is_grid_aligned,
        d_center=d_center,
    )

    return PlaneGeometricFeatures(
        plane_id=plane_idx,
        point_count=len(pts),
        normal=n,
        centroid=c,
        angle_deg=round(angle_deg, 1),
        robust_length_m=round(robust_length, 2),
        robust_height_m=round(robust_height, 2),
        planar_area_m2=round(planar_area, 2),
        point_density_pts_m2=round(density, 1),
        spatial_fill_factor=round(fill_factor, 2),
        normal_consistency=round(norm_consistency, 3),
        z_continuity=round(z_continuity, 2),
        d_center_m=round(d_center, 2),
        is_grid_aligned=is_grid_aligned,
        wall_score=round(score, 2),
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
) -> float:
    """Calculate an explainable multi-feature wall-likeness score in [0.0, 1.0]."""
    s_h = min(robust_height / 8.0, 1.0)
    s_l = min(robust_length / 5.0, 1.0)
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
    if not is_grid_aligned:
        score *= 0.80
    return float(max(0.0, min(score, 1.0)))


def classify_features(
    wall_score: float,
    robust_length: float,
    robust_height: float,
    density: float,
    fill_factor: float,
    is_grid_aligned: bool,
    d_center: float,
) -> WallClassification:
    """Classify a vertical plane candidate into one of the required categories."""
    is_central_chord = (d_center < 1.3) and (fill_factor < 0.32 or density < 30.0 or not is_grid_aligned)
    is_thin_slice = (fill_factor < 0.22) or (density < 18.0)

    if is_central_chord or is_thin_slice:
        return WallClassification.MEP_EQUIPMENT

    if wall_score >= 0.58 and is_grid_aligned and robust_height >= 2.0 and robust_length >= 1.8:
        return WallClassification.ARCHITECTURAL_WALL

    if wall_score >= 0.38 and (is_grid_aligned or d_center > 2.2) and robust_height >= 1.2 and robust_length >= 0.8:
        return WallClassification.WALL_FRAGMENT

    if robust_height < 1.0 and robust_length < 1.0:
        return WallClassification.FURNITURE_OBJECT

    if robust_length < 0.80 and robust_height >= 2.0:
        return WallClassification.STRUCTURAL_NONWALL

    if not is_grid_aligned and d_center < 2.0:
        return WallClassification.MEP_EQUIPMENT

    return WallClassification.OTHER


def format_diagnostic_table(features_list: list[PlaneGeometricFeatures], max_display: int = 40) -> str:
    """Format the exact Section 14 diagnostic table with numerical evidence."""
    lines = [
        "Plane | Vertical | Area (m²) | Length (m) | Height (m) | Density (pts/m²) | Z continuity | Wall score | Classification",
        "------|----------|-----------|------------|------------|------------------|--------------|------------|-------------------",
    ]
    display_items = features_list[:max_display]
    for f in display_items:
        lines.append(
            f"P{f.plane_id:02d}   | True     | {f.planar_area_m2:9.2f} | {f.robust_length_m:10.2f} | {f.robust_height_m:10.2f} | {f.point_density_pts_m2:16.1f} | {f.z_continuity:12.2f} | {f.wall_score:10.2f} | {f.classification.value}"
        )
    if len(features_list) > max_display:
        lines.append(f"... and {len(features_list) - max_display} more vertical planes audited.")
    return "\n".join(lines)
