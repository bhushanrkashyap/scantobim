#!/usr/bin/env python3
"""
run_stage2_demo.py
──────────────────
Scan-to-BIM Stage 2 Pipeline Demonstration & Audit Tool.

Usage:
  # Synthetic Benchmark Mode (clearly announced):
  python3 run_stage2_demo.py

  # External Real Scan Mode:
  python3 run_stage2_demo.py /path/to/ACTUAL_SCAN.ply
"""

from __future__ import annotations

import argparse
import math
import os
import sys
import time
from pathlib import Path

# Ensure Matplotlib config uses a local/writable directory on macOS
os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib_stb")

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

import numpy as np
import open3d as o3d

from agent.models import SegmentShape
from agent.tools.detection_config import (
    NORMAL_MAX_NN,
    NORMAL_RADIUS_VOXEL_MULT,
    SOR_MIN_NEIGHBORS,
    SOR_NB_NEIGHBORS,
    SOR_NEIGHBOR_DIVISOR,
    SOR_STD_RATIO,
    VOXEL_SIZE_M,
)
from agent.tools.geometry_tools import estimate_normals
from agent.tools.scan_tools import (
    detect_planes,
    downsample,
    remove_statistical_outliers,
)
from agent.stage2_pipeline import (
    PlaneFeatures,
    Stage2Result,
    WallClassification,
    extract_plane_features,
    extract_vertical_surfaces,
    format_diagnostic_table,
    group_wall_fragments,
    infer_dominant_orientations,
    pair_opposing_wall_faces,
    run_stage2_pipeline,
)



def generate_synthetic_benchmark(
    length_a: float = 4.0,
    length_b: float = 3.0,
    height: float = 3.0,
    thickness: float = 0.20,
    pts_per_m2: float = 600.0,
    noise_m: float = 0.003,
    seed: int = 42,
) -> tuple[np.ndarray, dict]:
    """Generate the canonical L-shaped dual-wall synthetic benchmark."""
    rng = np.random.default_rng(seed)
    chunks = []

    # Wall A faces (along X)
    n_pts_a = int(length_a * height * pts_per_m2)
    for y_face in (0.0, thickness):
        x_pts = rng.uniform(0.0, length_a, n_pts_a)
        y_pts = np.full(n_pts_a, y_face)
        z_pts = rng.uniform(0.0, height, n_pts_a)
        face = np.column_stack([x_pts, y_pts, z_pts])
        face += rng.normal(0.0, noise_m, face.shape)
        chunks.append(face)

    # Wall B faces (along Y)
    n_pts_b = int(length_b * height * pts_per_m2)
    for x_face in (0.0, thickness):
        x_pts = np.full(n_pts_b, x_face)
        y_pts = rng.uniform(0.0, length_b, n_pts_b)
        z_pts = rng.uniform(0.0, height, n_pts_b)
        face = np.column_stack([x_pts, y_pts, z_pts])
        face += rng.normal(0.0, noise_m, face.shape)
        chunks.append(face)

    # Horizontal Floor plane (Z = 0)
    n_pts_floor = int(length_a * length_b * (pts_per_m2 * 0.4))
    x_floor = rng.uniform(0.0, length_a, n_pts_floor)
    y_floor = rng.uniform(0.0, length_b, n_pts_floor)
    z_floor = np.zeros(n_pts_floor)
    floor_pts = np.column_stack([x_floor, y_floor, z_floor])
    floor_pts += rng.normal(0.0, noise_m, floor_pts.shape)
    chunks.append(floor_pts)

    all_pts = np.ascontiguousarray(np.vstack(chunks), dtype=np.float64)
    info = {
        "expected_walls": 2,
        "wall_a": {"length_m": length_a, "height_m": height, "orientation_deg": 0.0},
        "wall_b": {"length_m": length_b, "height_m": height, "orientation_deg": 90.0},
    }
    return all_pts, info


def load_input_points(file_path: Path) -> tuple[np.ndarray, str]:
    """Load points from file, returning (points_array_metres, detected_unit)."""
    ext = file_path.suffix.lower()

    if ext in (".ply", ".pcd", ".xyz"):
        pcd = o3d.io.read_point_cloud(str(file_path))
        if pcd.is_empty():
            raise ValueError(f"Empty point cloud loaded from {file_path}")
        pts = np.asarray(pcd.points, dtype=np.float64)
    elif ext == ".npy":
        pts = np.load(str(file_path))
        if pts.ndim != 2 or pts.shape[1] < 3:
            raise ValueError(f"NPY array must have shape (N, 3), got {pts.shape}")
        pts = np.ascontiguousarray(pts[:, :3], dtype=np.float64)
    elif ext == ".e57":
        try:
            from agent.tools.scan_tools import _load_e57
            pcd = _load_e57(file_path)
            pts = np.asarray(pcd.points, dtype=np.float64)
        except ImportError:
            raise RuntimeError(
                "E57 file format requires 'pye57' package, which is not installed. "
                "Please convert your scan to PLY or install pye57."
            )
    elif ext in (".las", ".laz"):
        try:
            from agent.tools.scan_tools import _load_las
            pcd = _load_las(file_path)
            pts = np.asarray(pcd.points, dtype=np.float64)
        except ImportError:
            raise RuntimeError("LAS/LAZ format requires 'laspy' package.")
    else:
        raise ValueError(f"Unsupported point cloud format: '{ext}'")

    span = pts.max(axis=0) - pts.min(axis=0)
    max_span = float(np.max(span))
    if max_span > 250.0:
        detected_unit = "millimetres (scaled to metres for processing)"
        pts_m = pts / 1000.0
    else:
        detected_unit = "metres"
        pts_m = pts

    return pts_m, detected_unit


def group_into_physical_walls(wall_segments: list[dict], max_thickness_m: float = 0.45) -> list[dict]:
    """Group opposing parallel faces of the same structural wall into physical walls."""
    if not wall_segments:
        return []

    used = [False] * len(wall_segments)
    physical_walls = []

    for i in range(len(wall_segments)):
        if used[i]:
            continue
        seg_a = wall_segments[i]
        tags_a = seg_a.get("tags", {})
        ax_a = np.array([tags_a.get("wall_axis_x", 1.0), tags_a.get("wall_axis_y", 0.0)], dtype=float)
        norm_a = np.linalg.norm(ax_a)
        if norm_a > 1e-6:
            ax_a /= norm_a
        normal_a = np.array([-ax_a[1], ax_a[0]], dtype=float)

        ca = seg_a["centroid"]
        dist_normal_a = float(ca[0] * normal_a[0] + ca[1] * normal_a[1])

        matched_idx = None
        for j in range(i + 1, len(wall_segments)):
            if used[j]:
                continue
            seg_b = wall_segments[j]
            tags_b = seg_b.get("tags", {})
            ax_b = np.array([tags_b.get("wall_axis_x", 1.0), tags_b.get("wall_axis_y", 0.0)], dtype=float)
            norm_b = np.linalg.norm(ax_b)
            if norm_b > 1e-6:
                ax_b /= norm_b

            dot = abs(float(np.dot(ax_a, ax_b)))
            if dot < 0.990:
                continue

            min_za, max_za = float(seg_a["mins"][2]), float(seg_a["maxs"][2])
            min_zb, max_zb = float(seg_b["mins"][2]), float(seg_b["maxs"][2])
            overlap_z = min(max_za, max_zb) - max(min_za, min_zb)
            min_h = max(min(max_za - min_za, max_zb - min_zb), 0.01)
            if min_h > 0 and (overlap_z / min_h < 0.25 and overlap_z < 0.50):
                continue

            cb = seg_b["centroid"]
            dist_normal_b = float(cb[0] * normal_a[0] + cb[1] * normal_a[1])
            face_dist = abs(dist_normal_a - dist_normal_b)

            if 0.05 <= face_dist <= max_thickness_m:
                matched_idx = j
                break

        used[i] = True
        if matched_idx is not None:
            used[matched_idx] = True
            seg_b = wall_segments[matched_idx]
            tags_b = seg_b.get("tags", {})
            cb = seg_b["centroid"]
            dist_normal_b = float(cb[0] * normal_a[0] + cb[1] * normal_a[1])
            face_thickness_mm = round(abs(dist_normal_a - dist_normal_b) * 1000.0, 1)

            mins = np.minimum(seg_a["mins"], seg_b["mins"])
            maxs = np.maximum(seg_a["maxs"], seg_b["maxs"])
            combined_pts = seg_a.get("point_count", 0) + seg_b.get("point_count", 0)

            c1 = seg_a.get("inlier_cloud")
            c2 = seg_b.get("inlier_cloud")
            combined_cloud = None
            if c1 is not None and c2 is not None:
                p1 = np.asarray(c1.points)
                p2 = np.asarray(c2.points)
                combined_cloud = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(np.vstack([p1, p2])))
            elif c1 is not None:
                combined_cloud = c1

            wall_length_mm = max(tags_a.get("wall_length_mm", 0.0), tags_b.get("wall_length_mm", 0.0))
            phys_tags = dict(tags_a)
            phys_tags["wall_thickness_mm"] = face_thickness_mm
            phys_tags["wall_length_mm"] = wall_length_mm
            phys_tags["wall_faces_paired"] = True
            phys_tags["paired_face_count"] = 2

            start_a = np.array([tags_a.get("_wall_start_x_m", ca[0]), tags_a.get("_wall_start_y_m", ca[1])])
            start_b = np.array([tags_b.get("_wall_start_x_m", cb[0]), tags_b.get("_wall_start_y_m", cb[1])])
            end_a = np.array([tags_a.get("_wall_end_x_m", ca[0]), tags_a.get("_wall_end_y_m", ca[1])])
            end_b = np.array([tags_b.get("_wall_end_x_m", cb[0]), tags_b.get("_wall_end_y_m", cb[1])])

            mid_start = 0.5 * (start_a + start_b)
            mid_end = 0.5 * (end_a + end_b)
            phys_tags["_wall_start_x_m"] = float(mid_start[0])
            phys_tags["_wall_start_y_m"] = float(mid_start[1])
            phys_tags["_wall_end_x_m"] = float(mid_end[0])
            phys_tags["_wall_end_y_m"] = float(mid_end[1])

            phys_wall = {
                "shape": SegmentShape.PLANE_VERTICAL,
                "mins": mins,
                "maxs": maxs,
                "centroid": (seg_a["centroid"] + seg_b["centroid"]) / 2.0,
                "point_count": combined_pts,
                "confidence": max(seg_a.get("confidence", 1.0), seg_b.get("confidence", 1.0)),
                "inlier_cloud": combined_cloud,
                "tags": phys_tags,
                "source_planes": 2,
            }
            physical_walls.append(phys_wall)
        else:
            single_wall = dict(seg_a)
            single_wall["source_planes"] = 1
            physical_walls.append(single_wall)

    return physical_walls


def save_real_artifacts(
    raw_pts: np.ndarray,
    pcd_stage1: o3d.geometry.PointCloud,
    all_planes: list[dict],
    vertical_candidates: list[dict],
    physical_walls: list[dict],
    output_dir: Path,
    curved_elements: list[dict] | None = None,
    rejected_planes: list[dict] | None = None,
) -> None:
    """Generate the exact PLY and PNG files required by the validation gate."""
    output_dir.mkdir(parents=True, exist_ok=True)

    # 1. 01_stage1_real.ply
    o3d.io.write_point_cloud(str(output_dir / "01_stage1_real.ply"), pcd_stage1)

    # 2. 02_ransac_real.ply — all detected planes colored distinctly
    colors_cycle = [
        [1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0], [1.0, 1.0, 0.0],
        [1.0, 0.0, 1.0], [0.0, 1.0, 1.0], [1.0, 0.5, 0.0], [0.5, 0.0, 0.5],
    ]
    ransac_pts, ransac_cols = [], []
    for idx, plane in enumerate(all_planes):
        cloud = plane.get("inlier_cloud")
        if cloud is not None:
            pts = np.asarray(cloud.points)
            col = colors_cycle[idx % len(colors_cycle)]
            ransac_pts.append(pts)
            ransac_cols.append(np.tile(col, (len(pts), 1)))

    if ransac_pts:
        pcd_ransac = o3d.geometry.PointCloud()
        pcd_ransac.points = o3d.utility.Vector3dVector(np.vstack(ransac_pts))
        pcd_ransac.colors = o3d.utility.Vector3dVector(np.vstack(ransac_cols))
        o3d.io.write_point_cloud(str(output_dir / "02_ransac_real.ply"), pcd_ransac)

    # 3. 03_vertical_candidates_real.ply
    vert_pts, vert_cols = [], []
    for idx, cand in enumerate(vertical_candidates):
        cloud = cand.get("inlier_cloud")
        if cloud is not None:
            pts = np.asarray(cloud.points)
            col = colors_cycle[idx % len(colors_cycle)]
            vert_pts.append(pts)
            vert_cols.append(np.tile(col, (len(pts), 1)))

    if vert_pts:
        pcd_vert = o3d.geometry.PointCloud()
        pcd_vert.points = o3d.utility.Vector3dVector(np.vstack(vert_pts))
        pcd_vert.colors = o3d.utility.Vector3dVector(np.vstack(vert_cols))
        o3d.io.write_point_cloud(str(output_dir / "03_vertical_candidates_real.ply"), pcd_vert)

    # 4. 04_physical_walls_real.ply
    wall_pts, wall_cols = [], []
    for idx, wall in enumerate(physical_walls):
        pts = wall.get("points")
        if pts is None:
            cloud = wall.get("inlier_cloud")
            if cloud is not None:
                pts = np.asarray(cloud.points)
        if pts is not None and len(pts) > 0:
            col = colors_cycle[idx % len(colors_cycle)]
            wall_pts.append(pts)
            wall_cols.append(np.tile(col, (len(pts), 1)))

    if wall_pts:
        pcd_walls = o3d.geometry.PointCloud()
        pcd_walls.points = o3d.utility.Vector3dVector(np.vstack(wall_pts))
        pcd_walls.colors = o3d.utility.Vector3dVector(np.vstack(wall_cols))
        o3d.io.write_point_cloud(str(output_dir / "04_physical_walls_real.ply"), pcd_walls)

    # Matplotlib Overlays against ORIGINAL point cloud
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    stride = max(1, len(raw_pts) // 15000)
    plot_raw = raw_pts[::stride]

    hex_colors = [
        "#e41a1c", "#377eb8", "#4daf4a", "#984ea3",
        "#ff7f00", "#ffff33", "#a65628", "#f781bf",
        "#1f77b4", "#ff7f0e", "#2ca02c", "#d62728",
    ]

    # 5. 05_wall_overlay_plan.png (Top / Plan View XY)
    fig, ax = plt.subplots(figsize=(10, 8))
    ax.scatter(plot_raw[:, 0], plot_raw[:, 1], c="lightgray", s=2, alpha=0.35, label="Original Cloud")
    for idx, wall in enumerate(physical_walls):
        c = hex_colors[idx % len(hex_colors)]
        tags = wall.get("tags", {})
        sx = tags.get("_wall_start_x_m", wall["centroid"][0])
        sy = tags.get("_wall_start_y_m", wall["centroid"][1])
        ex = tags.get("_wall_end_x_m", wall["centroid"][0])
        ey = tags.get("_wall_end_y_m", wall["centroid"][1])
        len_m = tags.get("wall_length_mm", 0.0) / 1000.0
        lbl = f"Wall {idx+1} ({len_m:.1f}m)" if idx < 12 else None
        ax.plot([sx, ex], [sy, ey], color=c, linewidth=3.5, label=lbl)
    ax.set_title("05_wall_overlay_plan (Top / Plan View XY - Architectural Walls)", fontsize=13, fontweight="bold")
    ax.set_xlabel("X (metres)")
    ax.set_ylabel("Y (metres)")
    ax.set_aspect("equal")
    ax.grid(True, linestyle="--", alpha=0.5)
    ax.legend(loc="upper right", framealpha=0.9, fontsize=8)
    plt.savefig(output_dir / "05_wall_overlay_plan.png", dpi=180, bbox_inches="tight")
    plt.close()

    # 6. 06_wall_overlay_3d.png (3D Perspective View)
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection="3d")
    ax.scatter(plot_raw[:, 0], plot_raw[:, 1], plot_raw[:, 2], c="lightgray", s=1.0, alpha=0.2, label="Original Cloud")
    for idx, wall in enumerate(physical_walls):
        c = hex_colors[idx % len(hex_colors)]
        w_pts = wall.get("points")
        if w_pts is None:
            cloud = wall.get("inlier_cloud")
            if cloud is not None:
                w_pts = np.asarray(cloud.points)
        if w_pts is not None and len(w_pts) > 0:
            w_sub = w_pts[::max(1, len(w_pts) // 1200)]
            lbl = f"Wall {idx+1}" if idx < 12 else None
            ax.scatter(w_sub[:, 0], w_sub[:, 1], w_sub[:, 2], color=c, s=6, alpha=0.8, label=lbl)
    ax.set_title("06_wall_overlay_3d (Physical Walls Overlaid on Original Cloud)", fontsize=13, fontweight="bold")
    ax.set_xlabel("X (m)")
    ax.set_ylabel("Y (m)")
    ax.set_zlabel("Z (m)")
    ax.legend(loc="upper right", framealpha=0.9, fontsize=8)
    plt.savefig(output_dir / "06_wall_overlay_3d.png", dpi=180, bbox_inches="tight")
    plt.close()

    # 7. 07_wall_overlay_elevation.png (Side / Elevation View XZ)
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.scatter(plot_raw[:, 0], plot_raw[:, 2], c="lightgray", s=2, alpha=0.35, label="Original Cloud")
    for idx, wall in enumerate(physical_walls):
        c = hex_colors[idx % len(hex_colors)]
        mins = wall["mins"]
        maxs = wall["maxs"]
        lbl = f"Wall {idx+1}" if idx < 12 else None
        x_span = [min(mins[0], maxs[0]), max(mins[0], maxs[0])]
        ax.fill_between(x_span, mins[2], maxs[2], color=c, alpha=0.35, label=lbl)
    ax.set_title("07_wall_overlay_elevation (Side / Elevation View XZ)", fontsize=13, fontweight="bold")
    ax.set_xlabel("X (metres)")
    ax.set_ylabel("Z (metres)")
    ax.set_aspect("equal")
    ax.grid(True, linestyle="--", alpha=0.5)
    ax.legend(loc="upper right", framealpha=0.9, fontsize=8)
    plt.savefig(output_dir / "07_wall_overlay_elevation.png", dpi=180, bbox_inches="tight")
    plt.close()

    # 8. 08_curved_overlay_3d.png (Curved Structural Surfaces)
    if curved_elements:
        fig = plt.figure(figsize=(10, 8))
        ax = fig.add_subplot(111, projection="3d")
        ax.scatter(plot_raw[:, 0], plot_raw[:, 1], plot_raw[:, 2], c="lightgray", s=1.0, alpha=0.2, label="Original Cloud")
        for idx, ce in enumerate(curved_elements):
            c_pts = ce.get("points")
            if c_pts is not None and len(c_pts) > 0:
                sub = c_pts[::max(1, len(c_pts) // 2000)]
                ax.scatter(sub[:, 0], sub[:, 1], sub[:, 2], color="darkturquoise", s=8, alpha=0.85, label=f"Curved Structural Core ({ce.get('facet_count', 0)} facets)")
        ax.set_title("08_curved_overlay_3d (Curved Structural Surfaces Overlaid on Cloud)", fontsize=13, fontweight="bold")
        ax.set_xlabel("X (m)")
        ax.set_ylabel("Y (m)")
        ax.set_zlabel("Z (m)")
        ax.legend(loc="upper right", framealpha=0.9, fontsize=9)
        plt.savefig(output_dir / "08_curved_overlay_3d.png", dpi=180, bbox_inches="tight")
        plt.close()

    # 9. 09_rejected_overlay_3d.png (Rejected MEP & Non-wall geometry)
    if rejected_planes:
        fig = plt.figure(figsize=(10, 8))
        ax = fig.add_subplot(111, projection="3d")
        ax.scatter(plot_raw[:, 0], plot_raw[:, 1], plot_raw[:, 2], c="lightgray", s=1.0, alpha=0.15, label="Original Cloud")
        rej_pts = []
        for rp in rejected_planes:
            if "points" in rp:
                rej_pts.append(rp["points"])
            elif rp.get("inlier_cloud") is not None:
                rej_pts.append(np.asarray(rp["inlier_cloud"].points))
        if rej_pts:
            all_rej = np.vstack(rej_pts)
            sub_rej = all_rej[::max(1, len(all_rej) // 2500)]
            ax.scatter(sub_rej[:, 0], sub_rej[:, 1], sub_rej[:, 2], color="crimson", s=6, alpha=0.6, label=f"Rejected MEP/Non-wall ({len(rejected_planes)} surfaces)")
        ax.set_title("09_rejected_overlay_3d (Rejected MEP & Non-Wall Geometry)", fontsize=13, fontweight="bold")
        ax.set_xlabel("X (m)")
        ax.set_ylabel("Y (m)")
        ax.set_zlabel("Z (m)")
        ax.legend(loc="upper right", framealpha=0.9, fontsize=9)
        plt.savefig(output_dir / "09_rejected_overlay_3d.png", dpi=180, bbox_inches="tight")
        plt.close()



def run_pipeline(
    raw_points_m: np.ndarray,
    source_name: str,
    detected_units: str,
    voxel_size_m: float = VOXEL_SIZE_M,
    sor_nb_neighbors: int = SOR_NB_NEIGHBORS,
    sor_std_ratio: float = SOR_STD_RATIO,
    min_inliers: int = 200,
    save_viz_dir: Path | None = None,
) -> dict:
    raw_count = len(raw_points_m)

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(raw_points_m)

    # ── Stage 1: Pre-processing ──────────────────────────────────────────────
    t0 = time.time()
    adaptive_sor_k = min(
        sor_nb_neighbors,
        max(SOR_MIN_NEIGHBORS, raw_count // SOR_NEIGHBOR_DIVISOR),
    )
    pcd_sor = remove_statistical_outliers(
        pcd, nb_neighbors=adaptive_sor_k, std_ratio=sor_std_ratio
    )
    sor_count = len(pcd_sor.points)

    pcd_voxel = downsample(pcd_sor, voxel_size_m=voxel_size_m)
    voxel_count = len(pcd_voxel.points)

    normal_radius = NORMAL_RADIUS_VOXEL_MULT * voxel_size_m
    normal_res = estimate_normals(
        pcd_voxel, radius=normal_radius, max_nn=NORMAL_MAX_NN
    )
    normals_estimated = (
        len(pcd_voxel.points) if normal_res.get("success") else 0
    )
    stage1_time = time.time() - t0

    # ── Stage 2: RANSAC Plane Detection ──────────────────────────────────────
    t1 = time.time()
    all_planes, remaining_cloud = detect_planes(
        pcd_voxel,
        min_inliers=min_inliers,
    )
    total_planes = len(all_planes)

    vertical_planes = [
        p for p in all_planes if p.get("shape") == SegmentShape.PLANE_VERTICAL
    ]
    non_vertical_planes = [
        p for p in all_planes if p.get("shape") != SegmentShape.PLANE_VERTICAL
    ]

    # ── Stage 2b: Production Stage 2 Pipeline Execution ──────────────────────
    pts_voxel = np.asarray(pcd_voxel.points)
    cloud_center_xy = pts_voxel[:, :2].mean(axis=0)

    # In external scan mode (many planes), run the multi-feature production Stage 2 pipeline
    if len(vertical_planes) > 4:
        stage2_res = run_stage2_pipeline(all_planes, cloud_center_xy=cloud_center_xy)

        print("\n" + "=" * 60)
        print("SECTION 16 DIAGNOSTIC TABLE: ALL VERTICAL PLANES")
        print("=" * 60)
        print(stage2_res.diagnostic_table)
        print("=" * 60 + "\n")

        architectural_candidates = [
            p for idx, p in enumerate(vertical_planes)
            if idx < len(stage2_res.features_list) and stage2_res.features_list[idx].classification in (WallClassification.ARCHITECTURAL_WALL, WallClassification.WALL_FRAGMENT)
        ]
        physical_wall_groups = stage2_res.physical_walls
        curved_elements = stage2_res.curved_elements
        rejected_planes = stage2_res.rejected_planes
        mep_rejected_count = stage2_res.mep_non_wall_surfaces
        total_rejected = mep_rejected_count
        stage2_time = stage2_res.runtime_s
    else:
        # Synthetic benchmark mode: pass vertical candidates directly
        merged_walls = group_wall_fragments(vertical_planes)
        physical_wall_groups = pair_opposing_wall_faces(merged_walls, max_thickness_m=0.45)
        architectural_candidates = vertical_planes
        curved_elements = []
        rejected_planes = non_vertical_planes
        mep_rejected_count = 0
        total_rejected = len(non_vertical_planes)
        stage2_time = time.time() - t1

    # ── Save Artifacts ───────────────────────────────────────────────────────
    t2 = time.time()
    if save_viz_dir is not None:
        save_real_artifacts(
            raw_pts=raw_points_m,
            pcd_stage1=pcd_voxel,
            all_planes=all_planes,
            vertical_candidates=architectural_candidates,
            physical_walls=physical_wall_groups,
            output_dir=save_viz_dir,
            curved_elements=curved_elements,
            rejected_planes=rejected_planes,
        )
    viz_time = time.time() - t2

    # ── Print Console Narration ──────────────────────────────────────────────
    print("Stage 1:")
    print(f"  Input points:     {raw_count:,}")
    print(f"  SOR points:       {sor_count:,}")
    print(f"  Voxel points:     {voxel_count:,} (voxel: {voxel_size_m*1000:.0f}mm)")
    print(f"  Normals:          {normals_estimated:,}")
    print(f"  Duration:         {stage1_time:.3f} s")

    print("\nStage 2:")
    print(f"  RANSAC planes:    {total_planes}")
    print(f"  Vertical planes:  {len(vertical_planes)}")
    print(f"  Architectural:    {len(architectural_candidates)}")
    print(f"  Curved elements:  {len(curved_elements)}")
    print(f"  MEP/Non-wall Rej: {mep_rejected_count}")
    print(f"  Physical walls:   {len(physical_wall_groups)}")
    print(f"  Duration:         {stage2_time:.3f} s")

    print("\n" + "=" * 50)
    print("STAGE 2 RESULT")
    print("=" * 50)
    print(f"Input points:                  {raw_count:,}")
    print(f"Stage 1 points:                {voxel_count:,}")
    print(f"RANSAC planes:                 {total_planes}")
    print(f"Vertical planes:               {len(vertical_planes)}")
    print(f"Architectural wall candidates: {len(architectural_candidates)}")
    print(f"Curved structural elements:    {len(curved_elements)}")
    print(f"MEP/Non-wall rejected:         {mep_rejected_count}")
    print(f"Total rejected:                {total_rejected}")
    print(f"Physical walls:                {len(physical_wall_groups)}\n")


    # Print compact wall table
    display_limit = min(20, len(physical_wall_groups))
    for idx in range(display_limit):
        wall = physical_wall_groups[idx]
        tags = wall.get("tags", {})
        length_m = tags.get("wall_length_mm", 0.0) / 1000.0
        mins, maxs = wall.get("mins", np.zeros(3)), wall.get("maxs", np.zeros(3))
        height_m = float(maxs[2] - mins[2])
        thickness_mm = tags.get("wall_thickness_mm", 200.0)
        paired = tags.get("wall_faces_paired", False)
        ax, ay = tags.get("wall_axis_x", 0.0), tags.get("wall_axis_y", 0.0)
        orient_deg = (math.degrees(math.atan2(ay, ax)) + 360.0) % 180.0
        conf = float(wall.get("confidence", 1.0))

        print(f"Wall {idx+1:02d}:")
        print(f"  length:      {length_m:.3f} m")
        print(f"  height:      {height_m:.3f} m")
        print(f"  thickness:   {thickness_mm:.1f} mm (paired: {paired})")
        print(f"  orientation: {orient_deg:.1f}°")
        print(f"  confidence:  {conf:.2f}")

    if len(physical_wall_groups) > display_limit:
        print(f"... and {len(physical_wall_groups) - display_limit} more physical wall facets (see report).")

    print("\n" + "=" * 50)
    if save_viz_dir:
        print(f"DEBUG OUTPUT: {save_viz_dir}")
    print("=" * 50)

    return {
        "raw_count": raw_count,
        "voxel_count": voxel_count,
        "total_planes": total_planes,
        "vertical_planes": len(vertical_planes),
        "architectural_candidates": len(architectural_candidates),
        "curved_elements": len(curved_elements),
        "mep_rejected": mep_rejected_count,
        "total_rejected": total_rejected,
        "physical_walls": len(physical_wall_groups),
        "stage1_time": stage1_time,
        "stage2_time": stage2_time,
        "viz_time": viz_time,
        "walls": physical_wall_groups,
    }


def main():
    parser = argparse.ArgumentParser(
        description="Scan-to-BIM Stage 2 Pipeline Demonstration & Audit Tool.",
    )
    parser.add_argument(
        "scan_path",
        nargs="?",
        default=None,
        help="Path to an input point cloud. If omitted, runs the synthetic benchmark.",
    )
    parser.add_argument(
        "--voxel-size",
        type=float,
        default=VOXEL_SIZE_M,
        help="Voxel downsample size in metres (default: 0.02 m)",
    )
    parser.add_argument(
        "--min-inliers",
        type=int,
        default=200,
        help="Minimum inliers for RANSAC plane detection (default: 200)",
    )
    parser.add_argument(
        "--save-viz",
        type=str,
        default="debug_output",
        help="Directory to save artifacts (default: 'debug_output')",
    )

    args = parser.parse_args()
    viz_dir = Path(args.save_viz)

    if args.scan_path is None:
        print("=" * 50)
        print("SCAN-TO-BIM STAGE 2")
        print("INPUT: SYNTHETIC BENCHMARK")
        print("==================================================")
        print("[NOTICE] No external real scan provided. Running synthetic benchmark.\n")
        raw_pts, _ = generate_synthetic_benchmark()
        run_pipeline(
            raw_points_m=raw_pts,
            source_name="SYNTHETIC BENCHMARK",
            detected_units="metres",
            voxel_size_m=args.voxel_size,
            min_inliers=args.min_inliers,
            save_viz_dir=viz_dir,
        )
    else:
        scan_path_obj = Path(args.scan_path)
        candidates = [
            scan_path_obj.resolve(),
            Path.cwd() / scan_path_obj,
            Path.cwd().parent / scan_path_obj,
            Path.cwd().parent.parent / scan_path_obj,
        ]
        if "STB_SCAN_DIR" in os.environ:
            candidates.insert(0, Path(os.environ["STB_SCAN_DIR"]) / scan_path_obj.name)
        scan_file = None
        for c in candidates:
            if c.exists() and c.is_file():
                scan_file = c.resolve()
                break

        if scan_file is None:
            print("=" * 50)
            print("ERROR: INPUT FILE NOT FOUND")
            print(f"Specified scan file does not exist: {args.scan_path}")
            print("=" * 50)
            sys.exit(1)

        print("=" * 50)
        print("SCAN-TO-BIM STAGE 2")
        print("INPUT: EXTERNAL REAL SCAN")
        print("==================================================")
        print(f"Source file: {scan_file.name}")
        print(f"File size:   {scan_file.stat().st_size / (1024*1024):.2f} MB\n")

        raw_pts, detected_units = load_input_points(scan_file)
        run_pipeline(
            raw_points_m=raw_pts,
            source_name=scan_file.name,
            detected_units=detected_units,
            voxel_size_m=args.voxel_size,
            min_inliers=args.min_inliers,
            save_viz_dir=viz_dir,
        )


if __name__ == "__main__":
    main()
