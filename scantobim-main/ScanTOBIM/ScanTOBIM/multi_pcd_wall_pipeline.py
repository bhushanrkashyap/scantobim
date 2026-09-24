from __future__ import annotations

import argparse
import json
import math
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Iterable

import numpy as np
import open3d as o3d
from sklearn.cluster import DBSCAN


@dataclass
class PipelineConfig:
    voxel_size: float = 0.03
    sor_nb_neighbors: int = 20
    sor_std_ratio: float = 1.8
    ror_nb_points: int = 8
    ror_radius: float = 0.08
    normal_radius: float = 0.12
    normal_max_nn: int = 40

    ransac_distance_threshold: float = 0.03
    ransac_n: int = 3
    ransac_num_iterations: int = 3000
    ransac_min_inliers: int = 180
    ransac_max_planes: int = 500
    min_remaining_points: int = 500

    # Vertical walls: |n dot z| close to 0.
    max_abs_nz_for_vertical: float = 0.25
    min_plane_area_m2: float = 0.8

    dbscan_eps: float = 0.10
    dbscan_min_samples: int = 25
    dbscan_min_cluster_points: int = 120

    merge_angle_deg: float = 10.0
    merge_offset_m: float = 0.20
    merge_gap_m: float = 0.60
    merge_z_overlap_m: float = 0.30


@dataclass
class WallSegment:
    segment_id: str
    source_file: str
    plane_eq: list[float]  # a,b,c,d in ax+by+cz+d=0
    points: np.ndarray
    centroid: np.ndarray
    length_m: float
    height_m: float
    thickness_m: float
    confidence: float


@dataclass
class MergedWall:
    wall_id: str
    plane_eq: list[float]
    points: np.ndarray
    source_segments: list[str]
    source_files: list[str]
    centroid: np.ndarray
    length_m: float
    height_m: float
    thickness_m: float
    confidence: float


def load_pcd_files(folder: Path) -> list[Path]:
    files = sorted(folder.glob("*.pcd"))
    if not files:
        raise FileNotFoundError(f"No .pcd files found in: {folder}")
    return files


def preprocess_cloud(pcd: o3d.geometry.PointCloud, cfg: PipelineConfig) -> o3d.geometry.PointCloud:
    pcd = pcd.voxel_down_sample(cfg.voxel_size)

    if len(pcd.points) == 0:
        return pcd

    pcd, _ = pcd.remove_statistical_outlier(
        nb_neighbors=cfg.sor_nb_neighbors,
        std_ratio=cfg.sor_std_ratio,
    )

    if len(pcd.points) == 0:
        return pcd

    pcd, _ = pcd.remove_radius_outlier(
        nb_points=cfg.ror_nb_points,
        radius=cfg.ror_radius,
    )

    if len(pcd.points) == 0:
        return pcd

    pcd.estimate_normals(
        search_param=o3d.geometry.KDTreeSearchParamHybrid(
            radius=cfg.normal_radius,
            max_nn=cfg.normal_max_nn,
        )
    )
    pcd.orient_normals_consistent_tangent_plane(20)
    return pcd


def build_vertical_plane_basis(normal: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    z_axis = np.array([0.0, 0.0, 1.0], dtype=float)
    u = np.cross(z_axis, normal)
    nu = np.linalg.norm(u)
    if nu < 1e-8:
        # Defensive fallback, should not happen for vertical planes.
        u = np.array([1.0, 0.0, 0.0], dtype=float)
    else:
        u /= nu
    v = z_axis
    return u, v


def fit_plane_svd(points: np.ndarray) -> np.ndarray:
    centroid = points.mean(axis=0)
    centered = points - centroid
    _, _, vh = np.linalg.svd(centered, full_matrices=False)
    normal = vh[-1]
    normal = normal / (np.linalg.norm(normal) + 1e-12)
    d = -float(np.dot(normal, centroid))
    return np.array([normal[0], normal[1], normal[2], d], dtype=float)


def project_dims_for_vertical_wall(points: np.ndarray, normal: np.ndarray) -> tuple[float, float, float]:
    u, _ = build_vertical_plane_basis(normal)
    proj_u = points @ u
    length_m = float(proj_u.max() - proj_u.min())

    z_vals = points[:, 2]
    height_m = float(z_vals.max() - z_vals.min())

    proj_n = points @ normal
    thickness_m = float(proj_n.max() - proj_n.min())
    return length_m, height_m, thickness_m


def estimate_plane_area(points: np.ndarray, normal: np.ndarray) -> float:
    # Recall-first rectangular extent estimate in plane coordinates.
    u, v = build_vertical_plane_basis(normal)
    pu = points @ u
    pv = points @ v
    return float((pu.max() - pu.min()) * (pv.max() - pv.min()))


def confidence_score(
    inlier_count: int,
    area_m2: float,
    abs_nz: float,
    residuals: np.ndarray,
) -> float:
    inlier_term = min(1.0, inlier_count / 1800.0)
    area_term = min(1.0, area_m2 / 12.0)
    vertical_term = max(0.0, 1.0 - (abs_nz / 0.30))
    rmse = float(np.sqrt(np.mean(residuals**2))) if residuals.size else 0.05
    fit_term = max(0.0, min(1.0, 1.0 - rmse / 0.06))
    return float(np.clip(0.35 * inlier_term + 0.25 * area_term + 0.20 * vertical_term + 0.20 * fit_term, 0.0, 1.0))


def extract_wall_segments_from_cloud(
    pcd: o3d.geometry.PointCloud,
    source_name: str,
    cfg: PipelineConfig,
) -> list[WallSegment]:
    segments: list[WallSegment] = []
    working = pcd
    seg_idx = 0

    while seg_idx < cfg.ransac_max_planes and len(working.points) >= cfg.min_remaining_points:
        plane_model, inliers = working.segment_plane(
            distance_threshold=cfg.ransac_distance_threshold,
            ransac_n=cfg.ransac_n,
            num_iterations=cfg.ransac_num_iterations,
        )

        if len(inliers) < cfg.ransac_min_inliers:
            break

        inlier_cloud = working.select_by_index(inliers)
        inlier_points = np.asarray(inlier_cloud.points)
        if inlier_points.shape[0] < cfg.ransac_min_inliers:
            working = working.select_by_index(inliers, invert=True)
            continue

        normal = np.array(plane_model[:3], dtype=float)
        n_norm = np.linalg.norm(normal)
        if n_norm < 1e-8:
            working = working.select_by_index(inliers, invert=True)
            continue
        normal /= n_norm
        d = float(plane_model[3]) / n_norm
        abs_nz = abs(float(normal[2]))

        if abs_nz > cfg.max_abs_nz_for_vertical:
            # Not a vertical candidate wall.
            working = working.select_by_index(inliers, invert=True)
            continue

        area = estimate_plane_area(inlier_points, normal)
        if area < cfg.min_plane_area_m2:
            working = working.select_by_index(inliers, invert=True)
            continue

        labels = DBSCAN(eps=cfg.dbscan_eps, min_samples=cfg.dbscan_min_samples).fit_predict(inlier_points)
        unique_labels = [lb for lb in np.unique(labels) if lb != -1]

        for lb in unique_labels:
            c_pts = inlier_points[labels == lb]
            if c_pts.shape[0] < cfg.dbscan_min_cluster_points:
                continue

            # Refit per cluster for better local plane and dimensions.
            c_plane = fit_plane_svd(c_pts)
            c_normal = c_plane[:3]
            c_normal /= np.linalg.norm(c_normal) + 1e-12

            if abs(float(c_normal[2])) > cfg.max_abs_nz_for_vertical:
                continue

            c_area = estimate_plane_area(c_pts, c_normal)
            if c_area < cfg.min_plane_area_m2:
                continue

            residuals = np.abs((c_pts @ c_normal) + c_plane[3])
            length_m, height_m, thickness_m = project_dims_for_vertical_wall(c_pts, c_normal)
            conf = confidence_score(c_pts.shape[0], c_area, abs(float(c_normal[2])), residuals)

            segment = WallSegment(
                segment_id=f"{source_name}_wall_{seg_idx:04d}_{int(lb):02d}",
                source_file=source_name,
                plane_eq=[float(c_plane[0]), float(c_plane[1]), float(c_plane[2]), float(c_plane[3])],
                points=c_pts,
                centroid=c_pts.mean(axis=0),
                length_m=length_m,
                height_m=height_m,
                thickness_m=thickness_m,
                confidence=conf,
            )
            segments.append(segment)

        # Always remove the fitted plane inliers and keep searching.
        working = working.select_by_index(inliers, invert=True)
        seg_idx += 1

    return segments


def plane_angle_and_offset_xy(plane_eq: list[float]) -> tuple[float, float, np.ndarray, np.ndarray]:
    a, b, c, d = plane_eq
    n_xy = np.array([a, b], dtype=float)
    nn = np.linalg.norm(n_xy)
    if nn < 1e-12:
        n_xy = np.array([1.0, 0.0], dtype=float)
        nn = 1.0
    n_xy = n_xy / nn
    # Canonical orientation so opposite normals map to same wall direction.
    if (n_xy[0] < 0) or (abs(n_xy[0]) < 1e-9 and n_xy[1] < 0):
        n_xy = -n_xy
        d = -d

    angle = math.degrees(math.atan2(float(n_xy[1]), float(n_xy[0])))
    rho = -float(d) / nn
    axis_xy = np.array([-n_xy[1], n_xy[0]], dtype=float)
    return angle, rho, n_xy, axis_xy


def projection_interval(points: np.ndarray, axis_xy: np.ndarray) -> tuple[float, float]:
    vals = points[:, :2] @ axis_xy
    return float(vals.min()), float(vals.max())


def z_interval(points: np.ndarray) -> tuple[float, float]:
    z = points[:, 2]
    return float(z.min()), float(z.max())


def intervals_overlap(a: tuple[float, float], b: tuple[float, float], slack: float) -> bool:
    return not (a[1] + slack < b[0] or b[1] + slack < a[0])


def can_merge_segments(seg_a: WallSegment, seg_b: WallSegment, cfg: PipelineConfig) -> bool:
    angle_a, rho_a, _, axis_a = plane_angle_and_offset_xy(seg_a.plane_eq)
    angle_b, rho_b, _, _ = plane_angle_and_offset_xy(seg_b.plane_eq)

    d_angle = abs(angle_a - angle_b)
    d_angle = min(d_angle, 180.0 - d_angle)
    if d_angle > cfg.merge_angle_deg:
        return False

    if abs(rho_a - rho_b) > cfg.merge_offset_m:
        return False

    int_a = projection_interval(seg_a.points, axis_a)
    int_b = projection_interval(seg_b.points, axis_a)
    if not intervals_overlap(int_a, int_b, cfg.merge_gap_m):
        return False

    z_a = z_interval(seg_a.points)
    z_b = z_interval(seg_b.points)
    if not intervals_overlap(z_a, z_b, cfg.merge_z_overlap_m):
        return False

    return True


def merge_segments_into_walls(segments: list[WallSegment], cfg: PipelineConfig) -> list[MergedWall]:
    groups: list[list[WallSegment]] = []

    for seg in segments:
        assigned = False
        for g in groups:
            if any(can_merge_segments(seg, member, cfg) for member in g):
                g.append(seg)
                assigned = True
                break
        if not assigned:
            groups.append([seg])

    merged: list[MergedWall] = []
    for i, group in enumerate(groups):
        pts = np.vstack([s.points for s in group])
        plane = fit_plane_svd(pts)
        normal = plane[:3]
        normal /= np.linalg.norm(normal) + 1e-12

        # Force vertical interpretation during metrics.
        normal[2] = 0.0 if abs(normal[2]) < 0.05 else normal[2]
        normal /= np.linalg.norm(normal) + 1e-12
        plane[:3] = normal
        plane[3] = -float(np.dot(normal, pts.mean(axis=0)))

        length_m, height_m, thickness_m = project_dims_for_vertical_wall(pts, normal)
        area_m2 = estimate_plane_area(pts, normal)
        residuals = np.abs((pts @ normal) + plane[3])
        confs = [s.confidence for s in group]
        conf = max(float(np.mean(confs)), confidence_score(len(pts), area_m2, abs(float(normal[2])), residuals))

        merged.append(
            MergedWall(
                wall_id=f"wall_{i:04d}",
                plane_eq=[float(plane[0]), float(plane[1]), float(plane[2]), float(plane[3])],
                points=pts,
                source_segments=[s.segment_id for s in group],
                source_files=sorted(set(s.source_file for s in group)),
                centroid=pts.mean(axis=0),
                length_m=length_m,
                height_m=height_m,
                thickness_m=thickness_m,
                confidence=float(np.clip(conf, 0.0, 1.0)),
            )
        )

    return merged


def points_to_pcd(points: np.ndarray, color: np.ndarray | None = None) -> o3d.geometry.PointCloud:
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)
    if color is not None:
        colors = np.repeat(color.reshape(1, 3), points.shape[0], axis=0)
        pcd.colors = o3d.utility.Vector3dVector(colors)
    return pcd


def color_palette(n: int) -> list[np.ndarray]:
    rng = np.random.default_rng(42)
    return [rng.uniform(0.1, 0.95, size=3) for _ in range(n)]


def save_outputs(walls: list[MergedWall], output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    per_wall_dir = output_dir / "walls"
    per_wall_dir.mkdir(parents=True, exist_ok=True)

    combined_points: list[np.ndarray] = []
    combined_colors: list[np.ndarray] = []
    colors = color_palette(len(walls))

    plane_records = []
    report_records = []

    for idx, wall in enumerate(walls):
        wall_pcd = points_to_pcd(wall.points)
        wall_file = per_wall_dir / f"{wall.wall_id}.pcd"
        o3d.io.write_point_cloud(str(wall_file), wall_pcd, write_ascii=False, compressed=False)

        combined_points.append(wall.points)
        combined_colors.append(np.repeat(colors[idx].reshape(1, 3), wall.points.shape[0], axis=0))

        plane_records.append(
            {
                "wall_id": wall.wall_id,
                "plane_eq": wall.plane_eq,
                "source_segments": wall.source_segments,
                "source_files": wall.source_files,
            }
        )

        report_records.append(
            {
                "wall_id": wall.wall_id,
                "num_points": int(wall.points.shape[0]),
                "length_m": round(wall.length_m, 3),
                "height_m": round(wall.height_m, 3),
                "thickness_m": round(wall.thickness_m, 3),
                "confidence": round(wall.confidence, 4),
                "centroid": [round(float(v), 4) for v in wall.centroid.tolist()],
                "num_source_scans": len(wall.source_files),
            }
        )

    if combined_points:
        all_points = np.vstack(combined_points)
        all_colors = np.vstack(combined_colors)
        combined_pcd = points_to_pcd(all_points)
        combined_pcd.colors = o3d.utility.Vector3dVector(all_colors)
        o3d.io.write_point_cloud(str(output_dir / "walls_combined.pcd"), combined_pcd, write_ascii=False, compressed=False)
        o3d.io.write_point_cloud(str(output_dir / "walls_colored.ply"), combined_pcd, write_ascii=False, compressed=False)

    (output_dir / "wall_planes.json").write_text(json.dumps(plane_records, indent=2), encoding="utf-8")
    summary = {
        "num_detected_walls": len(walls),
        "walls": report_records,
    }
    (output_dir / "wall_report.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")


def print_summary(walls: list[MergedWall]) -> None:
    print(f"Detected walls: {len(walls)}")
    for wall in walls:
        print(
            f"{wall.wall_id}: length={wall.length_m:.2f}m "
            f"height={wall.height_m:.2f}m confidence={wall.confidence:.3f} "
            f"scans={len(wall.source_files)}"
        )


def run_pipeline(input_dir: Path, output_dir: Path, cfg: PipelineConfig) -> list[MergedWall]:
    files = load_pcd_files(input_dir)
    all_segments: list[WallSegment] = []

    for pcd_file in files:
        pcd = o3d.io.read_point_cloud(str(pcd_file))
        if len(pcd.points) == 0:
            print(f"[WARN] Empty cloud skipped: {pcd_file.name}")
            continue

        pre = preprocess_cloud(pcd, cfg)
        if len(pre.points) == 0:
            print(f"[WARN] Cloud empty after preprocessing: {pcd_file.name}")
            continue

        segs = extract_wall_segments_from_cloud(pre, pcd_file.name, cfg)
        print(f"[INFO] {pcd_file.name}: wall segments={len(segs)}")
        all_segments.extend(segs)

    merged = merge_segments_into_walls(all_segments, cfg)
    save_outputs(merged, output_dir)
    print_summary(merged)
    return merged


def build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Multi-PCD wall detection with RANSAC + DBSCAN (recall-first)")
    p.add_argument("--input-dir", required=True, type=Path, help="Folder containing .pcd files")
    p.add_argument("--output-dir", required=True, type=Path, help="Output folder")

    p.add_argument("--voxel-size", type=float, default=0.03)
    p.add_argument("--ransac-dist", type=float, default=0.03)
    p.add_argument("--min-inliers", type=int, default=180)
    p.add_argument("--dbscan-eps", type=float, default=0.10)
    p.add_argument("--dbscan-min-samples", type=int, default=25)
    p.add_argument("--dbscan-min-cluster", type=int, default=120)
    p.add_argument("--max-abs-nz", type=float, default=0.25)
    p.add_argument("--min-plane-area", type=float, default=0.8)
    p.add_argument("--merge-angle-deg", type=float, default=10.0)
    p.add_argument("--merge-offset-m", type=float, default=0.20)
    p.add_argument("--merge-gap-m", type=float, default=0.60)
    return p


def config_from_args(args: argparse.Namespace) -> PipelineConfig:
    cfg = PipelineConfig()
    cfg.voxel_size = args.voxel_size
    cfg.ransac_distance_threshold = args.ransac_dist
    cfg.ransac_min_inliers = args.min_inliers
    cfg.dbscan_eps = args.dbscan_eps
    cfg.dbscan_min_samples = args.dbscan_min_samples
    cfg.dbscan_min_cluster_points = args.dbscan_min_cluster
    cfg.max_abs_nz_for_vertical = args.max_abs_nz
    cfg.min_plane_area_m2 = args.min_plane_area
    cfg.merge_angle_deg = args.merge_angle_deg
    cfg.merge_offset_m = args.merge_offset_m
    cfg.merge_gap_m = args.merge_gap_m
    return cfg


def main() -> None:
    args = build_arg_parser().parse_args()
    cfg = config_from_args(args)
    run_pipeline(args.input_dir, args.output_dir, cfg)


if __name__ == "__main__":
    main()
