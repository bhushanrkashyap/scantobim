"""Advanced geometry algorithms for Scan-to-BIM segmentation pipeline.

Implements:
  A3  — Surface normal estimation (Open3D KD-tree hybrid)
  B2  — RANSAC cylinder fitting (custom — Open3D has no segment_cylinder)
  B3  — PCA with linearity / planarity / sphericity signatures
  B4  — Oriented bounding box (Open3D OBB)
  B5  — RANSAC sphere fitting (custom 4-point solver)
  C2  — Region growing (normal-angle flood fill via KDTreeFlann)
  C3  — Euclidean clustering (scipy KDTree radius search)
  D3  — Z-profile histogram stair-step counter
  E1  — Spatial index (scipy KDTree over segment centroids)
  E2  — MEP connectivity graph (networkx)
"""

from __future__ import annotations

import numpy as np
import structlog

logger = structlog.get_logger()


# ── Geometry Validation ──────────────────────────────────────────────────────


def validate_geometry_segment(segment: dict) -> tuple[bool, str | None]:
    """Validate that a geometry segment meets minimum quality standards.

    Returns:
        (is_valid, error_message) tuple. If valid, error_message is None.
    """
    try:
        mins = np.asarray(segment.get("mins"), dtype=float)
        maxs = np.asarray(segment.get("maxs"), dtype=float)
        centroid = np.asarray(segment.get("centroid"), dtype=float)

        if len(mins) != 3 or len(maxs) != 3 or len(centroid) != 3:
            return False, "Invalid coordinate dimensions"

        # Check bounding box validity
        if np.any(maxs < mins):
            return False, f"Invalid bounding box: max < min (max={maxs}, min={mins})"

        extents_mm = maxs - mins

        # ── Degenerate box detection ──────────────────────────────────────────
        # A segment with any dimension < 10 mm is too small to be useful
        # (sub-centimeter features are noise or measurement error)
        MIN_DIMENSION_MM = 10.0
        if np.any(extents_mm < MIN_DIMENSION_MM):
            return False, (
                f"Degenerate bounding box: dimensions {extents_mm} mm "
                f"(minimum {MIN_DIMENSION_MM} mm)"
            )

        # Check that centroid is actually inside the bounding box
        if np.any(centroid < mins - 1.0) or np.any(centroid > maxs + 1.0):
            return False, (f"Centroid {centroid} outside bounding box [{mins}, {maxs}]")

        # Check for NaN or Inf values
        if np.any(~np.isfinite(extents_mm)) or np.any(~np.isfinite(centroid)):
            return False, "Contains NaN or Inf values"

        # Valid segment
        return True, None

    except Exception as e:
        return False, f"Validation exception: {e!s}"


def filter_valid_segments(segments: list[dict]) -> tuple[list[dict], list[tuple[int, str]]]:
    """Filter out degenerate/invalid segments from a list.

    Returns:
        (valid_segments, rejected_segments_with_reasons)
        where rejected_segments_with_reasons is a list of (index, reason) tuples.
    """
    valid = []
    rejected = []

    for idx, seg in enumerate(segments):
        is_valid, error_msg = validate_geometry_segment(seg)
        if is_valid:
            valid.append(seg)
        else:
            rejected.append((idx, error_msg))
            logger.warning(
                "segment_rejected_invalid_geometry",
                segment_index=idx,
                shape=seg.get("shape"),
                reason=error_msg,
            )

    if rejected:
        logger.info(
            "geometry_validation_results",
            input_segments=len(segments),
            valid_segments=len(valid),
            rejected_segments=len(rejected),
        )

    return valid, rejected


# ── A3: Surface Normal Estimation ────────────────────────────────────────────


def estimate_normals(
    pcd: open3d.geometry.PointCloud,
    radius: float = 0.05,
    max_nn: int = 30,
    adaptive: bool = False,
    max_retries: int = 3,
    voxel_start: float = 0.01,
    voxel_max: float = 0.10,
    preserve_original: bool = True,
) -> dict:
    """
    A3 - Surface Normal Estimation

    Expected pipeline:

        A2 SOR
            ↓
        A1 Voxel Downsample
            ↓
        A3 Surface Normal Estimation

    Returns:
        {
            "pcd": PointCloud with normals,
            "original_points": ndarray,
            "normalized_points": ndarray,
            "origin": ndarray,
            "success": bool,
            "error": str | None,
        }
    """

    import copy

    import numpy as np
    import open3d as o3d
    from scipy.spatial import QhullError

    result = {
        "pcd": None,
        "original_points": None,
        "normalized_points": None,
        "origin": None,
        "success": False,
        "error": None,
    }

    try:
        pts = np.asarray(pcd.points)

        if len(pts) == 0:
            result["error"] = "Point cloud is empty."
            return result

        # ---------------------------------------------------------
        # Preserve original coordinates (optional)
        # ---------------------------------------------------------
        if preserve_original:
            result["original_points"] = pts.copy()

        # ---------------------------------------------------------
        # Coordinate normalization
        # ---------------------------------------------------------
        origin = pts.mean(axis=0)
        pts_norm = pts - origin

        result["origin"] = origin
        result["normalized_points"] = pts_norm.copy()

        x_flat = np.allclose(pts_norm[:, 0], 0)
        y_flat = np.allclose(pts_norm[:, 1], 0)

        if x_flat or y_flat:
            logger.error(
                "coordinate_flattening_detected",
                x_flat=x_flat,
                y_flat=y_flat,
            )
        else:
            logger.info(
                "coordinate_normalization_valid",
                x_range=(
                    float(pts_norm[:, 0].min()),
                    float(pts_norm[:, 0].max()),
                ),
                y_range=(
                    float(pts_norm[:, 1].min()),
                    float(pts_norm[:, 1].max()),
                ),
            )

        # ---------------------------------------------------------
        # Create working cloud without deep-copying the original
        # ---------------------------------------------------------
        pcd_proc = o3d.geometry.PointCloud()

        pcd_proc.points = o3d.utility.Vector3dVector(pts_norm)

        if pcd.has_colors():
            pcd_proc.colors = pcd.colors

        voxel_size = voxel_start
        retries = 0

        while retries <= max_retries and voxel_size <= voxel_max:
            try:
                if adaptive and voxel_size > 0:
                    pcd_work = copy.deepcopy(pcd_proc)

                    pcd_work = pcd_work.voxel_down_sample(voxel_size=voxel_size)

                    logger.info(
                        "adaptive_voxelization",
                        voxel_size=voxel_size,
                        points=len(pcd_work.points),
                    )

                else:
                    pcd_work = pcd_proc

                logger.info(
                    "starting_normal_estimation",
                    points=len(pcd_work.points),
                    radius=radius,
                    max_nn=max_nn,
                )

                pcd_work.estimate_normals(
                    search_param=o3d.geometry.KDTreeSearchParamHybrid(
                        radius=radius,
                        max_nn=max_nn,
                    )
                )

                logger.info(
                    "finished_normal_estimation",
                    points=len(pcd_work.points),
                )

                pcd_work.orient_normals_consistent_tangent_plane(k=15)

                logger.info(
                    "normals_estimated",
                    points=len(pcd_work.points),
                    radius=radius,
                    max_nn=max_nn,
                    voxel_size=voxel_size,
                )

                result["pcd"] = pcd_work
                result["success"] = True

                return result

            except Exception as exc:
                if "QH6347" in str(exc) or isinstance(exc, QhullError):
                    logger.warning(
                        "qhull_precision_failure",
                        error=str(exc),
                        voxel_size=voxel_size,
                    )

                    retries += 1
                    voxel_size *= 2

                else:
                    logger.error(
                        "normal_estimation_failed",
                        error=str(exc),
                    )

                    result["error"] = str(exc)
                    return result

        result["error"] = "QHull precision failure after maximum retries"

        return result

    except Exception as exc:
        logger.error(
            "estimate_normals_unexpected_failure",
            error=str(exc),
        )
        result["error"] = str(exc)

        return result


# ── B3: Principal Component Analysis ─────────────────────────────────────────


def pca_fit(pts: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Full PCA via eigendecomposition of the covariance matrix.

    Returns:
        eigenvalues:  shape (3,) — sorted descending (largest variance first)
        eigenvectors: shape (3, 3) — columns are principal axes (same order)
        centroid:     shape (3,) — mean of pts
    """
    centroid = pts.mean(axis=0)
    cov = np.cov((pts - centroid).T)
    eigenvalues, eigenvectors = np.linalg.eigh(cov)
    # eigh returns ascending order — reverse to descending
    idx = np.argsort(eigenvalues)[::-1]
    return eigenvalues[idx], eigenvectors[:, idx], centroid


def pca_linearity(ev: np.ndarray) -> float:
    """(λ₁ − λ₂) / λ₁ — high → elongated (beam, pipe)."""
    return float((ev[0] - ev[1]) / max(ev[0], 1e-9))


def pca_planarity(ev: np.ndarray) -> float:
    """(λ₂ − λ₃) / λ₁ — high → flat plane (wall, floor)."""
    return float((ev[1] - ev[2]) / max(ev[0], 1e-9))


def pca_sphericity(ev: np.ndarray) -> float:
    """λ₃ / λ₁ — high → compact sphere-like (valve body, sprinkler)."""
    return float(ev[2] / max(ev[0], 1e-9))


# ── PCA Wall Thickness Refinement ─────────────────────────────────────────────
# Adapted from Cloud2BIM walls/pca_wall_fitting.py::identify_wall_faces_and_refine_thickness


def refine_wall_thickness_pca(
    inlier_pts: np.ndarray,
    wall_normal: np.ndarray,
    wall_axis: np.ndarray | None = None,
    min_thickness_mm: float = 100.0,
    max_thickness_mm: float = 600.0,
    default_thickness_mm: float = 200.0,
    resolution_m: float = 0.02,
) -> tuple[float, str]:
    """Estimate true wall thickness by projecting inlier points onto the wall normal.

    Cloud2BIM method: rotate points into wall-local frame where:
      - X = along wall axis (run direction)
      - Y = perpendicular to wall (thickness direction = normal)
      - Z = vertical

    Then analyse the Y-distribution to detect opposing face peaks.

    Returns:
        (thickness_mm, mode) where mode is one of:
          "measured_dual_peak"    — two face peaks detected
          "measured_percentile"   — single face, 10th-90th percentile spread
          "default_fallback"      — insufficient points or degenerate geometry
    """
    if inlier_pts is None or len(inlier_pts) < 15:
        return default_thickness_mm, "default_fallback"

    # Ensure wall_normal is 2D horizontal (we project XY only)
    normal_2d = np.array([wall_normal[0], wall_normal[1]], dtype=float)
    norm_n = np.linalg.norm(normal_2d)
    if norm_n < 1e-6:
        return default_thickness_mm, "default_fallback"
    normal_2d = normal_2d / norm_n

    # Project all inlier points onto the wall normal direction (thickness axis)
    pts_xy = inlier_pts[:, :2]
    proj_thickness = pts_xy @ normal_2d  # 1D projection onto normal

    y_min = float(proj_thickness.min())
    y_max = float(proj_thickness.max())
    spread_m = y_max - y_min

    if spread_m < resolution_m:
        # All points on a single surface — use default
        return default_thickness_mm, "default_fallback"

    # Build histogram of the projected distribution
    bins = np.arange(y_min, y_max + resolution_m, resolution_m)
    if len(bins) < 3:
        return default_thickness_mm, "default_fallback"

    hist, edges = np.histogram(proj_thickness, bins=bins)

    if hist.max() > 0:
        try:
            from scipy.signal import find_peaks

            thresh = 0.25 * hist.max()
            min_dist_bins = max(1, int(0.08 / resolution_m))
            peaks, _ = find_peaks(hist, height=thresh, distance=min_dist_bins)
        except ImportError:
            peaks = np.array([], dtype=int)
    else:
        peaks = np.array([], dtype=int)

    if len(peaks) >= 2:
        # Dual-face detection: pick the two most prominent peaks
        top2 = peaks[np.argsort(hist[peaks])[-2:]]
        y1 = float(0.5 * (edges[top2[0]] + edges[top2[0] + 1]))
        y2 = float(0.5 * (edges[top2[1]] + edges[top2[1] + 1]))
        thickness_m = abs(y2 - y1)
        thickness_mm = float(np.clip(thickness_m * 1000.0, min_thickness_mm, max_thickness_mm))
        logger.debug(
            "wall_thickness_dual_peak",
            thickness_mm=round(thickness_mm, 1),
            face1_m=round(min(y1, y2), 3),
            face2_m=round(max(y1, y2), 3),
        )
        return round(thickness_mm, 1), "measured_dual_peak"

    # Single-face fallback: use 10th-90th percentile spread
    p10 = float(np.percentile(proj_thickness, 10))
    p90 = float(np.percentile(proj_thickness, 90))
    thickness_m = max(0.0, p90 - p10)
    thickness_mm = thickness_m * 1000.0

    if thickness_mm < min_thickness_mm:
        # Spread too narrow — single surface scan, use default
        return default_thickness_mm, "default_fallback"

    thickness_mm = float(np.clip(thickness_mm, min_thickness_mm, max_thickness_mm))
    logger.debug(
        "wall_thickness_percentile",
        thickness_mm=round(thickness_mm, 1),
        p10_m=round(p10, 3),
        p90_m=round(p90, 3),
    )
    return round(thickness_mm, 1), "measured_percentile"


# ── B4: Oriented Bounding Box ─────────────────────────────────────────────────


def get_obb(pts: np.ndarray) -> dict:
    """Compute oriented bounding box using Open3D.

    Returns dict with:
        extent:   (3,) numpy array — half-extents along OBB axes, sorted descending
        rotation: (3, 3) numpy array — rotation matrix from OBB local to world
        center:   (3,) numpy array — OBB center in world coordinates
    """
    import open3d as o3d

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(pts)
    obb = pcd.get_oriented_bounding_box()
    extent = np.asarray(obb.extent)
    return {
        "extent": np.sort(np.abs(extent))[::-1],  # descending
        "rotation": np.asarray(obb.R),
        "center": np.asarray(obb.center),
    }


# ── B2: RANSAC Cylinder Fitting ───────────────────────────────────────────────


def ransac_cylinder_fit(
    pts: np.ndarray,
    normals: np.ndarray,
    threshold: float = 0.02,
    max_iterations: int = 1000,
    min_inlier_ratio: float = 0.40,
) -> dict | None:
    """Fit a cylinder to a point cluster using RANSAC.

    Algorithm per iteration:
      1. Sample 2 points p1, p2.
      2. Axis direction = cross(n1, n2), normalised.
      3. Project all points onto the plane perpendicular to axis.
      4. Project p1 onto that plane → candidate centre.
      5. Radius = distance from candidate centre to projected p1.
      6. Count inliers where |distance_to_axis − radius| < threshold.

    Returns None if insufficient inliers.

    Returns dict with:
        axis:     unit vector (3,) — cylinder axis direction
        center:   point on axis (3,) — closest to cloud centroid
        radius:   float — metres
        inliers:  int   — number of inlier points
    """
    if len(pts) < 10 or normals is None or len(normals) < 10:
        return None

    n = len(pts)
    best_inliers = 0
    best_result: dict | None = None
    # NQA-1 determinism: local fixed seed so this RANSAC fit is reproducible
    # across runs regardless of the pipeline-level seed. Different cylinder
    # clusters still get independent sampling because the input points differ.
    rng = np.random.default_rng(0)

    for _ in range(max_iterations):
        idx = rng.choice(n, size=2, replace=False)
        p1, p2 = pts[idx[0]], pts[idx[1]]
        n1, n2 = normals[idx[0]], normals[idx[1]]

        axis = np.cross(n1, n2)
        axis_norm = np.linalg.norm(axis)
        if axis_norm < 1e-6:
            continue
        axis /= axis_norm

        # Project all points onto plane perpendicular to axis through p1
        proj = pts - np.outer(pts.dot(axis), axis)
        p1_proj = p1 - axis * (p1.dot(axis))

        # Use p1_proj as candidate centre, compute radius as its distance to p2_proj
        p2_proj = p2 - axis * (p2.dot(axis))
        radius = float(np.linalg.norm(p1_proj - p2_proj))
        if radius < 0.005:  # < 5mm — degenerate
            continue

        # Distance from each projected point to candidate centre p1_proj
        diffs = proj - p1_proj
        dist_to_axis = np.linalg.norm(diffs, axis=1)
        inlier_mask = np.abs(dist_to_axis - radius) < threshold
        inlier_count = int(inlier_mask.sum())

        if inlier_count > best_inliers:
            best_inliers = inlier_count
            # Refine centre as mean of inlier projections
            inlier_proj = proj[inlier_mask]
            centre_proj = inlier_proj.mean(axis=0)
            best_result = {
                "axis": axis.copy(),
                "center": centre_proj,
                "radius": float(np.median(np.linalg.norm(inlier_proj - centre_proj, axis=1))),
                "inliers": inlier_count,
            }

    if best_result is None or best_inliers < int(n * min_inlier_ratio):
        return None

    logger.debug(
        "ransac_cylinder",
        inliers=best_inliers,
        total=n,
        radius_m=round(best_result["radius"], 3),
    )
    return best_result


# ── B5: RANSAC Sphere Fitting ─────────────────────────────────────────────────


def ransac_sphere_fit(
    pts: np.ndarray,
    threshold: float = 0.015,
    max_iterations: int = 500,
    min_inlier_ratio: float = 0.35,
) -> dict | None:
    """Fit a sphere using RANSAC with a 4-point algebraic solver.

    Algebraic form: ‖p − c‖² = r²  →  linear system in (cx, cy, cz, r²−‖c‖²).
    Sampling 4 non-collinear points gives an exact solution.

    Returns dict with:
        center:  (3,) float — sphere centre
        radius:  float      — metres
        inliers: int        — number of inlier points
    """
    if len(pts) < 10:
        return None

    n = len(pts)
    best_inliers = 0
    best_result: dict | None = None
    # NQA-1 determinism — fixed seed per the same rationale as ransac_cylinder_fit
    rng = np.random.default_rng(0)

    for _ in range(max_iterations):
        idx = rng.choice(n, size=4, replace=False)
        spts = pts[idx]

        # Build 4×4 linear system: A @ [2cx, 2cy, 2cz, r²-||c||²]ᵀ = b
        # where each row is: [xi, yi, zi, 1] and b_i = xi² + yi² + zi²
        A = np.hstack([spts, np.ones((4, 1))])
        b = (spts**2).sum(axis=1)

        try:
            sol, _, rank, _ = np.linalg.lstsq(A, b, rcond=None)
        except np.linalg.LinAlgError:
            continue

        if rank < 4:
            continue

        cx, cy, cz = sol[0] / 2, sol[1] / 2, sol[2] / 2
        center = np.array([cx, cy, cz])
        r_sq = sol[3] + cx**2 + cy**2 + cz**2
        if r_sq <= 0:
            continue
        radius = float(np.sqrt(r_sq))
        if radius < 0.005 or radius > 2.0:  # 5 mm – 2 m range
            continue

        dist = np.linalg.norm(pts - center, axis=1)
        inlier_mask = np.abs(dist - radius) < threshold
        inlier_count = int(inlier_mask.sum())

        if inlier_count > best_inliers:
            best_inliers = inlier_count
            inlier_pts = pts[inlier_mask]
            refined_center = center if inlier_count == 0 else inlier_pts.mean(axis=0)
            refined_radius = float(np.median(np.linalg.norm(inlier_pts - refined_center, axis=1)))
            best_result = {
                "center": refined_center,
                "radius": refined_radius,
                "inliers": inlier_count,
            }

    if best_result is None or best_inliers < int(n * min_inlier_ratio):
        return None

    logger.debug(
        "ransac_sphere",
        inliers=best_inliers,
        total=n,
        radius_m=round(best_result["radius"], 3),
    )
    return best_result


# ── C2: Region Growing ────────────────────────────────────────────────────────


def region_growing(
    pcd: open3d.geometry.PointCloud,
    angle_threshold_deg: float = 15.0,
    radius: float = 0.05,
    min_cluster_size: int = 50,
    max_cluster_size: int = 100_000,
    color_weight: float = 0.2,
    color_threshold: float = 0.15,
) -> list[dict]:
    """
    RGB-aware region growing segmentation using geometry, normals, and color similarity.
    Returns a list of dicts with 'indices', 'dominant_color', 'color_variance', 'semantic_color'.
    """
    import open3d as o3d

    # Optional dependency: scikit-learn
    try:
        from sklearn.cluster import KMeans  # type: ignore
    except ImportError:
        KMeans = None

    pts = np.asarray(pcd.points)
    n = len(pts)
    if n < min_cluster_size:
        return []

    # Region growing depends on normal-angle continuity. If normals are missing
    # (or stale/invalid), estimate them in-place instead of skipping.
    normals_valid = False
    if pcd.has_normals():
        normals = np.asarray(pcd.normals)
        normals_valid = len(normals) == n and np.isfinite(normals).all()

    if not normals_valid:
        try:
            # Scale radius gently with region-growing radius and cloud size.
            normal_radius = max(radius * 1.5, 0.08)
            normal_max_nn = 20 if n < 5_000 else 30
            pcd.estimate_normals(
                o3d.geometry.KDTreeSearchParamHybrid(
                    radius=normal_radius,
                    max_nn=normal_max_nn,
                )
            )
            # Best-effort orientation to improve local consistency.
            pcd.orient_normals_consistent_tangent_plane(k=10 if n < 5_000 else 15)
            normals = np.asarray(pcd.normals)
            normals_valid = len(normals) == n and np.isfinite(normals).all()
            if normals_valid:
                logger.info(
                    "region_growing_normals_estimated",
                    points=n,
                    radius=round(normal_radius, 3),
                    max_nn=normal_max_nn,
                )
        except Exception as exc:
            logger.warning("region_growing_normals_estimation_failed", error=str(exc), points=n)

    if not normals_valid:
        logger.warning("region_growing_skipped: normals unavailable after estimation", points=n)
        return []

    if pcd.has_colors():
        colors = np.asarray(pcd.colors)
    else:
        colors = np.zeros((n, 3))

    visited = np.zeros(n, dtype=bool)
    cos_thresh = float(np.cos(np.radians(angle_threshold_deg)))

    tree = o3d.geometry.KDTreeFlann(pcd)
    clusters: list[dict] = []

    for seed in range(n):
        if visited[seed]:
            continue

        region = [seed]
        visited[seed] = True
        queue = [seed]

        while queue:
            current = queue.pop()
            [_, neighbours, _] = tree.search_radius_vector_3d(pts[current], radius)
            for nb in neighbours:
                if visited[nb]:
                    continue
                # Accept neighbour if normal angle < threshold and color similarity
                cos_angle = abs(float(np.dot(normals[current], normals[nb])))
                color_dist = np.linalg.norm(colors[current] - colors[nb])
                if cos_angle >= cos_thresh and color_dist <= color_threshold:
                    visited[nb] = True
                    region.append(nb)
                    queue.append(nb)
                    if len(region) >= max_cluster_size:
                        queue.clear()
                        break

        if len(region) >= min_cluster_size:
            region_indices = np.array(region, dtype=np.intp)
            region_colors = colors[region_indices]
            # Dominant color: use KMeans if available, otherwise fall back to mean
            try:
                if KMeans is not None:
                    kmeans = KMeans(n_clusters=1, n_init=1, random_state=0).fit(region_colors)
                    dominant_color = kmeans.cluster_centers_[0]
                else:
                    dominant_color = region_colors.mean(axis=0)
            except Exception:
                dominant_color = region_colors.mean(axis=0)

            color_variance = float(np.mean(np.var(region_colors, axis=0)))
            # Semantic visualization color (e.g., map dominant color to a palette or use directly)
            semantic_color = dominant_color
            clusters.append(
                {
                    "indices": region_indices,
                    "dominant_color": dominant_color,
                    "color_variance": color_variance,
                    "semantic_color": semantic_color,
                }
            )

    logger.info("region_growing_done", clusters=len(clusters))
    return clusters


# ── C3: Euclidean Clustering ─────────────────────────────────────────────────


def euclidean_cluster(
    pcd: open3d.geometry.PointCloud,
    tolerance: float = 0.05,
    min_size: int = 50,
    max_size: int = 100_000,
) -> list[np.ndarray]:
    """Euclidean clustering via scipy KDTree radius search.

    Unlike DBSCAN, this has no density requirement — good for sparse
    isolated equipment clusters.

    Returns:
        List of index arrays, each one cluster.
    """
    from scipy.spatial import KDTree

    pts = np.asarray(pcd.points)
    n = len(pts)
    if n < min_size:
        return []

    tree = KDTree(pts)
    visited = np.zeros(n, dtype=bool)
    clusters: list[np.ndarray] = []

    for seed in range(n):
        if visited[seed]:
            continue

        cluster = []
        queue = [seed]
        visited[seed] = True

        while queue:
            current = queue.pop()
            neighbours = tree.query_ball_point(pts[current], r=tolerance)
            for nb in neighbours:
                if not visited[nb]:
                    visited[nb] = True
                    cluster.append(nb)
                    queue.append(nb)
                    if len(cluster) >= max_size:
                        queue.clear()
                        break

        if len(cluster) >= min_size:
            clusters.append(np.array(cluster, dtype=np.intp))

    logger.info("euclidean_cluster_done", clusters=len(clusters))
    return clusters


# ── D3: Z-Profile Histogram (Stair Step Counter) ─────────────────────────────


def count_stair_steps(
    inlier_pts: np.ndarray,
    min_steps: int = 2,
    max_steps: int = 30,
) -> int:
    """Count stair treads by analysing Z-coordinate histogram peaks.

    A stair will show discrete horizontal bands in its Z-profile.
    A ramp will show a smooth gradient with no pronounced peaks.

    Args:
        inlier_pts: (N, 3) array of inlier plane points in metres.
        min_steps:  Minimum step count before returning non-zero.
        max_steps:  Upper bound on realistic step count.

    Returns:
        Number of detected stair treads (0 if ramp-like or indeterminate).
    """
    z_vals = inlier_pts[:, 2]
    z_range = float(z_vals.max() - z_vals.min())
    if z_range < 0.12:  # < 12 cm rise — too shallow for a stair
        return 0

    # Histogram bin width tuned for typical riser bands (10-20 cm) while still
    # detecting shallow scans where risers are only partially observed.
    n_bins = max(24, min(int(z_range / 0.02), 220))
    hist, bin_edges = np.histogram(z_vals, bins=n_bins)
    if hist.max() <= 0:
        return 0

    # Smooth histogram to suppress quantization noise.
    kernel = np.array([1.0, 2.0, 3.0, 2.0, 1.0], dtype=float)
    kernel /= kernel.sum()
    hist_sm = np.convolve(hist.astype(float), kernel, mode="same")

    bin_h = z_range / max(n_bins, 1)
    min_peak_distance_bins = max(1, int(round(0.09 / max(bin_h, 1e-6))))
    prominence = max(2.0, float(hist_sm.max()) * 0.08)

    try:
        from scipy.signal import find_peaks

        peaks, _ = find_peaks(
            hist_sm,
            prominence=prominence,
            distance=min_peak_distance_bins,
        )
    except ImportError:
        # Fallback local-maxima detector when scipy isn't available.
        logger.warning("scipy not available — using fallback stair peak detector")
        peaks = []
        for i in range(1, len(hist_sm) - 1):
            if (
                hist_sm[i] >= hist_sm[i - 1]
                and hist_sm[i] > hist_sm[i + 1]
                and hist_sm[i] >= prominence
            ):
                if not peaks or (i - peaks[-1]) >= min_peak_distance_bins:
                    peaks.append(i)
        peaks = np.asarray(peaks, dtype=int)

    if len(peaks) < min_steps:
        return 0

    peak_z = 0.5 * (bin_edges[peaks] + bin_edges[peaks + 1])
    peak_z = np.sort(peak_z)

    # Keep only peak sequences with plausible riser spacing.
    if len(peak_z) >= 2:
        dz = np.diff(peak_z)
        riser_ok = (dz >= 0.08) & (dz <= 0.30)
        if np.any(riser_ok):
            run = best = 1
            for ok in riser_ok:
                if ok:
                    run += 1
                    best = max(best, run)
                else:
                    run = 1
            step_count = best
        else:
            step_count = len(peaks)
    else:
        step_count = len(peaks)

    if step_count < min_steps or step_count > max_steps:
        return 0

    logger.debug("stair_steps_detected", count=step_count, z_range_m=round(z_range, 2))
    return step_count


# ── E1: Spatial Index ─────────────────────────────────────────────────────────


class SegmentSpatialIndex:
    """KD-tree index over segment centroids for O(log n) proximity queries.

    Replaces the O(n²) brute-force loop in detect_valves() and similar code.
    """

    def __init__(self, segments: list[dict]) -> None:
        from scipy.spatial import KDTree

        self._segments = segments
        if not segments:
            self._tree = None
            self._centroids = np.empty((0, 3))
            return

        self._centroids = np.array([s["centroid"] for s in segments])
        self._tree = KDTree(self._centroids)

    def query_radius(self, point: np.ndarray, radius: float) -> list[int]:
        """Return indices of segments whose centroids are within `radius` metres."""
        if self._tree is None:
            return []
        return list(self._tree.query_ball_point(point, r=radius))

    def nearest(self, point: np.ndarray, k: int = 1) -> list[int]:
        """Return indices of the k nearest segment centroids."""
        if self._tree is None or len(self._segments) == 0:
            return []
        k = min(k, len(self._segments))
        _, indices = self._tree.query(point, k=k)
        if isinstance(indices, np.integer):
            return [int(indices)]
        return [int(i) for i in indices]


# ── E2: MEP Connectivity Graph ────────────────────────────────────────────────


def build_mep_graph(
    segments: list[dict],
    connection_radius: float = 0.30,
) -> object | None:
    """Build a proximity graph connecting MEP segments (pipes, ducts, conduits).

    Nodes:  segment indices (int)
    Edges:  added when two segment centroids are within `connection_radius` metres.

    Args:
        segments:          List of segment dicts (from detect_clusters / detect_valves).
        connection_radius: Max centroid-to-centroid distance to consider connected.

    Returns:
        networkx.Graph if networkx is installed, None otherwise.
    """
    try:
        import networkx as nx
    except ImportError:
        logger.warning("networkx not installed — MEP graph skipped. pip install networkx")
        return None

    from agent.models import SegmentShape

    mep_shapes = {SegmentShape.CYLINDER, SegmentShape.BOX, SegmentShape.VALVE_CANDIDATE}
    mep_segs = [(i, s) for i, s in enumerate(segments) if s.get("shape") in mep_shapes]

    G = nx.Graph()
    for i, _ in mep_segs:
        G.add_node(i)

    if len(mep_segs) < 2:
        return G

    centroids = np.array([s["centroid"] for _, s in mep_segs])
    original_indices = [i for i, _ in mep_segs]

    from scipy.spatial import KDTree

    tree = KDTree(centroids)
    pairs = tree.query_pairs(r=connection_radius)

    for a, b in pairs:
        G.add_edge(original_indices[a], original_indices[b])

    n_components = nx.number_connected_components(G)
    logger.info(
        "mep_graph_built",
        nodes=G.number_of_nodes(),
        edges=G.number_of_edges(),
        components=n_components,
    )
    return G


def snap_mep_endpoints(segments: list[dict], G: networkx.Graph) -> None:
    """Snap endpoints of connected MEP cylinders/boxes to ensure topological continuity.

    For each edge in the connectivity graph G, find the intersection/closest points
    between the two segments and update their explicit start/end tags.
    """
    if G is None or G.number_of_edges() == 0:
        return

    from agent.models import SegmentShape

    for u, v in G.edges():
        seg_u, seg_v = segments[u], segments[v]

        # Snap cylinders (Pipes, Conduits) and elongated Boxes (Ducts, Cable Trays)
        allowed_shapes = {SegmentShape.CYLINDER, SegmentShape.BOX}
        if seg_u["shape"] not in allowed_shapes or seg_v["shape"] not in allowed_shapes:
            continue

        # For BOX, ensure it is elongated (axis-like)
        def _is_elongated(s):
            if s["shape"] == SegmentShape.CYLINDER:
                return True
            ext = s["maxs"] - s["mins"]
            return max(ext) / max(np.sort(ext)[1], 0.01) > 2.0

        if not _is_elongated(seg_u) or not _is_elongated(seg_v):
            continue

        # Get axis data
        axis_u = np.array(
            [
                seg_u["tags"].get("cyl_axis_x", 0),
                seg_u["tags"].get("cyl_axis_y", 0),
                seg_u["tags"].get("cyl_axis_z", 0),
            ]
        )
        axis_v = np.array(
            [
                seg_v["tags"].get("cyl_axis_x", 0),
                seg_v["tags"].get("cyl_axis_y", 0),
                seg_v["tags"].get("cyl_axis_z", 0),
            ]
        )

        if np.linalg.norm(axis_u) < 1e-6 or np.linalg.norm(axis_v) < 1e-6:
            continue

        # Reconstruct current endpoints from centroid and BB length if not already tagged
        def _get_endpoints(s, axis):
            bb_len = max(s["maxs"] - s["mins"])
            c = s["centroid"]
            return c - axis * (bb_len / 2.0), c + axis * (bb_len / 2.0)

        start_u, end_u = _get_endpoints(seg_u, axis_u)
        start_v, end_v = _get_endpoints(seg_v, axis_v)

        # Solve for closest points between two lines: P(s) = P0 + s*D0, Q(t) = Q0 + t*D1
        p0, d0 = seg_u["centroid"], axis_u
        q0, d1 = seg_v["centroid"], axis_v

        w0 = p0 - q0
        a = np.dot(d0, d0)
        b = np.dot(d0, d1)
        c = np.dot(d1, d1)
        d = np.dot(d0, w0)
        e = np.dot(d1, w0)
        denom = a * c - b * b

        if abs(denom) < 1e-6:
            continue  # Parallel

        sc = (b * e - c * d) / denom
        tc = (a * e - b * d) / denom

        intersect_p = p0 + sc * d0
        intersect_q = q0 + tc * d1
        midpoint = (intersect_p + intersect_q) / 2.0

        # Snap the closer endpoint of each segment to the midpoint
        def _snap(s, start, end, target):
            d_start = np.linalg.norm(start - target)
            d_end = np.linalg.norm(end - target)
            if d_start < d_end:
                new_start, new_end = target, end
            else:
                new_start, new_end = start, target

            s["tags"]["cyl_start_x_mm"] = round(new_start[0] * 1000.0, 1)
            s["tags"]["cyl_start_y_mm"] = round(new_start[1] * 1000.0, 1)
            s["tags"]["cyl_start_z_mm"] = round(new_start[2] * 1000.0, 1)
            s["tags"]["cyl_end_x_mm"] = round(new_end[0] * 1000.0, 1)
            s["tags"]["cyl_end_y_mm"] = round(new_end[1] * 1000.0, 1)
            s["tags"]["cyl_end_z_mm"] = round(new_end[2] * 1000.0, 1)

        _snap(seg_u, start_u, end_u, midpoint)
        _snap(seg_v, start_v, end_v, midpoint)


# ── Stage 2: Orientation Canonicalization & Plane Classification ─────────────


def canonicalize_wall_axis(axis: np.ndarray | list[float] | tuple[float, ...]) -> np.ndarray:
    """Canonicalize a 2D wall axis vector so its angle lies in [0, 180°).

    Ensures that axis vectors v and -v produce the identical canonical unit vector.
    Preserves exact continuous arbitrary orientation angle without snapping.

    Rule:
      If ax < -1e-6 or (abs(ax) <= 1e-6 and ay < 0.0), negate both components.
    """
    arr = np.asarray(axis, dtype=float)[:2]
    norm = float(np.linalg.norm(arr))
    if norm < 1e-9:
        return np.array([1.0, 0.0], dtype=float)
    ax, ay = arr[0] / norm, arr[1] / norm
    if ax < -1e-6 or (abs(ax) <= 1e-6 and ay < 0.0):
        ax, ay = -ax, -ay
    return np.array([ax, ay], dtype=float)


def wall_angle_degrees(axis: np.ndarray | list[float] | tuple[float, ...]) -> float:
    """Compute the continuous orientation angle in [0, 180°) from a wall axis."""
    c_axis = canonicalize_wall_axis(axis)
    angle = float((np.degrees(np.arctan2(c_axis[1], c_axis[0])) + 360.0) % 180.0)
    return round(angle, 4)


def classify_plane_orientation(
    normal: np.ndarray | list[float],
    up_axis: np.ndarray | list[float] | None = None,
    vertical_max: float = 0.25,
    horizontal_min: float = 0.85,
) -> str:
    """Classify plane orientation from its normal vector.

    Uses the absolute vertical component |n · up|:
      - |n · up| < vertical_max       → 'plane_vertical'
      - |n · up| >= horizontal_min    → 'plane_horizontal'
      - otherwise                     → 'plane_sloped'

    Normal sign (+n vs -n) produces the identical orientation class.
    """
    n = np.asarray(normal, dtype=float)
    norm_n = float(np.linalg.norm(n))
    if norm_n > 1e-9:
        n = n / norm_n
    else:
        n = np.array([0.0, 0.0, 1.0], dtype=float)

    if up_axis is None:
        up = np.array([0.0, 0.0, 1.0], dtype=float)
    else:
        up = np.asarray(up_axis, dtype=float)
        norm_up = float(np.linalg.norm(up))
        if norm_up > 1e-9:
            up = up / norm_up

    abs_n_up = abs(float(np.dot(n, up)))
    if abs_n_up < vertical_max:
        return "plane_vertical"
    elif abs_n_up >= horizontal_min:
        return "plane_horizontal"
    else:
        return "plane_sloped"


# ── Explicit Named Unit Conversions ──────────────────────────────────────────
# Standard conversion factors: 1 meter = 1000 mm, 1 foot = 0.3048 meters = 304.8 mm
MM_PER_METER: float = 1000.0
METERS_PER_MM: float = 0.001
FEET_PER_METER: float = 1.0 / 0.3048
METERS_PER_FOOT: float = 0.3048
FEET_PER_MM: float = 1.0 / 304.8
MM_PER_FOOT: float = 304.8


def meters_to_mm(meters: float) -> float:
    """Convert meters to millimeters."""
    return float(meters) * MM_PER_METER


def mm_to_meters(mm: float) -> float:
    """Convert millimeters to meters."""
    return float(mm) * METERS_PER_MM


def meters_to_revit_feet(meters: float) -> float:
    """Convert meters to Revit decimal feet."""
    return float(meters) * FEET_PER_METER


def mm_to_revit_feet(mm: float) -> float:
    """Convert millimeters to Revit decimal feet."""
    return float(mm) * FEET_PER_MM


def revit_feet_to_meters(feet: float) -> float:
    """Convert Revit decimal feet to meters."""
    return float(feet) * METERS_PER_FOOT


def revit_feet_to_mm(feet: float) -> float:
    """Convert Revit decimal feet to millimeters."""
    return float(feet) * MM_PER_FOOT


def compute_survey_transform(points_xyz: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Compute rigid survey-to-local 4x4 transform to center scan near origin.
    
    Returns:
        centered_points: (N, 3) translated points
        T_survey_to_local: (4, 4) translation matrix
        origin_offset: (3,) translation vector in survey coordinates
    """
    if len(points_xyz) == 0:
        return points_xyz, np.eye(4), np.zeros(3)
    
    min_pt = points_xyz.min(axis=0)
    max_pt = points_xyz.max(axis=0)
    origin_offset = (min_pt + max_pt) * 0.5
    
    centered_points = points_xyz - origin_offset
    
    T_survey_to_local = np.eye(4, dtype=float)
    T_survey_to_local[0:3, 3] = -origin_offset
    
    return centered_points, T_survey_to_local, origin_offset

