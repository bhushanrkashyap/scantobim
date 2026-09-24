"""Multi-scan Registration Tools — Phase 2 Spatial Foundation.

Registers N scan stations into a single unified point cloud.

Architecture:
    Reference Cloud + Moving Cloud
    → preprocessing (voxel downsample + normal estimation)
    → FPFH feature extraction
    → RANSAC global alignment
    → ICP refinement
    → registration validation

Modes:
    Mode A (Single Scan):  Returns REGISTRATION_NOT_REQUIRED immediately.
    Mode B (Multi-Scan):   Full FPFH → RANSAC → ICP pipeline with validation.

Coordinate convention:
    Open3D ICP operates in METRES internally.
    load_scan_as_o3d()       uses UnitResolution for scaling.
    merged_pcd_to_numpy_mm() multiplies by 1000 (m → mm) on output.

Drift note:
    Pairwise chained ICP accumulates drift for N > ~5 stations.
    For high-accuracy surveys with many stations, a global bundle adjustment
    pass (open3d.pipelines.registration.global_optimization) should follow.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
import structlog

from agent.tools.coordinate_system import (
    REGISTRATION_FAILED,
    REGISTRATION_NOT_REQUIRED,
    REGISTRATION_SUCCESS,
)
from agent.tools.unit_resolution import UnitResolution, resolve_units

if TYPE_CHECKING:
    import open3d as o3d

logger = structlog.get_logger()

# ── Constants ─────────────────────────────────────────────────────────────────

_DEFAULT_VOXEL_SIZE = 0.05  # metres — 5 cm voxel for ICP downsampling
_DEFAULT_MAX_ITER = 50  # ICP max iterations per pair
_NORMAL_RADIUS = 0.10  # metres — normal estimation search radius
_NORMAL_MAX_NN = 30  # max neighbours for normal estimation


# ── Result Containers ─────────────────────────────────────────────────────────


@dataclass
class RegistrationQuality:
    """Quantitative registration quality metrics."""

    status: str = REGISTRATION_SUCCESS
    fitness: float = 0.0  # fraction of points within threshold
    rmse: float = 0.0  # metres
    inlier_count: int = 0
    inlier_ratio: float = 0.0
    overlap: float = 0.0
    transform_determinant: float = 1.0
    rotation_orthogonality_error: float = 0.0
    method: str = "none"
    is_valid: bool = True
    failure_reasons: list[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "status": self.status,
            "is_valid": self.is_valid,
            "fitness": round(self.fitness, 6),
            "rmse_m": round(self.rmse, 6),
            "rmse_mm": round(self.rmse * 1000.0, 3),
            "inlier_count": self.inlier_count,
            "inlier_ratio": round(self.inlier_ratio, 4),
            "overlap": round(self.overlap, 4),
            "transform_determinant": round(self.transform_determinant, 8),
            "rotation_orthogonality_error": round(
                self.rotation_orthogonality_error, 8
            ),
            "method": self.method,
            "failure_reasons": self.failure_reasons,
        }


@dataclass
class _PairResult:
    station_index: int
    source_file: str
    transform: np.ndarray  # 4×4 float64
    rmse: float  # metres
    inlier_ratio: float
    quality: RegistrationQuality = field(default_factory=RegistrationQuality)
    method: str = "FPFH_RANSAC_ICP"


# ── Mode Resolution ──────────────────────────────────────────────────────────


def resolve_registration_mode(
    file_paths: list[Path],
) -> tuple[str, str]:
    """Determine whether registration is required.

    Args:
        file_paths: List of scan file paths.

    Returns:
        (status, reason)
        status: REGISTRATION_NOT_REQUIRED or REGISTRATION_REQUIRED
    """
    if len(file_paths) <= 1:
        return (
            REGISTRATION_NOT_REQUIRED,
            f"Single scan ({len(file_paths)} file) — registration not required",
        )
    return (
        "REGISTRATION_REQUIRED",
        f"{len(file_paths)} scans detected — multi-scan registration required",
    )


# ── Public API ────────────────────────────────────────────────────────────────


def load_scan_as_o3d(
    file_path: Path,
    unit_resolution: UnitResolution | None = None,
) -> o3d.geometry.PointCloud:
    """Load a scan file into an Open3D PointCloud in METRES.

    Uses UnitResolution for proper scaling instead of fragile heuristics.

    Args:
        file_path: Path to the scan file.
        unit_resolution: Optional pre-computed UnitResolution. If None,
                         resolution is performed automatically.

    Returns:
        Open3D PointCloud with coordinates in metres.
    """
    import open3d as o3d

    if not file_path.exists():
        raise FileNotFoundError(f"Scan file not found: {file_path}")

    suffix = file_path.suffix.lower()

    if suffix in (".las", ".laz"):
        pcd = _load_las(file_path)
    elif suffix == ".e57":
        pcd = _load_e57(file_path)
    else:
        pcd = o3d.io.read_point_cloud(str(file_path))
        if len(pcd.points) == 0:
            raise RuntimeError(f"Open3D loaded 0 points from {file_path.name}")

    # Resolve units and scale to metres
    pts = np.asarray(pcd.points)
    if unit_resolution is None:
        unit_resolution = resolve_units(
            points=pts,
            file_path=file_path,
            file_format=suffix.lstrip("."),
        )

    if abs(unit_resolution.scale_to_meters - 1.0) > 1e-6:
        pcd.points = o3d.utility.Vector3dVector(
            pts * unit_resolution.scale_to_meters
        )
        logger.info(
            "scan_scaled_to_meters",
            file=file_path.name,
            source_unit=unit_resolution.detected_unit,
            scale=unit_resolution.scale_to_meters,
        )

    logger.info(
        "scan_loaded",
        file=file_path.name,
        points=len(pcd.points),
        unit_status=unit_resolution.status,
    )
    return pcd


def downsample_for_registration(
    pcd: o3d.geometry.PointCloud,
    voxel_size: float = _DEFAULT_VOXEL_SIZE,
) -> o3d.geometry.PointCloud:
    """Voxel-downsample and estimate normals for registration.

    Args:
        pcd:        Input Open3D PointCloud (metres).
        voxel_size: Voxel grid size in metres.

    Returns:
        Downsampled PointCloud with estimated normals.
    """
    import open3d as o3d

    down = pcd.voxel_down_sample(voxel_size)
    if len(down.points) >= 3:
        down.estimate_normals(
            o3d.geometry.KDTreeSearchParamHybrid(
                radius=voxel_size * 2,
                max_nn=_NORMAL_MAX_NN,
            )
        )
        if len(down.points) >= 15:
            try:
                down.orient_normals_consistent_tangent_plane(k=min(15, len(down.points) - 1))
            except (RuntimeError, ValueError) as exc:
                logger.debug("orient_normals_skipped", error=str(exc))
    logger.debug(
        "downsampled_for_registration",
        before=len(pcd.points),
        after=len(down.points),
    )
    return down


def extract_fpfh_features(
    pcd: o3d.geometry.PointCloud,
    voxel_size: float = _DEFAULT_VOXEL_SIZE,
) -> o3d.pipelines.registration.Feature:
    """Extract FPFH features for correspondence generation.

    Parameters are derived from voxel_size to adapt to point density.

    Args:
        pcd: Downsampled PointCloud with normals.
        voxel_size: Voxel size used for downsampling.

    Returns:
        Open3D Feature object containing FPFH descriptors.
    """
    import open3d as o3d

    radius = voxel_size * 5.0  # FPFH search radius
    fpfh = o3d.pipelines.registration.compute_fpfh_feature(
        pcd,
        o3d.geometry.KDTreeSearchParamHybrid(
            radius=radius,
            max_nn=100,
        ),
    )
    logger.debug(
        "fpfh_extracted",
        n_features=fpfh.num(),
        feature_dim=fpfh.dimension(),
        radius=round(radius, 4),
    )
    return fpfh


def ransac_global_registration(
    source: o3d.geometry.PointCloud,
    target: o3d.geometry.PointCloud,
    source_fpfh: o3d.pipelines.registration.Feature,
    target_fpfh: o3d.pipelines.registration.Feature,
    voxel_size: float = _DEFAULT_VOXEL_SIZE,
) -> tuple[np.ndarray, RegistrationQuality]:
    """RANSAC-based global registration using FPFH correspondences.

    Validates the result for plausibility before returning.

    Args:
        source: Source PointCloud (downsampled, with normals).
        target: Target PointCloud (downsampled, with normals).
        source_fpfh: FPFH features for source.
        target_fpfh: FPFH features for target.
        voxel_size: Voxel size for distance threshold.

    Returns:
        (transform_4x4, quality)
    """
    import open3d as o3d

    distance_threshold = voxel_size * 1.5

    result = o3d.pipelines.registration.registration_ransac_based_on_feature_matching(
        source,
        target,
        source_fpfh,
        target_fpfh,
        mutual_filter=True,
        max_correspondence_distance=distance_threshold,
        estimation_method=o3d.pipelines.registration.TransformationEstimationPointToPoint(
            False
        ),
        ransac_n=3,
        checkers=[
            o3d.pipelines.registration.CorrespondenceCheckerBasedOnEdgeLength(
                0.9
            ),
            o3d.pipelines.registration.CorrespondenceCheckerBasedOnDistance(
                distance_threshold
            ),
        ],
        criteria=o3d.pipelines.registration.RANSACConvergenceCriteria(
            100000, 0.999
        ),
    )

    T = np.array(result.transformation, dtype=np.float64)
    quality = _validate_transform(T, result.fitness, result.inlier_rmse, "RANSAC")

    logger.info(
        "ransac_done",
        fitness=round(quality.fitness, 4),
        rmse_m=round(quality.rmse, 6),
        inlier_count=len(result.correspondence_set),
    )
    return T, quality


def register_pair_icp(
    source: o3d.geometry.PointCloud,
    target: o3d.geometry.PointCloud,
    voxel_size: float = _DEFAULT_VOXEL_SIZE,
    max_iterations: int = _DEFAULT_MAX_ITER,
    initial_transform: np.ndarray | None = None,
) -> tuple[np.ndarray, RegistrationQuality]:
    """Register source onto target using point-to-plane ICP.

    Args:
        source:            Source PointCloud (to be transformed).
        target:            Target PointCloud (reference frame).
        voxel_size:        Correspondence distance threshold = voxel_size * 1.5.
        max_iterations:    Maximum ICP iterations.
        initial_transform: Optional initial alignment (e.g. from RANSAC).

    Returns:
        (transform_4x4, quality)
    """
    import open3d as o3d

    threshold = voxel_size * 1.5
    init_T = initial_transform if initial_transform is not None else np.eye(4)

    result = o3d.pipelines.registration.registration_icp(
        source,
        target,
        threshold,
        init_T,
        o3d.pipelines.registration.TransformationEstimationPointToPlane(),
        o3d.pipelines.registration.ICPConvergenceCriteria(
            max_iteration=max_iterations
        ),
    )

    T = np.array(result.transformation, dtype=np.float64)
    quality = _validate_transform(T, result.fitness, result.inlier_rmse, "ICP")

    logger.debug(
        "icp_pair_done",
        fitness=round(quality.fitness, 4),
        rmse_m=round(quality.rmse, 6),
    )
    return T, quality


def register_pair_full(
    source: o3d.geometry.PointCloud,
    target: o3d.geometry.PointCloud,
    voxel_size: float = _DEFAULT_VOXEL_SIZE,
    max_icp_iterations: int = _DEFAULT_MAX_ITER,
) -> tuple[np.ndarray, RegistrationQuality]:
    """Full registration pipeline: FPFH → RANSAC → ICP for one pair.

    If ICP degrades the result, the RANSAC-only transform is retained.

    Returns:
        (transform_4x4, quality)
    """
    # Downsample and extract features
    src_down = downsample_for_registration(source, voxel_size)
    tgt_down = downsample_for_registration(target, voxel_size)
    src_fpfh = extract_fpfh_features(src_down, voxel_size)
    tgt_fpfh = extract_fpfh_features(tgt_down, voxel_size)

    # RANSAC global alignment
    T_ransac, q_ransac = ransac_global_registration(
        src_down, tgt_down, src_fpfh, tgt_fpfh, voxel_size
    )

    # If RANSAC global registration failed, do not attempt local ICP
    if not q_ransac.is_valid:
        logger.warning(
            "ransac_global_registration_failed",
            fitness=round(q_ransac.fitness, 4),
            rmse_m=round(q_ransac.rmse, 6),
            reasons=q_ransac.failure_reasons,
        )
        return T_ransac, q_ransac

    # ICP refinement
    T_icp, q_icp = register_pair_icp(
        src_down,
        tgt_down,
        voxel_size,
        max_icp_iterations,
        initial_transform=T_ransac,
    )

    # Degradation check: keep RANSAC if ICP made things worse
    if q_icp.fitness < q_ransac.fitness * 0.9 or q_icp.rmse > q_ransac.rmse * 1.5:
        logger.warning(
            "icp_degradation_detected",
            ransac_fitness=round(q_ransac.fitness, 4),
            icp_fitness=round(q_icp.fitness, 4),
            ransac_rmse=round(q_ransac.rmse, 6),
            icp_rmse=round(q_icp.rmse, 6),
            action="retaining_ransac_transform",
        )
        q_ransac.method = "FPFH_RANSAC (ICP degraded)"
        return T_ransac, q_ransac

    q_icp.method = "FPFH_RANSAC_ICP"
    return T_icp, q_icp


def register_all_scans(
    file_paths: list[Path],
    voxel_size: float = _DEFAULT_VOXEL_SIZE,
) -> tuple[o3d.geometry.PointCloud, list[_PairResult], str]:
    """Register all scan stations into one unified point cloud.

    Station 0 is the world reference frame (identity transform).
    Each subsequent station uses FPFH → RANSAC → ICP against the merged cloud.

    Args:
        file_paths: Ordered list of scan file paths (≥2 required).
        voxel_size: ICP voxel size in metres.

    Returns:
        (merged_pcd, pair_results, registration_status)
    """
    if len(file_paths) < 2:
        raise ValueError("At least 2 scan files are required for registration.")

    results: list[_PairResult] = []

    # Load all scans
    raw_pcds = [load_scan_as_o3d(p) for p in file_paths]

    # Station 0 — identity
    results.append(
        _PairResult(
            station_index=0,
            source_file=file_paths[0].name,
            transform=np.eye(4, dtype=np.float64),
            rmse=0.0,
            inlier_ratio=1.0,
            method="reference",
        )
    )

    accumulated_transforms = [np.eye(4, dtype=np.float64)]
    merged = raw_pcds[0]

    for i in range(1, len(file_paths)):
        source = raw_pcds[i]

        T_pair, quality = register_pair_full(source, merged, voxel_size)

        # Validate registration quality
        if quality.fitness < 0.3 or quality.rmse > 0.5:
            logger.warning(
                "registration_quality_poor",
                station=i,
                fitness=round(quality.fitness, 4),
                rmse_m=round(quality.rmse, 6),
            )

        # Chain the transform
        T_world = accumulated_transforms[-1] @ T_pair
        accumulated_transforms.append(T_world)

        # Transform full-resolution source and merge
        source_world = source.transform(T_world)
        merged = merged + source_world

        results.append(
            _PairResult(
                station_index=i,
                source_file=file_paths[i].name,
                transform=T_world,
                rmse=quality.rmse * 1000.0,  # mm for downstream reporting convention
                inlier_ratio=quality.fitness,
                quality=quality,
                method=quality.method,
            )
        )

        logger.info(
            "station_registered",
            station=i,
            file=file_paths[i].name,
            rmse_mm=round(quality.rmse * 1000, 2),
            method=quality.method,
        )

    # Light downsample at stitch boundaries
    merged = merged.voxel_down_sample(voxel_size / 2)

    return merged, results


def merged_pcd_to_numpy_mm(merged_pcd: o3d.geometry.PointCloud) -> np.ndarray:
    """Convert a merged Open3D PointCloud (metres) to (N,3) numpy array in mm."""
    pts_m = np.asarray(merged_pcd.points, dtype=np.float64)
    return pts_m * 1000.0


def global_rmse_mm(pair_results: list[_PairResult]) -> float:
    """Compute a single global RMSE across all station pairs (mm)."""
    non_ref = [r.rmse for r in pair_results if r.station_index > 0]
    if not non_ref:
        return 0.0
    return float(np.sqrt(np.mean(np.square(non_ref))))


# ── Validation ────────────────────────────────────────────────────────────────


def _validate_transform(
    T: np.ndarray,
    fitness: float,
    rmse: float,
    method: str,
) -> RegistrationQuality:
    """Validate a 4×4 transform for plausibility."""
    try:
        T_arr = np.asarray(T, dtype=np.float64)
    except (ValueError, TypeError, AttributeError):
        T_arr = np.array([])

    fit_val = float(fitness) if isinstance(fitness, (int, float)) else 0.0
    rmse_val = float(rmse) if isinstance(rmse, (int, float)) else float("inf")

    if T_arr.shape != (4, 4) or not np.all(np.isfinite(T_arr)):
        logger.warning("transform_invalid_shape_or_non_finite", method=method, shape=getattr(T_arr, "shape", None))
        return RegistrationQuality(
            status=REGISTRATION_FAILED,
            fitness=0.0,
            rmse=float("inf"),
            is_valid=False,
            method=method,
            failure_reasons=[f"Invalid transform shape {getattr(T_arr, 'shape', None)} or non-finite values"],
        )

    R = T_arr[:3, :3]
    det_r = float(np.linalg.det(R))
    orth_err = float(np.max(np.abs(R.T @ R - np.eye(3))))

    is_good = (fit_val >= 0.3 and rmse_val <= 0.2 and abs(det_r - 1.0) < 1e-2 and orth_err < 1e-2)
    status = REGISTRATION_SUCCESS if is_good else REGISTRATION_FAILED

    failure_reasons = []
    if fit_val < 0.3:
        failure_reasons.append(f"Low fitness {fit_val:.3f} < 0.3")
    if rmse_val > 0.2:
        failure_reasons.append(f"High RMSE {rmse_val:.3f} > 0.2m")
    if abs(det_r - 1.0) >= 1e-2:
        failure_reasons.append(f"Improper rotation det(R)={det_r:.4f}")
    if orth_err >= 1e-2:
        failure_reasons.append(f"Non-orthogonal rotation error={orth_err:.4f}")

    return RegistrationQuality(
        status=status,
        fitness=fit_val,
        rmse=rmse_val,
        inlier_ratio=fit_val,
        transform_determinant=det_r,
        rotation_orthogonality_error=orth_err,
        method=method,
        is_valid=is_good,
        failure_reasons=failure_reasons,
    )


# ── Private helpers ───────────────────────────────────────────────────────────


def _build_merged_down(
    pcd: o3d.geometry.PointCloud,
    voxel_size: float,
) -> o3d.geometry.PointCloud:
    """Downsample the current merged cloud for use as registration target."""
    import open3d as o3d

    down = pcd.voxel_down_sample(voxel_size)
    down.estimate_normals(
        o3d.geometry.KDTreeSearchParamHybrid(
            radius=voxel_size * 2,
            max_nn=_NORMAL_MAX_NN,
        )
    )
    down.orient_normals_consistent_tangent_plane(k=15)
    return down


def _load_las(file_path: Path) -> o3d.geometry.PointCloud:
    """Load a .las/.laz file using laspy, return Open3D PointCloud."""
    import open3d as o3d

    try:
        import laspy
    except ImportError as exc:
        raise RuntimeError(
            "laspy is required to load .las/.laz files. "
            "Install with: pip install laspy[lazrs]"
        ) from exc

    las = laspy.read(str(file_path))
    xs = las.x.scaled_array()
    ys = las.y.scaled_array()
    zs = las.z.scaled_array()
    pts = np.column_stack([xs, ys, zs]).astype(np.float64)

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(pts)
    return pcd


def _load_e57(file_path: Path) -> o3d.geometry.PointCloud:
    """Attempt to load an .e57 file via Open3D."""
    import open3d as o3d

    try:
        pcd = o3d.io.read_point_cloud(str(file_path))
        if len(pcd.points) == 0:
            raise RuntimeError("Open3D loaded 0 points from E57")
        return pcd
    except Exception as exc:
        raise RuntimeError(
            f"Failed to load {file_path.name} as E57. "
            "Open3D may not have been compiled with E57 support. "
            "Convert to PLY first using CloudCompare: "
            "File → Save As → PLY (Binary)"
        ) from exc
