"""Multi-scan ICP registration tools for Scan-to-BIM.

Sprint 3.5 — Feature 1: Multi-scan ICP Registration

Registers N scan stations into a single unified point cloud using Open3D
point-to-plane ICP. The merged cloud is returned in millimetres, matching
the coordinate convention used throughout the rest of the codebase.

Coordinate convention:
    Open3D ICP operates in METRES internally.
    load_scan_as_o3d()       divides coordinates by 1000 (mm → m) on load.
    merged_pcd_to_numpy_mm() multiplies by 1000 (m → mm) on output.
    No other function in this module performs unit conversion.

Drift note:
    Pairwise chained ICP accumulates drift for N > ~5 stations.
    For high-accuracy surveys with many stations, a global bundle adjustment
    pass (open3d.pipelines.registration.global_optimization) should follow.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
import structlog

if TYPE_CHECKING:
    import open3d as o3d

logger = structlog.get_logger()

# ── Constants ─────────────────────────────────────────────────────────────────

_DEFAULT_VOXEL_SIZE = 0.05  # metres — 5 cm voxel for ICP downsampling
_DEFAULT_MAX_ITER = 50  # ICP max iterations per pair
_NORMAL_RADIUS = 0.10  # metres — normal estimation search radius
_NORMAL_MAX_NN = 30  # max neighbours for normal estimation


# ── Internal result container ─────────────────────────────────────────────────


@dataclass
class _PairResult:
    station_index: int
    source_file: str
    transform: np.ndarray  # 4×4 float64
    rmse: float  # metres
    inlier_ratio: float


# ── Public API ────────────────────────────────────────────────────────────────


def load_scan_as_o3d(file_path: Path) -> o3d.geometry.PointCloud:
    """Load a scan file into an Open3D PointCloud in METRES.

    Supports .ply, .pcd, .xyz directly via Open3D.
    Supports .las / .laz via laspy (resampled to Open3D).
    E57 files are attempted via Open3D; if that fails (compile-time flag),
    raises RuntimeError with a CloudCompare conversion hint.

    Args:
        file_path: Path to the scan file.

    Returns:
        Open3D PointCloud with coordinates in metres.

    Raises:
        RuntimeError: If the file cannot be loaded.
        FileNotFoundError: If the file does not exist.
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
        # .ply, .pcd, .xyz — Open3D handles natively
        pcd = o3d.io.read_point_cloud(str(file_path))
        if len(pcd.points) == 0:
            raise RuntimeError(f"Open3D loaded 0 points from {file_path.name}")

    # Convert mm → m if the bounding box diagonal suggests mm-scale coordinates
    pts = np.asarray(pcd.points)
    if len(pts) > 0:
        diag = np.linalg.norm(pts.max(axis=0) - pts.min(axis=0))
        if diag > 500:
            # Likely in millimetres — divide by 1000
            pcd.points = o3d.utility.Vector3dVector(pts / 1000.0)

    logger.info("scan_loaded", file=file_path.name, points=len(pcd.points))
    return pcd


def downsample_for_icp(
    pcd: o3d.geometry.PointCloud,
    voxel_size: float = _DEFAULT_VOXEL_SIZE,
) -> o3d.geometry.PointCloud:
    """Voxel-downsample and estimate normals — required for point-to-plane ICP.

    Args:
        pcd:        Input Open3D PointCloud (metres).
        voxel_size: Voxel grid size in metres.

    Returns:
        New downsampled PointCloud with normals estimated.
    """
    import open3d as o3d

    down = pcd.voxel_down_sample(voxel_size)
    down.estimate_normals(
        o3d.geometry.KDTreeSearchParamHybrid(
            radius=voxel_size * 2,
            max_nn=_NORMAL_MAX_NN,
        )
    )
    down.orient_normals_consistent_tangent_plane(k=15)
    logger.debug("downsampled_for_icp", before=len(pcd.points), after=len(down.points))
    return down


def register_pair_icp(
    source: o3d.geometry.PointCloud,
    target: o3d.geometry.PointCloud,
    voxel_size: float = _DEFAULT_VOXEL_SIZE,
    max_iterations: int = _DEFAULT_MAX_ITER,
) -> tuple[np.ndarray, float, float]:
    """Register source onto target using point-to-plane ICP.

    Args:
        source:         Source PointCloud (to be transformed).
        target:         Target PointCloud (reference frame).
        voxel_size:     Correspondence distance threshold = voxel_size * 1.5.
        max_iterations: Maximum ICP iterations.

    Returns:
        (transform_4x4, rmse_metres, inlier_ratio)
        transform_4x4 is a 4×4 numpy float64 array.
    """
    import open3d as o3d

    threshold = voxel_size * 1.5

    result = o3d.pipelines.registration.registration_icp(
        source,
        target,
        threshold,
        np.eye(4),
        o3d.pipelines.registration.TransformationEstimationPointToPlane(),
        o3d.pipelines.registration.ICPConvergenceCriteria(max_iteration=max_iterations),
    )

    rmse = result.inlier_rmse
    inlier_ratio = result.fitness  # fraction of points within threshold
    transform = np.array(result.transformation, dtype=np.float64)

    logger.debug(
        "icp_pair_done",
        rmse_m=round(rmse, 6),
        inlier_ratio=round(inlier_ratio, 4),
    )
    return transform, rmse, inlier_ratio


def register_all_scans(
    file_paths: list[Path],
    voxel_size: float = _DEFAULT_VOXEL_SIZE,
) -> tuple[o3d.geometry.PointCloud, list[_PairResult]]:
    """Register all scan stations into one unified point cloud.

    Station 0 is the world reference frame (identity transform).
    Each subsequent station is registered against the already-merged cloud
    (growing reference strategy), then transformed and merged.

    Args:
        file_paths: Ordered list of scan file paths (>=2 required).
        voxel_size: ICP voxel size in metres.

    Returns:
        (merged_pcd, list_of_pair_results)
        merged_pcd is in METRES with all stations in station-0 frame.

    Raises:
        ValueError: If fewer than 2 file paths provided.
    """

    if len(file_paths) < 2:
        raise ValueError("At least 2 scan files are required for ICP registration.")

    results: list[_PairResult] = []

    # Load and downsample all scans
    raw_pcds = [load_scan_as_o3d(p) for p in file_paths]
    down_pcds = [downsample_for_icp(p, voxel_size) for p in raw_pcds]

    # Station 0 — identity, already in reference frame
    results.append(
        _PairResult(
            station_index=0,
            source_file=file_paths[0].name,
            transform=np.eye(4, dtype=np.float64),
            rmse=0.0,
            inlier_ratio=1.0,
        )
    )

    # Accumulate transforms: T_i = T_{i-1} @ T_pair_i
    accumulated_transforms = [np.eye(4, dtype=np.float64)]

    # Merge starts with the full (non-downsampled) station-0 cloud
    merged = raw_pcds[0]

    for i in range(1, len(file_paths)):
        target_down = _build_merged_down(merged, voxel_size)
        source_down = down_pcds[i]

        T_pair, rmse, inlier_ratio = register_pair_icp(source_down, target_down, voxel_size)

        # Chain the transform
        T_world = accumulated_transforms[-1] @ T_pair
        accumulated_transforms.append(T_world)

        # Transform full-resolution source and merge
        source_world = raw_pcds[i].transform(T_world)
        merged = merged + source_world

        results.append(
            _PairResult(
                station_index=i,
                source_file=file_paths[i].name,
                transform=T_world,
                rmse=rmse * 1000,  # convert m → mm for the result
                inlier_ratio=inlier_ratio,
            )
        )

        logger.info(
            "station_registered",
            station=i,
            file=file_paths[i].name,
            rmse_mm=round(rmse * 1000, 2),
        )

    # Light final downsample to remove duplicate points at stitch boundaries
    merged = merged.voxel_down_sample(voxel_size / 2)

    return merged, results


def merged_pcd_to_numpy_mm(merged_pcd: o3d.geometry.PointCloud) -> np.ndarray:
    """Convert a merged Open3D PointCloud (metres) to (N,3) numpy array in mm.

    This is the hand-off point back to the existing segmentation pipeline
    which operates entirely in millimetres.
    """
    pts_m = np.asarray(merged_pcd.points, dtype=np.float64)
    return pts_m * 1000.0


def global_rmse_mm(pair_results: list[_PairResult]) -> float:
    """Compute a single global RMSE across all station pairs (mm).

    Station 0 has RMSE=0 (reference); the global value is the RMS of all
    non-reference station RMSEs, or 0.0 if only one station.
    """
    non_ref = [r.rmse for r in pair_results if r.station_index > 0]
    if not non_ref:
        return 0.0
    return float(np.sqrt(np.mean(np.square(non_ref))))


# ── Private helpers ───────────────────────────────────────────────────────────


def _build_merged_down(
    pcd: o3d.geometry.PointCloud,
    voxel_size: float,
) -> o3d.geometry.PointCloud:
    """Downsample the current merged cloud for use as ICP target."""
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
    """Load a .las/.laz file using laspy, return Open3D PointCloud in metres."""
    import open3d as o3d

    try:
        import laspy
    except ImportError as exc:
        raise RuntimeError(
            "laspy is required to load .las/.laz files. Install with: pip install laspy[lazrs]"
        ) from exc

    las = laspy.read(str(file_path))
    # laspy returns coordinates in the file's native unit (usually metres for
    # survey-grade scanners, but sometimes feet). We assume metres here.
    xs = las.x.scaled_array()
    ys = las.y.scaled_array()
    zs = las.z.scaled_array()
    pts = np.column_stack([xs, ys, zs]).astype(np.float64)

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(pts)
    return pcd


def _load_e57(file_path: Path) -> o3d.geometry.PointCloud:
    """Attempt to load an .e57 file via Open3D (requires E57 build flag)."""
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
