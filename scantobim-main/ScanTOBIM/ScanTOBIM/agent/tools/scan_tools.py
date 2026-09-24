"""Point cloud ingestion and segmentation pipeline.

Loads E57/LAZ files, runs RANSAC plane detection and DBSCAN clustering
to extract GeometrySegments from raw scan data.

Sprint 1 additions:
  - detect_scan_format()  — sniff file magic bytes, return format string
  - validate_scan_file()  — check size, format, min point count
  - ScanFileInfo          — dataclass returned by validate_scan_file
"""

from __future__ import annotations

import os
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
import structlog

_LAST_DOWNSAMPLED_POINTS: np.ndarray | None = None
_CURRENT_MULTI_RES_PCD: Any = None


def get_last_downsampled_points() -> np.ndarray | None:
    """Return downsampled points (N, 3) in metres from the most recent segmentation run."""
    return _LAST_DOWNSAMPLED_POINTS


def get_current_multi_res_pcd():
    """Return the authoritative MultiResolutionPointCloud instance from the current scan run."""
    return _CURRENT_MULTI_RES_PCD


def set_current_multi_res_pcd(pcd) -> None:
    """Set the authoritative MultiResolutionPointCloud instance."""
    global _CURRENT_MULTI_RES_PCD
    _CURRENT_MULTI_RES_PCD = pcd

if TYPE_CHECKING:
    import open3d

# ── MULTI-SHAPE SPLITTER FOR REVIT ─────────────────────────────


def split_segment_for_revit(segment: GeometrySegment):
    """Split large segments into valid sub-segments when explicitly enabled.

    Disabled by default because aggressive tiling can explode element counts and
    generate many low-value/degenerate fragments.
    """

    import copy
    import uuid

    enable_split = os.environ.get("STB_ENABLE_SEGMENT_SPLIT", "0").strip().lower() in {
        "1",
        "true",
        "yes",
        "on",
    }
    if not enable_split:
        return [segment]

    bb = segment.bounding_box

    w = bb.max_x - bb.min_x
    d = bb.max_y - bb.min_y
    h = bb.max_z - bb.min_z

    # ✅ Only split LARGE objects
    if max(w, d) < 1500:
        return [segment]

    # Larger default tile to avoid over-fragmentation; still configurable.
    tile = max(int(float(os.environ.get("STB_SEGMENT_SPLIT_TILE_MM", "600"))), 100)
    min_size = max(float(os.environ.get("STB_SEGMENT_MIN_SIZE_MM", "120")), 30.0)

    segments = []

    nx = max(1, int(w // tile))
    ny = max(1, int(d // tile))

    for i in range(nx):
        for j in range(ny):
            new_seg = copy.copy(segment)
            new_seg.segment_id = uuid.uuid4().hex[:32]

            new_bb = copy.copy(segment.bounding_box)

            x0 = bb.min_x + i * tile
            x1 = min(bb.min_x + (i + 1) * tile, bb.max_x)

            y0 = bb.min_y + j * tile
            y1 = min(bb.min_y + (j + 1) * tile, bb.max_y)

            # ✅ VALIDATION (prevents degenerate geometry)
            if (x1 - x0) < min_size:
                continue
            if (y1 - y0) < min_size:
                continue

            new_bb.min_x = x0
            new_bb.max_x = x1
            new_bb.min_y = y0
            new_bb.max_y = y1

            # ✅ enforce thickness
            if (new_bb.max_z - new_bb.min_z) < 50:
                new_bb.max_z = new_bb.min_z + 50

            new_seg.bounding_box = new_bb
            # Scale point count by covered XY area so tiled fragments have
            # more realistic densities for downstream classification.
            if isinstance(getattr(segment, "point_count", None), int) and w > 0 and d > 0:
                tile_area = max((x1 - x0) * (y1 - y0), 1.0)
                full_area = max(w * d, 1.0)
                ratio = tile_area / full_area
                new_seg.point_count = max(1, int(segment.point_count * ratio))
            segments.append(new_seg)

    return segments if segments else [segment]


from agent.models import (
    BoundingBox,
    GeometrySegment,
    Point3D,
    SegmentShape,
)
from agent.tools.geometry_tools import (
    SegmentSpatialIndex,
    count_stair_steps,
    estimate_normals,
    euclidean_cluster,
    get_obb,
    pca_fit,
    pca_linearity,
    ransac_cylinder_fit,
    ransac_sphere_fit,
    region_growing,
)


# --- Robust Normal Estimation Pipeline ---
def robust_normal_estimation(
    pcd,
    radius=0.10,
    max_nn=30,
    voxel_start=0.01,
    voxel_max=0.10,
    max_retries=3,
    remove_outliers=True,
    preserve_original=True,
):
    """
    Wrapper for robust normal estimation with adaptive voxel downsampling,
    outlier removal, coordinate normalization, and QHull error handling.
    Returns dict with 'pcd', 'original_points', 'origin', 'success', 'error'.
    """
    from agent.tools.geometry_tools import estimate_normals

    return estimate_normals(
        pcd,
        radius=radius,
        max_nn=max_nn,
        adaptive=True,
        max_retries=max_retries,
        voxel_start=voxel_start,
        voxel_max=voxel_max,
        remove_outliers=remove_outliers,
        preserve_original=preserve_original,
    )


from agent.bim_reconstruction_fix import detect_levels, run_auto_bim_pipeline
from agent.classifier import classify_segment, run_bim_pipeline_v2
from agent.tools.detection_config import (
    MAX_PLANES,
    NORMAL_MAX_NN,
    NORMAL_RADIUS_VOXEL_MULT,
    SEMANTIC_ENABLE,
    SEMANTIC_FALLBACK_TO_GEOMETRY,
    SEMANTIC_MIN_WALL_POINTS,
    SEMANTIC_MODEL,
    SEMANTIC_WALL_CLASS_ID,
    SEMANTIC_WALL_THRESHOLD,
    SOR_MIN_NEIGHBORS,
    SOR_NB_NEIGHBORS,
    SOR_NEIGHBOR_DIVISOR,
    SOR_STD_RATIO,
    VOXEL_SIZE_M,
)
from agent.tools.semantic_models import predict_wall_mask, set_last_semantic_report

logger = structlog.get_logger()

# ── Sprint 1: Upload validation ────────────────────────────────────────────────

# Supported formats and their magic byte signatures
_FORMAT_MAGIC: dict[str, bytes] = {
    ".e57": b"ASTM-E57",  # E57 ASTM magic header
    ".las": b"LASF",  # LAS/LAZ standard magic
    ".laz": b"LASF",
    ".ply": b"ply",  # PLY ASCII/binary
    ".pcd": b"# .PCD",  # Point Cloud Data format
    ".xyz": b"",  # plain text — no magic, accept by extension
}

SUPPORTED_EXTENSIONS = set(_FORMAT_MAGIC.keys())

# Size limits
MAX_SCAN_SIZE_MB = int(os.environ.get("MAX_SCAN_SIZE_MB", "5000"))
MIN_SCAN_SIZE_BYTES = 1024  # anything smaller is certainly corrupt

# Point count requirements
MIN_POINT_COUNT = 10_000  # absolute minimum for meaningful segmentation


@dataclass
class ScanFileInfo:
    """Result of validate_scan_file()."""

    path: Path
    format: str  # ".e57", ".las", etc.
    size_bytes: int
    size_mb: float
    valid: bool
    errors: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    point_count: int | None = None  # filled after loading, if available


def detect_scan_format(file_path: Path) -> str:
    ext = file_path.suffix.lower()
    if ext not in SUPPORTED_EXTENSIONS:
        # Check magic bytes if extension is missing or unknown
        try:
            with open(file_path, "rb") as fh:
                header = fh.read(8)
            for e, magic in _FORMAT_MAGIC.items():
                if magic and header.startswith(magic):
                    return e
        except Exception:
            pass
        raise ValueError(
            f"Unsupported file format: '{ext}'. Accepted: {sorted(SUPPORTED_EXTENSIONS)}"
        )

    # Verify magic bytes for formats that have them
    magic = _FORMAT_MAGIC[ext]
    if magic:
        with open(file_path, "rb") as fh:
            header = fh.read(len(magic))
        if header != magic:
            raise ValueError(
                f"File '{file_path.name}' has extension '{ext}' but magic bytes "
                f"do not match. Expected {magic!r}, got {header!r}. "
                "The file may be corrupt or misnamed."
            )

    return ext


def validate_scan_file(
    file_path: Path,
    max_size_mb: float = MAX_SCAN_SIZE_MB,
    min_size_bytes: int = MIN_SCAN_SIZE_BYTES,
) -> ScanFileInfo:
    """Validate an uploaded scan file before processing.

    Checks:
      1. File exists and is readable
      2. Extension is a supported format
      3. Magic bytes match declared format
      4. File size is within bounds (not empty, not too large)

    Returns a ScanFileInfo — check .valid and .errors before proceeding.
    """
    errors: list[str] = []
    warnings: list[str] = []

    # Existence check
    if not file_path.exists():
        return ScanFileInfo(
            path=file_path,
            format="unknown",
            size_bytes=0,
            size_mb=0.0,
            valid=False,
            errors=[f"File not found: {file_path}"],
        )

    size_bytes = file_path.stat().st_size
    size_mb = size_bytes / (1024 * 1024)

    # Format detection
    try:
        fmt = detect_scan_format(file_path)
    except ValueError as exc:
        return ScanFileInfo(
            path=file_path,
            format="unknown",
            size_bytes=size_bytes,
            size_mb=size_mb,
            valid=False,
            errors=[str(exc)],
        )

    # Size checks
    if size_bytes < min_size_bytes:
        errors.append(f"File is too small ({size_bytes} bytes). It may be empty or corrupt.")
    if size_mb > max_size_mb:
        errors.append(
            f"File size {size_mb:.1f} MB exceeds the limit of {max_size_mb} MB "
            f"for your current tier. "
            "Contact support to increase the limit or split the scan."
        )

    # Size warning thresholds
    if 500 < size_mb <= max_size_mb:
        warnings.append(
            f"Large scan file ({size_mb:.0f} MB) — processing may take several minutes."
        )

    point_count = count_file_points(file_path)
    if point_count >= 0:
        if point_count == 0:
            errors.append("Scan file contains no points.")
        elif point_count < MIN_POINT_COUNT:
            warnings.append(
                f"Scan has only {point_count:,} points. Minimum recommended is {MIN_POINT_COUNT:,} "
                "for meaningful segmentation."
            )
    else:
        point_count = None

    valid = len(errors) == 0
    logger.info(
        "scan_file_validated",
        path=str(file_path),
        format=fmt,
        size_mb=round(size_mb, 2),
        point_count=point_count,
        valid=valid,
        errors=errors,
    )
    return ScanFileInfo(
        path=file_path,
        format=fmt,
        size_bytes=size_bytes,
        size_mb=size_mb,
        valid=valid,
        errors=errors,
        warnings=warnings,
        point_count=point_count,
    )


def _count_ply_points(file_path: Path) -> int:
    with open(file_path, "rb") as fh:
        header_bytes = b""
        while True:
            chunk = fh.read(4096)
            if not chunk:
                break
            header_bytes += chunk
            if b"end_header" in header_bytes:
                header_bytes = header_bytes[
                    : header_bytes.index(b"end_header") + len(b"end_header")
                ]
                break
        if b"end_header" not in header_bytes:
            return -1

    header = header_bytes.decode("ascii", errors="ignore")
    for line in header.splitlines():
        parts = line.strip().split()
        if len(parts) >= 3 and parts[0].lower() == "element" and parts[1].lower() == "vertex":
            try:
                return int(parts[2])
            except ValueError:
                return -1
    return -1


def _count_pcd_points(file_path: Path) -> int:
    with open(file_path, "rb") as fh:
        for _ in range(128):
            line = fh.readline()
            if not line:
                break
            try:
                text = line.decode("ascii", errors="ignore").strip()
            except UnicodeDecodeError:
                continue
            parts = text.split()
            if len(parts) >= 2 and parts[0].lower() == "points":
                try:
                    return int(parts[1])
                except ValueError:
                    return -1
            if parts and parts[0].lower() == "data":
                break
    return -1


def _count_xyz_points(file_path: Path, max_bytes: int = 200 * 1024 * 1024) -> int:
    if file_path.stat().st_size > max_bytes:
        return -1
    count = 0
    with open(file_path, "rb") as fh:
        for line in fh:
            if line.strip():
                count += 1
    return count


def count_file_points(file_path: Path) -> int:
    """Quick point count from file header (no full load needed).

    Uses laspy header for LAS/LAZ and header metadata for PLY/PCD.
    For XYZ files, counts lines for reasonably-sized files.
    Returns -1 if count cannot be determined cheaply.
    """
    ext = file_path.suffix.lower()
    if ext in (".las", ".laz"):
        try:
            import laspy

            with laspy.open(str(file_path)) as f:
                return int(f.header.point_count)
        except Exception:  # noqa: BLE001
            return -1
    if ext == ".ply":
        return _count_ply_points(file_path)
    if ext == ".pcd":
        return _count_pcd_points(file_path)
    if ext == ".xyz":
        return _count_xyz_points(file_path)
    return -1  # other formats require loading


def _validate_min_points(point_count: int, source: str = "file") -> None:
    """Raise ValueError if point count is below the minimum threshold."""
    if 0 < point_count < MIN_POINT_COUNT:
        raise ValueError(
            f"Scan has only {point_count:,} points from {source}. "
            f"Minimum required is {MIN_POINT_COUNT:,} for meaningful segmentation. "
            "Ensure the scan covers the full area and was not cropped too aggressively."
        )


"""Loads the point cloud data """


def load_point_cloud(file_path: Path) -> open3d.geometry.PointCloud:
    """Load a point cloud from E57, LAZ, LAS, PLY, or PCD file."""
    import open3d as o3d

    suffix = file_path.suffix.lower()
    if suffix not in SUPPORTED_EXTENSIONS:
        try:
            suffix = detect_scan_format(file_path)
        except Exception:
            pass

    if suffix in (".laz", ".las"):
        return _load_las(file_path)
    elif suffix == ".e57":
        return _load_e57(file_path)
    elif suffix in (".ply", ".pcd", ".xyz"):
        pcd = o3d.io.read_point_cloud(str(file_path))
        if pcd.is_empty():
            raise ValueError(
                f"Empty point cloud loaded from {file_path}. "
                "Check the file is a valid, non-empty scan export."
            )
        return pcd
    else:
        raise ValueError(f"Unsupported format: {suffix}")


def _load_las(file_path: Path) -> open3d.geometry.PointCloud:
    """Load LAS/LAZ via laspy, convert to Open3D with RGB preservation."""
    import laspy
    import open3d as o3d

    las = laspy.read(str(file_path))
    points = np.vstack((las.x, las.y, las.z)).T

    # Scale inspection: authoritative unit resolution (Phase 2)
    try:
        from agent.tools.unit_resolution import resolve_units
        unit_res = resolve_units(points=points, file_path=file_path, file_format="las")
        if unit_res.status in ("UNIT_VERIFIED", "UNIT_INFERRED") and abs(unit_res.scale_to_meters - 1.0) > 1e-6:
            points = points * unit_res.scale_to_meters
            logger.info("las_unit_scaled_to_meters", scale=unit_res.scale_to_meters, unit=unit_res.detected_unit, status=unit_res.status)
        else:
            logger.info("las_unit_resolved", unit=unit_res.detected_unit, status=unit_res.status)
    except Exception as exc:
        logger.warning("las_unit_check_warning", error=str(exc))

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)

    # Preserve RGB if present
    if hasattr(las, "red") and hasattr(las, "green") and hasattr(las, "blue"):
        # LAS colors are often 16-bit
        c_max = 65535.0 if np.max(las.red) > 255 else 255.0
        colors = np.vstack((las.red, las.green, las.blue)).T / c_max
        pcd.colors = o3d.utility.Vector3dVector(colors)
    elif hasattr(las, "intensity"):
        # Map intensity to grayscale if no RGB
        intensities = np.asarray(las.intensity).astype(float)
        i_max = np.max(intensities) if np.max(intensities) > 0 else 1.0
        colors = np.tile(intensities[:, None] / i_max, (1, 3))
        pcd.colors = o3d.utility.Vector3dVector(colors)

    return pcd


def _load_e57(file_path: Path) -> open3d.geometry.PointCloud:
    """Load E57 with authoritative multi-resolution source preservation (Level 0) and adaptive detection (Level 1)."""
    global _CURRENT_MULTI_RES_PCD
    import open3d as o3d

    # Authoritative multi-resolution loading preserving all raw points in Level 0
    try:
        from agent.tools.multi_res_pcd import MultiResolutionPointCloud

        logger.info("e57_multi_res_loading_start", file=str(file_path))
        multi = MultiResolutionPointCloud.from_e57(file_path)
        _CURRENT_MULTI_RES_PCD = multi
        lvl1_pts = multi.get_level1_adaptive(base_voxel_size_m=0.05, target_max_points=1_500_000)
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(lvl1_pts.astype(np.float64, copy=False))
        logger.info(
            "e57_multi_res_loaded_success",
            file=str(file_path),
            level0_points=multi.loaded_point_count,
            level1_points=len(lvl1_pts),
        )
        return pcd
    except Exception as exc:
        logger.warning("e57_multi_res_load_failed_falling_back", error=str(exc))

    try:
        import pye57
        import pye57.e57 as pye57_internal

        # pye57 defaults cartesian buffers to float64 (dtype code 'd'), which
        # can OOM on very large scans. Force float32 buffers unless disabled.
        if os.environ.get("STB_E57_FORCE_FLOAT32", "1").strip().lower() in {"1", "true", "yes", "on"}:
            for axis_field in ("cartesianX", "cartesianY", "cartesianZ"):
                if axis_field in pye57_internal.SUPPORTED_POINT_FIELDS:
                    pye57_internal.SUPPORTED_POINT_FIELDS[axis_field] = "f"

        e57 = pye57.E57(str(file_path))

        # Hard memory guard for very large E57 files. Sampling is deterministic
        # (stride-based) so repeated runs on the same file are reproducible.
        max_total_points = max(int(os.environ.get("STB_E57_MAX_POINTS", "500000")), MIN_POINT_COUNT)

        # Keep memory usage bounded: allocate exactly one XYZ buffer and fill it.
        sampled_points = np.empty((max_total_points, 3), dtype=np.float64)
        write_idx = 0
        raw_total = 0

        # Colors are optional for segmentation and expensive for huge E57 scans.
        # Default off to avoid extra allocations; can be enabled when needed.
        load_colors = os.environ.get("STB_E57_LOAD_COLORS", "0").strip().lower() in {
            "1", "true", "yes", "on"
        }
        sampled_colors: list[np.ndarray] = []

        for i in range(e57.scan_count):
            remaining_budget = max_total_points - write_idx
            if remaining_budget <= 0:
                logger.info("e57_sampling_budget_exhausted", scan_index=i, max_total_points=max_total_points)
                break

            # read_scan() can explode RAM on huge datasets while transforming to
            # global coordinates. read_scan_raw() avoids that expensive allocation.
            try:
                data = e57.read_scan_raw(i, ignore_unsupported_fields=True)
            except TypeError:
                data = e57.read_scan_raw(i)

            if not all(k in data for k in ("cartesianX", "cartesianY", "cartesianZ")):
                continue

            # Keep zero-copy views for full arrays; cast only sampled slices.
            x = np.asarray(data["cartesianX"])
            y = np.asarray(data["cartesianY"])
            z = np.asarray(data["cartesianZ"])

            n_scan = int(x.shape[0])
            if n_scan == 0:
                continue

            raw_total += n_scan
            stride = max(1, int(np.ceil(n_scan / float(remaining_budget))))

            xs = x[::stride]
            ys = y[::stride]
            zs = z[::stride]

            if xs.shape[0] > remaining_budget:
                xs = xs[:remaining_budget]
                ys = ys[:remaining_budget]
                zs = zs[:remaining_budget]

            xs = xs.astype(np.float32, copy=False)
            ys = ys.astype(np.float32, copy=False)
            zs = zs.astype(np.float32, copy=False)
            n_kept = int(xs.shape[0])
            if n_kept == 0:
                continue

            end_idx = write_idx + n_kept
            sampled_points[write_idx:end_idx, 0] = xs
            sampled_points[write_idx:end_idx, 1] = ys
            sampled_points[write_idx:end_idx, 2] = zs

            if load_colors:
                if "colorRed" in data and "colorGreen" in data and "colorBlue" in data:
                    r = np.asarray(data["colorRed"])[::stride][:n_kept].astype(np.float32, copy=False) / 255.0
                    g = np.asarray(data["colorGreen"])[::stride][:n_kept].astype(np.float32, copy=False) / 255.0
                    b = np.asarray(data["colorBlue"])[::stride][:n_kept].astype(np.float32, copy=False) / 255.0
                    sampled_colors.append(np.column_stack((r, g, b)).astype(np.float64, copy=False))
                elif "intensity" in data:
                    intensities = np.asarray(data["intensity"])[::stride][:n_kept].astype(np.float32, copy=False)
                    i_max = np.max(intensities) if np.max(intensities) > 0 else 1.0
                    sampled_colors.append(np.tile((intensities / i_max)[:, None], (1, 3)).astype(np.float64, copy=False))
                else:
                    sampled_colors.append(np.zeros((n_kept, 3), dtype=np.float64))

            write_idx = end_idx

        if write_idx == 0:
            logger.error("e57_empty_point_cloud", file=str(file_path))
            raise ValueError(f"E57 file contains no points: {file_path}")

        points = sampled_points[:write_idx]
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(points)

        if load_colors and sampled_colors:
            pcd.colors = o3d.utility.Vector3dVector(np.vstack(sampled_colors))
        # Empty point cloud validation
        if len(points) == 0:
            logger.error("e57_empty_point_cloud", file=str(file_path))
            raise ValueError(f"E57 file contains no points: {file_path}")

        logger.info(
            "e57_loaded",
            file=str(file_path),
            points=len(points),
            raw_points=raw_total,
            sampled_points=write_idx,
            max_points=max_total_points,
            rgb=load_colors,
        )
        return pcd
    except ImportError:
        # Fallback to Open3D if pye57 is not installed
        logger.warning("pye57_not_installed_falling_back_to_o3d", path=str(file_path))
        pcd = o3d.io.read_point_cloud(str(file_path))
        if pcd.is_empty():
            logger.error("e57_fallback_empty", file=str(file_path))
            raise ValueError(
                f"Native E57 support failed for {file_path}. "
                "pye57 is not installed and Open3D failed to load the file. "
                "Install pye57 or convert to PLY/LAS."
            )
        logger.info("e57_fallback_loaded", file=str(file_path), points=len(pcd.points))
        return pcd
    except Exception as exc:
        logger.error("e57_load_failed", error=str(exc), file=str(file_path))
        raise ValueError(f"Failed to load E57 file: {exc}") from exc


""""Voxel down sampling algorithm"""


def downsample(pcd: open3d.geometry.PointCloud, voxel_size_m: float = 0.01):
    """Voxel downsample to reduce density. Default 10mm grid."""
    down = pcd.voxel_down_sample(voxel_size=voxel_size_m)
    logger.info(
        "downsampled",
        original=len(pcd.points),
        result=len(down.points),
        voxel_m=voxel_size_m,
    )
    return down


def remove_statistical_outliers(
    pcd: open3d.geometry.PointCloud,
    nb_neighbors: int = 20,
    std_ratio: float = 2.0,
):
    """
    Statistical Outlier Removal (SOR).

    Removes isolated points whose average distance to neighbours
    is significantly larger than the local mean.

    Args:
        pcd: Input point cloud
        nb_neighbors: Number of nearest neighbours
        std_ratio: Standard deviation threshold

    Returns:
        Filtered point cloud
    """

    filtered_pcd, ind = pcd.remove_statistical_outlier(
        nb_neighbors=nb_neighbors,
        std_ratio=std_ratio,
    )

    removed = len(pcd.points) - len(filtered_pcd.points)

    logger.info(
        "sor_completed",
        original_points=len(pcd.points),
        filtered_points=len(filtered_pcd.points),
        removed_points=removed,
        nb_neighbors=nb_neighbors,
        std_ratio=std_ratio,
    )

    return filtered_pcd


def sor_algorithm(pcd: open3d.geometry.PointCloud) -> open3d.geometry.PointCloud:
    """Remove statistical outliers with a neighbor count suited to cloud size."""
    point_count = len(pcd.points)
    if point_count == 0:
        return pcd

    neighbor_count = min(
        SOR_NB_NEIGHBORS,
        max(SOR_MIN_NEIGHBORS, point_count // SOR_NEIGHBOR_DIVISOR),
    )
    return remove_statistical_outliers(
        pcd,
        nb_neighbors=neighbor_count,
        std_ratio=SOR_STD_RATIO,
    )


def preprocess_point_cloud(
    file_path: Path,
    voxel_size_m: float = 0.01,
    nb_neighbors: int = 20,
    std_ratio: float = 2.0,
):
    """
    Standard scan preprocessing pipeline.

    Pipeline:
        Load → Voxel Downsample → SOR → Normal Estimation

    Args:
        file_path: Input point cloud file
        voxel_size_m: Voxel size in metres
        nb_neighbors: SOR neighbour count
        std_ratio: SOR standard deviation threshold

    Returns:
        Open3D PointCloud with normals estimated.
    """

    from agent.tools.geometry_tools import estimate_normals

    # Load scan
    pcd = load_point_cloud(file_path)

    logger.info(
        "preprocessing_started",
        file=str(file_path),
        points=len(pcd.points),
    )

    # Voxel downsampling
    pcd = downsample(
        pcd,
        voxel_size_m=voxel_size_m,
    )

    # Statistical Outlier Removal
    pcd = remove_statistical_outliers(
        pcd,
        nb_neighbors=nb_neighbors,
        std_ratio=std_ratio,
    )

    # Normal estimation
    normal_result = estimate_normals(
        pcd,
        adaptive=False,  # already voxelized
        preserve_original=True,
    )

    if not normal_result["success"]:
        raise ValueError(f"Normal estimation failed: {normal_result['error']}")

    processed_pcd = normal_result["pcd"]

    logger.info(
        "preprocessing_completed",
        points=len(processed_pcd.points),
        normals=processed_pcd.has_normals(),
    )

    return processed_pcd


def detect_planes(
    pcd: open3d.geometry.PointCloud,
    max_planes: int = MAX_PLANES,
    min_inliers: int = 200,
    distance_threshold: float = 0.025,
) -> list[dict]:
    """Iterative RANSAC plane detection. Returns list of plane info dicts."""
    planes = []
    remaining = pcd
    points_array = np.asarray(remaining.points)

    up_axis = None
    try:
        from agent.tools.coordinate_system import GLOBAL_TRANSFORM
        if hasattr(GLOBAL_TRANSFORM, "up_axis_estimated"):
            up_axis = np.asarray(GLOBAL_TRANSFORM.up_axis_estimated, dtype=float)
        elif hasattr(GLOBAL_TRANSFORM, "up_axis"):
            up_axis = np.asarray(GLOBAL_TRANSFORM.up_axis, dtype=float)
    except Exception:
        pass
    if up_axis is None:
        up_axis_name = os.environ.get("STB_UP_AXIS", "z").strip().lower()
        up_axis_map = {
            "x": np.array([1.0, 0.0, 0.0], dtype=float),
            "y": np.array([0.0, 1.0, 0.0], dtype=float),
            "z": np.array([0.0, 0.0, 1.0], dtype=float),
        }
        up_axis = up_axis_map.get(up_axis_name, up_axis_map["z"])

    # Vertical/horizontal plane thresholds measured against the configured up-axis.
    vertical_normal_up_max = float(os.environ.get("STB_VERTICAL_NORMAL_UP_MAX", "0.30"))
    horizontal_normal_up_min = float(os.environ.get("STB_HORIZONTAL_NORMAL_UP_MIN", "0.85"))

    for i in range(max_planes):
        if len(points_array) < min_inliers:
            break

        plane_model, inlier_indices = remaining.segment_plane(
            distance_threshold=distance_threshold,
            ransac_n=3,
            num_iterations=1000,
        )

        if len(inlier_indices) < min_inliers:
            break
        inlier_cloud = remaining.select_by_index(inlier_indices)
        remaining = remaining.select_by_index(inlier_indices, invert=True)
        points_array = np.asarray(remaining.points)

        a, b, c, _d = plane_model
        normal = np.array([a, b, c])
        normal = normal / np.linalg.norm(normal)

        inlier_pts = np.asarray(inlier_cloud.points)
        centroid = inlier_pts.mean(axis=0)
        mins = inlier_pts.min(axis=0)
        maxs = inlier_pts.max(axis=0)

        # Extract dominant color and variance
        dom_color, color_var = _get_dominant_color(inlier_cloud)
        # Classify plane orientation by normal component along configured up-axis:
        #   |n·up| < vertical_normal_up_max           → vertical plane
        #   |n·up| ≥ horizontal_normal_up_min         → horizontal plane
        #   otherwise                                  → sloped plane
        abs_n_up = abs(float(np.dot(normal, up_axis)))
        if abs_n_up < vertical_normal_up_max:
            shape = SegmentShape.PLANE_VERTICAL
        elif abs_n_up >= horizontal_normal_up_min:
            shape = SegmentShape.PLANE_HORIZONTAL
        else:
            shape = SegmentShape.PLANE_SLOPED

        # D3: Z-profile histogram — count stair treads for sloped planes.
        # Store step count in tags so the classifier can distinguish stair vs ramp.
        # Also tag horizontal planes with point density for grating detection.
        tags: dict = {"dominant_color": dom_color, "color_variance": color_var}
        if shape == SegmentShape.PLANE_SLOPED:
            step_count = count_stair_steps(inlier_pts)
            tags["stair_steps"] = step_count
        elif shape == SegmentShape.PLANE_HORIZONTAL:
            xy_span_m2 = max(
                (maxs[0] - mins[0]) * (maxs[1] - mins[1]), 0.01
            )  # metres² (coords are metres in open3d)
            density_per_m2 = len(inlier_pts) / xy_span_m2
            tags["point_density_per_m2"] = density_per_m2
            tags["slab_height_band"] = int(round(float(centroid[2]) / 0.20))  # 200 mm bands

            # Densitiy+height slab profiling with morphological cleanup + contour count.
            try:
                from scipy.ndimage import binary_closing, label

                grid_res = 0.10  # 10 cm
                gx = int((maxs[0] - mins[0]) / grid_res) + 1
                gy = int((maxs[1] - mins[1]) / grid_res) + 1
                if gx > 8 and gy > 8 and gx <= 500 and gy <= 500:
                    occ = np.zeros((gy, gx), dtype=bool)
                    xi = np.clip(((inlier_pts[:, 0] - mins[0]) / grid_res).astype(int), 0, gx - 1)
                    yi = np.clip(((inlier_pts[:, 1] - mins[1]) / grid_res).astype(int), 0, gy - 1)
                    occ[yi, xi] = True
                    occ = binary_closing(occ, structure=np.ones((2, 2), dtype=bool))
                    lbl, n_contours = label(occ)
                    if n_contours > 0:
                        area_cells = np.bincount(lbl.ravel())[1:]
                        primary_cells = int(area_cells.max()) if len(area_cells) else 0
                        tags["slab_contour_count"] = int(n_contours)
                        tags["slab_primary_area_m2"] = round(
                            float(primary_cells) * (grid_res**2), 3
                        )
                        tags["slab_profile_method"] = "density_morph_contour"
            except Exception as exc:
                logger.debug("slab_contour_profile_failed", error=str(exc))

            # Floor boundary extraction via 2D PCA / OBB in XY plane
            try:
                from agent.tools.geometry_tools import get_obb

                obb = get_obb(inlier_pts)
                rot = obb["rotation"]
                ext = obb["extent"]
                floor_ax = np.array(rot[:, 0], dtype=float, copy=True)
                floor_ay = np.array(rot[:, 1], dtype=float, copy=True)

                floor_ax[2] = 0.0
                norm_x = np.linalg.norm(floor_ax)
                if norm_x > 1e-6:
                    floor_ax = floor_ax / norm_x

                floor_ay[2] = 0.0
                norm_y = np.linalg.norm(floor_ay)
                if norm_y > 1e-6:
                    floor_ay = floor_ay / norm_y

                tags["floor_axis_x"] = round(float(floor_ax[0]), 6)
                tags["floor_axis_y"] = round(float(floor_ax[1]), 6)
                tags["floor_length_mm"] = round(float(ext[0] * 2 * 1000.0), 1)
                tags["floor_width_mm"] = round(float(ext[1] * 2 * 1000.0), 1)
                tags["_floor_center_x_m"] = float(obb["center"][0])
                tags["_floor_center_y_m"] = float(obb["center"][1])
                tags["floor_axis_source"] = "obb"
            except Exception as exc:
                logger.debug("floor_obb_failed", error=str(exc))

        # USIBD LOA — σ of perpendicular distance from inliers to fitted plane.
        # plane_model = (a, b, c, d) already normalized by Open3D.
        try:
            from agent.tools.loa_tools import plane_sigma_mm

            tags["loa_sigma_mm"] = round(plane_sigma_mm(inlier_pts, tuple(plane_model)), 3)
        except Exception as exc:
            logger.debug("plane_sigma_failed", error=str(exc))

        # ── Wall axis extraction for PLANE_VERTICAL ────────────────────────
        # Derive the true wall run direction from the RANSAC face normal so
        # Revit receives an axis-aligned centreline rather than an AABB diagonal.
        if shape == SegmentShape.PLANE_VERTICAL:
            from agent.tools.geometry_tools import canonicalize_wall_axis, wall_angle_degrees, refine_wall_thickness_pca
            from agent.tools.detection_config import DEFAULT_WALL_THICKNESS_MM

            wall_axis = np.cross(normal, up_axis)
            wall_axis_norm = np.linalg.norm(wall_axis)

            if wall_axis_norm > 1e-6:
                wall_axis = wall_axis / wall_axis_norm
                wall_axis_2d = canonicalize_wall_axis(wall_axis[:2])
                wall_axis = np.array([wall_axis_2d[0], wall_axis_2d[1], 0.0], dtype=float)
                angle_deg = wall_angle_degrees(wall_axis_2d)

                # Project inlier points onto wall axis → true wall length
                proj_along = inlier_pts @ wall_axis
                wall_length_m = float(proj_along.max() - proj_along.min())
                # Centreline endpoints in world metres (coord_origin not yet subtracted)
                half_len = wall_length_m / 2.0
                wall_start_m = centroid - wall_axis * half_len
                wall_end_m = centroid + wall_axis * half_len

                tags["wall_axis_x"] = round(float(wall_axis_2d[0]), 6)
                tags["wall_axis_y"] = round(float(wall_axis_2d[1]), 6)
                tags["wall_angle_deg"] = round(angle_deg, 2)
                tags["wall_length_mm"] = round(wall_length_m * 1000.0, 1)
                # PCA-based thickness measurement (Cloud2BIM method)
                _pca_thick, _pca_mode = refine_wall_thickness_pca(
                    inlier_pts, normal, wall_axis
                )
                tags["wall_thickness_mm"] = round(_pca_thick, 1)
                tags["thickness_mode"] = _pca_mode
                # Store start/end in metres; segment_to_model() applies coord_origin
                tags["_wall_start_x_m"] = float(wall_start_m[0])
                tags["_wall_start_y_m"] = float(wall_start_m[1])
                tags["_wall_end_x_m"] = float(wall_end_m[0])
                tags["_wall_end_y_m"] = float(wall_end_m[1])
                tags["wall_axis_source"] = "ransac"

                # Debug log: wall orientation angle in XY plane
                logger.debug(
                    "wall_axis_computed",
                    index=i,
                    axis_x=round(float(wall_axis_2d[0]), 4),
                    axis_y=round(float(wall_axis_2d[1]), 4),
                    angle_deg=angle_deg,
                    wall_length_mm=tags["wall_length_mm"],
                    wall_thickness_mm=tags["wall_thickness_mm"],
                )
            else:
                # Normal is nearly vertical — fall back to PCA on inlier points
                from agent.tools.geometry_tools import pca_fit as _pca_fit

                ev, evec, _ = _pca_fit(inlier_pts)
                # First principal axis (largest variance) = wall run direction
                pca_axis = evec[:, 0]
                # Remove component along configured up-axis; keep horizontal run direction.
                pca_axis = pca_axis - np.dot(pca_axis, up_axis) * up_axis
                pca_norm = np.linalg.norm(pca_axis)
                if pca_norm > 1e-6:
                    pca_axis /= pca_norm
                    pca_axis_2d = canonicalize_wall_axis(pca_axis[:2])
                    pca_axis = np.array([pca_axis_2d[0], pca_axis_2d[1], 0.0], dtype=float)
                    angle_deg = wall_angle_degrees(pca_axis_2d)

                    proj_along = inlier_pts @ pca_axis
                    wall_length_m = float(proj_along.max() - proj_along.min())
                    half_len = wall_length_m / 2.0
                    wall_start_m = centroid - pca_axis * half_len
                    wall_end_m = centroid + pca_axis * half_len

                    tags["wall_axis_x"] = round(float(pca_axis_2d[0]), 6)
                    tags["wall_axis_y"] = round(float(pca_axis_2d[1]), 6)
                    tags["wall_angle_deg"] = round(angle_deg, 2)
                    tags["wall_length_mm"] = round(wall_length_m * 1000.0, 1)
                    # PCA-based thickness measurement (Cloud2BIM method)
                    _pca_thick2, _pca_mode2 = refine_wall_thickness_pca(
                        inlier_pts, normal, pca_axis
                    )
                    tags["wall_thickness_mm"] = round(_pca_thick2, 1)
                    tags["thickness_mode"] = _pca_mode2
                    tags["_wall_start_x_m"] = float(wall_start_m[0])
                    tags["_wall_start_y_m"] = float(wall_start_m[1])
                    tags["_wall_end_x_m"] = float(wall_end_m[0])
                    tags["_wall_end_y_m"] = float(wall_end_m[1])
                    tags["wall_axis_source"] = "pca_fallback"

        planes.append(
            {
                "shape": shape,
                "normal": normal,
                "centroid": centroid,
                "mins": mins,
                "maxs": maxs,
                "point_count": len(inlier_pts),
                "confidence": min(len(inlier_pts) / 1000.0, 1.0),
                "inlier_cloud": inlier_cloud,
                "tags": tags,
            }
        )

        logger.info(
            "plane_detected",
            index=i,
            shape=shape.value,
            inliers=len(inlier_pts),
            normal_z=round(float(normal[2]), 3),
            normal_up=round(abs_n_up, 3),
            up_axis=up_axis_name,
        )

    return planes, remaining


# ── Wall segment merging ───────────────────────────────────────────────────────


def merge_coplanar_walls(
    plane_segments: list[dict],
    angle_tolerance_rad: float = 0.035,  # ≈ 2° — co-directional threshold
    gap_threshold_m: float = 0.30,  # 300 mm — endpoint gap closure
    lateral_offset_threshold_m: float = 0.20,  # 200 mm — avoid merging nearby parallel walls
) -> list[dict]:
    """Merge adjacent co-directional PLANE_VERTICAL segments into longer walls.

    Algorithm:
      1. Only processes PLANE_VERTICAL segments that carry wall axis tags.
      2. Groups segments by quantised orientation angle (±angle_tolerance_rad)
         and by storey band (Z-range overlap required).
        3. Within each group, iteratively merges endpoint-adjacent pairs whose
            run-direction gap is ≤ gap_threshold_m and whose perpendicular offset
            is ≤ lateral_offset_threshold_m.
      4. Non-wall shapes and walls without axis data pass through unchanged.

    Returns the original list with qualifying walls merged.
    """
    # Partition: walls with axis data vs everything else
    wall_segs: list[dict] = []
    other_segs: list[dict] = []
    for seg in plane_segments:
        if seg.get("shape") == SegmentShape.PLANE_VERTICAL and "wall_axis_x" in seg.get("tags", {}):
            wall_segs.append(seg)
        else:
            other_segs.append(seg)

    if not wall_segs:
        return plane_segments  # nothing to merge

    # Bucket walls by direction angle
    def _angle_bucket(seg: dict) -> float:
        ax = seg["tags"]["wall_axis_x"]
        ay = seg["tags"]["wall_axis_y"]
        angle = np.arctan2(ay, ax)
        # Fold into [0, π) — a wall and its reverse are the same direction
        if angle < 0:
            angle += np.pi
        return round(angle / angle_tolerance_rad) * angle_tolerance_rad

    from collections import defaultdict

    buckets: dict[float, list[dict]] = defaultdict(list)
    for seg in wall_segs:
        buckets[_angle_bucket(seg)].append(seg)

    merged_walls: list[dict] = []
    total_merged = 0

    for bucket_angle, group in buckets.items():
        # Within a bucket, further partition by Z storey band (floor level)
        # Two walls are on the same storey if their Z centroids are within 1 m
        group_sorted = sorted(group, key=lambda s: float(s["centroid"][2]))
        storey_groups: list[list[dict]] = []
        for seg in group_sorted:
            placed = False
            for sg in storey_groups:
                ref_seg = sg[0]
                ref_z = float(ref_seg["centroid"][2])
                min_za, max_za = float(seg["mins"][2]), float(seg["maxs"][2])
                min_zr, max_zr = float(ref_seg["mins"][2]), float(ref_seg["maxs"][2])
                overlap_z = min(max_za, max_zr) - max(min_za, min_zr)
                min_h = max(min(max_za - min_za, max_zr - min_zr), 0.01)
                z_overlap_ratio = overlap_z / min_h
                if abs(float(seg["centroid"][2]) - ref_z) <= 1.0 or z_overlap_ratio >= 0.30:
                    sg.append(seg)
                    placed = True
                    break
            if not placed:
                storey_groups.append([seg])

        for storey in storey_groups:
            if len(storey) == 1:
                merged_walls.append(storey[0])
                continue

            # Retrieve representative axis (from first segment — all co-directional)
            ref_ax = np.array(
                [
                    storey[0]["tags"]["wall_axis_x"],
                    storey[0]["tags"]["wall_axis_y"],
                    0.0,
                ]
            )
            ref_ax /= max(np.linalg.norm(ref_ax), 1e-9)
            ref_axis_xy = np.array([ref_ax[0], ref_ax[1]], dtype=float)
            ref_normal_xy = np.array([-ref_axis_xy[1], ref_axis_xy[0]], dtype=float)

            # Project each wall's midpoint onto the shared axis
            # Sort segments by projection value to process left-to-right
            def _mid_proj(seg: dict) -> float:
                c = seg["centroid"]
                return float(c[0] * ref_ax[0] + c[1] * ref_ax[1])

            storey_sorted = sorted(storey, key=_mid_proj)

            # Iterative greedy merge
            result: list[dict] = [storey_sorted[0]]
            for candidate in storey_sorted[1:]:
                prev = result[-1]

                # Compute endpoints in world metres using stored tags
                def _endpoints(s: dict):
                    sx = s["tags"].get("_wall_start_x_m", s["centroid"][0])
                    sy = s["tags"].get("_wall_start_y_m", s["centroid"][1])
                    ex = s["tags"].get("_wall_end_x_m", s["centroid"][0])
                    ey = s["tags"].get("_wall_end_y_m", s["centroid"][1])
                    return np.array([sx, sy], dtype=float), np.array([ex, ey], dtype=float)

                def _run_interval(s: np.ndarray, e: np.ndarray) -> tuple[float, float]:
                    p0 = float(np.dot(s, ref_axis_xy))
                    p1 = float(np.dot(e, ref_axis_xy))
                    return (p0, p1) if p0 <= p1 else (p1, p0)

                def _line_offset(s: np.ndarray, e: np.ndarray) -> float:
                    # Use midpoint offset to reduce endpoint jitter sensitivity.
                    return 0.5 * (float(np.dot(s, ref_normal_xy)) + float(np.dot(e, ref_normal_xy)))

                prev_s, prev_e = _endpoints(prev)
                cand_s, cand_e = _endpoints(candidate)

                prev_min, prev_max = _run_interval(prev_s, prev_e)
                cand_min, cand_max = _run_interval(cand_s, cand_e)
                if cand_min > prev_max:
                    gap = cand_min - prev_max
                elif prev_min > cand_max:
                    gap = prev_min - cand_max
                else:
                    gap = 0.0
                lateral_offset = abs(_line_offset(prev_s, prev_e) - _line_offset(cand_s, cand_e))

                if gap <= gap_threshold_m and lateral_offset <= lateral_offset_threshold_m:
                    # Merge by spanning min/max projections so reversed endpoint tags are handled.
                    endpoints = [prev_s, prev_e, cand_s, cand_e]
                    projected = sorted(
                        ((float(np.dot(pt, ref_axis_xy)), pt) for pt in endpoints), key=lambda item: item[0]
                    )
                    new_start_xy = projected[0][1]
                    new_end_xy = projected[-1][1]
                    new_length_m = float(np.linalg.norm(new_end_xy - new_start_xy))
                    new_centroid = np.array(
                        [
                            0.5 * float(new_start_xy[0] + new_end_xy[0]),
                            0.5 * float(new_start_xy[1] + new_end_xy[1]),
                            float(
                        (
                            prev["centroid"][2] * prev["point_count"]
                            + candidate["centroid"][2] * candidate["point_count"]
                        )
                        / (prev["point_count"] + candidate["point_count"])
                            ),
                        ],
                        dtype=float,
                    )

                    merged_tags = dict(prev["tags"])
                    merged_tags["wall_length_mm"] = round(new_length_m * 1000.0, 1)
                    merged_tags["_wall_start_x_m"] = float(new_start_xy[0])
                    merged_tags["_wall_start_y_m"] = float(new_start_xy[1])
                    merged_tags["_wall_end_x_m"] = float(new_end_xy[0])
                    merged_tags["_wall_end_y_m"] = float(new_end_xy[1])
                    merged_tags["wall_merged"] = True

                    merged_seg = dict(prev)
                    merged_seg["centroid"] = new_centroid.copy()
                    merged_seg["point_count"] = prev["point_count"] + candidate["point_count"]
                    merged_seg["confidence"] = max(prev["confidence"], candidate["confidence"])
                    # Expand AABB
                    merged_seg["mins"] = np.minimum(prev["mins"], candidate["mins"])
                    merged_seg["maxs"] = np.maximum(prev["maxs"], candidate["maxs"])
                    merged_seg["tags"] = merged_tags

                    # Merge inlier points and cloud if present
                    pts_to_merge = []
                    for s_item in (prev, candidate):
                        if "points" in s_item and s_item["points"] is not None:
                            pts_to_merge.append(np.asarray(s_item["points"]))
                        elif s_item.get("inlier_cloud") is not None:
                            pts_to_merge.append(np.asarray(s_item["inlier_cloud"].points))
                    if pts_to_merge:
                        comb_pts = np.vstack(pts_to_merge)
                        merged_seg["points"] = comb_pts
                        try:
                            import open3d as _o3d
                            comb_pcd = _o3d.geometry.PointCloud()
                            comb_pcd.points = _o3d.utility.Vector3dVector(comb_pts)
                            merged_seg["inlier_cloud"] = comb_pcd
                        except Exception:
                            pass

                    result[-1] = merged_seg
                    total_merged += 1
                else:
                    result.append(candidate)

            merged_walls.extend(result)
    if total_merged > 0:
        logger.info(
            "wall_merge_complete",
            input_walls=len(wall_segs),
            output_walls=len(merged_walls),
            merges_performed=total_merged,
        )

    return other_segs + merged_walls


def merge_walls_hybrid(
    segments: list[dict],
    angle_bucket_deg: float = 5.0,
    offset_bucket_m: float = 0.20,
    storey_bucket_m: float = 1.20,
    run_gap_threshold_m: float = 0.45,
    lateral_offset_threshold_m: float = 0.20,
) -> list[dict]:
    """Hybrid wall merge using wall orientation + room/surface offset bins.

    This complements endpoint-only merging by grouping collinear wall surfaces
    that represent the same room boundary but were fragmented by scan gaps.
    """
    walls = [
        s
        for s in segments
        if s.get("shape") == SegmentShape.PLANE_VERTICAL and "wall_axis_x" in s.get("tags", {})
    ]
    others = [
        s
        for s in segments
        if s.get("shape") != SegmentShape.PLANE_VERTICAL or "wall_axis_x" not in s.get("tags", {})
    ]
    if len(walls) < 2:
        return segments

    from collections import defaultdict

    buckets: dict[tuple[int, int, int], list[dict]] = defaultdict(list)

    for w in walls:
        ax = float(w["tags"].get("wall_axis_x", 0.0))
        ay = float(w["tags"].get("wall_axis_y", 0.0))
        a_norm = max(np.linalg.norm([ax, ay]), 1e-9)
        ax, ay = ax / a_norm, ay / a_norm

        # Surface-normal offset: groups walls lying on same boundary line.
        nx, ny = -ay, ax
        c = w["centroid"]
        offset = float(nx * c[0] + ny * c[1])
        angle = float(np.degrees(np.arctan2(ay, ax)))
        if angle < 0.0:
            angle += 180.0
        storey = float(c[2])

        key = (
            int(round(angle / max(angle_bucket_deg, 1e-6))),
            int(round(offset / max(offset_bucket_m, 1e-6))),
            int(round(storey / max(storey_bucket_m, 1e-6))),
        )
        buckets[key].append(w)

    def _endpoints(seg: dict) -> tuple[np.ndarray, np.ndarray]:
        tags = seg.get("tags", {})
        sx = float(tags.get("_wall_start_x_m", seg["centroid"][0]))
        sy = float(tags.get("_wall_start_y_m", seg["centroid"][1]))
        ex = float(tags.get("_wall_end_x_m", seg["centroid"][0]))
        ey = float(tags.get("_wall_end_y_m", seg["centroid"][1]))
        return np.array([sx, sy], dtype=float), np.array([ex, ey], dtype=float)

    def _merge_chain(parts: list[dict], axis: np.ndarray) -> dict:
        base = dict(parts[0])
        base["mins"] = np.minimum.reduce([p["mins"] for p in parts])
        base["maxs"] = np.maximum.reduce([p["maxs"] for p in parts])
        total_pts = sum(int(p["point_count"]) for p in parts)
        base["point_count"] = total_pts
        base["confidence"] = max(float(p["confidence"]) for p in parts)
        base["centroid"] = sum(p["centroid"] * float(p["point_count"]) for p in parts) / max(
            total_pts, 1
        )

        # Keep precise wall run endpoints for downstream centerline export.
        endpoint_pairs = [_endpoints(p) for p in parts]
        projected = []
        for start, end in endpoint_pairs:
            projected.append((float(np.dot(start, axis)), start))
            projected.append((float(np.dot(end, axis)), end))
        projected.sort(key=lambda item: item[0])
        start_pt = projected[0][1]
        end_pt = projected[-1][1]

        base_tags = dict(base.get("tags", {}))
        base_tags["wall_hybrid_merged"] = True
        base_tags["wall_hybrid_group_size"] = len(parts)
        base_tags["_wall_start_x_m"] = float(start_pt[0])
        base_tags["_wall_start_y_m"] = float(start_pt[1])
        base_tags["_wall_end_x_m"] = float(end_pt[0])
        base_tags["_wall_end_y_m"] = float(end_pt[1])
        base_tags["wall_length_mm"] = round(float(np.linalg.norm(end_pt - start_pt)) * 1000.0, 1)
        base["tags"] = base_tags

        # Merge inlier points and cloud if present
        pts_to_merge = []
        for p in parts:
            if "points" in p and p["points"] is not None:
                pts_to_merge.append(np.asarray(p["points"]))
            elif p.get("inlier_cloud") is not None:
                pts_to_merge.append(np.asarray(p["inlier_cloud"].points))
        if pts_to_merge:
            comb_pts = np.vstack(pts_to_merge)
            base["points"] = comb_pts
            try:
                import open3d as _o3d
                comb_pcd = _o3d.geometry.PointCloud()
                comb_pcd.points = _o3d.utility.Vector3dVector(comb_pts)
                base["inlier_cloud"] = comb_pcd
            except Exception:
                pass

        return base

    merged: list[dict] = []
    for group in buckets.values():
        if len(group) == 1:
            merged.append(group[0])
            continue

        # Sort along wall run axis so only close fragments merge into one boundary.
        ax = float(group[0]["tags"].get("wall_axis_x", 1.0))
        ay = float(group[0]["tags"].get("wall_axis_y", 0.0))
        axis = np.array([ax, ay], dtype=float)
        axis /= max(np.linalg.norm(axis), 1e-9)

        def _proj(seg: dict) -> float:
            c = seg["centroid"]
            return float(c[0] * axis[0] + c[1] * axis[1])

        parts = sorted(group, key=_proj)

        # Merge only adjacent fragments; do not collapse distinct walls that share
        # orientation/offset bucket but are separated along the run direction.
        chains: list[list[dict]] = [[parts[0]]]
        for candidate in parts[1:]:
            prev = chains[-1][-1]
            prev_s, prev_e = _endpoints(prev)
            cand_s, cand_e = _endpoints(candidate)

            prev_proj = sorted([float(np.dot(prev_s, axis)), float(np.dot(prev_e, axis))])
            cand_proj = sorted([float(np.dot(cand_s, axis)), float(np.dot(cand_e, axis))])
            gap = cand_proj[0] - prev_proj[1]
            normal = np.array([-axis[1], axis[0]], dtype=float)
            prev_offset = 0.5 * (float(np.dot(prev_s, normal)) + float(np.dot(prev_e, normal)))
            cand_offset = 0.5 * (float(np.dot(cand_s, normal)) + float(np.dot(cand_e, normal)))
            lateral_offset = abs(prev_offset - cand_offset)

            if gap <= run_gap_threshold_m and lateral_offset <= lateral_offset_threshold_m:
                chains[-1].append(candidate)
            else:
                chains.append([candidate])

        for chain in chains:
            if len(chain) == 1:
                merged.append(chain[0])
            else:
                merged.append(_merge_chain(chain, axis))

    return others + merged


def merge_walls(
    segments: list[dict],
    angle_tolerance_rad: float = 0.087,
    gap_threshold_m: float = 1.20,
    lateral_offset_threshold_m: float = 0.20,
) -> list[dict]:
    """Production wall merge function using Stage 2 Union-Find grouping and face pairing.

    Replaces the legacy 5° / 1.2m bucket greedy merger with the robust Stage 2
    Union-Find grouping and opposing face pairing pipeline.
    Preserves other non-wall segments unchanged.
    """
    from agent.stage2_pipeline import group_wall_fragments, pair_opposing_wall_faces

    wall_segs = [s for s in segments if s.get("shape") == SegmentShape.PLANE_VERTICAL]
    other_segs = [s for s in segments if s.get("shape") != SegmentShape.PLANE_VERTICAL]

    if not wall_segs:
        return segments

    # 1. Group coplanar/collinear wall fragments using Union-Find
    grouped = group_wall_fragments(
        wall_segs,
        angle_tol_rad=angle_tolerance_rad,
        lateral_offset_m=lateral_offset_threshold_m,
        fragment_gap_m=gap_threshold_m,
    )

    # 2. Pair opposing wall faces with controlled thickness
    physical = pair_opposing_wall_faces(
        grouped,
        max_thickness_m=0.45,
    )

    return other_segs + physical


def merge_coplanar_horizontal(
    segments: list[dict],
    z_tolerance_m: float = 0.05,
    gap_threshold_m: float = 0.50,
) -> list[dict]:
    """Merge horizontal planes that share the same Z-elevation and are adjacent."""
    horizontal_segs = [s for s in segments if s["shape"] == SegmentShape.PLANE_HORIZONTAL]
    other_segs = [s for s in segments if s["shape"] != SegmentShape.PLANE_HORIZONTAL]

    if not horizontal_segs:
        return other_segs

    floors = []
    ceilings = []
    for s in horizontal_segs:
        if s["normal"] is not None and s["normal"][2] > 0:
            floors.append(s)
        else:
            ceilings.append(s)

    merged_horizontal = []
    total_merged = 0

    import math

    for group in (floors, ceilings):
        if not group:
            continue

        group.sort(key=lambda s: s["centroid"][2])

        z_groups = []
        current_z_group = [group[0]]
        current_z = group[0]["centroid"][2]

        for s in group[1:]:
            z = s["centroid"][2]
            if abs(z - current_z) <= z_tolerance_m:
                current_z_group.append(s)
            else:
                z_groups.append(current_z_group)
                current_z_group = [s]
                current_z = z
        z_groups.append(current_z_group)

        for z_group in z_groups:
            if len(z_group) == 1:
                merged_horizontal.append(z_group[0])
                continue

            result = [z_group[0]]
            for cand in z_group[1:]:
                merged = False
                for i, prev in enumerate(result):
                    dx = max(
                        0.0,
                        max(prev["mins"][0], cand["mins"][0])
                        - min(prev["maxs"][0], cand["maxs"][0]),
                    )
                    dy = max(
                        0.0,
                        max(prev["mins"][1], cand["mins"][1])
                        - min(prev["maxs"][1], cand["maxs"][1]),
                    )
                    gap = math.sqrt(dx * dx + dy * dy)

                    if gap <= gap_threshold_m:
                        new_centroid = (
                            prev["centroid"] * prev["point_count"]
                            + cand["centroid"] * cand["point_count"]
                        ) / (prev["point_count"] + cand["point_count"])

                        merged_seg = dict(prev)
                        merged_seg["centroid"] = new_centroid.copy()
                        merged_seg["point_count"] = prev["point_count"] + cand["point_count"]
                        merged_seg["confidence"] = max(prev["confidence"], cand["confidence"])
                        merged_seg["mins"] = np.minimum(prev["mins"], cand["mins"])
                        merged_seg["maxs"] = np.maximum(prev["maxs"], cand["maxs"])

                        merged_tags = dict(prev.get("tags", {}))
                        merged_tags["floor_merged"] = True

                        if (
                            "floor_axis_x" in prev.get("tags", {})
                            and prev["point_count"] >= cand["point_count"]
                        ):
                            merged_tags["floor_length_mm"] = round(
                                float(merged_seg["maxs"][0] - merged_seg["mins"][0]) * 1000.0, 1
                            )
                            merged_tags["floor_width_mm"] = round(
                                float(merged_seg["maxs"][1] - merged_seg["mins"][1]) * 1000.0, 1
                            )
                            merged_tags["_floor_center_x_m"] = float(
                                (merged_seg["maxs"][0] + merged_seg["mins"][0]) / 2.0
                            )
                            merged_tags["_floor_center_y_m"] = float(
                                (merged_seg["maxs"][1] + merged_seg["mins"][1]) / 2.0
                            )
                            merged_tags["floor_axis_x"] = 1.0
                            merged_tags["floor_axis_y"] = 0.0
                            merged_tags["floor_axis_source"] = "merged_aabb"

                        merged_seg["tags"] = merged_tags
                        result[i] = merged_seg
                        total_merged += 1
                        merged = True
                        break

                if not merged:
                    result.append(cand)

            merged_horizontal.extend(result)

    if total_merged > 0:
        logger.info(
            "horizontal_merge_complete",
            input_planes=len(horizontal_segs),
            output_planes=len(merged_horizontal),
            merges_performed=total_merged,
        )

    return other_segs + merged_horizontal


def merge_collinear_cylinders(
    segments: list[dict],
    angle_tolerance_deg: float = 3.0,
    radius_tolerance_pct: float = 0.20,
    gap_threshold_m: float = 1.0,
) -> list[dict]:
    """Chain adjacent CYLINDER segments with collinear axes and similar radii."""

    opt_in = os.environ.get("STB_ENABLE_CYLINDER_CHAIN_MERGE", "0").strip().lower() in {
        "1",
        "true",
        "yes",
        "on",
    }
    disable_merge = os.environ.get("STB_DISABLE_CYLINDER_CHAIN_MERGE", "0").strip().lower() in {
        "1",
        "true",
        "yes",
        "on",
    }
    if disable_merge:
        return segments
    if not opt_in and os.environ.get("STB_ENABLE_CYLINDER_CHAIN_MERGE") is not None:
        return segments

    import math

    def _segment_line(seg: dict) -> tuple[np.ndarray, np.ndarray] | None:
        tags = seg.get("tags", {}) or {}
        start_keys = ("_cyl_start_x_m", "_cyl_start_y_m", "_cyl_start_z_m")
        end_keys = ("_cyl_end_x_m", "_cyl_end_y_m", "_cyl_end_z_m")
        start_vals = [tags.get(k) for k in start_keys]
        end_vals = [tags.get(k) for k in end_keys]
        if all(v is not None for v in start_vals + end_vals):
            try:
                start = np.array(
                    [float(start_vals[0]), float(start_vals[1]), float(start_vals[2])], dtype=float
                )
                end = np.array(
                    [float(end_vals[0]), float(end_vals[1]), float(end_vals[2])], dtype=float
                )
                if np.linalg.norm(end - start) > 1e-9:
                    return start, end
            except (TypeError, ValueError):
                pass

        axis = np.array(
            [
                float(tags.get("cyl_axis_x", 0.0)),
                float(tags.get("cyl_axis_y", 0.0)),
                float(tags.get("cyl_axis_z", 0.0)),
            ],
            dtype=float,
        )
        axis_norm = np.linalg.norm(axis)
        if axis_norm <= 1e-9:
            return None
        axis = axis / axis_norm

        centroid = np.asarray(seg.get("centroid", np.zeros(3)), dtype=float)
        mins = np.asarray(seg.get("mins", np.zeros(3)), dtype=float)
        maxs = np.asarray(seg.get("maxs", np.zeros(3)), dtype=float)
        extents = maxs - mins
        proj_len = float(np.dot(extents, np.abs(axis)))
        half_len = max(proj_len * 0.5, 1e-6)
        return centroid - axis * half_len, centroid + axis * half_len

    def _segment_extent(line: tuple[np.ndarray, np.ndarray]) -> float:
        return float(np.linalg.norm(line[1] - line[0]))

    def _axial_gap(
        a_line: tuple[np.ndarray, np.ndarray],
        b_line: tuple[np.ndarray, np.ndarray],
        axis: np.ndarray,
    ) -> float:
        a_start, a_end = a_line
        b_start, b_end = b_line
        a_proj = np.array([float(np.dot(a_start, axis)), float(np.dot(a_end, axis))], dtype=float)
        b_proj = np.array([float(np.dot(b_start, axis)), float(np.dot(b_end, axis))], dtype=float)
        a_min, a_max = float(np.min(a_proj)), float(np.max(a_proj))
        b_min, b_max = float(np.min(b_proj)), float(np.max(b_proj))
        if a_max < b_min:
            return b_min - a_max
        if b_max < a_min:
            return a_min - b_max
        return 0.0

    def _perpendicular_offset(
        a_line: tuple[np.ndarray, np.ndarray],
        b_line: tuple[np.ndarray, np.ndarray],
        axis: np.ndarray,
    ) -> float:
        a_start, a_end = a_line
        b_start, b_end = b_line
        a_vec = a_end - a_start
        b_vec = b_end - b_start
        if np.linalg.norm(a_vec) <= 1e-9 or np.linalg.norm(b_vec) <= 1e-9:
            return float(np.linalg.norm(np.cross(b_start - a_start, axis)))
        return float(np.linalg.norm(np.cross(b_start - a_start, axis)))

    cylinders = [s for s in segments if s["shape"] == SegmentShape.CYLINDER]
    other_segs = [s for s in segments if s["shape"] != SegmentShape.CYLINDER]

    if not cylinders:
        return other_segs

    merged = []
    used = set()
    total_merged = 0
    angle_tol = math.cos(math.radians(angle_tolerance_deg))

    for i, curr in enumerate(cylinders):
        if i in used:
            continue

        group = [curr]
        used.add(i)

        c_tags = curr.get("tags", {})
        if "cyl_axis_x" not in c_tags or "fitted_radius_mm" not in c_tags:
            merged.append(curr)
            continue

        c_axis = np.array([c_tags["cyl_axis_x"], c_tags["cyl_axis_y"], c_tags["cyl_axis_z"]])
        c_rad = c_tags["fitted_radius_mm"]
        c_line = _segment_line(curr)

        for j in range(i + 1, len(cylinders)):
            if j in used:
                continue

            cand = cylinders[j]
            cand_tags = cand.get("tags", {})
            if "cyl_axis_x" not in cand_tags or "fitted_radius_mm" not in cand_tags:
                continue

            cand_axis = np.array(
                [cand_tags["cyl_axis_x"], cand_tags["cyl_axis_y"], cand_tags["cyl_axis_z"]]
            )
            cand_rad = cand_tags["fitted_radius_mm"]
            cand_line = _segment_line(cand)

            if abs(c_rad - cand_rad) / max(c_rad, 1.0) > radius_tolerance_pct:
                continue

            dot = abs(np.dot(c_axis, cand_axis))
            if dot < angle_tol:
                continue

            if c_line is None or cand_line is None:
                continue

            axis = c_axis / max(np.linalg.norm(c_axis), 1e-9)
            perp_offset = _perpendicular_offset(c_line, cand_line, axis)
            radius_mean_m = 0.5 * ((c_rad / 1000.0) + (cand_rad / 1000.0))
            gap_limit_m = max(0.05, radius_mean_m * 2.5) + max(0.0, gap_threshold_m)
            if perp_offset > max(radius_mean_m * 2.5, 0.05):
                continue

            c_len = _segment_extent(c_line)
            cand_len = _segment_extent(cand_line)
            axial_gap = _axial_gap(c_line, cand_line, axis)
            if axial_gap <= gap_limit_m + 0.5 * min(c_len, cand_len):
                group.append(cand)
                used.add(j)

        if len(group) == 1:
            merged.append(group[0])
        else:
            base = dict(group[0])
            all_pts_count = sum(s["point_count"] for s in group)
            new_centroid = sum(s["centroid"] * s["point_count"] for s in group) / all_pts_count

            base["centroid"] = new_centroid
            base["point_count"] = all_pts_count
            base["confidence"] = max(s["confidence"] for s in group)

            mins = np.min([s["mins"] for s in group], axis=0)
            maxs = np.max([s["maxs"] for s in group], axis=0)
            base["mins"] = mins
            base["maxs"] = maxs

            if "tags" not in base or base["tags"] is None:
                base["tags"] = {}
            else:
                base["tags"] = dict(base["tags"])
            base["tags"]["cylinder_merged"] = True

            merged.append(base)
            total_merged += len(group) - 1

    if total_merged > 0:
        logger.info(
            "cylinder_merge_complete",
            input_cylinders=len(cylinders),
            output_cylinders=len(merged),
            merges_performed=total_merged,
        )

    return other_segs + merged


def merge_collinear_boxes(segments: list[dict], gap_m: float = 0.5) -> list[dict]:
    """Merge adjacent/collinear BOX segments (Stairs, Railings, Beams).

    Uses centroid proximity and bounding box alignment to consolidate fragmented
    linear assemblies before Revit generation.
    """

    enable_merge = os.environ.get("STB_ENABLE_BOX_CHAIN_MERGE", "0").strip().lower() in {
        "1",
        "true",
        "yes",
        "on",
    }
    if not enable_merge:
        return segments

    if len(segments) < 2:
        return segments

    from agent.models import SegmentShape

    mergable_shapes = {SegmentShape.BOX, SegmentShape.PLANE_SLOPED}

    def _is_linear_candidate(seg: dict) -> bool:
        mins = np.asarray(seg.get("mins"), dtype=float)
        maxs = np.asarray(seg.get("maxs"), dtype=float)
        dims_mm = np.maximum((maxs - mins) * 1000.0, 0.0)
        dims_mm.sort()
        minor, mid, major = dims_mm
        # Merge only elongated runs; keep compact components (bins, extinguishers,
        # boxes around doors) as independent elements.
        if major < 700.0:
            return False
        if mid < 1.0:
            return False
        if (major / mid) < 2.2:
            return False
        if minor > 600.0:
            return False
        return True

    candidates = [s for s in segments if s["shape"] in mergable_shapes and _is_linear_candidate(s)]
    others = [
        s for s in segments if s["shape"] not in mergable_shapes or not _is_linear_candidate(s)
    ]

    if not candidates:
        return segments

    merged_indices = set()
    result = []

    for i in range(len(candidates)):
        if i in merged_indices:
            continue

        curr = candidates[i]
        for j in range(i + 1, len(candidates)):
            if j in merged_indices:
                continue

            other = candidates[j]

            # Distance check
            dist = np.linalg.norm(curr["centroid"] - other["centroid"])
            if dist > 3.0:  # Skip far away clusters
                continue

            # Overlap or adjacency check
            inter_min = np.maximum(curr["mins"], other["mins"])
            inter_max = np.minimum(curr["maxs"], other["maxs"])
            inter_dims = inter_max - inter_min

            # If they overlap or are very close
            is_near = False
            if np.all(inter_dims > -gap_m):
                is_near = True

            if is_near:
                # Merge curr into other or vice versa
                curr["mins"] = np.minimum(curr["mins"], other["mins"])
                curr["maxs"] = np.maximum(curr["maxs"], other["maxs"])
                curr["centroid"] = (curr["mins"] + curr["maxs"]) / 2.0
                curr["point_count"] += other["point_count"]
                curr["confidence"] = max(curr["confidence"], other["confidence"])
                merged_indices.add(j)

        result.append(curr)

    return others + result


def segregate_by_color(
    pcd: open3d.geometry.PointCloud,
) -> tuple[list[open3d.geometry.PointCloud], list[str]]:
    """Segregate a point cloud into per-color tracing groups + grayscale structure.

    Workflow:
      1) Extract dominant color buckets from the full cloud (quantized RGB)
      2) Create one sub-cloud per dominant color for tracing
      3) Route remaining low-saturation points to grayscale structural group
      4) Keep any leftover colored points in a misc colored group
    """
    if not pcd.has_colors():
        logger.warning("color_trace_unavailable", reason="point_cloud_has_no_rgb")
        return [pcd], ["uncolored"]

    colors = np.asarray(pcd.colors)
    if len(colors) == 0:
        logger.warning("color_trace_unavailable", reason="empty_rgb_array")
        return [pcd], ["uncolored"]

    c_max = colors.max(axis=1)
    c_min = colors.min(axis=1)
    saturation = c_max - c_min

    # Use a lower saturation threshold so lightly tinted scan data is not
    # collapsed into grayscale-only runs.
    mask_color = saturation > 0.05
    mask_gray = ~mask_color

    # Quantize to 32-step RGB buckets (0..224) to extract robust dominant colours
    # from noisy scan colour measurements.
    q = np.clip((colors * 255.0).astype(np.int32), 0, 255)
    q = (q // 32) * 32

    # Extract dominant colored buckets from saturated points.
    sat_idx = np.where(mask_color)[0]
    dominant_keys: list[tuple[int, int, int]] = []
    if len(sat_idx) > 0:
        q_sat = q[sat_idx]
        keys, counts = np.unique(q_sat, axis=0, return_counts=True)
        order = np.argsort(counts)[::-1]
        total_sat = max(len(sat_idx), 1)

        for idx in order:
            key = tuple(int(v) for v in keys[idx])
            count = int(counts[idx])
            ratio = count / total_sat
            # Keep meaningful colour buckets; ignore tiny noise colours.
            if count < 120 and ratio < 0.01:
                continue
            dominant_keys.append(key)
            if len(dominant_keys) >= 12:
                break

    # If no dominant saturated colors were found, still try to extract major
    # buckets from the full cloud (common for low-saturation scanner exports).
    if not dominant_keys:
        keys_all, counts_all = np.unique(q, axis=0, return_counts=True)
        order_all = np.argsort(counts_all)[::-1]
        total = max(len(colors), 1)
        for idx in order_all:
            key = tuple(int(v) for v in keys_all[idx])
            count = int(counts_all[idx])
            ratio = count / total
            if count < 150 and ratio < 0.015:
                continue
            dominant_keys.append(key)
            if len(dominant_keys) >= 8:
                break

    clouds = []
    tags = []

    assigned = np.zeros(len(colors), dtype=bool)

    # Per-colour groups first (trace coloured runs/objects before classification)
    for r, g, b in dominant_keys:
        mask = (q[:, 0] == r) & (q[:, 1] == g) & (q[:, 2] == b) & mask_color & (~assigned)
        idx = np.where(mask)[0]
        if len(idx) < 100:
            continue
        clouds.append(pcd.select_by_index(idx.tolist()))
        tags.append(f"color_{r:02x}{g:02x}{b:02x}")
        assigned[idx] = True

    # Remaining colored points (not dominant enough to be individual groups)
    idx_color_misc = np.where(mask_color & (~assigned))[0]
    if len(idx_color_misc) > 100:
        clouds.append(pcd.select_by_index(idx_color_misc.tolist()))
        tags.append("colored_misc")
        assigned[idx_color_misc] = True

    idx_gray = np.where(mask_gray & (~assigned))[0]
    if len(idx_gray) > 100:
        clouds.append(pcd.select_by_index(idx_gray.tolist()))
        tags.append("grayscale_structural")

    if not clouds:
        return [pcd], ["uncolored"]

    logger.info("color_groups_extracted", groups=len(tags), tags=tags)

    return clouds, tags


def detect_clusters(
    pcd: open3d.geometry.PointCloud,
    min_cluster_size: int = 20,
) -> list[dict]:
    """C1 3-pass DBSCAN clustering on residual points after plane extraction.

    Pass 1 eps=0.10 m — large MEP equipment, tanks, HVAC units.
    Pass 2 eps=0.05 m — standard pipes, ducts, beams.
    Pass 3 eps=0.02 m — fine detail: conduits, small valves, sprinklers.

    Shape classification uses B3 PCA (eigendecomposition) and B4 OBB.
    B2 RANSAC cylinder refines CYLINDER detections when normals are available.
    B5 RANSAC sphere catches micro-compact fixtures (valves, sprinkler heads).
    """
    points = np.asarray(pcd.points)
    if len(points) < min_cluster_size:
        return []

    has_normals = pcd.has_normals()
    normals_arr = np.asarray(pcd.normals) if has_normals else None

    # C1: 3-pass DBSCAN — collect all cluster point-sets with deduplication
    claimed = np.zeros(len(points), dtype=bool)
    all_cluster_pts: list[tuple[np.ndarray, np.ndarray | None]] = []  # (pts, normals_or_none)

    for eps in (0.10, 0.05, 0.02):
        unclaimed_idx = np.where(~claimed)[0]
        if len(unclaimed_idx) < min_cluster_size:
            break

        sub_pcd = pcd.select_by_index(unclaimed_idx.tolist())
        labels = np.array(
            sub_pcd.cluster_dbscan(eps=eps, min_points=min_cluster_size, print_progress=False)
        )
        if len(labels) == 0:
            continue

        for label in set(labels) - {-1}:
            mask = labels == label
            local_pts = np.asarray(sub_pcd.points)[mask]
            if len(local_pts) < min_cluster_size:
                continue
            # Map back to global indices and mark claimed
            global_indices = unclaimed_idx[mask]
            claimed[global_indices] = True
            local_normals = normals_arr[global_indices] if has_normals else None
            all_cluster_pts.append((local_pts, local_normals))

        logger.debug("dbscan_pass", eps=eps, new_clusters=len(all_cluster_pts))

    results = []

    for cluster_pts, cluster_normals in all_cluster_pts:
        _fitted_radius_mm = None  # set by RANSAC cylinder when confirmed
        _cyl_fit = None  # full RANSAC cylinder dict for LOA σ computation

        # B3: PCA for shape signatures
        try:
            eigenvalues, _, _ = pca_fit(cluster_pts)
            linearity = pca_linearity(eigenvalues)
        except np.linalg.LinAlgError:
            linearity = 0.0

        # B4: OBB for accurate extents (falls back to AABB on failure)
        try:
            obb_info = get_obb(cluster_pts)
            extents = obb_info["extent"]  # sorted descending
        except Exception:
            extents_raw = cluster_pts.max(axis=0) - cluster_pts.min(axis=0)
            extents = np.sort(extents_raw)[::-1]

        minor1, minor2, major = float(extents[2]), float(extents[1]), float(extents[0])
        max_other = max(minor1, minor2, 0.001)
        centroid = cluster_pts.mean(axis=0)

        # Use AABB for bounding box min/max (OBB centre + AABB for range queries)
        mins = cluster_pts.min(axis=0)
        maxs = cluster_pts.max(axis=0)
        aabb_extents = maxs - mins

        is_elongated = (major >= 3 * max_other) or (linearity > 0.65)
        is_compact_equipment = (not is_elongated) and (minor1 >= 0.20)

        if is_elongated:
            dominant_axis = int(np.argmax(aabb_extents))
            other_extents = [aabb_extents[i] for i in range(3) if i != dominant_axis]
            cross_ratio = max(other_extents) / max(min(other_extents), 0.001)

            is_rectangular = cross_ratio > 1.5

            # B2: RANSAC cylinder refinement for elongated round clusters
            if is_rectangular or not has_normals:
                shape = SegmentShape.BOX
                confidence = min(len(cluster_pts) / 400.0, 1.0)
                z_c = float(centroid[2])
                z_tag = "railing" if 0.8 <= z_c <= 1.2 else ("duct" if z_c > 1.5 else "beam")
                logger.debug(
                    f"{z_tag}_cluster", points=len(cluster_pts), centroid_z_m=round(z_c, 2)
                )
            else:
                # Attempt RANSAC cylinder fitting to confirm round cross-section
                cyl = ransac_cylinder_fit(cluster_pts, cluster_normals) if has_normals else None
                if cyl is not None:
                    shape = SegmentShape.CYLINDER
                    confidence = min(cyl["inliers"] / max(len(cluster_pts), 1) * 1.2, 1.0)
                    # Store RANSAC-fitted radius for Revit geometry (avoids AABB overestimate)
                    _fitted_radius_mm = round(cyl["radius"] * 1000.0, 1)
                    _cyl_fit = cyl  # captured for LOA σ computation below
                    logger.debug(
                        "cylinder_ransac_confirmed",
                        radius_m=round(cyl["radius"], 3),
                        inliers=cyl["inliers"],
                    )
                else:
                    shape = SegmentShape.BOX
                    confidence = min(len(cluster_pts) / 500.0, 1.0)
                    _fitted_radius_mm = None
                    logger.debug("cylinder_fit_failed_fallback_to_box", points=len(cluster_pts))

        elif is_compact_equipment:
            shape = SegmentShape.BOX
            confidence = min(len(cluster_pts) / 600.0, 1.0)
            logger.debug(
                "equipment_cluster",
                points=len(cluster_pts),
                extents_m=[round(float(e), 2) for e in aabb_extents],
            )

        elif major < 0.25 and len(cluster_pts) >= 8:
            # Micro-fixture — try B5 sphere fitting first
            sph = ransac_sphere_fit(cluster_pts) if len(cluster_pts) >= 10 else None
            if sph is not None:
                shape = SegmentShape.CYLINDER  # classifier will sub-classify
                confidence = min(sph["inliers"] / max(len(cluster_pts), 1) * 1.1, 0.85)
                logger.debug(
                    "sphere_fixture",
                    radius_m=round(sph["radius"], 3),
                    centroid_z_m=round(float(centroid[2]), 2),
                )
            else:
                shape = SegmentShape.CYLINDER
                confidence = min(len(cluster_pts) / 30.0, 0.75)
                z_tag = "sprinkler" if float(centroid[2]) > 2.5 else "floor_fixture"
                logger.debug(
                    f"micro_fixture_{z_tag}",
                    points=len(cluster_pts),
                    centroid_z_m=round(float(centroid[2]), 2),
                )
        else:
            continue  # too small / ambiguous

        tags: dict = {}
        # Preserve RANSAC-fitted radius for downstream geometry builders
        if _fitted_radius_mm is not None:
            tags["fitted_radius_mm"] = _fitted_radius_mm

        if _cyl_fit is not None:
            try:
                cyl_axis = np.asarray(_cyl_fit["axis"], dtype=float)
                cyl_norm = np.linalg.norm(cyl_axis)
                if cyl_norm > 1e-9:
                    cyl_axis = cyl_axis / cyl_norm
                    # Force cylinder axis to be horizontal or vertical (orthogonal alignment)
                    abs_axis = np.abs(cyl_axis)
                    max_idx = np.argmax(abs_axis)
                    ortho_axis = np.zeros(3)
                    ortho_axis[max_idx] = np.sign(cyl_axis[max_idx])
                    cyl_axis = ortho_axis
                else:
                    cyl_axis = np.array([1.0, 0.0, 0.0])  # fallback

                tags["cyl_axis_x"] = float(cyl_axis[0])
                tags["cyl_axis_y"] = float(cyl_axis[1])
                tags["cyl_axis_z"] = float(cyl_axis[2])

                # Persist explicit cylinder endpoints in metres for downstream BIM
                # creation so pipe runs use fitted axis extents instead of AABB fallback.
                proj_t = (cluster_pts - centroid) @ cyl_axis
                t_min = float(proj_t.min())
                t_max = float(proj_t.max())
                cyl_start = centroid + cyl_axis * t_min
                cyl_end = centroid + cyl_axis * t_max
                tags["_cyl_start_x_m"] = float(cyl_start[0])
                tags["_cyl_start_y_m"] = float(cyl_start[1])
                tags["_cyl_start_z_m"] = float(cyl_start[2])
                tags["_cyl_end_x_m"] = float(cyl_end[0])
                tags["_cyl_end_y_m"] = float(cyl_end[1])
                tags["_cyl_end_z_m"] = float(cyl_end[2])
            except Exception as exc:
                logger.debug("cylinder_endpoint_tagging_failed", error=str(exc))
                tags["cyl_axis_x"] = float(_cyl_fit["axis"][0])
                tags["cyl_axis_y"] = float(_cyl_fit["axis"][1])
                tags["cyl_axis_z"] = float(_cyl_fit["axis"][2])

        # USIBD LOA — σ from fit residual.
        # Cylinders: use RANSAC cylinder residual (available when _cyl was set)
        # Boxes / other: use AABB nearest-face σ as a conservative proxy
        try:
            from agent.tools.loa_tools import aabb_sigma_mm, cylinder_sigma_mm

            if shape == SegmentShape.CYLINDER and _cyl_fit is not None:
                # Any point on the axis works for perpendicular-distance σ;
                # the cluster centroid lies arbitrarily close to the axis for
                # well-fit pipes, so it's a safe reference.
                sigma = cylinder_sigma_mm(
                    cluster_pts,
                    axis_point_m=centroid,
                    axis_direction=np.array(_cyl_fit["axis"]),
                    radius_m=_cyl_fit["radius"],
                )
            else:
                sigma = aabb_sigma_mm(cluster_pts)
            tags["loa_sigma_mm"] = round(float(sigma), 3)
        except Exception as exc:
            logger.debug("cluster_sigma_failed", error=str(exc))

        # Extract color for the cluster
        sub_pcd = pcd.select_by_index(
            np.where(np.all(np.isin(points, cluster_pts), axis=1))[0].tolist()
        )
        dom_color, color_var = _get_dominant_color(sub_pcd)

        results.append(
            {
                "shape": shape,
                "normal": None,
                "centroid": centroid,
                "mins": mins,
                "maxs": maxs,
                "point_count": len(cluster_pts),
                "confidence": confidence,
                "tags": {**tags, "dominant_color": dom_color, "color_variance": color_var},
            }
        )

    return results


def detect_valves(
    pipe_clusters: list[dict],
    residual_pcd: open3d.geometry.PointCloud,
) -> list[dict]:
    """Detect inline valves by topology: compact clusters co-located on pipe axes.

    Algorithm:
    1. Re-cluster the residual point cloud with tight DBSCAN (eps=3 cm, min_pts=5)
       to resolve small bodies that the coarser pass missed.
    2. Keep only compact candidates (all extents 0.05–0.50 m).
    3. E1: Use SegmentSpatialIndex for O(log n) proximity check — is this cluster
       near any known pipe axis (within 3× pipe radius and pipe length + 0.5 m margin)?
    4. Survivors are emitted as VALVE_CANDIDATE so the classifier can map them
       directly to ElementType.VALVE without further geometry checks.
    """
    if not pipe_clusters:
        return []

    pts = np.asarray(residual_pcd.points)
    if len(pts) < 5:
        return []

    labels = np.array(residual_pcd.cluster_dbscan(eps=0.03, min_points=5, print_progress=False))

    # E1: Build spatial index over pipe centroids for fast proximity queries
    pipe_index = SegmentSpatialIndex(pipe_clusters)

    valves = []
    for label in set(labels) - {-1}:
        mask = labels == label
        cluster_pts = pts[mask]

        mins = cluster_pts.min(axis=0)
        maxs = cluster_pts.max(axis=0)
        extents = maxs - mins  # metres
        centroid = cluster_pts.mean(axis=0)

        # Must be a compact body — not a full pipe run, not noise
        if not all(0.05 <= float(e) <= 0.50 for e in extents):
            continue

        # Geometric check: reject slender collinear fragments (likely pipe segments)
        if len(cluster_pts) >= 10:
            cov = np.cov(cluster_pts, rowvar=False)
            evals = np.linalg.eigvalsh(cov)
            evals = np.sort(evals)[::-1]
            tot = float(np.sum(evals))
            if tot > 1e-9:
                linearity = (evals[0] - evals[1]) / tot
                if linearity > 0.88:
                    continue  # Slender collinear run, not a valve body/flange

        # E1: Spatial index query — candidate pipe neighbours within 2 m
        nearby_pipe_indices = pipe_index.query_radius(centroid, radius=2.0)
        if not nearby_pipe_indices:
            continue

        # Proximity check: is this cluster on a pipe axis?
        on_pipe = False
        for pipe_idx in nearby_pipe_indices:
            pipe = pipe_clusters[pipe_idx]
            p_mins = pipe["mins"]
            p_maxs = pipe["maxs"]
            p_ext = p_maxs - p_mins
            p_centroid = pipe["centroid"]

            dom_ax = int(np.argmax(p_ext))
            cross_axes = [i for i in range(3) if i != dom_ax]

            # Cross-section radius estimate from minor extents
            p_radius = max(float(p_ext[cross_axes[0]]), float(p_ext[cross_axes[1]]), 0.02) / 2.0

            # Distance from valve centroid to pipe centreline (cross-section plane)
            cross_dist = float(
                np.sqrt(sum((centroid[ax] - p_centroid[ax]) ** 2 for ax in cross_axes))
            )

            # How far along the pipe axis is the valve?
            along_dist = abs(float(centroid[dom_ax]) - float(p_centroid[dom_ax]))
            pipe_half_len = float(p_ext[dom_ax]) / 2.0

            # Accept if: laterally within 3× pipe radius AND axially within pipe + 0.5 m margin
            if cross_dist <= max(p_radius * 3.0, 0.25) and along_dist <= pipe_half_len + 0.50:
                on_pipe = True
                break

        if on_pipe:
            confidence = min(len(cluster_pts) / 25.0, 0.78)
            valves.append(
                {
                    "shape": SegmentShape.VALVE_CANDIDATE,
                    "normal": None,
                    "centroid": centroid,
                    "mins": mins,
                    "maxs": maxs,
                    "point_count": len(cluster_pts),
                    "confidence": confidence,
                    "tags": {},
                }
            )
            logger.info(
                "valve_detected",
                points=len(cluster_pts),
                centroid_z_m=round(float(centroid[2]), 2),
                extents_m=[round(float(e), 3) for e in extents],
            )

    return valves


def _detect_slab_openings(plane: dict, grid_res: float, min_cells: int) -> list[dict]:
    """Find rectangular voids (penetrations) in a horizontal floor/ceiling slab.

    Projects inlier points to XY, builds an occupancy grid, then finds rectangular
    empty regions — these are slab openings (service penetrations, lift shafts, etc.).
    """
    inlier_cloud = plane.get("inlier_cloud")
    if inlier_cloud is None:
        return []

    pts = np.asarray(inlier_cloud.points)
    if len(pts) < 100:
        return []

    x_coords = pts[:, 0]
    y_coords = pts[:, 1]
    z_mid = float(pts[:, 2].mean())

    x_min, x_max = x_coords.min(), x_coords.max()
    y_min, y_max = y_coords.min(), y_coords.max()

    if (x_max - x_min) < 0.5 or (y_max - y_min) < 0.5:
        return []

    gx = int((x_max - x_min) / grid_res) + 1
    gy = int((y_max - y_min) / grid_res) + 1

    if gx > 400 or gy > 400:
        return []

    occ = np.zeros((gy, gx), dtype=bool)
    xi = np.clip(((x_coords - x_min) / grid_res).astype(int), 0, gx - 1)
    yi = np.clip(((y_coords - y_min) / grid_res).astype(int), 0, gy - 1)
    occ[yi, xi] = True

    # D2: Morphological gap-filling — close small scan holes before void detection.
    # Erosion+dilation (closing) fills 1-cell gaps caused by sparse scan coverage
    # without merging true slab openings that are typically ≥ 3 cells wide.
    try:
        from scipy.ndimage import binary_closing

        struct = np.ones((2, 2), dtype=bool)  # 2-cell structuring element
        occ = binary_closing(occ, structure=struct)
    except ImportError:
        pass  # scipy unavailable — proceed without gap-filling

    # Find contiguous empty rectangular regions in 2D
    slab_plan_area_m2 = float((x_max - x_min) * (y_max - y_min))
    slab_thickness = float(plane["maxs"][2] - plane["mins"][2])
    min_opening_area_m2 = max(0.01, float(os.environ.get("STB_MIN_SLAB_OPENING_AREA_M2", "0.30")))
    min_opening_side_m = max(0.10, float(os.environ.get("STB_MIN_SLAB_OPENING_SIDE_M", "0.35")))
    max_opening_span_m = max(
        min_opening_side_m, float(os.environ.get("STB_MAX_SLAB_OPENING_SPAN_M", "2.40"))
    )
    max_opening_area_ratio = min(
        0.95, max(0.01, float(os.environ.get("STB_MAX_SLAB_OPENING_AREA_RATIO", "0.30")))
    )
    max_opening_aspect = max(1.0, float(os.environ.get("STB_MAX_SLAB_OPENING_ASPECT", "6.0")))
    max_openings_per_plane = max(1, int(os.environ.get("STB_MAX_SLAB_OPENINGS_PER_PLANE", "4")))
    nms_iou_threshold = min(
        0.95, max(0.10, float(os.environ.get("STB_SLAB_OPENING_NMS_IOU", "0.45")))
    )
    results = []

    for row in range(gy):
        for col in range(gx):
            if occ[row, col]:
                continue  # occupied — not a void start

            # Expand rectangle as far as possible
            r_end = row
            while r_end < gy and not occ[r_end, col]:
                r_end += 1
            c_end = col
            while c_end < gx and not any(occ[row:r_end, c_end]):
                c_end += 1

            h_cells = r_end - row
            w_cells = c_end - col
            if h_cells < min_cells or w_cells < min_cells:
                continue

            # Mark as occupied so we don't re-detect
            occ[row:r_end, col:c_end] = True

            cx = x_min + (col + w_cells / 2) * grid_res
            cy = y_min + (row + h_cells / 2) * grid_res
            wx = w_cells * grid_res
            wy = h_cells * grid_res
            t = max(slab_thickness, 0.10)

            area_m2 = float(wx * wy)
            if min(wx, wy) < min_opening_side_m or max(wx, wy) > max_opening_span_m:
                continue
            if slab_plan_area_m2 > 1e-6 and (area_m2 / slab_plan_area_m2) > max_opening_area_ratio:
                continue
            aspect = max(wx, wy) / max(min(wx, wy), 1e-6)
            if area_m2 < min_opening_area_m2 or aspect > max_opening_aspect:
                continue

            results.append(
                {
                    "shape": SegmentShape.VOID,
                    "normal": np.array([0.0, 0.0, 1.0]),
                    "centroid": np.array([cx, cy, z_mid]),
                    "mins": np.array([cx - wx / 2, cy - wy / 2, z_mid - t / 2]),
                    "maxs": np.array([cx + wx / 2, cy + wy / 2, z_mid + t / 2]),
                    "point_count": 0,
                    "confidence": min(0.5 + (w_cells * h_cells) / 1000.0, 0.85),
                    "tags": {
                        "opening_area_m2": round(area_m2, 3),
                        "opening_aspect_ratio": round(float(aspect), 3),
                    },
                }
            )
            logger.info(
                "slab_opening_detected",
                width_m=round(wx, 2),
                depth_m=round(wy, 2),
                z_m=round(z_mid, 2),
            )

    if len(results) <= max_openings_per_plane:
        return results

    # Suppress duplicates caused by sparse/noisy occupancy holes.
    def _xy_iou(a: dict, b: dict) -> float:
        a_min, a_max = a["mins"], a["maxs"]
        b_min, b_max = b["mins"], b["maxs"]
        ix = max(0.0, min(float(a_max[0]), float(b_max[0])) - max(float(a_min[0]), float(b_min[0])))
        iy = max(0.0, min(float(a_max[1]), float(b_max[1])) - max(float(a_min[1]), float(b_min[1])))
        inter = ix * iy
        if inter <= 0.0:
            return 0.0
        a_area = max(0.0, (float(a_max[0]) - float(a_min[0])) * (float(a_max[1]) - float(a_min[1])))
        b_area = max(0.0, (float(b_max[0]) - float(b_min[0])) * (float(b_max[1]) - float(b_min[1])))
        denom = a_area + b_area - inter
        return inter / denom if denom > 1e-9 else 0.0

    ranked = sorted(
        results,
        key=lambda item: float((item.get("tags") or {}).get("opening_area_m2", 0.0)),
        reverse=True,
    )
    pruned: list[dict] = []
    for candidate in ranked:
        if any(_xy_iou(candidate, kept) >= nms_iou_threshold for kept in pruned):
            continue
        pruned.append(candidate)
        if len(pruned) >= max_openings_per_plane:
            break

    logger.info("slab_opening_pruned", detected=len(results), kept=len(pruned), z_m=round(z_mid, 2))
    return pruned


def detect_openings(planes: list[dict]) -> list[dict]:
    """Find door/window openings in vertical planes and slab openings in horizontal planes.

    Vertical planes  → door (void reaches floor) or window (floating void).
    Horizontal planes → slab opening / penetration (void through a floor or ceiling slab).

    Algorithm:
    1. For each vertical RANSAC plane, project inlier points onto the wall's 2D face
       (u = along-wall horizontal, v = vertical/z).
    2. Build a 10 cm occupancy grid on that face.
    3. Scan column-by-column for contiguous empty vertical runs (≥ 1 m tall, ≥ 30 cm wide).
    4. Merge horizontally adjacent void columns into rectangular openings.
    5. Emit each void as a VOID-shape segment; min_z tells the classifier whether
       the opening is a door (min_z ≈ floor) or a window (floating sill).
    """
    MIN_VOID_W_CELLS = 3  # ≥ 30 cm wide
    MIN_VOID_H_CELLS = 10  # ≥ 1 m tall  (vertical walls)
    MIN_SLAB_CELLS = 3  # ≥ 30 cm in each direction (slab openings)
    GRID_RES = 0.10  # metres
    UP = np.array([0.0, 0.0, 1.0])

    openings = []

    def _xy_iou(a: dict, b: dict) -> float:
        a_min, a_max = a["mins"], a["maxs"]
        b_min, b_max = b["mins"], b["maxs"]
        ix = max(0.0, min(float(a_max[0]), float(b_max[0])) - max(float(a_min[0]), float(b_min[0])))
        iy = max(0.0, min(float(a_max[1]), float(b_max[1])) - max(float(a_min[1]), float(b_min[1])))
        inter = ix * iy
        if inter <= 0.0:
            return 0.0
        a_area = max(0.0, (float(a_max[0]) - float(a_min[0])) * (float(a_max[1]) - float(a_min[1])))
        b_area = max(0.0, (float(b_max[0]) - float(b_min[0])) * (float(b_max[1]) - float(b_min[1])))
        denom = a_area + b_area - inter
        return inter / denom if denom > 1e-9 else 0.0

    for plane in planes:
        if plane["shape"] == SegmentShape.PLANE_HORIZONTAL:
            openings.extend(_detect_slab_openings(plane, GRID_RES, MIN_SLAB_CELLS))
            continue

        if plane["shape"] != SegmentShape.PLANE_VERTICAL:
            continue

        inlier_cloud = plane.get("inlier_cloud")
        if inlier_cloud is not None:
            pts = np.asarray(inlier_cloud.points)
        elif "points" in plane and plane["points"] is not None:
            pts = np.asarray(plane["points"])
        else:
            continue
        if len(pts) < 100:
            continue

        normal = plane["normal"]
        centroid = plane["centroid"]

        # Wall-local coordinate axes
        u_axis = np.cross(normal, UP)
        u_norm = np.linalg.norm(u_axis)
        if u_norm < 1e-6:
            continue
        u_axis /= u_norm

        u_coords = (pts - centroid) @ u_axis
        v_coords = pts[:, 2]

        u_min, u_max = u_coords.min(), u_coords.max()
        v_min, v_max = v_coords.min(), v_coords.max()
        u_span = u_max - u_min
        v_span = v_max - v_min

        if u_span < 0.5 or v_span < 0.5:
            continue

        # Adapt grid resolution for very large walls so opening detection does not
        # silently skip them.
        grid_res = GRID_RES
        grid_u = int(u_span / grid_res) + 1
        grid_v = int(v_span / grid_res) + 1

        while (grid_u > 400 or grid_v > 250) and grid_res < 0.50:
            grid_res *= 1.5
            grid_u = int(u_span / grid_res) + 1
            grid_v = int(v_span / grid_res) + 1

        # Final cap to avoid huge allocations in pathological scenes.
        if grid_u > 800 or grid_v > 500:
            continue

        # Build boolean occupancy grid (True = point present)
        occ = np.zeros((grid_v, grid_u), dtype=bool)
        ui = np.clip(((u_coords - u_min) / grid_res).astype(int), 0, grid_u - 1)
        vi = np.clip(((v_coords - v_min) / grid_res).astype(int), 0, grid_v - 1)
        occ[vi, ui] = True

        # D2: Morphological closing to fill 1-cell scan gaps on wall faces.
        # Prevents false void detection from sparse scan coverage.
        try:
            from scipy.ndimage import binary_closing

            struct = np.ones((2, 2), dtype=bool)
            occ = binary_closing(occ, structure=struct)
        except ImportError:
            pass

        # Per-column: find the longest empty vertical run
        col_voids: list[tuple[int, int] | None] = []
        for col in range(grid_u):
            empty = np.where(~occ[:, col])[0]
            if len(empty) < MIN_VOID_H_CELLS:
                col_voids.append(None)
                continue
            # Split into contiguous runs, keep longest
            splits = np.split(empty, np.where(np.diff(empty) > 1)[0] + 1)
            longest = max(splits, key=len)
            if len(longest) >= MIN_VOID_H_CELLS:
                col_voids.append((int(longest[0]), int(longest[-1])))
            else:
                col_voids.append(None)

        # Merge adjacent columns whose empty runs overlap
        i = 0
        while i < grid_u:
            if col_voids[i] is None:
                i += 1
                continue

            r_min, r_max = col_voids[i]
            j = i + 1
            while j < grid_u and col_voids[j] is not None:
                cr_min, cr_max = col_voids[j]
                overlap_min = max(r_min, cr_min)
                overlap_max = min(r_max, cr_max)
                if overlap_max - overlap_min >= MIN_VOID_H_CELLS:
                    r_min, r_max = overlap_min, overlap_max
                    j += 1
                else:
                    break

            w_cells = j - i
            h_cells = r_max - r_min

            if w_cells >= MIN_VOID_W_CELLS and h_cells >= MIN_VOID_H_CELLS:
                u_centre_m = u_min + (i + w_cells / 2) * grid_res
                v_bottom_m = v_min + r_min * grid_res
                v_top_m = v_min + r_max * grid_res
                v_centre_m = (v_bottom_m + v_top_m) / 2

                # Reconstruct 3D void centre
                void_centre = centroid.copy()
                void_centre += u_centre_m * u_axis
                void_centre[2] = v_centre_m

                # Estimate wall thickness from its minor extents
                extent = plane["maxs"] - plane["mins"]
                wall_t = max(float(np.sort(extent)[0]), 0.10)  # metres

                w_m = w_cells * grid_res
                h_m = h_cells * grid_res
                sill_h_m = max(v_bottom_m - v_min, 0.0)
                aspect = w_m / max(h_m, 1e-6)

                # Histogram-derived opening heuristics (sill + aspect + dimensions).
                is_door_like = (
                    sill_h_m <= 0.20
                    and 0.60 <= w_m <= 2.20
                    and 1.80 <= h_m <= 2.80
                    and 0.22 <= aspect <= 1.20
                )
                is_window_like = (
                    0.40 <= sill_h_m <= 1.60
                    and 0.30 <= w_m <= 3.20
                    and 0.30 <= h_m <= 2.20
                    and 0.20 <= aspect <= 4.50
                )
                if not (is_door_like or is_window_like):
                    i = j if j > i else i + 1
                    continue

                hw, hd, hh = w_m / 2, wall_t / 2, h_m / 2

                mins_3d = void_centre - np.array([hw, hd, hh])
                maxs_3d = void_centre + np.array([hw, hd, hh])
                # Correct Z to actual grid extents (not symmetric about centre)
                mins_3d[2] = v_bottom_m
                maxs_3d[2] = v_top_m

                confidence = min(0.5 + (w_cells * h_cells) / 2000.0, 0.88)
                opening_tag = "door" if v_bottom_m < (v_min + 0.15) else "window"

                plane_tags = plane.get("tags", {})
                host_wall_id = str(plane.get("segment_id") or plane.get("id") or plane_tags.get("segment_id", ""))
                wall_ax = float(plane_tags.get("wall_axis_x", u_axis[0]))
                wall_ay = float(plane_tags.get("wall_axis_y", u_axis[1]))
                wall_ang = float(plane_tags.get("wall_angle_deg", 0.0))

                start_x_m = float(plane_tags.get("_wall_start_x_m", centroid[0]))
                start_y_m = float(plane_tags.get("_wall_start_y_m", centroid[1]))
                start_vec = np.array([start_x_m, start_y_m], dtype=float)
                ax_vec = np.array([wall_ax, wall_ay], dtype=float)
                if np.linalg.norm(ax_vec) > 1e-6:
                    ax_vec /= np.linalg.norm(ax_vec)
                along_wall_dist_mm = round(float(np.dot(void_centre[:2] - start_vec, ax_vec)) * 1000.0, 1)

                openings.append(
                    {
                        "shape": SegmentShape.VOID,
                        "normal": normal,
                        "centroid": void_centre,
                        "mins": mins_3d,
                        "maxs": maxs_3d,
                        "point_count": 0,
                        "confidence": confidence,
                        # wall_thickness_mm drives containment_penetration / penetration_seal rules
                        "tags": {
                            "host_wall_id": host_wall_id,
                            "wall_axis_x": round(wall_ax, 6),
                            "wall_axis_y": round(wall_ay, 6),
                            "wall_angle_deg": round(wall_ang, 2),
                            "opening_dist_along_wall_mm": along_wall_dist_mm,
                            "wall_thickness_mm": wall_t * 1000.0,
                            "opening_width_mm": round(w_m * 1000.0, 1),
                            "opening_height_mm": round(h_m * 1000.0, 1),
                            "opening_bottom_z_mm": round(v_bottom_m * 1000.0, 1),
                            "opening_top_z_mm": round(v_top_m * 1000.0, 1),
                            "opening_sill_mm": round(sill_h_m * 1000.0, 1),
                            "opening_aspect_ratio": round(float(aspect), 3),
                            "opening_histogram_method": "grid_runs_v2",
                        },
                    }
                )

                logger.info(
                    "opening_detected",
                    type=opening_tag,
                    width_m=round(w_m, 2),
                    height_m=round(h_m, 2),
                    v_bottom_m=round(v_bottom_m, 2),
                )

            i = j if j > i else i + 1

    # Global suppression for slab openings across multiple horizontal planes.
    max_total_slab_openings = max(1, int(os.environ.get("STB_MAX_SLAB_OPENINGS_TOTAL", "8")))
    slab_global_nms_iou = min(
        0.95, max(0.10, float(os.environ.get("STB_SLAB_OPENING_GLOBAL_NMS_IOU", "0.35")))
    )

    slab_openings = [
        o
        for o in openings
        if "opening_area_m2" in (o.get("tags") or {})
        and "wall_thickness_mm" not in (o.get("tags") or {})
    ]
    if len(slab_openings) > max_total_slab_openings:
        ranked = sorted(
            slab_openings,
            key=lambda item: float((item.get("tags") or {}).get("opening_area_m2", 0.0)),
            reverse=True,
        )
        kept_slab: list[dict] = []
        for candidate in ranked:
            if any(_xy_iou(candidate, kept) >= slab_global_nms_iou for kept in kept_slab):
                continue
            kept_slab.append(candidate)
            if len(kept_slab) >= max_total_slab_openings:
                break

        kept_ids = {id(item) for item in kept_slab}
        openings = [
            o
            for o in openings
            if "opening_area_m2" not in (o.get("tags") or {})
            or "wall_thickness_mm" in (o.get("tags") or {})
            or id(o) in kept_ids
        ]
        logger.info(
            "slab_opening_global_pruned",
            detected=len(slab_openings),
            kept=len(kept_slab),
            total_openings=len(openings),
        )

    return openings


def assign_surface_zones(segments: list[dict]) -> None:
    """Assign sub-zone ids using actual wall surfaces (not wall center axes)."""
    wall_surfaces = []
    for idx, seg in enumerate(segments):
        if seg.get("shape") != SegmentShape.PLANE_VERTICAL:
            continue
        mins = seg.get("mins")
        maxs = seg.get("maxs")
        if mins is None or maxs is None:
            continue
        span_x = float(maxs[0] - mins[0])
        span_y = float(maxs[1] - mins[1])
        if max(span_x, span_y) < 0.8:
            continue
        run_axis = "y" if span_y >= span_x else "x"
        wall_surfaces.append(
            {
                "id": f"ws_{idx:04d}",
                "run_axis": run_axis,
                "x": float(seg["centroid"][0]),
                "y": float(seg["centroid"][1]),
                "z": float(seg["centroid"][2]),
                "mins": mins,
                "maxs": maxs,
            }
        )

    if len(wall_surfaces) < 4:
        return

    def _overlaps_1d(a0: float, a1: float, b0: float, b1: float, tol: float = 0.20) -> bool:
        return min(a1, b1) >= max(a0, b0) - tol

    for seg in segments:
        c = seg.get("centroid")
        if c is None:
            continue
        cx, cy, cz = float(c[0]), float(c[1]), float(c[2])
        storey_key = int(round(cz / 3.0))

        west = east = south = north = None
        for ws in wall_surfaces:
            wz_key = int(round(float(ws["z"]) / 3.0))
            if abs(wz_key - storey_key) > 1:
                continue

            mins = ws["mins"]
            maxs = ws["maxs"]
            if ws["run_axis"] == "y" and _overlaps_1d(mins[1], maxs[1], cy - 0.5, cy + 0.5):
                wx = float(ws["x"])
                if wx <= cx and (west is None or wx > west[0]):
                    west = (wx, ws["id"])
                if wx >= cx and (east is None or wx < east[0]):
                    east = (wx, ws["id"])
            if ws["run_axis"] == "x" and _overlaps_1d(mins[0], maxs[0], cx - 0.5, cx + 0.5):
                wy = float(ws["y"])
                if wy <= cy and (south is None or wy > south[0]):
                    south = (wy, ws["id"])
                if wy >= cy and (north is None or wy < north[0]):
                    north = (wy, ws["id"])

        if west and east and south and north:
            zone_surface_id = f"zone_s{storey_key}_{west[1]}_{east[1]}_{south[1]}_{north[1]}"
            tags = seg.setdefault("tags", {})
            tags["zone_surface_id"] = zone_surface_id
            tags["zone_surface_method"] = "wall_surface_bounded"
            tags["zone_storey_key"] = storey_key
            # Placeholder non-graphical attributes for IFC/CDE workflows.
            tags.setdefault("zone_acoustic_class", "unspecified")
            tags.setdefault("zone_thermal_class", "unspecified")
            tags.setdefault("zone_usage_class", "unspecified")


def _to_mm(meters: np.ndarray) -> np.ndarray:
    """Convert metres to millimetres."""
    return meters * 1000.0


def _get_dominant_color(pcd: open3d.geometry.PointCloud) -> tuple[str | None, float | None]:
    """Calculate the average color and variance of a point cloud. Returns (Hex, Variance)."""
    if not pcd.has_colors():
        return None, None

    colors = np.asarray(pcd.colors)
    if len(colors) == 0:
        return None, None

    avg_rgb = np.mean(colors, axis=0)
    variance = float(np.mean(np.var(colors, axis=0)))

    # Convert to Hex
    r, g, b = (avg_rgb * 255).astype(int)
    hex_color = f"#{r:02x}{g:02x}{b:02x}"

    return hex_color, round(variance, 4)


def segment_to_model(
    seg: dict,
    zone_id: str,
    scan_station_id: str | None = None,
    source_file: str | None = None,
    coord_origin_m: np.ndarray | None = None,
) -> GeometrySegment:
    """Convert a raw segment dict to a GeometrySegment model (in mm).

    coord_origin_m: survey-coordinate offset in metres to subtract before
    converting to mm. Pass the cloud's min-XYZ so Revit receives coordinates
    near the origin regardless of the scanner's absolute survey position.
    """
    origin = coord_origin_m if coord_origin_m is not None else np.zeros(3)
    mins_mm = _to_mm(seg["mins"] - origin)
    maxs_mm = _to_mm(seg["maxs"] - origin)
    centroid_mm = _to_mm(seg["centroid"] - origin)

    normal = None
    if seg.get("normal") is not None:
        n = seg["normal"]
        normal = Point3D(x=float(n[0]), y=float(n[1]), z=float(n[2]))

    # USIBD LOA — derive tier from σ before serialising
    tags = dict(seg.get("tags", {}))

    # ── Wall axis coordinate correction ──────────────────────────────────────
    # detect_planes() stored _wall_start/end_*_m in world metres.
    # Apply the same coord_origin_m subtraction used for centroid/mins/maxs,
    # then convert to mm and publish as the final public tag names.
    _WALL_M_KEYS = ("_wall_start_x_m", "_wall_start_y_m", "_wall_end_x_m", "_wall_end_y_m")
    _WALL_MM_KEYS = ("wall_start_x_mm", "wall_start_y_mm", "wall_end_x_mm", "wall_end_y_mm")
    _ORIGIN_IDX = (0, 1, 0, 1)  # x,y,x,y index into origin array

    if all(k in tags for k in _WALL_M_KEYS):
        for m_key, mm_key, oi in zip(_WALL_M_KEYS, _WALL_MM_KEYS, _ORIGIN_IDX):
            corrected_m = tags.pop(m_key) - float(origin[oi])
            tags[mm_key] = round(corrected_m * 1000.0, 1)
    else:
        # Clean up any partial private keys that may have been set
        for m_key in _WALL_M_KEYS:
            tags.pop(m_key, None)

    # Revit wall contract hardening: make sure every vertical plane exported as a
    # wall-like segment has usable centerline + thickness metadata.
    if seg.get("shape") == SegmentShape.PLANE_VERTICAL:
        from agent.tools.geometry_tools import canonicalize_wall_axis, wall_angle_degrees
        from agent.tools.detection_config import (
            DEFAULT_WALL_THICKNESS_MM,
            MAX_WALL_THICKNESS_MM,
            MIN_WALL_THICKNESS_MM,
        )

        have_endpoints = all(k in tags for k in _WALL_MM_KEYS)
        have_axis = ("wall_axis_x" in tags and "wall_axis_y" in tags)

        dx_mm = float(maxs_mm[0] - mins_mm[0])
        dy_mm = float(maxs_mm[1] - mins_mm[1])

        if not have_axis:
            n = seg.get("normal")
            if n is not None:
                n_xy = np.array([float(n[0]), float(n[1])], dtype=float)
                n_norm = float(np.linalg.norm(n_xy))
            else:
                n_xy = np.zeros(2, dtype=float)
                n_norm = 0.0

            if n_norm > 1e-9:
                ax = -n_xy[1] / n_norm
                ay = n_xy[0] / n_norm
            else:
                if dx_mm >= dy_mm:
                    ax, ay = 1.0, 0.0
                else:
                    ax, ay = 0.0, 1.0

            c_axis = canonicalize_wall_axis([ax, ay])
            tags["wall_axis_x"] = round(float(c_axis[0]), 6)
            tags["wall_axis_y"] = round(float(c_axis[1]), 6)
            tags.setdefault("wall_axis_source", "bbox_fallback")
        else:
            c_axis = canonicalize_wall_axis([float(tags["wall_axis_x"]), float(tags["wall_axis_y"])])
            tags["wall_axis_x"] = round(float(c_axis[0]), 6)
            tags["wall_axis_y"] = round(float(c_axis[1]), 6)

        tags["wall_angle_deg"] = wall_angle_degrees([tags["wall_axis_x"], tags["wall_axis_y"]])

        if not have_endpoints:
            ax = float(tags["wall_axis_x"])
            ay = float(tags["wall_axis_y"])
            axis = np.array([ax, ay], dtype=float)
            cxy = np.array([float(centroid_mm[0]), float(centroid_mm[1])], dtype=float)
            corners = np.array(
                [
                    [float(mins_mm[0]), float(mins_mm[1])],
                    [float(mins_mm[0]), float(maxs_mm[1])],
                    [float(maxs_mm[0]), float(mins_mm[1])],
                    [float(maxs_mm[0]), float(maxs_mm[1])],
                ],
                dtype=float,
            )
            proj = corners @ axis
            length_mm = max(float(proj.max() - proj.min()), 100.0)
            half = 0.5 * length_mm
            start = cxy - axis * half
            end = cxy + axis * half

            tags["wall_start_x_mm"] = round(float(start[0]), 1)
            tags["wall_start_y_mm"] = round(float(start[1]), 1)
            tags["wall_end_x_mm"] = round(float(end[0]), 1)
            tags["wall_end_y_mm"] = round(float(end[1]), 1)
            tags.setdefault("wall_length_mm", round(float(length_mm), 1))

        if "wall_start_x_mm" in tags and "wall_start_y_mm" in tags:
            z_mid_mm = float(centroid_mm[2])
            tags["start_xyz_mm"] = [float(tags["wall_start_x_mm"]), float(tags["wall_start_y_mm"]), z_mid_mm]
            tags["end_xyz_mm"] = [float(tags["wall_end_x_mm"]), float(tags["wall_end_y_mm"]), z_mid_mm]

        # Bound wall thickness
        if "wall_thickness_mm" in tags:
            try:
                curr_t = float(tags["wall_thickness_mm"])
                tags["wall_thickness_mm"] = round(
                    float(np.clip(curr_t, MIN_WALL_THICKNESS_MM, MAX_WALL_THICKNESS_MM)), 1
                )
            except (TypeError, ValueError):
                tags["wall_thickness_mm"] = DEFAULT_WALL_THICKNESS_MM
        else:
            # Check axis-alignment: if axis-aligned, min(dx, dy) represents true face extent
            is_axis_aligned = (
                abs(abs(float(tags.get("wall_axis_x", 0.0))) - 1.0) < 0.05
                or abs(float(tags.get("wall_axis_x", 0.0))) < 0.05
            )
            aabb_extent_mm = min(dx_mm, dy_mm)
            if is_axis_aligned and MIN_WALL_THICKNESS_MM <= aabb_extent_mm <= 800.0:
                tags["wall_thickness_mm"] = round(aabb_extent_mm, 1)
                tags["thickness_mode"] = "aabb_face_extent"
            else:
                # Angled wall or single-face: AABB min(dx,dy) is diagonal envelope (can be 10m+).
                # Fall back to default wall thickness.
                tags["wall_thickness_mm"] = round(
                    float(np.clip(DEFAULT_WALL_THICKNESS_MM, MIN_WALL_THICKNESS_MM, MAX_WALL_THICKNESS_MM)), 1
                )
                tags["thickness_mode"] = "inferred_single_face"

    # ── Cylinder axis endpoint coordinate correction ───────────────────────
    # detect_clusters() stores _cyl_start/end_*_m in world metres.
    # Apply coord-origin subtraction and expose public *_mm tags for Revit.
    _CYL_M_KEYS = (
        "_cyl_start_x_m",
        "_cyl_start_y_m",
        "_cyl_start_z_m",
        "_cyl_end_x_m",
        "_cyl_end_y_m",
        "_cyl_end_z_m",
    )
    _CYL_MM_KEYS = (
        "cyl_start_x_mm",
        "cyl_start_y_mm",
        "cyl_start_z_mm",
        "cyl_end_x_mm",
        "cyl_end_y_mm",
        "cyl_end_z_mm",
    )
    _CYL_ORIGIN_IDX = (0, 1, 2, 0, 1, 2)

    if all(k in tags for k in _CYL_M_KEYS):
        for m_key, mm_key, oi in zip(_CYL_M_KEYS, _CYL_MM_KEYS, _CYL_ORIGIN_IDX):
            corrected_m = tags.pop(m_key) - float(origin[oi])
            tags[mm_key] = round(corrected_m * 1000.0, 1)
    else:
        for m_key in _CYL_M_KEYS:
            tags.pop(m_key, None)

    # ── Floor OBB coordinate correction ────────────────────────────────────
    if "_floor_center_x_m" in tags and "_floor_center_y_m" in tags:
        c_x = tags.pop("_floor_center_x_m") - float(origin[0])
        c_y = tags.pop("_floor_center_y_m") - float(origin[1])
        tags["floor_center_x_mm"] = round(c_x * 1000.0, 1)
        tags["floor_center_y_mm"] = round(c_y * 1000.0, 1)
    if "loa_sigma_mm" in tags:
        try:
            from agent.tools.loa_tools import classify_loa_tier

            tier = classify_loa_tier(float(tags["loa_sigma_mm"]))
            tags["loa_tier"] = tier.value
        except Exception:
            pass  # leave sigma alone; consumer can recompute tier

    # ── Semantic Classification ──────────────────────────────────────────────
    # Create a temporary segment for classification
    temp_seg = GeometrySegment(
        zone_id=zone_id,
        shape=seg["shape"],
        normal=normal,
        centroid=Point3D(x=float(centroid_mm[0]), y=float(centroid_mm[1]), z=float(centroid_mm[2])),
        bounding_box=BoundingBox(
            min_x=float(mins_mm[0]),
            min_y=float(mins_mm[1]),
            min_z=float(mins_mm[2]),
            max_x=float(maxs_mm[0]),
            max_y=float(maxs_mm[1]),
            max_z=float(maxs_mm[2]),
        ),
        point_count=seg["point_count"],
        confidence=seg["confidence"],
        tags=tags,
    )

    sub_segments = split_segment_for_revit(temp_seg)

    models = []

    for sub_seg in sub_segments:
        try:
            element_type, _discipline, safety = classify_segment(sub_seg)

            models.append(
                GeometrySegment(
                    segment_id=sub_seg.segment_id,
                    zone_id=sub_seg.zone_id,
                    shape=sub_seg.shape,
                    normal=sub_seg.normal,
                    centroid=sub_seg.centroid,
                    bounding_box=sub_seg.bounding_box,
                    point_count=sub_seg.point_count,
                    confidence=sub_seg.confidence,
                    scan_station_id=scan_station_id,
                    source_file=source_file,
                    tags=sub_seg.tags,
                    element_type=element_type,
                    semantic_confidence=safety.confidence,
                    classification_reason=safety.classification_reason,
                )
            )

        except Exception as e:
            logger.warning("segment_to_model_failed", segment=sub_seg.segment_id, error=str(e))

    return models


def _bb_volume(seg: dict) -> float:
    """Bounding box volume in cubic metres from raw segment dict."""
    mins, maxs = seg["mins"], seg["maxs"]
    dx = float(maxs[0] - mins[0])
    dy = float(maxs[1] - mins[1])
    dz = float(maxs[2] - mins[2])
    return max(dx, 0.0) * max(dy, 0.0) * max(dz, 0.0)


def _bb_iou(a: dict, b: dict) -> float:
    """Intersection-over-union of two AABB segments (raw dicts with mins/maxs)."""
    a_min, a_max = a["mins"], a["maxs"]
    b_min, b_max = b["mins"], b["maxs"]

    inter_min = np.maximum(a_min, b_min)
    inter_max = np.minimum(a_max, b_max)
    inter_dims = np.maximum(inter_max - inter_min, 0.0)
    inter_vol = float(inter_dims[0] * inter_dims[1] * inter_dims[2])

    if inter_vol == 0.0:
        return 0.0

    vol_a = _bb_volume(a)
    vol_b = _bb_volume(b)
    union_vol = vol_a + vol_b - inter_vol
    if union_vol <= 0.0:
        return 0.0
    return inter_vol / union_vol


def _bb_containment(small: dict, large: dict) -> float:
    """Fraction of *small*'s volume contained inside *large*'s BB."""
    s_min, s_max = small["mins"], small["maxs"]
    l_min, l_max = large["mins"], large["maxs"]

    inter_min = np.maximum(s_min, l_min)
    inter_max = np.minimum(s_max, l_max)
    inter_dims = np.maximum(inter_max - inter_min, 0.0)
    inter_vol = float(inter_dims[0] * inter_dims[1] * inter_dims[2])

    vol_small = _bb_volume(small)
    if vol_small <= 0.0:
        return 0.0
    return inter_vol / vol_small


def _resolve_overlaps(
    segments: list[dict],
    iou_threshold: float = 0.50,
    containment_threshold: float = 0.85,
) -> list[dict]:
    """Post-segmentation deduplication across detection stages.

    Two rules:
      1. **IoU merge** — if two segments of the *same* shape overlap with
         IoU ≥ iou_threshold, keep the one with higher confidence (absorb
         the other's point count).
      2. **Containment absorb** — if a smaller segment is ≥ containment_threshold
         contained inside a larger one (regardless of shape), drop the smaller
         one unless it is a VALVE_CANDIDATE (valves are expected to sit inside
         a pipe's bounding box).

    Returns a new list with duplicates removed.
    """
    if len(segments) <= 1:
        return segments

    protected_shapes = {
        SegmentShape.CYLINDER,
        SegmentShape.PLANE_VERTICAL,
        SegmentShape.PLANE_SLOPED,
        SegmentShape.VALVE_CANDIDATE,
    }

    n = len(segments)
    dropped: set[int] = set()

    for i in range(n):
        if i in dropped:
            continue
        for j in range(i + 1, n):
            if j in dropped:
                continue

            a, b = segments[i], segments[j]

            # Rule 1: same-shape merge for pipe-like cylinders.
            if a["shape"] == b["shape"]:
                if a["shape"] == SegmentShape.CYLINDER:
                    try:
                        a_radius_m = float(a.get("tags", {}).get("fitted_radius_mm", 0.0)) / 1000.0
                        b_radius_m = float(b.get("tags", {}).get("fitted_radius_mm", 0.0)) / 1000.0
                    except (AttributeError, TypeError, ValueError):
                        a_radius_m = b_radius_m = 0.0

                    if a_radius_m <= 0.0 or b_radius_m <= 0.0:
                        radius_tag = a.get("tags", {}).get("radius_mm") or b.get("tags", {}).get(
                            "radius_mm"
                        )
                        if radius_tag is not None:
                            try:
                                radius_mm = float(radius_tag)
                                if radius_mm > 0.0:
                                    a_radius_m = radius_mm / 1000.0
                                    b_radius_m = radius_mm / 1000.0
                            except (TypeError, ValueError):
                                pass

                    centroid_dist = float(
                        np.linalg.norm(
                            np.asarray(a["centroid"], dtype=float)
                            - np.asarray(b["centroid"], dtype=float)
                        )
                    )
                    radius_sum_m = max(a_radius_m, 1e-6) + max(b_radius_m, 1e-6)
                    clearance_margin_m = max(0.01, 0.02 * min(a_radius_m, b_radius_m))
                    if (
                        a_radius_m > 0.0
                        and b_radius_m > 0.0
                        and centroid_dist <= (radius_sum_m + clearance_margin_m)
                    ):
                        if a["confidence"] >= b["confidence"]:
                            a["point_count"] += b["point_count"]
                            dropped.add(j)
                        else:
                            b["point_count"] += a["point_count"]
                            dropped.add(i)
                            break
                        continue
                else:
                    iou = _bb_iou(a, b)
                    if iou >= iou_threshold:
                        # Keep higher confidence; absorb point count
                        if a["confidence"] >= b["confidence"]:
                            a["point_count"] += b["point_count"]
                            dropped.add(j)
                        else:
                            b["point_count"] += a["point_count"]
                            dropped.add(i)
                            break  # i is dropped, move to next i
                        continue

            # Rule 2: containment absorb (skip core buildable primitives).
            # Pipes/walls/stairs are often volumetrically contained by envelopes,
            # so containment-only dropping can erase the very elements we need.
            if a["shape"] in protected_shapes or b["shape"] in protected_shapes:
                continue

            vol_a = _bb_volume(a)
            vol_b = _bb_volume(b)

            if vol_a <= vol_b:
                small, large, small_idx = a, b, i
            else:
                small, large, small_idx = b, a, j

            if _bb_containment(small, large) >= containment_threshold:
                dropped.add(small_idx)
                if small_idx == i:
                    break  # i is dropped

    kept = [s for idx, s in enumerate(segments) if idx not in dropped]

    if dropped:
        logger.info(
            "overlap_resolution",
            input_segments=n,
            dropped=len(dropped),
            output_segments=len(kept),
        )

    return kept


def run_segmentation(
    file_path: Path | None,
    zone_id: str | None = None,
    voxel_size_m: float = VOXEL_SIZE_M,
    max_planes: int = MAX_PLANES,
    min_inliers: int = 80,
    preloaded_points: np.ndarray | None = None,
    seed: int | None = None,
    semantic_enabled: bool | None = None,
    semantic_model: str | None = None,
    semantic_checkpoint: Path | None = None,
    semantic_wall_threshold: float | None = None,
    semantic_fallback_to_geometry: bool | None = None,
) -> list[GeometrySegment]:
    """Full segmentation pipeline: load → downsample → planes → cylinders → models.

    Args:
        preloaded_points: Optional (N, 3) numpy array in mm.  When supplied,
            file_path is ignored and the array is used directly, converted to
            an Open3D PointCloud in metres for the rest of the pipeline.
        seed: Optional integer to pin every RNG (random / numpy / Open3D).
            Falls back to STB_RANDOM_SEED env var, then to 42. The actual
            seed applied is returned in the segments' tags["random_seed"]
            so downstream consumers can reproduce the run.
    """
    import open3d as o3d

    # NQA-1 determinism — pin every RNG before any sampling begins
    from agent.determinism import set_global_seed

    applied_seed = set_global_seed(seed)

    zone_id = zone_id or str(uuid.uuid4())[:8]

    if preloaded_points is not None:
        logger.info("segmentation_start_preloaded", points=len(preloaded_points), zone_id=zone_id)
        pcd = o3d.geometry.PointCloud()
        # Force C-contiguous float64 — older Open3D versions silently discard
        # non-contiguous or non-float64 arrays passed to Vector3dVector.
        pts_m = np.ascontiguousarray(preloaded_points / 1000.0, dtype=np.float64)
        pcd.points = o3d.utility.Vector3dVector(pts_m)
        if len(pcd.points) == 0:
            raise ValueError(
                f"Open3D rejected the preloaded_points array ({len(preloaded_points)} rows). "
                "Ensure the array is shape (N, 3) float64 with no NaN/Inf values."
            )
    else:
        logger.info("segmentation_start", file=str(file_path), zone_id=zone_id)
        pcd = load_point_cloud(file_path)
    logger.info("loaded", points=len(pcd.points))

    # ── STAGE 1 · Pre-process (Adaptive: Voxel first for large clouds, then SOR) ──
    n_raw = len(pcd.points)
    if n_raw == 0:
        raise ValueError("Point cloud is empty before preprocessing.")

    # For clouds > 200k points, voxel downsample first to eliminate 90s KDTree bottleneck
    if n_raw > 200_000:
        pcd = downsample(pcd, voxel_size_m)
        sor_pts = len(pcd.points)
        sor_neighbors = min(
            SOR_NB_NEIGHBORS,
            max(SOR_MIN_NEIGHBORS, sor_pts // SOR_NEIGHBOR_DIVISOR),
        )
        pcd = remove_statistical_outliers(
            pcd,
            nb_neighbors=sor_neighbors,
            std_ratio=SOR_STD_RATIO,
        )
    else:
        sor_neighbors = min(
            SOR_NB_NEIGHBORS,
            max(SOR_MIN_NEIGHBORS, n_raw // SOR_NEIGHBOR_DIVISOR),
        )
        pcd = remove_statistical_outliers(
            pcd,
            nb_neighbors=sor_neighbors,
            std_ratio=SOR_STD_RATIO,
        )
        pcd = downsample(pcd, voxel_size_m)
    n_pts = len(pcd.points)
    global _LAST_DOWNSAMPLED_POINTS
    _LAST_DOWNSAMPLED_POINTS = np.asarray(pcd.points, dtype=np.float64).copy()
    if n_pts == 0:
        raise ValueError(
            f"No points after voxel downsampling (voxel_size_m={voxel_size_m}). "
            "voxel_size_m may be too large for the scan density."
        )
    if n_pts < min_inliers:
        raise ValueError(
            f"Too few points after preprocessing: {n_pts} points "
            f"(minimum for plane detection: {min_inliers}). "
            f"Reduce voxel_size_m below {voxel_size_m} or increase scan density."
        )

    # A3: Surface normal estimation — required for B2 RANSAC cylinder and C2 region growing.
    # estimate_normals returns a NEW mean-centred cloud; capture it, then translate the
    # origin back so absolute coords (and the coord-normalisation below) stay correct.
    normal_result = estimate_normals(
        pcd, radius=NORMAL_RADIUS_VOXEL_MULT * voxel_size_m, max_nn=NORMAL_MAX_NN
    )
    if not normal_result["success"]:
        raise RuntimeError(
            f"Normal estimation failed: {normal_result['error']}\n"
            "Often Qhull precision errors on large or nearly coplanar clouds — "
            "increase voxel_size_m to reduce point count."
        )
    pcd = normal_result["pcd"]
    pcd.translate(normal_result["origin"])

    # Issue #17: normalise survey coordinates so Revit receives values near origin.
    # Real scanners may report coordinates in the millions (absolute survey coords).
    # Subtract the cloud's min-XYZ before converting to mm.
    pts_min = np.asarray(pcd.points).min(axis=0)
    normalize_threshold = float(os.environ.get("STB_COORD_NORMALIZE_THRESHOLD_M", "100.0"))
    force_normalize = os.environ.get("STB_NORMALIZE_COORDS", "0").strip().lower() in {"1", "true", "yes"}
    if force_normalize or np.any(np.abs(pts_min) > normalize_threshold):
        coord_origin_m = pts_min
        logger.info(
            "coord_normalisation_applied",
            origin_m=[round(float(v), 3) for v in coord_origin_m],
        )
    else:
        coord_origin_m = np.zeros(3)

    # Optional semantic front-end: predict wall mask, then keep the existing
    # geometric backend (RANSAC/merge/axis) unchanged.
    effective_semantic_enabled = SEMANTIC_ENABLE if semantic_enabled is None else semantic_enabled
    effective_model = (semantic_model or os.environ.get("STB_SEMANTIC_MODEL") or SEMANTIC_MODEL).strip()
    effective_threshold = (
        SEMANTIC_WALL_THRESHOLD
        if semantic_wall_threshold is None
        else float(semantic_wall_threshold)
    )
    effective_fallback = (
        SEMANTIC_FALLBACK_TO_GEOMETRY
        if semantic_fallback_to_geometry is None
        else semantic_fallback_to_geometry
    )

    if effective_semantic_enabled:
        points_m = np.asarray(pcd.points, dtype=np.float64)
        semantic_result = predict_wall_mask(
            points_m=points_m,
            model=effective_model,
            wall_threshold=effective_threshold,
            wall_class_id=SEMANTIC_WALL_CLASS_ID,
            checkpoint_path=semantic_checkpoint,
        )
        if semantic_result.success and semantic_result.wall_mask is not None:
            wall_indices = np.flatnonzero(semantic_result.wall_mask)
            if len(wall_indices) >= max(min_inliers, SEMANTIC_MIN_WALL_POINTS):
                pcd = pcd.select_by_index(wall_indices.tolist())
                logger.info(
                    "semantic_wall_mask_applied",
                    model=effective_model,
                    kept_points=len(wall_indices),
                    total_points=len(points_m),
                    threshold=effective_threshold,
                )
            else:
                msg = (
                    "Semantic wall mask produced too few points "
                    f"({len(wall_indices)}); minimum required is "
                    f"{max(min_inliers, SEMANTIC_MIN_WALL_POINTS)}."
                )
                if effective_fallback:
                    logger.warning("semantic_mask_too_small_fallback", message=msg)
                    set_last_semantic_report(
                        {
                            "enabled": True,
                            "model": effective_model,
                            "success": True,
                            "provider": "fallback",
                            "message": msg,
                            "wall_points": int(len(wall_indices)),
                            "total_points": int(len(points_m)),
                            "fallback": True,
                        }
                    )
                else:
                    raise ValueError(msg)
        else:
            msg = (
                f"Semantic wall masking failed for model '{effective_model}': "
                f"{semantic_result.message}"
            )
            if effective_fallback:
                logger.warning("semantic_mask_failed_fallback", message=msg)
                set_last_semantic_report(
                    {
                        "enabled": True,
                        "model": effective_model,
                        "success": True,
                        "provider": "fallback",
                        "message": msg,
                        "wall_points": 0,
                        "total_points": int(len(np.asarray(pcd.points))),
                        "fallback": True,
                    }
                )
            else:
                raise RuntimeError(msg)
    else:
        set_last_semantic_report(
            {
                "enabled": False,
                "model": None,
                "success": False,
                "provider": "none",
                "message": "semantic stage disabled",
                "wall_points": 0,
                "total_points": int(len(np.asarray(pcd.points))),
            }
        )

    pcds, color_tags = segregate_by_color(pcd)

    all_planes = []
    all_residuals = []

    for sub_pcd, c_tag in zip(pcds, color_tags):
        planes, residual = detect_planes(sub_pcd, max_planes=max_planes, min_inliers=min_inliers)
        for p in planes:
            p["tags"] = p.get("tags", {})
            p["tags"]["color_group"] = c_tag
            if c_tag.startswith("color_") and len(c_tag) == 12:
                p["tags"]["trace_color_hex"] = "#" + c_tag.split("_", 1)[1].upper()
        all_planes.extend(planes)
        all_residuals.append(residual)

    enable_wall_merge = os.environ.get("STB_ENABLE_WALL_MERGE", "1").strip().lower() in {
        "1",
        "true",
        "yes",
        "on",
    }

    if enable_wall_merge:
        wall_angle_tol_rad = float(os.environ.get("STB_WALL_MERGE_ANGLE_TOL_RAD", "0.087"))
        wall_gap_m = float(os.environ.get("STB_WALL_MERGE_GAP_M", "1.20"))
        wall_lateral_m = float(os.environ.get("STB_WALL_MERGE_LATERAL_M", "0.20"))

        planes = merge_walls(
            all_planes,
            angle_tolerance_rad=wall_angle_tol_rad,
            gap_threshold_m=wall_gap_m,
            lateral_offset_threshold_m=wall_lateral_m,
        )
    else:
        planes = all_planes

    planes = merge_coplanar_horizontal(planes)

    all_clusters = []
    all_valves = []
    all_rg_segments = []
    all_ec_segments = []

    for residual, c_tag in zip(all_residuals, color_tags):
        # Adaptive residual clustering: coarse voxel scans need looser/shorter
        # neighborhoods so sparse primitives (stairs/pipe fragments) survive.
        cluster_radius = max(0.03, min(0.06, voxel_size_m * 1.25))
        # RANSAC planes are the authoritative wall detector; residual clusters
        # are for MEP and fixtures only, so reject tiny scan fragments.
        default_cluster_min_size = 24 if voxel_size_m >= 0.04 else 20
        try:
            cluster_min_size = max(
            20,
                int(
                    float(
                        os.environ.get(
                            "STB_RESIDUAL_MIN_CLUSTER_SIZE", str(default_cluster_min_size)
                        )
                    )
                ),
            )
        except ValueError:
            cluster_min_size = default_cluster_min_size

        clusters = detect_clusters(residual)
        for c in clusters:
            c["tags"] = c.get("tags", {})
            c["tags"]["color_group"] = c_tag
            if c_tag.startswith("color_") and len(c_tag) == 12:
                c["tags"]["trace_color_hex"] = "#" + c_tag.split("_", 1)[1].upper()

            # ── Level assignment for component tracking ──────────────────────
            # Assign each cluster to a storey band (every 3m) for multi-level analysis
            c_z = float(c.get("centroid", [0, 0, 0])[2])
            c_level = int(round(c_z / 3000.0))  # 3m per level
            c["tags"]["assigned_level"] = c_level

        pipe_clusters = [c for c in clusters if c["shape"] == SegmentShape.CYLINDER]
        valves = detect_valves(pipe_clusters, residual)
        for v in valves:
            v["tags"] = v.get("tags", {})
            v["tags"]["color_group"] = c_tag
            if c_tag.startswith("color_") and len(c_tag) == 12:
                v["tags"]["trace_color_hex"] = "#" + c_tag.split("_", 1)[1].upper()

            # Level assignment for valves too
            v_z = float(v.get("centroid", [0, 0, 0])[2])
            v_level = int(round(v_z / 3000.0))
            v["tags"]["assigned_level"] = v_level

        all_clusters.extend(clusters)
        all_valves.extend(valves)

        # C2: Region growing on the residual point cloud
        rg_segments: list[dict] = []
        try:
            rg_clusters = region_growing(
                residual,
                angle_threshold_deg=15.0,
                radius=cluster_radius,
                min_cluster_size=cluster_min_size,
            )
            from agent.tools.loa_tools import aabb_sigma_mm as _aabb_sigma

            for rg_cluster in rg_clusters:
                # geometry_tools.region_growing returns dicts with indices and color
                # metadata; older callers expected raw index arrays.
                if isinstance(rg_cluster, dict):
                    idx_arr = rg_cluster.get("indices")
                    if idx_arr is None:
                        continue
                else:
                    idx_arr = rg_cluster

                idx_arr = np.asarray(idx_arr, dtype=np.intp)
                if idx_arr.size < 400:
                    continue

                rg_pts = np.asarray(residual.points)[idx_arr]
                rg_mins = rg_pts.min(axis=0)
                rg_maxs = rg_pts.max(axis=0)
                spans = rg_maxs - rg_mins  # metres

                # Must be a coherent 3D object: spans in all 3 dims >= 0.20m and volume >= 0.02 m^3
                if (spans < 0.20).any() or (spans[0] * spans[1] * spans[2] < 0.02):
                    continue

                rg_segments.append(
                    {
                        "shape": SegmentShape.BOX,
                        "normal": None,
                        "centroid": rg_pts.mean(axis=0),
                        "mins": rg_mins,
                        "maxs": rg_maxs,
                        "point_count": len(idx_arr),
                        "confidence": min(len(idx_arr) / 1000.0, 0.70),
                        "tags": {
                            "loa_sigma_mm": round(float(_aabb_sigma(rg_pts)), 3),
                            "color_group": c_tag,
                            **(
                                {"trace_color_hex": "#" + c_tag.split("_", 1)[1].upper()}
                                if c_tag.startswith("color_") and len(c_tag) == 12
                                else {}
                            ),
                        },
                    }
                )
        except Exception as exc:
            logger.warning("region_growing_failed", error=str(exc))
        all_rg_segments.extend(rg_segments)

        # C3: Euclidean clustering — only retain genuine coherent equipment bodies
        ec_segments: list[dict] = []
        try:
            ec_clusters = euclidean_cluster(
                residual,
                tolerance=cluster_radius,
                min_size=cluster_min_size,
            )
            from agent.tools.loa_tools import aabb_sigma_mm as _aabb_sigma_ec

            for idx_arr in ec_clusters:
                # Discard small stray noise — valves on pipe axes are already detected by detect_valves()
                if len(idx_arr) < 500:
                    continue

                ec_pts = np.asarray(residual.points)[idx_arr]
                ec_mins = ec_pts.min(axis=0)
                ec_maxs = ec_pts.max(axis=0)
                spans = ec_maxs - ec_mins

                # Genuine equipment box requires substantial 3D volume
                if (spans < 0.25).any() or (spans[0] * spans[1] * spans[2] < 0.04):
                    continue

                ec_segments.append(
                    {
                        "shape": SegmentShape.BOX,
                        "normal": None,
                        "centroid": ec_pts.mean(axis=0),
                        "mins": ec_mins,
                        "maxs": ec_maxs,
                        "point_count": len(idx_arr),
                        "confidence": min(len(idx_arr) / 1500.0, 0.65),
                        "tags": {
                            "loa_sigma_mm": round(float(_aabb_sigma_ec(ec_pts)), 3),
                            "color_group": c_tag,
                            **(
                                {"trace_color_hex": "#" + c_tag.split("_", 1)[1].upper()}
                                if c_tag.startswith("color_") and len(c_tag) == 12
                                else {}
                            ),
                        },
                    }
                )
        except Exception as exc:
            logger.warning("euclidean_clustering_failed", error=str(exc))
        all_ec_segments.extend(ec_segments)

    # Global merges and overlap resolution
    all_clusters = merge_collinear_cylinders(all_clusters)
    openings = detect_openings(planes)

    all_segments = planes + all_clusters + openings + all_valves + all_rg_segments + all_ec_segments

    # ── Topology Cleanup: Merge fragmented stairs/railings ────────────────────
    all_segments = merge_collinear_boxes(all_segments)

    # ── Overlap resolution: merge/deduplicate segments across detection stages ──
    enable_overlap_resolve = os.environ.get("STB_ENABLE_OVERLAP_RESOLVE", "0").strip().lower() in {
        "1",
        "true",
        "yes",
        "on",
    }
    if enable_overlap_resolve:
        all_segments = _resolve_overlaps(all_segments)

    # Surface-based sub-zoning from real wall surfaces.
    try:
        assign_surface_zones(all_segments)
    except Exception as exc:
        logger.warning("zone_surface_assignment_failed", error=str(exc))

    # E2: MEP connectivity graph — trace pipe/duct system topology.
    try:
        from agent.tools.geometry_tools import build_mep_graph, snap_mep_endpoints

        mep_graph = build_mep_graph(all_segments, connection_radius=0.25)
        if mep_graph is not None:
            # Snap endpoints to ensure topological continuity in Revit
            snap_mep_endpoints(all_segments, mep_graph)
            logger.info(
                "mep_graph_finalized",
                nodes=mep_graph.number_of_nodes(),
                edges=mep_graph.number_of_edges(),
            )
    except Exception as exc:
        logger.warning("mep_graph_failed", error=str(exc))

    source_name = file_path.name if file_path is not None else None

    # NQA-1 provenance — every segment gets the seed that produced it
    for seg in all_segments:
        seg.setdefault("tags", {})
        seg["tags"]["random_seed"] = applied_seed

    models = []

    for seg in all_segments:
        result = segment_to_model(
            seg,
            zone_id,
            source_file=source_name,
            coord_origin_m=coord_origin_m,
        )

        if isinstance(result, list):
            models.extend(result)
        else:
            models.append(result)

    enable_bim_patch = os.environ.get("STB_ENABLE_BIM_PIPELINE", "1").strip().lower() in {
        "1",
        "true",
        "yes",
        "on",
    }
    if enable_bim_patch and models:
        try:
            enable_auto_level_bim = os.environ.get(
                "STB_ENABLE_AUTO_LEVEL_BIM", "1"
            ).strip().lower() in {"1", "true", "yes", "on"}

            if enable_auto_level_bim:
                try:
                    levels = detect_levels(models)
                    if len(levels) >= 2:
                        levels_arr = np.asarray(levels, dtype=float)
                        level_spans = list(zip(levels[:-1], levels[1:]))
                        level_spans = [
                            (float(z0), float(z1))
                            for z0, z1 in level_spans
                            if float(z1) > float(z0)
                        ]

                        for model in models:
                            model.tags.setdefault("bim", {})
                            model.tags["bim"]["auto_levels_mm"] = [float(v) for v in levels]

                        if level_spans:
                            # ── Auto-level assignment: walls and columns only ──────────
                            # NOTE: Stairs, ramps, and ladders are excluded because their
                            # geometry is intrinsically multi-level (they connect two floors).
                            # Modifying their Z extents to match a single floor span destroys
                            # their actual detected shape.
                            auto_level_types = {
                                "wall",
                                "bund_wall",
                                "column",
                                "beam",
                                "railing",
                            }
                            for model in models:
                                et = getattr(model, "element_type", None)
                                et_value = getattr(et, "value", et)
                                if (
                                    not isinstance(et_value, str)
                                    or et_value.lower() not in auto_level_types
                                ):
                                    continue

                                zc = float(model.centroid.z)
                                idx = int(np.searchsorted(levels_arr, zc, side="right") - 1)
                                idx = max(0, min(idx, len(level_spans) - 1))
                                z0, z1 = level_spans[idx]

                                # Validate level span is reasonable (typical floor height is 3–5 m)
                                level_height = z1 - z0
                                if level_height < 100.0 or level_height > 7000.0:
                                    continue

                                # For walls and beams, only constrain Z if the original segment
                                # is already mostly within one level (don't expand existing geometry)
                                original_dz = float(
                                    model.bounding_box.max_z - model.bounding_box.min_z
                                )
                                if original_dz > level_height * 0.80:
                                    # Element already spans most of a level — respect original extents
                                    continue

                                model.bounding_box.min_z = z0
                                model.bounding_box.max_z = z1
                                model.centroid.z = (z0 + z1) / 2.0
                                model.tags.setdefault("bim", {}).update(
                                    {
                                        "auto_level_base_mm": z0,
                                        "auto_level_top_mm": z1,
                                        "auto_level_applied": True,
                                    }
                                )

                    # Replace raw wall/floor fragments with reconstructed level-spanning
                    # storey models so Revit receives proper enclosed geometry.
                    auto_nodes = run_auto_bim_pipeline(models)
                    reconstructed_map = {}
                    for n in auto_nodes.values():
                        seg = getattr(n, "segment", None)
                        if seg is None:
                            continue
                        if seg.tags.get("host_wall_id"):
                            for m in models:
                                if m.segment_id == seg.segment_id:
                                    m.tags.update(seg.tags)
                        is_structural_rebuild = bool(
                            seg.tags.get("reconstructed_wall")
                            or seg.tags.get("reconstructed_floor")
                            or seg.tags.get("reconstructed_ceiling")
                            or seg.tags.get("reconstructed_stair")
                        )
                        if is_structural_rebuild:
                            reconstructed_map[seg.segment_id] = seg
                    if reconstructed_map:
                        replacement_by_source_id: dict[str, GeometrySegment] = {}
                        for seg in reconstructed_map.values():
                            source_ids = seg.tags.get("reconstructed_from_segment_ids") or [
                                seg.segment_id
                            ]
                            for source_id in source_ids:
                                if source_id:
                                    replacement_by_source_id[str(source_id)] = seg

                        rebuilt_models: list[GeometrySegment] = []
                        emitted_replacement_ids: set[str] = set()

                        for model in models:
                            replacement = replacement_by_source_id.get(str(model.segment_id))
                            if replacement is None:
                                rebuilt_models.append(model)
                                continue

                            if replacement.segment_id in emitted_replacement_ids:
                                continue

                            rebuilt_models.append(replacement)
                            emitted_replacement_ids.add(replacement.segment_id)

                        # Append synthetic reconstructions that do not map to an original source segment.
                        for seg in reconstructed_map.values():
                            if seg.segment_id in emitted_replacement_ids:
                                continue
                            rebuilt_models.append(seg)
                            emitted_replacement_ids.add(seg.segment_id)

                        models = rebuilt_models

                    auto_hosts = sum(1 for n in auto_nodes.values() if n.host)
                    logger.info(
                        "auto_level_bim_applied",
                        nodes=len(auto_nodes),
                        hosted_nodes=auto_hosts,
                    )
                except Exception as exc:
                    logger.warning("auto_level_bim_failed", error=str(exc))

            nodes = run_bim_pipeline_v2(models)
            by_segment_id = {node.segment.segment_id: node for node in nodes.values()}

            for model in models:
                node = by_segment_id.get(model.segment_id)
                if node is None:
                    continue

                base_conf = float(model.semantic_confidence or model.confidence)
                model.semantic_confidence = max(0.0, min(1.0, base_conf * float(node.confidence)))

                model.tags.setdefault("bim", {})
                model.tags["bim"].update(
                    {
                        "node_id": node.id,
                        "host": node.host,
                        "contains": node.children,
                        "connected_to": node.connections,
                        "parametric": node.parametric,
                        "topology_confidence": round(float(node.confidence), 4),
                    }
                )
                model.tags.setdefault("ifc", {})
                model.tags["ifc"].update(
                    {
                        "ifc_ready": True,
                        "ifc_node_id": node.id,
                        "ifc_host": node.host,
                        "ifc_connections": node.connections,
                        "ifc_parametric": node.parametric,
                    }
                )
        except Exception as exc:
            logger.warning("bim_pipeline_failed", error=str(exc))

    # ── Final geometry validation — reject degenerate boxes ──────────────────
    # Convert GeometrySegment objects to dicts for validation, then filter
    segment_dicts = [
        {
            "mins": np.array(
                [
                    float(m.bounding_box.min_x),
                    float(m.bounding_box.min_y),
                    float(m.bounding_box.min_z),
                ]
            ),
            "maxs": np.array(
                [
                    float(m.bounding_box.max_x),
                    float(m.bounding_box.max_y),
                    float(m.bounding_box.max_z),
                ]
            ),
            "centroid": np.array([float(m.centroid.x), float(m.centroid.y), float(m.centroid.z)]),
            "shape": m.shape,
        }
        for m in models
    ]
    from agent.tools.geometry_tools import filter_valid_segments

    _valid_dicts, _rejected = filter_valid_segments(segment_dicts)

    if _rejected:
        # Build set of rejected indices
        rejected_indices = {idx for idx, _ in _rejected}
        models = [m for idx, m in enumerate(models) if idx not in rejected_indices]
        logger.info(
            "degenerate_segments_filtered",
            original_count=len(segment_dicts),
            filtered_count=len(models),
            rejected_count=len(_rejected),
        )

    logger.info(
        "segmentation_complete",
        zone_id=zone_id,
        planes=len(planes),
        clusters=len(clusters),
        openings=len(openings),
        valves=len(valves),
        region_growing=len(rg_segments),
        euclidean_clusters=len(ec_segments),
        total_segments=len(models),
        random_seed=applied_seed,
    )
    models = _merge_circular_tanks(models)
    return models


def generate_synthetic_segments(
    zone_id: str = "zone-001",
    num_walls: int = 15,
    num_floors: int = 5,
    num_columns: int = 3,
    num_beams: int = 4,
    num_pipes: int = 2,
    num_stairs: int = 1,
    num_doors: int = 4,
    num_windows: int = 6,
    num_railings: int = 2,
    num_ducts: int = 3,
    num_conduits: int = 4,
    num_cable_trays: int = 2,
    num_hvac_units: int = 1,
    num_tanks: int = 1,
    num_pumps: int = 1,
    num_slab_openings: int = 3,
    num_sprinklers: int = 6,
    num_electrical_panels: int = 2,
    num_kerbs: int = 2,
    num_drains: int = 3,
    num_valves: int = 4,
    # New nuclear / industrial types (default 1 each for demo)
    num_ladders: int = 2,
    num_gratings: int = 2,
    num_overhead_cranes: int = 1,
    num_hatches: int = 2,
    num_trenches: int = 1,
    num_bund_walls: int = 2,
    num_safety_relief_valves: int = 2,
    num_strainers: int = 2,
    num_expansion_joints: int = 2,
    num_fire_dampers: int = 2,
    num_penetration_seals: int = 1,
    num_pipe_supports: int = 4,
    num_pressure_vessels: int = 1,
    num_heat_exchangers: int = 1,
    num_compressors: int = 1,
    num_pressurizers: int = 1,
    num_steam_generators: int = 1,
    num_emergency_diesel_generators: int = 1,
    num_seismic_isolators: int = 2,
    num_containment_penetrations: int = 1,
    num_radiation_monitors: int = 2,
    num_transformers: int = 1,
    num_switchgear: int = 1,
    num_ups_systems: int = 1,
    num_junction_boxes: int = 4,
    num_lighting_fittings: int = 6,
    num_fire_hydrants: int = 2,
    num_deluge_valves: int = 1,
    num_fire_alarm_panels: int = 1,
    num_smoke_detectors: int = 6,
) -> list[GeometrySegment]:
    """Generate synthetic segments for demo/testing without a real scan file."""
    segments = []
    rng = np.random.default_rng(42)

    for i in range(num_walls):
        x = rng.uniform(0, 50000)
        y = rng.uniform(0, 30000)
        z_base = 0.0
        height = rng.uniform(2800, 3500)
        thickness = rng.uniform(150, 300)
        length = rng.uniform(3000, 8000)

        segments.append(
            GeometrySegment(
                zone_id=zone_id,
                shape=SegmentShape.PLANE_VERTICAL,
                normal=Point3D(x=0.0, y=1.0, z=0.0),
                centroid=Point3D(x=x + length / 2, y=y, z=z_base + height / 2),
                bounding_box=BoundingBox(
                    min_x=x,
                    min_y=y - thickness / 2,
                    min_z=z_base,
                    max_x=x + length,
                    max_y=y + thickness / 2,
                    max_z=z_base + height,
                ),
                point_count=int(rng.uniform(500, 5000)),
                confidence=float(rng.uniform(0.75, 0.98)),
                source_file="synthetic",
            )
        )

    for i in range(num_floors):
        x = rng.uniform(0, 50000)
        y = rng.uniform(0, 30000)
        z = rng.uniform(0, 200) if i == 0 else rng.uniform(2800, 3500) * (i)
        width = rng.uniform(5000, 15000)
        depth = rng.uniform(5000, 15000)

        # Floor (low z) and matching ceiling (floor z + storey height)
        segments.append(
            GeometrySegment(
                zone_id=zone_id,
                shape=SegmentShape.PLANE_HORIZONTAL,
                normal=Point3D(x=0.0, y=0.0, z=1.0),
                centroid=Point3D(x=x + width / 2, y=y + depth / 2, z=z),
                bounding_box=BoundingBox(
                    min_x=x,
                    min_y=y,
                    min_z=z - 150,
                    max_x=x + width,
                    max_y=y + depth,
                    max_z=z + 150,
                ),
                point_count=int(rng.uniform(1000, 8000)),
                confidence=float(rng.uniform(0.80, 0.99)),
                source_file="synthetic",
            )
        )
        # Ceiling at top of storey
        ceiling_z = z + rng.uniform(2800, 3200)
        segments.append(
            GeometrySegment(
                zone_id=zone_id,
                shape=SegmentShape.PLANE_HORIZONTAL,
                normal=Point3D(x=0.0, y=0.0, z=-1.0),
                centroid=Point3D(x=x + width / 2, y=y + depth / 2, z=ceiling_z),
                bounding_box=BoundingBox(
                    min_x=x,
                    min_y=y,
                    min_z=ceiling_z - 50,
                    max_x=x + width,
                    max_y=y + depth,
                    max_z=ceiling_z + 50,
                ),
                point_count=int(rng.uniform(800, 5000)),
                confidence=float(rng.uniform(0.75, 0.95)),
                source_file="synthetic",
            )
        )

    for i in range(num_columns):
        x = rng.uniform(0, 50000)
        y = rng.uniform(0, 30000)
        size = rng.uniform(300, 600)
        height = rng.uniform(2800, 3500)

        segments.append(
            GeometrySegment(
                zone_id=zone_id,
                shape=SegmentShape.PLANE_VERTICAL,
                normal=Point3D(x=0.707, y=0.707, z=0.0),
                centroid=Point3D(x=x, y=y, z=height / 2),
                bounding_box=BoundingBox(
                    min_x=x - size / 2,
                    min_y=y - size / 2,
                    min_z=0,
                    max_x=x + size / 2,
                    max_y=y + size / 2,
                    max_z=height,
                ),
                point_count=int(rng.uniform(200, 1000)),
                confidence=float(rng.uniform(0.70, 0.95)),
                source_file="synthetic",
            )
        )

    for i in range(num_beams):
        x = rng.uniform(0, 50000)
        y = rng.uniform(0, 30000)
        z = rng.uniform(2600, 3200)  # suspended near ceiling
        length = rng.uniform(4000, 12000)
        width_b = rng.uniform(100, 300)  # flange width
        depth_b = rng.uniform(200, 600)  # beam depth

        # Beam runs along X axis
        segments.append(
            GeometrySegment(
                zone_id=zone_id,
                shape=SegmentShape.BOX,
                normal=None,
                centroid=Point3D(x=x + length / 2, y=y, z=z),
                bounding_box=BoundingBox(
                    min_x=x,
                    min_y=y - width_b / 2,
                    min_z=z - depth_b / 2,
                    max_x=x + length,
                    max_y=y + width_b / 2,
                    max_z=z + depth_b / 2,
                ),
                point_count=int(rng.uniform(200, 1500)),
                confidence=float(rng.uniform(0.65, 0.90)),
                source_file="synthetic",
            )
        )

    for i in range(num_stairs):
        x = rng.uniform(0, 40000)
        y = rng.uniform(0, 20000)
        run_length = rng.uniform(3000, 5000)
        run_width = rng.uniform(1200, 2400)
        rise = rng.uniform(2800, 3600)

        segments.append(
            GeometrySegment(
                zone_id=zone_id,
                shape=SegmentShape.PLANE_SLOPED,
                normal=Point3D(x=0.0, y=-0.5, z=0.866),  # ~30° slope
                centroid=Point3D(x=x + run_length / 2, y=y + run_width / 2, z=rise / 2),
                bounding_box=BoundingBox(
                    min_x=x,
                    min_y=y,
                    min_z=0,
                    max_x=x + run_length,
                    max_y=y + run_width,
                    max_z=rise,
                ),
                point_count=int(rng.uniform(300, 2000)),
                confidence=float(rng.uniform(0.60, 0.85)),
                source_file="synthetic",
            )
        )

    for i in range(num_pipes):
        x = rng.uniform(0, 50000)
        y = rng.uniform(0, 30000)
        z = rng.uniform(2500, 3200)
        length = rng.uniform(3000, 10000)
        radius = rng.uniform(25, 150)

        segments.append(
            GeometrySegment(
                zone_id=zone_id,
                shape=SegmentShape.CYLINDER,
                normal=None,
                centroid=Point3D(x=x + length / 2, y=y, z=z),
                bounding_box=BoundingBox(
                    min_x=x,
                    min_y=y - radius,
                    min_z=z - radius,
                    max_x=x + length,
                    max_y=y + radius,
                    max_z=z + radius,
                ),
                point_count=int(rng.uniform(100, 800)),
                confidence=float(rng.uniform(0.60, 0.90)),
                source_file="synthetic",
            )
        )

    for i in range(num_doors):
        x = rng.uniform(0, 50000)
        y = rng.uniform(0, 30000)
        w = rng.uniform(800, 1200)
        h = rng.uniform(2000, 2400)
        t = 250  # door thickness (wall depth)

        segments.append(
            GeometrySegment(
                zone_id=zone_id,
                shape=SegmentShape.VOID,
                normal=Point3D(x=0.0, y=1.0, z=0.0),
                centroid=Point3D(x=x + w / 2, y=y, z=h / 2),
                bounding_box=BoundingBox(
                    min_x=x,
                    min_y=y - t / 2,
                    min_z=0,  # min_z ≈ 0 → door
                    max_x=x + w,
                    max_y=y + t / 2,
                    max_z=h,
                ),
                point_count=0,
                confidence=float(rng.uniform(0.65, 0.85)),
                source_file="synthetic",
            )
        )

    for i in range(num_windows):
        x = rng.uniform(0, 50000)
        y = rng.uniform(0, 30000)
        w = rng.uniform(600, 1500)
        h = rng.uniform(800, 1400)
        sill = rng.uniform(800, 1200)  # sill height > 300 mm → window
        t = 250

        segments.append(
            GeometrySegment(
                zone_id=zone_id,
                shape=SegmentShape.VOID,
                normal=Point3D(x=0.0, y=1.0, z=0.0),
                centroid=Point3D(x=x + w / 2, y=y, z=sill + h / 2),
                bounding_box=BoundingBox(
                    min_x=x,
                    min_y=y - t / 2,
                    min_z=sill,
                    max_x=x + w,
                    max_y=y + t / 2,
                    max_z=sill + h,
                ),
                point_count=0,
                confidence=float(rng.uniform(0.60, 0.80)),
                source_file="synthetic",
            )
        )

    for i in range(num_railings):
        x = rng.uniform(0, 50000)
        y = rng.uniform(0, 30000)
        length = rng.uniform(2000, 8000)
        rail_z = rng.uniform(900, 1100)  # railing centroid Z in railing-height band

        segments.append(
            GeometrySegment(
                zone_id=zone_id,
                shape=SegmentShape.BOX,
                normal=None,
                centroid=Point3D(x=x + length / 2, y=y, z=rail_z),
                bounding_box=BoundingBox(
                    min_x=x,
                    min_y=y - 50,
                    min_z=rail_z - 50,
                    max_x=x + length,
                    max_y=y + 50,
                    max_z=rail_z + 50,
                ),
                point_count=int(rng.uniform(80, 400)),
                confidence=float(rng.uniform(0.60, 0.82)),
                source_file="synthetic",
            )
        )

    # ── MEP: Rectangular ducts (elevated, large cross-section BOX) ──────────────
    for i in range(num_ducts):
        x = rng.uniform(0, 50000)
        y = rng.uniform(0, 30000)
        z = rng.uniform(2700, 3100)  # near ceiling
        length = rng.uniform(5000, 15000)
        duct_w = rng.uniform(300, 1200)  # wide face
        duct_h = rng.uniform(200, 600)  # narrow face

        segments.append(
            GeometrySegment(
                zone_id=zone_id,
                shape=SegmentShape.BOX,
                normal=None,
                centroid=Point3D(x=x + length / 2, y=y, z=z),
                bounding_box=BoundingBox(
                    min_x=x,
                    min_y=y - duct_w / 2,
                    min_z=z - duct_h / 2,
                    max_x=x + length,
                    max_y=y + duct_w / 2,
                    max_z=z + duct_h / 2,
                ),
                point_count=int(rng.uniform(300, 2000)),
                confidence=float(rng.uniform(0.70, 0.90)),
                source_file="synthetic",
            )
        )

    # ── MEP: Conduits (small CYLINDER, any elevation) ───────────────────────────
    for i in range(num_conduits):
        x = rng.uniform(0, 50000)
        y = rng.uniform(0, 30000)
        z = rng.uniform(1000, 3000)
        length = rng.uniform(2000, 10000)
        r = rng.uniform(12, 30)  # small radius (< 60 mm diameter)

        segments.append(
            GeometrySegment(
                zone_id=zone_id,
                shape=SegmentShape.CYLINDER,
                normal=None,
                centroid=Point3D(x=x + length / 2, y=y, z=z),
                bounding_box=BoundingBox(
                    min_x=x,
                    min_y=y - r,
                    min_z=z - r,
                    max_x=x + length,
                    max_y=y + r,
                    max_z=z + r,
                ),
                point_count=int(rng.uniform(60, 300)),
                confidence=float(rng.uniform(0.55, 0.80)),
                source_file="synthetic",
            )
        )

    # ── MEP: Cable trays (thin flat BOX, horizontal) ────────────────────────────
    for i in range(num_cable_trays):
        x = rng.uniform(0, 50000)
        y = rng.uniform(0, 30000)
        z = rng.uniform(2400, 3000)
        length = rng.uniform(4000, 12000)
        tray_w = rng.uniform(300, 600)  # wide
        tray_h = rng.uniform(50, 120)  # very shallow

        segments.append(
            GeometrySegment(
                zone_id=zone_id,
                shape=SegmentShape.BOX,
                normal=None,
                centroid=Point3D(x=x + length / 2, y=y, z=z),
                bounding_box=BoundingBox(
                    min_x=x,
                    min_y=y - tray_w / 2,
                    min_z=z - tray_h / 2,
                    max_x=x + length,
                    max_y=y + tray_w / 2,
                    max_z=z + tray_h / 2,
                ),
                point_count=int(rng.uniform(150, 800)),
                confidence=float(rng.uniform(0.60, 0.82)),
                source_file="synthetic",
            )
        )

    # ── MEP: HVAC units (large compact BOX, elevated on plant deck) ─────────────
    for i in range(num_hvac_units):
        x = rng.uniform(0, 50000)
        y = rng.uniform(0, 30000)
        z = rng.uniform(1200, 2000)
        sz = rng.uniform(800, 2000)  # roughly cubic

        segments.append(
            GeometrySegment(
                zone_id=zone_id,
                shape=SegmentShape.BOX,
                normal=None,
                centroid=Point3D(x=x + sz / 2, y=y + sz / 2, z=z),
                bounding_box=BoundingBox(
                    min_x=x,
                    min_y=y,
                    min_z=z - sz / 2,
                    max_x=x + sz,
                    max_y=y + sz,
                    max_z=z + sz / 2,
                ),
                point_count=int(rng.uniform(500, 3000)),
                confidence=float(rng.uniform(0.65, 0.88)),
                source_file="synthetic",
            )
        )

    # ── MEP: Tanks (large CYLINDER, near floor or on base frame) ────────────────
    for i in range(num_tanks):
        x = rng.uniform(0, 50000)
        y = rng.uniform(0, 30000)
        r = rng.uniform(400, 1500)  # large radius > 600 mm half → diameter > 1200 mm
        h = rng.uniform(1500, 5000)
        z_base = rng.uniform(0, 500)

        segments.append(
            GeometrySegment(
                zone_id=zone_id,
                shape=SegmentShape.CYLINDER,
                normal=None,
                centroid=Point3D(x=x, y=y, z=z_base + h / 2),
                bounding_box=BoundingBox(
                    min_x=x - r,
                    min_y=y - r,
                    min_z=z_base,
                    max_x=x + r,
                    max_y=y + r,
                    max_z=z_base + h,
                ),
                point_count=int(rng.uniform(500, 4000)),
                confidence=float(rng.uniform(0.70, 0.92)),
                source_file="synthetic",
            )
        )

    # ── MEP: Pumps (medium compact BOX at floor level) ───────────────────────────
    for i in range(num_pumps):
        x = rng.uniform(0, 50000)
        y = rng.uniform(0, 30000)
        pw = rng.uniform(500, 1000)  # pump body size
        ph = rng.uniform(400, 800)
        pz = rng.uniform(200, 700)  # low centroid Z → at floor

        segments.append(
            GeometrySegment(
                zone_id=zone_id,
                shape=SegmentShape.BOX,
                normal=None,
                centroid=Point3D(x=x + pw / 2, y=y + pw / 2, z=pz),
                bounding_box=BoundingBox(
                    min_x=x,
                    min_y=y,
                    min_z=0,
                    max_x=x + pw,
                    max_y=y + pw,
                    max_z=pz + ph / 2,
                ),
                point_count=int(rng.uniform(300, 1500)),
                confidence=float(rng.uniform(0.60, 0.85)),
                source_file="synthetic",
            )
        )

    # ── Slab openings (thin VOID through horizontal slab) ────────────────────────
    for i in range(num_slab_openings):
        x = rng.uniform(0, 50000)
        y = rng.uniform(0, 30000)
        z = rng.uniform(0, 3000)  # floor/ceiling elevation
        ow = rng.uniform(500, 2000)  # opening width
        od = rng.uniform(500, 2000)  # opening depth
        t = rng.uniform(150, 350)  # slab thickness

        segments.append(
            GeometrySegment(
                zone_id=zone_id,
                shape=SegmentShape.VOID,
                normal=Point3D(x=0.0, y=0.0, z=1.0),
                centroid=Point3D(x=x + ow / 2, y=y + od / 2, z=z),
                bounding_box=BoundingBox(
                    min_x=x,
                    min_y=y,
                    min_z=z - t / 2,
                    max_x=x + ow,
                    max_y=y + od,
                    max_z=z + t / 2,
                ),
                point_count=int(rng.uniform(50, 150)),
                confidence=float(rng.uniform(0.60, 0.82)),
                source_file="synthetic",
            )
        )

    # ── Sprinkler heads (tiny CYLINDER at ceiling level > 2500 mm) ───────────────
    for i in range(num_sprinklers):
        x = rng.uniform(0, 50000)
        y = rng.uniform(0, 30000)
        z = rng.uniform(2600, 3200)  # ceiling level
        r = rng.uniform(30, 80)  # sprinkler head radius

        segments.append(
            GeometrySegment(
                zone_id=zone_id,
                shape=SegmentShape.CYLINDER,
                normal=None,
                centroid=Point3D(x=x, y=y, z=z),
                bounding_box=BoundingBox(
                    min_x=x - r,
                    min_y=y - r,
                    min_z=z - r,
                    max_x=x + r,
                    max_y=y + r,
                    max_z=z + r,
                ),
                point_count=int(rng.uniform(8, 50)),
                confidence=float(rng.uniform(0.50, 0.75)),
                source_file="synthetic",
            )
        )

    # ── Electrical panels (thin flat BOX on wall) ────────────────────────────────
    for i in range(num_electrical_panels):
        x = rng.uniform(0, 50000)
        y = rng.uniform(0, 30000)
        z = rng.uniform(1000, 1600)  # centroid at switch/panel height
        pw = rng.uniform(500, 1200)  # face width
        ph = rng.uniform(600, 1800)  # face height
        pd = rng.uniform(100, 250)  # panel depth (thin)

        segments.append(
            GeometrySegment(
                zone_id=zone_id,
                shape=SegmentShape.BOX,
                normal=Point3D(x=0.0, y=1.0, z=0.0),
                centroid=Point3D(x=x + pw / 2, y=y, z=z),
                bounding_box=BoundingBox(
                    min_x=x,
                    min_y=y - pd / 2,
                    min_z=z - ph / 2,
                    max_x=x + pw,
                    max_y=y + pd / 2,
                    max_z=z + ph / 2,
                ),
                point_count=int(rng.uniform(100, 600)),
                confidence=float(rng.uniform(0.60, 0.82)),
                source_file="synthetic",
            )
        )

    # ── Kerbs (very low BOX along building perimeter) ────────────────────────────
    for i in range(num_kerbs):
        x = rng.uniform(0, 50000)
        y = rng.uniform(0, 30000)
        length = rng.uniform(2000, 10000)
        kh = rng.uniform(100, 300)  # kerb height
        kw = rng.uniform(100, 250)  # kerb width

        segments.append(
            GeometrySegment(
                zone_id=zone_id,
                shape=SegmentShape.BOX,
                normal=None,
                centroid=Point3D(x=x + length / 2, y=y, z=kh / 2),
                bounding_box=BoundingBox(
                    min_x=x,
                    min_y=y - kw / 2,
                    min_z=0,
                    max_x=x + length,
                    max_y=y + kw / 2,
                    max_z=kh,
                ),
                point_count=int(rng.uniform(100, 600)),
                confidence=float(rng.uniform(0.65, 0.85)),
                source_file="synthetic",
            )
        )

    # ── Floor drains (tiny CYLINDER at floor level < 200 mm) ─────────────────────
    for i in range(num_drains):
        x = rng.uniform(0, 50000)
        y = rng.uniform(0, 30000)
        r = rng.uniform(40, 120)  # drain grate radius

        segments.append(
            GeometrySegment(
                zone_id=zone_id,
                shape=SegmentShape.CYLINDER,
                normal=None,
                centroid=Point3D(x=x, y=y, z=r),  # centroid close to floor
                bounding_box=BoundingBox(
                    min_x=x - r,
                    min_y=y - r,
                    min_z=0,
                    max_x=x + r,
                    max_y=y + r,
                    max_z=r * 2,
                ),
                point_count=int(rng.uniform(8, 60)),
                confidence=float(rng.uniform(0.45, 0.70)),
                source_file="synthetic",
            )
        )

    # ── Valves (compact VALVE_CANDIDATE on a pipe run) ──────────────────────────
    for i in range(num_valves):
        x = rng.uniform(0, 50000)
        y = rng.uniform(0, 30000)
        z = rng.uniform(1000, 3000)  # match pipe elevation
        vw = rng.uniform(80, 300)  # valve body width
        vh = rng.uniform(100, 350)  # valve body height (including actuator)
        vd = rng.uniform(80, 250)  # valve body depth

        segments.append(
            GeometrySegment(
                zone_id=zone_id,
                shape=SegmentShape.VALVE_CANDIDATE,
                normal=None,
                centroid=Point3D(x=x, y=y, z=z),
                bounding_box=BoundingBox(
                    min_x=x - vw / 2,
                    min_y=y - vd / 2,
                    min_z=z - vh / 2,
                    max_x=x + vw / 2,
                    max_y=y + vd / 2,
                    max_z=z + vh / 2,
                ),
                point_count=int(rng.uniform(15, 120)),
                confidence=float(rng.uniform(0.50, 0.75)),
                source_file="synthetic",
            )
        )

    # ── Access: Ladders (vertical BOX on vessel/rack face) ───────────────────────
    for _i in range(num_ladders):
        x = rng.uniform(0, 50000)
        y = rng.uniform(0, 30000)
        h = rng.uniform(2500, 6000)
        w = rng.uniform(350, 500)
        d = rng.uniform(150, 300)
        segments.append(
            GeometrySegment(
                zone_id=zone_id,
                shape=SegmentShape.BOX,
                normal=None,
                centroid=Point3D(x=x, y=y, z=h / 2),
                bounding_box=BoundingBox(
                    min_x=x - w / 2,
                    min_y=y - d / 2,
                    min_z=0,
                    max_x=x + w / 2,
                    max_y=y + d / 2,
                    max_z=h,
                ),
                point_count=int(rng.uniform(80, 400)),
                confidence=float(rng.uniform(0.55, 0.80)),
                source_file="synthetic",
            )
        )

    # ── Civil: Gratings (sparse PLANE_HORIZONTAL over sump/pit) ─────────────────
    for _i in range(num_gratings):
        x = rng.uniform(0, 50000)
        y = rng.uniform(0, 30000)
        z = rng.uniform(0, 500)
        gw = rng.uniform(1000, 4000)
        gd = rng.uniform(1000, 4000)
        segments.append(
            GeometrySegment(
                zone_id=zone_id,
                shape=SegmentShape.PLANE_HORIZONTAL,
                normal=Point3D(x=0, y=0, z=1),
                centroid=Point3D(x=x + gw / 2, y=y + gd / 2, z=z),
                bounding_box=BoundingBox(
                    min_x=x, min_y=y, min_z=z - 30, max_x=x + gw, max_y=y + gd, max_z=z + 30
                ),
                # very low point count → low density → grating rule triggers
                point_count=int(rng.uniform(5, 15)),
                confidence=float(rng.uniform(0.50, 0.72)),
                source_file="synthetic",
            )
        )

    # ── Structural: Overhead crane (very high BOX spanning bay) ─────────────────
    for _i in range(num_overhead_cranes):
        x = rng.uniform(0, 30000)
        y = rng.uniform(0, 15000)
        span = rng.uniform(10000, 25000)
        w = rng.uniform(400, 800)
        h = rng.uniform(600, 1200)
        z_ctr = rng.uniform(6000, 12000)
        segments.append(
            GeometrySegment(
                zone_id=zone_id,
                shape=SegmentShape.BOX,
                normal=None,
                centroid=Point3D(x=x + span / 2, y=y, z=z_ctr),
                bounding_box=BoundingBox(
                    min_x=x,
                    min_y=y - w / 2,
                    min_z=z_ctr - h / 2,
                    max_x=x + span,
                    max_y=y + w / 2,
                    max_z=z_ctr + h / 2,
                ),
                point_count=int(rng.uniform(500, 3000)),
                confidence=float(rng.uniform(0.70, 0.90)),
                source_file="synthetic",
            )
        )

    # ── Electrical: Transformers (large floor-mounted BOX) ────────────────────────
    for _i in range(num_transformers):
        x = rng.uniform(0, 50000)
        y = rng.uniform(0, 30000)
        tw = rng.uniform(700, 1500)
        td = rng.uniform(700, 1200)
        th = rng.uniform(900, 3000)
        z_ctr = th / 2
        segments.append(
            GeometrySegment(
                zone_id=zone_id,
                shape=SegmentShape.BOX,
                normal=None,
                centroid=Point3D(x=x + tw / 2, y=y + td / 2, z=z_ctr),
                bounding_box=BoundingBox(
                    min_x=x, min_y=y, min_z=0, max_x=x + tw, max_y=y + td, max_z=th
                ),
                point_count=int(rng.uniform(300, 2000)),
                confidence=float(rng.uniform(0.68, 0.90)),
                source_file="synthetic",
            )
        )

    # ── Electrical: Switchgear rows (tall narrow BOX row) ────────────────────────
    for _i in range(num_switchgear):
        x = rng.uniform(0, 50000)
        y = rng.uniform(0, 30000)
        sl = rng.uniform(2000, 6000)
        sd = rng.uniform(500, 680)
        sh = rng.uniform(1800, 2400)
        segments.append(
            GeometrySegment(
                zone_id=zone_id,
                shape=SegmentShape.BOX,
                normal=None,
                centroid=Point3D(x=x + sl / 2, y=y, z=sh / 2),
                bounding_box=BoundingBox(
                    min_x=x, min_y=y - sd / 2, min_z=0, max_x=x + sl, max_y=y + sd / 2, max_z=sh
                ),
                point_count=int(rng.uniform(200, 1200)),
                confidence=float(rng.uniform(0.65, 0.88)),
                source_file="synthetic",
            )
        )

    # ── Electrical: UPS / battery racks (long low profile BOX) ───────────────────
    for _i in range(num_ups_systems):
        x = rng.uniform(0, 50000)
        y = rng.uniform(0, 30000)
        ul = rng.uniform(2500, 5000)
        uw = rng.uniform(250, 650)
        uh = rng.uniform(250, 650)
        z_ctr = uh / 2 + rng.uniform(0, 400)
        segments.append(
            GeometrySegment(
                zone_id=zone_id,
                shape=SegmentShape.BOX,
                normal=None,
                centroid=Point3D(x=x + ul / 2, y=y, z=z_ctr),
                bounding_box=BoundingBox(
                    min_x=x,
                    min_y=y - uw / 2,
                    min_z=z_ctr - uh / 2,
                    max_x=x + ul,
                    max_y=y + uw / 2,
                    max_z=z_ctr + uh / 2,
                ),
                point_count=int(rng.uniform(150, 900)),
                confidence=float(rng.uniform(0.62, 0.85)),
                source_file="synthetic",
            )
        )

    # ── Electrical: Junction boxes (tiny wall BOX) ────────────────────────────────
    for _i in range(num_junction_boxes):
        x = rng.uniform(0, 50000)
        y = rng.uniform(0, 30000)
        z = rng.uniform(800, 2000)
        jw = rng.uniform(60, 95)
        jh = rng.uniform(80, 280)
        jd = rng.uniform(40, 90)
        segments.append(
            GeometrySegment(
                zone_id=zone_id,
                shape=SegmentShape.BOX,
                normal=None,
                centroid=Point3D(x=x, y=y, z=z),
                bounding_box=BoundingBox(
                    min_x=x - jw / 2,
                    min_y=y - jd / 2,
                    min_z=z - jh / 2,
                    max_x=x + jw / 2,
                    max_y=y + jd / 2,
                    max_z=z + jh / 2,
                ),
                point_count=int(rng.uniform(8, 40)),
                confidence=float(rng.uniform(0.45, 0.68)),
                source_file="synthetic",
            )
        )

    # ── Electrical: Lighting fittings (small elongated BOX near ceiling) ──────────
    for _i in range(num_lighting_fittings):
        x = rng.uniform(0, 50000)
        y = rng.uniform(0, 30000)
        z = rng.uniform(2600, 3200)
        ll = rng.uniform(300, 1800)
        lw = rng.uniform(60, 190)
        lh = rng.uniform(50, 120)
        segments.append(
            GeometrySegment(
                zone_id=zone_id,
                shape=SegmentShape.BOX,
                normal=None,
                centroid=Point3D(x=x + ll / 2, y=y, z=z),
                bounding_box=BoundingBox(
                    min_x=x,
                    min_y=y - lw / 2,
                    min_z=z - lh / 2,
                    max_x=x + ll,
                    max_y=y + lw / 2,
                    max_z=z + lh / 2,
                ),
                point_count=int(rng.uniform(10, 80)),
                confidence=float(rng.uniform(0.50, 0.72)),
                source_file="synthetic",
            )
        )

    # ── Fire: Fire hydrants (CYLINDER at hose-connection height 600–900 mm) ──────
    for _i in range(num_fire_hydrants):
        x = rng.uniform(0, 50000)
        y = rng.uniform(0, 30000)
        z = rng.uniform(600, 900)
        r = rng.uniform(50, 100)
        h = rng.uniform(80, 200)
        segments.append(
            GeometrySegment(
                zone_id=zone_id,
                shape=SegmentShape.CYLINDER,
                normal=None,
                centroid=Point3D(x=x, y=y, z=z),
                bounding_box=BoundingBox(
                    min_x=x - r,
                    min_y=y - r,
                    min_z=z - h / 2,
                    max_x=x + r,
                    max_y=y + r,
                    max_z=z + h / 2,
                ),
                point_count=int(rng.uniform(15, 80)),
                confidence=float(rng.uniform(0.55, 0.78)),
                source_file="synthetic",
            )
        )

    # ── Fire: Deluge valves (large VALVE_CANDIDATE on fire main) ─────────────────
    for _i in range(num_deluge_valves):
        x = rng.uniform(0, 50000)
        y = rng.uniform(0, 30000)
        z = rng.uniform(1000, 2500)
        dw = rng.uniform(250, 400)
        dd = rng.uniform(200, 350)
        dh = rng.uniform(350, 600)
        segments.append(
            GeometrySegment(
                zone_id=zone_id,
                shape=SegmentShape.VALVE_CANDIDATE,
                normal=None,
                centroid=Point3D(x=x, y=y, z=z),
                bounding_box=BoundingBox(
                    min_x=x - dw / 2,
                    min_y=y - dd / 2,
                    min_z=z - dh / 2,
                    max_x=x + dw / 2,
                    max_y=y + dd / 2,
                    max_z=z + dh / 2,
                ),
                point_count=int(rng.uniform(20, 120)),
                confidence=float(rng.uniform(0.55, 0.80)),
                source_file="synthetic",
            )
        )

    # ── Fire: Fire alarm panels (thin wall BOX at 1400–1800 mm) ──────────────────
    for _i in range(num_fire_alarm_panels):
        x = rng.uniform(0, 50000)
        y = rng.uniform(0, 30000)
        z = rng.uniform(1400, 1800)
        fw = rng.uniform(200, 590)
        fh = rng.uniform(250, 680)
        fd = rng.uniform(50, 95)
        segments.append(
            GeometrySegment(
                zone_id=zone_id,
                shape=SegmentShape.BOX,
                normal=None,
                centroid=Point3D(x=x + fw / 2, y=y, z=z),
                bounding_box=BoundingBox(
                    min_x=x,
                    min_y=y - fd / 2,
                    min_z=z - fh / 2,
                    max_x=x + fw,
                    max_y=y + fd / 2,
                    max_z=z + fh / 2,
                ),
                point_count=int(rng.uniform(30, 200)),
                confidence=float(rng.uniform(0.55, 0.78)),
                source_file="synthetic",
            )
        )

    # ── Fire: Smoke detectors (tiny ceiling CYLINDER > 2800 mm) ──────────────────
    for _i in range(num_smoke_detectors):
        x = rng.uniform(0, 50000)
        y = rng.uniform(0, 30000)
        z = rng.uniform(2850, 3200)
        r = rng.uniform(25, 70)
        h = rng.uniform(20, 75)
        segments.append(
            GeometrySegment(
                zone_id=zone_id,
                shape=SegmentShape.CYLINDER,
                normal=None,
                centroid=Point3D(x=x, y=y, z=z),
                bounding_box=BoundingBox(
                    min_x=x - r,
                    min_y=y - r,
                    min_z=z - h / 2,
                    max_x=x + r,
                    max_y=y + r,
                    max_z=z + h / 2,
                ),
                point_count=int(rng.uniform(5, 30)),
                confidence=float(rng.uniform(0.45, 0.70)),
                source_file="synthetic",
            )
        )

    logger.info(
        "synthetic_segments_generated",
        zone_id=zone_id,
        walls=num_walls,
        floors=num_floors,
        ceilings=num_floors,
        columns=num_columns,
        beams=num_beams,
        stairs=num_stairs,
        doors=num_doors,
        windows=num_windows,
        railings=num_railings,
        pipes=num_pipes,
        ducts=num_ducts,
        conduits=num_conduits,
        cable_trays=num_cable_trays,
        hvac_units=num_hvac_units,
        tanks=num_tanks,
        pumps=num_pumps,
        slab_openings=num_slab_openings,
        sprinklers=num_sprinklers,
        electrical_panels=num_electrical_panels,
        kerbs=num_kerbs,
        drains=num_drains,
        valves=num_valves,
        nuclear_industrial_added=30,
    )
    return segments


# --- PATCH: Large, dense, complex structure for full BIM scene ---
LARGE_BUILDING_KWARGS = dict(
    num_walls=80,
    num_floors=24,
    num_columns=40,
    num_beams=40,
    num_pipes=24,
    num_stairs=8,
    num_doors=40,
    num_windows=60,
    num_railings=20,
    num_ducts=20,
    num_conduits=20,
    num_cable_trays=12,
    num_hvac_units=8,
    num_tanks=8,
    num_pumps=8,
    num_slab_openings=16,
    num_sprinklers=40,
    num_electrical_panels=12,
    num_kerbs=12,
    num_drains=16,
    num_valves=24,
    num_ladders=8,
    num_gratings=8,
    num_overhead_cranes=4,
    num_hatches=8,
    num_trenches=4,
    num_bund_walls=8,
    num_safety_relief_valves=8,
    num_strainers=8,
    num_expansion_joints=8,
    num_fire_dampers=8,
    num_penetration_seals=4,
    num_pipe_supports=16,
    num_pressure_vessels=4,
    num_heat_exchangers=4,
    num_compressors=4,
    num_pressurizers=4,
    num_steam_generators=4,
    num_emergency_diesel_generators=4,
    num_seismic_isolators=8,
    num_containment_penetrations=4,
    num_radiation_monitors=8,
    num_transformers=4,
    num_switchgear=4,
    num_ups_systems=4,
    num_junction_boxes=16,
    num_lighting_fittings=40,
    num_fire_hydrants=8,
    num_deluge_valves=4,
    num_fire_alarm_panels=4,
    num_smoke_detectors=40,
)


def generate_large_synthetic_segments(zone_id: str = "zone-001"):
    """Generate a large, dense, complex synthetic BIM structure."""
    return generate_synthetic_segments(zone_id=zone_id, **LARGE_BUILDING_KWARGS)


# --- END PATCH ---


def _merge_circular_tanks(models: list) -> list:
    """Detect circular layouts of vertical segments (slats/pipes) and replace them with single large cylinders."""
    import numpy as np

    try:
        from sklearn.cluster import DBSCAN
    except ModuleNotFoundError:
        logger.warning(
            "merge_circular_tanks_skipped",
            reason="scikit-learn not installed",
        )
        return models
    from uuid import uuid4

    from agent.models import BoundingBox, ElementType, GeometrySegment, Point3D, SegmentShape

    candidates = []
    c_pts = []

    for idx, model in enumerate(models):
        et_value = getattr(model.element_type, "value", model.element_type)
        if not isinstance(et_value, str):
            continue
        et_str = et_value.lower()
        shape_value = getattr(model.shape, "value", model.shape)
        shape_str = str(shape_value).lower() if shape_value else ""
        if et_str in {"pipe", "conduit", "column", "unknown", "beam", "drainage", "kerb"}:
            dx = model.bounding_box.max_x - model.bounding_box.min_x
            dy = model.bounding_box.max_y - model.bounding_box.min_y
            dz = model.bounding_box.max_z - model.bounding_box.min_z
            cross = max(dx, dy)
            # Accept vertical cylinders OR tall narrow boxes (tank slats)
            is_vert_cyl = shape_str == "cylinder" and dz > max(dx, dy) * 1.0
            is_slat_box = shape_str in ("box", "cylinder") and dz > 300.0 and cross < 800.0
            if is_vert_cyl or is_slat_box:
                c_pts.append([model.centroid.x, model.centroid.y, model.centroid.z])
                candidates.append(idx)

    if len(c_pts) < 8:
        return models

    c_pts = np.array(c_pts)
    coords_xy = c_pts[:, :2]

    detected_tanks = []
    # Test multiple eps values to robustly find tanks of different densities/proximity
    for eps in [250.0, 300.0, 350.0, 400.0, 500.0, 600.0, 700.0]:
        db = DBSCAN(eps=eps, min_samples=6).fit(coords_xy)
        labels = db.labels_
        unique_labels = set(labels) - {-1}

        for label in unique_labels:
            mask = labels == label
            local_pts = c_pts[mask]
            local_cand_indices = [candidates[i] for i in np.where(mask)[0]]

            # Compute centroid
            xc, yc = local_pts[:, :2].mean(axis=0)
            dists = np.linalg.norm(local_pts[:, :2] - [xc, yc], axis=1)

            # Radius filter: keep points 200 to 2000 mm from centroid to isolate the tank perimeter
            inlier_mask = (dists >= 200.0) & (dists <= 2000.0)
            inliers = local_pts[inlier_mask]
            inlier_cands = [local_cand_indices[i] for i in np.where(inlier_mask)[0]]

            if len(inliers) >= 6:
                xc_new, yc_new = inliers[:, :2].mean(axis=0)
                dists_new = np.linalg.norm(inliers[:, :2] - [xc_new, yc_new], axis=1)
                r_mean = float(dists_new.mean())
                r_std = float(dists_new.std())
                ratio = r_std / max(r_mean, 1.0)

                # Adaptive thresholds: larger tanks need more evidence and stricter circularity
                if r_mean >= 700.0:
                    # Large tank (storage vessel): radius 700–2000mm, need 20+ members, ratio < 0.40
                    min_inliers, max_ratio = 20, 0.40
                elif r_mean >= 350.0:
                    # Medium tank: radius 350–700mm, need 8+ members, ratio < 0.40
                    min_inliers, max_ratio = 8, 0.40
                else:
                    # Small: radius 200–350mm, need 6+ members, ratio < 0.40
                    min_inliers, max_ratio = 6, 0.40

                # Check circularity criteria
                if r_mean >= 200.0 and ratio < max_ratio and len(inliers) >= min_inliers:
                    min_z = float(inliers[:, 2].min())
                    max_z = float(inliers[:, 2].max())
                    detected_tanks.append(
                        {
                            "xc": xc_new,
                            "yc": yc_new,
                            "r": r_mean,
                            "r_std": r_std,
                            "ratio": ratio,
                            "min_z": min_z,
                            "max_z": max_z,
                            "inliers_count": len(inliers),
                            "cand_indices": set(inlier_cands),
                        }
                    )

    # Deduplicate overlapping tank detections
    unique_tanks = []
    for tank in sorted(detected_tanks, key=lambda t: t["ratio"]):
        overlap = False
        for ut in unique_tanks:
            dist = np.sqrt((tank["xc"] - ut["xc"]) ** 2 + (tank["yc"] - ut["yc"]) ** 2)
            if dist < 600.0:  # within 600mm is the same physical tank
                overlap = True
                break
        if not overlap:
            unique_tanks.append(tank)
    merged_indices = set()
    new_models = []

    for tank in unique_tanks:
        xc = tank["xc"]
        yc = tank["yc"]
        radius = tank["r"]
        min_z = tank["min_z"]
        max_z = tank["max_z"]
        height = max_z - min_z
        centroid_z = (min_z + max_z) / 2.0
        member_indices = list(tank["cand_indices"])

        new_id = str(uuid4())
        new_seg = GeometrySegment(
            segment_id=new_id,
            zone_id=models[member_indices[0]].zone_id,
            shape=SegmentShape.CYLINDER,
            element_type=ElementType.PRESSURE_VESSEL,
            centroid=Point3D(x=float(xc), y=float(yc), z=float(centroid_z)),
            bounding_box=BoundingBox(
                min_x=float(xc - radius),
                max_x=float(xc + radius),
                min_y=float(yc - radius),
                max_y=float(yc + radius),
                min_z=float(min_z),
                max_z=float(max_z),
            ),
            point_count=int(sum(models[i].point_count for i in member_indices)),
            confidence=0.98,
            source_file=models[member_indices[0]].source_file,
            tags={
                "diameter_mm": float(2.0 * radius),
                "radius_mm": float(radius),
                "fitted_radius_mm": float(radius),
                "height_mm": float(height),
                "cyl_axis_x": 0.0,
                "cyl_axis_y": 0.0,
                "cyl_axis_z": 1.0,
                "dominant_color": "#7f7f7f",
                "reconstructed_tank": True,
                "tank_floor_z_mm": float(min_z),
                "tank_top_z_mm": float(max_z),
            },
        )
        new_models.append(new_seg)
        merged_indices.update(member_indices)

    final_models = [m for idx, m in enumerate(models) if idx not in merged_indices]
    final_models.extend(new_models)
    return final_models
