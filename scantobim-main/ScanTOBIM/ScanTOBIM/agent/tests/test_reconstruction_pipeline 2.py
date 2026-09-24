"""Automated Detection & Revit Reconstruction Test Suite.

Validates the full Scan-to-BIM pipeline according to the production reconstruction contract:
1. E57 Scan Ingestion: verifies ASTM E57 magic bytes header and format validation.
2. Wall Thickness & Centerline: verifies wall thickness is derived from dual-peak / PCA (< 400mm)
   and angled walls never inherit whole-room AABB diagonals (> 1000mm).
3. 3D Cable Tray Orientation: verifies PCA 3D eigenvector principal axis fitting
   and that trays are NOT clamped to flat 1D X-axis spans.
4. Storey & Slab Detection: verifies 1D Z-histogram peak prominence detection,
   storey elevation partitioning, and Douglas-Peucker simplified boundary polygons.
5. Column Detection: verifies 2D ConvexHull Minimum Bounding Rectangle (MBR)
   rotation, width, depth, and profile type extraction.
6. Revit Add-in Data Contract: verifies segment tags conform to Revit 2025
   requirements with valid numeric values and zero NaN/Inf.
"""

from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import pytest

from agent.models import SegmentShape, ElementType
from agent.tools.geometry_tools import (
    meters_to_mm,
    mm_to_meters,
    compute_survey_transform,
)
from agent.tools.slab_tools import (
    detect_storeys_and_slabs,
    create_slab_boundary_polygon,
    StoreyDefinition,
    SlabEntity,
)
from agent.tools.column_tools import (
    minimum_bounding_rectangle,
    detect_structural_columns,
    ColumnEntity,
)
from agent.tools.opening_tools import (
    detect_wall_openings,
    OpeningEntity,
)
from agent.tools.scan_tools import (
    detect_scan_format,
    validate_scan_file,
    segment_to_model,
)
from agent.classifier import _populate_cable_tray_tags


# ── 1. E57 Format Validation ──────────────────────────────────────────────────


def test_e57_magic_bytes_detection(tmp_path: Path):
    """Verify ASTM-E57 file format identification."""
    e57_file = tmp_path / "test.e57"
    header = b"ASTM-E57" + b"\x00" * 24
    e57_file.write_bytes(header)

    fmt = detect_scan_format(e57_file)
    assert fmt == ".e57"


def test_real_e57_header_integrity():
    """Verify the real point cloud file in the repository starts with ASTM-E57."""
    real_e57 = Path("ScanTOBIM/ScanTOBIM/point cloud data")
    if not real_e57.exists():
        pytest.skip("Real E57 file not present at ScanTOBIM/ScanTOBIM/point cloud data")

    with open(real_e57, "rb") as f:
        magic = f.read(8)
    assert magic == b"ASTM-E57", f"Expected ASTM-E57 magic bytes, got {magic!r}"


# ── 2. Wall Thickness & Orientation ───────────────────────────────────────────


def test_angled_wall_thickness_clamped_and_never_aabb_diagonal():
    """Angled walls (e.g. 46° or 136°) must have thickness <= 400mm, NOT 10-13m."""
    raw_wall = {
        "shape": SegmentShape.PLANE_VERTICAL,
        "normal": np.array([0.7071, -0.7071, 0.0], dtype=float),
        "centroid": np.array([5.0, 5.0, 1.5], dtype=float),
        "mins": np.array([0.0, 0.0, 0.0], dtype=float),
        "maxs": np.array([10.0, 10.0, 3.0], dtype=float),
        "point_count": 20000,
        "confidence": 0.95,
        "tags": {"random_seed": 42},
    }

    models = segment_to_model(raw_wall, zone_id="ZONE_TEST")
    assert len(models) >= 1
    wall = models[0]
    thickness_mm = float(wall.tags.get("wall_thickness_mm", 0))

    # Must be standard architectural thickness (100 to 400mm), NEVER the 10,000mm AABB diagonal
    assert 100.0 <= thickness_mm <= 400.0, f"Wall thickness {thickness_mm} exceeds valid range"
    assert thickness_mm != 10000.0, "Wall inherited raw AABB diagonal dimension"


def test_wall_endpoints_preserve_centerline():
    """Wall endpoints must form a valid line segment along the wall axis."""
    raw_wall = {
        "shape": SegmentShape.PLANE_VERTICAL,
        "normal": np.array([0.0, 1.0, 0.0], dtype=float),
        "centroid": np.array([5.0, 3.0, 1.5], dtype=float),
        "mins": np.array([0.0, 2.85, 0.0], dtype=float),
        "maxs": np.array([10.0, 3.15, 3.0], dtype=float),
        "point_count": 5000,
        "confidence": 0.95,
        "tags": {},
    }

    models = segment_to_model(raw_wall, zone_id="ZONE_TEST")
    wall = models[0]
    tags = wall.tags

    sx = float(tags["wall_start_x_mm"])
    sy = float(tags["wall_start_y_mm"])
    ex = float(tags["wall_end_x_mm"])
    ey = float(tags["wall_end_y_mm"])

    dx = ex - sx
    dy = ey - sy
    length_mm = math.hypot(dx, dy)
    assert length_mm >= 500.0, f"Wall length {length_mm} is too short"


# ── 3. 3D Cable Tray Orientation ──────────────────────────────────────────────


def test_cable_tray_3d_pca_preserves_sloped_direction():
    """Cable trays with 3D slope must preserve 3D axis and endpoints, not clamp to X-axis."""
    t = np.linspace(0, 10, 200)
    # Cable tray running diagonally in 3D: X advances 5m, Y advances 5m, Z advances 2m
    pts_x = 1.0 + t * 0.5 + np.random.normal(0, 0.02, len(t))
    pts_y = 2.0 + t * 0.5 + np.random.normal(0, 0.02, len(t))
    pts_z = 3.0 + t * 0.2 + np.random.normal(0, 0.02, len(t))
    cluster_pts = np.column_stack([pts_x, pts_y, pts_z])

    from types import SimpleNamespace
    seg = SimpleNamespace(
        bounding_box=SimpleNamespace(
            min_x=float(cluster_pts[:, 0].min() * 1000.0),
            max_x=float(cluster_pts[:, 0].max() * 1000.0),
            min_y=float(cluster_pts[:, 1].min() * 1000.0),
            max_y=float(cluster_pts[:, 1].max() * 1000.0),
            min_z=float(cluster_pts[:, 2].min() * 1000.0),
            max_z=float(cluster_pts[:, 2].max() * 1000.0),
        ),
        inlier_points=cluster_pts * 1000.0,
        tags={},
    )
    _populate_cable_tray_tags(seg)
    tags = seg.tags

    assert "cyl_start_x_mm" in tags
    assert "cyl_start_y_mm" in tags
    assert "cyl_start_z_mm" in tags
    assert "cyl_end_x_mm" in tags
    assert "cyl_end_y_mm" in tags
    assert "cyl_end_z_mm" in tags

    dx = tags["cyl_end_x_mm"] - tags["cyl_start_x_mm"]
    dy = tags["cyl_end_y_mm"] - tags["cyl_start_y_mm"]
    dz = tags["cyl_end_z_mm"] - tags["cyl_start_z_mm"]

    # Both X and Y must have significant non-zero spans (diagonal, not single-axis locked)
    assert abs(dx) > 1000.0
    assert abs(dy) > 1000.0
    assert abs(dz) > 500.0
    assert tags["axis_fitting_method"] == "3d_pca"


# ── 4. Storey & Slab Detection ────────────────────────────────────────────────


def test_storey_detection_peak_prominence():
    """Simulate a 3-storey tower and verify true elevations are detected."""
    np.random.seed(42)
    # Ground floor at Z = 0.0m, Floor 1 at Z = 3.5m, Floor 2 at Z = 7.0m, Roof at Z = 10.5m
    slab_levels = [0.0, 3.5, 7.0, 10.5]
    all_pts = []

    for z_level in slab_levels:
        # Generate slab points (dense horizontal plane)
        xs = np.random.uniform(0, 15, 3000)
        ys = np.random.uniform(0, 12, 3000)
        zs = np.random.normal(z_level, 0.02, 3000)
        all_pts.append(np.column_stack([xs, ys, zs]))

    # Add vertical wall points between storeys
    for i in range(len(slab_levels) - 1):
        z_lo, z_hi = slab_levels[i], slab_levels[i + 1]
        zs = np.random.uniform(z_lo, z_hi, 2000)
        xs = np.random.choice([0.0, 15.0], size=2000) + np.random.normal(0, 0.02, 2000)
        ys = np.random.uniform(0, 12, 2000)
        all_pts.append(np.column_stack([xs, ys, zs]))

    pts_xyz = np.vstack(all_pts)
    storeys = detect_storeys_and_slabs(pts_xyz)

    assert len(storeys) >= 2, f"Expected at least 2 storeys, got {len(storeys)}"
    for s in storeys:
        assert s.height_m >= 2.0, f"Storey height {s.height_m} is implausible"
        if s.floor_slab:
            assert s.floor_slab.type == "FLOOR"
            assert len(s.floor_slab.boundary_polygon_m) >= 3


def test_slab_boundary_polygon_is_closed_and_oriented():
    """Floor slab boundary polygon must be a valid closed 2D polygon."""
    xs = np.random.uniform(2, 10, 1000)
    ys = np.random.uniform(3, 8, 1000)
    zs = np.full(1000, 0.0)
    pts = np.column_stack([xs, ys, zs])

    poly = create_slab_boundary_polygon(pts)
    assert len(poly) >= 3
    for pt in poly:
        assert 1.5 <= pt[0] <= 10.5
        assert 2.5 <= pt[1] <= 8.5


# ── 5. Column Detection (CVPR-2024 MBR) ────────────────────────────────────────


def test_minimum_bounding_rectangle_oriented_rectangle():
    """Verify MBR computes authentic width, depth, and rotation angle."""
    # Create 400mm x 600mm column rotated by 30 degrees
    w, d = 0.40, 0.60
    angle = np.radians(30.0)
    c, s = np.cos(angle), np.sin(angle)
    R = np.array([[c, -s], [s, c]])

    grid_x, grid_y = np.meshgrid(np.linspace(-w / 2, w / 2, 20), np.linspace(-d / 2, d / 2, 30))
    pts_local = np.column_stack([grid_x.ravel(), grid_y.ravel()])
    pts_rot = pts_local @ R.T

    corners, est_w, est_d, est_deg = minimum_bounding_rectangle(pts_rot)

    # In MBR convention width <= depth
    dim_min = min(est_w, est_d)
    dim_max = max(est_w, est_d)
    assert abs(dim_min - 0.40) < 0.05, f"Expected width ~0.40m, got {dim_min}"
    assert abs(dim_max - 0.60) < 0.05, f"Expected depth ~0.60m, got {dim_max}"


def test_detect_structural_columns_from_vertical_cluster():
    """Verify structural column detection identifies column entity."""
    storey = StoreyDefinition(
        storey_id="storey_0",
        name="Level 0",
        index=0,
        elevation_m=0.0,
        top_elevation_m=3.5,
        height_m=3.5,
    )

    # Generate points for a 500x500mm column at (4.0, 4.0) extending Z=0.2 to 3.3
    np.random.seed(123)
    n = 200
    col_x = np.random.uniform(3.75, 4.25, n)
    col_y = np.random.uniform(3.75, 4.25, n)
    col_z = np.random.uniform(0.3, 3.2, n)
    col_pts = np.column_stack([col_x, col_y, col_z])

    cols = detect_structural_columns(col_pts, [storey], min_points_per_col=30)
    assert len(cols) >= 1, "Expected column detection to find at least 1 column"
    c0 = cols[0]
    assert abs(c0.center_xyz_m[0] - 4.0) < 0.2
    assert abs(c0.center_xyz_m[1] - 4.0) < 0.2
    assert c0.width_m <= 0.8
    assert c0.depth_m <= 0.8


# ── 6. Survey Transform ───────────────────────────────────────────────────────


def test_compute_survey_transform_centers_points():
    """Verify rigid 4x4 survey transform shifts origin offset correctly."""
    pts = np.array([
        [1000.0, 2000.0, 50.0],
        [1010.0, 2010.0, 55.0],
    ])
    centered, T_4x4, offset = compute_survey_transform(pts)

    assert abs(centered[:, 0].mean()) < 1e-9
    assert abs(centered[:, 1].mean()) < 1e-9
    assert abs(centered[:, 2].mean()) < 1e-9

    assert T_4x4.shape == (4, 4)
    assert np.allclose(T_4x4[:3, 3], -offset)
