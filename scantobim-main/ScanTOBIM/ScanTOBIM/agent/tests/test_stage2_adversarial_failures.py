"""Adversarial failure mode test suite for Scan-to-BIM Stage 2.

Covers all 12 failure modes required by Section 19:
1. Large vertical MEP panel
2. Equipment casing
3. Cylindrical equipment
4. Pipe support
5. Large architectural wall
6. Fragmented architectural wall
7. Wall with door opening
8. Wall with window opening
9. Wall partially occluded by MEP
10. Nearby parallel equipment surface
11. Corner wall (perpendicular)
12. Small legitimate wall fragment belonging to a large wall
"""

import sys

# Remove mock open3d if injected by parent conftest so real open3d is used
for _mod in list(sys.modules):
    if _mod == "open3d" or _mod.startswith("open3d."):
        del sys.modules[_mod]

import math
import numpy as np
import open3d as o3d
import pytest

from agent.models import SegmentShape
from agent.tools.scan_tools import merge_coplanar_walls
from agent.tools.wall_classifier import (
    WallClassification,
    compute_wall_likeness_score,
    classify_features,
    extract_plane_features,
)


def _make_plane_dict(
    points: np.ndarray,
    normal: np.ndarray,
    wall_axis: np.ndarray,
    normals: np.ndarray | None = None,
) -> dict:
    """Helper to create a standard plane dictionary with Open3D PointCloud and raw arrays."""
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)
    if normals is not None:
        pcd.normals = o3d.utility.Vector3dVector(normals)

    c = points.mean(axis=0)
    mins = points.min(axis=0)
    maxs = points.max(axis=0)
    length = float(np.ptp(points @ np.array([wall_axis[0], wall_axis[1], 0.0])))

    return {
        "shape": SegmentShape.PLANE_VERTICAL,
        "normal": normal,
        "centroid": c,
        "mins": mins,
        "maxs": maxs,
        "points": points,
        "normals": normals,
        "point_count": len(points),
        "confidence": 1.0,
        "inlier_cloud": pcd,
        "tags": {
            "wall_axis_x": float(wall_axis[0]),
            "wall_axis_y": float(wall_axis[1]),
            "wall_length_mm": round(length * 1000.0, 1),
            "_wall_start_x_m": float(c[0] - wall_axis[0] * length / 2.0),
            "_wall_start_y_m": float(c[1] - wall_axis[1] * length / 2.0),
            "_wall_end_x_m": float(c[0] + wall_axis[0] * length / 2.0),
            "_wall_end_y_m": float(c[1] + wall_axis[1] * length / 2.0),
        },
    }


# 1. Large vertical MEP panel: Non-aligned interior panel cutting through center
def test_reject_large_vertical_mep_panel():
    rng = np.random.default_rng(101)
    u = np.linspace(-1.1, 1.1, 30)
    z = np.linspace(0.0, 2.0, 30)
    U, Z = np.meshgrid(u, z)
    rad = math.radians(72.0)
    x = U.ravel() * math.cos(rad) + 0.1
    y = U.ravel() * math.sin(rad) + 0.2
    pts = np.column_stack([x, y, Z.ravel()])

    plane = _make_plane_dict(
        points=pts,
        normal=np.array([-math.sin(rad), math.cos(rad), 0.0]),
        wall_axis=np.array([math.cos(rad), math.sin(rad)]),
    )
    dom_angles = [0.0, 90.0]
    center_xy = np.array([0.0, 0.0])

    feat = extract_plane_features(0, plane, center_xy, dom_angles)
    assert feat.classification == WallClassification.MEP_EQUIPMENT
    assert feat.wall_score < 0.60


# 2. Equipment casing: small box-like enclosure
def test_reject_equipment_casing():
    u = np.linspace(-0.4, 0.4, 15)
    z = np.linspace(0.0, 0.8, 15)
    U, Z = np.meshgrid(u, z)
    pts = np.column_stack([U.ravel(), np.zeros(len(U.ravel())), Z.ravel()])

    plane = _make_plane_dict(
        points=pts,
        normal=np.array([0.0, 1.0, 0.0]),
        wall_axis=np.array([1.0, 0.0]),
    )
    feat = extract_plane_features(1, plane, np.array([0.0, 0.0]), [0.0, 90.0])
    assert feat.classification in (
        WallClassification.FURNITURE_OBJECT,
        WallClassification.MEP_EQUIPMENT,
    )


# 3. Cylindrical equipment: tangential patch with curved surface normals
def test_reject_cylindrical_equipment():
    theta = np.linspace(-0.35, 0.35, 30)
    z = np.linspace(0.0, 2.5, 30)
    T, Z = np.meshgrid(theta, z)
    x = np.cos(T.ravel())
    y = np.sin(T.ravel())
    pts = np.column_stack([x, y, Z.ravel()])
    normals = np.column_stack([np.cos(T.ravel()), np.sin(T.ravel()), np.zeros(len(x))])

    plane = _make_plane_dict(
        points=pts,
        normal=np.array([1.0, 0.0, 0.0]),
        wall_axis=np.array([0.0, 1.0]),
        normals=normals,
    )
    feat = extract_plane_features(2, plane, np.array([0.0, 0.0]), [0.0, 90.0])
    assert feat.classification in (
        WallClassification.MEP_EQUIPMENT,
        WallClassification.STRUCTURAL_NONWALL,
    )


# 4. Pipe support: narrow vertical strip
def test_reject_pipe_support():
    u = np.linspace(-0.06, 0.06, 6)
    z = np.linspace(0.0, 2.8, 30)
    U, Z = np.meshgrid(u, z)
    pts = np.column_stack([U.ravel(), np.zeros(len(U.ravel())), Z.ravel()])

    plane = _make_plane_dict(
        points=pts,
        normal=np.array([0.0, 1.0, 0.0]),
        wall_axis=np.array([1.0, 0.0]),
    )
    feat = extract_plane_features(3, plane, np.array([0.0, 0.0]), [0.0, 90.0])
    assert feat.classification in (
        WallClassification.STRUCTURAL_NONWALL,
        WallClassification.MEP_EQUIPMENT,
    )


# 5. Large architectural wall: full scale building envelope
def test_accept_large_architectural_wall():
    u = np.linspace(-3.0, 3.0, 50)
    z = np.linspace(0.0, 3.5, 40)
    U, Z = np.meshgrid(u, z)
    pts = np.column_stack([U.ravel(), np.full(len(U.ravel()), 4.0), Z.ravel()])
    norms = np.tile(np.array([0.0, 1.0, 0.0]), (len(pts), 1))

    plane = _make_plane_dict(
        points=pts,
        normal=np.array([0.0, 1.0, 0.0]),
        wall_axis=np.array([1.0, 0.0]),
        normals=norms,
    )
    feat = extract_plane_features(4, plane, np.array([0.0, 0.0]), [0.0, 90.0])
    assert feat.classification == WallClassification.ARCHITECTURAL_WALL
    assert feat.wall_score >= 0.70


# 6. Fragmented architectural wall: collinear segments
def test_accept_fragmented_architectural_wall():
    u1 = np.linspace(0.0, 2.5, 30)
    z1 = np.linspace(0.0, 3.0, 30)
    U1, Z1 = np.meshgrid(u1, z1)
    pts1 = np.column_stack([U1.ravel(), np.full(len(U1.ravel()), 3.0), Z1.ravel()])

    u2 = np.linspace(3.3, 5.8, 30)
    z2 = np.linspace(0.0, 3.0, 30)
    U2, Z2 = np.meshgrid(u2, z2)
    pts2 = np.column_stack([U2.ravel(), np.full(len(U2.ravel()), 3.0), Z2.ravel()])

    p1 = _make_plane_dict(pts1, np.array([0.0, 1.0, 0.0]), np.array([1.0, 0.0]))
    p2 = _make_plane_dict(pts2, np.array([0.0, 1.0, 0.0]), np.array([1.0, 0.0]))

    merged = merge_coplanar_walls([p1, p2], gap_threshold_m=1.20)
    assert len(merged) == 1
    assert merged[0]["tags"].get("wall_merged") is True
    assert merged[0]["tags"]["wall_length_mm"] > 5000.0


# 7. Wall with door opening: 1.0m doorway gap between wall sections
def test_accept_wall_with_door_opening():
    u_left = np.linspace(-3.0, -0.5, 30)
    u_right = np.linspace(0.5, 3.0, 30)
    z = np.linspace(0.0, 3.0, 30)

    U_l, Z_l = np.meshgrid(u_left, z)
    pts_l = np.column_stack([U_l.ravel(), np.full(len(U_l.ravel()), 2.0), Z_l.ravel()])

    U_r, Z_r = np.meshgrid(u_right, z)
    pts_r = np.column_stack([U_r.ravel(), np.full(len(U_r.ravel()), 2.0), Z_r.ravel()])

    p_left = _make_plane_dict(pts_l, np.array([0.0, 1.0, 0.0]), np.array([1.0, 0.0]))
    p_right = _make_plane_dict(pts_r, np.array([0.0, 1.0, 0.0]), np.array([1.0, 0.0]))

    merged = merge_coplanar_walls([p_left, p_right], gap_threshold_m=1.20)
    assert len(merged) == 1
    assert merged[0]["tags"]["wall_length_mm"] >= 5800.0


# 8. Wall with window opening: wall regions flanking a window punch opening
def test_accept_wall_with_window_opening():
    u_l = np.linspace(0.0, 2.0, 25)
    u_r = np.linspace(3.0, 5.0, 25)
    z = np.linspace(0.0, 3.0, 25)

    U1, Z1 = np.meshgrid(u_l, z)
    U2, Z2 = np.meshgrid(u_r, z)
    pts1 = np.column_stack([U1.ravel(), np.zeros(len(U1.ravel())), Z1.ravel()])
    pts2 = np.column_stack([U2.ravel(), np.zeros(len(U2.ravel())), Z2.ravel()])

    p1 = _make_plane_dict(pts1, np.array([0.0, 1.0, 0.0]), np.array([1.0, 0.0]))
    p2 = _make_plane_dict(pts2, np.array([0.0, 1.0, 0.0]), np.array([1.0, 0.0]))

    merged = merge_coplanar_walls([p1, p2], gap_threshold_m=1.20)
    assert len(merged) == 1


# 9. Wall partially occluded by MEP: wall retains planar support
def test_accept_wall_partially_occluded_by_mep():
    u = np.linspace(-2.5, 2.5, 40)
    z = np.linspace(0.0, 3.0, 30)
    U, Z = np.meshgrid(u, z)
    mask = ~((np.abs(U.ravel()) < 0.5) & (Z.ravel() > 0.5) & (Z.ravel() < 2.0))
    pts = np.column_stack([U.ravel()[mask], np.full(np.sum(mask), 3.5), Z.ravel()[mask]])

    plane = _make_plane_dict(pts, np.array([0.0, 1.0, 0.0]), np.array([1.0, 0.0]))
    feat = extract_plane_features(5, plane, np.array([0.0, 0.0]), [0.0, 90.0])
    assert feat.classification == WallClassification.ARCHITECTURAL_WALL
    assert feat.wall_score >= 0.60


# 10. Nearby parallel equipment surface: should NOT be merged with wall
def test_reject_nearby_parallel_equipment_surface():
    u = np.linspace(0.0, 3.0, 30)
    z = np.linspace(0.0, 3.0, 30)
    U, Z = np.meshgrid(u, z)

    pts_wall = np.column_stack([U.ravel(), np.zeros(len(U.ravel())), Z.ravel()])
    pts_eq = np.column_stack([U.ravel(), np.full(len(U.ravel()), 0.50), Z.ravel()])

    p_wall = _make_plane_dict(pts_wall, np.array([0.0, 1.0, 0.0]), np.array([1.0, 0.0]))
    p_eq = _make_plane_dict(pts_eq, np.array([0.0, 1.0, 0.0]), np.array([1.0, 0.0]))

    merged = merge_coplanar_walls([p_wall, p_eq], lateral_offset_threshold_m=0.30)
    assert len(merged) == 2


# 11. Corner wall: two perpendicular walls meeting at 90 degrees
def test_accept_corner_wall():
    u = np.linspace(0.0, 3.5, 35)
    z = np.linspace(0.0, 3.0, 30)
    U, Z = np.meshgrid(u, z)

    pts_a = np.column_stack([U.ravel(), np.full(len(U.ravel()), 2.5), Z.ravel()])
    pts_b = np.column_stack([np.full(len(U.ravel()), 2.5), U.ravel(), Z.ravel()])

    p_a = _make_plane_dict(pts_a, np.array([0.0, 1.0, 0.0]), np.array([1.0, 0.0]))
    p_b = _make_plane_dict(pts_b, np.array([1.0, 0.0, 0.0]), np.array([0.0, 1.0]))

    feat_a = extract_plane_features(6, p_a, np.array([0.0, 0.0]), [0.0, 90.0])
    feat_b = extract_plane_features(7, p_b, np.array([0.0, 0.0]), [0.0, 90.0])

    assert feat_a.classification == WallClassification.ARCHITECTURAL_WALL
    assert feat_b.classification == WallClassification.ARCHITECTURAL_WALL

    merged = merge_coplanar_walls([p_a, p_b])
    assert len(merged) == 2


# 12. Small legitimate wall fragment belonging to a large wall
def test_accept_small_legitimate_fragment():
    u_large = np.linspace(0.0, 4.0, 40)
    z_large = np.linspace(0.0, 3.0, 30)
    Ul, Zl = np.meshgrid(u_large, z_large)
    pts_large = np.column_stack([Ul.ravel(), np.full(len(Ul.ravel()), 3.0), Zl.ravel()])

    u_small = np.linspace(4.5, 5.5, 15)
    z_small = np.linspace(0.0, 1.8, 18)
    Us, Zs = np.meshgrid(u_small, z_small)
    pts_small = np.column_stack([Us.ravel(), np.full(len(Us.ravel()), 3.0), Zs.ravel()])

    p_large = _make_plane_dict(pts_large, np.array([0.0, 1.0, 0.0]), np.array([1.0, 0.0]))
    p_small = _make_plane_dict(pts_small, np.array([0.0, 1.0, 0.0]), np.array([1.0, 0.0]))

    feat_small = extract_plane_features(8, p_small, np.array([0.0, 0.0]), [0.0, 90.0])
    assert feat_small.classification in (
        WallClassification.WALL_FRAGMENT,
        WallClassification.ARCHITECTURAL_WALL,
    )

    merged = merge_coplanar_walls([p_large, p_small], gap_threshold_m=1.0)
    assert len(merged) == 1
    assert merged[0]["tags"]["wall_length_mm"] >= 5300.0


# 13. Curved surface represented by multiple planar facets → NOT one wall per facet
def test_curved_surface_facets_not_one_wall_per_facet():
    from agent.stage2_pipeline import (
        compute_curvature_evidence,
        run_stage2_pipeline,
    )

    # Create 6 tangential facets approximating a cylindrical tower of radius 1.5m
    radius = 1.5
    angles_deg = [10.0, 35.0, 60.0, 85.0, 110.0, 135.0]
    facets = []
    z_vals = np.linspace(0.0, 15.0, 25)

    for idx, ang in enumerate(angles_deg):
        rad = math.radians(ang)
        n = np.array([math.cos(rad), math.sin(rad), 0.0])
        c = np.array([radius * math.cos(rad), radius * math.sin(rad), 7.5])
        u_vals = np.linspace(-0.5, 0.5, 15)
        U, Z = np.meshgrid(u_vals, z_vals)
        ax = np.array([-math.sin(rad), math.cos(rad), 0.0])
        x = c[0] + U.ravel() * ax[0]
        y = c[1] + U.ravel() * ax[1]
        pts = np.column_stack([x, y, Z.ravel()])

        p = _make_plane_dict(pts, n, ax[:2])
        p["centroid"] = c
        facets.append(p)

    cloud_center = np.array([0.0, 0.0])

    # Verify curvature evidence is high
    for idx in range(len(facets)):
        c_ev = compute_curvature_evidence(idx, facets, cloud_center)
        assert c_ev >= 0.50, f"Facet {idx} curvature evidence {c_ev} should be >= 0.50"

    # Run stage2 pipeline on these facets
    result = run_stage2_pipeline(facets, cloud_center_xy=cloud_center)

    # Mandatory check: Must NOT create 6 individual architectural walls!
    assert len(result.physical_walls) == 0, f"Expected 0 individual walls, got {len(result.physical_walls)}"
    assert result.curved_structural_elements == 1, f"Expected 1 curved structural element, got {result.curved_structural_elements}"


# 14. Isolated small vertical object → REJECT
def test_reject_isolated_small_vertical_object():
    from agent.stage2_pipeline import extract_plane_features

    # Small 200mm x 200mm vertical plate (e.g. electrical bracket/junction box face)
    u = np.linspace(-0.1, 0.1, 10)
    z = np.linspace(1.2, 1.4, 10)
    U, Z = np.meshgrid(u, z)
    pts = np.column_stack([U.ravel(), np.full(len(U.ravel()), 5.0), Z.ravel()])

    plane = _make_plane_dict(pts, np.array([0.0, 1.0, 0.0]), np.array([1.0, 0.0]))
    feat = extract_plane_features(99, plane, np.array([0.0, 0.0]), [0.0, 90.0])

    assert feat.classification in (
        WallClassification.FURNITURE_OBJECT,
        WallClassification.MEP_EQUIPMENT,
    )
    assert feat.classification != WallClassification.ARCHITECTURAL_WALL

