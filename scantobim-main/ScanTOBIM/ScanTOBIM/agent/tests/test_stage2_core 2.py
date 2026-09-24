"""Comprehensive tests for Scan-to-BIM Stage 2 plane detection, classification, orientation, and merging."""

import math
import numpy as np
import pytest

from agent.models import (
    BoundingBox,
    ElementType,
    GeometrySegment,
    Point3D,
    SegmentShape,
)
from agent.tools.detection_config import (
    DEFAULT_WALL_THICKNESS_MM,
    MAX_WALL_THICKNESS_MM,
    MIN_WALL_THICKNESS_MM,
)
from agent.tools.geometry_tools import (
    canonicalize_wall_axis,
    classify_plane_orientation,
    count_stair_steps,
    wall_angle_degrees,
)
from agent.stage2_pipeline import group_wall_fragments, pair_opposing_wall_faces
from agent.classifier import classify_segment


def test_a_vertical_plane_classification():
    """Test A: Plane with normal ~ [0.707, 0.707, 0] -> VERTICAL, wall angle ~ 45 deg."""
    normal = np.array([1.0 / math.sqrt(2), 1.0 / math.sqrt(2), 0.0])
    ori = classify_plane_orientation(normal)
    assert ori == SegmentShape.PLANE_VERTICAL or ori == "plane_vertical"

    # Wall angle in XY plane
    angle = wall_angle_degrees(normal)
    assert 40.0 <= angle <= 50.0 or 130.0 <= angle <= 140.0


def test_b_opposite_normals_canonicalization():
    """Test B: Plane with normal +n and -n map to the same canonical wall axis and orientation class."""
    n_pos = np.array([0.70710678, 0.70710678, 0.0])
    n_neg = np.array([-0.70710678, -0.70710678, 0.0])

    ori_pos = classify_plane_orientation(n_pos)
    ori_neg = classify_plane_orientation(n_neg)
    assert ori_pos in (SegmentShape.PLANE_VERTICAL, "plane_vertical")
    assert ori_neg in (SegmentShape.PLANE_VERTICAL, "plane_vertical")

    # Opposite wall face normals must map to identical canonical axis (modulo 180 deg)
    ax_pos = canonicalize_wall_axis(n_pos)
    ax_neg = canonicalize_wall_axis(n_neg)
    np.testing.assert_allclose(ax_pos, ax_neg, atol=1e-5)

    # Angles must match
    ang_pos = wall_angle_degrees(n_pos)
    ang_neg = wall_angle_degrees(n_neg)
    assert abs(ang_pos - ang_neg) < 1e-4


def test_c_horizontal_floor():
    """Test C: Plane with normal ~ [0, 0, 1] -> HORIZONTAL, semantic class FLOOR (if low Z)."""
    normal = np.array([0.0, 0.0, 1.0])
    ori = classify_plane_orientation(normal)
    assert ori in (SegmentShape.PLANE_HORIZONTAL, "plane_horizontal")

    seg = GeometrySegment(
        zone_id="zone-001",
        confidence=0.95,
        shape=SegmentShape.PLANE_HORIZONTAL,
        centroid=Point3D(x=0.0, y=0.0, z=50.0),
        bounding_box=BoundingBox(min_x=0.0, min_y=0.0, min_z=0.0, max_x=5000.0, max_y=5000.0, max_z=100.0),
        normal=Point3D(x=0.0, y=0.0, z=1.0),
        point_count=5000,
    )
    elem_type, disc, safety = classify_segment(seg)
    assert elem_type == ElementType.FLOOR


def test_d_horizontal_ceiling():
    """Test D: Plane with normal ~ [0, 0, -1] -> HORIZONTAL, semantic class CEILING (if high Z)."""
    normal = np.array([0.0, 0.0, -1.0])
    ori = classify_plane_orientation(normal)
    assert ori in (SegmentShape.PLANE_HORIZONTAL, "plane_horizontal")

    seg = GeometrySegment(
        zone_id="zone-001",
        confidence=0.95,
        shape=SegmentShape.PLANE_HORIZONTAL,
        centroid=Point3D(x=0.0, y=0.0, z=2800.0),
        bounding_box=BoundingBox(min_x=0.0, min_y=0.0, min_z=2700.0, max_x=5000.0, max_y=5000.0, max_z=2900.0),
        normal=Point3D(x=0.0, y=0.0, z=-1.0),
        point_count=5000,
    )
    elem_type, disc, safety = classify_segment(seg)
    assert elem_type == ElementType.CEILING


def test_e_sloped_plane():
    """Test E: Plane with normal ~ [0, 0.707, 0.707] -> SLOPED (slope angle ~45 deg)."""
    normal = np.array([0.0, 0.70710678, 0.70710678])
    ori = classify_plane_orientation(normal)
    assert ori in (SegmentShape.PLANE_SLOPED, "plane_sloped")


def test_f_arbitrary_wall_angle_preservation():
    """Test F: Angled wall at ~45.8 deg is preserved and not snapped to 0 or 90."""
    wall_deg = 45.8
    rad = math.radians(wall_deg)
    # Face normal perpendicular to wall axis
    normal = np.array([-math.sin(rad), math.cos(rad), 0.0])
    # Wall axis points along the wall
    axis = np.array([-normal[1], normal[0]])
    measured_deg = wall_angle_degrees(axis)
    assert abs(measured_deg - wall_deg) < 0.5


def test_g_second_wall_family_preservation():
    """Test G: Second wall family at ~135.8 deg is preserved."""
    wall_deg = 135.8
    rad = math.radians(wall_deg)
    normal = np.array([-math.sin(rad), math.cos(rad), 0.0])
    axis = np.array([-normal[1], normal[0]])
    measured_deg = wall_angle_degrees(axis)
    assert abs(measured_deg - wall_deg) < 0.5


def test_h_perpendicular_walls_remain_distinct():
    """Test H: Walls at 45.8 deg and 135.8 deg are treated as distinct perpendicular groups."""
    rad_a = math.radians(45.8)
    normal_a = np.array([-math.sin(rad_a), math.cos(rad_a), 0.0])

    rad_b = math.radians(135.8)
    normal_b = np.array([-math.sin(rad_b), math.cos(rad_b), 0.0])

    axis_a = canonicalize_wall_axis(normal_a)
    axis_b = canonicalize_wall_axis(normal_b)

    # Dot product of axes should be near 0 (perpendicular)
    dot = abs(float(np.dot(axis_a, axis_b)))
    assert dot < 0.1  # Perpendicular!

    # Segments representing these two walls
    seg_a = {
        "plane_id": 1,
        "normal": normal_a,
        "centroid": np.array([0.0, 0.0, 1.5]),
        "mins": np.array([-2.0, -2.0, 0.0]),
        "maxs": np.array([2.0, 2.0, 3.0]),
        "tags": {"wall_angle_deg": 45.8, "wall_axis_x": axis_a[0], "wall_axis_y": axis_a[1]},
    }
    seg_b = {
        "plane_id": 2,
        "normal": normal_b,
        "centroid": np.array([0.0, 0.0, 1.5]),
        "mins": np.array([-2.0, -2.0, 0.0]),
        "maxs": np.array([2.0, 2.0, 3.0]),
        "tags": {"wall_angle_deg": 135.8, "wall_axis_x": axis_b[0], "wall_axis_y": axis_b[1]},
    }

    groups = group_wall_fragments([seg_a, seg_b])
    assert len(groups) == 2  # Not merged together!


def test_i_multi_height_wall_fragments_merge():
    """Test I: Multi-height wall fragments from the same physical wall merge via Union-Find."""
    norm = np.array([1.0, 0.0, 0.0])
    axis = canonicalize_wall_axis(norm)

    # Lower fragment (Z: 0 to 1.8m)
    seg_lower = {
        "plane_id": 10,
        "normal": norm,
        "centroid": np.array([0.0, 2.0, 0.9]),
        "mins": np.array([-0.05, 0.0, 0.0]),
        "maxs": np.array([0.05, 4.0, 1.8]),
        "tags": {"wall_angle_deg": 90.0, "wall_axis_x": axis[0], "wall_axis_y": axis[1]},
    }
    # Upper fragment (Z: 1.5 to 3.0m) - overlapping Z by 0.3m
    seg_upper = {
        "plane_id": 11,
        "normal": norm,
        "centroid": np.array([0.0, 2.0, 2.25]),
        "mins": np.array([-0.05, 0.0, 1.5]),
        "maxs": np.array([0.05, 4.0, 3.0]),
        "tags": {"wall_angle_deg": 90.0, "wall_axis_x": axis[0], "wall_axis_y": axis[1]},
    }

    groups = group_wall_fragments([seg_lower, seg_upper])
    assert len(groups) == 1
    merged = groups[0]
    # Merged Z extent spans 0.0 to 3.0
    assert merged["mins"][2] <= 0.01
    assert merged["maxs"][2] >= 2.99


def test_j_stair_vs_ramp():
    """Test J: count_stair_steps and classifier distinguish stepped profile (STAIR) vs smooth slope (RAMP)."""
    # Stair points: 5 discrete steps along Z
    rng = np.random.RandomState(42)
    stair_pts = []
    for step_idx in range(5):
        z_val = step_idx * 0.18  # 180mm riser
        x_val = step_idx * 0.28  # 280mm tread
        x = rng.uniform(x_val, x_val + 0.28, 200)
        y = rng.uniform(0.0, 1.0, 200)
        z = np.full(200, z_val) + rng.normal(0.0, 0.005, 200)
        stair_pts.append(np.column_stack([x, y, z]))
    stair_pts = np.vstack(stair_pts)
    n_steps = count_stair_steps(stair_pts)
    assert n_steps >= 3

    # Segment with detected stair steps classifies as STAIR (coordinates in mm)
    stair_seg = GeometrySegment(
        zone_id="zone-001",
        confidence=0.95,
        shape=SegmentShape.PLANE_SLOPED,
        centroid=Point3D(x=1000.0, y=500.0, z=500.0),
        bounding_box=BoundingBox(min_x=0.0, min_y=0.0, min_z=0.0, max_x=2000.0, max_y=1000.0, max_z=1000.0),
        point_count=len(stair_pts),
        tags={"stair_steps": n_steps},
    )
    elem_stair, _, _ = classify_segment(stair_seg)
    assert elem_stair == ElementType.STAIR

    # Segment without stair steps on smooth/shallow slope classifies as RAMP
    ramp_seg = GeometrySegment(
        zone_id="zone-001",
        confidence=0.95,
        shape=SegmentShape.PLANE_SLOPED,
        centroid=Point3D(x=2500.0, y=500.0, z=250.0),
        bounding_box=BoundingBox(min_x=0.0, min_y=0.0, min_z=0.0, max_x=5000.0, max_y=1000.0, max_z=500.0),
        point_count=2000,
        tags={"stair_steps": 0},
    )
    elem_ramp, _, _ = classify_segment(ramp_seg)
    assert elem_ramp == ElementType.RAMP




def test_k_single_face_wall_thickness():
    """Test K: Single-face wall receives configured 200mm thickness bounded within [100, 600]."""
    norm = np.array([0.0, 1.0, 0.0])
    axis = canonicalize_wall_axis(norm)
    single_face = {
        "plane_id": 20,
        "normal": norm,
        "centroid": np.array([2.5, 0.0, 1.5]),
        "mins": np.array([0.0, -0.05, 0.0]),
        "maxs": np.array([5.0, 0.05, 3.0]),
        "tags": {"wall_angle_deg": 0.0, "wall_axis_x": axis[0], "wall_axis_y": axis[1]},
    }
    physical_walls = pair_opposing_wall_faces([single_face], max_thickness_m=0.45)
    assert len(physical_walls) == 1
    w = physical_walls[0]
    thick_mm = w["tags"]["wall_thickness_mm"]
    assert MIN_WALL_THICKNESS_MM <= thick_mm <= MAX_WALL_THICKNESS_MM
    assert thick_mm == DEFAULT_WALL_THICKNESS_MM
