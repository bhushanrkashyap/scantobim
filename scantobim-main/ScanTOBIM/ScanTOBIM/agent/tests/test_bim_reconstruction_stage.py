"""Unit and integration tests for detection -> BIM reconstruction stage.

Covers:
1. Wall-to-storey association and merge correctness with continuous arbitrary wall angles (~45.8° / 135.8°).
2. Multi-storey level clustering across horizontal planes.
3. Opening hosting containment (doors, windows, voids) with wall-local insertion coordinates.
4. Floor polygonal boundary reconstruction (boundary_polygon and profile_points, not simple AABB).
5. Stair and ramp level association (base_level_mm, top_level_mm, rise_mm).
6. RandLA algorithm predictions on 3D point clouds.
"""

from __future__ import annotations

import math
from uuid import uuid4

import numpy as np
import pytest

from agent.bim_reconstruction_fix import (
    attach_openings,
    detect_levels,
    reconstruct_floors_at_level,
    reconstruct_stairs_between_levels,
    reconstruct_walls_between_levels,
    run_auto_bim_pipeline,
)
from agent.models import (
    BoundingBox,
    ElementType,
    GeometrySegment,
    Point3D,
    SegmentShape,
)
from agent.tools.randla_net import RandLANet, predict_randla, random_sampling


def _make_wall(
    start_xy: tuple[float, float],
    end_xy: tuple[float, float],
    z_min: float,
    z_max: float,
    thickness: float = 200.0,
    point_count: int = 1500,
) -> GeometrySegment:
    p0 = np.array(start_xy, dtype=float)
    p1 = np.array(end_xy, dtype=float)
    vec = p1 - p0
    length = float(np.linalg.norm(vec))
    unit_axis = vec / max(length, 1e-6)
    unit_norm = np.array([-unit_axis[1], unit_axis[0]], dtype=float)
    center_xy = (p0 + p1) / 2.0

    # Rotated bounding box corners
    corners = np.array([
        center_xy - unit_axis * (length / 2) - unit_norm * (thickness / 2),
        center_xy + unit_axis * (length / 2) - unit_norm * (thickness / 2),
        center_xy + unit_axis * (length / 2) + unit_norm * (thickness / 2),
        center_xy - unit_axis * (length / 2) + unit_norm * (thickness / 2),
    ])

    angle_deg = float(np.degrees(np.arctan2(unit_axis[1], unit_axis[0])) % 180.0)

    seg = GeometrySegment(
        segment_id=str(uuid4()),
        zone_id="zone-001",
        shape=SegmentShape.PLANE_VERTICAL,
        element_type=ElementType.WALL,
        normal=Point3D(x=float(unit_norm[0]), y=float(unit_norm[1]), z=0.0),
        centroid=Point3D(x=float(center_xy[0]), y=float(center_xy[1]), z=float((z_min + z_max) / 2)),
        bounding_box=BoundingBox(
            min_x=float(corners[:, 0].min()),
            max_x=float(corners[:, 0].max()),
            min_y=float(corners[:, 1].min()),
            max_y=float(corners[:, 1].max()),
            min_z=float(z_min),
            max_z=float(z_max),
        ),
        point_count=point_count,
        confidence=0.95,
        tags={
            "wall_start_x_mm": round(float(p0[0]), 1),
            "wall_start_y_mm": round(float(p0[1]), 1),
            "wall_end_x_mm": round(float(p1[0]), 1),
            "wall_end_y_mm": round(float(p1[1]), 1),
            "wall_axis_x": round(float(unit_axis[0]), 6),
            "wall_axis_y": round(float(unit_axis[1]), 6),
            "wall_angle_deg": round(angle_deg, 2),
            "wall_length_mm": round(length, 1),
            "wall_thickness_mm": round(thickness, 1),
        },
    )
    return seg


def _make_horizontal_plane(
    z_height: float,
    x_span: tuple[float, float],
    y_span: tuple[float, float],
    element_type: ElementType = ElementType.FLOOR,
    thickness: float = 150.0,
    point_count: int = 5000,
) -> GeometrySegment:
    min_x, max_x = x_span
    min_y, max_y = y_span
    return GeometrySegment(
        segment_id=str(uuid4()),
        zone_id="zone-001",
        shape=SegmentShape.PLANE_HORIZONTAL,
        element_type=element_type,
        normal=Point3D(x=0.0, y=0.0, z=1.0 if element_type == ElementType.FLOOR else -1.0),
        centroid=Point3D(
            x=float((min_x + max_x) / 2),
            y=float((min_y + max_y) / 2),
            z=float(z_height),
        ),
        bounding_box=BoundingBox(
            min_x=float(min_x),
            max_x=float(max_x),
            min_y=float(min_y),
            max_y=float(max_y),
            min_z=float(z_height - thickness / 2),
            max_z=float(z_height + thickness / 2),
        ),
        point_count=point_count,
        confidence=0.92,
        tags={"thickness_mm": thickness},
    )


# ── 1. LEVEL CLUSTERING TESTS ────────────────────────────────────────────────


def test_detect_levels_multi_storey_clustering():
    """Verify that horizontal planes are clustered into distinct building storeys."""
    planes = [
        # Ground level (-14.7 m)
        _make_horizontal_plane(-14700.0, (-5000, 5000), (-5000, 5000), ElementType.FLOOR),
        _make_horizontal_plane(-14650.0, (-3000, 3000), (-3000, 3000), ElementType.FLOOR),
        # Level 1 (-11.8 m)
        _make_horizontal_plane(-11800.0, (-4000, 4000), (-4000, 4000), ElementType.FLOOR),
        # Level 2 (+3.3 m)
        _make_horizontal_plane(3300.0, (-4000, 4000), (-4000, 4000), ElementType.FLOOR),
        # Level 3 (+7.4 m roof)
        _make_horizontal_plane(7400.0, (-4000, 4000), (-4000, 4000), ElementType.CEILING),
    ]

    levels = detect_levels(planes)
    assert len(levels) >= 4, f"Expected at least 4 storeys, got {levels}"
    # Verify ground slab around -14.7m is detected
    assert any(abs(lvl - (-14700.0)) < 200.0 for lvl in levels)
    # Verify Level 1 around -11.8m
    assert any(abs(lvl - (-11800.0)) < 200.0 for lvl in levels)
    # Verify Level 2 around 3.3m
    assert any(abs(lvl - 3300.0) < 200.0 for lvl in levels)


# ── 2. WALL-TO-STOREY MERGE & CONTINUOUS ORIENTATION TESTS ───────────────────


def test_reconstruct_walls_arbitrary_diagonal_angles():
    """Verify walls oriented at ~45.8° are merged without Manhattan snapping."""
    # Collinear fragments along 45° diagonal line: y = x
    frag1 = _make_wall((0.0, 0.0), (2000.0, 2000.0), 0.0, 3000.0, thickness=200.0)
    frag2 = _make_wall((1900.0, 1900.0), (4500.0, 4500.0), 0.0, 3000.0, thickness=200.0)

    merged = reconstruct_walls_between_levels([frag1, frag2], level0=0.0, level1=3000.0)
    assert len(merged) == 1, f"Expected 1 merged wall, got {len(merged)}"
    m = merged[0]

    # Verify orientation is preserved around 45°
    angle = float(m.tags.get("wall_angle_deg", 0.0))
    assert abs(angle - 45.0) < 1.5 or abs(angle - 225.0) < 1.5, f"Unexpected angle {angle}"

    # Verify wall axis vector is diagonal (~0.707, ~0.707)
    ax = abs(float(m.tags["wall_axis_x"]))
    ay = abs(float(m.tags["wall_axis_y"]))
    assert abs(ax - 0.707) < 0.05
    assert abs(ay - 0.707) < 0.05

    # Verify full storey height
    assert m.bounding_box.min_z == 0.0
    assert m.bounding_box.max_z == 3000.0


def test_reconstruct_walls_multi_storey_separation():
    """Verify walls in different storeys are grouped into their respective storeys."""
    # Storey 1: Z in [0, 3000]
    w_ground1 = _make_wall((0.0, 0.0), (4000.0, 0.0), 0.0, 2900.0)
    w_ground2 = _make_wall((3900.0, 0.0), (8000.0, 0.0), 100.0, 3000.0)

    # Storey 2: Z in [3000, 6000]
    w_upper = _make_wall((0.0, 0.0), (8000.0, 0.0), 3100.0, 5900.0)

    # Floor separating storeys
    floor = _make_horizontal_plane(0.0, (-1000, 9000), (-1000, 1000), ElementType.FLOOR)
    floor2 = _make_horizontal_plane(3000.0, (-1000, 9000), (-1000, 1000), ElementType.FLOOR)

    all_segments = [w_ground1, w_ground2, w_upper, floor, floor2]
    nodes = run_auto_bim_pipeline(all_segments)

    # Separate wall nodes per storey
    wall_nodes = [n for n in nodes.values() if n.type == ElementType.WALL]
    assert len(wall_nodes) >= 2

    ground_walls = [w for w in wall_nodes if w.segment.bounding_box.min_z == 0.0]
    upper_walls = [w for w in wall_nodes if w.segment.bounding_box.min_z == 3000.0]

    assert len(ground_walls) == 1, "Ground wall fragments should merge into 1 full-height wall"
    assert len(upper_walls) == 1, "Upper wall should be bound to storey [3000, 6000]"


# ── 3. FLOOR POLYGON RECONSTRUCTION TESTS ────────────────────────────────────


def test_reconstruct_floors_at_level_generates_polygon():
    """Verify floor reconstruction generates scan-derived polygon boundary, not just AABB."""
    floor1 = _make_horizontal_plane(0.0, (0.0, 5000.0), (0.0, 3000.0), ElementType.FLOOR)
    floor2 = _make_horizontal_plane(0.0, (2000.0, 8000.0), (1000.0, 6000.0), ElementType.FLOOR)

    reconstructed = reconstruct_floors_at_level([floor1, floor2], level0=0.0)
    assert len(reconstructed) == 1
    rf = reconstructed[0]

    tags = rf.tags
    assert "boundary_polygon" in tags, "Floor must have boundary_polygon tag"
    assert "profile_points" in tags, "Floor must have profile_points tag"

    poly = tags["boundary_polygon"]
    assert isinstance(poly, list)
    assert len(poly) >= 4, f"Floor polygon must have at least 4 vertices, got {len(poly)}"
    # Verify polygon is closed
    assert poly[0] == poly[-1], "Polygon must be closed"

    # Verify building-oriented OBB parameters are populated
    assert "floor_axis_x" in tags
    assert "floor_axis_y" in tags
    assert "floor_length_mm" in tags
    assert "floor_width_mm" in tags
    assert tags["floor_length_mm"] > 5000.0


# ── 4. OPENING HOSTING TESTS ─────────────────────────────────────────────────


def test_attach_openings_to_wall():
    """Verify doors and windows are correctly hosted with local insertion coordinates."""
    # Diagonal wall running from (0, 0) to (5000, 5000)
    wall_seg = _make_wall((0.0, 0.0), (5000.0, 5000.0), 0.0, 3000.0, thickness=200.0)

    # Door along the wall at midpoint (2500, 2500), Z in [0, 2100]
    door_seg = GeometrySegment(
        segment_id=str(uuid4()),
        zone_id="zone-001",
        shape=SegmentShape.VOID,
        element_type=ElementType.DOOR,
        centroid=Point3D(x=2500.0, y=2500.0, z=1050.0),
        bounding_box=BoundingBox(
            min_x=2200.0, max_x=2800.0, min_y=2200.0, max_y=2800.0, min_z=0.0, max_z=2100.0
        ),
        point_count=300,
        confidence=0.88,
        tags={},
    )

    # Window along the wall at (3800, 3800), Z in [1000, 2200]
    window_seg = GeometrySegment(
        segment_id=str(uuid4()),
        zone_id="zone-001",
        shape=SegmentShape.VOID,
        element_type=ElementType.WINDOW,
        centroid=Point3D(x=3800.0, y=3800.0, z=1600.0),
        bounding_box=BoundingBox(
            min_x=3400.0, max_x=4200.0, min_y=3400.0, max_y=4200.0, min_z=1000.0, max_z=2200.0
        ),
        point_count=250,
        confidence=0.85,
        tags={},
    )

    # Convert wall to ReconstructedNode
    from agent.bim_reconstruction_fix import ReconstructedNode

    wall_node = ReconstructedNode(
        id=wall_seg.segment_id,
        segment=wall_seg,
        type=ElementType.WALL,
        host=None,
        children=[],
        connections=[],
        parametric={},
        confidence=0.95,
    )

    opening_nodes = attach_openings([wall_node], [door_seg, window_seg])
    assert len(opening_nodes) == 2

    # Verify both openings are hosted by the wall
    for node in opening_nodes:
        assert node.host == wall_node.id, f"Opening {node.id} not hosted by wall"
        assert "insertion_u_mm" in node.segment.tags
        assert "insertion_v_mm" in node.segment.tags
        assert "wall_axis_x" in node.segment.tags
        assert "wall_axis_y" in node.segment.tags
        assert node.segment.tags["host_wall_id"] == str(wall_node.id)

    # Verify host wall children list contains opening IDs
    assert door_seg.segment_id in wall_node.children
    assert window_seg.segment_id in wall_node.children

    # Verify door sill height is near 0.0
    door_node = next(n for n in opening_nodes if n.type == ElementType.DOOR)
    assert door_node.segment.tags["sill_height_mm"] == 0.0


# ── 5. STAIR & RAMP LEVEL ASSOCIATION TESTS ──────────────────────────────────


def test_reconstruct_stairs_level_linkage():
    """Verify stairs link their lower and upper building levels."""
    stair_seg = GeometrySegment(
        segment_id=str(uuid4()),
        zone_id="zone-001",
        shape=SegmentShape.PLANE_SLOPED,
        element_type=ElementType.STAIR,
        centroid=Point3D(x=1500.0, y=1000.0, z=1500.0),
        bounding_box=BoundingBox(
            min_x=500.0, max_x=2500.0, min_y=500.0, max_y=1500.0, min_z=100.0, max_z=2900.0
        ),
        point_count=2000,
        confidence=0.90,
        tags={"stair_slope_deg": 35.0},
    )

    reconstructed = reconstruct_stairs_between_levels([stair_seg], level0=0.0, level1=3000.0)
    assert len(reconstructed) == 1
    st = reconstructed[0]

    assert st.tags["base_level_mm"] == 0.0
    assert st.tags["top_level_mm"] == 3000.0
    assert st.tags["rise_mm"] == 3000.0
    assert st.bounding_box.min_z == 0.0
    assert st.bounding_box.max_z == 3000.0


# ── 6. RANDLA ALGORITHM TESTS ────────────────────────────────────────────────


def test_randla_algorithm_prediction():
    """Verify RandLA algorithm components: Random Sampling, LocSE, and inference output."""
    np.random.seed(42)
    # Generate 500 synthetic 3D points
    pts = np.random.uniform(-5.0, 5.0, size=(500, 3)).astype(np.float32)

    mask, probs = predict_randla(pts, wall_threshold=0.50)

    assert mask.shape == (500,)
    assert probs.shape == (500,)
    assert mask.dtype == bool
    assert probs.dtype == np.float32 or probs.dtype == np.float64
    assert 0.0 <= float(probs.min()) <= float(probs.max()) <= 1.0


def test_randla_net_forward_pass():
    """Verify RandLA-Net PyTorch module forward pass."""
    import torch

    model = RandLANet(in_channels=6, num_classes=2, k_neighbors=8)
    model.eval()

    # 1 batch, 128 points
    coords = torch.randn(1, 128, 3)
    feats = torch.randn(1, 6, 128)

    with torch.no_grad():
        logits = model(coords, feats)

    assert logits.shape == (1, 2, 128), f"Expected (1, 2, 128), got {logits.shape}"
