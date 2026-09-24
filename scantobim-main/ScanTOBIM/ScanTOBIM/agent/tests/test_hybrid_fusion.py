"""Unit and integration tests for Hybrid AI + Geometric Fusion Engine.

Verifies:
1. PTv3 Semantic Adapter tensor signature classification and confidence scores.
2. YOLOv8 Column Candidate Generator + 2D ConvexHull MBR + Spatial Grid NMS.
3. GroundingDINO Door/Window Opening candidate detection and 3D back-projection.
4. A-Scan2BIM corner-edge candidate generation and wall candidate ranking.
5. HybridFusionEngine end-to-end fusion and audit report metrics.
"""

from __future__ import annotations

import numpy as np
import pytest

from agent.adapters.ascan2bim_wall_adapter import AScan2BimWallAdapter, WallCandidateProposal
from agent.adapters.fusion_engine import HybridFusionEngine, FusionAuditReport
from agent.adapters.grounding_dino_adapter import GroundingDinoAdapter, VerifiedOpening
from agent.adapters.ptv3_semantic_adapter import Ptv3SemanticAdapter
from agent.adapters.yolo_column_adapter import YoloColumnAdapter, VerifiedColumn
from agent.tools.slab_tools import StoreyDefinition, SlabEntity


def test_ptv3_semantic_adapter_vertical_wall():
    """Test PTv3 semantic adapter accurately classifies vertical planar point cloud as wall."""
    adapter = Ptv3SemanticAdapter()
    # Create synthetic vertical planar points (wall along X with small Y variance)
    xs = np.linspace(0, 5, 50)
    zs = np.linspace(0, 3, 30)
    X, Z = np.meshgrid(xs, zs)
    Y = np.zeros_like(X) + np.random.normal(0, 0.01, X.shape)
    pts = np.column_stack([X.ravel(), Y.ravel(), Z.ravel()])

    res = adapter.classify_segment("wall_01", pts, normal=np.array([0.0, 1.0, 0.0]), shape="plane_vertical")
    assert res.predicted_class == "wall"
    assert res.confidence >= 0.85
    assert "wall" in res.class_probabilities
    assert res.status == "executed"


def test_ptv3_semantic_adapter_horizontal_floor():
    """Test PTv3 semantic adapter accurately classifies horizontal planar points as floor."""
    adapter = Ptv3SemanticAdapter()
    xs = np.linspace(-5, 5, 50)
    ys = np.linspace(-5, 5, 50)
    X, Y = np.meshgrid(xs, ys)
    Z = np.zeros_like(X) + 0.1
    pts = np.column_stack([X.ravel(), Y.ravel(), Z.ravel()])

    res = adapter.classify_segment("floor_01", pts, normal=np.array([0.0, 0.0, 1.0]), shape="plane_horizontal")
    assert res.predicted_class == "floor"
    assert res.confidence >= 0.80


def test_yolo_column_adapter_candidate_generation_and_nms():
    """Test YOLOv8 column candidate generation and spatial NMS deduplication."""
    adapter = YoloColumnAdapter(conf_thresh=0.25)

    # Synthetic storey
    storey = StoreyDefinition(
        storey_id="storey_0",
        name="Ground Floor",
        index=0,
        elevation_m=0.0,
        top_elevation_m=3.5,
        height_m=3.5,
        floor_slab=None,
    )

    # Create two synthetic columns: one at (2, 2) and one duplicate at (2.2, 2.1)
    pts_col1 = np.random.uniform(low=[-0.2, -0.2, 0.2], high=[0.2, 0.2, 3.3], size=(200, 3)) + np.array([2.0, 2.0, 0.0])
    pts_col2 = np.random.uniform(low=[-0.2, -0.2, 0.2], high=[0.2, 0.2, 3.3], size=(180, 3)) + np.array([2.2, 2.1, 0.0])
    # Distant column at (8, 8)
    pts_col3 = np.random.uniform(low=[-0.25, -0.25, 0.2], high=[0.25, 0.25, 3.3], size=(250, 3)) + np.array([8.0, 8.0, 0.0])

    all_pts = np.vstack([pts_col1, pts_col2, pts_col3])

    verified_cols, stats = adapter.detect_columns(all_pts, [storey], min_inliers_per_col=30, nms_radius_m=1.0)

    assert stats["dl_candidates_generated"] >= 2
    # Spatial NMS must merge duplicate cluster within 1.0m radius
    assert stats["fused_by_nms"] >= 1
    # Only 2 distinct physical columns should remain
    assert len(verified_cols) == 2
    for c in verified_cols:
        assert c.height_m == pytest.approx(3.5, rel=1e-2)
        assert 0.15 <= c.width_m <= 1.20
        assert 0.15 <= c.depth_m <= 1.20


def test_grounding_dino_opening_detection():
    """Test GroundingDINO adapter identifies void/opening along wall centerline."""
    adapter = GroundingDinoAdapter()

    # Synthetic wall from (0, 0) to (6, 0), height 3.0m
    wall_seg = {
        "segment_id": "wall_test_01",
        "tags": {
            "storey_id": "storey_0",
            "wall_start_x_mm": 0.0,
            "wall_start_y_mm": 0.0,
            "wall_end_x_mm": 6000.0,
            "wall_end_y_mm": 0.0,
            "wall_thickness_mm": 200.0,
        },
        "bounding_box": {
            "min_x": 0.0,
            "max_x": 6000.0,
            "min_y": -100.0,
            "max_y": 100.0,
            "min_z": 0.0,
            "max_z": 3000.0,
        },
    }

    # Points on wall except a window opening at u in [2.0, 3.2], z in [1.0, 2.2]
    pts = []
    for u in np.linspace(0.1, 5.9, 60):
        for z in np.linspace(0.1, 2.9, 30):
            if 2.0 <= u <= 3.2 and 1.0 <= z <= 2.2:
                continue  # Void representing opening
            pts.append([u, 0.0, z])
    points_xyz = np.array(pts)

    openings = adapter.detect_openings_on_wall(wall_seg, points_xyz, min_width_m=0.8, min_height_m=0.8)
    assert len(openings) >= 1
    op = openings[0]
    assert op.host_wall_id == "wall_test_01"
    assert op.type in ("WINDOW", "DOOR")
    assert op.width_m >= 0.7
    assert op.height_m >= 0.7


def test_ascan2bim_wall_candidate_reasoning():
    """Test A-Scan2BIM corner-edge reasoning proposes ranked wall candidate proposals."""
    adapter = AScan2BimWallAdapter(min_wall_length_m=1.0, raster_res_m=0.10)

    # Synthetic L-shaped wall: (0, 0) -> (4, 0) and (0, 0) -> (0, 4)
    pts_w1 = np.column_stack([np.linspace(0, 4, 150), np.zeros(150), np.random.uniform(0.5, 2.5, 150)])
    pts_w2 = np.column_stack([np.zeros(150), np.linspace(0, 4, 150), np.random.uniform(0.5, 2.5, 150)])
    points = np.vstack([pts_w1, pts_w2])

    proposals = adapter.generate_wall_candidates(points, "storey_0", base_z_m=0.0, top_z_m=3.0)
    assert len(proposals) >= 1
    prop = proposals[0]
    assert isinstance(prop, WallCandidateProposal)
    assert prop.inlier_support >= 50
    assert prop.source == "ascan2bim_corner_edge"


def test_hybrid_fusion_engine_end_to_end():
    """Test HybridFusionEngine fuses geometric and DL candidates into BIM elements."""
    engine = HybridFusionEngine()

    # Synthetic floor points and wall points
    xs = np.linspace(0, 6, 40)
    ys = np.linspace(0, 6, 40)
    X, Y = np.meshgrid(xs, ys)
    floor_pts = np.column_stack([X.ravel(), Y.ravel(), np.zeros_like(X.ravel())])

    # Wall points
    wall_pts = np.column_stack([
        np.linspace(0, 6, 100),
        np.zeros(100),
        np.random.uniform(0.1, 2.9, 100),
    ])

    all_pts = np.vstack([floor_pts, wall_pts])

    raw_wall = {
        "segment_id": "wall_main_01",
        "element_type": "WALL",
        "shape": "plane_vertical",
        "confidence": 0.92,
        "point_count": len(wall_pts),
        "centroid": [3000.0, 0.0, 1500.0],
        "bounding_box": {
            "min_x": 0.0,
            "max_x": 6000.0,
            "min_y": -100.0,
            "max_y": 100.0,
            "min_z": 0.0,
            "max_z": 3000.0,
        },
        "tags": {
            "wall_start_x_mm": 0.0,
            "wall_start_y_mm": 0.0,
            "wall_end_x_mm": 6000.0,
            "wall_end_y_mm": 0.0,
            "wall_thickness_mm": 200.0,
        },
        "inlier_points": (wall_pts * 1000.0).tolist(),
    }

    fused_segments, storeys, audit = engine.fuse_and_verify(
        raw_segments=[raw_wall],
        downsampled_points_m=all_pts,
        zone_id="TEST_ZONE",
    )

    assert isinstance(audit, FusionAuditReport)
    assert audit.fused_elements_count >= 1
    assert len(audit.models_executed) == 4
    model_names = [m["model"] for m in audit.models_executed]
    assert "PTV3_PPT" in model_names
    assert "YOLOv8" in model_names
    assert "GroundingDINO" in model_names
    assert "A_Scan2BIM" in model_names
