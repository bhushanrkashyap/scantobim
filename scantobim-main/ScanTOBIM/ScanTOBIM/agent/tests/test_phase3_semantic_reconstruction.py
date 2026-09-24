"""Phase 3 Comprehensive Test Suite — Semantic Understanding & Architectural Reconstruction.

Mandatory 21 Test Cases:
 1. semantic adapter discovery
 2. checkpoint missing
 3. checkpoint invalid
 4. supported labels
 5. unsupported labels
 6. semantic output provenance
 7. semantic/geometric fusion
 8. instance extraction
 9. duplicate suppression
10. transformed input invariance
11. storey discovery
12. slab fitting
13. wall plane fitting
14. wall thickness estimation
15. wall local coordinate frame
16. wall fragment merging
17. corner handling
18. spatial NMS
19. column fitting
20. invalid geometry rejection
21. source support calculation
"""

from __future__ import annotations

import math
import numpy as np
import pytest

from agent.phase3.columns.reconstructor import (
    compute_oriented_cross_section,
    reconstruct_column_from_points,
)
from agent.phase3.fusion.engine import (
    compute_geometry_score,
    compute_source_support_score,
    fuse_semantic_and_geometric,
)
from agent.phase3.instance.adapter import InstanceMask
from agent.phase3.instance.deduplication import deduplicate_instances
from agent.phase3.instance.geometric_clustering import GeometricInstanceAdapter
from agent.phase3.pipeline import run_phase3_reconstruction
from agent.phase3.provenance.tracker import ProvenanceTracker
from agent.phase3.semantic.geometric_adapter import GeometricSemanticAdapter
from agent.phase3.semantic.kpconv_adapter import KPConvSemanticAdapter
from agent.phase3.semantic.ptv2_adapter import PTv2SemanticAdapter
from agent.phase3.semantic.randla_adapter import RandLANetSemanticAdapter
from agent.phase3.semantic.selector import SemanticModelSelector
from agent.phase3.slabs.reconstructor import (
    extract_2d_boundary_polygon,
    reconstruct_slab_from_points,
)
from agent.phase3.status import (
    ALGORITHMIC_ADAPTER,
    FAILED,
    GEOMETRIC_ADAPTER,
    REAL_NEURAL_INFERENCE,
    UNAVAILABLE,
)
from agent.phase3.storeys.detector import detect_storeys_from_point_cloud
from agent.phase3.taxonomy import (
    CANONICAL_LABELS,
    UNSUPPORTED_MODEL_LABEL,
    map_to_canonical,
)
from agent.phase3.validation.validator import (
    validate_column_geometry,
    validate_slab_geometry,
    validate_wall_geometry,
)
from agent.phase3.walls.reconstructor import (
    detect_wall_junctions,
    fit_vertical_plane,
    measure_wall_thickness,
    merge_collinear_wall_fragments,
    reconstruct_wall_from_points,
)
from agent.tools.coordinate_system import AuthoritativeTransform


# ── Synthetic Helpers ─────────────────────────────────────────────────────────

def make_synthetic_wall_points(
    length_m: float = 4.0,
    height_m: float = 3.0,
    thickness_m: float = 0.20,
    num_points: int = 500,
    x_offset: float = 0.0,
    y_offset: float = 0.0,
    z_offset: float = 0.0,
    angle_deg: float = 0.0,
) -> np.ndarray:
    """Generate two opposing parallel faces simulating a real physical wall."""
    np.random.seed(42)
    n_per_face = num_points // 2
    u = np.random.uniform(0, length_m, n_per_face)
    w = np.random.uniform(0, height_m, n_per_face)

    f1 = np.column_stack([u, np.full(n_per_face, -thickness_m / 2.0), w])
    f2 = np.column_stack([u, np.full(n_per_face, thickness_m / 2.0), w])
    pts = np.vstack([f1, f2])

    rad = np.radians(angle_deg)
    rot = np.array([[np.cos(rad), -np.sin(rad), 0], [np.sin(rad), np.cos(rad), 0], [0, 0, 1]])
    pts = np.dot(pts, rot.T)

    pts += np.array([x_offset, y_offset, z_offset])
    return pts


def make_synthetic_slab_points(
    width_m: float = 5.0,
    length_m: float = 5.0,
    thickness_m: float = 0.15,
    num_points: int = 400,
    elevation_m: float = 0.0,
) -> np.ndarray:
    """Generate points simulating a horizontal floor or ceiling slab."""
    np.random.seed(42)
    x = np.random.uniform(0, width_m, num_points)
    y = np.random.uniform(0, length_m, num_points)
    z = np.random.uniform(elevation_m, elevation_m + thickness_m, num_points)
    return np.column_stack([x, y, z])


def make_synthetic_column_points(
    width_m: float = 0.35,
    depth_m: float = 0.35,
    height_m: float = 3.0,
    num_points: int = 200,
    center_xy: tuple[float, float] = (2.0, 2.0),
    z_bottom: float = 0.0,
) -> np.ndarray:
    """Generate points around a structural column profile."""
    np.random.seed(42)
    x = np.random.uniform(-width_m / 2, width_m / 2, num_points) + center_xy[0]
    y = np.random.uniform(-depth_m / 2, depth_m / 2, num_points) + center_xy[1]
    z = np.random.uniform(z_bottom, z_bottom + height_m, num_points)
    return np.column_stack([x, y, z])


# ── Test Suite ────────────────────────────────────────────────────────────────

class TestPhase3SemanticReconstruction:
    """Mandatory test cases covering Phase 3 semantic understanding & reconstruction."""

    # 1. Semantic adapter discovery
    def test_semantic_adapter_discovery(self):
        selector = SemanticModelSelector()
        adapters = selector.adapters
        assert "ptv2" in adapters
        assert "randlanet" in adapters
        assert "kpconv" in adapters
        assert "geometric" in adapters

    # 2. Checkpoint missing
    def test_checkpoint_missing(self):
        adapter = PTv2SemanticAdapter(checkpoint_path="/nonexistent/ptv2.pt")
        loaded = adapter.load()
        assert loaded is False
        assert adapter.status == UNAVAILABLE
        assert adapter.health_check() is False

    # 3. Checkpoint invalid
    def test_checkpoint_invalid(self, tmp_path):
        bad_ckpt = tmp_path / "corrupted.pt"
        bad_ckpt.write_text("not a valid pytorch binary state dict")
        adapter = PTv2SemanticAdapter(checkpoint_path=str(bad_ckpt))
        loaded = adapter.load()
        assert loaded is False
        assert adapter.status in (FAILED, UNAVAILABLE)

    # 4. Supported labels
    def test_supported_labels(self):
        adapter = GeometricSemanticAdapter()
        labels = adapter.supported_labels()
        assert "WALL" in labels
        assert "FLOOR" in labels
        assert "CEILING" in labels
        assert "COLUMN" in labels

    # 5. Unsupported labels
    def test_unsupported_labels(self):
        assert map_to_canonical("alien_spaceship") == UNSUPPORTED_MODEL_LABEL
        assert map_to_canonical("unsupported_random_label_xyz") == UNSUPPORTED_MODEL_LABEL
        assert map_to_canonical("") == "UNKNOWN"

    # 6. Semantic output provenance
    def test_semantic_output_provenance(self):
        adapter = GeometricSemanticAdapter()
        pts = make_synthetic_wall_points(num_points=100)
        res = adapter.infer(pts)
        assert res.model_name == "GeometricTensorClassifier"
        assert res.status == GEOMETRIC_ADAPTER
        assert len(res.labels) == len(pts)
        assert len(res.probabilities) == len(pts)
        assert "diagnostics" in res.to_dict()

    # 7. Semantic/geometric fusion
    def test_semantic_geometric_fusion(self):
        pts = make_synthetic_wall_points(num_points=100)
        fusion = fuse_semantic_and_geometric(
            semantic_score=0.90,
            points=pts,
            expected_type="WALL",
            residual_rmse_m=0.01,
            has_valid_storey=True,
        )
        assert fusion.overall_confidence > 0.70
        assert fusion.planarity > 0.55
        assert fusion.verticality > 0.80
        assert fusion.source_support_score > 0.70

    # 8. Instance extraction
    def test_instance_extraction(self):
        wall_pts = make_synthetic_wall_points(num_points=100, x_offset=0.0)
        floor_pts = make_synthetic_slab_points(num_points=100)
        pts = np.vstack([wall_pts, floor_pts])
        labels = ["WALL"] * 100 + ["FLOOR"] * 100

        adapter = GeometricInstanceAdapter(cluster_distance_tol_m=0.5, min_cluster_points=20)
        res = adapter.segment(pts, labels)
        assert len(res.instances) >= 2
        classes = {inst.semantic_class for inst in res.instances}
        assert "WALL" in classes
        assert "FLOOR" in classes

    # 9. Duplicate suppression
    def test_duplicate_suppression(self):
        pts_idx = np.arange(100)
        inst1 = InstanceMask(
            instance_id="W1",
            semantic_class="WALL",
            point_indices=pts_idx,
            confidence=0.95,
            centroid_m=np.array([2.0, 0.0, 1.5]),
            bounding_box_min_m=np.array([0.0, -0.1, 0.0]),
            bounding_box_max_m=np.array([4.0, 0.1, 3.0]),
            point_count=100,
        )
        inst2 = InstanceMask(
            instance_id="W2",
            semantic_class="WALL",
            point_indices=pts_idx,
            confidence=0.80,
            centroid_m=np.array([2.01, 0.01, 1.5]),
            bounding_box_min_m=np.array([0.0, -0.1, 0.0]),
            bounding_box_max_m=np.array([4.0, 0.1, 3.0]),
            point_count=100,
        )
        dedup = deduplicate_instances([inst1, inst2])
        assert len(dedup.active_instances) == 1
        assert dedup.active_instances[0].instance_id == "W1"
        assert len(dedup.suppressed_candidates) == 1

    # 10. Transformed input invariance
    def test_transformed_input_invariance(self):
        w_pts = make_synthetic_wall_points(length_m=4.0, height_m=3.0, x_offset=0, y_offset=0, z_offset=0)
        f_pts = make_synthetic_slab_points(width_m=4.0, length_m=4.0, elevation_m=0.0)
        base_cloud = np.vstack([w_pts, f_pts])

        R = np.array([[0, -1, 0], [1, 0, 0], [0, 0, 1]])
        trans_cloud = np.dot(base_cloud, R.T) + np.array([50000.0, -30000.0, 120.0])

        s_base = detect_storeys_from_point_cloud(base_cloud)
        s_trans = detect_storeys_from_point_cloud(trans_cloud)

        assert len(s_base) == len(s_trans)
        assert abs(s_base[0].height_m - s_trans[0].height_m) < 0.25

    # 11. Storey discovery
    def test_storey_discovery(self):
        f1 = make_synthetic_slab_points(elevation_m=0.0, num_points=300)
        f2 = make_synthetic_slab_points(elevation_m=3.5, num_points=300)
        w = make_synthetic_wall_points(height_m=3.5, num_points=400)
        pts = np.vstack([f1, f2, w])

        storeys = detect_storeys_from_point_cloud(pts, min_floor_clearance_m=2.0)
        assert len(storeys) >= 2
        assert abs(storeys[0].elevation_m - 0.0) < 0.35
        assert abs(storeys[1].elevation_m - 3.5) < 0.35

    # 12. Slab fitting
    def test_slab_fitting(self):
        pts = make_synthetic_slab_points(width_m=4.0, length_m=4.0, thickness_m=0.20, num_points=300)
        slab = reconstruct_slab_from_points(pts, np.arange(len(pts)))
        assert slab is not None
        assert abs(slab.thickness_m - 0.20) < 0.06
        assert slab.area_m2 > 10.0
        assert len(slab.boundary_polygon_m) >= 3

    # 13. Wall plane fitting
    def test_wall_plane_fitting(self):
        pts = make_synthetic_wall_points(length_m=5.0, height_m=2.8, angle_deg=30.0)
        normal, d, rmse = fit_vertical_plane(pts)
        assert rmse < 0.15
        assert abs(normal[2]) < 1e-5

    # 14. Wall thickness estimation
    def test_wall_thickness_estimation(self):
        pts = make_synthetic_wall_points(thickness_m=0.25, num_points=500)
        normal, _, _ = fit_vertical_plane(pts)
        thick, method = measure_wall_thickness(pts, normal)
        assert abs(thick - 0.25) < 0.08
        assert "BIMODAL" in method or "PERCENTILE" in method or "GEOMETRIC" in method

    # 15. Wall local coordinate frame
    def test_wall_local_coordinate_frame(self):
        pts = make_synthetic_wall_points(length_m=6.0, height_m=3.0, angle_deg=45.0)
        wall = reconstruct_wall_from_points(pts, np.arange(len(pts)))
        assert wall is not None
        frame = wall.local_frame
        long_ax = np.array(frame.longitudinal_axis)
        trans_ax = np.array(frame.transverse_axis)
        norm_ax = np.array(frame.normal_axis)

        assert abs(np.linalg.norm(long_ax) - 1.0) < 1e-4
        assert abs(np.linalg.norm(trans_ax) - 1.0) < 1e-4
        assert abs(np.dot(long_ax, trans_ax)) < 1e-4

    # 16. Wall fragment merging
    def test_wall_fragment_merging(self):
        pts1 = make_synthetic_wall_points(length_m=2.5, x_offset=0.0)
        pts2 = make_synthetic_wall_points(length_m=3.0, x_offset=3.0)
        w1 = reconstruct_wall_from_points(pts1, np.arange(len(pts1)), wall_id="W1")
        w2 = reconstruct_wall_from_points(pts2, np.arange(len(pts2)), wall_id="W2")
        assert w1 is not None and w2 is not None

        merged = merge_collinear_wall_fragments([w1, w2], endpoint_gap_tol_m=1.0)
        assert len(merged) == 1
        assert merged[0].length_m >= 5.5

    # 17. Corner handling
    def test_corner_handling(self):
        pts1 = make_synthetic_wall_points(length_m=3.0, angle_deg=0.0, x_offset=0.0, y_offset=0.0)
        pts2 = make_synthetic_wall_points(length_m=3.0, angle_deg=90.0, x_offset=0.0, y_offset=0.0)
        w1 = reconstruct_wall_from_points(pts1, np.arange(len(pts1)), wall_id="W1")
        w2 = reconstruct_wall_from_points(pts2, np.arange(len(pts2)), wall_id="W2")
        assert w1 is not None and w2 is not None

        detect_wall_junctions([w1, w2], proximity_tol_m=0.5)
        assert any("L_JUNCTION" in j for j in w1.junction_connections)
        assert any("L_JUNCTION" in j for j in w2.junction_connections)

    # 18. Spatial NMS
    def test_spatial_nms(self):
        pts1 = np.arange(50)
        pts2 = np.arange(40)
        pts3 = np.arange(100, 150)
        i1 = InstanceMask("C1", "COLUMN", pts1, 0.95, np.array([1, 1, 0]), np.array([0.8, 0.8, 0]), np.array([1.2, 1.2, 3]), 50)
        i2 = InstanceMask("C2", "COLUMN", pts2, 0.80, np.array([1.02, 1.02, 0]), np.array([0.8, 0.8, 0]), np.array([1.2, 1.2, 3]), 40)
        i3 = InstanceMask("C3", "COLUMN", pts3, 0.90, np.array([5, 5, 0]), np.array([4.8, 4.8, 0]), np.array([5.2, 5.2, 3]), 50)

        res = deduplicate_instances([i1, i2, i3])
        assert len(res.active_instances) == 2
        active_ids = {inst.instance_id for inst in res.active_instances}
        assert "C1" in active_ids
        assert "C3" in active_ids
        assert "C2" not in active_ids

    # 19. Column fitting
    def test_column_fitting(self):
        pts = make_synthetic_column_points(width_m=0.40, depth_m=0.30, height_m=2.8, num_points=150)
        col = reconstruct_column_from_points(pts, np.arange(len(pts)))
        assert col is not None
        dim_max = max(col.width_m, col.depth_m)
        dim_min = min(col.width_m, col.depth_m)
        assert abs(dim_max - 0.40) < 0.12
        assert abs(dim_min - 0.30) < 0.12
        assert abs(col.height_m - 2.8) < 0.15

    # 20. Invalid geometry rejection
    def test_invalid_geometry_rejection(self):
        few_pts = np.array([[0, 0, 0], [1, 1, 1]])
        rep = validate_wall_geometry((0, 0, 0), (1, 0, 0), 1.0, 2.5, 0.20, few_pts)
        assert rep.is_valid is False
        assert rep.status == "REJECTED_SUPPORT"

        rep2 = validate_wall_geometry((0, 0, 0), (5, 0, 0), 5.0, 3.0, 5.0, make_synthetic_wall_points(num_points=100))
        assert rep2.is_valid is False
        assert rep2.status == "REJECTED_GEOMETRY"

    # 21. Source support calculation
    def test_source_support_calculation(self):
        score, breakdown = compute_source_support_score(
            point_count=350,
            spatial_coverage_m2=4.5,
            residual_rmse_m=0.008,
            min_required_points=50,
        )
        assert score > 0.85
        assert "density_factor" in breakdown
        assert "continuity_factor" in breakdown


def test_end_to_end_phase3_pipeline():
    """Verify complete Phase 3 pipeline execution on realistic building scene."""
    w1 = make_synthetic_wall_points(length_m=5.0, height_m=3.0, angle_deg=0.0, num_points=300)
    w2 = make_synthetic_wall_points(length_m=4.0, height_m=3.0, angle_deg=90.0, num_points=300)
    f = make_synthetic_slab_points(width_m=6.0, length_m=6.0, elevation_m=0.0, num_points=300)
    c = make_synthetic_column_points(center_xy=(2.5, 2.5), num_points=150)
    scene_points = np.vstack([w1, w2, f, c])

    res = run_phase3_reconstruction(
        points=scene_points,
        source_file="test_scene.pcd",
    )
    assert res.status in ("PHASE_3_PASS", "PHASE_3_PASS_WITH_CONFIG")
    assert res.storeys_count >= 1
    assert res.walls_count >= 1
    assert res.slabs_count >= 1
    assert len(res.accepted_candidates) >= 2

    for cand in res.accepted_candidates:
        d = cand.to_dict()
        assert "candidate_id" in d
        assert "geometry" in d
        assert "level" in d
        assert "source" in d
        assert "semantic_evidence" in d
        assert "geometric_evidence" in d
        assert "validation" in d
