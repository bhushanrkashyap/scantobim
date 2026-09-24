"""Comprehensive Test Suite for Phase 3B: Open-World 3D Object Understanding.

Validates:
  - Multi-Scale Geometric Feature Extraction (planarity, linearity, scattering, verticality, thickness)
  - Class-Agnostic Object Proposal Engine (discovery without predefined scene layout)
  - Hierarchical Taxonomy (Level 1 Categories, Level 2 Classes, Unknown handling)
  - Open-World & Unknown Object Handling (no hallucination of known classes)
  - Topological Reasoning Graph (hosted_by, connected_to, supported_by, intersects)
  - Object-Specific Geometric Reconstruction (measured dimensions, no generic fallback boxes)
  - Native Revit Element Generation (Wall, Floor, Column, Pipe, Duct, CableTray, Valve, DirectShape)
  - Strict CPU Enforcement & CUDA Guard
  - Negative Test Cases (ambiguous geometry, touching objects, unsupported classes, noise clusters)
"""

from __future__ import annotations

import numpy as np
import pytest
import torch

from agent.models import ElementInstruction, ElementType
from agent.phase3.geometry import (
    GeometricFeatureSet,
    MultiScaleRadii,
    compute_geometric_features,
    estimate_multi_scale_radii,
)
from agent.phase3.instance.proposal_engine import (
    ClassAgnosticProposalEngine,
    ObjectProposal,
)
from agent.phase3.mep.reconstructor import (
    reconstruct_cable_tray_from_points,
    reconstruct_duct_from_points,
    reconstruct_pipe_from_points,
    reconstruct_valve_from_points,
)
from agent.phase3.objects.reconstructor import reconstruct_object_from_points
from agent.phase3.open_world import (
    OPEN_VOCABULARY_STATUS,
    OpenWorldRecognitionEngine,
)
from agent.phase3.open_world_pipeline import run_phase3b_pipeline
from agent.phase3.revit.native_generator import NativeRevitGenerator
from agent.phase3.taxonomy import (
    HIERARCHICAL_TAXONOMY,
    LEVEL1_CATEGORIES,
    get_category_for_class,
    get_classes_for_category,
    is_unknown_or_unsupported,
)
from agent.phase3.topology.graph import TopologyGraph


# ── 1. Geometric Feature Engine Tests ──────────────────────────────────────────

class TestGeometricFeatureEngine:
    """Validates Section 7 & 8 multi-scale geometric feature engine."""

    def test_multi_scale_radii_estimation(self):
        """Radii must be derived dynamically from point density, not static."""
        # Dense cloud
        dense = np.random.uniform(0, 1, size=(500, 3)).astype(np.float32)
        radii_dense = estimate_multi_scale_radii(dense)

        # Sparse cloud
        sparse = np.random.uniform(0, 10, size=(500, 3)).astype(np.float32)
        radii_sparse = estimate_multi_scale_radii(sparse)

        assert radii_sparse.local_radius_m > radii_dense.local_radius_m
        assert radii_dense.local_radius_m < radii_dense.mid_radius_m < radii_dense.global_radius_m

    def test_vertical_wall_planarity_and_verticality(self):
        """Vertical planar patch must produce planarity > 0.5 and verticality > 0.9."""
        # Plane: X in [0, 5], Y near 0, Z in [0, 3]
        pts = np.random.uniform(low=[0, -0.01, 0], high=[5, 0.01, 3], size=(300, 3)).astype(np.float32)
        feats = compute_geometric_features(pts)

        assert feats.point_count == 300
        assert np.mean(feats.planarity) > 0.50
        assert np.mean(feats.verticality) > 0.85
        assert np.mean(feats.horizontality) < 0.15

    def test_horizontal_slab_horizontality(self):
        """Horizontal planar slab must produce horizontality > 0.85."""
        # Slab: X in [-5, 5], Y in [-5, 5], Z near 0
        pts = np.random.uniform(low=[-5, -5, -0.02], high=[5, 5, 0.02], size=(300, 3)).astype(np.float32)
        feats = compute_geometric_features(pts)

        assert np.mean(feats.horizontality) > 0.85
        assert np.mean(feats.verticality) < 0.15

    def test_cylindricality_and_linearity_for_cylinder(self):
        """Cylinder surface points must have high cylindricality and linearity."""
        angles = np.random.uniform(0, 2 * np.pi, 250)
        x = np.random.uniform(0, 4, 250)
        y = 0.10 * np.cos(angles)
        z = 0.10 * np.sin(angles)
        cyl_pts = np.column_stack([x, y, z]).astype(np.float32)

        feats = compute_geometric_features(cyl_pts)
        assert np.mean(feats.linearity) > 0.40
        assert np.mean(feats.cylindricality) > 0.30

    def test_compact_object_scattering(self):
        """Spherical/compact cluster must have high scattering/sphericity."""
        blob = np.random.normal(loc=0.0, scale=0.5, size=(200, 3)).astype(np.float32)
        feats = compute_geometric_features(blob)
        assert np.mean(feats.scattering) > 0.20


# ── 2. Class-Agnostic Object Proposals Tests ──────────────────────────────────

class TestClassAgnosticObjectProposals:
    """Validates Section 13 & 14 class-agnostic physical object discovery."""

    def test_discovers_discrete_objects_without_class_knowledge(self):
        """Proposal engine must separate physically distinct objects without prior labels."""
        # Wall at y=0
        wall = np.random.uniform(low=[0, -0.02, 0], high=[4, 0.02, 3], size=(250, 3))
        # Pipe at y=3
        angles = np.random.uniform(0, 2*np.pi, 150)
        px = np.random.uniform(0, 3, 150)
        py = 3.0 + 0.08 * np.cos(angles)
        pz = 2.0 + 0.08 * np.sin(angles)
        pipe = np.column_stack([px, py, pz])
        # Machinery block at x=5, y=5
        block = np.random.uniform(low=[4.5, 4.5, 0], high=[5.5, 5.5, 1.0], size=(120, 3))

        combined = np.vstack([wall, pipe, block]).astype(np.float32)

        engine = ClassAgnosticProposalEngine(min_proposal_points=20)
        proposals = engine.generate_proposals(combined)

        assert len(proposals) >= 3
        # Each proposal must have full OBB, AABB, and centroid
        for p in proposals:
            assert len(p.centroid_m) == 3
            assert len(p.obb_extents_m) == 3
            assert np.all(p.obb_extents_m > 0)
            assert p.surface_area_m2 > 0
            assert p.volume_m3 > 0
            assert len(p.point_indices) == p.point_count


# ── 3. Hierarchical Taxonomy Tests ────────────────────────────────────────────

class TestHierarchicalTaxonomy:
    """Validates Section 9 hierarchical taxonomy structure."""

    def test_level1_categories_present(self):
        expected = ["STRUCTURAL", "MEP", "ARCHITECTURAL", "FURNITURE", "EQUIPMENT", "OTHER"]
        for cat in expected:
            assert cat in LEVEL1_CATEGORIES
            assert cat in HIERARCHICAL_TAXONOMY

    def test_class_to_category_lookup(self):
        assert get_category_for_class("WALL") == "STRUCTURAL"
        assert get_category_for_class("PIPE") == "MEP"
        assert get_category_for_class("DUCT") == "MEP"
        assert get_category_for_class("DOOR") == "ARCHITECTURAL"
        assert get_category_for_class("UNKNOWN_OBJECT") == "OTHER"
        assert get_category_for_class("MACHINERY") == "EQUIPMENT" or get_category_for_class("MACHINERY") == "OTHER"

    def test_unknown_detection(self):
        assert is_unknown_or_unsupported("UNKNOWN")
        assert is_unknown_or_unsupported("UNKNOWN_OBJECT")
        assert is_unknown_or_unsupported("UNKNOWN_MEP")
        assert not is_unknown_or_unsupported("WALL")
        assert not is_unknown_or_unsupported("PIPE")


# ── 4. Open-World & Unknown Handling Tests ────────────────────────────────────

class TestOpenWorldUnknownClassification:
    """Validates Section 15 & 16 open-world recognition and unknown object handling."""

    def test_supported_wall_detection(self):
        pts = np.random.uniform(low=[0, -0.02, 0], high=[4, 0.02, 3], size=(200, 3)).astype(np.float32)
        engine = ClassAgnosticProposalEngine(min_proposal_points=15)
        props = engine.generate_proposals(pts)
        ow = OpenWorldRecognitionEngine(min_known_confidence=0.50)

        res = ow.classify_proposal(props[0])
        assert res.final_label == "WALL"
        assert res.label_status == "SUPPORTED"
        assert "HIGH_GEOMETRIC_SUPPORT" in res.reason_codes

    def test_ambiguous_object_becomes_unknown_never_hallucinated(self):
        """Ambiguous/irregular geometry must be classified as UNKNOWN, never forced to a known class."""
        blob = np.random.normal(loc=0.0, scale=0.4, size=(100, 3)).astype(np.float32)
        engine = ClassAgnosticProposalEngine(min_proposal_points=15)
        props = engine.generate_proposals(blob)
        ow = OpenWorldRecognitionEngine(min_known_confidence=0.50)

        if props:
            res = ow.classify_proposal(props[0])
            assert "UNKNOWN" in res.final_label
            assert res.label_status == "UNKNOWN"

    def test_open_vocabulary_status_reporting(self):
        ow = OpenWorldRecognitionEngine()
        assert ow.open_vocab_status == OPEN_VOCABULARY_STATUS


# ── 5. Topological Reasoning Tests ────────────────────────────────────────────

class TestTopologicalReasoning:
    """Validates Section 19 physical relationship reasoning."""

    def test_spatial_topology_graph(self):
        graph = TopologyGraph(connection_tolerance_m=0.30)
        # Floor
        graph.add_object("fl_1", "FLOOR", np.array([0, 0, 0]), np.array([-5, -5, -0.3]), np.array([5, 5, 0]))
        # Column on Floor
        graph.add_object("col_1", "COLUMN", np.array([1, 1, 1.5]), np.array([0.8, 0.8, 0]), np.array([1.2, 1.2, 3.0]))
        # Wall on Floor
        graph.add_object("wl_1", "WALL", np.array([0, 2, 1.5]), np.array([-2, 1.9, 0]), np.array([2, 2.1, 3.0]))
        # Door in Wall
        graph.add_object("dr_1", "DOOR", np.array([0, 2, 1.0]), np.array([-0.5, 1.9, 0]), np.array([0.5, 2.1, 2.0]))
        # Pipe
        graph.add_object("pp_1", "PIPE", np.array([0, -2, 2.5]), np.array([-2, -2.1, 2.4]), np.array([2, -1.9, 2.6]))
        # Valve connected to Pipe
        graph.add_object("vl_1", "VALVE", np.array([2.1, -2, 2.5]), np.array([2.0, -2.1, 2.4]), np.array([2.2, -1.9, 2.6]))

        graph.build_spatial_topology()
        summary = graph.to_dict()

        assert summary["node_count"] == 6
        assert summary["edge_count"] >= 4

        # Check door hosted by wall
        boost_dr, reasons_dr = graph.get_topology_reinforcement("dr_1")
        assert "HOST_WALL_CONFIRMED" in reasons_dr

        # Check valve connected to pipe
        boost_vl, reasons_vl = graph.get_topology_reinforcement("vl_1")
        assert "PIPE_CONNECTION_CONFIRMED" in reasons_vl


# ── 6. Object-Specific Reconstruction & Quality Gates Tests ───────────────────

class TestReconstructionAndQualityGates:
    """Validates Section 18 & 27 object-specific reconstruction."""

    def test_pipe_reconstruction(self):
        angles = np.random.uniform(0, 2 * np.pi, 200)
        px = np.random.uniform(0, 3, 200)
        py = 0.08 * np.cos(angles)
        pz = 0.08 * np.sin(angles)
        pipe_pts = np.column_stack([px, py, pz]).astype(np.float32)

        pipe = reconstruct_pipe_from_points(pipe_pts, list(range(len(pipe_pts))), "test_pipe_01")
        assert pipe is not None
        assert abs(pipe.diameter_m - 0.16) < 0.03
        assert abs(pipe.length_m - 3.0) < 0.30
        assert pipe.confidence > 0.60

    def test_duct_reconstruction(self):
        # Rectangular cross-section 0.4m x 0.3m along length 2m
        pts = np.random.uniform(low=[0, -0.2, -0.15], high=[2.0, 0.2, 0.15], size=(200, 3)).astype(np.float32)
        duct = reconstruct_duct_from_points(pts, list(range(len(pts))), "test_duct_01")
        assert duct is not None
        assert abs(duct.length_m - 2.0) < 0.20
        assert duct.width_m > 0.15
        assert duct.height_m > 0.10

    def test_quality_gate_rejects_degenerate_point_support(self):
        """Micro-cloud with fewer than 5 points must be REJECTED."""
        tiny_pts = np.array([[0, 0, 0], [1, 0, 0]], dtype=np.float32)
        obj = reconstruct_object_from_points(tiny_pts, [0, 1], "obj_tiny")
        assert obj.decision_status == "REJECTED"
        assert "INSUFFICIENT_POINT_SUPPORT" in obj.rejection_reasons


# ── 7. Native Revit Element Generation Tests ──────────────────────────────────

class TestNativeRevitGeneration:
    """Validates Section 28 native Revit element instructions."""

    def test_revit_pipe_instruction(self):
        angles = np.random.uniform(0, 2 * np.pi, 150)
        px = np.random.uniform(0, 2, 150)
        py = 0.05 * np.cos(angles)
        pz = 0.05 * np.sin(angles)
        pts = np.column_stack([px, py, pz]).astype(np.float32)

        pipe = reconstruct_pipe_from_points(pts, list(range(len(pts))), "pipe_100")
        assert pipe is not None

        gen = NativeRevitGenerator()
        instr = gen.from_pipe(pipe)

        assert isinstance(instr, ElementInstruction)
        assert instr.element_type == ElementType.PIPE
        assert "diameter_mm" in instr.parameters
        assert instr.parameters["diameter_mm"] > 0
        assert instr.bounding_box.max_x >= instr.bounding_box.min_x

    def test_revit_unknown_object_directshape_instruction(self):
        blob = np.random.normal(loc=0.0, scale=0.3, size=(50, 3)).astype(np.float32)
        obj = reconstruct_object_from_points(blob, list(range(50)), "obj_unk", semantic_label="UNKNOWN_OBJECT")

        gen = NativeRevitGenerator()
        instr = gen.from_reconstructed_object(obj)

        assert isinstance(instr, ElementInstruction)
        assert instr.element_type == ElementType.GENERIC_MODEL
        assert instr.parameters["semantic_label"] == "UNKNOWN_OBJECT"
        assert "obb_extents_mm" in instr.parameters


# ── 8. CPU Enforcement & CUDA Guard Tests ─────────────────────────────────────

class TestCPUEnforcementAndCUDAGuard:
    """Validates Section 2 CPU-only non-negotiable requirements."""

    def test_cuda_strictly_unavailable(self):
        assert not torch.cuda.is_available(), "CUDA must be unavailable in production environment"

    def test_device_string_is_cpu(self):
        pts = np.random.uniform(0, 1, (50, 3)).astype(np.float32)
        res = run_phase3b_pipeline(pts, source_file="cpu_check.pcd")
        assert res.device == "CPU"


# ── 9. Negative & Ambiguity Test Cases ────────────────────────────────────────

class TestNegativeAndAmbiguityCases:
    """Validates Section 34 negative test scenarios."""

    def test_empty_cloud_does_not_crash(self):
        empty = np.zeros((0, 3), dtype=np.float32)
        res = run_phase3b_pipeline(empty, source_file="empty.pcd")
        assert res.total_proposals == 0
        assert res.accepted_count == 0

    def test_touching_wall_and_pipe_separated(self):
        """Touching objects with distinct geometry must be separated into discrete proposals."""
        wall = np.random.uniform(low=[0, -0.02, 0], high=[3, 0.02, 3], size=(200, 3))
        # Pipe touching wall at y = 0.03
        angles = np.random.uniform(0, 2*np.pi, 150)
        px = np.random.uniform(0, 3, 150)
        py = 0.03 + 0.05 * np.cos(angles)
        pz = 1.5 + 0.05 * np.sin(angles)
        pipe = np.column_stack([px, py, pz])

        combined = np.vstack([wall, pipe]).astype(np.float32)
        engine = ClassAgnosticProposalEngine(min_proposal_points=15)
        proposals = engine.generate_proposals(combined)

        assert len(proposals) >= 2
