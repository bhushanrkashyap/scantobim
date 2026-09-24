"""Open-World 3D Object Understanding End-to-End Pipeline for Scan-to-BIM.

Phase 3B Master Pipeline Implementation (Section 5):
Architecture:
  REAL POINT CLOUD
  ↓
  INPUT INSPECTION
  ↓
  UNIT RESOLUTION & COORDINATE NORMALIZATION
  ↓
  ORIENTATION & DENSITY / RESOLUTION ESTIMATION
  ↓
  MULTI-SCALE REPRESENTATIONS
  ↓
  GEOMETRIC FEATURE EXTRACTION
  ↓
  SEMANTIC SEGMENTATION (Active CPU Neural Model / RandLA-Net)
  ↓
  CLASS-AGNOSTIC OBJECT PROPOSAL GENERATION
  ↓
  INSTANCE SEPARATION & PROVENANCE PRESERVATION
  ↓
  OPEN-VOCABULARY / CATEGORY RECOGNITION
  ↓
  SEMANTIC + GEOMETRIC FUSION
  ↓
  TOPOLOGICAL REASONING
  ↓
  OBJECT VALIDATION & RECONSTRUCTION QUALITY GATES
  ↓
  OBJECT CONFIDENCE CALIBRATION & REASON CODES
  ↓
  UNKNOWN / REVIEW HANDLING
  ↓
  PARAMETRIC BIM RECONSTRUCTION
  ↓
  NATIVE REVIT ELEMENTS

STRICT CPU ENFORCEMENT — NO CUDA OPERATORS — NO FABRICATED DETECTIONS.
"""

from __future__ import annotations

import os
import time
from dataclasses import dataclass, field
from typing import Any
import numpy as np
import structlog

from agent.models import ElementInstruction
from agent.phase3.columns.reconstructor import reconstruct_column_from_points
from agent.phase3.geometry.features import (
    GeometricFeatureSet,
    compute_geometric_features,
    estimate_multi_scale_radii,
)
from agent.phase3.instance.adapter import InstanceMask
from agent.phase3.instance.proposal_engine import ClassAgnosticProposalEngine, ObjectProposal
from agent.phase3.mep.reconstructor import (
    ReconstructedCableTray,
    ReconstructedDuct,
    ReconstructedPipe,
    ReconstructedValve,
    reconstruct_cable_tray_from_points,
    reconstruct_duct_from_points,
    reconstruct_pipe_from_points,
    reconstruct_valve_from_points,
)
from agent.phase3.objects.reconstructor import (
    ReconstructedObject,
    reconstruct_object_from_points,
)
from agent.phase3.open_world.engine import (
    OPEN_VOCABULARY_STATUS,
    OpenWorldRecognitionEngine,
    SemanticClassificationResult,
)
from agent.phase3.provenance.tracker import (
    BIMArchitecturalCandidate,
    ProvenanceTracker,
)
from agent.phase3.revit.native_generator import NativeRevitGenerator
from agent.phase3.semantic.selector import SemanticModelSelector
from agent.phase3.slabs.reconstructor import reconstruct_slab_from_points
from agent.phase3.status import REAL_NEURAL_INFERENCE
from agent.phase3.storeys.detector import detect_storeys_from_point_cloud
from agent.phase3.taxonomy import HIERARCHICAL_TAXONOMY, LEVEL1_CATEGORIES, get_category_for_class
from agent.phase3.topology.graph import TopologyGraph
from agent.phase3.validation.validator import (
    validate_column_geometry,
    validate_slab_geometry,
    validate_wall_geometry,
)
from agent.phase3.walls.reconstructor import (
    detect_wall_junctions,
    merge_collinear_wall_fragments,
    reconstruct_wall_from_points,
)
from agent.tools.coordinate_system import GLOBAL_TRANSFORM, AuthoritativeTransform

logger = structlog.get_logger(__name__)


@dataclass
class Phase3BExecutionResult:
    """Complete execution outcome of the Phase 3B Open-World 3D Object pipeline."""
    status: str  # "PHASE_3B_PASS", etc.
    source_file: str
    total_source_points: int
    processed_points: int
    semantic_model: str
    semantic_model_status: str
    checkpoint_hash: str | None
    device: str
    runtime_s: float
    memory_usage_mb: float

    # Quantitative counts
    total_proposals: int
    supported_objects_count: int
    unknown_objects_count: int
    review_required_count: int
    rejected_count: int
    accepted_count: int

    # Categorical counts
    per_class_counts: dict[str, int]
    per_category_counts: dict[str, int]
    per_class_confidence: dict[str, float]

    # Reconstructed elements
    walls: list[Any] = field(default_factory=list)
    slabs: list[Any] = field(default_factory=list)
    columns: list[Any] = field(default_factory=list)
    pipes: list[ReconstructedPipe] = field(default_factory=list)
    ducts: list[ReconstructedDuct] = field(default_factory=list)
    cable_trays: list[ReconstructedCableTray] = field(default_factory=list)
    valves: list[ReconstructedValve] = field(default_factory=list)
    other_objects: list[ReconstructedObject] = field(default_factory=list)

    # Topology & Revit
    topology_summary: dict[str, Any] = field(default_factory=dict)
    revit_instructions: list[ElementInstruction] = field(default_factory=list)
    diagnostics: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "status": self.status,
            "source_file": self.source_file,
            "provenance": {
                "total_source_points": self.total_source_points,
                "processed_points": self.processed_points,
                "semantic_model": self.semantic_model,
                "semantic_model_status": self.semantic_model_status,
                "checkpoint_hash": self.checkpoint_hash,
                "device": self.device,
                "runtime_s": round(self.runtime_s, 4),
                "memory_usage_mb": round(self.memory_usage_mb, 2),
            },
            "counts": {
                "total_proposals": self.total_proposals,
                "supported_objects": self.supported_objects_count,
                "unknown_objects": self.unknown_objects_count,
                "accepted": self.accepted_count,
                "review_required": self.review_required_count,
                "rejected": self.rejected_count,
                "revit_instructions": len(self.revit_instructions),
            },
            "per_class_counts": self.per_class_counts,
            "per_category_counts": self.per_category_counts,
            "per_class_confidence": {k: round(v, 4) for k, v in self.per_class_confidence.items()},
            "topology": self.topology_summary,
            "diagnostics": self.diagnostics,
        }


def run_phase3b_pipeline(
    points: np.ndarray,
    normals: np.ndarray | None = None,
    source_file: str = "canonical_point_cloud",
    source_point_count: int | None = None,
    transform: AuthoritativeTransform | None = None,
    preferred_semantic_model: str | None = None,
) -> Phase3BExecutionResult:
    """Execute complete Phase 3B Open-World 3D Object Understanding Pipeline."""
    t0 = time.time()
    device_str = "CPU"
    N = len(points)
    total_pts = source_point_count if source_point_count is not None else N

    logger.info("phase3b_pipeline_started", points=N, source_file=source_file, device=device_str)

    # 1. Coordinate Normalization & Authoritative Transform
    trans = transform or GLOBAL_TRANSFORM
    tracker = ProvenanceTracker(source_file=source_file, transform=trans)
    revit_gen = NativeRevitGenerator(zone_id="ZONE_MAIN")

    # 2. Density & Multi-scale Estimation
    radii = estimate_multi_scale_radii(points)

    # 3. Geometric Feature Engine
    features = compute_geometric_features(
        points,
        k_neighbors=min(24, max(4, N)),
        precomputed_radii=radii,
    )
    use_normals = normals if normals is not None else features.normals

    # 4. Semantic Segmentation via Active Model Selector
    selector = SemanticModelSelector(preferred_model=preferred_semantic_model)
    _model_key, semantic_adapter, rationale = selector.select_active_adapter()
    semantic_res = semantic_adapter.infer(points, normals=use_normals)
    ckpt_hash = getattr(semantic_res, "checkpoint_hash", None)

    # 5. Class-Agnostic Object Proposal Generation (Section 13)
    proposal_engine = ClassAgnosticProposalEngine(
        min_proposal_points=max(15, min(30, N // 200)),
        normal_smoothness_deg=35.0,
    )
    proposals = proposal_engine.generate_proposals(points, normals=use_normals)

    # 6. Storey Understanding
    storeys = detect_storeys_from_point_cloud(points, normals=use_normals)
    primary_storey_id = storeys[0].storey_id if storeys else "LEVEL_00"
    primary_elevation = storeys[0].elevation_m if storeys else 0.0

    # 7. Open-World Recognition & Unknown Handling (Sections 15 & 16)
    ow_engine = OpenWorldRecognitionEngine(min_known_confidence=0.50)
    classified_proposals: list[tuple[ObjectProposal, SemanticClassificationResult]] = []

    for prop in proposals:
        class_res = ow_engine.classify_proposal(
            proposal=prop,
            neural_labels=semantic_res.labels,
            neural_probs=semantic_res.probabilities,
        )
        classified_proposals.append((prop, class_res))

    # 8. Topology Graph Construction (Section 19)
    top_graph = TopologyGraph(connection_tolerance_m=0.30)
    for prop, class_res in classified_proposals:
        top_graph.add_object(
            object_id=prop.proposal_id,
            semantic_label=class_res.final_label,
            centroid_m=prop.centroid_m,
            bbox_min_m=prop.bbox_min_m,
            bbox_max_m=prop.bbox_max_m,
            obb_extents_m=prop.obb_extents_m,
            point_count=prop.point_count,
        )
    top_graph.build_spatial_topology()

    # Apply topological reinforcement feedback
    for prop, class_res in classified_proposals:
        boost, reasons = top_graph.get_topology_reinforcement(prop.proposal_id)
        if boost > 0:
            class_res.confidence = min(0.99, class_res.confidence + boost)
            class_res.reason_codes.extend(reasons)

    # 9. Object-Specific Reconstruction & Quality Gates (Sections 18 & 27)
    reconstructed_walls: list[Any] = []
    reconstructed_slabs: list[Any] = []
    reconstructed_columns: list[Any] = []
    reconstructed_pipes: list[ReconstructedPipe] = []
    reconstructed_ducts: list[ReconstructedDuct] = []
    reconstructed_cable_trays: list[ReconstructedCableTray] = []
    reconstructed_valves: list[ReconstructedValve] = []
    reconstructed_others: list[ReconstructedObject] = []

    accepted_count = 0
    review_required_count = 0
    rejected_count = 0
    revit_instructions: list[ElementInstruction] = []

    per_class_counts: dict[str, int] = {}
    per_category_counts: dict[str, int] = {}
    class_conf_accum: dict[str, list[float]] = {}

    for prop, class_res in classified_proposals:
        lbl = class_res.final_label
        cat = class_res.level1_category
        conf = class_res.confidence
        c_pts = points[prop.point_indices]
        c_src = prop.point_indices

        per_class_counts[lbl] = per_class_counts.get(lbl, 0) + 1
        per_category_counts[cat] = per_category_counts.get(cat, 0) + 1
        if lbl not in class_conf_accum:
            class_conf_accum[lbl] = []
        class_conf_accum[lbl].append(conf)

        # A. WALL Reconstruction
        if lbl == "WALL":
            wall = reconstruct_wall_from_points(
                points=c_pts,
                source_indices=c_src,
                wall_id=prop.proposal_id,
                storey_id=primary_storey_id,
            )
            if wall is not None:
                val = validate_wall_geometry(
                    start_pt=wall.start_point_m,
                    end_pt=wall.end_point_m,
                    length_m=wall.length_m,
                    height_m=wall.height_m,
                    thickness_m=wall.thickness_m,
                    points=c_pts,
                )
                if val.is_valid:
                    reconstructed_walls.append(wall)
                    accepted_count += 1
                    cand = tracker.build_wall_candidate(
                        wall_id=wall.wall_id,
                        storey_id=wall.storey_id,
                        storey_elevation=primary_elevation,
                        start_pt_canon=wall.start_point_m,
                        end_pt_canon=wall.end_point_m,
                        length_m=wall.length_m,
                        height_m=wall.height_m,
                        thickness_m=wall.thickness_m,
                        orientation_deg=wall.orientation_deg,
                        local_frame=wall.local_frame.to_dict(),
                        source_indices=wall.source_point_indices,
                        raw_source_points=c_pts,
                        confidence=conf,
                        validation_report=val.to_dict(),
                        semantic_evidence={"model": semantic_res.model_name, "status": semantic_res.status},
                        geometric_evidence={"plane": list(wall.plane_equation)},
                    )
                    revit_instructions.append(revit_gen.from_architectural_candidate(cand))
                else:
                    rejected_count += 1
            else:
                rejected_count += 1

        # B. FLOOR / SLAB / CEILING Reconstruction
        elif lbl in ("FLOOR", "CEILING", "SLAB"):
            slab = reconstruct_slab_from_points(
                points=c_pts,
                source_indices=c_src,
                slab_id=prop.proposal_id,
                storey_id=primary_storey_id,
                slab_type=lbl,
            )
            if slab is not None:
                val = validate_slab_geometry(
                    boundary_polygon=slab.boundary_polygon_m,
                    thickness_m=slab.thickness_m,
                    points=c_pts,
                )
                if val.is_valid:
                    reconstructed_slabs.append(slab)
                    accepted_count += 1
                    cand = tracker.build_slab_candidate(
                        slab_id=slab.slab_id,
                        storey_id=slab.storey_id,
                        slab_type=slab.slab_type,
                        top_elevation_m=slab.top_elevation_m,
                        bottom_elevation_m=slab.bottom_elevation_m,
                        thickness_m=slab.thickness_m,
                        boundary_polygon_m=slab.boundary_polygon_m,
                        area_m2=slab.area_m2,
                        source_indices=slab.source_point_indices,
                        confidence=conf,
                        validation_report=val.to_dict(),
                        semantic_evidence={"model": semantic_res.model_name, "status": semantic_res.status},
                        geometric_evidence={"plane": list(slab.plane_equation)},
                    )
                    revit_instructions.append(revit_gen.from_architectural_candidate(cand))
                else:
                    rejected_count += 1
            else:
                rejected_count += 1

        # C. COLUMN Reconstruction
        elif lbl == "COLUMN":
            col = reconstruct_column_from_points(
                points=c_pts,
                source_indices=c_src,
                column_id=prop.proposal_id,
                storey_id=primary_storey_id,
            )
            if col is not None:
                val = validate_column_geometry(
                    center_xyz=col.center_xyz_m,
                    width_m=col.width_m,
                    depth_m=col.depth_m,
                    height_m=col.height_m,
                    points=c_pts,
                )
                if val.is_valid:
                    reconstructed_columns.append(col)
                    accepted_count += 1
                    cand = tracker.build_column_candidate(
                        column_id=col.column_id,
                        storey_id=col.storey_id,
                        storey_elevation=primary_elevation,
                        center_canon=col.center_xyz_m,
                        width_m=col.width_m,
                        depth_m=col.depth_m,
                        height_m=col.height_m,
                        rotation_deg=col.rotation_deg,
                        profile_type=col.profile_type,
                        source_indices=col.source_point_indices,
                        confidence=conf,
                        validation_report=val.to_dict(),
                        semantic_evidence={"model": semantic_res.model_name, "status": semantic_res.status},
                        geometric_evidence={"profile": col.profile_type, "rotation_deg": col.rotation_deg},
                    )
                    revit_instructions.append(revit_gen.from_architectural_candidate(cand))
                else:
                    rejected_count += 1
            else:
                rejected_count += 1

        # D. PIPE Reconstruction
        elif lbl == "PIPE":
            pipe = reconstruct_pipe_from_points(
                points=c_pts,
                source_indices=c_src,
                pipe_id=prop.proposal_id,
            )
            if pipe is not None:
                reconstructed_pipes.append(pipe)
                accepted_count += 1
                revit_instructions.append(revit_gen.from_pipe(pipe))
            else:
                rejected_count += 1

        # E. DUCT Reconstruction
        elif lbl == "DUCT":
            duct = reconstruct_duct_from_points(
                points=c_pts,
                source_indices=c_src,
                duct_id=prop.proposal_id,
            )
            if duct is not None:
                reconstructed_ducts.append(duct)
                accepted_count += 1
                revit_instructions.append(revit_gen.from_duct(duct))
            else:
                rejected_count += 1

        # F. CABLE TRAY Reconstruction
        elif lbl == "CABLE_TRAY":
            tray = reconstruct_cable_tray_from_points(
                points=c_pts,
                source_indices=c_src,
                tray_id=prop.proposal_id,
            )
            if tray is not None:
                reconstructed_cable_trays.append(tray)
                accepted_count += 1
                revit_instructions.append(revit_gen.from_cable_tray(tray))
            else:
                rejected_count += 1

        # G. VALVE Reconstruction
        elif lbl == "VALVE":
            valve = reconstruct_valve_from_points(
                points=c_pts,
                source_indices=c_src,
                valve_id=prop.proposal_id,
            )
            if valve is not None:
                reconstructed_valves.append(valve)
                accepted_count += 1
                revit_instructions.append(revit_gen.from_valve(valve))
            else:
                rejected_count += 1

        # H. UNKNOWN_OBJECT / OTHER / MACHINERY Reconstruction (Section 16 & 18)
        else:
            rec_obj = reconstruct_object_from_points(
                points=c_pts,
                source_indices=c_src,
                object_id=prop.proposal_id,
                semantic_label=lbl,
                level1_category=cat,
                confidence=conf,
                confidence_reason_codes=class_res.reason_codes,
            )
            reconstructed_others.append(rec_obj)
            if rec_obj.decision_status == "ACCEPTED":
                accepted_count += 1
                revit_instructions.append(revit_gen.from_reconstructed_object(rec_obj))
            elif rec_obj.decision_status == "REVIEW_REQUIRED":
                review_required_count += 1
                # Still output to Revit with DirectShape review flag
                revit_instructions.append(revit_gen.from_reconstructed_object(rec_obj))
            elif rec_obj.decision_status == "UNCERTAIN":
                review_required_count += 1
                revit_instructions.append(revit_gen.from_reconstructed_object(rec_obj))
            else:
                rejected_count += 1

    runtime = time.time() - t0

    # Memory usage estimate
    import resource
    rusage = resource.getrusage(resource.RUSAGE_SELF)
    mem_mb = float(rusage.ru_maxrss / (1024.0 * 1024.0 if os.uname().sysname == "Darwin" else 1024.0))

    per_class_conf = {k: float(np.mean(v)) for k, v in class_conf_accum.items()} if class_conf_accum else {}

    unknown_count = sum(v for k, v in per_class_counts.items() if "UNKNOWN" in k)
    supported_count = sum(v for k, v in per_class_counts.items() if "UNKNOWN" not in k)

    status_str = "PHASE_3B_PASS" if (len(proposals) > 0 and (accepted_count + review_required_count) > 0) else "PHASE_3B_REVIEW_REQUIRED"

    return Phase3BExecutionResult(
        status=status_str,
        source_file=source_file,
        total_source_points=total_pts,
        processed_points=N,
        semantic_model=semantic_res.model_name,
        semantic_model_status=semantic_res.status,
        checkpoint_hash=ckpt_hash,
        device=device_str,
        runtime_s=runtime,
        memory_usage_mb=mem_mb,
        total_proposals=len(proposals),
        supported_objects_count=supported_count,
        unknown_objects_count=unknown_count,
        review_required_count=review_required_count,
        rejected_count=rejected_count,
        accepted_count=accepted_count,
        per_class_counts=per_class_counts,
        per_category_counts=per_category_counts,
        per_class_confidence=per_class_conf,
        walls=reconstructed_walls,
        slabs=reconstructed_slabs,
        columns=reconstructed_columns,
        pipes=reconstructed_pipes,
        ducts=reconstructed_ducts,
        cable_trays=reconstructed_cable_trays,
        valves=reconstructed_valves,
        other_objects=reconstructed_others,
        topology_summary=top_graph.to_dict(),
        revit_instructions=revit_instructions,
        diagnostics={
            "selection_rationale": rationale,
            "open_vocabulary_status": OPEN_VOCABULARY_STATUS,
            "radii_used": radii.to_dict(),
        },
    )
