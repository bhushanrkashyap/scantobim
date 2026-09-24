"""Phase 3 Orchestration Pipeline — Semantic Understanding & Architectural Reconstruction.

Architecture:
  Canonical Point Cloud
  ↓
  Semantic Understanding (Model Selector -> Real Neural / Geometric Fallback)
  ↓
  Instance Separation (Geometric Clustering Fallback / Neural Instance)
  ↓
  Instance Deduplication (NMS & Overlap Suppression)
  ↓
  Storey Understanding (Data-derived level detection)
  ↓
  Slab/Floor Reconstruction (Measured thickness & 2D boundary polygons)
  ↓
  Wall Reconstruction (Measured thickness, 5-vector local frame, fragment merge)
  ↓
  Column Reconstruction (PCA & MBR scan-derived dimensions)
  ↓
  Geometric Validation & Source Support (Rejects fabricated or weak candidates)
  ↓
  BIM-Ready Architectural Candidates (Structured JSON contract with provenance)
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import structlog

from agent.phase3.columns.reconstructor import reconstruct_column_from_points
from agent.phase3.instance.deduplication import deduplicate_instances
from agent.phase3.instance.geometric_clustering import GeometricInstanceAdapter
from agent.phase3.provenance.tracker import (
    BIMArchitecturalCandidate,
    ProvenanceTracker,
)
from agent.phase3.semantic.selector import SemanticModelSelector
from agent.phase3.slabs.reconstructor import reconstruct_slab_from_points
from agent.phase3.storeys.detector import detect_storeys_from_point_cloud
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

logger = structlog.get_logger()


@dataclass
class Phase3ReconstructionResult:
    """Execution outcome of Phase 3 pipeline."""

    status: str  # "PHASE_3_PASS", etc.
    source_file: str
    point_count: int
    semantic_model: str
    semantic_model_status: str
    storeys_count: int
    slabs_count: int
    walls_count: int
    columns_count: int
    accepted_candidates: list[BIMArchitecturalCandidate]
    rejected_candidates: list[dict[str, Any]]
    runtime_s: float
    diagnostics: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "status": self.status,
            "source_file": self.source_file,
            "point_count": self.point_count,
            "semantic_model": self.semantic_model,
            "semantic_model_status": self.semantic_model_status,
            "counts": {
                "storeys": self.storeys_count,
                "slabs": self.slabs_count,
                "walls": self.walls_count,
                "columns": self.columns_count,
                "total_accepted": len(self.accepted_candidates),
                "total_rejected": len(self.rejected_candidates),
            },
            "runtime_s": round(self.runtime_s, 4),
            "accepted_candidates": [c.to_dict() for c in self.accepted_candidates],
            "rejected_candidates": self.rejected_candidates,
            "diagnostics": self.diagnostics,
        }


def run_phase3_reconstruction(
    points: np.ndarray,
    normals: np.ndarray | None = None,
    source_file: str = "canonical_cloud.pcd",
    transform: AuthoritativeTransform | None = None,
    preferred_semantic_model: str | None = None,
) -> Phase3ReconstructionResult:
    """Execute complete Phase 3 semantic understanding and architectural reconstruction pipeline."""
    t0 = time.time()
    logger.info("phase3_reconstruction_started", points=len(points), source_file=source_file)

    trans = transform or GLOBAL_TRANSFORM
    tracker = ProvenanceTracker(source_file=source_file, transform=trans)

    # 0. Surface normal estimation if not provided
    if normals is None and len(points) > 0:
        from agent.phase3.semantic.geometric_adapter import estimate_point_normals
        normals = estimate_point_normals(points)

    # 1. Semantic Understanding
    selector = SemanticModelSelector(preferred_model=preferred_semantic_model)
    _model_key, semantic_adapter, rationale = selector.select_active_adapter()
    semantic_res = semantic_adapter.infer(points, normals=normals)

    # 2. Instance Separation
    inst_adapter = GeometricInstanceAdapter()
    inst_res = inst_adapter.segment(points, semantic_labels=semantic_res.labels, normals=normals)

    # 3. Instance Deduplication & NMS
    dedup_res = deduplicate_instances(inst_res.instances)
    active_instances = dedup_res.active_instances

    # 4. Storey Understanding
    storeys = detect_storeys_from_point_cloud(points, normals=normals)
    primary_storey_id = storeys[0].storey_id if storeys else "LEVEL_00"
    primary_elevation = storeys[0].elevation_m if storeys else 0.0

    accepted: list[BIMArchitecturalCandidate] = []
    rejected: list[dict[str, Any]] = []

    # 5. Slab / Floor Reconstruction
    slab_instances = [inst for inst in active_instances if inst.semantic_class in ("FLOOR", "CEILING")]
    reconstructed_slabs = []
    for inst in slab_instances:
        slab = reconstruct_slab_from_points(
            points=points[inst.point_indices],
            source_indices=inst.point_indices,
            slab_id=inst.instance_id,
            storey_id=primary_storey_id,
            slab_type=inst.semantic_class,
        )
        if slab is not None:
            # Validate
            val = validate_slab_geometry(
                boundary_polygon=slab.boundary_polygon_m,
                thickness_m=slab.thickness_m,
                points=points[inst.point_indices],
            )
            if val.is_valid:
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
                    confidence=slab.confidence,
                    validation_report=val.to_dict(),
                    semantic_evidence={"model": semantic_res.model_name, "status": semantic_res.status},
                    geometric_evidence={"plane": list(slab.plane_equation)},
                )
                accepted.append(cand)
                reconstructed_slabs.append(slab)
            else:
                rejected.append({"id": slab.slab_id, "type": slab.slab_type, "reasons": val.reasons})
        else:
            rejected.append({"id": inst.instance_id, "type": inst.semantic_class, "reasons": ["Slab fitting failed"]})

    # 6. Wall Reconstruction
    wall_instances = [inst for inst in active_instances if inst.semantic_class == "WALL"]
    raw_walls = []
    for inst in wall_instances:
        wall = reconstruct_wall_from_points(
            points=points[inst.point_indices],
            source_indices=inst.point_indices,
            wall_id=inst.instance_id,
            storey_id=primary_storey_id,
        )
        if wall is not None:
            raw_walls.append(wall)
        else:
            rejected.append({"id": inst.instance_id, "type": "WALL", "reasons": ["Wall geometric fit failed"]})

    # Fragment merging and junction detection
    merged_walls = merge_collinear_wall_fragments(raw_walls)
    detect_wall_junctions(merged_walls)

    for wall in merged_walls:
        val = validate_wall_geometry(
            start_pt=wall.start_point_m,
            end_pt=wall.end_point_m,
            length_m=wall.length_m,
            height_m=wall.height_m,
            thickness_m=wall.thickness_m,
            points=points[wall.source_point_indices],
        )
        if val.is_valid:
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
                raw_source_points=points[wall.source_point_indices],
                confidence=wall.confidence,
                validation_report=val.to_dict(),
                semantic_evidence={"model": semantic_res.model_name, "status": semantic_res.status},
                geometric_evidence={
                    "plane": list(wall.plane_equation),
                    "measurement_method": wall.thickness_measurement_method,
                    "junctions": wall.junction_connections,
                },
            )
            accepted.append(cand)
        else:
            rejected.append({"id": wall.wall_id, "type": "WALL", "reasons": val.reasons})

    # 7. Column Reconstruction
    col_instances = [inst for inst in active_instances if inst.semantic_class == "COLUMN"]
    for inst in col_instances:
        col = reconstruct_column_from_points(
            points=points[inst.point_indices],
            source_indices=inst.point_indices,
            column_id=inst.instance_id,
            storey_id=primary_storey_id,
        )
        if col is not None:
            val = validate_column_geometry(
                center_xyz=col.center_xyz_m,
                width_m=col.width_m,
                depth_m=col.depth_m,
                height_m=col.height_m,
                points=points[inst.point_indices],
            )
            if val.is_valid:
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
                    confidence=col.confidence,
                    validation_report=val.to_dict(),
                    semantic_evidence={"model": semantic_res.model_name, "status": semantic_res.status},
                    geometric_evidence={
                        "profile": col.profile_type,
                        "rotation_deg": col.rotation_deg,
                    },
                )
                accepted.append(cand)
            else:
                rejected.append({"id": col.column_id, "type": "COLUMN", "reasons": val.reasons})
        else:
            rejected.append({"id": inst.instance_id, "type": "COLUMN", "reasons": ["Column geometric fit failed"]})

    runtime = time.time() - t0
    status_str = "PHASE_3_PASS" if (len(storeys) > 0 and len(accepted) > 0) else "PHASE_3_PASS_WITH_CONFIG"

    return Phase3ReconstructionResult(
        status=status_str,
        source_file=source_file,
        point_count=len(points),
        semantic_model=semantic_res.model_name,
        semantic_model_status=semantic_res.status,
        storeys_count=len(storeys),
        slabs_count=len([c for c in accepted if c.semantic_type in ("FLOOR", "CEILING")]),
        walls_count=len([c for c in accepted if c.semantic_type == "WALL"]),
        columns_count=len([c for c in accepted if c.semantic_type == "COLUMN"]),
        accepted_candidates=accepted,
        rejected_candidates=rejected,
        runtime_s=runtime,
        diagnostics={
            "selection_rationale": rationale,
            "dedup_suppressed_count": len(dedup_res.suppressed_candidates),
        },
    )
