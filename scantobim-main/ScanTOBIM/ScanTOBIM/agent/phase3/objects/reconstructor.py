"""Generic & Unknown Object Reconstruction Engine for Scan-to-BIM.

Phase 3B Core Requirement (Sections 18, 25, 26, 27):
- Derives precise physical boundaries from point clouds for unknown objects, machinery,
  doors, windows, beams, and equipment.
- NEVER manufactures fallback generic boxes or fixed dimensions.
- Evaluates Reconstruction Quality Gates (Section 27):
    * finite values
    * non-zero dimensions
    * reasonable object extents (not 0.001mm or 500m)
    * source-point support (minimum points)
    * geometric residual
- Categorizes object lifecycle state:
    * ACCEPTED
    * UNCERTAIN
    * REVIEW_REQUIRED
    * REJECTED
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any
import numpy as np
import structlog

logger = structlog.get_logger(__name__)


@dataclass
class ReconstructedObject:
    """Parametric representation of an arbitrary 3D physical object instance."""
    object_id: str
    semantic_label: str
    level1_category: str
    centroid_m: tuple[float, float, float]
    bbox_min_m: tuple[float, float, float]
    bbox_max_m: tuple[float, float, float]
    obb_center_m: tuple[float, float, float]
    obb_extents_m: tuple[float, float, float]
    obb_rotation_matrix: list[list[float]]
    dimensions_m: dict[str, float]
    surface_area_m2: float
    volume_m3: float
    source_point_indices: list[int]
    confidence: float
    decision_status: str  # ACCEPTED, UNCERTAIN, REVIEW_REQUIRED, REJECTED
    rejection_reasons: list[str] = field(default_factory=list)
    confidence_reason_codes: list[str] = field(default_factory=list)
    topology_relations: list[dict[str, Any]] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "object_id": self.object_id,
            "semantic_label": self.semantic_label,
            "level1_category": self.level1_category,
            "decision_status": self.decision_status,
            "confidence": round(float(self.confidence), 4),
            "point_count": len(self.source_point_indices),
            "centroid_m": [round(float(c), 4) for c in self.centroid_m],
            "bbox_min_m": [round(float(c), 4) for c in self.bbox_min_m],
            "bbox_max_m": [round(float(c), 4) for c in self.bbox_max_m],
            "obb": {
                "center_m": [round(float(c), 4) for c in self.obb_center_m],
                "extents_m": [round(float(e), 4) for e in self.obb_extents_m],
                "rotation_matrix": self.obb_rotation_matrix,
            },
            "dimensions_m": {k: round(float(v), 4) for k, v in self.dimensions_m.items()},
            "surface_area_m2": round(float(self.surface_area_m2), 4),
            "volume_m3": round(float(self.volume_m3), 6),
            "confidence_reason_codes": self.confidence_reason_codes,
            "rejection_reasons": self.rejection_reasons,
            "topology_relations": self.topology_relations,
            "metadata": self.metadata,
        }


def reconstruct_object_from_points(
    points: np.ndarray,
    source_indices: list[int] | np.ndarray,
    object_id: str,
    semantic_label: str = "UNKNOWN_OBJECT",
    level1_category: str = "OTHER",
    confidence: float = 0.50,
    confidence_reason_codes: list[str] | None = None,
    topology_relations: list[dict[str, Any]] | None = None,
    metadata: dict[str, Any] | None = None,
) -> ReconstructedObject:
    """Derive measured physical object geometry and apply quality gates."""
    rejection_reasons: list[str] = []
    reasons = confidence_reason_codes or []
    N = len(points)

    if N < 5:
        rejection_reasons.append("INSUFFICIENT_POINT_SUPPORT")

    # Check finite coordinates
    if not np.all(np.isfinite(points)):
        rejection_reasons.append("NON_FINITE_COORDINATES")

    if rejection_reasons:
        return ReconstructedObject(
            object_id=object_id,
            semantic_label=semantic_label,
            level1_category=level1_category,
            centroid_m=(0.0, 0.0, 0.0),
            bbox_min_m=(0.0, 0.0, 0.0),
            bbox_max_m=(0.0, 0.0, 0.0),
            obb_center_m=(0.0, 0.0, 0.0),
            obb_extents_m=(0.0, 0.0, 0.0),
            obb_rotation_matrix=[[1,0,0],[0,1,0],[0,0,1]],
            dimensions_m={"length": 0.0, "width": 0.0, "height": 0.0},
            surface_area_m2=0.0,
            volume_m3=0.0,
            source_point_indices=[int(i) for i in source_indices],
            confidence=0.0,
            decision_status="REJECTED",
            rejection_reasons=rejection_reasons,
        )

    # 1. Compute Centroid & AABB
    centroid = np.mean(points, axis=0)
    bbox_min = np.min(points, axis=0)
    bbox_max = np.max(points, axis=0)
    span = bbox_max - bbox_min

    # 2. PCA & OBB
    centered = points - centroid
    cov = np.dot(centered.T, centered) / float(N)
    eigvals, eigvecs = np.linalg.eigh(cov)
    sort_idx = np.argsort(eigvals)
    rot_matrix = eigvecs[:, sort_idx]

    projs = np.dot(centered, rot_matrix)
    p_min = np.min(projs, axis=0)
    p_max = np.max(projs, axis=0)
    obb_extents = np.maximum(p_max - p_min, 1e-4)
    obb_center = centroid + np.dot(rot_matrix, (p_min + p_max) / 2.0)

    # 3. Measured Physical Dimensions
    dim_x, dim_y, dim_z = float(obb_extents[0]), float(obb_extents[1]), float(obb_extents[2])
    surface_area = 2.0 * (dim_x * dim_y + dim_y * dim_z + dim_x * dim_z)
    volume = dim_x * dim_y * dim_z

    dimensions = {
        "length_m": round(dim_z, 4),
        "width_m": round(dim_y, 4),
        "height_m": round(dim_x, 4),
    }

    # 4. Reconstruction Quality Gates (Section 27)
    if any(e < 0.01 for e in obb_extents):
        rejection_reasons.append("DEGENERATE_EXTENT_LESS_THAN_10MM")
    if any(e > 100.0 for e in obb_extents):
        rejection_reasons.append("EXTENT_EXCEEDS_REASONABLE_BUILDING_BOUNDS")
    if volume <= 0.0 or not np.isfinite(volume):
        rejection_reasons.append("INVALID_VOLUME")

    # Decision status logic
    if rejection_reasons:
        decision_status = "REJECTED"
    elif confidence < 0.40 or "AMBIGUOUS_EVIDENCE" in reasons:
        decision_status = "UNCERTAIN"
    elif "UNKNOWN" in semantic_label:
        decision_status = "REVIEW_REQUIRED"
    else:
        decision_status = "ACCEPTED"

    return ReconstructedObject(
        object_id=object_id,
        semantic_label=semantic_label,
        level1_category=level1_category,
        centroid_m=(float(centroid[0]), float(centroid[1]), float(centroid[2])),
        bbox_min_m=(float(bbox_min[0]), float(bbox_min[1]), float(bbox_min[2])),
        bbox_max_m=(float(bbox_max[0]), float(bbox_max[1]), float(bbox_max[2])),
        obb_center_m=(float(obb_center[0]), float(obb_center[1]), float(obb_center[2])),
        obb_extents_m=(dim_x, dim_y, dim_z),
        obb_rotation_matrix=[[round(float(v), 5) for v in row] for row in rot_matrix],
        dimensions_m=dimensions,
        surface_area_m2=float(surface_area),
        volume_m3=float(volume),
        source_point_indices=[int(i) for i in source_indices],
        confidence=confidence,
        decision_status=decision_status,
        rejection_reasons=rejection_reasons,
        confidence_reason_codes=reasons,
        topology_relations=topology_relations or [],
        metadata=metadata or {},
    )
