"""Source Point Provenance & BIM-Ready Architectural Candidate Contract — Phase 3.

Engineering Rules:
- Every architectural candidate retains traceable provenance back to source point cloud.
- Coordinates stored in both canonical frame [metres] and source frame [source units].
- Full evidence tracking: semantic, geometric, source support, and validation.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np

from agent.tools.coordinate_system import GLOBAL_TRANSFORM, AuthoritativeTransform


@dataclass
class BIMArchitecturalCandidate:
    """Structured BIM-ready candidate element."""

    candidate_id: str
    semantic_type: str  # "WALL", "FLOOR", "CEILING", "COLUMN"
    geometry: dict[str, Any]
    level: dict[str, Any]
    source: dict[str, Any]
    semantic_evidence: dict[str, Any]
    geometric_evidence: dict[str, Any]
    confidence: float
    validation: dict[str, Any]
    canonical_geometry: dict[str, Any] = field(default_factory=dict)
    source_geometry: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "candidate_id": self.candidate_id,
            "semantic_type": self.semantic_type,
            "geometry": self.geometry,
            "canonical_geometry": self.canonical_geometry or self.geometry,
            "source_geometry": self.source_geometry,
            "level": self.level,
            "source": self.source,
            "semantic_evidence": self.semantic_evidence,
            "geometric_evidence": self.geometric_evidence,
            "confidence": round(self.confidence, 4),
            "validation": self.validation,
        }


class ProvenanceTracker:
    """Builds authoritative BIM candidates with complete source coordinate provenance."""

    def __init__(
        self,
        source_file: str,
        transform: AuthoritativeTransform | None = None,
    ) -> None:
        self.source_file = source_file
        self.transform = transform or GLOBAL_TRANSFORM

    def to_source_point(self, pt: tuple[float, float, float]) -> tuple[float, float, float]:
        """Convert a canonical 3D point back into scan source coordinates."""
        if hasattr(self.transform, "to_source"):
            return self.transform.to_source(pt)
        res = self.transform.transform_points(np.asarray(pt, dtype=float), direction="inverse")[0]
        return (float(res[0]), float(res[1]), float(res[2]))

    def build_wall_candidate(
        self,
        wall_id: str,
        storey_id: str,
        storey_elevation: float,
        start_pt_canon: tuple[float, float, float],
        end_pt_canon: tuple[float, float, float],
        length_m: float,
        height_m: float,
        thickness_m: float,
        orientation_deg: float,
        local_frame: dict[str, Any],
        source_indices: np.ndarray,
        raw_source_points: np.ndarray,
        confidence: float,
        validation_report: dict[str, Any],
        semantic_evidence: dict[str, Any],
        geometric_evidence: dict[str, Any],
    ) -> BIMArchitecturalCandidate:
        start_src = self.to_source_point(start_pt_canon)
        end_src = self.to_source_point(end_pt_canon)

        geom_canon = {
            "start_point_m": [round(float(v), 4) for v in start_pt_canon],
            "end_point_m": [round(float(v), 4) for v in end_pt_canon],
            "length_m": round(length_m, 4),
            "height_m": round(height_m, 4),
            "thickness_m": round(thickness_m, 4),
            "orientation_deg": round(orientation_deg, 2),
            "local_frame": local_frame,
        }

        geom_src = {
            "start_point_source": [round(float(v), 4) for v in start_src],
            "end_point_source": [round(float(v), 4) for v in end_src],
            "source_units": self.transform.source_units,
        }

        idx_sample = source_indices[:100].tolist() if len(source_indices) > 0 else []

        return BIMArchitecturalCandidate(
            candidate_id=wall_id,
            semantic_type="WALL",
            geometry=geom_canon,
            canonical_geometry=geom_canon,
            source_geometry=geom_src,
            level={
                "storey_id": storey_id,
                "elevation_m": round(storey_elevation, 4),
            },
            source={
                "source_file": self.source_file,
                "point_count": len(source_indices),
                "point_indices_sample": idx_sample,
                "source_units": self.transform.source_units,
                "transform_id": self.transform.transform_id,
            },
            semantic_evidence=semantic_evidence,
            geometric_evidence=geometric_evidence,
            confidence=confidence,
            validation=validation_report,
        )

    def build_column_candidate(
        self,
        column_id: str,
        storey_id: str,
        storey_elevation: float,
        center_canon: tuple[float, float, float],
        width_m: float,
        depth_m: float,
        height_m: float,
        rotation_deg: float,
        profile_type: str,
        source_indices: np.ndarray,
        confidence: float,
        validation_report: dict[str, Any],
        semantic_evidence: dict[str, Any],
        geometric_evidence: dict[str, Any],
    ) -> BIMArchitecturalCandidate:
        center_src = self.to_source_point(center_canon)

        geom_canon = {
            "center_xyz_m": [round(float(v), 4) for v in center_canon],
            "width_m": round(width_m, 4),
            "depth_m": round(depth_m, 4),
            "height_m": round(height_m, 4),
            "rotation_deg": round(rotation_deg, 2),
            "profile_type": profile_type,
        }
        geom_src = {
            "center_xyz_source": [round(float(v), 4) for v in center_src],
            "source_units": self.transform.source_units,
        }

        return BIMArchitecturalCandidate(
            candidate_id=column_id,
            semantic_type="COLUMN",
            geometry=geom_canon,
            canonical_geometry=geom_canon,
            source_geometry=geom_src,
            level={
                "storey_id": storey_id,
                "elevation_m": round(storey_elevation, 4),
            },
            source={
                "source_file": self.source_file,
                "point_count": len(source_indices),
                "point_indices_sample": source_indices[:100].tolist() if len(source_indices) > 0 else [],
                "source_units": self.transform.source_units,
                "transform_id": self.transform.transform_id,
            },
            semantic_evidence=semantic_evidence,
            geometric_evidence=geometric_evidence,
            confidence=confidence,
            validation=validation_report,
        )

    def build_slab_candidate(
        self,
        slab_id: str,
        storey_id: str,
        slab_type: str,
        top_elevation_m: float,
        bottom_elevation_m: float,
        thickness_m: float,
        boundary_polygon_m: list[tuple[float, float]],
        area_m2: float,
        source_indices: np.ndarray,
        confidence: float,
        validation_report: dict[str, Any],
        semantic_evidence: dict[str, Any],
        geometric_evidence: dict[str, Any],
    ) -> BIMArchitecturalCandidate:
        geom_canon = {
            "top_elevation_m": round(top_elevation_m, 4),
            "bottom_elevation_m": round(bottom_elevation_m, 4),
            "thickness_m": round(thickness_m, 4),
            "area_m2": round(area_m2, 3),
            "boundary_polygon_m": [
                [round(float(pt[0]), 4), round(float(pt[1]), 4)] for pt in boundary_polygon_m
            ],
        }

        return BIMArchitecturalCandidate(
            candidate_id=slab_id,
            semantic_type=slab_type,
            geometry=geom_canon,
            canonical_geometry=geom_canon,
            source_geometry={"source_units": self.transform.source_units},
            level={
                "storey_id": storey_id,
                "elevation_m": round(top_elevation_m, 4),
            },
            source={
                "source_file": self.source_file,
                "point_count": len(source_indices),
                "point_indices_sample": source_indices[:100].tolist() if len(source_indices) > 0 else [],
                "source_units": self.transform.source_units,
                "transform_id": self.transform.transform_id,
            },
            semantic_evidence=semantic_evidence,
            geometric_evidence=geometric_evidence,
            confidence=confidence,
            validation=validation_report,
        )
