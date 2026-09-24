"""Native Revit Element Instruction Generator for Scan-to-BIM.

Phase 3B Core Requirement (Section 28):
- Prefer native Revit element generation wherever possible:
    * Wall.Create (Wall)
    * Floor.Create (Floor / Slab / Ceiling)
    * FamilyInstance (Column, Door, Window, Valve, Equipment)
    * Pipe.Create (Pipe)
    * Duct.Create (Duct)
    * CableTray.Create (Cable Tray)
    * Opening APIs (Openings / Doors / Windows)
    * DirectShape fallback ONLY when no native parametric representation is possible (complex machinery/unknown).
- Maintains:
    * object_id
    * source_point_ids
    * semantic_label
    * confidence
    * model provenance
  in element parameters.
"""

from __future__ import annotations

import uuid
from typing import Any
import numpy as np
import structlog

from agent.models import (
    BoundingBox,
    Discipline,
    ElementInstruction,
    ElementType,
    Point3D,
    SafetyCategory,
)
from agent.phase3.mep.reconstructor import (
    ReconstructedCableTray,
    ReconstructedDuct,
    ReconstructedPipe,
    ReconstructedValve,
)
from agent.phase3.objects.reconstructor import ReconstructedObject
from agent.phase3.provenance.tracker import BIMArchitecturalCandidate

logger = structlog.get_logger(__name__)


class NativeRevitGenerator:
    """Translates reconstructed 3D architectural, MEP, and unknown objects into native Revit instructions."""

    def __init__(self, zone_id: str = "ZONE_MAIN", session_id: str = "PHASE_3B_SESSION") -> None:
        self.zone_id = zone_id
        self.session_id = session_id

    def from_architectural_candidate(
        self,
        candidate: BIMArchitecturalCandidate,
    ) -> ElementInstruction:
        """Create native instruction for walls, slabs, floors, or columns."""
        geom = candidate.canonical_geometry or candidate.geometry
        sem_type = candidate.semantic_type.upper()
        storey_id = candidate.level.get("storey_id", "LEVEL_00") if isinstance(candidate.level, dict) else "LEVEL_00"

        if sem_type == "WALL":
            elem_type = ElementType.WALL
            discipline = Discipline.ARCHITECTURAL
            start_m = geom.get("start_point_m", geom.get("start_point_canonical", [0, 0, 0]))
            end_m = geom.get("end_point_m", geom.get("end_point_canonical", [1, 0, 0]))
            height_mm = float(geom.get("height_m", 3.0)) * 1000.0
            thickness_mm = float(geom.get("thickness_m", 0.2)) * 1000.0
            params = {
                "start_point": [c * 1000.0 for c in start_m],
                "end_point": [c * 1000.0 for c in end_m],
                "height_mm": height_mm,
                "thickness_mm": thickness_mm,
                "orientation_deg": geom.get("orientation_deg", 0.0),
                "storey_id": storey_id,
            }
        elif sem_type in ("FLOOR", "SLAB"):
            elem_type = ElementType.FLOOR
            discipline = Discipline.STRUCTURAL
            poly = geom.get("boundary_polygon_m", geom.get("boundary_polygon_canonical", [[0, 0], [1, 0], [1, 1], [0, 1]]))
            thickness_mm = float(geom.get("thickness_m", 0.25)) * 1000.0
            elev_mm = float(geom.get("top_elevation_m", geom.get("top_elevation_canonical", 0.0))) * 1000.0
            params = {
                "boundary_polygon": [[pt[0] * 1000.0, pt[1] * 1000.0] for pt in poly],
                "elevation_mm": elev_mm,
                "thickness_mm": thickness_mm,
                "storey_id": storey_id,
            }
        elif sem_type == "CEILING":
            elem_type = ElementType.CEILING
            discipline = Discipline.ARCHITECTURAL
            poly = geom.get("boundary_polygon_m", geom.get("boundary_polygon_canonical", [[0, 0], [1, 0], [1, 1], [0, 1]]))
            thickness_mm = float(geom.get("thickness_m", 0.15)) * 1000.0
            elev_mm = float(geom.get("top_elevation_m", geom.get("top_elevation_canonical", 2.8))) * 1000.0
            params = {
                "boundary_polygon": [[pt[0] * 1000.0, pt[1] * 1000.0] for pt in poly],
                "elevation_mm": elev_mm,
                "thickness_mm": thickness_mm,
                "storey_id": storey_id,
            }
        elif sem_type == "COLUMN":
            elem_type = ElementType.COLUMN
            discipline = Discipline.STRUCTURAL
            ctr_m = geom.get("center_m", geom.get("center_canonical", [0, 0, 0]))
            params = {
                "center_point": [c * 1000.0 for c in ctr_m],
                "width_mm": float(geom.get("width_m", 0.4)) * 1000.0,
                "depth_mm": float(geom.get("depth_m", 0.4)) * 1000.0,
                "height_mm": float(geom.get("height_m", 3.0)) * 1000.0,
                "rotation_deg": float(geom.get("rotation_deg", 0.0)),
                "storey_id": storey_id,
            }
        else:
            elem_type = ElementType.GENERIC_MODEL
            discipline = Discipline.CIVIL
            params = geom

        # Common metadata
        pt_count = candidate.source.get("point_count", 0) if isinstance(candidate.source, dict) else 0
        params.update({
            "object_id": candidate.candidate_id,
            "semantic_label": candidate.semantic_type,
            "confidence": candidate.confidence,
            "source_point_count": pt_count,
            "model_provenance": candidate.semantic_evidence,
        })

        centroid_xyz = Point3D(x=0.0, y=0.0, z=0.0)
        return ElementInstruction(
            instruction_id=str(uuid.uuid4()),
            segment_id=str(uuid.uuid4()),
            zone_id=self.zone_id,
            element_type=elem_type,
            discipline=discipline,
            safety_category=SafetyCategory.NS,
            bounding_box=BoundingBox(
                min_x=-10000.0,
                min_y=-10000.0,
                min_z=-1000.0,
                max_x=10000.0,
                max_y=10000.0,
                max_z=10000.0,
            ),
            centroid=centroid_xyz,
            parameters=params,
        )

    def from_pipe(self, pipe: ReconstructedPipe) -> ElementInstruction:
        """Create native Revit Pipe.Create instruction."""
        start_mm = [c * 1000.0 for c in pipe.start_point_m]
        end_mm = [c * 1000.0 for c in pipe.end_point_m]
        mid_mm = [(s + e) / 2.0 for s, e in zip(start_mm, end_mm)]

        params = {
            "start_point": start_mm,
            "end_point": end_mm,
            "diameter_mm": float(pipe.diameter_m * 1000.0),
            "length_mm": float(pipe.length_m * 1000.0),
            "slope": pipe.slope,
            "system_type": "Hydronic Supply",
            "object_id": pipe.pipe_id,
            "semantic_label": "PIPE",
            "confidence": pipe.confidence,
            "residual_rmse_mm": float(pipe.residual_rmse_m * 1000.0),
            "source_point_count": len(pipe.source_point_indices),
        }

        min_x = min(start_mm[0], end_mm[0]) - 100.0
        min_y = min(start_mm[1], end_mm[1]) - 100.0
        min_z = min(start_mm[2], end_mm[2]) - 100.0
        max_x = max(start_mm[0], end_mm[0]) + 100.0
        max_y = max(start_mm[1], end_mm[1]) + 100.0
        max_z = max(start_mm[2], end_mm[2]) + 100.0

        return ElementInstruction(
            instruction_id=str(uuid.uuid4()),
            segment_id=str(uuid.uuid4()),
            zone_id=self.zone_id,
            element_type=ElementType.PIPE,
            discipline=Discipline.MEP,
            safety_category=SafetyCategory.NS,
            bounding_box=BoundingBox(
                min_x=min_x,
                min_y=min_y,
                min_z=min_z,
                max_x=max_x,
                max_y=max_y,
                max_z=max_z,
            ),
            centroid=Point3D(x=mid_mm[0], y=mid_mm[1], z=mid_mm[2]),
            parameters=params,
            revit_family_hint="Standard Pipe",
        )

    def from_duct(self, duct: ReconstructedDuct) -> ElementInstruction:
        """Create native Revit Duct.Create instruction."""
        start_mm = [c * 1000.0 for c in duct.start_point_m]
        end_mm = [c * 1000.0 for c in duct.end_point_m]
        mid_mm = [(s + e) / 2.0 for s, e in zip(start_mm, end_mm)]

        params = {
            "start_point": start_mm,
            "end_point": end_mm,
            "width_mm": float(duct.width_m * 1000.0),
            "height_mm": float(duct.height_m * 1000.0),
            "length_mm": float(duct.length_m * 1000.0),
            "system_type": "Supply Air",
            "object_id": duct.duct_id,
            "semantic_label": "DUCT",
            "confidence": duct.confidence,
            "source_point_count": len(duct.source_point_indices),
        }

        min_x = min(start_mm[0], end_mm[0]) - 200.0
        min_y = min(start_mm[1], end_mm[1]) - 200.0
        min_z = min(start_mm[2], end_mm[2]) - 200.0
        max_x = max(start_mm[0], end_mm[0]) + 200.0
        max_y = max(start_mm[1], end_mm[1]) + 200.0
        max_z = max(start_mm[2], end_mm[2]) + 200.0

        return ElementInstruction(
            instruction_id=str(uuid.uuid4()),
            segment_id=str(uuid.uuid4()),
            zone_id=self.zone_id,
            element_type=ElementType.DUCT,
            discipline=Discipline.MEP,
            safety_category=SafetyCategory.NS,
            bounding_box=BoundingBox(
                min_x=min_x,
                min_y=min_y,
                min_z=min_z,
                max_x=max_x,
                max_y=max_y,
                max_z=max_z,
            ),
            centroid=Point3D(x=mid_mm[0], y=mid_mm[1], z=mid_mm[2]),
            parameters=params,
            revit_family_hint="Rectangular Duct",
        )

    def from_cable_tray(self, tray: ReconstructedCableTray) -> ElementInstruction:
        """Create native Revit CableTray.Create instruction."""
        start_mm = [c * 1000.0 for c in tray.start_point_m]
        end_mm = [c * 1000.0 for c in tray.end_point_m]
        mid_mm = [(s + e) / 2.0 for s, e in zip(start_mm, end_mm)]

        params = {
            "start_point": start_mm,
            "end_point": end_mm,
            "width_mm": float(tray.width_m * 1000.0),
            "depth_mm": float(tray.depth_m * 1000.0),
            "length_mm": float(tray.length_m * 1000.0),
            "object_id": tray.tray_id,
            "semantic_label": "CABLE_TRAY",
            "confidence": tray.confidence,
            "source_point_count": len(tray.source_point_indices),
        }

        min_x = min(start_mm[0], end_mm[0]) - 150.0
        min_y = min(start_mm[1], end_mm[1]) - 150.0
        min_z = min(start_mm[2], end_mm[2]) - 150.0
        max_x = max(start_mm[0], end_mm[0]) + 150.0
        max_y = max(start_mm[1], end_mm[1]) + 150.0
        max_z = max(start_mm[2], end_mm[2]) + 150.0

        return ElementInstruction(
            instruction_id=str(uuid.uuid4()),
            segment_id=str(uuid.uuid4()),
            zone_id=self.zone_id,
            element_type=ElementType.CABLE_TRAY,
            discipline=Discipline.ELECTRICAL,
            safety_category=SafetyCategory.NS,
            bounding_box=BoundingBox(
                min_x=min_x,
                min_y=min_y,
                min_z=min_z,
                max_x=max_x,
                max_y=max_y,
                max_z=max_z,
            ),
            centroid=Point3D(x=mid_mm[0], y=mid_mm[1], z=mid_mm[2]),
            parameters=params,
        )

    def from_valve(self, valve: ReconstructedValve) -> ElementInstruction:
        """Create native Revit Valve FamilyInstance instruction."""
        ctr_mm = [c * 1000.0 for c in valve.center_m]
        params = {
            "center_point": ctr_mm,
            "length_mm": float(valve.length_m * 1000.0),
            "width_mm": float(valve.width_m * 1000.0),
            "height_mm": float(valve.height_m * 1000.0),
            "connected_pipe_id": valve.connected_pipe_id,
            "object_id": valve.valve_id,
            "semantic_label": "VALVE",
            "confidence": valve.confidence,
            "source_point_count": len(valve.source_point_indices),
        }

        return ElementInstruction(
            instruction_id=str(uuid.uuid4()),
            segment_id=str(uuid.uuid4()),
            zone_id=self.zone_id,
            element_type=ElementType.VALVE,
            discipline=Discipline.MEP,
            safety_category=SafetyCategory.NS,
            bounding_box=BoundingBox(
                min_x=ctr_mm[0] - 250.0,
                min_y=ctr_mm[1] - 250.0,
                min_z=ctr_mm[2] - 250.0,
                max_x=ctr_mm[0] + 250.0,
                max_y=ctr_mm[1] + 250.0,
                max_z=ctr_mm[2] + 250.0,
            ),
            centroid=Point3D(x=ctr_mm[0], y=ctr_mm[1], z=ctr_mm[2]),
            parameters=params,
            revit_family_hint="Ball Valve",
        )

    def from_reconstructed_object(self, obj: ReconstructedObject) -> ElementInstruction:
        """Create Revit instruction for generic, door, window, or UNKNOWN objects."""
        ctr_mm = [c * 1000.0 for c in obj.centroid_m]
        min_mm = [c * 1000.0 for c in obj.bbox_min_m]
        max_mm = [c * 1000.0 for c in obj.bbox_max_m]

        lbl = obj.semantic_label.upper()
        if "DOOR" in lbl:
            elem_type = ElementType.DOOR
            discipline = Discipline.ARCHITECTURAL
            family_hint = "Single-Flush"
        elif "WINDOW" in lbl:
            elem_type = ElementType.WINDOW
            discipline = Discipline.ARCHITECTURAL
            family_hint = "Fixed Window"
        elif "BEAM" in lbl:
            elem_type = ElementType.BEAM
            discipline = Discipline.STRUCTURAL
            family_hint = "W-Wide Flange"
        elif "EQUIPMENT" in lbl or "MACHINERY" in lbl or "PUMP" in lbl:
            elem_type = ElementType.TANK
            discipline = Discipline.MEP
            family_hint = "Mechanical Equipment"
        else:
            elem_type = ElementType.GENERIC_MODEL
            discipline = Discipline.CIVIL
            family_hint = "DirectShape_Unknown_Object"

        params = {
            "object_id": obj.object_id,
            "semantic_label": obj.semantic_label,
            "level1_category": obj.level1_category,
            "decision_status": obj.decision_status,
            "confidence": obj.confidence,
            "dimensions_mm": {k: float(v * 1000.0) for k, v in obj.dimensions_m.items()},
            "obb_center_mm": [c * 1000.0 for c in obj.obb_center_m],
            "obb_extents_mm": [c * 1000.0 for c in obj.obb_extents_m],
            "rotation_matrix": obj.obb_rotation_matrix,
            "surface_area_m2": obj.surface_area_m2,
            "volume_m3": obj.volume_m3,
            "source_point_count": len(obj.source_point_indices),
            "confidence_reason_codes": obj.confidence_reason_codes,
            "topology_relations": obj.topology_relations,
        }

        min_x = min_mm[0]
        min_y = min_mm[1]
        min_z = min_mm[2]
        max_x = max(min_x + 10.0, max_mm[0])
        max_y = max(min_y + 10.0, max_mm[1])
        max_z = max(min_z + 10.0, max_mm[2])

        return ElementInstruction(
            instruction_id=str(uuid.uuid4()),
            segment_id=str(uuid.uuid4()),
            zone_id=self.zone_id,
            element_type=elem_type,
            discipline=discipline,
            safety_category=SafetyCategory.NS,
            bounding_box=BoundingBox(
                min_x=min_x,
                min_y=min_y,
                min_z=min_z,
                max_x=max_x,
                max_y=max_y,
                max_z=max_z,
            ),
            centroid=Point3D(x=ctr_mm[0], y=ctr_mm[1], z=ctr_mm[2]),
            parameters=params,
            revit_family_hint=family_hint,
        )
