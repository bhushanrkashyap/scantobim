"""IFC 4.x export for Scan-to-BIM sessions.

Produces an IFC 4 (ISO 16739-1:2018) file from the classified element
instructions so the model can be consumed by any IFC-compliant tool
(Navisworks, Solibri, ACC, BIMcollab, etc.) without requiring Revit.

ifcopenshell is an optional dependency — if not installed, export returns
a structured error so the rest of the pipeline still works.

Usage (Python):
    from agent.tools.ifc_export import export_session_to_ifc
    ifc_bytes = export_session_to_ifc(session_id, instructions, results)

Usage (API):
    GET /sessions/{session_id}/export/ifc → application/octet-stream
"""

from __future__ import annotations

import io
import uuid
from datetime import datetime, timezone
from typing import TYPE_CHECKING

import structlog

if TYPE_CHECKING:
    from agent.models import ActionResult, ElementInstruction

logger = structlog.get_logger()

# ── IFC category → IFC class mapping ─────────────────────────────────────────

_IFC_CLASS_MAP: dict[str, str] = {
    # Structural
    "wall": "IfcWall",
    "floor": "IfcSlab",
    "ceiling": "IfcCovering",
    "column": "IfcColumn",
    "beam": "IfcBeam",
    "stair": "IfcStair",
    "ramp": "IfcRamp",
    "ladder": "IfcBuildingElementProxy",
    "grating": "IfcPlate",
    "overhead_crane": "IfcTransportElement",
    # Architectural
    "door": "IfcDoor",
    "window": "IfcWindow",
    "railing": "IfcRailing",
    "hatch": "IfcOpeningElement",
    # Civil
    "trench": "IfcCivilElement",
    "bund_wall": "IfcWall",
    "kerb": "IfcCivilElement",
    "slab_opening": "IfcOpeningElement",
    # MEP — piping / inline
    "pipe": "IfcPipeSegment",
    "conduit": "IfcCableCarrierSegment",
    "duct": "IfcDuctSegment",
    "cable_tray": "IfcCableCarrierSegment",
    "valve": "IfcValve",
    "safety_relief_valve": "IfcValve",
    "strainer": "IfcPipeFitting",
    "expansion_joint": "IfcPipeFitting",
    "fire_damper": "IfcDamper",
    "penetration_seal": "IfcBuildingElementProxy",
    "pipe_support": "IfcBuildingElementProxy",
    # MEP — vessels / equipment
    "tank": "IfcTank",
    "pressure_vessel": "IfcTank",
    "heat_exchanger": "IfcHeatExchanger",
    "pump": "IfcPump",
    "compressor": "IfcCompressor",
    "hvac_equipment": "IfcUnitaryEquipment",
    "sprinkler": "IfcFireSuppressionTerminal",
    "drainage": "IfcSanitaryTerminal",
    "fire_hydrant": "IfcFireSuppressionTerminal",
    "deluge_valve": "IfcValve",
    # Nuclear
    "pressurizer": "IfcTank",
    "steam_generator": "IfcHeatExchanger",
    "emergency_diesel_generator": "IfcElectricGenerator",
    "seismic_isolator": "IfcBuildingElementProxy",
    "containment_penetration": "IfcOpeningElement",
    "radiation_monitor": "IfcSensor",
    # Electrical
    "electrical_panel": "IfcDistributionControlElement",
    "transformer": "IfcTransformer",
    "switchgear": "IfcSwitchingDevice",
    "ups_system": "IfcElectricDistributionBoard",
    "junction_box": "IfcJunctionBox",
    "lighting_fitting": "IfcLightFixture",
    # Fire / life safety
    "fire_alarm_panel": "IfcAlarm",
    "smoke_detector": "IfcAlarm",
}

from agent.tools.detection_config import (
    MIN_ELEMENT_DIM_MM,
    SLAB_THICKNESS_MM,
    WALL_THICKNESS_MM,
)

# Planar element categories. A scanned surface is ~zero-thickness on one axis, which
# gives a degenerate (zero-volume) bounding box that Revit will not render — this is why
# walls disappear while 3D elements (pipes/equipment) show. Walls are thin on a horizontal
# axis; slabs (floor/ceiling/grating) are thin on Z. Give the thin axis a real thickness.
_WALL_TYPES = frozenset({"wall", "bund_wall"})
_SLAB_TYPES = frozenset({"floor", "ceiling", "grating"})


def _solid_dims(element_type: str, dx: float, dy: float, dz: float) -> tuple[float, float, float]:
    """Clamp bounding-box dimensions (mm) so planar elements render as solids.
    Only ever enlarges a dimension to a thickness floor — never shrinks a real one."""
    if element_type in _WALL_TYPES:
        dx, dy = max(dx, WALL_THICKNESS_MM), max(dy, WALL_THICKNESS_MM)
    elif element_type in _SLAB_TYPES:
        dz = max(dz, SLAB_THICKNESS_MM)
    return (max(dx, MIN_ELEMENT_DIM_MM), max(dy, MIN_ELEMENT_DIM_MM), max(dz, MIN_ELEMENT_DIM_MM))


def _ifc_class_for(element_type: str) -> str:
    return _IFC_CLASS_MAP.get(element_type, "IfcBuildingElementProxy")


# ── Main export function ───────────────────────────────────────────────────────


def export_session_to_ifc(
    session_id: str,
    instructions: list[ElementInstruction],
    results: list[ActionResult] | None = None,
    site_name: str = "ScanToBIM Export",
) -> bytes:
    """Export a session's classified elements to an IFC 4 file.

    Each ElementInstruction becomes one IfcProduct with:
    - IfcLocalPlacement derived from the segment centroid
    - IfcBoundingBox representation from the bounding box
    - Pset_ScanToBIM property set with safety category, discipline, audit ID

    Returns the IFC file as raw bytes (UTF-8 encoded STEP Physical File).
    Raises ImportError with a helpful message if ifcopenshell is not installed.
    """
    try:
        import ifcopenshell  # type: ignore
        import ifcopenshell.api  # type: ignore
    except ImportError as exc:
        raise ImportError(
            "ifcopenshell is required for IFC export. Install it with: pip install ifcopenshell"
        ) from exc

    logger.info("ifc_export_start", session_id=session_id[:8], elements=len(instructions))

    # ── Create IFC model ──────────────────────────────────────────────────────
    model = ifcopenshell.file(schema="IFC4")

    # Project hierarchy
    project = ifcopenshell.api.run(
        "root.create_entity", model, ifc_class="IfcProject", name=f"ScanToBIM-{session_id[:8]}"
    )
    ifcopenshell.api.run("unit.assign_unit", model, length={"is_metric": True, "raw": "MILLIMETRE"})

    site = ifcopenshell.api.run("root.create_entity", model, ifc_class="IfcSite", name=site_name)
    building = ifcopenshell.api.run(
        "root.create_entity", model, ifc_class="IfcBuilding", name="Main Building"
    )
    storey = ifcopenshell.api.run(
        "root.create_entity", model, ifc_class="IfcBuildingStorey", name="Level 1"
    )

    ifcopenshell.api.run("aggregate.assign_object", model, product=site, relating_object=project)
    ifcopenshell.api.run("aggregate.assign_object", model, product=building, relating_object=site)
    ifcopenshell.api.run("aggregate.assign_object", model, product=storey, relating_object=building)

    context = ifcopenshell.api.run(
        "context.add_context",
        model,
        context_type="Model",
        context_identifier="Body",
        target_view="MODEL_VIEW",
    )

    # Build result lookup: segment_id → element_id
    result_map: dict[str, str] = {}
    if results:
        for r in results:
            if r.success and r.element_id:
                result_map[r.instruction_id] = r.element_id

    # ── Add one product per instruction ──────────────────────────────────────
    for instr in instructions:
        ifc_class = _ifc_class_for(instr.element_type.value)
        bb = instr.bounding_box
        cx = (bb.min_x + bb.max_x) / 2.0
        cy = (bb.min_y + bb.max_y) / 2.0
        cz = (bb.min_z + bb.max_z) / 2.0

        # Local placement at centroid
        placement = model.createIfcAxis2Placement3D(
            model.createIfcCartesianPoint((cx, cy, cz)),
        )
        local_placement = model.createIfcLocalPlacement(None, placement)

        # Bounding box representation — clamp planar elements (walls/slabs) to a real
        # thickness so the box is non-degenerate; a zero-thickness box does not render.
        dx, dy, dz = _solid_dims(
            instr.element_type.value,
            bb.max_x - bb.min_x,
            bb.max_y - bb.min_y,
            bb.max_z - bb.min_z,
        )
        bbox_item = model.createIfcBoundingBox(
            model.createIfcCartesianPoint((-dx / 2.0, -dy / 2.0, -dz / 2.0)),
            dx,
            dy,
            dz,
        )
        shape_rep = model.createIfcShapeRepresentation(context, "Body", "BoundingBox", [bbox_item])
        product_def = model.createIfcProductDefinitionShape(None, None, [shape_rep])

        # Create the product
        product = model.create_entity(
            ifc_class,
            GlobalId=ifcopenshell.guid.new(),
            OwnerHistory=None,
            Name=instr.element_type.value,
            Description=f"seg:{instr.segment_id[:8]} zone:{instr.zone_id}",
            ObjectPlacement=local_placement,
            Representation=product_def,
        )

        # Assign to storey
        ifcopenshell.api.run(
            "spatial.assign_container", model, product=product, relating_structure=storey
        )

        # Pset_ScanToBIM — compliance properties
        revit_id = result_map.get(instr.instruction_id, "")
        pset = ifcopenshell.api.run("pset.add_pset", model, product=product, name="Pset_ScanToBIM")
        ifcopenshell.api.run(
            "pset.edit_pset",
            model,
            pset=pset,
            properties={
                "ScanToBIM_SafetyCategory": instr.safety_category.value,
                "ScanToBIM_Discipline": instr.discipline.value,
                "ScanToBIM_DSEARZone": instr.dsear_zone.value,
                "ScanToBIM_SessionId": session_id,
                "ScanToBIM_SegmentId": instr.segment_id,
                "ScanToBIM_ZoneId": instr.zone_id,
                "ScanToBIM_RevitElementId": revit_id,
                "ScanToBIM_ExportedAt": datetime.now(timezone.utc).isoformat(),
            },
        )

    # ── Serialise to bytes ────────────────────────────────────────────────────
    buf = io.BytesIO()
    model.write(buf)
    ifc_bytes = buf.getvalue()

    logger.info(
        "ifc_export_complete",
        session_id=session_id[:8],
        elements=len(instructions),
        size_kb=round(len(ifc_bytes) / 1024, 1),
    )
    return ifc_bytes


# ── Fallback STEP file builder (no ifcopenshell) ───────────────────────────────


def export_session_to_ifc_minimal(
    session_id: str,
    instructions: list[ElementInstruction],
) -> bytes:
    """Minimal STEP Physical File fallback — no ifcopenshell required.

    Produces a valid IFC4 STEP file header with element stubs.
    Not fully schema-valid but readable by most IFC viewers for inspection.
    Only use this as a last resort — prefer export_session_to_ifc() with
    ifcopenshell installed.
    """
    now = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%S")
    lines: list[str] = [
        "ISO-10303-21;",
        "HEADER;",
        f"FILE_DESCRIPTION(('ScanToBIM IFC4 Export','Session:{session_id}'),'2;1');",
        f"FILE_NAME('{session_id}.ifc','{now}',('ScanToBIM Agent'),('ScanToBIM'),",
        "  'ifcopenshell fallback','ScanToBIM v0.1','');",
        "FILE_SCHEMA(('IFC4'));",
        "ENDSEC;",
        "DATA;",
    ]

    for idx, instr in enumerate(instructions, start=1):
        ifc_class = _ifc_class_for(instr.element_type.value)
        gid = str(uuid.uuid4()).replace("-", "")[:22]
        bb = instr.bounding_box
        cx = round((bb.min_x + bb.max_x) / 2.0, 1)
        cy = round((bb.min_y + bb.max_y) / 2.0, 1)
        cz = round((bb.min_z + bb.max_z) / 2.0, 1)

        p_id = 1000 + idx * 10
        ax_id = p_id + 1
        pl_id = p_id + 2
        lp_id = p_id + 3
        bb_id = p_id + 4
        sr_id = p_id + 5
        pd_id = p_id + 6
        el_id = p_id + 7

        lines += [
            f"#{p_id}=IFCCARTESIANPOINT(({cx},{cy},{cz}));",
            f"#{ax_id}=IFCAXIS2PLACEMENT3D(#{p_id},$,$);",
            f"#{pl_id}=IFCLOCALPLACEMENT($,#{ax_id});",
            f"#{bb_id}=IFCBOUNDINGBOX(IFCCARTESIANPOINT((0.,0.,0.)),",
            f"  {round(bb.max_x - bb.min_x, 1)},{round(bb.max_y - bb.min_y, 1)},{round(bb.max_z - bb.min_z, 1)});",
            f"#{sr_id}=IFCSHAPEREPRESENTATION($,'Body','BoundingBox',(#{bb_id}));",
            f"#{pd_id}=IFCPRODUCTDEFINITIONSHAPE($,$,(#{sr_id}));",
            (
                f"#{el_id}={ifc_class.upper()}('{gid}',$,'{instr.element_type.value}',"
                f"'seg:{instr.segment_id[:8]}',$,#{pl_id},#{pd_id},$);"
            ),
        ]

    lines += ["ENDSEC;", "END-ISO-10303-21;"]
    return "\n".join(lines).encode("utf-8")
