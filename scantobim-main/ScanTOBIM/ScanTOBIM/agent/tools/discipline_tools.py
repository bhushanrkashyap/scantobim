"""
discipline_tools.py  —  P3 Discipline Agent tools
──────────────────────────────────────────────────
8 tools available to each Discipline Agent (structural, MEP, fire_protection …).

Each tool wraps an orchestrator action and returns a serialisable dict so it
can be used directly as a GPT-4o tool-call response or called imperatively.

Safety contract (enforced before every create_* call):
  1. AssertNotSC1   — raises SafetyCategoryViolationError for SC1
  2. AssertSC2Gate  — raises SC2GateRequired for SC2 without approval
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import structlog

from agent.models import (
    Discipline,
    ElementInstruction,
    ElementType,
    GeometrySegment,
    SafetyCategory,
)

if TYPE_CHECKING:
    from agent.orchestrator import ScanToBIMOrchestrator

logger = structlog.get_logger()


# ── Tool 1-5: Element creation ────────────────────────────────────────────────


def _create_element(
    instruction: ElementInstruction,
    session_id: str,
    orchestrator: ScanToBIMOrchestrator,
) -> dict:
    """Shared dispatch logic for all element creation tools."""
    from agent.safety_gate import SafetyCategoryViolationError

    try:
        result = orchestrator._dispatch_to_revit(instruction, session_id)
        return {
            "success": result.success,
            "element_id": result.element_id,
            "instruction_id": result.instruction_id,
            "duration_ms": result.duration_ms,
            "error": result.error,
        }
    except SafetyCategoryViolationError as exc:
        logger.error("sc1_hard_block", instruction_id=instruction.instruction_id[:8])
        return {"success": False, "error": f"SC1_BLOCKED: {exc}"}
    except Exception as exc:
        return {"success": False, "error": str(exc)}


def create_wall(
    instruction: ElementInstruction,
    session_id: str,
    orchestrator: ScanToBIMOrchestrator,
) -> dict:
    """Create a wall element in Revit from a classified scan segment."""
    assert instruction.element_type == ElementType.WALL, "Expected WALL instruction"
    return _create_element(instruction, session_id, orchestrator)


def create_floor(
    instruction: ElementInstruction,
    session_id: str,
    orchestrator: ScanToBIMOrchestrator,
) -> dict:
    """Create a floor/slab element in Revit from a classified scan segment."""
    assert instruction.element_type == ElementType.FLOOR, "Expected FLOOR instruction"
    return _create_element(instruction, session_id, orchestrator)


def create_column(
    instruction: ElementInstruction,
    session_id: str,
    orchestrator: ScanToBIMOrchestrator,
) -> dict:
    """Create a structural column in Revit from a classified scan segment."""
    assert instruction.element_type == ElementType.COLUMN, "Expected COLUMN instruction"
    return _create_element(instruction, session_id, orchestrator)


def create_pipe(
    instruction: ElementInstruction,
    session_id: str,
    orchestrator: ScanToBIMOrchestrator,
) -> dict:
    """Create a pipe element in Revit (MEP discipline)."""
    assert instruction.element_type == ElementType.PIPE, "Expected PIPE instruction"
    return _create_element(instruction, session_id, orchestrator)


def create_beam(
    instruction: ElementInstruction,
    session_id: str,
    orchestrator: ScanToBIMOrchestrator,
) -> dict:
    """Create a structural beam in Revit from a classified scan segment."""
    assert instruction.element_type == ElementType.BEAM, "Expected BEAM instruction"
    return _create_element(instruction, session_id, orchestrator)


# --- Auto-generated discipline tools for all supported element types ---
def create_ceiling(instruction, session_id, orchestrator):
    assert instruction.element_type == ElementType.CEILING, "Expected CEILING instruction"
    return _create_element(instruction, session_id, orchestrator)


def create_stair(instruction, session_id, orchestrator):
    assert instruction.element_type == ElementType.STAIR, "Expected STAIR instruction"
    return _create_element(instruction, session_id, orchestrator)


def create_ramp(instruction, session_id, orchestrator):
    assert instruction.element_type == ElementType.RAMP, "Expected RAMP instruction"
    return _create_element(instruction, session_id, orchestrator)


def create_ladder(instruction, session_id, orchestrator):
    assert instruction.element_type == ElementType.LADDER, "Expected LADDER instruction"
    return _create_element(instruction, session_id, orchestrator)


def create_grating(instruction, session_id, orchestrator):
    assert instruction.element_type == ElementType.GRATING, "Expected GRATING instruction"
    return _create_element(instruction, session_id, orchestrator)


def create_overhead_crane(instruction, session_id, orchestrator):
    assert instruction.element_type == ElementType.OVERHEAD_CRANE, (
        "Expected OVERHEAD_CRANE instruction"
    )
    return _create_element(instruction, session_id, orchestrator)


def create_door(instruction, session_id, orchestrator):
    assert instruction.element_type == ElementType.DOOR, "Expected DOOR instruction"
    return _create_element(instruction, session_id, orchestrator)


def create_window(instruction, session_id, orchestrator):
    assert instruction.element_type == ElementType.WINDOW, "Expected WINDOW instruction"
    return _create_element(instruction, session_id, orchestrator)


def create_railing(instruction, session_id, orchestrator):
    assert instruction.element_type == ElementType.RAILING, "Expected RAILING instruction"
    return _create_element(instruction, session_id, orchestrator)


def create_hatch(instruction, session_id, orchestrator):
    assert instruction.element_type == ElementType.HATCH, "Expected HATCH instruction"
    return _create_element(instruction, session_id, orchestrator)


def create_trench(instruction, session_id, orchestrator):
    assert instruction.element_type == ElementType.TRENCH, "Expected TRENCH instruction"
    return _create_element(instruction, session_id, orchestrator)


def create_bund_wall(instruction, session_id, orchestrator):
    assert instruction.element_type == ElementType.BUND_WALL, "Expected BUND_WALL instruction"
    return _create_element(instruction, session_id, orchestrator)


def create_kerb(instruction, session_id, orchestrator):
    assert instruction.element_type == ElementType.KERB, "Expected KERB instruction"
    return _create_element(instruction, session_id, orchestrator)


def create_slab_opening(instruction, session_id, orchestrator):
    assert instruction.element_type == ElementType.SLAB_OPENING, "Expected SLAB_OPENING instruction"
    return _create_element(instruction, session_id, orchestrator)


def create_conduit(instruction, session_id, orchestrator):
    assert instruction.element_type == ElementType.CONDUIT, "Expected CONDUIT instruction"
    return _create_element(instruction, session_id, orchestrator)


def create_wire(instruction, session_id, orchestrator):
    assert instruction.element_type == ElementType.WIRE, "Expected WIRE instruction"
    return _create_element(instruction, session_id, orchestrator)


def create_duct(instruction, session_id, orchestrator):
    assert instruction.element_type == ElementType.DUCT, "Expected DUCT instruction"
    return _create_element(instruction, session_id, orchestrator)


def create_cable_tray(instruction, session_id, orchestrator):
    assert instruction.element_type == ElementType.CABLE_TRAY, "Expected CABLE_TRAY instruction"
    return _create_element(instruction, session_id, orchestrator)


def create_valve(instruction, session_id, orchestrator):
    assert instruction.element_type == ElementType.VALVE, "Expected VALVE instruction"
    return _create_element(instruction, session_id, orchestrator)


def create_safety_relief_valve(instruction, session_id, orchestrator):
    assert instruction.element_type == ElementType.SAFETY_RELIEF_VALVE, (
        "Expected SAFETY_RELIEF_VALVE instruction"
    )
    return _create_element(instruction, session_id, orchestrator)


def create_strainer(instruction, session_id, orchestrator):
    assert instruction.element_type == ElementType.STRAINER, "Expected STRAINER instruction"
    return _create_element(instruction, session_id, orchestrator)


def create_expansion_joint(instruction, session_id, orchestrator):
    assert instruction.element_type == ElementType.EXPANSION_JOINT, (
        "Expected EXPANSION_JOINT instruction"
    )
    return _create_element(instruction, session_id, orchestrator)


def create_fire_damper(instruction, session_id, orchestrator):
    assert instruction.element_type == ElementType.FIRE_DAMPER, "Expected FIRE_DAMPER instruction"
    return _create_element(instruction, session_id, orchestrator)


def create_penetration_seal(instruction, session_id, orchestrator):
    assert instruction.element_type == ElementType.PENETRATION_SEAL, (
        "Expected PENETRATION_SEAL instruction"
    )
    return _create_element(instruction, session_id, orchestrator)


def create_pipe_support(instruction, session_id, orchestrator):
    assert instruction.element_type == ElementType.PIPE_SUPPORT, "Expected PIPE_SUPPORT instruction"
    return _create_element(instruction, session_id, orchestrator)


def create_tank(instruction, session_id, orchestrator):
    assert instruction.element_type == ElementType.TANK, "Expected TANK instruction"
    return _create_element(instruction, session_id, orchestrator)


def create_pressure_vessel(instruction, session_id, orchestrator):
    assert instruction.element_type == ElementType.PRESSURE_VESSEL, (
        "Expected PRESSURE_VESSEL instruction"
    )
    return _create_element(instruction, session_id, orchestrator)


def create_heat_exchanger(instruction, session_id, orchestrator):
    assert instruction.element_type == ElementType.HEAT_EXCHANGER, (
        "Expected HEAT_EXCHANGER instruction"
    )
    return _create_element(instruction, session_id, orchestrator)


def create_pump(instruction, session_id, orchestrator):
    assert instruction.element_type == ElementType.PUMP, "Expected PUMP instruction"
    return _create_element(instruction, session_id, orchestrator)


def create_compressor(instruction, session_id, orchestrator):
    assert instruction.element_type == ElementType.COMPRESSOR, "Expected COMPRESSOR instruction"
    return _create_element(instruction, session_id, orchestrator)


def create_mechanical_equipment_receiver(instruction, session_id, orchestrator):
    """Create a mechanical equipment receiver (air/gas receiver tank) in Revit.

    Receivers are upright stocky cylinders classified dynamically via the
    element_type_rules.json config.  No hardcoded dimensions exist here —
    add or adjust receiver rules by editing that file.
    """
    assert instruction.element_type == ElementType.MECHANICAL_EQUIPMENT_RECEIVER, (
        "Expected MECHANICAL_EQUIPMENT_RECEIVER instruction"
    )
    return _create_element(instruction, session_id, orchestrator)


def create_hvac_equipment(instruction, session_id, orchestrator):
    assert instruction.element_type == ElementType.HVAC_EQUIPMENT, (
        "Expected HVAC_EQUIPMENT instruction"
    )
    return _create_element(instruction, session_id, orchestrator)


def create_sprinkler(instruction, session_id, orchestrator):
    assert instruction.element_type == ElementType.SPRINKLER, "Expected SPRINKLER instruction"
    return _create_element(instruction, session_id, orchestrator)


def create_drainage(instruction, session_id, orchestrator):
    assert instruction.element_type == ElementType.DRAINAGE, "Expected DRAINAGE instruction"
    return _create_element(instruction, session_id, orchestrator)


def create_fire_hydrant(instruction, session_id, orchestrator):
    assert instruction.element_type == ElementType.FIRE_HYDRANT, "Expected FIRE_HYDRANT instruction"
    return _create_element(instruction, session_id, orchestrator)


def create_deluge_valve(instruction, session_id, orchestrator):
    assert instruction.element_type == ElementType.DELUGE_VALVE, "Expected DELUGE_VALVE instruction"
    return _create_element(instruction, session_id, orchestrator)


def create_pressurizer(instruction, session_id, orchestrator):
    assert instruction.element_type == ElementType.PRESSURIZER, "Expected PRESSURIZER instruction"
    return _create_element(instruction, session_id, orchestrator)


def create_steam_generator(instruction, session_id, orchestrator):
    assert instruction.element_type == ElementType.STEAM_GENERATOR, (
        "Expected STEAM_GENERATOR instruction"
    )
    return _create_element(instruction, session_id, orchestrator)


def create_emergency_diesel_generator(instruction, session_id, orchestrator):
    assert instruction.element_type == ElementType.EMERGENCY_DIESEL_GENERATOR, (
        "Expected EMERGENCY_DIESEL_GENERATOR instruction"
    )
    return _create_element(instruction, session_id, orchestrator)


def create_seismic_isolator(instruction, session_id, orchestrator):
    assert instruction.element_type == ElementType.SEISMIC_ISOLATOR, (
        "Expected SEISMIC_ISOLATOR instruction"
    )
    return _create_element(instruction, session_id, orchestrator)


def create_containment_penetration(instruction, session_id, orchestrator):
    assert instruction.element_type == ElementType.CONTAINMENT_PENETRATION, (
        "Expected CONTAINMENT_PENETRATION instruction"
    )
    return _create_element(instruction, session_id, orchestrator)


def create_radiation_monitor(instruction, session_id, orchestrator):
    assert instruction.element_type == ElementType.RADIATION_MONITOR, (
        "Expected RADIATION_MONITOR instruction"
    )
    return _create_element(instruction, session_id, orchestrator)


def create_electrical_panel(instruction, session_id, orchestrator):
    assert instruction.element_type == ElementType.ELECTRICAL_PANEL, (
        "Expected ELECTRICAL_PANEL instruction"
    )
    return _create_element(instruction, session_id, orchestrator)


def create_transformer(instruction, session_id, orchestrator):
    assert instruction.element_type == ElementType.TRANSFORMER, "Expected TRANSFORMER instruction"
    return _create_element(instruction, session_id, orchestrator)


def create_switchgear(instruction, session_id, orchestrator):
    assert instruction.element_type == ElementType.SWITCHGEAR, "Expected SWITCHGEAR instruction"
    return _create_element(instruction, session_id, orchestrator)


def create_ups_system(instruction, session_id, orchestrator):
    assert instruction.element_type == ElementType.UPS_SYSTEM, "Expected UPS_SYSTEM instruction"
    return _create_element(instruction, session_id, orchestrator)


def create_junction_box(instruction, session_id, orchestrator):
    assert instruction.element_type == ElementType.JUNCTION_BOX, "Expected JUNCTION_BOX instruction"
    return _create_element(instruction, session_id, orchestrator)


def create_lighting_fitting(instruction, session_id, orchestrator):
    assert instruction.element_type == ElementType.LIGHTING_FITTING, (
        "Expected LIGHTING_FITTING instruction"
    )
    return _create_element(instruction, session_id, orchestrator)


def create_fire_alarm_panel(instruction, session_id, orchestrator):
    assert instruction.element_type == ElementType.FIRE_ALARM_PANEL, (
        "Expected FIRE_ALARM_PANEL instruction"
    )
    return _create_element(instruction, session_id, orchestrator)


def create_smoke_detector(instruction, session_id, orchestrator):
    assert instruction.element_type == ElementType.SMOKE_DETECTOR, (
        "Expected SMOKE_DETECTOR instruction"
    )
    return _create_element(instruction, session_id, orchestrator)


# ── Tool 6: Validate element geometry ────────────────────────────────────────


def validate_element(
    instruction: ElementInstruction,
    segment: GeometrySegment,
    tolerance_mm: float = 10.0,
) -> dict:
    """
    Run a deviation check between the element instruction and source segment.

    Returns  {element_id, deviation_mm, passed, tolerance_mm}
    """
    from agent.deviation_check import check

    result = check(instruction, segment, tolerance_mm=tolerance_mm)
    logger.info(
        "element_validated",
        element_id=result.element_id[:16],
        deviation_mm=result.deviation_mm,
        passed=result.passed,
    )
    return {
        "element_id": result.element_id,
        "segment_id": result.segment_id,
        "deviation_mm": result.deviation_mm,
        "passed": result.passed,
        "tolerance_mm": result.tolerance_mm,
    }


# ── Tool 7: Detect spatial clash ──────────────────────────────────────────────


def detect_clash(
    instruction_a: ElementInstruction,
    instruction_b: ElementInstruction,
    clearance_mm: float = 0.0,
) -> dict:
    """
    Simple AABB overlap test between two element instructions.

    Returns  {clash: bool, clash_type, clearance_mm, element_a_id, element_b_id}

    In production this uses the Open3D spatial registry R-tree.
    For the PoV we use bounding-box intersection.
    """

    def _overlap_1d(a_min, a_max, b_min, b_max) -> float:
        """Positive = overlap, negative = gap."""
        return min(a_max, b_max) - max(a_min, b_min)

    bb_a = instruction_a.bounding_box
    bb_b = instruction_b.bounding_box

    ox = _overlap_1d(bb_a.min_x, bb_a.max_x, bb_b.min_x, bb_b.max_x)
    oy = _overlap_1d(bb_a.min_y, bb_a.max_y, bb_b.min_y, bb_b.max_y)
    oz = _overlap_1d(bb_a.min_z, bb_a.max_z, bb_b.min_z, bb_b.max_z)

    hard_clash = ox > 0 and oy > 0 and oz > 0
    soft_clash = not hard_clash and ox > -clearance_mm and oy > -clearance_mm and oz > -clearance_mm

    gap_mm = -min(ox, oy, oz) if not hard_clash else 0.0

    clash_type = "hard" if hard_clash else ("soft" if soft_clash else "none")
    logger.info(
        "clash_checked",
        a=instruction_a.instruction_id[:8],
        b=instruction_b.instruction_id[:8],
        clash_type=clash_type,
        gap_mm=round(gap_mm, 1),
    )

    return {
        "clash": hard_clash or soft_clash,
        "clash_type": clash_type,
        "clearance_mm": round(gap_mm, 1),
        "element_a_id": instruction_a.instruction_id,
        "element_b_id": instruction_b.instruction_id,
    }


# ── Tool 8: Discipline status ─────────────────────────────────────────────────


def get_discipline_status(
    session_id: str,
    discipline: Discipline,
    orchestrator: ScanToBIMOrchestrator,
) -> dict:
    """
    Returns a summary of created/blocked/pending counts for one discipline.
    """
    instructions = orchestrator._instructions.get(session_id, [])
    disc_instrs = [i for i in instructions if i.discipline == discipline]

    results = orchestrator._results.get(session_id, [])
    ids = {i.instruction_id for i in disc_instrs}
    disc_results = [r for r in results if r.instruction_id in ids]

    created = sum(1 for r in disc_results if r.success)
    blocked = sum(1 for r in disc_results if not r.success)
    pending = len(disc_instrs) - len(disc_results)

    sc_counts = {
        "SC1": sum(1 for i in disc_instrs if i.safety_category == SafetyCategory.SC1),
        "SC2": sum(1 for i in disc_instrs if i.safety_category == SafetyCategory.SC2),
        "SC3": sum(1 for i in disc_instrs if i.safety_category == SafetyCategory.SC3),
        "NS": sum(1 for i in disc_instrs if i.safety_category == SafetyCategory.NS),
    }

    return {
        "session_id": session_id,
        "discipline": discipline.value,
        "total": len(disc_instrs),
        "created": created,
        "blocked": blocked,
        "pending": pending,
        "safety_breakdown": sc_counts,
    }
