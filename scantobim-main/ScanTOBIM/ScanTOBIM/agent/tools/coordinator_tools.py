"""
coordinator_tools.py  —  P3 Site Coordinator Agent
───────────────────────────────────────────────────
7 tools the Site Coordinator Agent calls to drive the full pipeline.

The Coordinator owns the session lifecycle:
  decompose_zones → dispatch_to_discipline → aggregate → CDE transition

These are plain functions (not LLM tool-call wrappers) so they work both
with GPT-4o tool-calling and with direct orchestrator calls.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import structlog

from agent.models import (
    ClashReport,
    ElementInstruction,
    ElementType,
    GeometrySegment,
)

if TYPE_CHECKING:
    from agent.cde_state import CDEStateService
    from agent.ncr import NCRService
    from agent.orchestrator import ScanToBIMOrchestrator
    from agent.safety_gate import SafetyGateService

logger = structlog.get_logger()


# ── Tool 1: Decompose site into processing zones ───────────────────────────────


def decompose_zones(
    segments: list[GeometrySegment],
    max_zones: int = 10,
) -> dict[str, list[GeometrySegment]]:
    """
    Partition segments by zone_id.

    Returns  {zone_id: [segments]} — at most max_zones zones.
    If more zones are present, overflow segments are merged into 'zone-overflow'.
    """
    zone_map: dict[str, list[GeometrySegment]] = {}
    for seg in segments:
        zone_map.setdefault(seg.zone_id, []).append(seg)

    if len(zone_map) > max_zones:
        overflow = {}
        for zid, segs in list(zone_map.items())[max_zones:]:
            overflow.setdefault("zone-overflow", []).extend(segs)
            del zone_map[zid]
        zone_map.update(overflow)

    logger.info(
        "zones_decomposed",
        zones=list(zone_map.keys()),
        total_segments=len(segments),
    )
    return zone_map


# ── Tool 2: Dispatch segments to discipline agents ─────────────────────────────


def dispatch_to_discipline(
    zone_id: str,
    segments: list[GeometrySegment],
) -> dict[str, list[GeometrySegment]]:
    """
    Split segments within a zone by the discipline each will be created under.

    Returns  {discipline_value: [segments]}
    The classification step decides final discipline; here we pre-sort by
    shape heuristic so discipline agents receive a focused workload.
    """
    from agent.classifier import classify_discipline, classify_element_type

    by_discipline: dict[str, list[GeometrySegment]] = {}
    for seg in segments:
        el_type = classify_element_type(seg)
        disc = classify_discipline(el_type)
        by_discipline.setdefault(disc.value, []).append(seg)

    logger.info(
        "segments_dispatched",
        zone_id=zone_id,
        dispatch_summary={d: len(s) for d, s in by_discipline.items()},
    )
    return by_discipline


# ── Tool 3: Check zone completion status ───────────────────────────────────────


def check_zone_completion(
    zone_id: str,
    orchestrator: ScanToBIMOrchestrator,
    session_id: str,
) -> dict:
    """
    Returns completion stats for a zone:
      {zone_id, total, created, blocked, pending, complete}
    """
    instructions = orchestrator._instructions.get(session_id, [])
    zone_instrs = [i for i in instructions if i.zone_id == zone_id]

    results = orchestrator._results.get(session_id, [])
    zone_res = [
        r for r in results if any(i.instruction_id == r.instruction_id for i in zone_instrs)
    ]

    created = sum(1 for r in zone_res if r.success)
    blocked = sum(1 for r in zone_res if not r.success)
    pending = len(zone_instrs) - len(zone_res)
    complete = pending == 0

    return {
        "zone_id": zone_id,
        "total": len(zone_instrs),
        "created": created,
        "blocked": blocked,
        "pending": pending,
        "complete": complete,
    }


# ── Tool 4: Escalate a safety gate ────────────────────────────────────────────


def escalate_safety_gate(
    gate_id: str,
    reason: str,
    safety_gate: SafetyGateService,
    escalation_upn: str = "project.director@example.com",
) -> dict:
    """
    Mark an SC2 gate as escalated when the 24h decision window has expired.
    Logs an audit event and notifies the escalation UPN.

    Returns {gate_id, escalated, escalation_upn, reason}
    """
    gate = safety_gate.get_gate(gate_id)
    if not gate:
        return {"gate_id": gate_id, "escalated": False, "error": "gate not found"}

    logger.warning(
        "safety_gate_escalated",
        gate_id=gate_id[:12],
        reason=reason,
        escalation_upn=escalation_upn,
    )

    # In production this would trigger a Teams/email notification
    return {
        "gate_id": gate_id,
        "escalated": True,
        "escalation_upn": escalation_upn,
        "reason": reason,
    }


# ── Tool 5: Get full session status ────────────────────────────────────────────


def get_session_status(
    session_id: str,
    orchestrator: ScanToBIMOrchestrator,
) -> dict:
    """
    Snapshot of session state including pending gates and safety breakdown.
    """
    return orchestrator.get_session_summary(session_id)


# ── Tool 6: Raise an NCR ───────────────────────────────────────────────────────


def raise_ncr(
    session_id: str,
    element_id: str,
    description: str,
    severity: str,
    ncr_service: NCRService,
    raised_by: str = "coordinator",
) -> dict:
    """
    Raise a Non-Conformance Report for a failed or non-compliant element.

    severity: 'CRITICAL' | 'MAJOR' | 'MINOR'
    Returns the created NCR record dict.
    """
    ncr = ncr_service.raise_ncr(
        session_id=session_id,
        element_id=element_id,
        description=description,
        severity=severity,
        raised_by=raised_by,
    )
    return ncr.model_dump()


# ── Tool 7: Detect spatial clashes between element bounding boxes ──────────────

# ── Clash whitelist: known-valid overlaps that should NOT be flagged ──────────

# Inline MEP elements that physically sit inside/on a pipe or duct
_INLINE_MEP: frozenset[ElementType] = frozenset(
    {
        ElementType.VALVE,
        ElementType.SAFETY_RELIEF_VALVE,
        ElementType.STRAINER,
        ElementType.EXPANSION_JOINT,
        ElementType.FIRE_DAMPER,
        ElementType.DELUGE_VALVE,
        ElementType.PIPE_SUPPORT,
    }
)

# Conduit types that legitimately pass through walls/floors
_CONDUIT_TYPES: frozenset[ElementType] = frozenset(
    {
        ElementType.PIPE,
        ElementType.CONDUIT,
        ElementType.DUCT,
        ElementType.CABLE_TRAY,
    }
)

# Planar types whose BBs may overlap in XY but are at different Z elevations
_HORIZONTAL_PLANES: frozenset[ElementType] = frozenset(
    {
        ElementType.FLOOR,
        ElementType.CEILING,
        ElementType.GRATING,
    }
)


def _is_whitelisted_pair(a: ElementInstruction, b: ElementInstruction) -> bool:
    """Return True if the (a, b) pair is a known-valid overlap.

    Whitelist rules:
      1. Valve/fitting on pipe — inline MEP element overlapping its parent conduit.
      2. Pipe through wall/floor — conduit penetrating a planar structural element
         (overlap typically in 1–2 axes only; real clash resolved by penetration seal).
      3. Stacked floor/ceiling — horizontal planes at different Z share XY footprint;
         only flag if their Z ranges actually overlap.
    """
    types = {a.element_type, b.element_type}

    # Rule 1: inline MEP on conduit
    if types & _INLINE_MEP and types & _CONDUIT_TYPES:
        return True

    # Rule 2: conduit through wall/floor/ceiling
    wall_floor = {
        ElementType.WALL,
        ElementType.FLOOR,
        ElementType.CEILING,
        ElementType.GRATING,
        ElementType.SLAB_OPENING,
    }
    if types & _CONDUIT_TYPES and types & wall_floor:
        return True

    # Rule 3: stacked horizontal planes — only whitelist if Z ranges do NOT overlap
    if a.element_type in _HORIZONTAL_PLANES and b.element_type in _HORIZONTAL_PLANES:
        bb_a, bb_b = a.bounding_box, b.bounding_box
        z_gap = max(bb_a.min_z, bb_b.min_z) - min(bb_a.max_z, bb_b.max_z)
        if z_gap >= 0:  # no Z overlap — separate planes stacked vertically
            return True

    return False


def detect_clashes(
    instructions: list[ElementInstruction],
    tolerance_mm: float = 20.0,
) -> list[ClashReport]:
    """Check all instruction bounding boxes for spatial intersection.

    Compares every pair of element instructions and reports:
      hard clash — BBs fully overlap in all three axes (clearance < 0)
      soft clash — BBs are within tolerance_mm of each other

    Known-valid overlaps (valve-on-pipe, pipe-through-wall, stacked floor/ceiling)
    are whitelisted and excluded from results.

    O(n²) — acceptable for element counts up to ~1 000 per zone.
    For larger element sets, wire the E1 SegmentSpatialIndex from geometry_tools.

    Args:
        instructions:  List of prepared ElementInstructions (post-classification).
        tolerance_mm:  Clearance threshold below which a soft clash is raised.

    Returns:
        List of ClashReport records (may be empty if no clashes found).
    """
    clashes: list[ClashReport] = []
    whitelisted = 0
    n = len(instructions)

    for i in range(n):
        a = instructions[i]
        for j in range(i + 1, n):
            b = instructions[j]

            if a.segment_id == b.segment_id:
                continue  # same segment — skip (shouldn't happen but guard anyway)

            bb_a = a.bounding_box
            bb_b = b.bounding_box

            # Gap along each axis — negative value means overlap in that axis
            gap_x = max(bb_a.min_x, bb_b.min_x) - min(bb_a.max_x, bb_b.max_x)
            gap_y = max(bb_a.min_y, bb_b.min_y) - min(bb_a.max_y, bb_b.max_y)
            gap_z = max(bb_a.min_z, bb_b.min_z) - min(bb_a.max_z, bb_b.max_z)

            # Effective clearance = maximum gap across all three axes.
            # Negative → hard clash (BBs overlap in all 3 axes simultaneously).
            clearance = max(gap_x, gap_y, gap_z)

            if clearance >= tolerance_mm:
                continue  # no clash

            # Skip known-valid overlaps
            if _is_whitelisted_pair(a, b):
                whitelisted += 1
                continue

            if clearance < 0:
                clashes.append(
                    ClashReport(
                        element_a_id=a.segment_id,
                        element_b_id=b.segment_id,
                        clash_type="hard",
                        clearance_mm=round(clearance, 1),
                        zone_id=a.zone_id,
                    )
                )
                logger.warning(
                    "hard_clash_detected",
                    a=a.element_type.value,
                    b=b.element_type.value,
                    clearance_mm=round(clearance, 1),
                    zone_id=a.zone_id,
                )
            else:
                clashes.append(
                    ClashReport(
                        element_a_id=a.segment_id,
                        element_b_id=b.segment_id,
                        clash_type="soft",
                        clearance_mm=round(clearance, 1),
                        zone_id=a.zone_id,
                    )
                )

    if clashes or whitelisted:
        logger.info(
            "clash_detection_complete",
            total_pairs=n * (n - 1) // 2,
            hard=sum(1 for c in clashes if c.clash_type == "hard"),
            soft=sum(1 for c in clashes if c.clash_type == "soft"),
            whitelisted=whitelisted,
        )

    return clashes


# ── Tool 9: Trigger CDE transition ────────────────────────────────────────────


def trigger_cde_transition(
    session_id: str,
    target_state: str,
    cde_service: CDEStateService,
    triggered_by: str = "coordinator",
) -> dict:
    """
    Request a CDE state transition:
      WIP → SHARED     (all elements created)
      SHARED → PUBLISHED  (no open NCRs, all SC2 approved)

    Returns {success, from_state, to_state, blocked_reason}
    """
    try:
        transition = cde_service.transition(
            session_id=session_id,
            target_state=target_state,
            triggered_by=triggered_by,
        )
        return {
            "success": True,
            "from_state": transition.from_state,
            "to_state": transition.to_state,
        }
    except Exception as exc:
        logger.warning(
            "cde_transition_blocked",
            session_id=session_id[:12],
            target=target_state,
            reason=str(exc),
        )
        return {
            "success": False,
            "blocked_reason": str(exc),
        }
