"""Tests for P5 CDE State Machine — cde_state.py"""

from __future__ import annotations

import pytest

from agent.audit import AuditLedger
from agent.cde_state import CDEStateService, CDETransitionError
from agent.models import CDEState, SafetyCategory
from agent.ncr import NCRService
from agent.orchestrator import ScanToBIMOrchestrator
from agent.safety_gate import SafetyGateService

# ── Fixtures ──────────────────────────────────────────────────────────────────


@pytest.fixture
def services():
    audit = AuditLedger()
    safety_gate = SafetyGateService(audit)
    orchestrator = ScanToBIMOrchestrator(audit=audit, safety_gate=safety_gate)
    ncr = NCRService(audit)
    cde = CDEStateService(
        audit=audit,
        orchestrator=orchestrator,
        safety_gate=safety_gate,
        ncr_service=ncr,
    )
    return audit, safety_gate, orchestrator, ncr, cde


@pytest.fixture
def empty_session(services):
    """A session with no instructions — 'all done' trivially."""
    _, _, orchestrator, _, _ = services
    session = orchestrator.create_session("Test Site")
    sid = session.session_id
    # Ensure empty instruction + result lists so pending == 0
    orchestrator._instructions[sid] = []
    orchestrator._results[sid] = []
    return sid


# ── Initial state ─────────────────────────────────────────────────────────────


def test_initial_state_is_wip(services, empty_session):
    _, _, _, _, cde = services
    state = cde.get_state(empty_session)
    assert state.current_state == CDEState.WIP


def test_get_state_creates_entry_on_first_call(services):
    _, _, _, _, cde = services
    sid = "nonexistent-session-001"
    state = cde.get_state(sid)
    assert state.session_id == sid
    assert state.current_state == CDEState.WIP


# ── WIP → SHARED ──────────────────────────────────────────────────────────────


def test_wip_to_shared_succeeds_when_no_pending(services, empty_session):
    _, _, _, _, cde = services
    t = cde.transition(empty_session, "SHARED", triggered_by="coordinator")
    assert t.from_state == CDEState.WIP
    assert t.to_state == CDEState.SHARED


def test_wip_to_shared_blocked_when_instructions_pending(services):
    _, _, orchestrator, _, cde = services
    session = orchestrator.create_session("Pending Site")
    sid = session.session_id
    # Add 1 instruction but no result → pending = 1

    from agent.models import (
        BoundingBox,
        Discipline,
        ElementInstruction,
        ElementType,
        Point3D,
    )

    bb = BoundingBox(min_x=0, min_y=0, min_z=0, max_x=100, max_y=100, max_z=100)
    instr = ElementInstruction(
        segment_id="seg-001",
        zone_id="z1",
        element_type=ElementType.WALL,
        discipline=Discipline.STRUCTURAL,
        safety_category=SafetyCategory.NS,
        bounding_box=bb,
        centroid=Point3D(x=50, y=50, z=50),
    )
    orchestrator._instructions[sid] = [instr]
    orchestrator._results[sid] = []  # nothing processed yet

    with pytest.raises(CDETransitionError, match="pending"):
        cde.transition(sid, "SHARED")


def test_state_recorded_in_transitions_list(services, empty_session):
    _, _, _, _, cde = services
    cde.transition(empty_session, "SHARED")
    state = cde.get_state(empty_session)
    assert len(state.transitions) == 1
    assert state.transitions[0].to_state == CDEState.SHARED


# ── SHARED → PUBLISHED ────────────────────────────────────────────────────────


def test_shared_to_published_succeeds_clean(services, empty_session):
    _, _, _, _, cde = services
    cde.transition(empty_session, "SHARED")
    t = cde.transition(empty_session, "PUBLISHED")
    assert t.to_state == CDEState.PUBLISHED


def test_published_blocked_by_open_ncr(services, empty_session):
    _, _, _, ncr, cde = services
    cde.transition(empty_session, "SHARED")
    ncr.raise_ncr(empty_session, "el-001", "Test deviation too high", "MAJOR")
    with pytest.raises(CDETransitionError, match="NCR"):
        cde.transition(empty_session, "PUBLISHED")


def test_published_allowed_after_ncr_resolved(services, empty_session):
    _, _, _, ncr, cde = services
    cde.transition(empty_session, "SHARED")
    record = ncr.raise_ncr(empty_session, "el-001", "Test NCR", "MINOR")
    ncr.resolve_ncr(record.ncr_id, "engineer@example.com", "Deviation within concession")
    t = cde.transition(empty_session, "PUBLISHED")
    assert t.to_state == CDEState.PUBLISHED


def test_published_blocked_by_broken_chain(services, empty_session):
    import sqlite3

    audit, _, _, _, cde = services
    cde.transition(empty_session, "SHARED")
    # Corrupt the stored event_hash in SQLite so verify_chain detects a mismatch
    with sqlite3.connect(audit.db_path) as conn:
        # SQLite doesn't support LIMIT in UPDATE — use rowid subquery
        conn.execute(
            """UPDATE audit_events SET event_hash='corrupted-000'
               WHERE rowid = (
                 SELECT rowid FROM audit_events WHERE session_id=? LIMIT 1
               )""",
            (empty_session,),
        )
    with pytest.raises(CDETransitionError, match="chain"):
        cde.transition(empty_session, "PUBLISHED")


# ── Invalid transitions ────────────────────────────────────────────────────────


def test_wip_to_published_directly_is_blocked(services, empty_session):
    _, _, _, _, cde = services
    with pytest.raises(CDETransitionError, match="Invalid transition"):
        cde.transition(empty_session, "PUBLISHED")


def test_unknown_target_state_raises(services, empty_session):
    _, _, _, _, cde = services
    with pytest.raises((CDETransitionError, ValueError)):
        cde.transition(empty_session, "ARCHIVED")


# ── Audit events emitted ──────────────────────────────────────────────────────


def test_cde_transition_appends_audit_event(services, empty_session):
    audit, _, _, _, cde = services
    cde.transition(empty_session, "SHARED")
    events = audit.get_session_events(empty_session)
    cde_events = [e for e in events if e.event_type.value == "cde_state_changed"]
    assert len(cde_events) >= 1
    assert cde_events[-1].detail["to_state"] == "SHARED"


def test_publish_emits_design_record_issued(services, empty_session):
    audit, _, _, _, cde = services
    cde.transition(empty_session, "SHARED")
    cde.transition(empty_session, "PUBLISHED")
    events = audit.get_session_events(empty_session)
    drp_events = [e for e in events if e.event_type.value == "design_record_issued"]
    assert len(drp_events) == 1
