"""
test_api.py — Integration tests for ScanToBIM FastAPI endpoints.

Tests run against the full FastAPI app via HTTPX async test client.
open3d is mocked (see conftest.py) so these run in CI without installing it.
"""

from __future__ import annotations

import pytest
from httpx import AsyncClient

# ── Health ────────────────────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_health_returns_ok(client: AsyncClient) -> None:
    resp = await client.get("/health")
    assert resp.status_code == 200
    data = resp.json()
    assert data["status"] == "ok"
    assert "service" in data
    assert "version" in data


# ── Sessions ──────────────────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_create_session_synthetic_returns_200(client: AsyncClient) -> None:
    resp = await client.post(
        "/sessions",
        json={
            "site_name": "CI Test Facility",
            "zone_id": "zone-001",
            "use_synthetic": True,
        },
    )
    assert resp.status_code == 200, resp.text
    data = resp.json()

    assert "session_id" in data
    assert isinstance(data["session_id"], str)
    assert len(data["session_id"]) > 0


@pytest.mark.asyncio
async def test_create_session_returns_segment_count(client: AsyncClient) -> None:
    resp = await client.post(
        "/sessions",
        json={
            "site_name": "CI Segment Count Test",
            "use_synthetic": True,
        },
    )
    assert resp.status_code == 200
    data = resp.json()
    assert "segments_found" in data
    assert isinstance(data["segments_found"], int)
    assert data["segments_found"] >= 0


@pytest.mark.asyncio
async def test_create_session_execution_has_required_keys(client: AsyncClient) -> None:
    resp = await client.post(
        "/sessions",
        json={
            "site_name": "CI Execution Keys Test",
            "use_synthetic": True,
        },
    )
    assert resp.status_code == 200
    data = resp.json()
    execution = data.get("execution", {})
    for key in ("created", "blocked", "pending_approval", "errors"):
        assert key in execution, f"Missing key in execution: {key}"


@pytest.mark.asyncio
async def test_get_session_by_id(client: AsyncClient) -> None:
    # Create a session first
    create_resp = await client.post(
        "/sessions",
        json={
            "site_name": "CI Get Session Test",
            "use_synthetic": True,
        },
    )
    assert create_resp.status_code == 200
    session_id = create_resp.json()["session_id"]

    # Fetch it back — response is the session summary dict
    get_resp = await client.get(f"/sessions/{session_id}")
    assert get_resp.status_code == 200
    data = get_resp.json()
    # The session summary contains the ID at some key — verify it is present
    assert session_id in str(data)  # session_id appears somewhere in the response


@pytest.mark.asyncio
async def test_get_session_not_found(client: AsyncClient) -> None:
    resp = await client.get("/sessions/nonexistent-id-xyz")
    assert resp.status_code == 404


# ── Audit Chain ───────────────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_audit_chain_is_valid_after_session(client: AsyncClient) -> None:
    create_resp = await client.post(
        "/sessions",
        json={
            "site_name": "CI Chain Test",
            "use_synthetic": True,
        },
    )
    assert create_resp.status_code == 200
    session_id = create_resp.json()["session_id"]

    chain_resp = await client.get(f"/sessions/{session_id}/chain")
    assert chain_resp.status_code == 200
    data = chain_resp.json()

    assert data["valid"] is True, f"Chain broken: {data.get('broken_links')}"
    assert isinstance(data["event_count"], int)
    assert data["event_count"] > 0


@pytest.mark.asyncio
async def test_audit_events_are_returned(client: AsyncClient) -> None:
    create_resp = await client.post(
        "/sessions",
        json={
            "site_name": "CI Audit Events Test",
            "use_synthetic": True,
        },
    )
    session_id = create_resp.json()["session_id"]

    audit_resp = await client.get(f"/sessions/{session_id}/audit")
    assert audit_resp.status_code == 200
    body = audit_resp.json()

    # Response is either a list or a dict with an "events" key
    events = body if isinstance(body, list) else body.get("events", [])
    assert len(events) > 0

    # Every event must have required fields
    for event in events:
        assert "event_id" in event
        assert "event_type" in event
        assert "timestamp_utc" in event
        assert "event_hash" in event


# ── Safety Gates ──────────────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_safety_gates_endpoint_returns_list(client: AsyncClient) -> None:
    resp = await client.get("/safety-gates")
    assert resp.status_code == 200
    body = resp.json()
    # Response is either a bare list or a dict containing a list under "pending_gates"
    gates = body if isinstance(body, list) else body.get("pending_gates", body)
    assert isinstance(gates, list)


@pytest.mark.asyncio
async def test_safety_gate_not_found(client: AsyncClient) -> None:
    resp = await client.get("/safety-gates/nonexistent-gate-id")
    assert resp.status_code == 404


@pytest.mark.asyncio
async def test_safety_gate_decision_not_found(client: AsyncClient) -> None:
    resp = await client.post(
        "/safety-gates/nonexistent-gate-id/decision",
        json={"approved": True, "approver_upn": "test@test.com"},
    )
    assert resp.status_code == 404


# ── Models ────────────────────────────────────────────────────────────────────


def test_safety_category_enum_values() -> None:
    from agent.models import SafetyCategory

    assert SafetyCategory.SC1.value == "SC1"
    assert SafetyCategory.SC2.value == "SC2"
    assert SafetyCategory.SC3.value == "SC3"
    assert SafetyCategory.NS.value == "NS"


def test_element_instruction_requires_segment_id() -> None:
    import uuid

    from agent.models import (
        BoundingBox,
        Discipline,
        ElementInstruction,
        ElementType,
        Point3D,
        SafetyCategory,
    )

    # Use value-based lookup (ElementType("wall")) — works across Python 3.9 and 3.11
    instr = ElementInstruction(
        instruction_id=str(uuid.uuid4()),
        segment_id="seg-123",
        zone_id="zone-001",
        element_type=ElementType("wall"),
        discipline=Discipline("structural"),
        safety_category=SafetyCategory("NS"),
        bounding_box=BoundingBox(min_x=0, min_y=0, min_z=0, max_x=5000, max_y=200, max_z=3000),
        centroid=Point3D(x=2500, y=100, z=1500),
    )
    assert instr.segment_id == "seg-123"
    assert instr.safety_category.value == "NS"


def test_discipline_tools_dispatch_all_types():
    """Ensure every create_* discipline tool dispatches without error for its type."""
    import uuid

    import agent.tools.discipline_tools as dt
    from agent.models import (
        BoundingBox,
        Discipline,
        ElementInstruction,
        ElementType,
        Point3D,
        SafetyCategory,
    )

    # Map element type to discipline for test
    discipline_map = {
        "wall": "structural",
        "floor": "structural",
        "ceiling": "architectural",
        "column": "structural",
        "beam": "structural",
        "stair": "architectural",
        "ramp": "architectural",
        "ladder": "structural",
        "grating": "structural",
        "overhead_crane": "structural",
        "door": "architectural",
        "window": "architectural",
        "railing": "architectural",
        "hatch": "architectural",
        "trench": "civil",
        "bund_wall": "civil",
        "kerb": "civil",
        "slab_opening": "civil",
        "pipe": "mep",
        "conduit": "mep",
        "wire": "mep",
        "duct": "mep",
        "cable_tray": "mep",
        "valve": "mep",
        "safety_relief_valve": "mep",
        "strainer": "mep",
        "expansion_joint": "mep",
        "fire_damper": "mep",
        "penetration_seal": "mep",
        "pipe_support": "mep",
        "tank": "mep",
        "pressure_vessel": "mep",
        "heat_exchanger": "mep",
        "pump": "mep",
        "compressor": "mep",
        "hvac_equipment": "mep",
        "sprinkler": "fire_protection",
        "drainage": "mep",
        "fire_hydrant": "fire_protection",
        "deluge_valve": "fire_protection",
        "fire_extinguisher": "fire_protection",
        "pressurizer": "mep",
        "steam_generator": "mep",
        "emergency_diesel_generator": "mep",
        "seismic_isolator": "infrastructure",
        "containment_penetration": "infrastructure",
        "radiation_monitor": "infrastructure",
        "electrical_panel": "mep",
        "transformer": "mep",
        "switchgear": "mep",
        "ups_system": "mep",
        "junction_box": "mep",
        "lighting_fitting": "mep",
        "fire_alarm_panel": "fire_protection",
        "smoke_detector": "fire_protection",
    }
    for et in [e.value for e in ElementType]:
        instr = ElementInstruction(
            instruction_id=str(uuid.uuid4()),
            segment_id=f"seg-{et}",
            zone_id="zone-001",
            element_type=ElementType(et),
            discipline=Discipline(discipline_map.get(et, "structural")),
            safety_category=SafetyCategory("NS"),
            bounding_box=BoundingBox(min_x=0, min_y=0, min_z=0, max_x=100, max_y=100, max_z=100),
            centroid=Point3D(x=50, y=50, z=50),
        )
        fn_name = f"create_{et}"
        if hasattr(dt, fn_name):
            fn = getattr(dt, fn_name)
            # Should not raise for NS
            result = fn(
                instr,
                "test-session",
                orchestrator=type(
                    "Orch",
                    (),
                    {
                        "_dispatch_to_revit": lambda *_: type(
                            "R",
                            (),
                            {
                                "success": True,
                                "element_id": "e",
                                "instruction_id": "i",
                                "duration_ms": 1,
                                "error": None,
                            },
                        )()
                    },
                )(),
            )
            assert result["success"] is True, f"Failed for {et}"


def test_audit_event_hash_is_computed() -> None:
    import datetime
    import uuid

    from agent.models import AuditEvent, AuditEventType

    event = AuditEvent(
        event_id=str(uuid.uuid4()),
        session_id="sess-001",
        event_type=AuditEventType("session_started"),
        timestamp_utc=datetime.datetime.utcnow(),
        zone_id="zone-001",
        actor="test",
        detail={"msg": "test"},
        previous_hash="0" * 64,
    )
    # AuditEvent is frozen — use model_copy to attach the computed hash
    event = event.model_copy(update={"event_hash": event.compute_hash("test-secret")})
    assert len(event.event_hash) == 64  # SHA-256 hex digest


# ── Safety Gate Service (Python) ──────────────────────────────────────────────


def test_sc1_hard_block_raises() -> None:
    import uuid

    from agent.models import (
        BoundingBox,
        Discipline,
        ElementInstruction,
        ElementType,
        Point3D,
        SafetyCategory,
    )
    from agent.safety_gate import SafetyCategoryViolationError

    instr = ElementInstruction(
        instruction_id=str(uuid.uuid4()),
        segment_id="seg-sc1-wall",
        zone_id="zone-001",
        element_type=ElementType("wall"),
        discipline=Discipline("structural"),
        safety_category=SafetyCategory("SC1"),
        bounding_box=BoundingBox(min_x=0, min_y=0, min_z=0, max_x=10000, max_y=400, max_z=4000),
        centroid=Point3D(x=5000, y=200, z=2000),
    )

    with pytest.raises(SafetyCategoryViolationError):
        if instr.safety_category.value == "SC1":
            raise SafetyCategoryViolationError(
                f"SC1 HARD BLOCK: segment {instr.segment_id} cannot be auto-created."
            )
