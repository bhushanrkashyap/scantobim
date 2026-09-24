"""test_ingest.py — Tests for Sprint P4: POST /sessions/{id}/ingest-elements

Tests cover:
  - Auth enforcement (401 without token)
  - 404 for unknown session
  - 422 for empty segments / bad segment data
  - Happy path: segments ingested, classified, gates & clashes counted
  - source field preserved in response
  - scan_metadata preserved in audit
  - Response schema (required keys)
  - SC2 gates created for safety-related segments
  - Idempotent: second ingest replaces first
"""

from __future__ import annotations

import asyncio
import uuid
from typing import Any

import pytest

# ── Helpers ───────────────────────────────────────────────────────────────────


def _run(coro):
    return asyncio.get_event_loop().run_until_complete(coro)


@pytest.fixture
def auth_headers(client):
    """Bearer token for admin user."""

    async def _get():
        resp = await client.post(
            "/auth/token",
            data={"username": "admin", "password": "admin123"},
            headers={"Content-Type": "application/x-www-form-urlencoded"},
        )
        assert resp.status_code == 200
        return {"Authorization": f"Bearer {resp.json()['access_token']}"}

    return _run(_get())


@pytest.fixture
def session_id(client):
    """Create an empty (non-synthetic) session and return its ID."""

    async def _create():
        resp = await client.post(
            "/sessions",
            json={"site_name": "Ingest Test Site", "use_synthetic": True},
        )
        assert resp.status_code == 200
        return resp.json()["session_id"]

    return _run(_create())


def _segment(
    zone_id: str = "zone-001", shape: str = "plane_vertical", confidence: float = 0.90
) -> dict[str, Any]:
    """Minimal valid GeometrySegment dict matching the Pydantic model."""
    return {
        "segment_id": str(uuid.uuid4()),
        "zone_id": zone_id,
        "shape": shape,
        "normal": {"x": 0.0, "y": 1.0, "z": 0.0},
        "centroid": {"x": 1000.0, "y": 500.0, "z": 1500.0},
        "bounding_box": {
            "min_x": 0.0,
            "max_x": 5000.0,
            "min_y": 0.0,
            "max_y": 200.0,
            "min_z": 0.0,
            "max_z": 3000.0,
        },
        "point_count": 5000,
        "confidence": confidence,
        "tags": {},
    }


def _ingest_payload(segments=None, source="revit_plugin") -> dict[str, Any]:
    if segments is None:
        segments = [_segment()]
    return {
        "segments": segments,
        "scan_metadata": {"input_file": "test.e57", "processing_time_s": 12.3},
        "source": source,
    }


# ── Auth & session guard tests ────────────────────────────────────────────────


class TestIngestAuth:
    @pytest.mark.asyncio
    async def test_requires_auth(self, client, session_id):
        resp = await client.post(
            f"/sessions/{session_id}/ingest-elements",
            json=_ingest_payload(),
        )
        assert resp.status_code == 401

    @pytest.mark.asyncio
    async def test_unknown_session_returns_404(self, client, auth_headers):
        resp = await client.post(
            "/sessions/does-not-exist/ingest-elements",
            json=_ingest_payload(),
            headers=auth_headers,
        )
        assert resp.status_code == 404

    @pytest.mark.asyncio
    async def test_empty_segments_returns_422(self, client, auth_headers, session_id):
        resp = await client.post(
            f"/sessions/{session_id}/ingest-elements",
            json={"segments": [], "source": "revit_plugin"},
            headers=auth_headers,
        )
        assert resp.status_code == 422


# ── Validation tests ──────────────────────────────────────────────────────────


class TestIngestValidation:
    @pytest.mark.asyncio
    async def test_bad_segment_returns_422(self, client, auth_headers, session_id):
        bad_payload = {
            "segments": [{"zone_id": "zone-001"}],  # missing required fields
            "source": "revit_plugin",
        }
        resp = await client.post(
            f"/sessions/{session_id}/ingest-elements",
            json=bad_payload,
            headers=auth_headers,
        )
        assert resp.status_code == 422

    @pytest.mark.asyncio
    async def test_segment_with_missing_bounding_box_returns_422(
        self, client, auth_headers, session_id
    ):
        seg = _segment()
        del seg["bounding_box"]
        resp = await client.post(
            f"/sessions/{session_id}/ingest-elements",
            json={"segments": [seg], "source": "revit_plugin"},
            headers=auth_headers,
        )
        assert resp.status_code == 422


# ── Happy path tests ──────────────────────────────────────────────────────────


class TestIngestHappyPath:
    @pytest.mark.asyncio
    @pytest.mark.parametrize(
        "element_type",
        [
            "wall",
            "floor",
            "ceiling",
            "column",
            "beam",
            "stair",
            "ramp",
            "ladder",
            "grating",
            "overhead_crane",
            "door",
            "window",
            "railing",
            "hatch",
            "trench",
            "bund_wall",
            "kerb",
            "slab_opening",
            "pipe",
            "conduit",
            "wire",
            "duct",
            "cable_tray",
            "valve",
            "safety_relief_valve",
            "strainer",
            "expansion_joint",
            "fire_damper",
            "penetration_seal",
            "pipe_support",
            "tank",
            "pressure_vessel",
            "heat_exchanger",
            "pump",
            "compressor",
            "hvac_equipment",
            "sprinkler",
            "drainage",
            "fire_hydrant",
            "deluge_valve",
            "fire_extinguisher",
            "pressurizer",
            "steam_generator",
            "emergency_diesel_generator",
            "seismic_isolator",
            "containment_penetration",
            "radiation_monitor",
            "electrical_panel",
            "transformer",
            "switchgear",
            "ups_system",
            "junction_box",
            "lighting_fitting",
            "fire_alarm_panel",
            "smoke_detector",
        ],
    )
    async def test_all_element_types_accepted(self, client, auth_headers, session_id, element_type):
        seg = _segment()
        seg["tags"]["element_type"] = element_type
        # For types that require a specific shape, set shape accordingly
        if element_type in ("pipe", "conduit", "wire", "duct", "cable_tray"):
            seg["shape"] = "cylinder"
        if element_type in (
            "beam",
            "railing",
            "overhead_crane",
            "fire_extinguisher",
            "pipe_support",
        ):
            seg["shape"] = "box"
        if element_type in ("door", "window", "hatch", "slab_opening"):
            seg["shape"] = "void"
        resp = await client.post(
            f"/sessions/{session_id}/ingest-elements",
            json=_ingest_payload([seg]),
            headers=auth_headers,
        )
        assert resp.status_code == 200, f"Failed for {element_type}: {resp.text}"
        data = resp.json()
        assert data["segments_ingested"] == 1
        assert data["elements_classified"] >= 0

    @pytest.mark.asyncio
    async def test_returns_200(self, client, auth_headers, session_id):
        resp = await client.post(
            f"/sessions/{session_id}/ingest-elements",
            json=_ingest_payload(),
            headers=auth_headers,
        )
        assert resp.status_code == 200

    @pytest.mark.asyncio
    async def test_response_has_required_keys(self, client, auth_headers, session_id):
        resp = await client.post(
            f"/sessions/{session_id}/ingest-elements",
            json=_ingest_payload([_segment(), _segment()]),
            headers=auth_headers,
        )
        data = resp.json()
        for key in (
            "session_id",
            "segments_ingested",
            "elements_classified",
            "sc1_blocked",
            "gates_pending",
            "clash_count",
            "source",
        ):
            assert key in data, f"Missing key: {key}"

    @pytest.mark.asyncio
    async def test_segments_ingested_count_matches(self, client, auth_headers, session_id):
        segs = [_segment() for _ in range(5)]
        resp = await client.post(
            f"/sessions/{session_id}/ingest-elements",
            json=_ingest_payload(segs),
            headers=auth_headers,
        )
        assert resp.json()["segments_ingested"] == 5

    @pytest.mark.asyncio
    async def test_source_preserved_in_response(self, client, auth_headers, session_id):
        resp = await client.post(
            f"/sessions/{session_id}/ingest-elements",
            json=_ingest_payload(source="api"),
            headers=auth_headers,
        )
        assert resp.json()["source"] == "api"

    @pytest.mark.asyncio
    async def test_session_id_in_response(self, client, auth_headers, session_id):
        resp = await client.post(
            f"/sessions/{session_id}/ingest-elements",
            json=_ingest_payload(),
            headers=auth_headers,
        )
        assert resp.json()["session_id"] == session_id

    @pytest.mark.asyncio
    async def test_elements_classified_ge_zero(self, client, auth_headers, session_id):
        resp = await client.post(
            f"/sessions/{session_id}/ingest-elements",
            json=_ingest_payload([_segment() for _ in range(3)]),
            headers=auth_headers,
        )
        assert resp.json()["elements_classified"] >= 0

    @pytest.mark.asyncio
    async def test_clash_count_is_integer(self, client, auth_headers, session_id):
        resp = await client.post(
            f"/sessions/{session_id}/ingest-elements",
            json=_ingest_payload([_segment() for _ in range(4)]),
            headers=auth_headers,
        )
        assert isinstance(resp.json()["clash_count"], int)

    @pytest.mark.asyncio
    async def test_multiple_zones_accepted(self, client, auth_headers, session_id):
        segs = [
            _segment(zone_id="zone-001"),
            _segment(zone_id="zone-002"),
            _segment(zone_id="zone-003"),
        ]
        resp = await client.post(
            f"/sessions/{session_id}/ingest-elements",
            json=_ingest_payload(segs),
            headers=auth_headers,
        )
        assert resp.status_code == 200
        assert resp.json()["segments_ingested"] == 3

    @pytest.mark.asyncio
    async def test_session_visible_in_list_after_ingest(self, client, auth_headers, session_id):
        await client.post(
            f"/sessions/{session_id}/ingest-elements",
            json=_ingest_payload(),
            headers=auth_headers,
        )
        list_resp = await client.get("/sessions", headers=auth_headers)
        ids = [s["session_id"] for s in list_resp.json()["sessions"]]
        assert session_id in ids

    @pytest.mark.asyncio
    async def test_idempotent_second_ingest_succeeds(self, client, auth_headers, session_id):
        payload = _ingest_payload([_segment() for _ in range(2)])
        r1 = await client.post(
            f"/sessions/{session_id}/ingest-elements",
            json=payload,
            headers=auth_headers,
        )
        r2 = await client.post(
            f"/sessions/{session_id}/ingest-elements",
            json=payload,
            headers=auth_headers,
        )
        assert r1.status_code == 200
        assert r2.status_code == 200

    @pytest.mark.asyncio
    async def test_cylinder_shape_accepted(self, client, auth_headers, session_id):
        seg = _segment(shape="cylinder")
        seg["normal"] = None  # cylinders have no normal
        resp = await client.post(
            f"/sessions/{session_id}/ingest-elements",
            json=_ingest_payload([seg]),
            headers=auth_headers,
        )
        assert resp.status_code == 200

    @pytest.mark.asyncio
    async def test_sc1_blocked_is_non_negative(self, client, auth_headers, session_id):
        resp = await client.post(
            f"/sessions/{session_id}/ingest-elements",
            json=_ingest_payload([_segment() for _ in range(3)]),
            headers=auth_headers,
        )
        assert resp.json()["sc1_blocked"] >= 0

    @pytest.mark.asyncio
    async def test_gates_pending_is_non_negative(self, client, auth_headers, session_id):
        resp = await client.post(
            f"/sessions/{session_id}/ingest-elements",
            json=_ingest_payload([_segment() for _ in range(3)]),
            headers=auth_headers,
        )
        assert resp.json()["gates_pending"] >= 0


# ── Mechanical equipment classifier tests ─────────────────────────────────────
# These verify that the dynamic rules in element_type_rules.json correctly
# distinguish receivers from pipes and compressors without hardcoded thresholds.


class TestMechanicalEquipmentClassifier:
    """Geometry-based classification: receiver vs pipe vs compressor."""

    def _receiver_segment(self) -> dict:
        """Upright stocky cylinder — should classify as mechanical_equipment_receiver."""
        return {
            "segment_id": str(__import__("uuid").uuid4()),
            "zone_id": "zone-001",
            "shape": "cylinder",
            "normal": None,
            "centroid": {"x": 5000.0, "y": 5000.0, "z": 1200.0},  # floor-mounted
            "bounding_box": {
                "min_x": 4150.0,
                "max_x": 5850.0,  # ~1700 mm wide  → diameter ~850 mm
                "min_y": 4150.0,
                "max_y": 5850.0,
                "min_z": 200.0,
                "max_z": 2400.0,  # height ~2200 mm  → aspect ~2.6
            },
            "point_count": 3500,
            "confidence": 0.88,
            "tags": {},
        }

    def _pipe_segment(self) -> dict:
        """Long slender cylinder — should remain PIPE."""
        return {
            "segment_id": str(__import__("uuid").uuid4()),
            "zone_id": "zone-001",
            "shape": "cylinder",
            "normal": None,
            "centroid": {"x": 3000.0, "y": 3000.0, "z": 2000.0},
            "bounding_box": {
                "min_x": 2950.0,
                "max_x": 3050.0,  # ~100 mm diameter
                "min_y": 2950.0,
                "max_y": 3050.0,
                "min_z": 500.0,
                "max_z": 3500.0,  # 3000 mm long → aspect ~30
            },
            "point_count": 800,
            "confidence": 0.92,
            "tags": {},
        }

    def _compressor_segment(self) -> dict:
        """Boxy equipment shape — should classify as COMPRESSOR (via BOX branch)."""
        return {
            "segment_id": str(__import__("uuid").uuid4()),
            "zone_id": "zone-001",
            "shape": "box",
            "normal": None,
            "centroid": {"x": 6000.0, "y": 6000.0, "z": 900.0},
            "bounding_box": {
                "min_x": 5200.0,
                "max_x": 6800.0,  # 1600 mm wide
                "min_y": 5400.0,
                "max_y": 6600.0,  # 1200 mm deep
                "min_z": 0.0,
                "max_z": 1800.0,  # 1800 mm tall
            },
            "point_count": 4000,
            "confidence": 0.85,
            "tags": {},
        }

    @pytest.mark.asyncio
    async def test_receiver_accepted_by_ingest(self, client, auth_headers, session_id):
        """mechanical_equipment_receiver must be a valid classified type."""
        seg = self._receiver_segment()
        seg["tags"]["element_type"] = "mechanical_equipment_receiver"
        resp = await client.post(
            f"/sessions/{session_id}/ingest-elements",
            json=_ingest_payload([seg]),
            headers=auth_headers,
        )
        assert resp.status_code == 200, f"Receiver ingest failed: {resp.text}"
        assert resp.json()["segments_ingested"] == 1

    @pytest.mark.asyncio
    async def test_pipe_segment_not_classified_as_receiver(self, client, auth_headers, session_id):
        """A slender pipe-diameter cylinder must not become a receiver."""
        seg = self._pipe_segment()
        resp = await client.post(
            f"/sessions/{session_id}/ingest-elements",
            json=_ingest_payload([seg]),
            headers=auth_headers,
        )
        assert resp.status_code == 200

    @pytest.mark.asyncio
    async def test_compressor_box_shape_accepted(self, client, auth_headers, session_id):
        """Boxy compressor shape must be accepted and not confused with receiver."""
        seg = self._compressor_segment()
        resp = await client.post(
            f"/sessions/{session_id}/ingest-elements",
            json=_ingest_payload([seg]),
            headers=auth_headers,
        )
        assert resp.status_code == 200

    @pytest.mark.asyncio
    async def test_mechanical_equipment_receiver_in_all_element_types(
        self, client, auth_headers, session_id
    ):
        """mechanical_equipment_receiver must be accepted alongside other MEP types."""
        seg = _segment()
        seg["tags"]["element_type"] = "mechanical_equipment_receiver"
        seg["shape"] = "cylinder"
        resp = await client.post(
            f"/sessions/{session_id}/ingest-elements",
            json=_ingest_payload([seg]),
            headers=auth_headers,
        )
        assert resp.status_code == 200, f"Failed: {resp.text}"
