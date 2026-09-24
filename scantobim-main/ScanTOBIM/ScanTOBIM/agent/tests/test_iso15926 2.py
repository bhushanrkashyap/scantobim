"""test_iso15926.py — ISO 15926 / CFIHOS handover mapping (oil-gas market).

Covers:
  • Every ElementType has an ISO 15926 class URI
  • Every ElementType has a CFIHOS attribute list
  • map_element produces expected shape
  • CFIHOS categories assigned correctly (PIPING, VESSEL, etc.)
  • generate_iso15926_json / generate_cfihos_csv
  • Coverage pct > 0 when attrs populate
  • Endpoint: auth, 404, 422 (no instructions), JSON, CSV
  • Bundle: 09_iso15926/ present when classified instructions exist
"""

from __future__ import annotations

import csv as _csv
import io
import uuid
import zipfile

import pytest
import pytest_asyncio
from httpx import ASGITransport, AsyncClient

from agent.models import (
    BoundingBox,
    Discipline,
    DSEARZone,
    ElementInstruction,
    ElementType,
    Point3D,
    SafetyCategory,
)
from agent.tools.iso15926_tools import (
    CFIHOS_ATTRIBUTES,
    ISO15926_CLASS_MAP,
    generate_cfihos_csv,
    generate_iso15926_json,
    map_element,
    map_session,
)

# ── Helpers ──────────────────────────────────────────────────────────────────


def _instruction(
    element_type: ElementType, zone_id: str = "Z1", params: dict | None = None
) -> ElementInstruction:
    return ElementInstruction(
        segment_id=str(uuid.uuid4()),
        zone_id=zone_id,
        element_type=element_type,
        discipline=Discipline.MEP,
        safety_category=SafetyCategory.NS,
        dsear_zone=DSEARZone.NONE,
        bounding_box=BoundingBox(min_x=0, min_y=0, min_z=0, max_x=1000, max_y=1000, max_z=1000),
        centroid=Point3D(x=500, y=500, z=500),
        parameters=params or {},
    )


# ═════════════════════════════════════════════════════════════════════════════
# 1. Mapping completeness
# ═════════════════════════════════════════════════════════════════════════════


class TestMappingCompleteness:
    def test_every_relevant_element_has_iso_class(self):
        """Every non-structural discipline ElementType we support
        should have an ISO 15926 class URI defined."""
        expected = {
            ElementType.PIPE,
            ElementType.VALVE,
            ElementType.SAFETY_RELIEF_VALVE,
            ElementType.PRESSURIZER,
            ElementType.STEAM_GENERATOR,
            ElementType.PUMP,
            ElementType.COMPRESSOR,
            ElementType.TANK,
            ElementType.PRESSURE_VESSEL,
            ElementType.HEAT_EXCHANGER,
            ElementType.ELECTRICAL_PANEL,
            ElementType.SWITCHGEAR,
            ElementType.WALL,
            ElementType.FLOOR,
            ElementType.CEILING,
        }
        for e in expected:
            assert e in ISO15926_CLASS_MAP, f"{e.value} missing ISO 15926 class"

    def test_every_class_maps_to_rdl_uri(self):
        for e, uri in ISO15926_CLASS_MAP.items():
            assert uri.startswith("http://"), f"{e.value}: bad URI {uri}"

    def test_every_mapped_element_has_cfihos_attrs(self):
        for e in ISO15926_CLASS_MAP:
            assert e in CFIHOS_ATTRIBUTES, f"{e.value} has no CFIHOS attrs"
            attrs = CFIHOS_ATTRIBUTES[e]
            # Universal attrs must be present
            for univ in (
                "tag_number",
                "functional_location",
                "description",
                "element_type",
                "iso15926_class",
            ):
                assert univ in attrs, f"{e.value} missing universal CFIHOS attr {univ}"


# ═════════════════════════════════════════════════════════════════════════════
# 2. map_element
# ═════════════════════════════════════════════════════════════════════════════


class TestMapElement:
    def test_piping_category(self):
        m = map_element(_instruction(ElementType.PIPE))
        assert m.cfihos_category == "PIPING"
        assert "nominal_diameter_mm" in (m.attributes.keys() | set(m.unpopulated))

    def test_vessel_category(self):
        m = map_element(_instruction(ElementType.PRESSURE_VESSEL))
        assert m.cfihos_category == "VESSEL"
        assert "asme_stamp" in m.unpopulated

    def test_rotating_category(self):
        m = map_element(_instruction(ElementType.PUMP))
        assert m.cfihos_category == "ROTATING_EQUIPMENT"
        assert "rated_power_kw" in m.unpopulated

    def test_electrical_category(self):
        m = map_element(_instruction(ElementType.SWITCHGEAR))
        assert m.cfihos_category == "ELECTRICAL"

    def test_structural_category(self):
        m = map_element(_instruction(ElementType.WALL))
        assert m.cfihos_category == "STRUCTURAL"

    def test_nuclear_primary_category(self):
        inst = _instruction(ElementType.PRESSURIZER)
        # Force SC1 on the instruction
        inst = inst.model_copy(update={"safety_category": SafetyCategory.SC1})
        m = map_element(inst)
        assert m.cfihos_category == "NUCLEAR_PRIMARY"
        assert m.attributes["nqa1_safety_class"] == "SC1"

    def test_tag_number_format(self):
        inst = _instruction(ElementType.PIPE, zone_id="unit-01")
        m = map_element(inst)
        tag = m.attributes["tag_number"]
        assert tag.startswith("UNIT-01-PIPE-")

    def test_diameter_auto_populated_when_param_given(self):
        m = map_element(_instruction(ElementType.PIPE, params={"diameter_mm": 100}))
        assert m.attributes.get("nominal_diameter_mm") == 100


# ═════════════════════════════════════════════════════════════════════════════
# 3. Bulk generators
# ═════════════════════════════════════════════════════════════════════════════


class TestBulk:
    def test_map_session_matches_instructions(self):
        insts = [
            _instruction(ElementType.PIPE),
            _instruction(ElementType.WALL),
            _instruction(ElementType.PUMP),
        ]
        maps = map_session(insts)
        assert len(maps) == 3
        assert [m.element_type for m in maps] == [
            "pipe",
            "wall",
            "pump",
        ]

    def test_json_doc_shape(self):
        insts = [_instruction(ElementType.PIPE)]
        doc = generate_iso15926_json("sess-xyz", map_session(insts))
        assert doc["session_id"] == "sess-xyz"
        assert doc["total_elements"] == 1
        assert "rdl_namespace" in doc
        assert "ISO 15926-4" in " ".join(doc["standard_refs"])
        assert "by_category" in doc
        assert "coverage_pct" in doc

    def test_csv_has_header_and_rows(self):
        insts = [_instruction(ElementType.PIPE), _instruction(ElementType.PUMP)]
        csv_str = generate_cfihos_csv(map_session(insts))
        rows = list(_csv.reader(io.StringIO(csv_str)))
        header = rows[0]
        assert "tag_number" in header
        assert "attribute_name" in header
        assert "status" in header
        # One row per attribute per instruction — at least 10 rows
        assert len(rows) > 10

    def test_csv_status_marks_pending_vs_populated(self):
        insts = [_instruction(ElementType.PIPE, params={"diameter_mm": 150})]
        csv_str = generate_cfihos_csv(map_session(insts))
        assert "POPULATED" in csv_str
        assert "PENDING" in csv_str  # e.g. design_pressure_bar not filled

    def test_coverage_pct_above_zero(self):
        doc = generate_iso15926_json(
            "s",
            map_session([_instruction(ElementType.PIPE)]),
        )
        assert doc["coverage_pct"] > 0


# ═════════════════════════════════════════════════════════════════════════════
# 4. API endpoint
# ═════════════════════════════════════════════════════════════════════════════


@pytest_asyncio.fixture
async def auth_client():
    from agent.main import app

    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as ac:
        resp = await ac.post(
            "/auth/token",
            data={"username": "admin", "password": "admin123"},
        )
        ac.headers["Authorization"] = f"Bearer {resp.json().get('access_token', '')}"
        yield ac


class TestEndpoint:
    @pytest.mark.asyncio
    async def test_requires_auth(self):
        from agent.main import app

        async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as ac:
            resp = await ac.get("/sessions/x/iso15926")
            assert resp.status_code == 401

    @pytest.mark.asyncio
    async def test_unknown_session_404(self, auth_client: AsyncClient):
        resp = await auth_client.get("/sessions/does-not-exist/iso15926")
        assert resp.status_code == 404

    @pytest.mark.asyncio
    async def test_no_instructions_422(self, auth_client: AsyncClient):
        from agent.main import orchestrator

        sess = orchestrator.create_session("ISO EP 1")
        resp = await auth_client.get(f"/sessions/{sess.session_id}/iso15926")
        assert resp.status_code == 422

    @pytest.mark.asyncio
    async def test_json_format(self, auth_client: AsyncClient):
        from agent.main import orchestrator

        sess = orchestrator.create_session("ISO EP 2")
        orchestrator._instructions[sess.session_id] = [
            _instruction(ElementType.PIPE),
            _instruction(ElementType.PUMP),
        ]
        resp = await auth_client.get(
            f"/sessions/{sess.session_id}/iso15926",
            params={"format": "json"},
        )
        assert resp.status_code == 200
        body = resp.json()
        assert body["total_elements"] == 2
        assert "mappings" in body

    @pytest.mark.asyncio
    async def test_csv_format(self, auth_client: AsyncClient):
        from agent.main import orchestrator

        sess = orchestrator.create_session("ISO EP 3")
        orchestrator._instructions[sess.session_id] = [
            _instruction(ElementType.PIPE),
        ]
        resp = await auth_client.get(
            f"/sessions/{sess.session_id}/iso15926",
            params={"format": "csv"},
        )
        assert resp.status_code == 200
        assert resp.headers["content-type"].startswith("text/csv")
        assert "attachment" in resp.headers["content-disposition"]
        assert "tag_number" in resp.text


# ═════════════════════════════════════════════════════════════════════════════
# 5. Handover bundle integration
# ═════════════════════════════════════════════════════════════════════════════


class TestIso15926InBundle:
    def test_iso15926_section_included_when_instructions_exist(self):
        from agent.main import audit, ncr_service, orchestrator
        from agent.tools.handover_bundle import build_handover_bundle

        sess = orchestrator.create_session("ISO Bundle 1")
        orchestrator._instructions[sess.session_id] = [
            _instruction(ElementType.PIPE),
            _instruction(ElementType.VALVE),
        ]
        zip_bytes, sections = build_handover_bundle(
            sess.session_id,
            orchestrator,
            audit,
            ncr_service,
        )
        zf = zipfile.ZipFile(io.BytesIO(zip_bytes))
        files = zf.namelist()

        assert "09_iso15926/iso15926_mapping.json" in files
        assert "09_iso15926/cfihos_attributes.csv" in files

        iso_section = next(s for s in sections if s.name == "09_iso15926")
        assert iso_section.included is True

    def test_iso15926_skipped_without_instructions(self):
        from agent.main import audit, ncr_service, orchestrator
        from agent.tools.handover_bundle import build_handover_bundle

        sess = orchestrator.create_session("ISO Bundle 2")
        _, sections = build_handover_bundle(
            sess.session_id,
            orchestrator,
            audit,
            ncr_service,
        )
        iso_section = next(s for s in sections if s.name == "09_iso15926")
        assert iso_section.included is False
        assert "classify_and_prepare" in iso_section.reason
