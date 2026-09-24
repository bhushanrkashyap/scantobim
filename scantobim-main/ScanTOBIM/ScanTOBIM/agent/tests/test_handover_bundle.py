"""test_handover_bundle.py — P2.2d consolidated handover zip.

Covers:
  • Empty session → sections skipped gracefully, bundle still valid
  • Session with segments → 02_loa present; 01_ifc skipped (no instructions)
  • Session with instructions → 01_ifc + 02_loa present
  • Session with registration → 04_registration present
  • Session with deviation report → 03_deviation present
  • Manifest + README always present
  • README lists included vs skipped sections
  • GET /sessions/{id}/handover-bundle endpoint (auth, 404, zip response)
"""

from __future__ import annotations

import io
import json
import uuid
import zipfile
from datetime import datetime, timezone

import pytest
import pytest_asyncio
from httpx import ASGITransport, AsyncClient

# ── Helpers ──────────────────────────────────────────────────────────────────


def _segment(x=1000.0, y=500.0, z=1500.0, seg_id=None, loa_sigma=None):
    from agent.models import BoundingBox, GeometrySegment, Point3D, SegmentShape

    tags = {}
    if loa_sigma is not None:
        tags["loa_sigma_mm"] = loa_sigma
    return GeometrySegment(
        segment_id=seg_id or str(uuid.uuid4()),
        zone_id="zone-A",
        shape=SegmentShape.BOX,
        centroid=Point3D(x=x, y=y, z=z),
        bounding_box=BoundingBox(
            min_x=x - 250,
            min_y=y - 250,
            min_z=z - 500,
            max_x=x + 250,
            max_y=y + 250,
            max_z=z + 500,
        ),
        point_count=500,
        confidence=0.85,
        tags=tags,
    )


def _registration_result(session_id: str):
    from agent.models import StationRegistrationDetail, StationRegistrationResult

    return StationRegistrationResult(
        session_id=session_id,
        station_count=2,
        stations=[
            StationRegistrationDetail(
                station_index=0,
                source_file="north.e57",
                transform_4x4=[[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]],
                rmse_mm=1.8,
                inlier_ratio=0.94,
            ),
            StationRegistrationDetail(
                station_index=1,
                source_file="south.e57",
                transform_4x4=[[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]],
                rmse_mm=2.2,
                inlier_ratio=0.91,
            ),
        ],
        global_rmse_mm=2.0,
        merged_point_count=500_000,
        registered_at=datetime.now(timezone.utc),
    )


def _deviation_report(session_id: str):
    from agent.models import (
        DeviationReport,
        DeviationStatus,
        ElementDeviationRecord,
    )

    return DeviationReport(
        session_id=session_id,
        generated_at=datetime.now(timezone.utc),
        ifc_element_count=2,
        scan_segment_count=2,
        matched_count=1,
        shifted_count=1,
        missing_count=0,
        extra_count=0,
        records=[
            ElementDeviationRecord(
                ifc_global_id="A",
                ifc_class="IfcWall",
                ifc_name="W1",
                scan_segment_id="s1",
                distance_mm=8.0,
                status=DeviationStatus.MATCHED,
                ifc_centroid_mm=[0, 0, 0],
                scan_centroid_mm=[0, 8, 0],
            ),
            ElementDeviationRecord(
                ifc_global_id="B",
                ifc_class="IfcWall",
                ifc_name="W2",
                scan_segment_id="s2",
                distance_mm=40.0,
                status=DeviationStatus.SHIFTED,
                ifc_centroid_mm=[1000, 0, 0],
                scan_centroid_mm=[1040, 0, 0],
            ),
        ],
    )


def _unzip(zip_bytes: bytes) -> dict[str, bytes]:
    """Return {name: bytes} mapping from a zip archive."""
    zf = zipfile.ZipFile(io.BytesIO(zip_bytes))
    return {n: zf.read(n) for n in zf.namelist()}


# ═════════════════════════════════════════════════════════════════════════════
# 1. build_handover_bundle — direct unit tests
# ═════════════════════════════════════════════════════════════════════════════


class TestEmptySession:
    def test_empty_session_still_produces_valid_zip(self):
        from agent.main import audit, ncr_service, orchestrator
        from agent.tools.handover_bundle import build_handover_bundle

        sess = orchestrator.create_session("Handover Empty 1")
        zip_bytes, sections = build_handover_bundle(
            sess.session_id,
            orchestrator,
            audit,
            ncr_service,
        )

        # zip should still open, and contain manifest + README
        files = _unzip(zip_bytes)
        assert "manifest.json" in files
        assert "README.md" in files

        # Quality section always present (audit trail exists even if empty)
        assert any(name.startswith("05_quality/") for name in files)

    def test_unknown_session_raises_key_error(self):
        from agent.main import audit, ncr_service, orchestrator
        from agent.tools.handover_bundle import build_handover_bundle

        with pytest.raises(KeyError):
            build_handover_bundle(
                "does-not-exist",
                orchestrator,
                audit,
                ncr_service,
            )

    def test_sections_mark_skipped_when_no_data(self):
        from agent.main import audit, ncr_service, orchestrator
        from agent.tools.handover_bundle import build_handover_bundle

        sess = orchestrator.create_session("Handover Empty 2")
        _, sections = build_handover_bundle(
            sess.session_id,
            orchestrator,
            audit,
            ncr_service,
        )

        # IFC, LOA, deviation, registration should all be skipped
        skipped_names = {s.name for s in sections if not s.included}
        assert "01_ifc" in skipped_names
        assert "02_loa" in skipped_names
        assert "03_deviation" in skipped_names
        assert "04_registration" in skipped_names
        # Quality is always included (audit ledger exists regardless)
        assert any(s.name == "05_quality" and s.included for s in sections)


class TestPartialSession:
    def test_session_with_segments_includes_loa(self):
        from agent.main import audit, ncr_service, orchestrator
        from agent.tools.handover_bundle import build_handover_bundle

        sess = orchestrator.create_session("Handover LOA 1")
        orchestrator._segments[sess.session_id] = [
            _segment(loa_sigma=1.5),  # LOA30
            _segment(loa_sigma=0.3),  # LOA40
        ]

        zip_bytes, sections = build_handover_bundle(
            sess.session_id,
            orchestrator,
            audit,
            ncr_service,
        )
        files = _unzip(zip_bytes)

        assert "02_loa/loa_report.json" in files
        loa_section = next(s for s in sections if s.name == "02_loa")
        assert loa_section.included

        # Content valid JSON with expected keys
        loa = json.loads(files["02_loa/loa_report.json"])
        assert loa["total_segments"] == 2
        assert loa["standard_version"] == "USIBD LOA v3.1"

    def test_session_with_registration_includes_reg_qa(self):
        from agent.main import audit, ncr_service, orchestrator
        from agent.tools.handover_bundle import build_handover_bundle

        sess = orchestrator.create_session("Handover Reg 1")
        orchestrator._registration[sess.session_id] = _registration_result(
            sess.session_id,
        )

        zip_bytes, _ = build_handover_bundle(
            sess.session_id,
            orchestrator,
            audit,
            ncr_service,
        )
        files = _unzip(zip_bytes)

        assert "04_registration/registration_qa.json" in files
        assert "04_registration/registration_qa.html" in files

        qa = json.loads(files["04_registration/registration_qa.json"])
        assert qa["station_count"] == 2
        assert qa["overall_category"] == "A"

    def test_session_with_deviation_report_includes_heatmaps(self):
        from agent.main import audit, ncr_service, orchestrator
        from agent.tools.handover_bundle import build_handover_bundle

        sess = orchestrator.create_session("Handover Dev 1")
        orchestrator._deviation_reports[sess.session_id] = _deviation_report(
            sess.session_id,
        )

        zip_bytes, _ = build_handover_bundle(
            sess.session_id,
            orchestrator,
            audit,
            ncr_service,
        )
        files = _unzip(zip_bytes)

        assert "03_deviation/deviation_report.json" in files
        assert "03_deviation/heatmap_plan.svg" in files
        assert "03_deviation/heatmap_elevation.svg" in files
        assert "03_deviation/heatmap_summary.json" in files

        # SVG actually contains drawn circles
        plan_svg = files["03_deviation/heatmap_plan.svg"].decode()
        assert "<svg" in plan_svg
        assert "<circle" in plan_svg

    def test_segments_plus_deviation_generates_loa_heatmap(self):
        from agent.main import audit, ncr_service, orchestrator
        from agent.tools.handover_bundle import build_handover_bundle

        sess = orchestrator.create_session("Handover Dev+LOA 1")
        orchestrator._segments[sess.session_id] = [
            _segment(seg_id="s1", loa_sigma=1.0),
            _segment(seg_id="s2", loa_sigma=3.0),
        ]
        orchestrator._deviation_reports[sess.session_id] = _deviation_report(
            sess.session_id,
        )

        zip_bytes, _ = build_handover_bundle(
            sess.session_id,
            orchestrator,
            audit,
            ncr_service,
        )
        files = _unzip(zip_bytes)
        # LOA-coloured variant should appear when segments exist
        assert "03_deviation/heatmap_plan_loa.svg" in files


# ═════════════════════════════════════════════════════════════════════════════
# 2. Manifest + README
# ═════════════════════════════════════════════════════════════════════════════


class TestManifestAndReadme:
    def test_manifest_schema(self):
        from agent.main import audit, ncr_service, orchestrator
        from agent.tools.handover_bundle import build_handover_bundle

        sess = orchestrator.create_session("Handover Manifest 1")
        zip_bytes, _ = build_handover_bundle(
            sess.session_id,
            orchestrator,
            audit,
            ncr_service,
        )
        manifest = json.loads(_unzip(zip_bytes)["manifest.json"])

        assert manifest["session_id"] == sess.session_id
        assert manifest["schema_version"] == "1.0"
        assert isinstance(manifest["sections"], list)
        # 7 sections: 01_ifc … 07_nqa1 (P2.4 added NQA-1 V&V package)
        assert len(manifest["sections"]) == 9  # + FMEA (08_fmea) + ISO 15926 (09_iso15926)
        section_names = {s["name"] for s in manifest["sections"]}
        assert {
            "01_ifc",
            "02_loa",
            "03_deviation",
            "04_registration",
            "05_quality",
            "06_drp",
            "07_nqa1",
        } <= section_names

        for s in manifest["sections"]:
            assert {"name", "title", "included", "file_count", "reason"} <= set(s.keys())

    def test_readme_lists_included_and_skipped(self):
        from agent.main import audit, ncr_service, orchestrator
        from agent.tools.handover_bundle import build_handover_bundle

        sess = orchestrator.create_session("Handover README 1")
        orchestrator._segments[sess.session_id] = [_segment(loa_sigma=1.0)]

        zip_bytes, _ = build_handover_bundle(
            sess.session_id,
            orchestrator,
            audit,
            ncr_service,
        )
        readme = _unzip(zip_bytes)["README.md"].decode()

        assert "Contents — included" in readme
        assert "Not present" in readme
        # Included: USIBD LOA
        assert "USIBD LOA" in readme
        # Skipped: IFC export
        assert "IFC export" in readme


# ═════════════════════════════════════════════════════════════════════════════
# 3. API endpoint
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


class TestHandoverEndpoint:
    @pytest.mark.asyncio
    async def test_requires_auth(self):
        from agent.main import app

        async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as ac:
            resp = await ac.get("/sessions/any/handover-bundle")
            assert resp.status_code == 401

    @pytest.mark.asyncio
    async def test_unknown_session_404(self, auth_client: AsyncClient):
        resp = await auth_client.get("/sessions/does-not-exist/handover-bundle")
        assert resp.status_code == 404

    @pytest.mark.asyncio
    async def test_returns_zip(self, auth_client: AsyncClient):
        from agent.main import orchestrator

        sess = orchestrator.create_session("Handover Endpoint 1")
        orchestrator._segments[sess.session_id] = [_segment(loa_sigma=1.0)]

        resp = await auth_client.get(
            f"/sessions/{sess.session_id}/handover-bundle",
        )
        assert resp.status_code == 200
        assert resp.headers["content-type"] == "application/zip"
        assert "filename=" in resp.headers["content-disposition"]

        # Response body is a valid zip with LOA present, IFC skipped
        files = _unzip(resp.content)
        assert "manifest.json" in files
        assert "02_loa/loa_report.json" in files
        assert "01_ifc/model.ifc" not in files

    @pytest.mark.asyncio
    async def test_custom_header_reports_sections(self, auth_client: AsyncClient):
        from agent.main import orchestrator

        sess = orchestrator.create_session("Handover Endpoint 2")
        orchestrator._segments[sess.session_id] = [_segment(loa_sigma=1.0)]

        resp = await auth_client.get(
            f"/sessions/{sess.session_id}/handover-bundle",
        )
        header = resp.headers.get("x-scantobim-sections", "")
        # Expected format: 01_ifc=skip,02_loa=ok,...
        assert "02_loa=ok" in header
        assert "01_ifc=skip" in header


# ═════════════════════════════════════════════════════════════════════════════
# 4. NQA-1 section inside the bundle (P2.4++ follow-up)
# ═════════════════════════════════════════════════════════════════════════════


class TestNqa1SectionInBundle:
    def test_nqa1_section_included_when_signature_service_passed(self):
        from agent.main import audit, ncr_service, nqa1_signatures, orchestrator
        from agent.tools.handover_bundle import build_handover_bundle

        sess = orchestrator.create_session("Handover NQA1 1")
        zip_bytes, _ = build_handover_bundle(
            sess.session_id,
            orchestrator,
            audit,
            ncr_service,
            signature_service=nqa1_signatures,
        )
        files = _unzip(zip_bytes)
        assert "07_nqa1/nqa1_package.html" in files
        assert "07_nqa1/nqa1_package.json" in files
        # Signature files only appear when service is wired
        assert "07_nqa1/signatures.json" in files
        assert "07_nqa1/signature_status.json" in files
        assert "07_nqa1/signature_chain_integrity.json" in files

    def test_signature_status_reflects_live_service(self):
        from agent.main import (
            _auto_preparer_sign,
            audit,
            ncr_service,
            nqa1_signatures,
            orchestrator,
        )
        from agent.tools.handover_bundle import build_handover_bundle

        sess = orchestrator.create_session("Handover NQA1 2")
        orchestrator.run_segmentation(
            sess.session_id,
            use_synthetic=True,
            zone_id="Z",
        )
        _auto_preparer_sign(sess.session_id)

        zip_bytes, _ = build_handover_bundle(
            sess.session_id,
            orchestrator,
            audit,
            ncr_service,
            signature_service=nqa1_signatures,
        )
        files = _unzip(zip_bytes)

        status = json.loads(files["07_nqa1/signature_status.json"])
        assert status["by_role"]["preparer"]["signed"] is True
        assert status["next_required"] == "verifier"

    def test_signature_files_skipped_without_service(self):
        from agent.main import audit, ncr_service, orchestrator
        from agent.tools.handover_bundle import build_handover_bundle

        sess = orchestrator.create_session("Handover NQA1 3")
        zip_bytes, _ = build_handover_bundle(
            sess.session_id,
            orchestrator,
            audit,
            ncr_service,
            # no signature_service
        )
        files = _unzip(zip_bytes)
        assert "07_nqa1/nqa1_package.html" in files
        assert "07_nqa1/signatures.json" not in files

    @pytest.mark.asyncio
    async def test_endpoint_includes_signatures(self, auth_client: AsyncClient):
        from agent.main import orchestrator

        sess = orchestrator.create_session("Handover Endpoint NQA1")
        orchestrator.run_segmentation(sess.session_id, use_synthetic=True, zone_id="Z")

        resp = await auth_client.get(
            f"/sessions/{sess.session_id}/handover-bundle",
        )
        assert resp.status_code == 200
        files = _unzip(resp.content)
        assert "07_nqa1/signatures.json" in files
        assert "07_nqa1/signature_chain_integrity.json" in files

        chain = json.loads(files["07_nqa1/signature_chain_integrity.json"])
        assert chain["valid"] is True
        # Endpoint auto-signs preparer, so at least 1 signature
        assert chain["signature_count"] >= 1
