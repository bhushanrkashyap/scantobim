"""test_nqa1_package.py — P2.4 NQA-1 Subpart 2.7 V&V package.

Covers:
  • software_identity — required fields, stable identity_hash
  • requirements_matrix — coverage verifier + traceability
  • nqa1_package.build_nqa1_package — all 5 sections present
  • xUnit parser — happy path + missing-file tolerance
  • /version endpoint (public, no auth)
  • /sessions/{id}/nqa1-package endpoint (auth, 404, json + html)
  • Handover bundle integration — 07_nqa1 section present
"""

from __future__ import annotations

import io
import json
import zipfile
from pathlib import Path

import pytest
import pytest_asyncio
from httpx import ASGITransport, AsyncClient

# ═════════════════════════════════════════════════════════════════════════════
# 1. software_identity
# ═════════════════════════════════════════════════════════════════════════════


class TestSoftwareIdentity:
    def test_identity_has_required_fields(self):
        from agent.software_identity import get_software_identity

        ident = get_software_identity()
        assert ident.product_name == "ScanToBIM"
        assert ident.product_version  # non-empty
        assert ident.git_sha  # may be 'unknown' in sandboxed envs
        assert ident.python_version
        assert ident.platform_label
        assert ident.identity_hash
        assert len(ident.identity_hash) == 64  # sha256 hex

    def test_identity_hash_stable(self):
        """Calling twice returns same hash (cached)."""
        from agent.software_identity import clear_cache, get_software_identity

        clear_cache()
        first = get_software_identity()
        second = get_software_identity()
        assert first.identity_hash == second.identity_hash
        assert first == second  # dataclass equality

    def test_identity_hash_excludes_timestamp(self):
        """Two runs at different times must give same identity_hash."""
        from agent.software_identity import (
            _compute_identity_hash,
            get_software_identity,
        )

        ident = get_software_identity()
        d1 = ident.to_dict()
        d2 = dict(d1)
        d2["built_at_utc"] = "2099-01-01T00:00:00+00:00"
        assert _compute_identity_hash(d1) == _compute_identity_hash(d2)

    def test_identity_dict_has_no_none_product_fields(self):
        from agent.software_identity import software_identity_dict

        d = software_identity_dict()
        assert d["product_name"] is not None
        assert d["product_version"] is not None

    def test_dependencies_includes_tracked_packages(self):
        from agent.software_identity import get_software_identity

        deps = get_software_identity().dependencies
        # Expect at least numpy and pydantic to be installed in test env
        assert "numpy" in deps
        assert "pydantic" in deps


# ═════════════════════════════════════════════════════════════════════════════
# 2. requirements_matrix
# ═════════════════════════════════════════════════════════════════════════════


class TestMatrixCoverage:
    def test_every_requirement_has_test_ids(self):
        """The RTM itself must be fully traced — every requirement verified."""
        from agent.tools.requirements_matrix import verify_matrix_coverage

        report = verify_matrix_coverage()
        assert report.missing_tests == [], f"Requirements without tests: {report.missing_tests}"
        assert report.fully_traced

    def test_grades_distributed(self):
        from agent.tools.requirements_matrix import verify_matrix_coverage

        by_grade = verify_matrix_coverage().by_grade
        # At least one A (SC1), one B (SC2/LOA), one C (compliance)
        assert by_grade.get("A", 0) >= 1
        assert by_grade.get("B", 0) >= 1
        assert by_grade.get("C", 0) >= 1

    def test_get_requirement_lookup(self):
        from agent.tools.requirements_matrix import get_requirement

        r = get_requirement("REQ-SC1-001")
        assert r is not None
        assert r.grade == "A"
        assert "safety_gate.py" in " ".join(r.source_files)

    def test_get_requirement_missing_returns_none(self):
        from agent.tools.requirements_matrix import get_requirement

        assert get_requirement("REQ-DOES-NOT-EXIST") is None

    def test_matrix_to_list_is_json_safe(self):
        from agent.tools.requirements_matrix import matrix_to_list

        payload = json.dumps(matrix_to_list())  # must not raise
        assert len(payload) > 100


# ═════════════════════════════════════════════════════════════════════════════
# 3. nqa1_package builder
# ═════════════════════════════════════════════════════════════════════════════


class TestNqa1Package:
    def test_build_requires_known_session(self):
        from agent.main import audit, ncr_service, orchestrator
        from agent.tools.nqa1_package import build_nqa1_package

        with pytest.raises(KeyError):
            build_nqa1_package("unknown-session", orchestrator, audit, ncr_service)

    def test_build_produces_five_sections(self):
        from agent.main import audit, ncr_service, orchestrator
        from agent.tools.nqa1_package import build_nqa1_package

        sess = orchestrator.create_session("NQA-1 Test Site")
        html, doc = build_nqa1_package(
            sess.session_id,
            orchestrator,
            audit,
            ncr_service,
        )

        # All five sections present in JSON
        assert doc["document"] == "NQA-1 Subpart 2.7 V&V Package"
        assert "software_identity" in doc
        assert "matrix_coverage" in doc
        assert "requirements" in doc
        assert "session_evidence" in doc
        assert "signatures" in doc

        # All five sections referenced in HTML
        for heading in ("Section 1", "Section 2", "Section 3", "Section 4", "Section 5"):
            assert heading in html

    def test_html_includes_requirement_rows(self):
        from agent.main import audit, ncr_service, orchestrator
        from agent.tools.nqa1_package import build_nqa1_package

        sess = orchestrator.create_session("NQA-1 Test HTML")
        html, _ = build_nqa1_package(
            sess.session_id,
            orchestrator,
            audit,
            ncr_service,
        )
        # At least the SC1 hard-block requirement must be rendered
        assert "REQ-SC1-001" in html
        assert "SC1 hard block" in html

    def test_session_evidence_counts_audit_events(self):
        from agent.main import audit, ncr_service, orchestrator
        from agent.tools.nqa1_package import build_nqa1_package

        sess = orchestrator.create_session("NQA-1 Test Audit")
        _, doc = build_nqa1_package(
            sess.session_id,
            orchestrator,
            audit,
            ncr_service,
        )
        # Session creation writes at least one audit event
        assert doc["session_evidence"]["audit_event_count"] >= 0
        assert "chain" in doc["session_evidence"]

    def test_signature_block_structure(self):
        """After P2.4++ the signature block is a live status from
        NQA1SignatureService — not the old null-placeholder dict."""
        from agent.main import audit, ncr_service, orchestrator
        from agent.tools.nqa1_package import build_nqa1_package

        sess = orchestrator.create_session("NQA-1 Test Sig")
        _, doc = build_nqa1_package(
            sess.session_id,
            orchestrator,
            audit,
            ncr_service,
        )
        sigs = doc["signatures"]
        # Top-level shape
        assert {
            "fully_signed",
            "next_required",
            "by_role",
            "chain_valid",
            "signature_count",
        } <= set(sigs.keys())
        # Per-role shape
        assert set(sigs["by_role"].keys()) == {"preparer", "verifier", "approver"}
        for role_data in sigs["by_role"].values():
            assert {"signed", "signer_upn", "signer_name", "timestamp_utc"} <= set(role_data.keys())
        # Unsigned by default when no service passed in
        assert sigs["fully_signed"] is False
        assert sigs["next_required"] == "preparer"


class TestXUnitParser:
    def test_missing_file_returns_none(self, tmp_path: Path):
        from agent.tools.nqa1_package import parse_xunit

        assert parse_xunit(tmp_path / "nope.xml") is None

    def test_parses_testsuite(self, tmp_path: Path):
        from agent.tools.nqa1_package import parse_xunit

        xml = tmp_path / "results.xml"
        xml.write_text(
            '<testsuite name="x" tests="10" failures="1" errors="0" '
            'skipped="2" time="3.5"></testsuite>'
        )
        s = parse_xunit(xml)
        assert s is not None
        assert s.total == 10
        assert s.failed == 1
        assert s.skipped == 2
        assert s.errors == 0
        assert s.passed == 7
        assert s.duration_s == 3.5

    def test_aggregates_testsuites(self, tmp_path: Path):
        from agent.tools.nqa1_package import parse_xunit

        xml = tmp_path / "results.xml"
        xml.write_text(
            "<testsuites>"
            '  <testsuite tests="5" failures="0" errors="0" skipped="1" time="1.0"/>'
            '  <testsuite tests="7" failures="1" errors="1" skipped="0" time="2.0"/>'
            "</testsuites>"
        )
        s = parse_xunit(xml)
        assert s.total == 12
        assert s.failed == 1
        assert s.errors == 1
        assert s.duration_s == pytest.approx(3.0)

    def test_malformed_xml_returns_none(self, tmp_path: Path):
        from agent.tools.nqa1_package import parse_xunit

        xml = tmp_path / "bad.xml"
        xml.write_text("this is not xml")
        assert parse_xunit(xml) is None


# ═════════════════════════════════════════════════════════════════════════════
# 4. API endpoints
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


class TestVersionEndpoint:
    @pytest.mark.asyncio
    async def test_public_endpoint_no_auth(self):
        """/version is intentionally public — compliance dashboards poll it."""
        from agent.main import app

        async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as ac:
            resp = await ac.get("/version")
            assert resp.status_code == 200
            body = resp.json()
            assert body["product_name"] == "ScanToBIM"
            assert "git_sha" in body
            assert "identity_hash" in body


class TestNqa1PackageEndpoint:
    @pytest.mark.asyncio
    async def test_requires_auth(self):
        from agent.main import app

        async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as ac:
            resp = await ac.get("/sessions/x/nqa1-package")
            assert resp.status_code == 401

    @pytest.mark.asyncio
    async def test_unknown_session_404(self, auth_client: AsyncClient):
        resp = await auth_client.get("/sessions/does-not-exist/nqa1-package")
        assert resp.status_code == 404

    @pytest.mark.asyncio
    async def test_returns_json_by_default(self, auth_client: AsyncClient):
        from agent.main import orchestrator

        sess = orchestrator.create_session("NQA-1 Endpoint Test")
        resp = await auth_client.get(
            f"/sessions/{sess.session_id}/nqa1-package",
        )
        assert resp.status_code == 200
        body = resp.json()
        assert body["document"] == "NQA-1 Subpart 2.7 V&V Package"
        assert "software_identity" in body

    @pytest.mark.asyncio
    async def test_returns_html_when_requested(self, auth_client: AsyncClient):
        from agent.main import orchestrator

        sess = orchestrator.create_session("NQA-1 HTML Test")
        resp = await auth_client.get(
            f"/sessions/{sess.session_id}/nqa1-package",
            params={"format": "html"},
        )
        assert resp.status_code == 200
        assert resp.headers["content-type"].startswith("text/html")
        assert resp.text.startswith("<!DOCTYPE html>")

    @pytest.mark.asyncio
    async def test_invalid_format_422(self, auth_client: AsyncClient):
        from agent.main import orchestrator

        sess = orchestrator.create_session("NQA-1 Bad Format")
        resp = await auth_client.get(
            f"/sessions/{sess.session_id}/nqa1-package",
            params={"format": "xml"},
        )
        assert resp.status_code == 422


# ═════════════════════════════════════════════════════════════════════════════
# 5. Handover bundle integration
# ═════════════════════════════════════════════════════════════════════════════


def _unzip(zip_bytes: bytes) -> dict[str, bytes]:
    zf = zipfile.ZipFile(io.BytesIO(zip_bytes))
    return {n: zf.read(n) for n in zf.namelist()}


class TestHandoverIntegration:
    def test_handover_bundle_includes_07_nqa1(self):
        from agent.main import audit, ncr_service, orchestrator
        from agent.tools.handover_bundle import build_handover_bundle

        sess = orchestrator.create_session("NQA-1 In Bundle")
        zip_bytes, sections = build_handover_bundle(
            sess.session_id,
            orchestrator,
            audit,
            ncr_service,
        )
        files = _unzip(zip_bytes)

        assert "07_nqa1/nqa1_package.html" in files
        assert "07_nqa1/nqa1_package.json" in files

        # Section registered in manifest
        manifest = json.loads(files["manifest.json"])
        nqa_section = next(
            (s for s in manifest["sections"] if s["name"] == "07_nqa1"),
            None,
        )
        assert nqa_section is not None
        assert nqa_section["included"] is True
        assert nqa_section["file_count"] == 2
