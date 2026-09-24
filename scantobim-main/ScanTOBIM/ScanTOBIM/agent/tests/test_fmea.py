"""test_fmea.py — Failure Mode and Effects Analysis report.

Covers:
  • FailureMode.rpn computed correctly
  • risk_tier thresholds (HIGH ≥ 200, MEDIUM ≥ 100, LOW ≥ 50)
  • Catalog integrity — every entry has tests cited
  • build_fmea_document returns expected shape
  • summarise() counts tiers + flags high-RPN items
  • render_fmea_html produces valid HTML with every mode + XSS escaping
  • GET /sessions/{id}/fmea endpoint (auth, 404, HTML, JSON)
  • FMEA included in handover bundle
"""

from __future__ import annotations

import json

import pytest
import pytest_asyncio
from httpx import ASGITransport, AsyncClient

from agent.tools.fmea import (
    FAILURE_MODES,
    FailureMode,
    build_fmea_document,
    render_fmea_html,
    summarise,
)

# ═════════════════════════════════════════════════════════════════════════════
# 1. FailureMode semantics
# ═════════════════════════════════════════════════════════════════════════════


class TestFailureMode:
    def _fm(self, s=5, l=5, d=5):
        return FailureMode(
            fmea_id="TEST-1",
            function="test.fn",
            failure_mode="test",
            effect_local="",
            effect_system="",
            severity=s,
            likelihood=l,
            detection=d,
            mitigation="",
            test_ids=("a",),
        )

    def test_rpn_is_product_of_three(self):
        fm = self._fm(s=6, l=4, d=3)
        assert fm.rpn == 72

    def test_high_tier_at_200(self):
        fm = self._fm(s=10, l=10, d=2)  # rpn=200
        assert fm.risk_tier == "HIGH"

    def test_medium_tier_100_to_199(self):
        fm = self._fm(s=6, l=6, d=3)  # rpn=108
        assert fm.risk_tier == "MEDIUM"

    def test_low_tier_50_to_99(self):
        fm = self._fm(s=5, l=5, d=2)  # rpn=50
        assert fm.risk_tier == "LOW"

    def test_negligible_under_50(self):
        fm = self._fm(s=3, l=3, d=3)  # rpn=27
        assert fm.risk_tier == "NEGLIGIBLE"

    def test_to_dict_includes_rpn_and_tier(self):
        fm = self._fm(s=6, l=6, d=3)
        d = fm.to_dict()
        assert d["rpn"] == 108
        assert d["risk_tier"] == "MEDIUM"
        assert isinstance(d["test_ids"], list)


# ═════════════════════════════════════════════════════════════════════════════
# 2. Catalog integrity
# ═════════════════════════════════════════════════════════════════════════════


class TestCatalog:
    def test_catalog_non_empty(self):
        assert len(FAILURE_MODES) >= 10

    def test_every_mode_has_unique_id(self):
        ids = [m.fmea_id for m in FAILURE_MODES]
        assert len(ids) == len(set(ids)), "duplicate FMEA ID detected"

    def test_every_mode_has_tests_cited(self):
        """Every failure mode must name at least one verifying test."""
        missing = [m.fmea_id for m in FAILURE_MODES if not m.test_ids]
        assert missing == [], f"FMEA entries without tests: {missing}"

    def test_scores_in_range(self):
        for m in FAILURE_MODES:
            for k, v in [
                ("severity", m.severity),
                ("likelihood", m.likelihood),
                ("detection", m.detection),
            ]:
                assert 1 <= v <= 10, f"{m.fmea_id}: {k}={v} out of 1..10"

    def test_high_rpn_have_strong_mitigations(self):
        """Every HIGH-RPN mode must have a non-empty mitigation."""
        for m in FAILURE_MODES:
            if m.risk_tier == "HIGH":
                assert m.mitigation, f"{m.fmea_id} HIGH but no mitigation"


# ═════════════════════════════════════════════════════════════════════════════
# 3. summarise() + build_fmea_document()
# ═════════════════════════════════════════════════════════════════════════════


class TestSummariseAndBuild:
    def test_summarise_shape(self):
        s = summarise(FAILURE_MODES)
        assert s["total_modes"] == len(FAILURE_MODES)
        assert set(s["by_tier"].keys()) == {"HIGH", "MEDIUM", "LOW", "NEGLIGIBLE"}
        assert sum(s["by_tier"].values()) == s["total_modes"]
        assert s["max_rpn"] >= 0
        assert isinstance(s["high_rpn_ids"], list)

    def test_build_document_shape(self):
        doc = build_fmea_document("sess-abc")
        assert doc["session_id"] == "sess-abc"
        assert doc["document"].startswith("FMEA")
        assert "NQA-1" in " ".join(doc["standard_refs"])
        assert len(doc["failure_modes"]) == len(FAILURE_MODES)
        assert "summary" in doc
        assert "scoring_guides" in doc
        for key in ("severity", "likelihood", "detection"):
            assert key in doc["scoring_guides"]


# ═════════════════════════════════════════════════════════════════════════════
# 4. HTML rendering
# ═════════════════════════════════════════════════════════════════════════════


class TestRenderFmeaHtml:
    def test_produces_valid_html(self):
        html = render_fmea_html(build_fmea_document("sess-1"))
        assert html.startswith("<!DOCTYPE html>")
        assert "</html>" in html

    def test_every_failure_mode_renders(self):
        html = render_fmea_html(build_fmea_document("sess-1"))
        for m in FAILURE_MODES:
            assert m.fmea_id in html

    def test_tier_colours_applied(self):
        html = render_fmea_html(build_fmea_document("sess-1"))
        # At least one tier label must appear
        assert ("HIGH" in html) or ("MEDIUM" in html) or ("LOW" in html)

    def test_escapes_special_chars(self):
        """User-supplied session_id is XML-escaped."""
        doc = build_fmea_document("<script>alert(1)</script>")
        html = render_fmea_html(doc)
        assert "<script>alert" not in html
        assert "&lt;script&gt;" in html


# ═════════════════════════════════════════════════════════════════════════════
# 5. API endpoint
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


class TestFmeaEndpoint:
    @pytest.mark.asyncio
    async def test_requires_auth(self):
        from agent.main import app

        async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as ac:
            resp = await ac.get("/sessions/x/fmea")
            assert resp.status_code == 401

    @pytest.mark.asyncio
    async def test_unknown_session_404(self, auth_client: AsyncClient):
        resp = await auth_client.get("/sessions/does-not-exist/fmea")
        assert resp.status_code == 404

    @pytest.mark.asyncio
    async def test_html_format(self, auth_client: AsyncClient):
        from agent.main import orchestrator

        sess = orchestrator.create_session("FMEA Endpoint 1")
        resp = await auth_client.get(
            f"/sessions/{sess.session_id}/fmea",
            params={"format": "html"},
        )
        assert resp.status_code == 200
        assert resp.headers["content-type"].startswith("text/html")
        assert "FMEA" in resp.text

    @pytest.mark.asyncio
    async def test_json_format(self, auth_client: AsyncClient):
        from agent.main import orchestrator

        sess = orchestrator.create_session("FMEA Endpoint 2")
        resp = await auth_client.get(
            f"/sessions/{sess.session_id}/fmea",
            params={"format": "json"},
        )
        assert resp.status_code == 200
        body = resp.json()
        assert body["session_id"] == sess.session_id
        assert len(body["failure_modes"]) >= 10

    @pytest.mark.asyncio
    async def test_invalid_format_422(self, auth_client: AsyncClient):
        from agent.main import orchestrator

        sess = orchestrator.create_session("FMEA Endpoint 3")
        resp = await auth_client.get(
            f"/sessions/{sess.session_id}/fmea",
            params={"format": "csv"},
        )
        assert resp.status_code == 422


# ═════════════════════════════════════════════════════════════════════════════
# 6. Handover bundle integration
# ═════════════════════════════════════════════════════════════════════════════


class TestFmeaInBundle:
    def test_fmea_section_always_included(self):
        """FMEA is static — it should always render regardless of session state."""
        import io
        import zipfile

        from agent.main import audit, ncr_service, orchestrator
        from agent.tools.handover_bundle import build_handover_bundle

        sess = orchestrator.create_session("Bundle FMEA 1")
        zip_bytes, sections = build_handover_bundle(
            sess.session_id,
            orchestrator,
            audit,
            ncr_service,
        )
        zf = zipfile.ZipFile(io.BytesIO(zip_bytes))
        files = zf.namelist()

        assert "08_fmea/fmea_report.html" in files
        assert "08_fmea/fmea_report.json" in files

        fmea_section = next(s for s in sections if s.name == "08_fmea")
        assert fmea_section.included is True

    def test_fmea_json_matches_catalog(self):
        import io
        import zipfile

        from agent.main import audit, ncr_service, orchestrator
        from agent.tools.handover_bundle import build_handover_bundle

        sess = orchestrator.create_session("Bundle FMEA 2")
        zip_bytes, _ = build_handover_bundle(
            sess.session_id,
            orchestrator,
            audit,
            ncr_service,
        )
        zf = zipfile.ZipFile(io.BytesIO(zip_bytes))
        doc = json.loads(zf.read("08_fmea/fmea_report.json"))
        assert len(doc["failure_modes"]) == len(FAILURE_MODES)
        assert doc["session_id"] == sess.session_id
