"""test_deviation_heatmap.py — P2.2b deviation heatmap renderer + endpoint.

Covers:
  • render_heatmap_svg — plan + elevation views, deviation + LOA colouring
  • compute_heatmap_summary — distribution buckets
  • Empty / no-centroid records → empty_svg fallback
  • GET /sessions/{id}/deviation-heatmap endpoint (auth, 404, SVG, JSON)
"""

from __future__ import annotations

import uuid
from datetime import datetime, timezone

import pytest
import pytest_asyncio
from httpx import ASGITransport, AsyncClient

# ── Helpers ──────────────────────────────────────────────────────────────────


def _record(
    distance_mm: float | None = 10.0,
    scan_centroid: list[float] | None = None,
    ifc_centroid: list[float] | None = None,
    seg_id: str | None = None,
    status: str = "MATCHED",
):
    from agent.models import DeviationStatus, ElementDeviationRecord

    return ElementDeviationRecord(
        ifc_global_id=f"IFC-{uuid.uuid4().hex[:6]}",
        ifc_class="IfcWall",
        ifc_name=f"W-{uuid.uuid4().hex[:4]}",
        scan_segment_id=seg_id or str(uuid.uuid4()),
        distance_mm=distance_mm,
        status=DeviationStatus(status),
        ifc_centroid_mm=ifc_centroid,
        scan_centroid_mm=scan_centroid,
    )


def _report(records: list | None = None, session_id: str | None = None):
    from agent.models import DeviationReport

    if records is None:
        records = [
            _record(distance_mm=2.0, scan_centroid=[0, 0, 1500]),
            _record(distance_mm=10.0, scan_centroid=[1000, 0, 1500]),
            _record(distance_mm=30.0, scan_centroid=[2000, 0, 1500]),
            _record(distance_mm=80.0, scan_centroid=[3000, 0, 1500]),
        ]
    return DeviationReport(
        session_id=session_id or str(uuid.uuid4()),
        generated_at=datetime.now(timezone.utc),
        ifc_element_count=len(records),
        scan_segment_count=len(records),
        matched_count=sum(1 for r in records if r.status.value == "MATCHED"),
        shifted_count=sum(1 for r in records if r.status.value == "SHIFTED"),
        missing_count=sum(1 for r in records if r.status.value == "MISSING"),
        extra_count=sum(1 for r in records if r.status.value == "EXTRA"),
        records=records,
    )


# ═════════════════════════════════════════════════════════════════════════════
# 1. render_heatmap_svg — unit tests
# ═════════════════════════════════════════════════════════════════════════════


class TestRenderHeatmapSvg:
    def test_plan_view_colour_by_deviation(self):
        from agent.tools.deviation_heatmap import render_heatmap_svg

        svg = render_heatmap_svg(_report(), view="plan", colour_by="deviation")
        assert svg.startswith("<svg")
        assert svg.endswith("</svg>")
        assert "Plan view" in svg
        assert "deviation" in svg

    def test_elevation_view_colour_by_loa(self):
        from agent.tools.deviation_heatmap import render_heatmap_svg

        loa_stmts = [
            {"segment_id": r.scan_segment_id, "effective_tier": "LOA30"} for r in _report().records
        ]
        svg = render_heatmap_svg(
            _report(),
            loa_statements=loa_stmts,
            view="elevation",
            colour_by="loa",
        )
        assert "Elevation" in svg
        assert "LOA" in svg

    def test_empty_report_returns_empty_svg(self):
        from agent.tools.deviation_heatmap import render_heatmap_svg

        svg = render_heatmap_svg(None)
        assert svg.startswith("<svg")
        assert "No deviation records" in svg

    def test_record_without_centroid_skipped(self):
        """A record with no centroid at all must not crash the renderer."""
        from agent.models import DeviationReport, DeviationStatus, ElementDeviationRecord
        from agent.tools.deviation_heatmap import render_heatmap_svg

        r = ElementDeviationRecord(
            ifc_global_id=None,
            ifc_class=None,
            ifc_name=None,
            scan_segment_id="x",
            distance_mm=5.0,
            status=DeviationStatus.MISSING,
            ifc_centroid_mm=None,
            scan_centroid_mm=None,
        )
        rep = DeviationReport(
            session_id=str(uuid.uuid4()),
            generated_at=datetime.now(timezone.utc),
            ifc_element_count=1,
            scan_segment_count=1,
            matched_count=0,
            shifted_count=0,
            missing_count=1,
            extra_count=0,
            records=[r],
        )
        svg = render_heatmap_svg(rep)
        assert svg.startswith("<svg")
        # Expect empty-fallback message because no plottable records
        assert "No spatial data" in svg or "No deviation records" in svg

    def test_falls_back_to_ifc_centroid_when_scan_missing(self):
        """If scan_centroid is None but ifc_centroid exists, use ifc."""
        from agent.tools.deviation_heatmap import render_heatmap_svg

        rep = _report(
            records=[_record(distance_mm=5.0, scan_centroid=None, ifc_centroid=[100, 200, 300])]
        )
        svg = render_heatmap_svg(rep)
        # Should NOT hit the empty fallback
        assert "No spatial data" not in svg
        # Should have at least one circle plotted
        assert "<circle" in svg

    def test_legend_contains_all_loa_tiers_when_loa_mode(self):
        from agent.tools.deviation_heatmap import render_heatmap_svg

        svg = render_heatmap_svg(
            _report(),
            loa_statements=[],
            view="plan",
            colour_by="loa",
        )
        for tier in ("LOA50", "LOA40", "LOA30", "LOA20", "LOA10"):
            assert tier in svg, f"legend missing {tier}"

    def test_tooltips_escape_special_chars(self):
        """XML-unsafe names in tooltips must be escaped."""
        from agent.models import (
            DeviationReport,
            DeviationStatus,
            ElementDeviationRecord,
        )
        from agent.tools.deviation_heatmap import render_heatmap_svg

        r = ElementDeviationRecord(
            ifc_global_id="X",
            ifc_class="IfcWall",
            ifc_name="Wall<A&B>",
            scan_segment_id="s",
            distance_mm=1.0,
            status=DeviationStatus.MATCHED,
            ifc_centroid_mm=None,
            scan_centroid_mm=[0, 0, 0],
        )
        rep = DeviationReport(
            session_id=str(uuid.uuid4()),
            generated_at=datetime.now(timezone.utc),
            ifc_element_count=1,
            scan_segment_count=1,
            matched_count=1,
            shifted_count=0,
            missing_count=0,
            extra_count=0,
            records=[r],
        )
        svg = render_heatmap_svg(rep)
        assert "&lt;" in svg
        assert "&amp;" in svg
        assert "&gt;" in svg
        # raw < or > must not appear in the escaped name
        assert "Wall<A&B>" not in svg


# ═════════════════════════════════════════════════════════════════════════════
# 2. compute_heatmap_summary
# ═════════════════════════════════════════════════════════════════════════════


class TestHeatmapSummary:
    def test_deviation_buckets(self):
        from agent.tools.deviation_heatmap import compute_heatmap_summary

        rep = _report()  # 2, 10, 30, 80
        summary = compute_heatmap_summary(rep)

        assert summary["total_records"] == 4
        assert summary["by_deviation_mm"]["<5"] == 1
        assert summary["by_deviation_mm"]["5-15"] == 1
        assert summary["by_deviation_mm"]["15-50"] == 1
        assert summary["by_deviation_mm"][">=50"] == 1
        assert summary["by_deviation_mm"]["unknown"] == 0

    def test_unknown_bucket_for_none_distance(self):
        from agent.tools.deviation_heatmap import compute_heatmap_summary

        rep = _report(records=[_record(distance_mm=None)])
        summary = compute_heatmap_summary(rep)
        assert summary["by_deviation_mm"]["unknown"] == 1

    def test_loa_distribution_when_statements_provided(self):
        from agent.tools.deviation_heatmap import compute_heatmap_summary

        stmts = [
            {"effective_tier": "LOA30"},
            {"effective_tier": "LOA30"},
            {"effective_tier": "LOA20"},
        ]
        summary = compute_heatmap_summary(_report(), stmts)
        assert summary["by_loa_tier"]["LOA30"] == 2
        assert summary["by_loa_tier"]["LOA20"] == 1

    def test_loa_distribution_none_when_no_statements(self):
        from agent.tools.deviation_heatmap import compute_heatmap_summary

        summary = compute_heatmap_summary(_report(), None)
        assert summary["by_loa_tier"] is None


# ═════════════════════════════════════════════════════════════════════════════
# 3. API endpoint — /sessions/{id}/deviation-heatmap
# ═════════════════════════════════════════════════════════════════════════════


@pytest_asyncio.fixture
async def auth_client():
    from agent.main import app

    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as ac:
        resp = await ac.post(
            "/auth/token",
            data={"username": "admin", "password": "admin123"},
        )
        token = resp.json().get("access_token", "")
        ac.headers["Authorization"] = f"Bearer {token}"
        yield ac


class TestHeatmapEndpoint:
    @pytest.mark.asyncio
    async def test_requires_auth(self):
        from agent.main import app

        async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as ac:
            resp = await ac.get("/sessions/any-id/deviation-heatmap")
            assert resp.status_code == 401

    @pytest.mark.asyncio
    async def test_unknown_session_404(self, auth_client: AsyncClient):
        resp = await auth_client.get("/sessions/does-not-exist/deviation-heatmap")
        assert resp.status_code == 404

    @pytest.mark.asyncio
    async def test_no_deviation_report_404(self, auth_client: AsyncClient):
        """Session exists but no deviation report generated → 404."""
        from agent.main import orchestrator

        sess = orchestrator.create_session("Heatmap Session A")
        resp = await auth_client.get(f"/sessions/{sess.session_id}/deviation-heatmap")
        assert resp.status_code == 404
        assert "deviation report" in resp.json()["detail"].lower()

    @pytest.mark.asyncio
    async def test_returns_svg(self, auth_client: AsyncClient):
        """Happy path — report exists, returns SVG content."""
        from agent.main import orchestrator

        sess = orchestrator.create_session("Heatmap Session B")
        orchestrator._deviation_reports[sess.session_id] = _report(
            session_id=sess.session_id,
        )

        resp = await auth_client.get(
            f"/sessions/{sess.session_id}/deviation-heatmap",
            params={"format": "svg", "view": "plan"},
        )
        assert resp.status_code == 200
        assert resp.headers["content-type"].startswith("image/svg+xml")
        assert resp.text.startswith("<svg")
        assert "</svg>" in resp.text

    @pytest.mark.asyncio
    async def test_returns_json_summary(self, auth_client: AsyncClient):
        from agent.main import orchestrator

        sess = orchestrator.create_session("Heatmap Session C")
        orchestrator._deviation_reports[sess.session_id] = _report(
            session_id=sess.session_id,
        )

        resp = await auth_client.get(
            f"/sessions/{sess.session_id}/deviation-heatmap",
            params={"format": "json"},
        )
        assert resp.status_code == 200
        body = resp.json()
        assert body["session_id"] == sess.session_id
        assert "by_deviation_mm" in body
        assert body["total_records"] == 4

    @pytest.mark.asyncio
    async def test_elevation_view_accepted(self, auth_client: AsyncClient):
        from agent.main import orchestrator

        sess = orchestrator.create_session("Heatmap Session D")
        orchestrator._deviation_reports[sess.session_id] = _report(
            session_id=sess.session_id,
        )

        resp = await auth_client.get(
            f"/sessions/{sess.session_id}/deviation-heatmap",
            params={"view": "elevation"},
        )
        assert resp.status_code == 200
        assert "Elevation" in resp.text

    @pytest.mark.asyncio
    async def test_invalid_view_returns_422(self, auth_client: AsyncClient):
        from agent.main import orchestrator

        sess = orchestrator.create_session("Heatmap Session E")
        orchestrator._deviation_reports[sess.session_id] = _report(
            session_id=sess.session_id,
        )

        resp = await auth_client.get(
            f"/sessions/{sess.session_id}/deviation-heatmap",
            params={"view": "oblique"},
        )
        assert resp.status_code == 422

    @pytest.mark.asyncio
    async def test_colour_by_loa(self, auth_client: AsyncClient):
        from agent.main import orchestrator

        sess = orchestrator.create_session("Heatmap Session F")
        orchestrator._deviation_reports[sess.session_id] = _report(
            session_id=sess.session_id,
        )
        resp = await auth_client.get(
            f"/sessions/{sess.session_id}/deviation-heatmap",
            params={"colour_by": "loa"},
        )
        assert resp.status_code == 200
        assert "LOA" in resp.text
