"""test_registration_report.py — P2.2c Registration QA report.

Covers:
  • classify_station — all four RICS categories (A/B/C/FAIL) + boundaries
  • worst_category aggregation
  • build_qa_report — session aggregation, overall_pass_fail logic
  • render_registration_html — HTML structure, escaping, colour banding
  • RegistrationQAReportModel — Pydantic validation
  • GET /sessions/{id}/registration-report endpoint
"""

from __future__ import annotations

import uuid
from datetime import datetime, timezone

import pytest
import pytest_asyncio
from httpx import ASGITransport, AsyncClient

from agent.models import (
    RegistrationQAReportModel,
    StationRegistrationDetail,
    StationRegistrationResult,
)
from agent.tools.registration_report import (
    RICSCategory,
    build_qa_report,
    classify_station,
    render_registration_html,
    worst_category,
)

# ── Helpers ──────────────────────────────────────────────────────────────────


def _station(
    idx: int, rmse_mm: float, inlier: float, source: str | None = None
) -> StationRegistrationDetail:
    return StationRegistrationDetail(
        station_index=idx,
        source_file=source or f"scan_{idx:02d}.e57",
        transform_4x4=[[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]],
        rmse_mm=rmse_mm,
        inlier_ratio=inlier,
    )


def _result(
    stations: list, global_rmse_mm: float, session_id: str | None = None
) -> StationRegistrationResult:
    return StationRegistrationResult(
        session_id=session_id or str(uuid.uuid4()),
        station_count=len(stations),
        stations=stations,
        global_rmse_mm=global_rmse_mm,
        merged_point_count=1_000_000,
        registered_at=datetime.now(timezone.utc),
    )


# ═════════════════════════════════════════════════════════════════════════════
# 1. Category classification
# ═════════════════════════════════════════════════════════════════════════════


class TestClassifyStation:
    def test_survey_grade_is_cat_a(self):
        assert classify_station(1.5, 0.95) == RICSCategory.A

    def test_bim_grade_is_cat_b(self):
        assert classify_station(8.0, 0.85) == RICSCategory.B

    def test_general_planning_is_cat_c(self):
        assert classify_station(30.0, 0.70) == RICSCategory.C

    def test_out_of_tolerance_is_fail(self):
        assert classify_station(100.0, 0.55) == RICSCategory.FAIL

    def test_cat_a_rmse_boundary(self):
        assert classify_station(3.0, 0.90) == RICSCategory.A
        assert classify_station(3.01, 0.90) == RICSCategory.B

    def test_cat_a_inlier_boundary(self):
        """A station at exactly 0.90 inlier passes Cat A; below it doesn't."""
        assert classify_station(2.0, 0.90) == RICSCategory.A
        assert classify_station(2.0, 0.89) == RICSCategory.B

    def test_cat_c_fails_when_inlier_too_low(self):
        """Within RMSE bound but inlier below Cat C floor."""
        assert classify_station(40.0, 0.55) == RICSCategory.FAIL


class TestWorstCategory:
    def test_same_returns_same(self):
        assert worst_category(RICSCategory.B, RICSCategory.B) == RICSCategory.B

    def test_a_vs_c_gives_c(self):
        assert worst_category(RICSCategory.A, RICSCategory.C) == RICSCategory.C

    def test_b_vs_fail_gives_fail(self):
        assert worst_category(RICSCategory.B, RICSCategory.FAIL) == RICSCategory.FAIL

    def test_commutative(self):
        assert worst_category(RICSCategory.A, RICSCategory.C) == worst_category(
            RICSCategory.C, RICSCategory.A
        )


# ═════════════════════════════════════════════════════════════════════════════
# 2. Session-level aggregation
# ═════════════════════════════════════════════════════════════════════════════


class TestBuildQAReport:
    def test_all_cat_a(self):
        r = _result([_station(i, 1.5, 0.95) for i in range(3)], global_rmse_mm=1.8)
        report = build_qa_report(r)
        assert report.overall_category == RICSCategory.A
        assert report.overall_pass_fail == "pass"
        assert report.station_count == 3
        assert all(s.category == RICSCategory.A for s in report.stations)

    def test_mixed_downgrades_overall(self):
        """One Cat-C station drops overall to Cat C even if others are A."""
        stations = [
            _station(0, 1.0, 0.95),  # A
            _station(1, 2.5, 0.92),  # A
            _station(2, 25.0, 0.65),  # C
        ]
        report = build_qa_report(_result(stations, global_rmse_mm=9.0))
        assert report.overall_category == RICSCategory.C
        assert report.overall_pass_fail == "pass"  # still passing

    def test_one_fail_drops_overall_to_fail(self):
        stations = [
            _station(0, 1.0, 0.95),  # A
            _station(1, 200.0, 0.10),  # FAIL
        ]
        report = build_qa_report(_result(stations, global_rmse_mm=80.0))
        assert report.overall_category == RICSCategory.FAIL
        assert report.overall_pass_fail == "fail"

    def test_global_rmse_forces_downgrade(self):
        """All stations Cat A, but global RMSE > 3 mm → overall Cat B."""
        stations = [_station(i, 2.0, 0.95) for i in range(3)]
        report = build_qa_report(_result(stations, global_rmse_mm=8.0))
        assert report.overall_category == RICSCategory.B

    def test_empty_stations(self):
        r = _result([], global_rmse_mm=0.0)
        report = build_qa_report(r)
        assert report.station_count == 0
        assert report.stations == []

    def test_to_dict_has_required_keys(self):
        r = _result([_station(0, 1.5, 0.95)], global_rmse_mm=1.5)
        d = build_qa_report(r).to_dict()
        for k in (
            "session_id",
            "generated_at_utc",
            "station_count",
            "stations",
            "global_rmse_mm",
            "overall_category",
            "overall_pass_fail",
            "merged_point_count",
            "standard_version",
        ):
            assert k in d


# ═════════════════════════════════════════════════════════════════════════════
# 3. HTML rendering
# ═════════════════════════════════════════════════════════════════════════════


class TestRenderRegistrationHtml:
    def test_produces_valid_html(self):
        r = _result([_station(0, 1.5, 0.95), _station(1, 2.0, 0.92)], global_rmse_mm=1.8)
        html = render_registration_html(build_qa_report(r))
        assert html.startswith("<!DOCTYPE html>")
        assert "</html>" in html

    def test_includes_station_data(self):
        r = _result(
            [_station(0, 4.2, 0.88, source="north_wing.e57")],
            global_rmse_mm=4.2,
        )
        html = render_registration_html(build_qa_report(r))
        assert "north_wing.e57" in html
        assert "4.200" in html or "4.2" in html

    def test_cat_c_shown_with_amber(self):
        r = _result([_station(0, 25.0, 0.70)], global_rmse_mm=25.0)
        html = render_registration_html(build_qa_report(r))
        assert "Cat C" in html
        assert "#f5b400" in html  # amber

    def test_fail_shown_with_red(self):
        r = _result([_station(0, 100.0, 0.50)], global_rmse_mm=100.0)
        html = render_registration_html(build_qa_report(r))
        assert "FAIL" in html
        assert "#e14d3d" in html

    def test_escapes_xss_in_filename(self):
        r = _result(
            [_station(0, 2.0, 0.95, source="<script>alert(1)</script>.e57")],
            global_rmse_mm=2.0,
        )
        html = render_registration_html(build_qa_report(r))
        assert "<script>" not in html
        assert "&lt;script&gt;" in html

    def test_empty_stations_renders_placeholder(self):
        r = _result([], global_rmse_mm=0.0)
        html = render_registration_html(build_qa_report(r))
        assert "No stations" in html


# ═════════════════════════════════════════════════════════════════════════════
# 4. Pydantic model
# ═════════════════════════════════════════════════════════════════════════════


class TestRegistrationQAReportModel:
    def test_valid_model_round_trip(self):
        r = _result([_station(0, 1.5, 0.95)], global_rmse_mm=1.5)
        report = build_qa_report(r)
        m = RegistrationQAReportModel(**report.to_dict())
        assert m.station_count == 1
        assert m.overall_category == "A"

    def test_rejects_bad_category(self):
        with pytest.raises(Exception):
            RegistrationQAReportModel(
                session_id=str(uuid.uuid4()),
                generated_at_utc="2026-01-01T00:00:00Z",
                station_count=0,
                stations=[],
                global_rmse_mm=1.0,
                overall_category="Z",
                overall_pass_fail="pass",
                merged_point_count=0,
            )

    def test_rejects_negative_rmse(self):
        with pytest.raises(Exception):
            RegistrationQAReportModel(
                session_id=str(uuid.uuid4()),
                generated_at_utc="2026-01-01T00:00:00Z",
                station_count=0,
                stations=[],
                global_rmse_mm=-1.0,
                overall_category="A",
                overall_pass_fail="pass",
                merged_point_count=0,
            )


# ═════════════════════════════════════════════════════════════════════════════
# 5. API endpoint — /sessions/{id}/registration-report
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


class TestRegistrationReportEndpoint:
    @pytest.mark.asyncio
    async def test_requires_auth(self):
        from agent.main import app

        async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as ac:
            resp = await ac.get("/sessions/any/registration-report")
            assert resp.status_code == 401

    @pytest.mark.asyncio
    async def test_unknown_session_404(self, auth_client: AsyncClient):
        resp = await auth_client.get("/sessions/does-not-exist/registration-report")
        assert resp.status_code == 404

    @pytest.mark.asyncio
    async def test_no_registration_result_404(self, auth_client: AsyncClient):
        from agent.main import orchestrator

        sess = orchestrator.create_session("Reg QA Session A")
        resp = await auth_client.get(f"/sessions/{sess.session_id}/registration-report")
        assert resp.status_code == 404
        assert "registration" in resp.json()["detail"].lower()

    @pytest.mark.asyncio
    async def test_returns_json_report(self, auth_client: AsyncClient):
        from agent.main import orchestrator

        sess = orchestrator.create_session("Reg QA Session B")
        orchestrator._registration[sess.session_id] = _result(
            [_station(0, 1.5, 0.95), _station(1, 2.0, 0.92)],
            global_rmse_mm=1.8,
            session_id=sess.session_id,
        )
        resp = await auth_client.get(
            f"/sessions/{sess.session_id}/registration-report",
            params={"format": "json"},
        )
        assert resp.status_code == 200
        body = resp.json()
        assert body["station_count"] == 2
        assert body["overall_category"] == "A"
        assert body["overall_pass_fail"] == "pass"

    @pytest.mark.asyncio
    async def test_returns_html(self, auth_client: AsyncClient):
        from agent.main import orchestrator

        sess = orchestrator.create_session("Reg QA Session C")
        orchestrator._registration[sess.session_id] = _result(
            [_station(0, 25.0, 0.70)],
            global_rmse_mm=25.0,
            session_id=sess.session_id,
        )
        resp = await auth_client.get(
            f"/sessions/{sess.session_id}/registration-report",
            params={"format": "html"},
        )
        assert resp.status_code == 200
        assert resp.headers["content-type"].startswith("text/html")
        assert resp.text.startswith("<!DOCTYPE html>")
        assert "Cat C" in resp.text

    @pytest.mark.asyncio
    async def test_invalid_format_422(self, auth_client: AsyncClient):
        from agent.main import orchestrator

        sess = orchestrator.create_session("Reg QA Session D")
        orchestrator._registration[sess.session_id] = _result(
            [_station(0, 1.5, 0.95)],
            global_rmse_mm=1.5,
            session_id=sess.session_id,
        )
        resp = await auth_client.get(
            f"/sessions/{sess.session_id}/registration-report",
            params={"format": "pdf"},
        )
        assert resp.status_code == 422
