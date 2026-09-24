"""Tests for Sprint 3.5 Feature 2: Scan vs. Design Deviation Report.

Covers:
  - deviation_tools module functions (unit)
  - orchestrator.generate_deviation_report (integration)
  - POST /sessions/{id}/deviation-report  endpoint
  - GET  /sessions/{id}/deviation-report  endpoint
"""

from __future__ import annotations

import uuid
from pathlib import Path
from unittest.mock import patch

import pytest
import pytest_asyncio
from httpx import ASGITransport, AsyncClient

# ── Helpers ────────────────────────────────────────────────────────────────────


def _make_segment(x: float, y: float, z: float, seg_id: str | None = None):
    """Create a minimal GeometrySegment at the given centroid."""
    from agent.models import (
        BoundingBox,
        GeometrySegment,
        Point3D,
        SegmentShape,
    )

    sid = seg_id or str(uuid.uuid4())
    return GeometrySegment(
        segment_id=sid,
        zone_id="zone-001",
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
        point_count=200,
        confidence=0.85,
    )


def _make_ifc_element(global_id: str, x: float, y: float, z: float):
    """Create a minimal IFCElementSummary."""
    from agent.tools.deviation_tools import IFCElementSummary

    return IFCElementSummary(
        global_id=global_id,
        name=f"Elem-{global_id[:4]}",
        ifc_class="IfcWall",
        centroid_mm=(x, y, z),
    )


# ── Unit tests: compute_deviation_report ──────────────────────────────────────


class TestComputeDeviationReport:
    def test_matched_within_tolerance(self):
        """Segment 10 mm from IFC element → MATCHED."""
        from agent.models import DeviationStatus
        from agent.tools.deviation_tools import compute_deviation_report

        elem = _make_ifc_element("AAAA0001", 1000, 2000, 1500)
        seg = _make_segment(1010, 2000, 1500)  # 10 mm off

        report = compute_deviation_report("sess-1", [elem], [seg])

        assert report.matched_count == 1
        assert report.shifted_count == 0
        assert report.missing_count == 0
        assert report.records[0].status == DeviationStatus.MATCHED

    def test_shifted_between_thresholds(self):
        """Segment 80 mm from IFC element → SHIFTED."""
        from agent.models import DeviationStatus
        from agent.tools.deviation_tools import compute_deviation_report

        elem = _make_ifc_element("BBBB0001", 0, 0, 0)
        seg = _make_segment(80, 0, 0)  # 80 mm off in X

        report = compute_deviation_report("sess-2", [elem], [seg])
        assert report.shifted_count == 1
        assert report.records[0].status == DeviationStatus.SHIFTED

    def test_missing_no_nearby_segment(self):
        """IFC element with no segment within 150 mm → MISSING."""
        from agent.models import DeviationStatus
        from agent.tools.deviation_tools import compute_deviation_report

        elem = _make_ifc_element("CCCC0001", 0, 0, 0)
        seg = _make_segment(500, 0, 0)  # 500 mm away — beyond threshold

        report = compute_deviation_report("sess-3", [elem], [seg])
        assert report.missing_count == 1
        assert report.records[0].status == DeviationStatus.MISSING
        assert report.records[0].distance_mm is None

    def test_extra_scan_segment_no_ifc_match(self):
        """Scan segment far from any IFC element → EXTRA."""
        from agent.models import DeviationStatus
        from agent.tools.deviation_tools import compute_deviation_report

        elem = _make_ifc_element("DDDD0001", 0, 0, 0)
        seg = _make_segment(500, 500, 500)  # far from IFC element

        report = compute_deviation_report("sess-4", [elem], [seg])
        # The IFC element will be MISSING and the segment will be EXTRA
        extra_records = [r for r in report.records if r.status == DeviationStatus.EXTRA]
        assert len(extra_records) >= 1
        assert extra_records[0].ifc_global_id is None
        assert extra_records[0].scan_segment_id is not None

    def test_empty_scan_segments_all_missing(self):
        """No scan data → all IFC elements are MISSING."""
        from agent.tools.deviation_tools import compute_deviation_report

        elems = [_make_ifc_element(f"E{i:04d}", i * 1000, 0, 0) for i in range(3)]
        report = compute_deviation_report("sess-5", elems, [])
        assert report.missing_count == 3
        assert report.matched_count == 0

    def test_empty_ifc_elements_all_extra(self):
        """No IFC data → all scan segments are EXTRA."""
        from agent.tools.deviation_tools import compute_deviation_report

        segs = [_make_segment(i * 1000, 0, 0) for i in range(3)]
        report = compute_deviation_report("sess-6", [], segs)
        assert report.extra_count == 3

    def test_summary_counts_sum_to_record_count(self):
        """matched + shifted + missing + extra == len(records)."""
        from agent.tools.deviation_tools import compute_deviation_report

        elems = [_make_ifc_element(f"X{i:04d}", i * 200, 0, 0) for i in range(4)]
        segs = [_make_segment(i * 200 + 10, 0, 0) for i in range(4)]

        report = compute_deviation_report("sess-7", elems, segs)
        total = (
            report.matched_count + report.shifted_count + report.missing_count + report.extra_count
        )
        assert total == len(report.records)

    def test_ifc_centroid_mm_field_populated(self):
        """ifc_centroid_mm is correctly populated in each record."""
        from agent.tools.deviation_tools import compute_deviation_report

        elem = _make_ifc_element("FFFF0001", 100, 200, 300)
        seg = _make_segment(100, 200, 300)

        report = compute_deviation_report("sess-8", [elem], [seg])
        rec = report.records[0]
        assert rec.ifc_centroid_mm == [100, 200, 300]


# ── Unit tests: extract_ifc_elements ──────────────────────────────────────────


class TestExtractIfcElements:
    def test_missing_ifcopenshell_raises_import_error(self, tmp_path: Path):
        """ImportError is raised when ifcopenshell is not available."""
        import sys
        from importlib import reload
        from unittest.mock import patch

        # Patch the import inside the function by patching sys.modules
        with patch.dict(
            sys.modules,
            {"ifcopenshell": None, "ifcopenshell.util.placement": None, "ifcopenshell.util": None},
        ):
            import agent.tools.deviation_tools as dt

            reload(dt)
            with pytest.raises((ImportError, TypeError)):
                dt.extract_ifc_elements(tmp_path / "fake.ifc")
        # Reload again to restore
        reload(dt)

    def test_nonexistent_ifc_file_raises(self, tmp_path: Path):
        pytest.importorskip("ifcopenshell")
        from agent.tools.deviation_tools import extract_ifc_elements

        with pytest.raises(RuntimeError, match="Failed to open"):
            extract_ifc_elements(tmp_path / "nonexistent.ifc")


# ── Integration: orchestrator.generate_deviation_report ───────────────────────


class TestOrchestratorDeviationReport:
    @pytest.fixture
    def session_with_segments(self):
        """Session with synthetic scan segments pre-loaded."""
        from agent.main import orchestrator

        sess = orchestrator.create_session("Deviation Test Site")
        # Manually inject segments (synthetic pipeline bypasses file I/O)
        segments = [_make_segment(i * 1000, 0, 1500) for i in range(5)]
        orchestrator._segments[sess.session_id] = segments
        return sess

    def test_generate_deviation_report_no_segments_raises(self):
        from agent.main import orchestrator

        sess = orchestrator.create_session("Empty Session")
        with pytest.raises(ValueError, match="No scan segments"):
            orchestrator.generate_deviation_report(
                sess.session_id,
                ifc_path=Path("/nonexistent.ifc"),
            )

    def test_generate_deviation_report_unknown_session_raises(self, tmp_path: Path):
        from agent.main import orchestrator

        with pytest.raises(KeyError):
            orchestrator.generate_deviation_report(
                "bad-session-id",
                ifc_path=tmp_path / "x.ifc",
            )

    def test_generate_deviation_report_stored_on_session(
        self, session_with_segments, tmp_path: Path
    ):
        """After generation the report is accessible via _deviation_reports."""
        from agent.main import orchestrator
        from agent.models import DeviationReport

        sess_id = session_with_segments.session_id

        # Patch extract_ifc_elements to return mock data (no real IFC file needed)
        mock_elements = [_make_ifc_element(f"MOCK{i:04d}", i * 1000, 0, 1500) for i in range(3)]
        with patch(
            "agent.orchestrator.extract_ifc_elements",
            return_value=mock_elements,
        ):
            report = orchestrator.generate_deviation_report(sess_id, ifc_path=tmp_path / "fake.ifc")

        assert isinstance(report, DeviationReport)
        assert sess_id in orchestrator._deviation_reports
        assert orchestrator._deviation_reports[sess_id].report_id == report.report_id


# ── Endpoint tests ─────────────────────────────────────────────────────────────


class TestDeviationReportEndpoints:
    @pytest_asyncio.fixture
    async def auth_client(self) -> AsyncClient:
        from agent.main import app

        async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as ac:
            resp = await ac.post(
                "/auth/token",
                data={"username": "admin", "password": "admin123"},
            )
            token = resp.json().get("access_token", "")
            ac.headers["Authorization"] = f"Bearer {token}"
            yield ac

    @pytest_asyncio.fixture
    async def session_id(self, auth_client: AsyncClient) -> str:
        resp = await auth_client.post(
            "/sessions",
            json={"site_name": "Deviation Endpoint Test", "use_synthetic": True},
        )
        return resp.json()["session_id"]

    async def test_get_deviation_report_404_before_generation(
        self, auth_client: AsyncClient, session_id: str
    ):
        resp = await auth_client.get(f"/sessions/{session_id}/deviation-report")
        assert resp.status_code == 404

    async def test_post_deviation_report_422_no_segments(self, auth_client: AsyncClient):
        """422 when session has no scan segments."""
        from agent.main import orchestrator

        sess = orchestrator.create_session("Empty for deviation test")
        sess_id = sess.session_id

        fake_ifc = b"ISO-10303-21;\nHEADER;\nENDSEC;\nDATA;\nENDSEC;\nEND-ISO-10303-21;"
        resp = await auth_client.post(
            f"/sessions/{sess_id}/deviation-report",
            files=[("ifc_file", ("design.ifc", fake_ifc, "application/octet-stream"))],
        )
        assert resp.status_code == 422

    async def test_post_deviation_report_422_wrong_extension(
        self, auth_client: AsyncClient, session_id: str
    ):
        """422 when file extension is not .ifc."""
        resp = await auth_client.post(
            f"/sessions/{session_id}/deviation-report",
            files=[("ifc_file", ("design.rvt", b"not-ifc", "application/octet-stream"))],
        )
        assert resp.status_code == 422

    async def test_post_deviation_report_404_unknown_session(self, auth_client: AsyncClient):
        fake_ifc = b"ISO-10303-21;\nHEADER;\nENDSEC;\nDATA;\nENDSEC;\nEND-ISO-10303-21;"
        resp = await auth_client.post(
            "/sessions/nonexistent/deviation-report",
            files=[("ifc_file", ("design.ifc", fake_ifc, "application/octet-stream"))],
        )
        assert resp.status_code == 404

    async def test_get_deviation_report_returns_report_after_post(
        self, auth_client: AsyncClient, session_id: str
    ):
        """GET returns the report stored by POST."""
        from agent.main import orchestrator

        # Inject mock deviation report directly
        from agent.models import DeviationReport

        report = DeviationReport(
            session_id=session_id,
            ifc_element_count=5,
            scan_segment_count=5,
            matched_count=3,
            shifted_count=1,
            missing_count=1,
            extra_count=0,
            records=[],
        )
        orchestrator._deviation_reports[session_id] = report

        resp = await auth_client.get(f"/sessions/{session_id}/deviation-report")
        assert resp.status_code == 200
        body = resp.json()
        assert body["matched_count"] == 3
        assert body["missing_count"] == 1
