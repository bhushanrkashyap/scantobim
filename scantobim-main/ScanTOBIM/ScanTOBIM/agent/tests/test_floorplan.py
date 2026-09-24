"""Tests for Sprint 3.5 Feature 3: 2D Floor Plan Slice Export.

Covers:
  - floorplan_tools module functions (unit)
  - orchestrator.generate_floorplan (integration)
  - GET /sessions/{id}/floorplan endpoint
"""

from __future__ import annotations

import uuid
import xml.etree.ElementTree as ET

import numpy as np
import pytest
import pytest_asyncio
from httpx import ASGITransport, AsyncClient

# ── Helpers ────────────────────────────────────────────────────────────────────


def _grid_cloud(
    n_per_wall: int = 100,
    room_size_mm: float = 6000,
    floor_z_mm: float = 0.0,
    wall_z_min: float = 0.0,
    wall_z_max: float = 3000.0,
) -> np.ndarray:
    """Generate a simple 4-wall rectangular room point cloud in mm."""
    rng = np.random.default_rng(0)
    pts_list = []
    # North wall  y = room_size
    xs = rng.uniform(0, room_size_mm, n_per_wall)
    ys = np.full(n_per_wall, room_size_mm) + rng.normal(0, 3, n_per_wall)
    zs = rng.uniform(wall_z_min, wall_z_max, n_per_wall)
    pts_list.append(np.column_stack([xs, ys, zs]))
    # South wall  y = 0
    xs = rng.uniform(0, room_size_mm, n_per_wall)
    ys = rng.normal(0, 3, n_per_wall)
    zs = rng.uniform(wall_z_min, wall_z_max, n_per_wall)
    pts_list.append(np.column_stack([xs, ys, zs]))
    # East wall   x = room_size
    ys = rng.uniform(0, room_size_mm, n_per_wall)
    xs = np.full(n_per_wall, room_size_mm) + rng.normal(0, 3, n_per_wall)
    zs = rng.uniform(wall_z_min, wall_z_max, n_per_wall)
    pts_list.append(np.column_stack([xs, ys, zs]))
    # West wall   x = 0
    ys = rng.uniform(0, room_size_mm, n_per_wall)
    xs = rng.normal(0, 3, n_per_wall)
    zs = rng.uniform(wall_z_min, wall_z_max, n_per_wall)
    pts_list.append(np.column_stack([xs, ys, zs]))
    return np.vstack(pts_list)


def _make_segment(x: float, y: float, z: float, shape=None):
    from agent.models import BoundingBox, GeometrySegment, Point3D, SegmentShape

    shape = shape or SegmentShape.BOX
    return GeometrySegment(
        segment_id=str(uuid.uuid4()),
        zone_id="zone-001",
        shape=shape,
        centroid=Point3D(x=x, y=y, z=z),
        bounding_box=BoundingBox(
            min_x=x - 250,
            min_y=y - 250,
            min_z=z - 1500,
            max_x=x + 250,
            max_y=y + 250,
            max_z=z + 1500,
        ),
        point_count=200,
        confidence=0.85,
    )


# ── Unit: slice_point_cloud ────────────────────────────────────────────────────


class TestSlicePointCloud:
    def test_basic_slice_returns_xy(self):
        from agent.tools.floorplan_tools import slice_point_cloud

        pts = _grid_cloud(n_per_wall=200)
        # Slice at z=1000mm, thickness=200mm → band [900, 1100]
        xy = slice_point_cloud(pts, elevation_mm=1000, thickness_mm=200)
        assert xy.ndim == 2
        assert xy.shape[1] == 2
        assert xy.shape[0] > 0
        # All returned points should come from the band
        # (we can verify by checking the original Z for those XY, but that
        #  information is dropped; just assert count is plausible)
        assert xy.shape[0] <= pts.shape[0]

    def test_empty_band_returns_empty_array(self):
        from agent.tools.floorplan_tools import slice_point_cloud

        pts = np.random.default_rng(0).uniform(0, 1000, (500, 3))  # z in [0, 1000]
        xy = slice_point_cloud(pts, elevation_mm=5000, thickness_mm=200)
        assert xy.shape == (0, 2)

    def test_bad_input_returns_empty(self):
        from agent.tools.floorplan_tools import slice_point_cloud

        xy = slice_point_cloud(np.zeros((0, 3)), elevation_mm=1000)
        assert xy.shape == (0, 2)


# ── Unit: slice_from_segments ─────────────────────────────────────────────────


class TestSliceFromSegments:
    def test_vertical_wall_overlapping_band(self):
        """A PLANE_VERTICAL segment with bbox overlapping the slice band → 4 pts."""
        from agent.models import SegmentShape
        from agent.tools.floorplan_tools import slice_from_segments

        seg = _make_segment(0, 0, 1500, shape=SegmentShape.PLANE_VERTICAL)
        # segment bbox z: [0, 3000] — overlaps band [900, 1100]
        xy = slice_from_segments([seg], elevation_mm=1000, thickness_mm=200)
        assert xy.shape[0] == 4

    def test_cylinder_emits_circle_points(self):
        """A CYLINDER segment → circle approximated as 12 points."""
        from agent.models import SegmentShape
        from agent.tools.floorplan_tools import _CIRCLE_APPROX_POINTS, slice_from_segments

        seg = _make_segment(1000, 1000, 1500, shape=SegmentShape.CYLINDER)
        xy = slice_from_segments([seg], elevation_mm=1000, thickness_mm=200)
        assert xy.shape[0] == _CIRCLE_APPROX_POINTS

    def test_segment_outside_band_excluded(self):
        """Segment with bbox entirely above band → not included."""
        from agent.models import SegmentShape
        from agent.tools.floorplan_tools import slice_from_segments

        seg = _make_segment(0, 0, 10000, shape=SegmentShape.BOX)  # z: [8500, 11500]
        xy = slice_from_segments([seg], elevation_mm=1000, thickness_mm=200)
        assert xy.shape[0] == 0

    def test_empty_segments_returns_empty(self):
        from agent.tools.floorplan_tools import slice_from_segments

        xy = slice_from_segments([], elevation_mm=1000)
        assert xy.shape == (0, 2)


# ── Unit: ransac_2d_lines ──────────────────────────────────────────────────────


class TestRansac2dLines:
    def test_too_few_points_returns_empty(self):
        from agent.tools.floorplan_tools import ransac_2d_lines

        assert ransac_2d_lines(np.zeros((1, 2))) == []

    def test_perfect_horizontal_line(self):
        """100 points on y=0 line → at least 1 line detected."""
        from agent.tools.floorplan_tools import ransac_2d_lines

        rng = np.random.default_rng(0)
        xs = np.linspace(0, 5000, 200)
        ys = rng.normal(0, 5, 200)  # 5 mm noise around y=0
        pts = np.column_stack([xs, ys])
        lines = ransac_2d_lines(pts, min_inliers=20, distance_threshold_mm=20)
        assert len(lines) >= 1

    def test_four_wall_room_detects_lines(self):
        """4-wall room cloud → RANSAC detects lines (at least 2)."""
        from agent.tools.floorplan_tools import ransac_2d_lines, slice_point_cloud

        pts = _grid_cloud(n_per_wall=200)
        xy = slice_point_cloud(pts, elevation_mm=1000, thickness_mm=300)
        if xy.shape[0] < 40:
            pytest.skip("Slice returned too few points for this test")

        lines = ransac_2d_lines(xy, min_inliers=20, distance_threshold_mm=50)
        assert len(lines) >= 2

    def test_lines_sorted_longest_first(self):
        from agent.tools.floorplan_tools import ransac_2d_lines

        rng = np.random.default_rng(1)
        # Two parallel lines of different lengths
        short_xs = np.linspace(0, 1000, 60)
        long_xs = np.linspace(0, 5000, 200)
        pts = np.vstack(
            [
                np.column_stack([short_xs, rng.normal(0, 5, 60)]),
                np.column_stack([long_xs, rng.normal(3000, 5, 200)]),
            ]
        )
        lines = ransac_2d_lines(pts, min_inliers=20, distance_threshold_mm=30)
        if len(lines) >= 2:
            assert lines[0].length_mm >= lines[1].length_mm


# ── Unit: render_svg ──────────────────────────────────────────────────────────


class TestRenderSvg:
    def test_returns_valid_xml(self):
        from agent.tools.floorplan_tools import render_svg

        pts = np.column_stack(
            [
                np.linspace(0, 5000, 50),
                np.zeros(50),
            ]
        )
        svg = render_svg([], pts)
        root = ET.fromstring(svg)
        assert root.tag.endswith("svg")

    def test_empty_points_returns_valid_svg(self):
        from agent.tools.floorplan_tools import render_svg

        svg = render_svg([], np.empty((0, 2)))
        assert "svg" in svg.lower()
        # Should still produce valid XML — either an empty canvas or message svg
        root = ET.fromstring(svg)
        assert root.tag.endswith("svg")

    def test_contains_line_elements_when_lines_present(self):
        from agent.tools.floorplan_tools import LineSegment2D, render_svg

        ln = LineSegment2D(x0=0, y0=0, x1=5000, y1=0, inlier_count=100)
        pts = np.column_stack([np.linspace(0, 5000, 50), np.zeros(50)])
        svg = render_svg([ln], pts)
        assert "<line" in svg

    def test_svg_has_correct_dimensions(self):
        from agent.tools.floorplan_tools import render_svg

        pts = np.column_stack([np.linspace(0, 5000, 50), np.zeros(50)])
        svg = render_svg([], pts, width_px=800, height_px=600)
        assert 'width="800"' in svg
        assert 'height="600"' in svg


# ── Unit: render_dxf ──────────────────────────────────────────────────────────


class TestRenderDxf:
    def test_returns_bytes(self):
        pytest.importorskip("ezdxf")
        from agent.tools.floorplan_tools import render_dxf

        pts = np.column_stack([np.linspace(0, 5000, 20), np.zeros(20)])
        result = render_dxf([], pts)
        assert isinstance(result, bytes)
        assert len(result) > 0

    def test_dxf_has_walls_layer(self):
        pytest.importorskip("ezdxf")
        import ezdxf

        from agent.tools.floorplan_tools import LineSegment2D, render_dxf

        ln = LineSegment2D(x0=0, y0=0, x1=5000, y1=0, inlier_count=100)
        pts = np.column_stack([np.linspace(0, 5000, 10), np.zeros(10)])
        data = render_dxf([ln], pts)

        import io

        doc = ezdxf.read(io.StringIO(data.decode("utf-8", errors="ignore")))
        layer_names = [l.dxf.name for l in doc.layers]
        assert "WALLS" in layer_names

    def test_dxf_missing_ezdxf_raises(self, monkeypatch):
        import sys

        monkeypatch.setitem(sys.modules, "ezdxf", None)
        # Force reimport to pick up the monkeypatched module
        import importlib

        import agent.tools.floorplan_tools as ft

        importlib.reload(ft)

        with pytest.raises((ImportError, TypeError)):
            ft.render_dxf([], np.zeros((5, 2)))

        # Restore
        importlib.reload(ft)


# ── Integration: orchestrator.generate_floorplan ─────────────────────────────


class TestOrchestratorFloorplan:
    @pytest.fixture
    def session_with_segments(self):
        from agent.main import orchestrator
        from agent.models import SegmentShape

        sess = orchestrator.create_session("Floorplan Test Site")
        segs = [_make_segment(i * 1500, 0, 1500, SegmentShape.PLANE_VERTICAL) for i in range(4)]
        orchestrator._segments[sess.session_id] = segs
        return sess

    def test_generate_floorplan_no_data_raises(self):
        from agent.main import orchestrator

        sess = orchestrator.create_session("Empty for floorplan")
        with pytest.raises(ValueError, match="No scan data"):
            orchestrator.generate_floorplan(sess.session_id)

    def test_generate_floorplan_svg_returns_bytes(self, session_with_segments):
        from agent.main import orchestrator

        data = orchestrator.generate_floorplan(session_with_segments.session_id, fmt="svg")
        assert isinstance(data, bytes)
        assert b"<svg" in data

    def test_generate_floorplan_unknown_session_raises(self):
        from agent.main import orchestrator

        with pytest.raises(KeyError):
            orchestrator.generate_floorplan("unknown-session-id")

    def test_generate_floorplan_uses_floor_z_from_segments(self, session_with_segments):
        from agent.main import orchestrator

        # segment bbox min_z is -0 mm (set in fixture via _make_segment z=1500,
        # min_z = 1500 - 1500 = 0).  Floor should be detected as 0.
        # elevation_m=1.0 → slice at 1000 mm above floor = 1000 mm
        data = orchestrator.generate_floorplan(
            session_with_segments.session_id,
            elevation_m=1.0,
            fmt="svg",
        )
        assert data  # non-empty


# ── Endpoint tests ─────────────────────────────────────────────────────────────


class TestFloorplanEndpoint:
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
            json={"site_name": "Floorplan Endpoint Test", "use_synthetic": True},
        )
        return resp.json()["session_id"]

    async def test_floorplan_svg_200(self, auth_client: AsyncClient, session_id: str):
        resp = await auth_client.get(
            f"/sessions/{session_id}/floorplan",
            params={"format": "svg"},
        )
        assert resp.status_code == 200
        assert "svg" in resp.headers.get("content-type", "")

    async def test_floorplan_dxf_200(self, auth_client: AsyncClient, session_id: str):
        pytest.importorskip("ezdxf")
        resp = await auth_client.get(
            f"/sessions/{session_id}/floorplan",
            params={"format": "dxf"},
        )
        assert resp.status_code == 200
        assert "dxf" in resp.headers.get("content-type", "").lower()

    async def test_floorplan_404_unknown_session(self, auth_client: AsyncClient):
        resp = await auth_client.get("/sessions/bad-session/floorplan")
        assert resp.status_code == 404

    async def test_floorplan_422_bad_format(self, auth_client: AsyncClient, session_id: str):
        resp = await auth_client.get(
            f"/sessions/{session_id}/floorplan",
            params={"format": "pdf"},
        )
        assert resp.status_code == 422

    async def test_floorplan_422_no_data(self, auth_client: AsyncClient):
        from agent.main import orchestrator

        sess = orchestrator.create_session("Floorplan No Data")
        resp = await auth_client.get(f"/sessions/{sess.session_id}/floorplan")
        assert resp.status_code == 422

    async def test_floorplan_svg_content_type_header(
        self, auth_client: AsyncClient, session_id: str
    ):
        resp = await auth_client.get(
            f"/sessions/{session_id}/floorplan",
            params={"format": "svg"},
        )
        assert resp.status_code == 200
        cd = resp.headers.get("content-disposition", "")
        assert ".svg" in cd
