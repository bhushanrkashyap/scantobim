"""Tests for Sprint 3.5 Feature 1: Multi-scan ICP Registration.

Covers:
  - registration_tools module functions (unit)
  - orchestrator.register_scans (integration)
  - POST /sessions/{id}/register-scans endpoint
"""

from __future__ import annotations

from pathlib import Path
from unittest.mock import patch

import numpy as np
import pytest
import pytest_asyncio
from httpx import ASGITransport, AsyncClient

# ── Helpers ────────────────────────────────────────────────────────────────────


def _make_ply_bytes(pts: np.ndarray) -> bytes:
    """Build a minimal ASCII PLY file from an (N, 3) numpy array in mm."""
    header = (
        "ply\n"
        "format ascii 1.0\n"
        f"element vertex {len(pts)}\n"
        "property float x\n"
        "property float y\n"
        "property float z\n"
        "end_header\n"
    )
    rows = "\n".join(f"{r[0]} {r[1]} {r[2]}" for r in pts)
    return (header + rows + "\n").encode("ascii")


def _write_ply(tmp_path: Path, pts: np.ndarray, name: str = "scan.ply") -> Path:
    p = tmp_path / name
    p.write_bytes(_make_ply_bytes(pts))
    return p


def _synthetic_wall_cloud(n: int = 300, offset_mm: float = 0.0) -> np.ndarray:
    """Flat vertical wall in XZ, with Y offset to simulate a separate station."""
    rng = np.random.default_rng(42)
    x = rng.uniform(0, 5000, n)
    y = np.full(n, offset_mm) + rng.normal(0, 2, n)  # 2 mm noise
    z = rng.uniform(0, 3000, n)
    return np.column_stack([x, y, z])


# ── Unit tests: registration_tools ────────────────────────────────────────────


class TestRegistrationTools:
    """Unit tests for functions in registration_tools.py.

    Open3D is mocked globally in conftest.py, so these tests verify the
    orchestration logic rather than the numerical ICP result.
    """

    def test_load_scan_as_o3d_missing_file(self, tmp_path: Path):
        from agent.tools.registration_tools import load_scan_as_o3d

        with pytest.raises(FileNotFoundError):
            load_scan_as_o3d(tmp_path / "nonexistent.ply")

    def test_load_scan_as_o3d_ply(self, tmp_path: Path):
        """load_scan_as_o3d returns an Open3D PointCloud (mocked)."""
        from agent.tools.registration_tools import load_scan_as_o3d

        pts = _synthetic_wall_cloud(100)
        p = _write_ply(tmp_path, pts)
        pcd = load_scan_as_o3d(p)
        # Mocked o3d returns the mock pcd object — just assert no exception
        assert pcd is not None

    def test_global_rmse_mm_single_station(self):
        from agent.tools.registration_tools import _PairResult, global_rmse_mm

        results = [_PairResult(0, "a.ply", np.eye(4), 0.0, 1.0)]
        assert global_rmse_mm(results) == 0.0

    def test_global_rmse_mm_multi_station(self):
        from agent.tools.registration_tools import _PairResult, global_rmse_mm

        results = [
            _PairResult(0, "a.ply", np.eye(4), 0.0, 1.0),
            _PairResult(1, "b.ply", np.eye(4), 3.0, 0.9),
            _PairResult(2, "c.ply", np.eye(4), 5.0, 0.8),
        ]
        expected = float(np.sqrt(np.mean([9.0, 25.0])))  # RMS of 3 and 5
        assert abs(global_rmse_mm(results) - expected) < 1e-9

    def test_merged_pcd_to_numpy_mm(self):
        """merged_pcd_to_numpy_mm converts metres to millimetres."""
        import open3d as o3d

        from agent.tools.registration_tools import merged_pcd_to_numpy_mm

        pcd = o3d.geometry.PointCloud()
        # The mock returns a MagicMock; we patch asarray to return known data
        with patch("numpy.asarray") as mock_arr:
            mock_arr.return_value = np.array([[1.0, 2.0, 3.0]])
            result = merged_pcd_to_numpy_mm(pcd)
            np.testing.assert_array_almost_equal(result, [[1000.0, 2000.0, 3000.0]])

    def test_register_all_scans_requires_two_files(self, tmp_path: Path):
        from agent.tools.registration_tools import register_all_scans

        p = _write_ply(tmp_path, _synthetic_wall_cloud(50))
        with pytest.raises(ValueError, match="At least 2"):
            register_all_scans([p])

    def test_register_all_scans_returns_result_structure(self, tmp_path: Path):
        """register_all_scans returns (merged_pcd, list of _PairResult)."""
        from agent.tools.registration_tools import register_all_scans

        p1 = _write_ply(tmp_path, _synthetic_wall_cloud(100, offset_mm=0), "s1.ply")
        p2 = _write_ply(tmp_path, _synthetic_wall_cloud(100, offset_mm=100), "s2.ply")

        merged, results = register_all_scans([p1, p2])
        assert len(results) == 2
        assert results[0].station_index == 0
        assert results[1].station_index == 1
        assert results[0].rmse == 0.0  # reference station
        assert results[0].inlier_ratio == 1.0


# ── Integration: orchestrator.register_scans ──────────────────────────────────


class TestOrchestratorRegisterScans:
    @pytest.fixture
    def session(self):
        """Create a live orchestrator session."""
        from agent.main import orchestrator

        sess = orchestrator.create_session("ICP Test Site")
        return sess

    def test_register_scans_stores_merged_cloud(self, tmp_path: Path, session):
        from agent.main import orchestrator

        p1 = _write_ply(tmp_path, _synthetic_wall_cloud(80, 0), "s1.ply")
        p2 = _write_ply(tmp_path, _synthetic_wall_cloud(80, 200), "s2.ply")

        result = orchestrator.register_scans(session.session_id, [p1, p2])

        assert session.session_id in orchestrator._registration
        assert session.session_id in orchestrator._merged_clouds
        cloud = orchestrator._merged_clouds[session.session_id]
        assert isinstance(cloud, np.ndarray)

    def test_register_scans_returns_station_registration_result(self, tmp_path: Path, session):
        from agent.main import orchestrator
        from agent.models import StationRegistrationResult

        p1 = _write_ply(tmp_path, _synthetic_wall_cloud(80, 0), "s1.ply")
        p2 = _write_ply(tmp_path, _synthetic_wall_cloud(80, 200), "s2.ply")

        result = orchestrator.register_scans(session.session_id, [p1, p2])

        assert isinstance(result, StationRegistrationResult)
        assert result.station_count == 2
        assert len(result.stations) == 2
        assert result.stations[0].transform_4x4 is not None

    def test_register_scans_unknown_session_raises(self, tmp_path: Path):
        from agent.main import orchestrator

        p1 = _write_ply(tmp_path, _synthetic_wall_cloud(50), "s1.ply")
        p2 = _write_ply(tmp_path, _synthetic_wall_cloud(50), "s2.ply")

        with pytest.raises(KeyError):
            orchestrator.register_scans("nonexistent-session", [p1, p2])

    def test_register_scans_single_file_raises(self, tmp_path: Path, session):
        from agent.main import orchestrator

        p1 = _write_ply(tmp_path, _synthetic_wall_cloud(50), "s1.ply")
        with pytest.raises(ValueError, match="At least 2"):
            orchestrator.register_scans(session.session_id, [p1])


# ── Endpoint tests ─────────────────────────────────────────────────────────────


class TestRegisterScansEndpoint:
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
        # use_synthetic=True so the session exists without needing a scan file
        resp = await auth_client.post(
            "/sessions",
            json={"site_name": "Register Test", "use_synthetic": True},
        )
        return resp.json()["session_id"]

    async def test_register_scans_404_unknown_session(
        self, auth_client: AsyncClient, tmp_path: Path
    ):
        pts = _synthetic_wall_cloud(50)
        ply = _make_ply_bytes(pts)
        resp = await auth_client.post(
            "/sessions/unknown-id/register-scans",
            files=[
                ("files", ("a.ply", ply, "application/octet-stream")),
                ("files", ("b.ply", ply, "application/octet-stream")),
            ],
        )
        assert resp.status_code == 404

    async def test_register_scans_422_single_file(
        self, auth_client: AsyncClient, session_id: str, tmp_path: Path
    ):
        ply = _make_ply_bytes(_synthetic_wall_cloud(50))
        resp = await auth_client.post(
            f"/sessions/{session_id}/register-scans",
            files=[("files", ("a.ply", ply, "application/octet-stream"))],
        )
        assert resp.status_code == 422

    async def test_register_scans_200_two_files(self, auth_client: AsyncClient, session_id: str):
        ply1 = _make_ply_bytes(_synthetic_wall_cloud(80, 0))
        ply2 = _make_ply_bytes(_synthetic_wall_cloud(80, 200))
        resp = await auth_client.post(
            f"/sessions/{session_id}/register-scans",
            files=[
                ("files", ("s1.ply", ply1, "application/octet-stream")),
                ("files", ("s2.ply", ply2, "application/octet-stream")),
            ],
            params={"run_pipeline": "false"},
        )
        assert resp.status_code == 200
        body = resp.json()
        assert body["station_count"] == 2
        assert "global_rmse_mm" in body
        assert "stations" in body
