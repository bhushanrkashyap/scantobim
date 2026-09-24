"""test_upload.py — Tests for Sprint 1: scan file upload + job tracking.

Tests cover:
  - POST /sessions/{id}/upload  with valid/invalid files
  - GET  /jobs/{job_id}         job status
  - GET  /jobs                  job list
  - ScanFileInfo validation helpers
  - detect_scan_format magic byte checks
  - count_file_points header read
"""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

# ── Helpers to build minimal valid file bytes ─────────────────────────────────


def _make_las_bytes() -> bytes:
    """Minimal LAS 1.2 file with correct magic bytes (LASF)."""
    # LAS file starts with 4-byte magic "LASF" followed by 240+ byte header
    magic = b"LASF"
    # Fill rest of a minimal 300-byte header with zeros
    header = magic + b"\x00" * 296
    return header


def _make_ply_bytes() -> bytes:
    """Minimal PLY ASCII file."""
    return (
        b"ply\nformat ascii 1.0\nelement vertex 3\n"
        b"property float x\nproperty float y\nproperty float z\n"
        b"end_header\n0 0 0\n1 1 1\n2 2 2\n"
    )


def _make_xyz_bytes() -> bytes:
    """Minimal XYZ point cloud text file."""
    return b"0.0 0.0 0.0\n1.0 1.0 1.0\n2.0 2.0 2.0\n"


# ── Unit tests: validate_scan_file / detect_scan_format ──────────────────────


class TestScanFileValidation:
    def test_detect_format_las(self, tmp_path):
        from agent.tools.scan_tools import detect_scan_format

        f = tmp_path / "test.las"
        f.write_bytes(_make_las_bytes())
        assert detect_scan_format(f) == ".las"

    def test_detect_format_laz(self, tmp_path):
        from agent.tools.scan_tools import detect_scan_format

        f = tmp_path / "test.laz"
        f.write_bytes(_make_las_bytes())  # .laz also uses LASF magic
        assert detect_scan_format(f) == ".laz"

    def test_detect_format_ply(self, tmp_path):
        from agent.tools.scan_tools import detect_scan_format

        f = tmp_path / "test.ply"
        f.write_bytes(_make_ply_bytes())
        assert detect_scan_format(f) == ".ply"

    def test_detect_format_xyz_no_magic(self, tmp_path):
        from agent.tools.scan_tools import detect_scan_format

        f = tmp_path / "test.xyz"
        f.write_bytes(_make_xyz_bytes())
        assert detect_scan_format(f) == ".xyz"  # no magic check for xyz

    def test_detect_format_unsupported_raises(self, tmp_path):
        from agent.tools.scan_tools import detect_scan_format

        f = tmp_path / "test.obj"
        f.write_bytes(b"# OBJ file")
        with pytest.raises(ValueError, match="Unsupported file format"):
            detect_scan_format(f)

    def test_detect_format_bad_magic_raises(self, tmp_path):
        from agent.tools.scan_tools import detect_scan_format

        # Write a .ply file with wrong magic
        f = tmp_path / "test.ply"
        f.write_bytes(b"WRONG_MAGIC" + b"\x00" * 50)
        with pytest.raises(ValueError, match="magic bytes do not match"):
            detect_scan_format(f)

    def test_validate_scan_file_valid_las(self, tmp_path):
        from agent.tools.scan_tools import validate_scan_file

        f = tmp_path / "scan.las"
        # Write 100KB of valid LAS
        f.write_bytes(_make_las_bytes() + b"\x00" * (100 * 1024))
        info = validate_scan_file(f)
        assert info.valid is True
        assert info.format == ".las"
        assert info.errors == []

    def test_validate_scan_file_too_small(self, tmp_path):
        from agent.tools.scan_tools import validate_scan_file

        f = tmp_path / "tiny.las"
        f.write_bytes(b"LASF")  # only 4 bytes — below MIN_SCAN_SIZE_BYTES
        info = validate_scan_file(f)
        assert info.valid is False
        assert any("too small" in e for e in info.errors)

    def test_validate_scan_file_too_large(self, tmp_path):
        from agent.tools.scan_tools import validate_scan_file

        f = tmp_path / "huge.las"
        f.write_bytes(_make_las_bytes() + b"\x00" * (1024 * 1024))
        info = validate_scan_file(f, max_size_mb=0.0001)  # artificially low limit
        assert info.valid is False
        assert any("exceeds" in e for e in info.errors)

    def test_validate_scan_file_not_found(self, tmp_path):
        from agent.tools.scan_tools import validate_scan_file

        info = validate_scan_file(tmp_path / "missing.las")
        assert info.valid is False
        assert any("not found" in e for e in info.errors)

    def test_validate_scan_file_size_warning(self, tmp_path):
        from agent.tools.scan_tools import validate_scan_file

        f = tmp_path / "big.las"
        # Write 600 MB worth — too slow to actually write, so patch stat
        f.write_bytes(_make_las_bytes() + b"\x00" * 1024)
        with patch.object(Path, "stat") as mock_stat:
            mock_stat.return_value = MagicMock(st_size=600 * 1024 * 1024)
            info = validate_scan_file(f, max_size_mb=2000)
        assert info.valid is True
        assert any("Large scan" in w for w in info.warnings)

    def test_count_file_points_las(self, tmp_path):
        from agent.tools.scan_tools import count_file_points

        f = tmp_path / "scan.las"
        # Build a proper context manager mock and inject laspy into sys.modules
        import sys

        mock_header = MagicMock()
        mock_header.point_count = 50_000
        mock_reader = MagicMock()
        mock_reader.header = mock_header
        mock_cm = MagicMock()
        mock_cm.__enter__ = MagicMock(return_value=mock_reader)
        mock_cm.__exit__ = MagicMock(return_value=False)
        mock_laspy = MagicMock()
        mock_laspy.open = MagicMock(return_value=mock_cm)
        with patch.dict(sys.modules, {"laspy": mock_laspy}):
            count = count_file_points(f)
        assert count == 50_000

    def test_count_file_points_ply_reads_header(self, tmp_path):
        from agent.tools.scan_tools import count_file_points

        f = tmp_path / "scan.ply"
        f.write_bytes(_make_ply_bytes())
        assert count_file_points(f) == 3

    def test_count_file_points_non_las_unknown_format_returns_minus_one(self, tmp_path):
        from agent.tools.scan_tools import count_file_points

        f = tmp_path / "scan.e57"
        f.write_bytes(b"ASTM-E57" + b"\x00" * 100)
        assert count_file_points(f) == -1


# ── API tests: upload endpoint ────────────────────────────────────────────────


@pytest.fixture
def auth_headers(client):
    """Get auth token for admin user."""
    import asyncio

    async def _get_token():
        resp = await client.post(
            "/auth/token",
            data={"username": "admin", "password": "admin123"},
            headers={"Content-Type": "application/x-www-form-urlencoded"},
        )
        assert resp.status_code == 200
        return {"Authorization": f"Bearer {resp.json()['access_token']}"}

    return asyncio.get_event_loop().run_until_complete(_get_token())


@pytest.fixture
def session_id(client):
    """Create a session and return its ID."""
    import asyncio

    async def _create():
        resp = await client.post(
            "/sessions",
            json={"site_name": "Upload Test Site", "use_synthetic": True},
        )
        assert resp.status_code == 200
        return resp.json()["session_id"]

    return asyncio.get_event_loop().run_until_complete(_create())


class TestUploadEndpoint:
    @pytest.mark.asyncio
    async def test_upload_unsupported_format_returns_422(self, client, auth_headers, session_id):
        resp = await client.post(
            f"/sessions/{session_id}/upload",
            files={"file": ("model.obj", b"# OBJ file", "application/octet-stream")},
            headers=auth_headers,
        )
        assert resp.status_code == 422
        assert "Unsupported" in resp.json()["detail"]

    @pytest.mark.asyncio
    async def test_upload_valid_las_returns_job(self, client, auth_headers, session_id):
        las_bytes = _make_las_bytes() + b"\x00" * (50 * 1024)  # 50 KB

        # Patch validate_scan_file where it is imported inside the endpoint
        with (
            patch("agent.tools.scan_tools.validate_scan_file") as mock_validate,
            patch("agent.main._run_pipeline_background"),
        ):
            from agent.tools.scan_tools import ScanFileInfo

            mock_validate.return_value = ScanFileInfo(
                path=Path("/tmp/test.las"),
                format=".las",
                size_bytes=len(las_bytes),
                size_mb=len(las_bytes) / 1024 / 1024,
                valid=True,
            )

            resp = await client.post(
                f"/sessions/{session_id}/upload",
                files={"file": ("scan.las", las_bytes, "application/octet-stream")},
                headers=auth_headers,
            )

        assert resp.status_code == 200
        data = resp.json()
        assert "job_id" in data
        assert data["session_id"] == session_id
        assert data["status"] == "queued"
        assert "progress_url" in data
        assert "job_url" in data

    @pytest.mark.asyncio
    async def test_upload_session_not_found_returns_404(self, client, auth_headers):
        resp = await client.post(
            "/sessions/nonexistent-session/upload",
            files={"file": ("scan.las", b"LASF" + b"\x00" * 300, "application/octet-stream")},
            headers=auth_headers,
        )
        assert resp.status_code == 404

    @pytest.mark.asyncio
    async def test_get_job_returns_status(self, client, auth_headers, session_id):
        las_bytes = _make_las_bytes() + b"\x00" * (50 * 1024)

        with (
            patch("agent.tools.scan_tools.validate_scan_file") as mock_validate,
            patch("agent.main._run_pipeline_background"),
        ):
            from agent.tools.scan_tools import ScanFileInfo

            mock_validate.return_value = ScanFileInfo(
                path=Path("/tmp/test.las"),
                format=".las",
                size_bytes=len(las_bytes),
                size_mb=0.05,
                valid=True,
            )
            upload_resp = await client.post(
                f"/sessions/{session_id}/upload",
                files={"file": ("scan.las", las_bytes, "application/octet-stream")},
                headers=auth_headers,
            )

        job_id = upload_resp.json()["job_id"]

        job_resp = await client.get(f"/jobs/{job_id}", headers=auth_headers)
        assert job_resp.status_code == 200
        data = job_resp.json()
        assert data["job_id"] == job_id
        assert data["session_id"] == session_id
        assert "status" in data

    @pytest.mark.asyncio
    async def test_get_job_not_found_returns_404(self, client, auth_headers):
        resp = await client.get("/jobs/nonexistent-job-id", headers=auth_headers)
        assert resp.status_code == 404

    @pytest.mark.asyncio
    async def test_list_jobs_returns_array(self, client, auth_headers):
        resp = await client.get("/jobs", headers=auth_headers)
        assert resp.status_code == 200
        data = resp.json()
        assert "jobs" in data
        assert "count" in data
        assert isinstance(data["jobs"], list)

    @pytest.mark.asyncio
    async def test_list_jobs_filter_by_session(self, client, auth_headers, session_id):
        resp = await client.get(f"/jobs?session_id={session_id}", headers=auth_headers)
        assert resp.status_code == 200
        jobs = resp.json()["jobs"]
        for job in jobs:
            assert job["session_id"] == session_id

    @pytest.mark.asyncio
    async def test_upload_requires_auth(self, client, session_id):
        """Upload without token should return 401."""
        resp = await client.post(
            f"/sessions/{session_id}/upload",
            files={"file": ("scan.las", b"LASF" + b"\x00" * 300, "application/octet-stream")},
        )
        assert resp.status_code == 401
