"""
conftest.py — shared pytest fixtures for ScanToBIM Agent tests.

Key concerns:
  1. open3d is NOT installed in CI — we mock it before any import that uses it.
  2. SQLite DB is created fresh per test session in a temp directory.
  3. FastAPI test client is provided as an async fixture.
"""

from __future__ import annotations

import sys
import uuid
from collections.abc import AsyncGenerator
from unittest.mock import MagicMock

import pytest
import pytest_asyncio
from httpx import ASGITransport, AsyncClient

# ─────────────────────────────────────────────────────────────────────────────
# 1. Mock open3d BEFORE importing any agent module that transitively uses it.
#    This allows the full test suite to run in CI without installing open3d.
# ─────────────────────────────────────────────────────────────────────────────


def _mock_open3d() -> None:
    """Inject a minimal open3d mock into sys.modules."""
    import numpy as np

    o3d = MagicMock()

    # Geometry namespace used by scan_tools.py
    pcd = MagicMock()
    pcd.points = np.zeros((500, 3))  # real ndarray so asarray() works

    o3d.geometry.PointCloud.return_value = pcd
    o3d.io.read_point_cloud.return_value = pcd
    o3d.utility.Vector3dVector.side_effect = lambda arr=None: np.asarray(arr) if arr is not None else np.zeros((500, 3))

    # RANSAC / segmentation return values
    plane_model = [0.0, 0.0, 1.0, 0.0]  # horizontal plane normal
    inliers = list(range(500))
    pcd.segment_plane.return_value = (plane_model, inliers)
    pcd.select_by_index.return_value = pcd
    pcd.voxel_down_sample.return_value = pcd
    pcd.cluster_dbscan.return_value = [0] * 200

    # A2: SOR — remove_statistical_outlier returns (pcd, indices)
    pcd.remove_statistical_outlier.return_value = (pcd, list(range(500)))

    # A3: normal estimation — has_normals, estimate_normals, orient_normals
    pcd.has_normals.return_value = False
    pcd.estimate_normals.return_value = None
    pcd.orient_normals_consistent_tangent_plane.return_value = None
    pcd.normals = np.zeros((500, 3))

    # B4: OBB
    obb = MagicMock()
    obb.extent = np.array([5.0, 0.2, 0.15])
    obb.R = np.eye(3)
    obb.center = np.zeros(3)
    pcd.get_oriented_bounding_box.return_value = obb

    # C2: KDTreeFlann for region growing
    kdtree = MagicMock()
    kdtree.search_radius_vector_3d.return_value = (0, [], [])
    o3d.geometry.KDTreeFlann.return_value = kdtree

    # KDTreeSearchParamHybrid for normal estimation
    o3d.geometry.KDTreeSearchParamHybrid.return_value = MagicMock()

    # get_axis_aligned_bounding_box
    aabb = MagicMock()
    aabb.get_min_bound.return_value = MagicMock(__mul__=lambda s, x: [0.0, 0.0, 0.0])
    aabb.get_max_bound.return_value = MagicMock(__mul__=lambda s, x: [5.0, 0.2, 3.0])
    pcd.get_axis_aligned_bounding_box.return_value = aabb

    # ICP registration result — must return a real 4×4 numpy array for transform
    icp_result = MagicMock()
    icp_result.transformation = np.eye(4, dtype=np.float64)
    icp_result.inlier_rmse = 0.003  # 3 mm in metres
    icp_result.fitness = 0.90
    o3d.pipelines.registration.registration_icp.return_value = icp_result
    o3d.pipelines.registration.TransformationEstimationPointToPlane.return_value = MagicMock()
    o3d.pipelines.registration.ICPConvergenceCriteria.return_value = MagicMock()

    # RANSAC registration & FPFH feature computation mocks
    ransac_result = MagicMock()
    ransac_result.transformation = np.eye(4, dtype=np.float64)
    ransac_result.inlier_rmse = 0.005
    ransac_result.fitness = 0.85
    ransac_result.correspondence_set = list(range(100))
    o3d.pipelines.registration.registration_ransac_based_on_feature_matching.return_value = ransac_result
    o3d.pipelines.registration.CorrespondenceCheckerBasedOnDistance.return_value = MagicMock()
    o3d.pipelines.registration.CorrespondenceCheckerBasedOnNormal.return_value = MagicMock()
    o3d.pipelines.registration.CorrespondenceCheckerBasedOnEdgeLength.return_value = MagicMock()
    o3d.pipelines.registration.RANSACConvergenceCriteria.return_value = MagicMock()

    fpfh_feat = MagicMock()
    fpfh_feat.dimension.return_value = 33
    fpfh_feat.num.return_value = 500
    o3d.pipelines.registration.compute_fpfh_feature.return_value = fpfh_feat

    # pcd.transform() — used in register_all_scans; returns the same pcd mock
    pcd.transform.return_value = pcd
    # pcd.__add__ — used for merged = merged + source_world
    pcd.__add__ = lambda self, other: pcd

    sys.modules["open3d"] = o3d
    sys.modules["open3d.geometry"] = o3d.geometry
    sys.modules["open3d.io"] = o3d.io
    sys.modules["open3d.utility"] = o3d.utility
    sys.modules["open3d.pipelines"] = o3d.pipelines
    sys.modules["open3d.pipelines.registration"] = o3d.pipelines.registration


_mock_open3d()


# ─────────────────────────────────────────────────────────────────────────────
# 2. Environment setup (must happen before agent imports use os.environ)
# ─────────────────────────────────────────────────────────────────────────────

import os
import tempfile

_tmp_dir = tempfile.mkdtemp(prefix="scantobim_test_")

os.environ.setdefault("AUDIT_HMAC_SECRET", "test-hmac-secret-not-for-production")
os.environ.setdefault("SCAN_AGENT_API_KEY", "test-api-key")
os.environ.setdefault("DATABASE_URL", f"sqlite:///{_tmp_dir}/test.db")
os.environ.setdefault("REVIT_BRIDGE_URL", "http://localhost:8766")
os.environ.setdefault("SCAN_FILES_DIR", _tmp_dir)
os.environ.setdefault("AGENT_PORT", "8765")


# ─────────────────────────────────────────────────────────────────────────────
# 3. Fixtures
# ─────────────────────────────────────────────────────────────────────────────


@pytest.fixture(scope="session")
def test_db_path() -> str:
    return f"{_tmp_dir}/test.db"


@pytest_asyncio.fixture
async def client() -> AsyncGenerator[AsyncClient, None]:
    """Async HTTPX test client for the FastAPI app."""
    from agent.main import app

    async with AsyncClient(
        transport=ASGITransport(app=app),
        base_url="http://test",
    ) as ac:
        yield ac


# ── Sample data fixtures ───────────────────────────────────────────────────────


@pytest.fixture
def ns_segment_dict() -> dict:
    """A minimal non-safety geometry segment payload (passes all gates)."""
    return {
        "site_name": "Test Facility NS",
        "zone_id": "zone-001",
        "use_synthetic": True,
    }


@pytest.fixture
def session_id() -> str:
    return str(uuid.uuid4())


@pytest.fixture(autouse=True)
def reset_o3d_pcd():
    import numpy as np
    if "open3d" in sys.modules:
        o3d = sys.modules["open3d"]
        if hasattr(o3d, "geometry") and hasattr(o3d.geometry, "PointCloud"):
            pcd = o3d.geometry.PointCloud.return_value
            pcd.points = np.zeros((500, 3))
    yield
    if "open3d" in sys.modules:
        o3d = sys.modules["open3d"]
        if hasattr(o3d, "geometry") and hasattr(o3d.geometry, "PointCloud"):
            pcd = o3d.geometry.PointCloud.return_value
            pcd.points = np.zeros((500, 3))

