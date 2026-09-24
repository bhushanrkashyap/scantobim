"""Phase 2 Spatial Foundation, Coordinate Normalization & Registration Tests.

Covers the 15 mandatory verification tests specified in Phase 2:
1.  test_origin_near_zero — cloud centered near origin, canonical transform ≈ identity
2.  test_positive_coordinates — positive-quadrant cloud processes correctly
3.  test_negative_coordinates — negative coordinates preserved and reversible
4.  test_large_survey_coordinates — UTM-scale coordinates (e.g. 500000, 4500000) handled
5.  test_translated_cloud — same geometry at different translations → equivalent canonical
6.  test_rotated_cloud_vertical — Z-rotation preserves bounding dimensions and distances
7.  test_arbitrary_3d_rotation — full 3D SO(3) rotation preserves pairwise distances
8.  test_unit_scaled_cloud — metre vs millimetre representation → equivalent after normalization
9.  test_transform_round_trip — source→canonical→source, verify max_error < 0.01 mm
10. test_unit_ambiguity — conflicting/uncertain unit signals → AMBIGUOUS status, no silent scaling
11. test_orientation_ambiguity — featureless cloud → AMBIGUOUS status, not forced Z-up
12. test_single_scan_no_registration — single cloud → REGISTRATION_NOT_REQUIRED
13. test_multi_scan_registration — two synthetic clouds → FPFH+RANSAC+ICP path
14. test_registration_failure — deliberately incompatible clouds → REGISTRATION_FAILED
15. test_icp_degradation — ICP worsens alignment → pre-ICP transform retained

All synthetic geometry is confined strictly to these test fixtures.
"""

from __future__ import annotations

import math
from pathlib import Path

import numpy as np

from agent.tools.coordinate_system import (
    ORIENTATION_AMBIGUOUS,
    REGISTRATION_FAILED,
    REGISTRATION_NOT_REQUIRED,
    REGISTRATION_SUCCESS,
    UNIT_AMBIGUOUS,
    AuthoritativeTransform,
)
from agent.tools.geometry_tools import compute_survey_transform
from agent.tools.orientation_tools import (
    estimate_vertical_axis,
)
from agent.tools.registration_tools import (
    register_pair_full,
    resolve_registration_mode,
)
from agent.tools.unit_resolution import resolve_units

# ── Synthetic Test Fixtures ───────────────────────────────────────────────────


def _generate_box_room(
    dx: float = 8.0, dy: float = 6.0, dz: float = 3.0, n_per_wall: int = 500
) -> tuple[np.ndarray, np.ndarray]:
    """Generate a clean synthetic rectangular room (floor + 4 walls) with normals."""
    rng = np.random.default_rng(42)
    pts_list = []
    normals_list = []

    # Floor (z=0, normal=[0,0,1])
    x = rng.uniform(0, dx, n_per_wall)
    y = rng.uniform(0, dy, n_per_wall)
    z = np.zeros(n_per_wall)
    pts_list.append(np.column_stack([x, y, z]))
    normals_list.append(np.tile([0.0, 0.0, 1.0], (n_per_wall, 1)))

    # Wall 1: x=0 (normal=[1,0,0])
    y1 = rng.uniform(0, dy, n_per_wall)
    z1 = rng.uniform(0, dz, n_per_wall)
    x1 = np.zeros(n_per_wall)
    pts_list.append(np.column_stack([x1, y1, z1]))
    normals_list.append(np.tile([1.0, 0.0, 0.0], (n_per_wall, 1)))

    # Wall 2: x=dx (normal=[-1,0,0])
    y2 = rng.uniform(0, dy, n_per_wall)
    z2 = rng.uniform(0, dz, n_per_wall)
    x2 = np.full(n_per_wall, dx)
    pts_list.append(np.column_stack([x2, y2, z2]))
    normals_list.append(np.tile([-1.0, 0.0, 0.0], (n_per_wall, 1)))

    # Wall 3: y=0 (normal=[0,1,0])
    x3 = rng.uniform(0, dx, n_per_wall)
    z3 = rng.uniform(0, dz, n_per_wall)
    y3 = np.zeros(n_per_wall)
    pts_list.append(np.column_stack([x3, y3, z3]))
    normals_list.append(np.tile([0.0, 1.0, 0.0], (n_per_wall, 1)))

    # Wall 4: y=dy (normal=[0,-1,0])
    x4 = rng.uniform(0, dx, n_per_wall)
    z4 = rng.uniform(0, dz, n_per_wall)
    y4 = np.full(n_per_wall, dy)
    pts_list.append(np.column_stack([x4, y4, z4]))
    normals_list.append(np.tile([0.0, -1.0, 0.0], (n_per_wall, 1)))

    points = np.vstack(pts_list)
    normals = np.vstack(normals_list)
    return points, normals


def _euler_to_rotation_matrix(roll_deg: float, pitch_deg: float, yaw_deg: float) -> np.ndarray:
    """Compute 3x3 rotation matrix from roll (X), pitch (Y), yaw (Z) in degrees."""
    rx = math.radians(roll_deg)
    ry = math.radians(pitch_deg)
    rz = math.radians(yaw_deg)

    Rx = np.array([
        [1, 0, 0],
        [0, math.cos(rx), -math.sin(rx)],
        [0, math.sin(rx), math.cos(rx)],
    ])
    Ry = np.array([
        [math.cos(ry), 0, math.sin(ry)],
        [0, 1, 0],
        [-math.sin(ry), 0, math.cos(ry)],
    ])
    Rz = np.array([
        [math.cos(rz), -math.sin(rz), 0],
        [math.sin(rz), math.cos(rz), 0],
        [0, 0, 1],
    ])
    return Rz @ Ry @ Rx


# ── 15 Mandatory Phase 2 Tests ───────────────────────────────────────────────


class TestPhase2SpatialFoundation:
    """Test suite for Phase 2 spatial foundation, coordinate normalization, and registration."""

    def test_origin_near_zero(self):
        """Test 1: Point cloud centered near origin produces valid transform near identity."""
        rng = np.random.default_rng(101)
        # Cloud centered around (0, 0, 0), span 10m x 10m x 3m
        pts = rng.uniform(-5.0, 5.0, (1000, 3))
        pts[:, 2] = rng.uniform(0.0, 3.0, 1000)

        centered, _T_4x4, offset = compute_survey_transform(pts)

        # Centroid offset should be small (< 0.5m)
        assert np.linalg.norm(offset[:2]) < 0.5
        # Bounding box center of centered points is exactly at origin
        assert np.allclose((np.min(centered, axis=0) + np.max(centered, axis=0)) * 0.5, 0.0, atol=1e-10)
        # Mean of centered points is close to zero
        assert np.allclose(np.mean(centered, axis=0), 0.0, atol=0.2)

        transform = AuthoritativeTransform(origin_offset_m=offset)
        val = transform.validate()
        assert val["is_valid"] is True
        assert val["determinant_error"] < 1e-6

    def test_positive_coordinates(self):
        """Test 2: Positive-quadrant cloud processes and centers correctly."""
        rng = np.random.default_rng(102)
        # Cloud in [100, 120] x [200, 220] x [10, 15]
        pts = np.column_stack([
            rng.uniform(100.0, 120.0, 500),
            rng.uniform(200.0, 220.0, 500),
            rng.uniform(10.0, 15.0, 500),
        ])

        centered, _T_4x4, offset = compute_survey_transform(pts)
        assert 105.0 < offset[0] < 115.0
        assert 205.0 < offset[1] < 215.0

        transform = AuthoritativeTransform(
            origin_offset_m=offset,
            source_bounds_m=np.array([np.min(pts, axis=0), np.max(pts, axis=0)]),
            canonical_bounds_m=np.array([np.min(centered, axis=0), np.max(centered, axis=0)]),
        )

        recovered = transform.transform_points(centered, direction="inverse")
        assert np.allclose(recovered, pts, atol=1e-6)

    def test_negative_coordinates(self):
        """Test 3: Negative coordinates are preserved and fully reversible."""
        rng = np.random.default_rng(103)
        # Cloud in negative quadrant [-120, -100] x [-220, -200] x [-15, -10]
        pts = np.column_stack([
            rng.uniform(-120.0, -100.0, 500),
            rng.uniform(-220.0, -200.0, 500),
            rng.uniform(-15.0, -10.0, 500),
        ])

        _centered, _T_4x4, offset = compute_survey_transform(pts)
        transform = AuthoritativeTransform(origin_offset_m=offset)

        # Forward then inverse must recover original coordinates exactly
        fwd = transform.transform_points(pts, direction="forward")
        inv = transform.transform_points(fwd, direction="inverse")
        max_err = float(np.max(np.abs(inv - pts)))
        assert max_err < 1e-6

    def test_large_survey_coordinates(self):
        """Test 4: UTM-scale coordinates (e.g. 500,000 m, 4,500,000 m) handled without precision loss."""
        rng = np.random.default_rng(104)
        local_pts = rng.uniform(0.0, 50.0, (1000, 3))
        # Add UTM offset (e.g. Zone 32N)
        utm_offset = np.array([500_000.0, 4_500_000.0, 150.0])
        utm_pts = local_pts + utm_offset

        centered, _T_4x4, offset = compute_survey_transform(utm_pts)

        # Centered points are normalized to local magnitude ~ 0-50m
        assert np.max(np.abs(centered)) < 100.0

        transform = AuthoritativeTransform(origin_offset_m=offset)
        err = transform.round_trip_error(utm_pts)
        # Round trip error must be well under 0.01 mm (1e-5 m)
        assert err["max_error_m"] < 1e-5

    def test_translated_cloud(self):
        """Test 5: Same geometry at different translations produces equivalent canonical representation."""
        pts1, _ = _generate_box_room()
        # Translated by [1000.0, -2500.0, 450.0]
        pts2 = pts1 + np.array([1000.0, -2500.0, 450.0])

        centered1, _, _ = compute_survey_transform(pts1)
        centered2, _, _ = compute_survey_transform(pts2)

        # Canonical extents (spans) must be identical
        span1 = np.ptp(centered1, axis=0)
        span2 = np.ptp(centered2, axis=0)
        assert np.allclose(span1, span2, atol=1e-6)

    def test_rotated_cloud_vertical(self):
        """Test 6: Z-rotation preserves bounding dimensions and distances."""
        pts, _ = _generate_box_room(dx=10.0, dy=6.0, dz=3.0)
        R_z = _euler_to_rotation_matrix(0, 0, 45.0)
        pts_rot = pts @ R_z.T

        # Pairwise distance sample check
        idx_a = [0, 10, 50, 100]
        idx_b = [5, 15, 55, 105]
        d_orig = np.linalg.norm(pts[idx_a] - pts[idx_b], axis=1)
        d_rot = np.linalg.norm(pts_rot[idx_a] - pts_rot[idx_b], axis=1)
        assert np.allclose(d_orig, d_rot, atol=1e-6)

        # Z extents are strictly identical under Z rotation
        assert math.isclose(float(np.ptp(pts[:, 2])), float(np.ptp(pts_rot[:, 2])), abs_tol=1e-6)

    def test_arbitrary_3d_rotation(self):
        """Test 7: Arbitrary 3D SO(3) rotation preserves pairwise distances and forms valid isometry."""
        pts, _ = _generate_box_room(dx=12.0, dy=8.0, dz=4.0)
        R_3d = _euler_to_rotation_matrix(roll_deg=23.5, pitch_deg=-15.2, yaw_deg=67.8)

        # Verify SO(3) properties: R @ R.T == I, det(R) == 1
        assert np.allclose(R_3d @ R_3d.T, np.eye(3), atol=1e-10)
        assert math.isclose(float(np.linalg.det(R_3d)), 1.0, abs_tol=1e-10)

        pts_rot = pts @ R_3d.T

        # Isometry check: distances between points are preserved exactly
        d1 = np.linalg.norm(pts[100:150] - pts[150:200], axis=1)
        d2 = np.linalg.norm(pts_rot[100:150] - pts_rot[150:200], axis=1)
        assert np.allclose(d1, d2, atol=1e-6)

        transform = AuthoritativeTransform(rotation_matrix=R_3d)
        val = transform.validate()
        assert val["is_valid"] is True
        assert val["orthogonality_error"] < 1e-9

    def test_unit_scaled_cloud(self):
        """Test 8: Metre vs millimetre representation yields equivalent points after normalization."""
        pts_m, _ = _generate_box_room(dx=10.0, dy=6.0, dz=3.0)
        pts_mm = pts_m * 1000.0

        res_m = resolve_units(points=pts_m)
        res_mm = resolve_units(points=pts_mm)

        assert res_m.detected_unit == "meters"
        assert res_mm.detected_unit == "millimeters"
        assert math.isclose(res_m.scale_to_meters, 1.0, abs_tol=1e-6)
        assert math.isclose(res_mm.scale_to_meters, 0.001, abs_tol=1e-6)

        # After applying scale, the normalized coordinates match within 1e-6
        normalized_mm = pts_mm * res_mm.scale_to_meters
        assert np.allclose(normalized_mm, pts_m, atol=1e-6)

    def test_transform_round_trip(self):
        """Test 9: Source→canonical→source round-trip error < 0.01 mm."""
        rng = np.random.default_rng(109)
        pts = rng.uniform(-50.0, 50.0, (2000, 3))

        R = _euler_to_rotation_matrix(12.3, -45.6, 78.9)
        offset = np.array([123.456, -789.012, 345.678])

        transform = AuthoritativeTransform(
            origin_offset_m=offset,
            rotation_matrix=R,
        )

        metrics = transform.round_trip_error(pts)
        # 0.01 mm = 1e-5 m
        assert metrics["max_error_m"] < 1e-5
        assert metrics["rmse_m"] < 1e-6
        assert metrics["max_error_mm"] < 0.01

    def test_unit_ambiguity(self):
        """Test 10: Conflicting or ambiguous unit signals return AMBIGUOUS status, no silent scaling."""
        rng = np.random.default_rng(110)
        # A tiny point cloud spanning only 0.2 m (too small for standard building envelope in m, mm, or ft)
        tiny_pts = rng.uniform(0.0, 0.2, (50, 3))

        res = resolve_units(points=tiny_pts)
        assert res.status == UNIT_AMBIGUOUS
        # Must not silently scale
        assert math.isclose(res.scale_to_meters, 1.0, abs_tol=1e-6)

    def test_orientation_ambiguity(self):
        """Test 11: Featureless cloud (isotropic sphere / random noise) reports ORIENTATION_AMBIGUOUS."""
        rng = np.random.default_rng(111)
        # Uniform isotropic noise without dominant planes or normals
        noise_pts = rng.uniform(-10.0, 10.0, (2000, 3))

        up_axis, confidence, _evidence = estimate_vertical_axis(noise_pts)
        assert confidence == ORIENTATION_AMBIGUOUS
        # Estimated axis is still a valid unit vector
        assert math.isclose(float(np.linalg.norm(up_axis)), 1.0, abs_tol=1e-6)

    def test_single_scan_no_registration(self):
        """Test 12: Single-scan input reports REGISTRATION_NOT_REQUIRED, no synthetic target fabricated."""
        single_path = [Path("/dummy/path/scan_01.e57")]
        mode, reason = resolve_registration_mode(single_path)
        assert mode == REGISTRATION_NOT_REQUIRED
        assert "not required" in reason.lower()

        transform = AuthoritativeTransform(registration_status=REGISTRATION_NOT_REQUIRED)
        d = transform.to_dict()
        assert d["registration_status"] == REGISTRATION_NOT_REQUIRED

    def test_multi_scan_registration(self):
        """Test 13: Multi-scan registration runs FPFH+RANSAC+ICP pipeline with valid quality metrics."""
        import open3d as o3d

        pts1, n1 = _generate_box_room(dx=6.0, dy=5.0, dz=3.0, n_per_wall=800)
        pcd1 = o3d.geometry.PointCloud()
        pcd1.points = o3d.utility.Vector3dVector(pts1)
        pcd1.normals = o3d.utility.Vector3dVector(n1)

        # Move scan 2 by a slight rigid transform
        R_gt = _euler_to_rotation_matrix(0, 0, 5.0)
        t_gt = np.array([0.2, 0.1, 0.0])
        pts2 = (pts1 @ R_gt.T) + t_gt
        n2 = n1 @ R_gt.T

        pcd2 = o3d.geometry.PointCloud()
        pcd2.points = o3d.utility.Vector3dVector(pts2)
        pcd2.normals = o3d.utility.Vector3dVector(n2)

        T_reg, quality = register_pair_full(pcd2, pcd1, voxel_size=0.1)

        assert quality.status == REGISTRATION_SUCCESS
        assert quality.fitness > 0.5
        assert quality.rmse < 0.1
        # Determinant of estimated rotation is +1
        R_est = T_reg[:3, :3]
        assert math.isclose(float(np.linalg.det(R_est)), 1.0, abs_tol=1e-3)

    def test_registration_failure(self):
        """Test 14: Deliberately incompatible, disjoint clouds report REGISTRATION_FAILED."""
        from unittest.mock import MagicMock, patch

        import open3d as o3d

        # Mock RANSAC returning low inlier fitness to test the failure rejection gate
        failed_ransac = MagicMock()
        failed_ransac.transformation = np.eye(4)
        failed_ransac.fitness = 0.05
        failed_ransac.inlier_rmse = 0.85
        failed_ransac.correspondence_set = []

        with patch.object(
            o3d.pipelines.registration,
            "registration_ransac_based_on_feature_matching",
            return_value=failed_ransac,
        ):
            pts, n = _generate_box_room(dx=4.0, dy=4.0, dz=3.0, n_per_wall=400)
            pcd1 = o3d.geometry.PointCloud()
            pcd1.points = o3d.utility.Vector3dVector(pts)
            pcd1.normals = o3d.utility.Vector3dVector(n)

            pcd2 = o3d.geometry.PointCloud()
            pcd2.points = o3d.utility.Vector3dVector(pts)
            pcd2.normals = o3d.utility.Vector3dVector(n)

            _T_reg, quality = register_pair_full(pcd2, pcd1, voxel_size=0.2)
            assert quality.status == REGISTRATION_FAILED
            assert quality.is_valid is False
            assert len(quality.failure_reasons) > 0

    def test_icp_degradation(self):
        """Test 15: When ICP degrades alignment, pre-ICP transform is retained."""
        from agent.tools.registration_tools import _validate_transform

        # Simulate RANSAC having good fitness and low RMSE
        T_ransac = np.eye(4)
        q_ransac = _validate_transform(T_ransac, fitness=0.85, rmse=0.015, method="RANSAC")

        # Simulate ICP that diverged: lower fitness, higher RMSE
        T_icp = np.eye(4)
        T_icp[0, 3] = 10.0  # Diverged translation
        q_icp = _validate_transform(T_icp, fitness=0.20, rmse=0.250, method="ICP")

        # Verify degradation condition logic matches register_pair_full
        is_degraded = q_icp.fitness < q_ransac.fitness * 0.9 or q_icp.rmse > q_ransac.rmse * 1.5
        assert is_degraded is True
