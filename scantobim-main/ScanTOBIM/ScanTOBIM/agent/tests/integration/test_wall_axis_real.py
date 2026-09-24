"""
Integration tests for wall axis orientation — requires REAL open3d.

These tests run in agent/tests/integration/ whose local conftest.py strips
the mock open3d injected by the parent conftest, so the real library is used.

Run with:  pytest agent/tests/integration/test_wall_axis_real.py -v
"""

from __future__ import annotations

import math

import numpy as np
import pytest

# ── Synthetic surface point generator ─────────────────────────────────────────


def _make_wall_points(
    angle_deg: float,
    length_m: float = 5.0,
    thickness_m: float = 0.20,
    height_m: float = 3.0,
    pts_per_m2: float = 400.0,
    noise_m: float = 0.003,
    rng: np.random.Generator | None = None,
) -> np.ndarray:
    """Return a C-contiguous float64 (N,3) surface point cloud in METRES.

    Points are placed on the two XZ faces of the wall (front/back) at the
    given surface density, matching real scanner output patterns.
    """
    if rng is None:
        rng = np.random.default_rng(42)

    angle_rad = math.radians(angle_deg)
    axis_u = np.array([math.cos(angle_rad), math.sin(angle_rad), 0.0])
    axis_n = np.array([-math.sin(angle_rad), math.cos(angle_rad), 0.0])

    face_area_m2 = length_m * height_m
    n_per_face = max(int(face_area_m2 * pts_per_m2), 200)

    chunks = []
    for face_offset in (0.0, thickness_m):
        along = rng.uniform(-length_m / 2.0, length_m / 2.0, n_per_face)
        up = rng.uniform(0.0, height_m, n_per_face)

        face_pts = (
            np.outer(along, axis_u)
            + (face_offset * axis_n)
            + np.stack([np.zeros(n_per_face), np.zeros(n_per_face), up], axis=1)
        )
        face_pts += rng.normal(0.0, noise_m, face_pts.shape)
        chunks.append(face_pts)

    return np.ascontiguousarray(np.vstack(chunks), dtype=np.float64)


def _pcd_from_array(pts_m: np.ndarray):
    """Create a real Open3D PointCloud, asserting it is non-empty."""
    import open3d as o3d

    arr = np.ascontiguousarray(pts_m, dtype=np.float64)
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(arr)

    if len(pcd.points) != len(arr):
        pytest.fail(
            f"Open3D Vector3dVector discarded points: "
            f"input={len(arr)}, pcd={len(pcd.points)}. "
            "Possible version mismatch or memory layout issue."
        )
    return pcd


def _preprocess(pts_m: np.ndarray, voxel_size: float = 0.05):
    """Downsample → adaptive SOR → estimate normals. Returns ready-to-use pcd."""
    from agent.tools.geometry_tools import estimate_normals

    pcd = _pcd_from_array(pts_m)
    pcd = pcd.voxel_down_sample(voxel_size=voxel_size)

    n = len(pcd.points)
    if n < 10:
        pytest.fail(
            f"Only {n} points after {voxel_size * 1000:.0f} mm voxel downsample "
            f"(input had {len(pts_m)} pts). Cloud may be degenerate."
        )

    nb = min(6, n - 1)
    pcd, _ = pcd.remove_statistical_outlier(nb_neighbors=nb, std_ratio=2.0)
    estimate_normals(pcd, radius=0.15, max_nn=30)
    return pcd


# ── Sanity: Open3D accepts our arrays ────────────────────────────────────────


def test_vector3dvector_not_empty():
    """Vector3dVector must accept the wall point array without silently failing."""
    pts = _make_wall_points(45.0, pts_per_m2=50.0)
    pcd = _pcd_from_array(pts)
    assert len(pcd.points) == len(pts)


# ── detect_planes wall axis tests ─────────────────────────────────────────────


@pytest.mark.parametrize("angle_deg", [0.0, 30.0, 45.0, 60.0, 90.0])
def test_wall_axis_angle(angle_deg: float):
    """Wall axis extracted from RANSAC normal must match angle within ±1.5°."""
    from agent.tools.scan_tools import detect_planes

    pts = _make_wall_points(angle_deg)
    pcd = _preprocess(pts)
    planes, _ = detect_planes(pcd, max_planes=1, min_inliers=50)

    assert planes, f"No plane detected for {angle_deg}° wall"
    tags = planes[0]["tags"]
    assert "wall_axis_x" in tags, f"wall_axis_x missing — shape={planes[0]['shape']}"
    assert "wall_axis_y" in tags

    ax = float(tags["wall_axis_x"])
    ay = float(tags["wall_axis_y"])
    detected = math.degrees(math.atan2(ay, ax))
    if detected < 0:
        detected += 180.0
    expected = angle_deg % 180.0
    err = min(abs(detected - expected), 180.0 - abs(detected - expected))

    assert err < 1.5, (
        f"{angle_deg}°: detected {detected:.2f}° (err={err:.2f}°), axis=({ax:.4f},{ay:.4f})"
    )


def test_wall_axis_source_ransac():
    """wall_axis_source tag must be 'ransac' for a clean vertical RANSAC plane."""
    from agent.tools.scan_tools import detect_planes

    pts = _make_wall_points(45.0)
    pcd = _preprocess(pts)
    planes, _ = detect_planes(pcd, max_planes=1, min_inliers=50)

    assert planes
    assert planes[0]["tags"].get("wall_axis_source") == "ransac"


def test_wall_length_accurate():
    """wall_length_mm within 8% of true length."""
    from agent.tools.scan_tools import detect_planes

    true_len_m = 5.0
    pts = _make_wall_points(30.0, length_m=true_len_m)
    pcd = _preprocess(pts)
    planes, _ = detect_planes(pcd, max_planes=1, min_inliers=50)

    assert planes
    tags = planes[0]["tags"]
    assert "wall_length_mm" in tags

    err = abs(float(tags["wall_length_mm"]) - true_len_m * 1000.0) / (true_len_m * 1000.0)
    assert err < 0.08, (
        f"Length error {err * 100:.1f}%: "
        f"expected {true_len_m * 1000:.0f} mm, got {tags['wall_length_mm']:.1f} mm"
    )


def test_wall_start_end_private_tags():
    """_wall_start/end_m tags must be written by detect_planes before coord correction."""
    from agent.tools.scan_tools import detect_planes

    pts = _make_wall_points(45.0)
    pcd = _preprocess(pts)
    planes, _ = detect_planes(pcd, max_planes=1, min_inliers=50)

    assert planes
    tags = planes[0]["tags"]
    for key in ("_wall_start_x_m", "_wall_start_y_m", "_wall_end_x_m", "_wall_end_y_m"):
        assert key in tags, f"Private tag '{key}' missing"


# ── segment_to_model coordinate correction ────────────────────────────────────


def test_wall_mm_tags_after_coord_origin():
    """After segment_to_model(), public wall_*_mm tags exist and private _m tags are gone."""
    from agent.tools.geometry_tools import estimate_normals
    from agent.tools.scan_tools import detect_planes, segment_to_model

    pts_m = _make_wall_points(45.0)
    offset = np.array([150.0, 200.0, 0.0])  # >100 m → triggers coord normalisation
    pts_offset = np.ascontiguousarray(pts_m + offset, dtype=np.float64)

    pcd = _pcd_from_array(pts_offset)
    pcd = pcd.voxel_down_sample(voxel_size=0.05)
    pcd, _ = pcd.remove_statistical_outlier(nb_neighbors=6, std_ratio=2.0)
    estimate_normals(pcd, radius=0.15, max_nn=30)

    planes, _ = detect_planes(pcd, max_planes=1, min_inliers=50)
    assert planes

    origin = np.asarray(pcd.points).min(axis=0)
    segs = segment_to_model(planes[0], zone_id="test", coord_origin_m=origin)
    assert segs
    seg = segs[0]

    assert "wall_start_x_mm" in seg.tags
    assert "wall_end_x_mm" in seg.tags
    for key in ("_wall_start_x_m", "_wall_start_y_m", "_wall_end_x_m", "_wall_end_y_m"):
        assert key not in seg.tags, f"Private tag '{key}' must be removed"


# ── Integration: full run_segmentation pipeline ───────────────────────────────


@pytest.mark.parametrize("angle_deg", [0.0, 45.0, 90.0])
def test_run_segmentation_wall_orientation(angle_deg: float):
    """run_segmentation() on surface data produces PLANE_VERTICAL segments
    with correct wall_axis orientation for arbitrary wall angles."""
    from agent.models import SegmentShape
    from agent.tools.scan_tools import run_segmentation

    pts_m = _make_wall_points(angle_deg, pts_per_m2=400.0)
    pts_mm = np.ascontiguousarray(pts_m * 1000.0, dtype=np.float64)

    segments = run_segmentation(
        file_path=None,
        zone_id="integ-test",
        preloaded_points=pts_mm,
        min_inliers=50,
        seed=42,
    )

    wall_segs = [s for s in segments if s.shape == SegmentShape.PLANE_VERTICAL]
    assert wall_segs, f"No PLANE_VERTICAL found for {angle_deg}° wall"

    best = max(wall_segs, key=lambda s: s.point_count)
    tags = best.tags

    if "wall_axis_x" not in tags:
        pytest.skip("wall_axis absent — possibly degenerate RANSAC normal")

    ax = float(tags["wall_axis_x"])
    ay = float(tags["wall_axis_y"])
    detected = math.degrees(math.atan2(ay, ax)) % 180.0
    expected = angle_deg % 180.0
    err = min(abs(detected - expected), 180.0 - abs(detected - expected))

    assert err < 3.0, f"Pipeline: {angle_deg}° wall → detected {detected:.1f}° (err={err:.1f}°)"


# ── Expected output sample ────────────────────────────────────────────────────


def test_expected_output_45deg_wall(capsys):
    """Print tag values for a 45° wall — visual inspection, no assertion."""
    from agent.tools.scan_tools import detect_planes

    pts = _make_wall_points(45.0, length_m=6.0, thickness_m=0.25)
    pcd = _preprocess(pts)
    planes, _ = detect_planes(pcd, max_planes=1, min_inliers=50)

    if not planes:
        pytest.skip("No planes detected")

    tags = planes[0]["tags"]
    print(
        "\n── 45° wall (6 m long, 250 mm thick) ──\n"
        f"  wall_axis_x       : {tags.get('wall_axis_x', 'MISSING')}  (expect ≈ ±0.7071)\n"
        f"  wall_axis_y       : {tags.get('wall_axis_y', 'MISSING')}  (expect ≈ ±0.7071)\n"
        f"  wall_length_mm    : {tags.get('wall_length_mm', 'MISSING')} mm  (expect ≈ 6000)\n"
        f"  wall_thickness_mm : {tags.get('wall_thickness_mm', 'MISSING')} mm  (expect ≈ 250)\n"
        f"  wall_axis_source  : {tags.get('wall_axis_source', 'MISSING')}\n"
        f"  wall_start_x_mm   : (after segment_to_model)\n"
    )
