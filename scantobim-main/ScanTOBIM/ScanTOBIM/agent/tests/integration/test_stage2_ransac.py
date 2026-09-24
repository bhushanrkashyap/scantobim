"""
test_stage2_ransac.py — Integration tests for real Open3D RANSAC plane detection.
Requires real open3d (enforced by local conftest.py).
"""

import math
import numpy as np
import pytest

from agent.models import SegmentShape


def _generate_box_room_pcd():
    import open3d as o3d
    rng = np.random.default_rng(123)
    chunks = []

    # Wall A along X (0 to 5m, height 3m)
    n = 2000
    x = rng.uniform(0.0, 5.0, n)
    y = np.zeros(n)
    z = rng.uniform(0.0, 3.0, n)
    chunks.append(np.column_stack([x, y, z]) + rng.normal(0, 0.002, (n, 3)))

    # Wall B along Y (0 to 4m, height 3m)
    y2 = rng.uniform(0.0, 4.0, n)
    x2 = np.zeros(n)
    z2 = rng.uniform(0.0, 3.0, n)
    chunks.append(np.column_stack([x2, y2, z2]) + rng.normal(0, 0.002, (n, 3)))

    # Floor at Z = 0
    xf = rng.uniform(0.0, 5.0, n)
    yf = rng.uniform(0.0, 4.0, n)
    zf = np.zeros(n)
    chunks.append(np.column_stack([xf, yf, zf]) + rng.normal(0, 0.002, (n, 3)))

    all_pts = np.vstack(chunks)
    pcd = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(all_pts))
    pcd.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.1, max_nn=30))
    return pcd


def test_ransac_extracts_vertical_and_horizontal_planes():
    from agent.tools.scan_tools import detect_planes

    pcd = _generate_box_room_pcd()
    planes, residual = detect_planes(pcd, min_inliers=100)

    assert len(planes) >= 3, f"Expected at least 3 planes (2 walls + floor), got {len(planes)}"
    verticals = [p for p in planes if p["shape"] == SegmentShape.PLANE_VERTICAL]
    horizontals = [p for p in planes if p["shape"] == SegmentShape.PLANE_HORIZONTAL]

    assert len(verticals) >= 2, f"Expected at least 2 vertical planes, got {len(verticals)}"
    assert len(horizontals) >= 1, f"Expected at least 1 horizontal plane, got {len(horizontals)}"


def test_ransac_wall_axes_are_orthogonal():
    from agent.tools.scan_tools import detect_planes

    pcd = _generate_box_room_pcd()
    planes, _ = detect_planes(pcd, min_inliers=100)
    verticals = [p for p in planes if p["shape"] == SegmentShape.PLANE_VERTICAL]

    # Find the top 2 vertical planes
    v1, v2 = verticals[0], verticals[1]
    ax1 = np.array([v1["tags"]["wall_axis_x"], v1["tags"]["wall_axis_y"]])
    ax2 = np.array([v2["tags"]["wall_axis_x"], v2["tags"]["wall_axis_y"]])

    dot_product = abs(float(np.dot(ax1, ax2)))
    assert dot_product < 0.08, f"Expected near-orthogonal walls (dot ~ 0), got dot={dot_product:.4f}"
