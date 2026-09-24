"""
Unit tests for wall axis orientation fix — NO open3d required.

Tests here cover:
  - merge_coplanar_walls() logic (pure dict/numpy operations)
  - WallGeometryHelper tag parsing (pure dict operations)

They run under the standard conftest mock and are fast (<1s total).

Run with:  pytest agent/tests/test_wall_axis.py -v
"""

from __future__ import annotations

import math

import numpy as np
import pytest

# ── Helpers ───────────────────────────────────────────────────────────────────


def _make_wall_seg(angle_deg: float, start_m: float, end_m: float, z_m: float = 0.0) -> dict:
    """Minimal wall segment dict — no Open3D needed."""
    from agent.models import SegmentShape

    angle_rad = math.radians(angle_deg)
    ax = math.cos(angle_rad)
    ay = math.sin(angle_rad)
    length_m = abs(end_m - start_m)
    cx = ((start_m + end_m) / 2.0) * ax
    cy = ((start_m + end_m) / 2.0) * ay

    return {
        "shape": SegmentShape.PLANE_VERTICAL,
        "normal": np.array([-ay, ax, 0.0]),
        "centroid": np.array([cx, cy, z_m + 1.5]),
        "mins": np.array([min(start_m * ax, end_m * ax), min(start_m * ay, end_m * ay), z_m]),
        "maxs": np.array([max(start_m * ax, end_m * ax), max(start_m * ay, end_m * ay), z_m + 3.0]),
        "point_count": max(int(length_m * 200), 1),
        "confidence": 0.85,
        "tags": {
            "wall_axis_x": round(ax, 6),
            "wall_axis_y": round(ay, 6),
            "wall_length_mm": round(length_m * 1000.0, 1),
            "wall_thickness_mm": 200.0,
            "_wall_start_x_m": start_m * ax,
            "_wall_start_y_m": start_m * ay,
            "_wall_end_x_m": end_m * ax,
            "_wall_end_y_m": end_m * ay,
            "wall_axis_source": "ransac",
        },
    }


# ── merge_coplanar_walls unit tests ───────────────────────────────────────────


def test_merge_two_adjacent_walls():
    """Two co-directional walls with ≤300 mm gap merge into one."""
    from agent.tools.scan_tools import merge_coplanar_walls

    seg1 = _make_wall_seg(0.0, 0.0, 3.0)
    seg2 = _make_wall_seg(0.0, 3.1, 6.0)  # 100 mm gap
    result = merge_coplanar_walls([seg1, seg2])
    assert len(result) == 1, f"Expected 1 merged wall, got {len(result)}"
    assert result[0]["tags"].get("wall_merged") is True
    assert result[0]["tags"]["wall_length_mm"] == pytest.approx(6000.0, abs=100.0)


def test_no_merge_large_gap():
    """Walls with >300 mm gap must NOT merge."""
    from agent.tools.scan_tools import merge_coplanar_walls

    seg1 = _make_wall_seg(0.0, 0.0, 3.0)
    seg2 = _make_wall_seg(0.0, 3.5, 6.0)  # 500 mm gap
    result = merge_coplanar_walls([seg1, seg2])
    assert len(result) == 2


def test_no_merge_different_directions():
    """Walls more than 2° apart must not merge."""
    from agent.tools.scan_tools import merge_coplanar_walls

    seg1 = _make_wall_seg(0.0, 0.0, 3.0)
    seg2 = _make_wall_seg(45.0, 3.0, 6.0)
    result = merge_coplanar_walls([seg1, seg2])
    assert len(result) == 2


def test_non_wall_segments_passthrough():
    """Non-PLANE_VERTICAL segments pass through merge unchanged."""
    from agent.models import SegmentShape
    from agent.tools.scan_tools import merge_coplanar_walls

    floor_seg = {
        "shape": SegmentShape.PLANE_HORIZONTAL,
        "normal": np.array([0.0, 0.0, 1.0]),
        "centroid": np.array([0.0, 0.0, 0.0]),
        "mins": np.zeros(3),
        "maxs": np.ones(3),
        "point_count": 500,
        "confidence": 0.9,
        "tags": {},
    }
    wall_seg = _make_wall_seg(30.0, 0.0, 4.0)
    result = merge_coplanar_walls([floor_seg, wall_seg])
    assert len(result) == 2
    shapes = {s["shape"] for s in result}
    assert SegmentShape.PLANE_HORIZONTAL in shapes


def test_merge_three_walls_chain():
    """Three chained walls that each gap ≤300 mm should collapse to one."""
    from agent.tools.scan_tools import merge_coplanar_walls

    seg1 = _make_wall_seg(0.0, 0.0, 3.0)
    seg2 = _make_wall_seg(0.0, 3.1, 6.0)  # 100 mm gap after seg1
    seg3 = _make_wall_seg(0.0, 6.15, 9.0)  # 150 mm gap after seg2
    result = merge_coplanar_walls([seg1, seg2, seg3])
    assert len(result) == 1, f"Expected 1 merged wall, got {len(result)}"
    assert result[0]["tags"]["wall_length_mm"] == pytest.approx(9000.0, abs=200.0)


def test_no_merge_parallel_offset_walls():
    """Parallel walls with close run gap but lateral separation must remain separate."""
    from agent.tools.scan_tools import merge_coplanar_walls

    seg1 = _make_wall_seg(0.0, 0.0, 3.0)
    seg2 = _make_wall_seg(0.0, 3.1, 6.0)
    seg2["tags"]["_wall_start_y_m"] = 0.35
    seg2["tags"]["_wall_end_y_m"] = 0.35
    seg2["centroid"][1] = 0.35
    seg2["mins"][1] = 0.35
    seg2["maxs"][1] = 0.35

    result = merge_coplanar_walls([seg1, seg2])
    assert len(result) == 2


def test_merge_handles_reversed_endpoint_tags():
    """Reversed start/end tags should still merge based on projected interval gap."""
    from agent.tools.scan_tools import merge_coplanar_walls

    seg1 = _make_wall_seg(0.0, 0.0, 3.0)
    seg2 = _make_wall_seg(0.0, 3.1, 6.0)
    seg2["tags"]["_wall_start_x_m"] = 6.0
    seg2["tags"]["_wall_end_x_m"] = 3.1

    result = merge_coplanar_walls([seg1, seg2])
    assert len(result) == 1
    assert result[0]["tags"].get("wall_merged") is True
    assert result[0]["tags"]["wall_length_mm"] == pytest.approx(6000.0, abs=100.0)


def test_wall_axis_tags_after_merge():
    """Merged wall must preserve wall_axis_x/y and wall_thickness_mm from input."""
    from agent.tools.scan_tools import merge_coplanar_walls

    seg1 = _make_wall_seg(30.0, 0.0, 3.0)
    seg2 = _make_wall_seg(30.0, 3.1, 6.0)
    result = merge_coplanar_walls([seg1, seg2])
    assert len(result) == 1
    tags = result[0]["tags"]
    assert "wall_axis_x" in tags
    assert "wall_axis_y" in tags
    assert tags["wall_thickness_mm"] == pytest.approx(200.0)


def test_empty_input_returns_empty():
    """merge_coplanar_walls([]) must return []."""
    from agent.tools.scan_tools import merge_coplanar_walls

    assert merge_coplanar_walls([]) == []


def test_single_wall_passthrough():
    """A single wall segment passes through unchanged (nothing to merge)."""
    from agent.tools.scan_tools import merge_coplanar_walls

    seg = _make_wall_seg(45.0, 0.0, 5.0)
    result = merge_coplanar_walls([seg])
    assert len(result) == 1
    assert result[0]["tags"].get("wall_merged") is not True


# ── Wall axis tag structure tests (dict-only, no open3d) ─────────────────────


def test_wall_axis_tags_present_in_seg():
    """Verify the tag dict produced by _make_wall_seg has all required keys."""
    seg = _make_wall_seg(45.0, 0.0, 5.0)
    required_keys = (
        "wall_axis_x",
        "wall_axis_y",
        "wall_length_mm",
        "wall_thickness_mm",
        "_wall_start_x_m",
        "_wall_start_y_m",
        "_wall_end_x_m",
        "_wall_end_y_m",
        "wall_axis_source",
    )
    for key in required_keys:
        assert key in seg["tags"], f"Tag '{key}' missing"


@pytest.mark.parametrize(
    "angle_deg,expected_ax,expected_ay",
    [
        (0.0, 1.0, 0.0),
        (90.0, 0.0, 1.0),
        (180.0, -1.0, 0.0),
    ],
)
def test_wall_axis_direction_math(angle_deg, expected_ax, expected_ay):
    """Wall axis direction vector matches expected values for cardinal angles."""
    seg = _make_wall_seg(angle_deg, 0.0, 5.0)
    ax = seg["tags"]["wall_axis_x"]
    ay = seg["tags"]["wall_axis_y"]
    assert abs(ax - expected_ax) < 0.001, f"axis_x: expected {expected_ax}, got {ax}"
    assert abs(ay - expected_ay) < 0.001, f"axis_y: expected {expected_ay}, got {ay}"


def test_wall_start_end_span_correct():
    """wall_start/end should span ±half_length along the wall axis."""
    seg = _make_wall_seg(0.0, 0.0, 6.0)  # 6 m X-axis wall
    sx = seg["tags"]["_wall_start_x_m"]
    ex = seg["tags"]["_wall_end_x_m"]
    assert abs(ex - sx - 6.0) < 0.01, f"Expected 6m span, got {ex - sx:.3f}"


# ── merge_coplanar_horizontal unit tests ─────────────────────────────────────


def test_merge_coplanar_horizontal_floors():
    from agent.models import SegmentShape
    from agent.tools.scan_tools import merge_coplanar_horizontal

    seg1 = {
        "shape": SegmentShape.PLANE_HORIZONTAL,
        "normal": np.array([0.0, 0.0, 1.0]),
        "centroid": np.array([2.0, 2.0, 0.0]),
        "mins": np.array([0.0, 0.0, 0.0]),
        "maxs": np.array([4.0, 4.0, 0.0]),
        "point_count": 100,
        "confidence": 0.8,
        "tags": {},
    }
    seg2 = {
        "shape": SegmentShape.PLANE_HORIZONTAL,
        "normal": np.array([0.0, 0.0, 1.0]),
        "centroid": np.array([6.0, 2.0, 0.0]),
        "mins": np.array([4.1, 0.0, 0.0]),  # gap of 0.1
        "maxs": np.array([8.0, 4.0, 0.0]),
        "point_count": 100,
        "confidence": 0.8,
        "tags": {},
    }
    result = merge_coplanar_horizontal([seg1, seg2], gap_threshold_m=0.50)
    assert len(result) == 1
    assert result[0]["tags"].get("floor_merged") is True
    assert np.allclose(result[0]["mins"], [0.0, 0.0, 0.0])
    assert np.allclose(result[0]["maxs"], [8.0, 4.0, 0.0])


def test_no_merge_different_z_floors():
    from agent.models import SegmentShape
    from agent.tools.scan_tools import merge_coplanar_horizontal

    seg1 = {
        "shape": SegmentShape.PLANE_HORIZONTAL,
        "normal": np.array([0.0, 0.0, 1.0]),
        "centroid": np.array([2.0, 2.0, 0.0]),
        "mins": np.array([0.0, 0.0, 0.0]),
        "maxs": np.array([4.0, 4.0, 0.0]),
        "point_count": 100,
        "confidence": 0.8,
        "tags": {},
    }
    seg2 = {
        "shape": SegmentShape.PLANE_HORIZONTAL,
        "normal": np.array([0.0, 0.0, 1.0]),
        "centroid": np.array([6.0, 2.0, 1.0]),  # Different Z
        "mins": np.array([4.1, 0.0, 1.0]),
        "maxs": np.array([8.0, 4.0, 1.0]),
        "point_count": 100,
        "confidence": 0.8,
        "tags": {},
    }
    result = merge_coplanar_horizontal([seg1, seg2], z_tolerance_m=0.05)
    assert len(result) == 2


# ── merge_collinear_cylinders unit tests ─────────────────────────────────────


def test_merge_collinear_cylinders(monkeypatch):
    monkeypatch.setenv("STB_ENABLE_CYLINDER_CHAIN_MERGE", "1")
    from agent.models import SegmentShape
    from agent.tools.scan_tools import merge_collinear_cylinders

    seg1 = {
        "shape": SegmentShape.CYLINDER,
        "centroid": np.array([0.0, 0.0, 0.0]),
        "mins": np.array([-1.0, -0.1, -0.1]),
        "maxs": np.array([1.0, 0.1, 0.1]),
        "point_count": 100,
        "confidence": 0.8,
        "tags": {
            "cyl_axis_x": 1.0,
            "cyl_axis_y": 0.0,
            "cyl_axis_z": 0.0,
            "fitted_radius_mm": 100.0,
        },
    }
    seg2 = {
        "shape": SegmentShape.CYLINDER,
        "centroid": np.array([2.5, 0.0, 0.0]),
        "mins": np.array([1.5, -0.1, -0.1]),
        "maxs": np.array([3.5, 0.1, 0.1]),
        "point_count": 100,
        "confidence": 0.8,
        "tags": {
            "cyl_axis_x": 1.0,
            "cyl_axis_y": 0.0,
            "cyl_axis_z": 0.0,
            "fitted_radius_mm": 105.0,  # within tolerance
        },
    }
    result = merge_collinear_cylinders([seg1, seg2], gap_threshold_m=1.0)
    assert len(result) == 1
    assert result[0]["tags"].get("cylinder_merged") is True
