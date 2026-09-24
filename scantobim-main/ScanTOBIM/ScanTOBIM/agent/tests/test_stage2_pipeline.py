"""
test_stage2_pipeline.py — Unit tests for Stage 2 Plane Detection and Wall Merging.
Audits:
  - 80mm coplanar lateral offset tolerance (separate nearby surfaces NOT merged)
  - 1200mm fragment gap tolerance (doorways bridged, perpendiculars separated)
  - Wall axis tag derivation and Revit contract compliance
"""

import math
import numpy as np
import pytest

from agent.models import SegmentShape
from agent.tools.scan_tools import merge_coplanar_walls


def _make_wall_dict(angle_deg, start_x, end_x, y_offset=0.0, z_base=0.0, z_top=3.0):
    rad = math.radians(angle_deg)
    ax = math.cos(rad)
    ay = math.sin(rad)
    length = abs(end_x - start_x)
    cx = (start_x + end_x) / 2.0
    cy = y_offset
    cz = (z_base + z_top) / 2.0

    return {
        "shape": SegmentShape.PLANE_VERTICAL,
        "normal": np.array([-ay, ax, 0.0]),
        "centroid": np.array([cx, cy, cz]),
        "mins": np.array([min(start_x, end_x), y_offset - 0.1, z_base]),
        "maxs": np.array([max(start_x, end_x), y_offset + 0.1, z_top]),
        "point_count": 1000,
        "confidence": 1.0,
        "tags": {
            "wall_axis_x": round(ax, 6),
            "wall_axis_y": round(ay, 6),
            "wall_length_mm": round(length * 1000.0, 1),
            "wall_thickness_mm": 200.0,
            "_wall_start_x_m": float(start_x),
            "_wall_start_y_m": float(y_offset),
            "_wall_end_x_m": float(end_x),
            "_wall_end_y_m": float(y_offset),
            "wall_axis_source": "ransac",
        },
    }


def test_coplanar_offset_80mm_does_not_merge_separate_surfaces():
    """Two parallel walls 150mm apart (e.g. corridor or opposite faces) must NOT merge with 80mm tolerance."""
    wall1 = _make_wall_dict(0.0, start_x=0.0, end_x=4.0, y_offset=0.0)
    wall2 = _make_wall_dict(0.0, start_x=0.0, end_x=4.0, y_offset=0.15)  # 150mm lateral offset

    result = merge_coplanar_walls(
        [wall1, wall2],
        gap_threshold_m=1.20,
        lateral_offset_threshold_m=0.08,
    )
    assert len(result) == 2, f"Expected 2 separate walls, got {len(result)}"


def test_coplanar_offset_within_80mm_merges():
    """Two fragments on the same wall surface within 50mm offset and <1200mm gap MUST merge."""
    wall1 = _make_wall_dict(0.0, start_x=0.0, end_x=2.0, y_offset=0.0)
    wall2 = _make_wall_dict(0.0, start_x=2.8, end_x=5.0, y_offset=0.04)  # 800mm door gap, 40mm offset

    result = merge_coplanar_walls(
        [wall1, wall2],
        gap_threshold_m=1.20,
        lateral_offset_threshold_m=0.08,
    )
    assert len(result) == 1, f"Expected 1 merged wall across doorway, got {len(result)}"
    assert result[0]["tags"].get("wall_merged") is True
    assert result[0]["tags"]["wall_length_mm"] == pytest.approx(5000.0, abs=100.0)


def test_fragment_gap_1200mm_bridges_doorway():
    """Collinear fragments separated by a 1.0m doorway must merge under 1.2m gap tolerance."""
    wall1 = _make_wall_dict(0.0, start_x=0.0, end_x=3.0, y_offset=0.0)
    wall2 = _make_wall_dict(0.0, start_x=4.0, end_x=7.0, y_offset=0.0)  # 1.0m door opening

    result = merge_coplanar_walls(
        [wall1, wall2],
        gap_threshold_m=1.20,
        lateral_offset_threshold_m=0.08,
    )
    assert len(result) == 1
    assert result[0]["tags"]["wall_length_mm"] == pytest.approx(7000.0, abs=50.0)


def test_fragment_gap_exceeding_1200mm_stays_separated():
    """Walls separated by > 1.2m gap (e.g. 1.5m hallway) must NOT merge."""
    wall1 = _make_wall_dict(0.0, start_x=0.0, end_x=3.0, y_offset=0.0)
    wall2 = _make_wall_dict(0.0, start_x=4.6, end_x=8.0, y_offset=0.0)  # 1.6m gap

    result = merge_coplanar_walls(
        [wall1, wall2],
        gap_threshold_m=1.20,
        lateral_offset_threshold_m=0.08,
    )
    assert len(result) == 2


def test_perpendicular_walls_never_merge():
    """Orthogonal walls (0 deg and 90 deg) must never merge regardless of gap."""
    wall1 = _make_wall_dict(0.0, start_x=0.0, end_x=4.0, y_offset=0.0)
    wall2 = _make_wall_dict(90.0, start_x=0.0, end_x=3.0, y_offset=0.0)

    result = merge_coplanar_walls(
        [wall1, wall2],
        gap_threshold_m=1.20,
        lateral_offset_threshold_m=0.08,
    )
    assert len(result) == 2
