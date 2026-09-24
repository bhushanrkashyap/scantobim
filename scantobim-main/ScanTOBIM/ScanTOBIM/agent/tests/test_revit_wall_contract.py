from __future__ import annotations

import numpy as np
import pytest

from agent.models import SegmentShape
from agent.tools.detection_config import DEFAULT_WALL_THICKNESS_MM
from agent.tools.scan_tools import segment_to_model


def _plane_vertical_without_wall_tags() -> dict:
    return {
        "shape": SegmentShape.PLANE_VERTICAL,
        "normal": np.array([0.0, 1.0, 0.0], dtype=float),
        "centroid": np.array([2.0, 3.0, 1.5], dtype=float),
        "mins": np.array([0.0, 2.8, 0.0], dtype=float),
        "maxs": np.array([4.0, 3.2, 3.0], dtype=float),
        "point_count": 1200,
        "confidence": 0.9,
        "tags": {
            "random_seed": 42,
        },
    }


def test_segment_to_model_synthesizes_wall_revit_contract_tags():
    raw = _plane_vertical_without_wall_tags()

    out = segment_to_model(raw, zone_id="zone-001", source_file="synthetic.e57")
    assert isinstance(out, list)
    assert len(out) >= 1

    wall = out[0]
    tags = wall.tags

    assert "wall_axis_x" in tags
    assert "wall_axis_y" in tags
    assert "wall_start_x_mm" in tags
    assert "wall_start_y_mm" in tags
    assert "wall_end_x_mm" in tags
    assert "wall_end_y_mm" in tags
    assert "wall_thickness_mm" in tags

    assert float(tags["wall_thickness_mm"]) >= 120.0

    sx = float(tags["wall_start_x_mm"])
    sy = float(tags["wall_start_y_mm"])
    ex = float(tags["wall_end_x_mm"])
    ey = float(tags["wall_end_y_mm"])
    length = float(np.linalg.norm([ex - sx, ey - sy]))
    assert length >= 100.0


def test_segment_to_model_angled_wall_does_not_inherit_aabb_thickness():
    """A 46° angled single-face wall must not get AABB-based thickness.

    Regression: the AABB of an angled wall face is the diagonal envelope of
    the plane, so min(dx, dy) ≈ 10,435 mm on the real user scan. That value
    was published as wall_thickness_mm, making Revit select monster wall
    types and fall back to whole-room DirectShape boxes.
    """
    raw = {
        "shape": SegmentShape.PLANE_VERTICAL,
        "normal": np.array([0.7274, -0.6862, -0.0012], dtype=float),
        "centroid": np.array([3.17, 3.98, 1.5], dtype=float),
        # Diagonal envelope of a 46° wall face — min(dx, dy) ≈ 10.4 m.
        "mins": np.array([-2.09, -1.11, 0.0], dtype=float),
        "maxs": np.array([8.34, 9.54, 3.0], dtype=float),
        "point_count": 36000,
        "confidence": 1.0,
        "tags": {"random_seed": 42},
    }

    out = segment_to_model(raw, zone_id="ZONE_PLY")
    wall = out[0]
    tags = wall.tags

    assert tags["thickness_mode"] == "inferred_single_face"
    assert float(tags["wall_thickness_mm"]) == pytest.approx(DEFAULT_WALL_THICKNESS_MM)
    # Centerline must still carry the true 46° orientation.
    assert float(tags["wall_angle_deg"]) == pytest.approx(46.7, abs=1.5)


def test_segment_to_model_axis_aligned_wall_keeps_aabb_face_extent():
    """Axis-aligned walls keep min(dx, dy) as thickness (true face extent)."""
    raw = {
        "shape": SegmentShape.PLANE_VERTICAL,
        "normal": np.array([0.0, 1.0, 0.0], dtype=float),
        "centroid": np.array([2.0, 3.0, 1.5], dtype=float),
        "mins": np.array([0.0, 2.8, 0.0], dtype=float),
        "maxs": np.array([4.0, 3.2, 3.0], dtype=float),
        "point_count": 1200,
        "confidence": 0.9,
        "tags": {},
    }

    out = segment_to_model(raw, zone_id="zone-001")
    tags = out[0].tags

    assert tags["thickness_mode"] == "aabb_face_extent"
    assert float(tags["wall_thickness_mm"]) == pytest.approx(400.0)


def test_segment_to_model_preserves_existing_wall_endpoint_tags():
    raw = _plane_vertical_without_wall_tags()
    raw["tags"] = {
        "wall_axis_x": 1.0,
        "wall_axis_y": 0.0,
        "wall_start_x_mm": 100.0,
        "wall_start_y_mm": 200.0,
        "wall_end_x_mm": 5100.0,
        "wall_end_y_mm": 200.0,
        "wall_thickness_mm": 180.0,
    }

    out = segment_to_model(raw, zone_id="zone-001")
    wall = out[0]
    tags = wall.tags

    assert float(tags["wall_start_x_mm"]) == 100.0
    assert float(tags["wall_end_x_mm"]) == 5100.0
    assert float(tags["wall_thickness_mm"]) == 180.0
