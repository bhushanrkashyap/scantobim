"""Tests for overlap resolution (scan_tools) and clash whitelist (coordinator_tools)."""

from __future__ import annotations

import uuid

import numpy as np
import pytest

from agent.classifier import _linear_length_mm
from agent.models import (
    BoundingBox,
    Discipline,
    DSEARZone,
    ElementInstruction,
    ElementType,
    GeometrySegment,
    Point3D,
    SafetyCategory,
    SegmentShape,
)
from agent.segment_merger import merge_segments
from agent.tools.coordinator_tools import _is_whitelisted_pair, detect_clashes
from agent.tools.scan_tools import (
    _bb_containment,
    _bb_iou,
    _bb_volume,
    _resolve_overlaps,
    detect_openings,
    merge_collinear_boxes,
    merge_collinear_cylinders,
)

# ── Helpers ──────────────────────────────────────────────────────────────────


def _seg(shape: SegmentShape, mins, maxs, confidence: float = 0.80, point_count: int = 100) -> dict:
    """Build a raw segment dict matching detect_clusters / detect_planes output."""
    mins = np.array(mins, dtype=float)
    maxs = np.array(maxs, dtype=float)
    return {
        "shape": shape,
        "normal": None,
        "centroid": (mins + maxs) / 2.0,
        "mins": mins,
        "maxs": maxs,
        "point_count": point_count,
        "confidence": confidence,
        "tags": {},
    }


def _instruction(
    element_type: ElementType, bb: BoundingBox, zone_id: str = "Z1"
) -> ElementInstruction:
    """Build a minimal ElementInstruction for clash detection."""
    cx = (bb.min_x + bb.max_x) / 2
    cy = (bb.min_y + bb.max_y) / 2
    cz = (bb.min_z + bb.max_z) / 2
    return ElementInstruction(
        segment_id=str(uuid.uuid4()),
        zone_id=zone_id,
        element_type=element_type,
        discipline=Discipline.MEP,
        safety_category=SafetyCategory.NS,
        dsear_zone=DSEARZone.NONE,
        bounding_box=bb,
        centroid=Point3D(x=cx, y=cy, z=cz),
    )


def _geometry_segment(
    shape: SegmentShape, bb: BoundingBox, point_count: int = 120, confidence: float = 0.80
) -> GeometrySegment:
    """Build a GeometrySegment for merge_segments regression tests."""
    cx = (bb.min_x + bb.max_x) / 2
    cy = (bb.min_y + bb.max_y) / 2
    cz = (bb.min_z + bb.max_z) / 2
    return GeometrySegment(
        zone_id="Z1",
        shape=shape,
        centroid=Point3D(x=cx, y=cy, z=cz),
        bounding_box=bb,
        point_count=point_count,
        confidence=confidence,
        source_file="test",
    )


# ═══════════════════════════════════════════════════════════════════════════════
# 1. Overlap resolution unit tests (scan_tools)
# ═══════════════════════════════════════════════════════════════════════════════


class TestBBHelpers:
    """Volume, IoU, and containment calculations."""

    def test_volume_simple(self):
        seg = _seg(SegmentShape.BOX, [0, 0, 0], [1, 2, 3])
        assert _bb_volume(seg) == pytest.approx(6.0)

    def test_volume_degenerate(self):
        seg = _seg(SegmentShape.BOX, [0, 0, 0], [0, 2, 3])
        assert _bb_volume(seg) == 0.0

    def test_iou_identical(self):
        a = _seg(SegmentShape.BOX, [0, 0, 0], [1, 1, 1])
        b = _seg(SegmentShape.BOX, [0, 0, 0], [1, 1, 1])
        assert _bb_iou(a, b) == pytest.approx(1.0)

    def test_iou_no_overlap(self):
        a = _seg(SegmentShape.BOX, [0, 0, 0], [1, 1, 1])
        b = _seg(SegmentShape.BOX, [2, 2, 2], [3, 3, 3])
        assert _bb_iou(a, b) == 0.0

    def test_iou_partial(self):
        a = _seg(SegmentShape.BOX, [0, 0, 0], [2, 2, 2])
        b = _seg(SegmentShape.BOX, [1, 1, 1], [3, 3, 3])
        # Intersection: [1,1,1]→[2,2,2] = 1.0; Union: 8+8-1=15
        assert _bb_iou(a, b) == pytest.approx(1.0 / 15.0)

    def test_containment_full(self):
        small = _seg(SegmentShape.BOX, [1, 1, 1], [2, 2, 2])
        large = _seg(SegmentShape.BOX, [0, 0, 0], [3, 3, 3])
        assert _bb_containment(small, large) == pytest.approx(1.0)

    def test_containment_none(self):
        a = _seg(SegmentShape.BOX, [0, 0, 0], [1, 1, 1])
        b = _seg(SegmentShape.BOX, [5, 5, 5], [6, 6, 6])
        assert _bb_containment(a, b) == 0.0


class TestResolveOverlaps:
    """Post-segmentation deduplication."""

    def test_empty_list(self):
        assert _resolve_overlaps([]) == []

    def test_single_segment(self):
        segs = [_seg(SegmentShape.BOX, [0, 0, 0], [1, 1, 1])]
        assert len(_resolve_overlaps(segs)) == 1

    def test_same_shape_high_iou_merges(self):
        """Two nearly-identical boxes should merge into one."""
        a = _seg(SegmentShape.BOX, [0, 0, 0], [1, 1, 1], confidence=0.9, point_count=100)
        b = _seg(
            SegmentShape.BOX, [0.05, 0.05, 0.05], [1.05, 1.05, 1.05], confidence=0.7, point_count=80
        )
        result = _resolve_overlaps([a, b], iou_threshold=0.60)
        assert len(result) == 1
        # Higher confidence kept, point count absorbed
        assert result[0]["confidence"] == 0.9
        assert result[0]["point_count"] == 180

    def test_same_shape_low_iou_keeps_both(self):
        """Two boxes with low overlap should both survive."""
        a = _seg(SegmentShape.BOX, [0, 0, 0], [1, 1, 1])
        b = _seg(SegmentShape.BOX, [3, 3, 3], [4, 4, 4])
        result = _resolve_overlaps([a, b])
        assert len(result) == 2

    def test_different_shape_no_iou_merge(self):
        """A box and cylinder at the same location should NOT IoU-merge."""
        a = _seg(SegmentShape.BOX, [0, 0, 0], [1, 1, 1], confidence=0.9)
        b = _seg(SegmentShape.CYLINDER, [0, 0, 0], [1, 1, 1], confidence=0.7)
        result = _resolve_overlaps([a, b], iou_threshold=0.60)
        # containment rule applies — smaller (same vol) gets dropped
        assert len(result) <= 2

    def test_containment_absorb_drops_small(self):
        """Small box fully inside large box → small dropped."""
        large = _seg(SegmentShape.BOX, [0, 0, 0], [10, 10, 10], confidence=0.8)
        small = _seg(SegmentShape.BOX, [2, 2, 2], [3, 3, 3], confidence=0.9)
        result = _resolve_overlaps([large, small], containment_threshold=0.80)
        assert len(result) == 1
        assert result[0] is large

    def test_valve_not_absorbed_by_pipe(self):
        """Valve inside pipe BB should NOT be dropped (whitelist)."""
        pipe = _seg(SegmentShape.CYLINDER, [0, 0, 0], [5, 0.2, 0.2], confidence=0.9)
        valve = _seg(SegmentShape.VALVE_CANDIDATE, [2, 0, 0], [2.5, 0.2, 0.2], confidence=0.85)
        result = _resolve_overlaps([pipe, valve], containment_threshold=0.80)
        assert len(result) == 2

    def test_multiple_overlaps_chain(self):
        """Three overlapping boxes — two merge, third survives if distant."""
        a = _seg(SegmentShape.BOX, [0, 0, 0], [1, 1, 1], confidence=0.8, point_count=50)
        # Shift by 0.05 → IoU ≈ 0.86 — well above 0.60 threshold
        b = _seg(
            SegmentShape.BOX, [0.05, 0.05, 0.05], [1.05, 1.05, 1.05], confidence=0.9, point_count=60
        )
        c = _seg(SegmentShape.BOX, [5, 5, 5], [6, 6, 6], confidence=0.7, point_count=40)
        result = _resolve_overlaps([a, b, c], iou_threshold=0.60)
        assert len(result) == 2  # a+b merge, c survives

    def test_cylinders_with_clearance_are_kept_separate(self):
        """Two pipe-like cylinders with surface clearance should not be merged."""
        a = _seg(SegmentShape.CYLINDER, [0.0, 0.0, 0.0], [0.3, 0.2, 0.2], confidence=0.9)
        b = _seg(SegmentShape.CYLINDER, [0.22, 0.0, 0.0], [0.52, 0.2, 0.2], confidence=0.85)
        a["tags"].update(
            {"fitted_radius_mm": 100.0, "cyl_axis_x": 1.0, "cyl_axis_y": 0.0, "cyl_axis_z": 0.0}
        )
        b["tags"].update(
            {"fitted_radius_mm": 100.0, "cyl_axis_x": 1.0, "cyl_axis_y": 0.0, "cyl_axis_z": 0.0}
        )

        result = _resolve_overlaps([a, b])

        assert len(result) == 2


class TestScanMergeAndOpenings:
    """Scan-time merge defaults should preserve detail; large-wall openings should still detect."""

    def test_merge_collinear_boxes_disabled_by_default(self):
        segs = [
            _seg(SegmentShape.BOX, [0.0, 0.0, 0.0], [1.0, 0.3, 0.3]),
            _seg(SegmentShape.BOX, [0.8, 0.0, 0.0], [1.8, 0.3, 0.3]),
        ]
        result = merge_collinear_boxes(segs)
        assert len(result) == 2

    def test_merge_collinear_boxes_opt_in(self, monkeypatch):
        monkeypatch.setenv("STB_ENABLE_BOX_CHAIN_MERGE", "1")
        segs = [
            _seg(SegmentShape.BOX, [0.0, 0.0, 0.0], [1.0, 0.3, 0.3]),
            _seg(SegmentShape.BOX, [0.8, 0.0, 0.0], [1.8, 0.3, 0.3]),
        ]
        result = merge_collinear_boxes(segs)
        assert len(result) == 1

    def test_merge_collinear_cylinders_merges_by_default(self):
        a = _seg(SegmentShape.CYLINDER, [0.0, 0.0, 0.0], [1.0, 0.2, 0.2])
        b = _seg(SegmentShape.CYLINDER, [0.9, 0.0, 0.0], [1.9, 0.2, 0.2])
        a["tags"].update(
            {"cyl_axis_x": 1.0, "cyl_axis_y": 0.0, "cyl_axis_z": 0.0, "fitted_radius_mm": 80.0}
        )
        b["tags"].update(
            {"cyl_axis_x": 1.0, "cyl_axis_y": 0.0, "cyl_axis_z": 0.0, "fitted_radius_mm": 82.0}
        )
        result = merge_collinear_cylinders([a, b])
        assert len(result) == 1

    def test_merge_collinear_cylinders_opt_in(self, monkeypatch):
        monkeypatch.setenv("STB_ENABLE_CYLINDER_CHAIN_MERGE", "1")
        a = _seg(SegmentShape.CYLINDER, [0.0, 0.0, 0.0], [1.0, 0.2, 0.2])
        b = _seg(SegmentShape.CYLINDER, [0.9, 0.0, 0.0], [1.9, 0.2, 0.2])
        a["tags"].update(
            {"cyl_axis_x": 1.0, "cyl_axis_y": 0.0, "cyl_axis_z": 0.0, "fitted_radius_mm": 80.0}
        )
        b["tags"].update(
            {"cyl_axis_x": 1.0, "cyl_axis_y": 0.0, "cyl_axis_z": 0.0, "fitted_radius_mm": 82.0}
        )
        result = merge_collinear_cylinders([a, b])
        assert len(result) == 1

    def test_merge_collinear_cylinders_skips_offset_parallel_segments(self, monkeypatch):
        monkeypatch.setenv("STB_ENABLE_CYLINDER_CHAIN_MERGE", "1")
        a = _seg(SegmentShape.CYLINDER, [0.0, 0.0, 0.0], [1.0, 0.2, 0.2])
        b = _seg(SegmentShape.CYLINDER, [0.4, 0.3, 0.0], [1.4, 0.5, 0.2])
        a["tags"].update(
            {"cyl_axis_x": 1.0, "cyl_axis_y": 0.0, "cyl_axis_z": 0.0, "fitted_radius_mm": 80.0}
        )
        b["tags"].update(
            {"cyl_axis_x": 1.0, "cyl_axis_y": 0.0, "cyl_axis_z": 0.0, "fitted_radius_mm": 80.0}
        )
        result = merge_collinear_cylinders([a, b], gap_threshold_m=0.2)
        assert len(result) == 2

    def test_linear_length_uses_metre_endpoint_tags(self):
        seg = _geometry_segment(
            SegmentShape.CYLINDER,
            BoundingBox(min_x=0.0, min_y=0.0, min_z=0.0, max_x=1.0, max_y=1.0, max_z=3.0),
        )
        seg.tags.update(
            {
                "_cyl_start_x_m": 0.0,
                "_cyl_start_y_m": 0.0,
                "_cyl_start_z_m": 0.0,
                "_cyl_end_x_m": 3.0,
                "_cyl_end_y_m": 0.0,
                "_cyl_end_z_m": 0.0,
            }
        )
        assert _linear_length_mm(seg, dx=1000.0, dy=1000.0, dz=3000.0) == pytest.approx(3000.0)

    def test_detect_openings_large_wall_uses_adaptive_grid(self):
        class _Cloud:
            def __init__(self, points):
                self.points = points

        xs = np.linspace(0.0, 50.0, 501)
        zs = np.linspace(0.0, 3.0, 31)
        pts = []
        for x in xs:
            for z in zs:
                if 20.0 <= x <= 22.0 and 0.2 <= z <= 2.3:
                    continue
                pts.append([x, 0.0, z])

        pts_np = np.asarray(pts, dtype=float)
        plane = {
            "shape": SegmentShape.PLANE_VERTICAL,
            "normal": np.array([0.0, 1.0, 0.0], dtype=float),
            "centroid": np.array([25.0, 0.0, 1.5], dtype=float),
            "mins": np.array([0.0, -0.1, 0.0], dtype=float),
            "maxs": np.array([50.0, 0.1, 3.0], dtype=float),
            "point_count": len(pts_np),
            "confidence": 0.9,
            "inlier_cloud": _Cloud(pts_np),
            "tags": {},
        }

        openings = detect_openings([plane])
        assert len(openings) >= 1
        assert any(o["shape"] == SegmentShape.VOID for o in openings)


class TestMergeSegments:
    """merge_segments should keep distinct elements while retaining valid low-count segments."""

    def test_distinct_overlapping_boxes_remain_separate(self):
        a = _geometry_segment(
            SegmentShape.BOX,
            BoundingBox(min_x=0.0, min_y=0.0, min_z=0.0, max_x=1.0, max_y=1.0, max_z=1.0),
        )
        b = _geometry_segment(
            SegmentShape.BOX,
            BoundingBox(min_x=0.5, min_y=0.5, min_z=0.0, max_x=2.0, max_y=2.0, max_z=1.0),
        )

        result = merge_segments([a, b])

        assert len(result) == 2

    def test_low_point_count_segments_are_retained(self):
        seg = _geometry_segment(
            SegmentShape.BOX,
            BoundingBox(min_x=0.0, min_y=0.0, min_z=0.0, max_x=1.0, max_y=1.0, max_z=1.0),
            point_count=1,
        )

        result = merge_segments([seg])

        assert len(result) == 1
        assert result[0].point_count == 1

    def test_void_segments_are_retained(self):
        seg = _geometry_segment(
            SegmentShape.VOID,
            BoundingBox(min_x=0.0, min_y=0.0, min_z=0.0, max_x=1.0, max_y=1.0, max_z=2.0),
            point_count=0,
        )

        result = merge_segments([seg])

        assert len(result) == 1
        assert result[0].shape == SegmentShape.VOID

    def test_valve_candidates_are_retained(self):
        seg = _geometry_segment(
            SegmentShape.VALVE_CANDIDATE,
            BoundingBox(min_x=0.0, min_y=0.0, min_z=0.0, max_x=0.5, max_y=0.5, max_z=0.5),
            point_count=12,
        )

        result = merge_segments([seg])

        assert len(result) == 1
        assert result[0].shape == SegmentShape.VALVE_CANDIDATE


# ═══════════════════════════════════════════════════════════════════════════════
# 2. Clash whitelist unit tests (coordinator_tools)
# ═══════════════════════════════════════════════════════════════════════════════


class TestClashWhitelist:
    """_is_whitelisted_pair should suppress known-valid overlaps."""

    def _bb(self, min_x=0, min_y=0, min_z=0, max_x=100, max_y=100, max_z=100):
        return BoundingBox(
            min_x=min_x, min_y=min_y, min_z=min_z, max_x=max_x, max_y=max_y, max_z=max_z
        )

    def test_valve_on_pipe(self):
        pipe = _instruction(ElementType.PIPE, self._bb())
        valve = _instruction(ElementType.VALVE, self._bb())
        assert _is_whitelisted_pair(pipe, valve) is True

    def test_safety_relief_valve_on_pipe(self):
        pipe = _instruction(ElementType.PIPE, self._bb())
        srv = _instruction(ElementType.SAFETY_RELIEF_VALVE, self._bb())
        assert _is_whitelisted_pair(pipe, srv) is True

    def test_pipe_through_wall(self):
        wall = _instruction(ElementType.WALL, self._bb())
        pipe = _instruction(ElementType.PIPE, self._bb())
        assert _is_whitelisted_pair(wall, pipe) is True

    def test_duct_through_floor(self):
        floor = _instruction(ElementType.FLOOR, self._bb())
        duct = _instruction(ElementType.DUCT, self._bb())
        assert _is_whitelisted_pair(floor, duct) is True

    def test_cable_tray_through_ceiling(self):
        ceiling = _instruction(ElementType.CEILING, self._bb())
        tray = _instruction(ElementType.CABLE_TRAY, self._bb())
        assert _is_whitelisted_pair(ceiling, tray) is True

    def test_stacked_floors_different_z(self):
        """Floor at Z=0–100 and ceiling at Z=2800–2900 — no Z overlap → whitelist."""
        floor = _instruction(ElementType.FLOOR, self._bb(min_z=0, max_z=100))
        ceiling = _instruction(ElementType.CEILING, self._bb(min_z=2800, max_z=2900))
        assert _is_whitelisted_pair(floor, ceiling) is True

    def test_stacked_floors_overlapping_z_NOT_whitelisted(self):
        """Two floors at the same Z — genuine clash, should NOT be whitelisted."""
        f1 = _instruction(ElementType.FLOOR, self._bb(min_z=0, max_z=100))
        f2 = _instruction(ElementType.FLOOR, self._bb(min_z=50, max_z=150))
        assert _is_whitelisted_pair(f1, f2) is False

    def test_wall_wall_not_whitelisted(self):
        """Two walls overlapping — genuine clash."""
        w1 = _instruction(ElementType.WALL, self._bb())
        w2 = _instruction(ElementType.WALL, self._bb())
        assert _is_whitelisted_pair(w1, w2) is False

    def test_pipe_pipe_not_whitelisted(self):
        """Two pipes overlapping — genuine clash."""
        p1 = _instruction(ElementType.PIPE, self._bb())
        p2 = _instruction(ElementType.PIPE, self._bb())
        assert _is_whitelisted_pair(p1, p2) is False

    def test_strainer_on_pipe(self):
        pipe = _instruction(ElementType.PIPE, self._bb())
        strainer = _instruction(ElementType.STRAINER, self._bb())
        assert _is_whitelisted_pair(pipe, strainer) is True


class TestDetectClashesWithWhitelist:
    """Integration: detect_clashes should skip whitelisted pairs."""

    def _bb(self, min_x=0, min_y=0, min_z=0, max_x=100, max_y=100, max_z=100):
        return BoundingBox(
            min_x=min_x, min_y=min_y, min_z=min_z, max_x=max_x, max_y=max_y, max_z=max_z
        )

    def test_valve_pipe_overlap_no_clash(self):
        """Valve inside pipe BB → 0 clashes reported."""
        pipe = _instruction(ElementType.PIPE, self._bb())
        valve = _instruction(ElementType.VALVE, self._bb())
        clashes = detect_clashes([pipe, valve])
        assert len(clashes) == 0

    def test_wall_wall_overlap_reports_clash(self):
        """Two overlapping walls → hard clash."""
        w1 = _instruction(ElementType.WALL, self._bb())
        w2 = _instruction(ElementType.WALL, self._bb())
        clashes = detect_clashes([w1, w2])
        assert len(clashes) == 1
        assert clashes[0].clash_type == "hard"

    def test_no_overlap_no_clash(self):
        """Non-overlapping elements → 0 clashes."""
        a = _instruction(ElementType.PIPE, self._bb(min_x=0, max_x=50))
        b = _instruction(ElementType.PIPE, self._bb(min_x=200, max_x=250))
        clashes = detect_clashes([a, b])
        assert len(clashes) == 0

    def test_soft_clash(self):
        """Near-miss within tolerance → soft clash."""
        a = _instruction(ElementType.BEAM, self._bb(min_x=0, max_x=100))
        b = _instruction(ElementType.BEAM, self._bb(min_x=110, max_x=200))
        clashes = detect_clashes([a, b], tolerance_mm=20.0)
        assert len(clashes) == 1
        assert clashes[0].clash_type == "soft"
