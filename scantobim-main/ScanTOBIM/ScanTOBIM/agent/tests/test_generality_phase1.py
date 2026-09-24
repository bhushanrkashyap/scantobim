"""Phase 1 Generality & Dataset-Independence Verification Tests.

Verifies:
1. Multi-storey floor vs ceiling classification using normal vectors and relative Z datum.
2. Opening (door vs window) classification using sill height relative to host wall/storey datum.
3. MEP equipment (pump vs HVAC, sprinkler, drainage) using relative elevation rel_z.
4. Non-beam box fallback mapping to GENERIC_MODEL rather than BEAM.
5. Removal of developer local paths across production adapters and scripts.
6. Completeness of standard architectural and structural mappings in family_map.json.
"""

from __future__ import annotations

import json
from pathlib import Path

from agent.classifier import classify_element_type
from agent.models import (
    BoundingBox,
    ElementType,
    GeometrySegment,
    Point3D,
    SegmentShape,
)


def _make_segment(
    shape: SegmentShape,
    min_xyz: tuple[float, float, float],
    max_xyz: tuple[float, float, float],
    normal: tuple[float, float, float] | None = None,
    tags: dict | None = None,
    point_count: int = 2000,
) -> GeometrySegment:
    cx = (min_xyz[0] + max_xyz[0]) / 2.0
    cy = (min_xyz[1] + max_xyz[1]) / 2.0
    cz = (min_xyz[2] + max_xyz[2]) / 2.0
    return GeometrySegment(
        zone_id="test_zone",
        shape=shape,
        normal=Point3D(x=normal[0], y=normal[1], z=normal[2]) if normal else None,
        centroid=Point3D(x=cx, y=cy, z=cz),
        bounding_box=BoundingBox(
            min_x=min_xyz[0],
            min_y=min_xyz[1],
            min_z=min_xyz[2],
            max_x=max_xyz[0],
            max_y=max_xyz[1],
            max_z=max_xyz[2],
        ),
        point_count=point_count,
        confidence=0.95,
        tags=tags or {},
    )


class TestMultiStoreyClassification:
    """Verifies classification operates relative to storey/floor datums, not absolute Z."""

    def test_upper_floor_slab_classified_as_floor(self):
        """Floor at Z=4000mm (+Z normal) must classify as FLOOR, not CEILING."""
        seg = _make_segment(
            shape=SegmentShape.PLANE_HORIZONTAL,
            min_xyz=(0, 0, 4000),
            max_xyz=(5000, 5000, 4200),
            normal=(0.0, 0.0, 1.0),
            tags={"floor_z_mm": 4000},
        )
        assert classify_element_type(seg) == ElementType.FLOOR

    def test_upper_floor_ceiling_classified_as_ceiling(self):
        """Ceiling at Z=6800mm (-Z normal) must classify as CEILING."""
        seg = _make_segment(
            shape=SegmentShape.PLANE_HORIZONTAL,
            min_xyz=(0, 0, 6800),
            max_xyz=(5000, 5000, 6850),
            normal=(0.0, 0.0, -1.0),
            tags={"floor_z_mm": 4000},
        )
        assert classify_element_type(seg) == ElementType.CEILING

    def test_basement_floor_classified_as_floor(self):
        """Basement slab at Z=-3500mm (+Z normal) must classify as FLOOR."""
        seg = _make_segment(
            shape=SegmentShape.PLANE_HORIZONTAL,
            min_xyz=(0, 0, -3500),
            max_xyz=(5000, 5000, -3300),
            normal=(0.0, 0.0, 1.0),
            tags={"floor_z_mm": -3500},
        )
        assert classify_element_type(seg) == ElementType.FLOOR

    def test_door_at_upper_storey_classified_as_door(self):
        """Door on Level 2 (min_z=4050mm, host wall base=4000mm) must classify as DOOR."""
        seg = _make_segment(
            shape=SegmentShape.VOID,
            min_xyz=(1000, 100, 4050),
            max_xyz=(1900, 300, 6150),
            tags={"host_wall_base_z_mm": 4000, "opening_sill_mm": 50},
        )
        assert classify_element_type(seg) == ElementType.DOOR

    def test_window_at_upper_storey_classified_as_window(self):
        """Window on Level 2 (min_z=4900mm, host wall base=4000mm) must classify as WINDOW."""
        seg = _make_segment(
            shape=SegmentShape.VOID,
            min_xyz=(2500, 100, 4900),
            max_xyz=(3700, 300, 6100),
            tags={"host_wall_base_z_mm": 4000, "opening_sill_mm": 900},
        )
        assert classify_element_type(seg) == ElementType.WINDOW

    def test_pump_on_upper_storey_classified_as_pump(self):
        """Pump on Level 2 (centroid_z=4300mm, floor_z_mm=4000mm -> rel_z=300mm) must be PUMP, not HVAC."""
        seg = _make_segment(
            shape=SegmentShape.BOX,
            min_xyz=(1000, 1000, 4050),
            max_xyz=(1600, 1600, 4550),
            tags={"floor_z_mm": 4000},
        )
        assert classify_element_type(seg) == ElementType.PUMP


class TestBoxClassificationGenerality:
    """Verifies that arbitrary box objects do not blindly collapse to structural BEAM."""

    def test_compact_floor_box_is_generic_model(self):
        """A compact box near the floor (rel_z < 1000) should be GENERIC_MODEL or equipment, not BEAM."""
        seg = _make_segment(
            shape=SegmentShape.BOX,
            min_xyz=(500, 500, 50),
            max_xyz=(850, 850, 400),
            tags={"floor_z_mm": 0},
        )
        assert classify_element_type(seg) != ElementType.BEAM
        assert classify_element_type(seg) == ElementType.GENERIC_MODEL

    def test_elevated_slender_horizontal_box_is_beam(self):
        """A slender horizontally-elongated box elevated near ceiling qualifies as BEAM."""
        seg = _make_segment(
            shape=SegmentShape.BOX,
            min_xyz=(0, 0, 2400),
            max_xyz=(4000, 300, 2800),
            tags={"floor_z_mm": 0},
        )
        assert classify_element_type(seg) == ElementType.BEAM


class TestCodebasePathGenerality:
    """Verifies no hardcoded developer paths exist in production files."""

    def test_no_developer_paths_in_production(self):
        repo_root = Path(__file__).resolve().parents[2]
        suspicious_paths = [
            "/Users/bhushanrkaashyap",
            r"C:\Users\RKAA6083",
            r"C:\Users\BAVA6928",
        ]
        production_files = [
            repo_root / "agent/adapters/yolo_column_adapter.py",
            repo_root / "agent/adapters/ptv3_semantic_adapter.py",
            repo_root / "run_stage2_demo.py",
            repo_root / "test_wall_detection.py",
        ]
        for fpath in production_files:
            assert fpath.exists(), f"File {fpath} not found"
            content = fpath.read_text(encoding="utf-8")
            for sp in suspicious_paths:
                assert sp not in content, f"Hardcoded path '{sp}' found in {fpath.name}"


class TestFamilyMapCompleteness:
    """Verifies that family_map.json contains standard architectural and structural categories."""

    def test_standard_categories_present(self):
        repo_root = Path(__file__).resolve().parents[2]
        fmap_path = repo_root / "revit-addin/family_map.json"
        assert fmap_path.exists()
        with open(fmap_path, "r", encoding="utf-8") as f:
            data = json.load(f)

        required_keys = ["column", "beam", "door", "window", "wall", "floor", "ceiling", "pipe", "duct", "cable_tray"]
        for key in required_keys:
            assert key in data, f"Missing architectural/structural key '{key}' in family_map.json"
            assert "category" in data[key], f"Key '{key}' missing 'category'"
            assert "family" in data[key], f"Key '{key}' missing 'family'"
