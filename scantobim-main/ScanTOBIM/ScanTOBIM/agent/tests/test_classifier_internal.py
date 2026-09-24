from __future__ import annotations

import pytest

from agent.bim_reconstruction_fix import _detail_parametric
from agent.classifier import classify_segment
from agent.models import BoundingBox, ElementType, GeometrySegment, Point3D, SegmentShape


def _segment(
    shape: SegmentShape,
    *,
    min_x: float,
    min_y: float,
    min_z: float,
    max_x: float,
    max_y: float,
    max_z: float,
    point_count: int = 600,
    confidence: float = 0.9,
) -> GeometrySegment:
    return GeometrySegment(
        zone_id="zone-001",
        shape=shape,
        centroid=Point3D(
            x=(min_x + max_x) / 2.0,
            y=(min_y + max_y) / 2.0,
            z=(min_z + max_z) / 2.0,
        ),
        bounding_box=BoundingBox(
            min_x=min_x,
            min_y=min_y,
            min_z=min_z,
            max_x=max_x,
            max_y=max_y,
            max_z=max_z,
        ),
        point_count=point_count,
        confidence=confidence,
    )


def test_cylinder_classifies_fire_extinguisher() -> None:
    seg = _segment(
        SegmentShape.CYLINDER,
        min_x=0.0,
        min_y=0.0,
        min_z=700.0,
        max_x=180.0,
        max_y=180.0,
        max_z=1300.0,
    )

    element_type, discipline, safety = classify_segment(seg)

    assert element_type.value == "fire_extinguisher"
    assert discipline.value == "fire_protection"
    assert safety.safety_category.value == "SC3"


def test_box_classifies_pipe_support() -> None:
    seg = _segment(
        SegmentShape.BOX,
        min_x=0.0,
        min_y=0.0,
        min_z=1800.0,
        max_x=80.0,
        max_y=120.0,
        max_z=2300.0,
    )

    element_type, discipline, safety = classify_segment(seg)

    assert element_type.value == "pipe_support"
    assert discipline.value == "mep"
    assert safety.safety_category.value == "NS"


def test_thin_long_box_without_tray_evidence_is_not_classified_as_cable_tray() -> None:
    seg = _segment(
        SegmentShape.BOX,
        min_x=0.0,
        min_y=0.0,
        min_z=4800.0,
        max_x=4000.0,
        max_y=180.0,
        max_z=5000.0,
        point_count=400,
    )

    element_type, discipline, safety = classify_segment(seg)

    assert element_type != ElementType.CABLE_TRAY
    assert discipline.value == "mep"
    assert safety.safety_category.value in {"NS", "SC3"}


def test_box_classifies_long_slim_horizontal_box_as_cable_tray() -> None:
    seg = _segment(
        SegmentShape.BOX,
        min_x=0.0,
        min_y=0.0,
        min_z=2400.0,
        max_x=4000.0,
        max_y=320.0,
        max_z=2520.0,
        point_count=400,
    )

    element_type, discipline, safety = classify_segment(seg)

    assert element_type == ElementType.CABLE_TRAY
    assert discipline.value == "mep"
    assert safety.safety_category.value == "NS"


def test_box_with_heavy_cross_section_is_not_classified_as_cable_tray() -> None:
    seg = _segment(
        SegmentShape.BOX,
        min_x=0.0,
        min_y=0.0,
        min_z=2200.0,
        max_x=4000.0,
        max_y=2000.0,
        max_z=2800.0,
        point_count=400,
    )

    element_type, discipline, safety = classify_segment(seg)

    assert element_type != ElementType.CABLE_TRAY
    assert discipline.value == "mep"
    assert safety.safety_category.value in {"NS", "SC3"}


def test_cable_tray_params_use_explicit_tags_when_present() -> None:
    seg = _segment(
        SegmentShape.BOX,
        min_x=0.0,
        min_y=0.0,
        min_z=2400.0,
        max_x=5000.0,
        max_y=1800.0,
        max_z=2600.0,
        point_count=400,
    )
    seg.tags.update({"width_mm": 450.0, "height_mm": 120.0, "tray_thickness_mm": 35.0})

    params = _detail_parametric(seg, ElementType.CABLE_TRAY)

    assert params["width_mm"] == pytest.approx(450.0)
    assert params["height_mm"] == pytest.approx(120.0)
    assert params["tray_thickness_mm"] == pytest.approx(35.0)


def test_box_classifies_slim_pipe_like_run_as_pipe() -> None:
    seg = _segment(
        SegmentShape.BOX,
        min_x=0.0,
        min_y=0.0,
        min_z=2400.0,
        max_x=4000.0,
        max_y=400.0,
        max_z=2650.0,
        point_count=400,
    )

    element_type, discipline, safety = classify_segment(seg)

    assert element_type == ElementType.PIPE
    assert discipline.value == "mep"
    assert safety.safety_category.value == "SC3"
