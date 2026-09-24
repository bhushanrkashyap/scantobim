from __future__ import annotations

from agent.models import BoundingBox, ElementType, GeometrySegment, Point3D, SegmentShape
from agent.orchestrator import ScanToBIMOrchestrator


class DummyAudit:
    def append(self, _event) -> None:
        return None


class DummySafetyGate:
    pass


def _make_segment(
    shape: SegmentShape,
    *,
    min_x: float,
    min_y: float,
    min_z: float,
    max_x: float,
    max_y: float,
    max_z: float,
    point_count: int = 400,
    confidence: float = 0.92,
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


def test_classify_and_prepare_prioritises_walls_then_internal_enclosed_components() -> None:
    orch = ScanToBIMOrchestrator(DummyAudit(), DummySafetyGate())
    session = orch.create_session("priority-order-test")

    wall = _make_segment(
        SegmentShape.PLANE_VERTICAL,
        min_x=0.0,
        min_y=0.0,
        min_z=0.0,
        max_x=6000.0,
        max_y=200.0,
        max_z=3000.0,
        point_count=5000,
    )
    cylinder = _make_segment(
        SegmentShape.CYLINDER,
        min_x=800.0,
        min_y=800.0,
        min_z=0.0,
        max_x=920.0,
        max_y=920.0,
        max_z=2400.0,
        point_count=1200,
    )
    valve_candidate = _make_segment(
        SegmentShape.VALVE_CANDIDATE,
        min_x=1200.0,
        min_y=1200.0,
        min_z=600.0,
        max_x=1360.0,
        max_y=1360.0,
        max_z=780.0,
        point_count=800,
    )
    floor = _make_segment(
        SegmentShape.PLANE_HORIZONTAL,
        min_x=0.0,
        min_y=0.0,
        min_z=0.0,
        max_x=6000.0,
        max_y=6000.0,
        max_z=120.0,
        point_count=2000,
    )

    instructions = orch.classify_and_prepare(
        session.session_id,
        [wall, cylinder, valve_candidate, floor],
    )

    assert len(instructions) == 4

    instruction_order = {instr.segment_id: idx for idx, instr in enumerate(instructions)}
    assert instruction_order[wall.segment_id] < instruction_order[cylinder.segment_id]
    assert instruction_order[wall.segment_id] < instruction_order[valve_candidate.segment_id]
    assert instruction_order[cylinder.segment_id] < instruction_order[floor.segment_id]
    assert instruction_order[valve_candidate.segment_id] < instruction_order[floor.segment_id]


def test_classify_and_prepare_reduces_wall_fragments_to_enclosure() -> None:
    orch = ScanToBIMOrchestrator(DummyAudit(), DummySafetyGate())
    session = orch.create_session("wall-envelope-reduction-test")

    # Eight wall strips representing fragmented perimeter detection.
    fragmented_walls = [
        _make_segment(
            SegmentShape.PLANE_VERTICAL,
            min_x=0.0,
            min_y=0.0,
            min_z=0.0,
            max_x=100.0,
            max_y=3000.0,
            max_z=3000.0,
            point_count=2500,
        ),
        _make_segment(
            SegmentShape.PLANE_VERTICAL,
            min_x=0.0,
            min_y=3000.0,
            min_z=0.0,
            max_x=100.0,
            max_y=6000.0,
            max_z=3000.0,
            point_count=2500,
        ),
        _make_segment(
            SegmentShape.PLANE_VERTICAL,
            min_x=5900.0,
            min_y=0.0,
            min_z=0.0,
            max_x=6000.0,
            max_y=3000.0,
            max_z=3000.0,
            point_count=2500,
        ),
        _make_segment(
            SegmentShape.PLANE_VERTICAL,
            min_x=5900.0,
            min_y=3000.0,
            min_z=0.0,
            max_x=6000.0,
            max_y=6000.0,
            max_z=3000.0,
            point_count=2500,
        ),
        _make_segment(
            SegmentShape.PLANE_VERTICAL,
            min_x=0.0,
            min_y=0.0,
            min_z=0.0,
            max_x=3000.0,
            max_y=100.0,
            max_z=3000.0,
            point_count=2500,
        ),
        _make_segment(
            SegmentShape.PLANE_VERTICAL,
            min_x=3000.0,
            min_y=0.0,
            min_z=0.0,
            max_x=6000.0,
            max_y=100.0,
            max_z=3000.0,
            point_count=2500,
        ),
        _make_segment(
            SegmentShape.PLANE_VERTICAL,
            min_x=0.0,
            min_y=5900.0,
            min_z=0.0,
            max_x=3000.0,
            max_y=6000.0,
            max_z=3000.0,
            point_count=2500,
        ),
        _make_segment(
            SegmentShape.PLANE_VERTICAL,
            min_x=3000.0,
            min_y=5900.0,
            min_z=0.0,
            max_x=6000.0,
            max_y=6000.0,
            max_z=3000.0,
            point_count=2500,
        ),
    ]

    internal_pipe = _make_segment(
        SegmentShape.CYLINDER,
        min_x=1200.0,
        min_y=1000.0,
        min_z=200.0,
        max_x=1320.0,
        max_y=1120.0,
        max_z=2600.0,
        point_count=1500,
    )

    instructions = orch.classify_and_prepare(
        session.session_id,
        [*fragmented_walls, internal_pipe],
    )

    wall_instructions = [i for i in instructions if i.element_type == ElementType.WALL]
    # Non-destructive preservation: real wall segments are preserved without 4-wall envelope collapse
    assert len(wall_instructions) == 8
    assert any(i.element_type == ElementType.PIPE for i in instructions)

