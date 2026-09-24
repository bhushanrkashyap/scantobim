from __future__ import annotations

from agent.classifier import classify_segment
from agent.models import BoundingBox, GeometrySegment, Point3D, SegmentShape
from agent.orchestrator import ScanToBIMOrchestrator
from agent.segment_merger import merge_segments
from agent.tools.scan_tools import generate_large_synthetic_segments


class DummyAudit:
    def append(self, _event) -> None:
        return None


class DummySafetyGate:
    pass


def _sample_segment(zone_id: str = "zone-001") -> GeometrySegment:
    return GeometrySegment(
        zone_id=zone_id,
        shape=SegmentShape.BOX,
        centroid=Point3D(x=1000.0, y=1000.0, z=500.0),
        bounding_box=BoundingBox(
            min_x=0.0,
            min_y=0.0,
            min_z=0.0,
            max_x=2000.0,
            max_y=2000.0,
            max_z=1000.0,
        ),
        point_count=100,
        confidence=0.9,
        source_file="synthetic",
    )


def test_large_synthetic_profile_is_full_bim_profile() -> None:
    segments = generate_large_synthetic_segments(zone_id="zone-profile")

    assert len(segments) >= 600
    assert all(segment.source_file == "synthetic" for segment in segments)
    assert sum(1 for segment in segments if segment.shape == SegmentShape.VOID) >= 58
    assert sum(1 for segment in segments if segment.shape == SegmentShape.VALVE_CANDIDATE) >= 14


def test_orchestrator_uses_large_synthetic_profile(monkeypatch) -> None:
    from agent import orchestrator as orchestrator_module

    sentinel = [_sample_segment(zone_id="zone-001")]

    def fake_large(zone_id: str):
        assert zone_id == "zone-001"
        return sentinel

    def fail_default(*args, **kwargs):
        raise AssertionError("default synthetic generator must not be used")

    monkeypatch.setattr(orchestrator_module, "generate_large_synthetic_segments", fake_large)
    monkeypatch.setattr(
        orchestrator_module,
        "generate_synthetic_segments",
        fail_default,
        raising=False,
    )

    orch = ScanToBIMOrchestrator(DummyAudit(), DummySafetyGate())
    session = orch.create_session("profile test")

    result = orch.run_segmentation(session.session_id, use_synthetic=True, zone_id="zone-001")

    assert result == sentinel
    assert orch._segments[session.session_id] == sentinel


def test_large_synthetic_profile_classifies_voids_without_errors() -> None:
    segments = merge_segments(generate_large_synthetic_segments(zone_id="zone-profile"))

    classified = []
    for segment in segments:
        if segment.shape == SegmentShape.VOID:
            element_type, discipline, safety = classify_segment(segment)
            classified.append((element_type.value, discipline.value, safety.safety_category.value))

    assert len(classified) >= 50
    assert all(
        item[0]
        in {
            "door",
            "window",
            "slab_opening",
            "hatch",
            "trench",
            "penetration_seal",
            "containment_penetration",
        }
        for item in classified
    )
