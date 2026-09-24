#!/usr/bin/env python3
"""Scan-to-BIM PoV Demo Runner.

Runs the full pipeline in-process without needing the API server.
Shows: segmentation → classification → safety gates → element creation → audit verification.
"""

import sys
from pathlib import Path

# Add agent to path
sys.path.insert(0, str(Path(__file__).parent))

from agent.audit import AuditLedger
from agent.models import SafetyGateDecision
from agent.orchestrator import ScanToBIMOrchestrator
from agent.safety_gate import SafetyGateService


def print_header(text: str):
    print(f"\n{'=' * 70}")
    print(f"  {text}")
    print(f"{'=' * 70}\n")


def main():
    print_header("SCAN-TO-BIM PROOF OF VALUE DEMO")
    print("Enterprise Agentic Scan-to-BIM for Critical Infrastructure\n")

    # ── Setup ──────────────────────────────────────────────────────────────
    db_path = Path("demo_audit.db")
    if db_path.exists():
        db_path.unlink()

    audit = AuditLedger(db_path=db_path, secret="demo-secret")
    safety = SafetyGateService(audit)
    orch = ScanToBIMOrchestrator(audit=audit, safety_gate=safety)

    # ── Step 1: Create Session ─────────────────────────────────────────────
    print_header("STEP 1: Create Session")
    session = orch.create_session("Demo Nuclear Facility — Unit 4")
    print(f"Session ID: {session.session_id}")
    print(f"Site: {session.site_name}")

    # ── Step 2: Segmentation ───────────────────────────────────────────────
    print_header("STEP 2: Point Cloud Segmentation (Synthetic Data)")
    segments = orch.run_segmentation(
        session.session_id,
        zone_id="zone-001",
        use_synthetic=True,
    )
    # Inject SC1 and SC2 test segments for demo
    from agent.models import BoundingBox, GeometrySegment, Point3D, SegmentShape

    sc1_segment = GeometrySegment(
        zone_id="zone-001",
        shape=SegmentShape.PLANE_VERTICAL,
        normal=Point3D(x=0, y=1, z=0),
        centroid=Point3D(x=5000, y=5000, z=2000),
        bounding_box=BoundingBox(
            min_x=0,
            min_y=4850,
            min_z=0,
            max_x=10000,
            max_y=5150,
            max_z=4000,
        ),
        point_count=8000,
        confidence=0.95,
        source_file="synthetic-sc1",
    )
    sc2_segment = GeometrySegment(
        zone_id="zone-001",
        shape=SegmentShape.PLANE_VERTICAL,
        normal=Point3D(x=0.707, y=0.707, z=0),
        centroid=Point3D(x=12000, y=8000, z=1700),
        bounding_box=BoundingBox(
            min_x=11700,
            min_y=7700,
            min_z=0,
            max_x=12300,
            max_y=8300,
            max_z=3400,
        ),
        point_count=400,
        confidence=0.65,
        source_file="synthetic-sc2",
    )
    segments.extend([sc1_segment, sc2_segment])

    print(f"Segments found: {len(segments)}")

    shapes = {}
    for s in segments:
        shapes[s.shape.value] = shapes.get(s.shape.value, 0) + 1
    for shape, count in shapes.items():
        print(f"  {shape}: {count}")

    # ── Step 3: Classification ─────────────────────────────────────────────
    print_header("STEP 3: Safety Classification")
    instructions = orch.classify_and_prepare(session.session_id, segments)

    cats = {}
    for inst in instructions:
        cats[inst.safety_category.value] = cats.get(inst.safety_category.value, 0) + 1
    for cat, count in cats.items():
        print(f"  {cat}: {count} elements")

    types = {}
    for inst in instructions:
        types[inst.element_type.value] = types.get(inst.element_type.value, 0) + 1
    print("\nElement types:")
    for t, count in types.items():
        print(f"  {t}: {count}")

    # ── Step 4: Execute with Safety Gates ──────────────────────────────────
    print_header("STEP 4: Execute Instructions (Safety Gates Active)")
    result = orch.execute_instructions(session.session_id)

    print(f"Elements created:     {result['created']}")
    print(f"Elements blocked:     {result['blocked']} (SC1 hard block)")
    print(f"Pending approval:     {result['pending_approval']} (SC2 gates)")
    print(f"Errors:               {result['errors']}")

    # ── Step 5: Approve SC2 Gates ──────────────────────────────────────────
    pending = safety.get_pending_gates(session.session_id)
    if pending:
        print_header("STEP 5: SC2 Safety Gate Approval")
        for gate in pending:
            print(
                f"  Gate {gate.gate_id[:8]}... — {gate.element_type.value} segment {gate.segment_id[:8]}..."
            )
            decision = SafetyGateDecision(
                gate_id=gate.gate_id,
                approved=True,
                approver_upn="senior.engineer@example.com",
                comments="Approved for PoV demo — structural review complete",
            )
            approved = safety.decide_gate(decision, session.session_id)
            print(
                f"    -> {'APPROVED' if approved else 'REJECTED'} by {decision.approver_upn}"
            )

            # Re-dispatch approved element
            for inst in orch._instructions[session.session_id]:
                if inst.segment_id == gate.segment_id:
                    inst.approval_signature = "approved"
                    inst.approver_upn = decision.approver_upn
                    r = orch._dispatch_to_revit(inst, session.session_id)
                    print(f"    -> Element created: {r.element_id}")

    # ── Step 6: Verify Audit Chain ─────────────────────────────────────────
    print_header("STEP 6: Audit Chain Integrity Verification")
    chain = audit.verify_chain(session.session_id)
    stats = audit.get_stats(session.session_id)

    print(f"Total audit events:   {chain['event_count']}")
    print(f"Chain integrity:      {'PASS' if chain['valid'] else 'FAIL'}")
    if chain["broken_links"]:
        print(f"  Broken links: {chain['broken_links']}")

    print("\nAudit events by type:")
    for event_type, count in sorted(stats["by_type"].items()):
        print(f"  {event_type}: {count}")

    # ── Summary ────────────────────────────────────────────────────────────
    print_header("DEMO COMPLETE")
    sess = orch.get_session_summary(session.session_id)
    s = sess["session"]
    print(f"Session:          {s['session_id'][:16]}...")
    print(f"Site:             {s['site_name']}")
    print(f"State:            {s['state']}")
    print(f"Segments:         {s['total_segments']}")
    print(f"Elements created: {s['total_elements_created']}")
    print(f"Audit events:     {stats['total_events']}")
    print(f"Chain integrity:  {'VALID' if chain['valid'] else 'BROKEN'}")
    print(f"\nAudit DB: {db_path.absolute()}")


if __name__ == "__main__":
    main()
