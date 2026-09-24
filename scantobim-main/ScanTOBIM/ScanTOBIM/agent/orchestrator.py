"""Agent orchestrator — drives the full Scan-to-BIM pipeline.

Coordinates: segmentation → coordinate alignment → clash detection →
             classification → safety gate → element dispatch → audit.
"""

from __future__ import annotations

import uuid
from collections.abc import Generator
from pathlib import Path

import numpy as np
import structlog

from agent.audit import AuditLedger
from agent.classifier import classify_segment
from agent.models import (
    ActionResult,
    AuditEvent,
    AuditEventType,
    BoundingBox,
    DeviationReport,
    ElementInstruction,
    ElementType,
    GeometrySegment,
    Point3D,
    SafetyCategory,
    SegmentShape,
    SessionState,
    SiteSession,
    StationRegistrationResult,
    SurveyControlPoint,
)
from agent.safety_gate import SafetyCategoryViolationError, SafetyGateService
from agent.segment_merger import merge_segments
from agent.tools.coordinator_tools import detect_clashes
from agent.tools.deviation_tools import compute_deviation_report, extract_ifc_elements
from agent.tools.floorplan_tools import (
    ransac_2d_lines,
    render_dxf,
    render_svg,
    slice_from_segments,
    slice_point_cloud,
)
from agent.tools.registration_tools import (
    global_rmse_mm,
    merged_pcd_to_numpy_mm,
    register_all_scans,
)
from agent.tools.scan_tools import (
    generate_large_synthetic_segments,
)
from agent.tools.scan_tools import (
    run_segmentation as _run_segmentation_from_file,
)
from agent.tools.survey_tools import (
    RigidTransform,
    apply_transform,
    compute_rigid_transform,
    scan_quality_check,
    validate_transform_quality,
)

logger = structlog.get_logger()

# Confidence threshold below which an NCR is auto-raised after element creation
_LOW_CONFIDENCE_THRESHOLD = 0.60

# Two-phase ordering:
#  1) Wall envelope creation first
#  2) Internal enclosed components next
#  3) Remaining elements last
_WALL_FIRST_TYPES: set[ElementType] = {
    ElementType.WALL,
    ElementType.BUND_WALL,
}

_WALL_ENVELOPE_MAX_PER_ZONE = 4

_INTERNAL_ENCLOSED_SHAPES: set[SegmentShape] = {
    SegmentShape.CYLINDER,
    SegmentShape.VALVE_CANDIDATE,
    SegmentShape.BOX,
    SegmentShape.VOID,
}

_SEGMENT_SHAPE_TIEBREAKER_PRIORITY: dict[SegmentShape, int] = {
    SegmentShape.PLANE_VERTICAL: 0,
    SegmentShape.VOID: 1,
    SegmentShape.CYLINDER: 2,
    SegmentShape.VALVE_CANDIDATE: 3,
    SegmentShape.BOX: 4,
    SegmentShape.PLANE_SLOPED: 5,
    SegmentShape.PLANE_HORIZONTAL: 6,
}


def _instruction_processing_key(
    element_type: ElementType,
    segment: GeometrySegment,
) -> tuple[int, int, float, int, float, float, float]:
    """Order instructions by walls first, then enclosed internals, then remaining."""
    if element_type in _WALL_FIRST_TYPES:
        phase = 0
    elif segment.shape in _INTERNAL_ENCLOSED_SHAPES:
        phase = 1
    else:
        phase = 2

    return (
        phase,
        _SEGMENT_SHAPE_TIEBREAKER_PRIORITY.get(segment.shape, 99),
        -float(segment.confidence),
        -int(segment.point_count),
        float(segment.centroid.z),
        float(segment.centroid.x),
        float(segment.centroid.y),
    )


def _make_wall_envelope_candidates(
    zone_items: list[tuple[ElementInstruction, GeometrySegment, ElementType]],
) -> list[tuple[ElementInstruction, GeometrySegment, ElementType]]:
    """Preserve all detected walls without destructive synthetic four-wall collapse.

    Buildings are not guaranteed to be four-wall Manhattan rectangles. Real detections
    (interior walls, angled walls, partial walls, and multi-room layouts) must be
    preserved intact rather than discarded and replaced by synthetic bounding walls.
    """
    return zone_items


def _reduce_wall_segmentation(
    prepared: list[tuple[ElementInstruction, GeometrySegment, ElementType]],
) -> list[tuple[ElementInstruction, GeometrySegment, ElementType]]:
    """Reduce wall fragmentation to enclosure walls and preserve internal detail."""
    by_zone: dict[str, list[tuple[ElementInstruction, GeometrySegment, ElementType]]] = {}
    for item in prepared:
        zone = item[1].zone_id
        by_zone.setdefault(zone, []).append(item)

    reduced: list[tuple[ElementInstruction, GeometrySegment, ElementType]] = []
    for zone_items in by_zone.values():
        wall_items = [item for item in zone_items if item[2] == ElementType.WALL]
        non_wall_items = [item for item in zone_items if item[2] != ElementType.WALL]
        reduced.extend(_make_wall_envelope_candidates(wall_items) if wall_items else [])
        reduced.extend(non_wall_items)

    return reduced


def _geometry_params(segment: GeometrySegment) -> dict:
    """Derive key geometry parameters from a segment's bounding box.

    These are sent to the Revit bridge so FamilyResolver can pick the
    correct family type size without the agent knowing Revit family names.
    Wall segments additionally carry orientation data (wall_axis_x/y,
    wall_start/end_x/y_mm, wall_length_mm, wall_thickness_mm) derived from
    the RANSAC face normal in detect_planes().
    """
    bb = segment.bounding_box
    w = bb.max_x - bb.min_x  # mm
    d = bb.max_y - bb.min_y  # mm
    h = bb.max_z - bb.min_z  # mm
    dims = sorted([w, d, h])
    minor, mid, major = dims
    diameter = (minor + mid) / 2.0  # avg of two cross-section dims

    params: dict = {
        "width_mm": round(w, 1),
        "depth_mm": round(d, 1),
        "height_mm": round(h, 1),
        "minor_mm": round(minor, 1),
        "major_mm": round(major, 1),
        "diameter_mm": round(diameter, 1),
        "aspect_ratio": round(major / max(diameter, 1.0), 2),
    }

    # Propagate face normal so C# can recompute wall axis if tag coords missing
    if segment.normal is not None:
        params["normal_x"] = round(segment.normal.x, 6)
        params["normal_y"] = round(segment.normal.y, 6)
        params["normal_z"] = round(segment.normal.z, 6)

    # Pass through all tags already computed by scan pipeline:
    # wall_axis_x/y, wall_start/end_x/y_mm, wall_length_mm,
    # wall_thickness_mm, stair_steps, point_density_per_m2, etc.
    params.update(segment.tags)

    return params


def _is_viable_for_instruction(segment: GeometrySegment) -> tuple[bool, str | None]:
    """Return whether a segment has enough geometry quality for BIM creation."""
    bb = segment.bounding_box
    dx = max(bb.max_x - bb.min_x, 0.0)
    dy = max(bb.max_y - bb.min_y, 0.0)
    dz = max(bb.max_z - bb.min_z, 0.0)
    minor, mid, major = sorted((dx, dy, dz))

    if major < 50.0:
        return False, "major_span_lt_50mm"

    # VOIDs can be thin and point-free by design (synthetic openings).
    if segment.shape != SegmentShape.VOID:
        if segment.point_count <= 0:
            return False, "non_void_zero_points"
        if mid < 20.0:
            return False, "mid_span_lt_20mm"
        if minor < 3.0:
            return False, "minor_span_lt_3mm"

    return True, None


class ScanToBIMOrchestrator:
    def export_segmentation_json(self, session_id: str, output_path: str) -> str:
        """
        Export segmentation results to a validated JSON file with all required metadata.
        Includes: semantic category, confidence, bounding box, RGB metadata, coordinate transforms,
        geometry metadata, and processing metadata. Validates JSON before writing.
        Segments with invalid geometry (e.g., missing normal for planes) are included with a geometry_error field.
        """
        import json

        from pydantic import ValidationError

        from agent.determinism import compute_output_hash

        session = self.sessions.get(session_id)
        if not session:
            raise ValueError(f"Session {session_id} not found")
        segments = self._segments.get(session_id, [])
        if not segments:
            raise ValueError(f"No segments found for session {session_id}")

        # Collect coordinate transform if present
        transform = self._transforms.get(session_id)
        transform_dict = transform.as_dict() if transform else None

        # Compute output hash for determinism
        output_hash = compute_output_hash(segments)

        # Provenance (if available)
        provenance = self._provenance.get(session_id, {})

        # Compose validated segment dicts
        segment_dicts = []
        for seg in segments:
            try:
                seg_dict = seg.model_dump(mode="json")
            except ValidationError as ve:
                seg_dict = {
                    "segment_id": getattr(seg, "segment_id", None),
                    "geometry_error": f"Pydantic validation failed: {ve}",
                }
            # Add geometry params for completeness
            seg_dict["geometry_params"] = _geometry_params(seg)

            # Geometry validation for required fields
            error_msgs = []
            # For planes, normal must be present
            if str(seg.shape) in {"plane_vertical", "plane_horizontal", "plane_sloped"}:
                if seg.normal is None:
                    error_msgs.append("Missing normal vector for plane segment.")
            # Add more shape-specific checks as needed
            # (e.g., bounding box validity, centroid, etc.)
            if error_msgs:
                seg_dict["geometry_error"] = "; ".join(error_msgs)
            segment_dicts.append(seg_dict)

        # Compose output JSON
        output = {
            "session_id": session_id,
            "site_name": getattr(session, "site_name", None),
            "output_hash": output_hash,
            "provenance": provenance,
            "coordinate_transform": transform_dict,
            "segments": segment_dicts,
            "generated_at_utc": getattr(provenance, "generated_at_utc", None) or None,
        }

        # Validate top-level structure (basic check)
        if not isinstance(output["segments"], list) or not output["segments"]:
            raise ValueError("Output JSON missing segments array or is empty")

        # Write to file
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(output, f, indent=2, ensure_ascii=False)

        logger.info(
            "segmentation_json_exported",
            path=output_path,
            segments=len(segment_dicts),
            output_hash=output_hash[:12],
        )
        return output_path

    def __init__(
        self,
        audit: AuditLedger,
        safety_gate: SafetyGateService,
        revit_bridge_url: str | None = None,
    ):
        self.audit = audit
        self.safety_gate = safety_gate
        self.revit_bridge_url = revit_bridge_url or "http://localhost:8766"
        self._bridge_unreachable = False

        # Per-session state
        self.sessions: dict[str, SiteSession] = {}
        self._instructions: dict[str, list[ElementInstruction]] = {}
        self._results: dict[str, list[ActionResult]] = {}
        self._segments: dict[str, list[GeometrySegment]] = {}
        self._transforms: dict[str, RigidTransform | None] = {}
        self._clashes: dict[str, list] = {}
        # Progress streaming: session_id → list of progress event strings
        self._progress: dict[str, list[str]] = {}
        # Sprint 3.5 — Scan Intelligence
        self._registration: dict[str, StationRegistrationResult] = {}
        self._merged_clouds: dict[str, np.ndarray] = {}  # (N,3) mm
        self._deviation_reports: dict[str, DeviationReport] = {}
        # P2.4 follow-up — NQA-1 determinism provenance per session
        self._provenance: dict[str, dict] = {}

    # ── Session lifecycle ──────────────────────────────────────────────────────

    def create_session(self, site_name: str) -> SiteSession:
        session = SiteSession(site_name=site_name)
        self.sessions[session.session_id] = session
        self._instructions[session.session_id] = []
        self._results[session.session_id] = []
        self._segments[session.session_id] = []
        self._transforms[session.session_id] = None
        self._clashes[session.session_id] = []
        self._progress[session.session_id] = []
        self._bridge_unreachable = False
        # Sprint 3.5
        # (no pre-init needed for _registration / _merged_clouds / _deviation_reports;
        #  they are written on demand and checked with .get())

        self.audit.append(
            AuditEvent(
                session_id=session.session_id,
                event_type=AuditEventType.SESSION_STARTED,
                actor="orchestrator",
                detail={"site_name": site_name},
            )
        )
        logger.info("session_created", session_id=session.session_id[:8], site=site_name)
        return session

    # ── Step 0 (optional): Scan quality gate ──────────────────────────────────

    def check_scan_quality(
        self,
        session_id: str,
        point_count: int,
        area_m2: float,
        noise_pts: int = 0,
        min_density: float = 100.0,
        raise_on_fail: bool = False,
    ) -> dict:
        """Pre-segmentation quality check.

        Call before run_segmentation() when working with real scan files.
        Returns the quality result dict.
        Raises ValueError if raise_on_fail=True and quality check fails.
        """
        result = scan_quality_check(
            point_count=point_count,
            area_m2=area_m2,
            noise_pts=noise_pts,
            min_density_pts_per_m2=min_density,
        )

        self.audit.append(
            AuditEvent(
                session_id=session_id,
                event_type=AuditEventType.SCAN_QUALITY_CHECKED,
                actor="orchestrator",
                detail=result.model_dump(),
            )
        )
        self._push_progress(session_id, "scan_quality", result.model_dump())

        if raise_on_fail and not result.passed:
            raise ValueError(f"Scan quality check failed: {'; '.join(result.errors)}")
        return result.model_dump()

    # ── Step 1: Segmentation ───────────────────────────────────────────────────

    def run_segmentation(
        self,
        session_id: str,
        file_path: Path | None = None,
        zone_id: str = "zone-001",
        use_synthetic: bool = False,
        preloaded_points: np.ndarray | None = None,
    ) -> list[GeometrySegment]:
        """Run segmentation on a scan file, merged cloud, or synthetic data.

        Args:
            preloaded_points: (N, 3) numpy array in mm produced by
                              register_scans().  When supplied, file_path is
                              ignored and this array is passed directly to the
                              segmentation pipeline, bypassing file I/O.
        """
        session = self.sessions[session_id]
        session.state = SessionState.RUNNING
        self._push_progress(session_id, "segmentation_start", {"zone_id": zone_id})
        try:
            # NQA-1 — compute input hash up front so provenance covers raw bytes
            input_hash = None
            input_source = None
            if file_path is not None and not use_synthetic and preloaded_points is None:
                try:
                    from agent.determinism import compute_input_hash

                    input_hash = compute_input_hash(file_path)
                    input_source = file_path.name
                except (OSError, FileNotFoundError) as exc:
                    logger.warning("input_hash_failed", file=str(file_path), error=str(exc))
            elif use_synthetic:
                input_source = "synthetic"
            elif preloaded_points is not None:
                input_source = "preloaded"

            if use_synthetic:
                logger.warning("using_synthetic_demo_data_test_fixture", session_id=session_id)
                segments = generate_large_synthetic_segments(zone_id=zone_id)
            elif preloaded_points is not None:
                segments = _run_segmentation_from_file(
                    None, zone_id=zone_id, preloaded_points=preloaded_points
                )
            else:
                if not file_path:
                    raise ValueError("file_path required when not using synthetic data")
                segments = _run_segmentation_from_file(file_path, zone_id=zone_id)

            self._segments[session_id] = segments
            session.zone_ids = list(set(s.zone_id for s in segments))
            session.total_segments = len(segments)

            # NQA-1 — record full provenance so a reviewer can reproduce this run
            self._record_provenance(
                session_id=session_id,
                segments=segments,
                input_hash=input_hash,
                input_source=input_source,
            )

            self._push_progress(
                session_id,
                "segmentation_done",
                {
                    "segments": len(segments),
                    "zones": len(session.zone_ids),
                },
            )
            logger.info(
                "segmentation_done",
                session_id=session_id[:8],
                segments=len(segments),
                zones=len(session.zone_ids),
            )
            return segments
        except Exception as exc:
            import traceback

            tb = traceback.format_exc()
            session.state = SessionState.FAILED
            self._push_progress(
                session_id, "ERROR", {"message": f"Segmentation failed: {exc}", "traceback": tb}
            )
            logger.error(
                "segmentation_failed", session_id=session_id[:8], error=str(exc), traceback=tb
            )
            raise

    # ── Step 2 (optional): Coordinate alignment ───────────────────────────────

    def align_coordinates(
        self,
        session_id: str,
        control_points: list[SurveyControlPoint],
        max_rmse_mm: float = 5.0,
    ) -> dict:
        """Align segment coordinates from scan-frame to project/world CRS.

        Call after run_segmentation() and before classify_and_prepare().
        Modifies the stored segments in-place.

        Args:
            session_id:     Active session.
            control_points: ≥3 pairs of scan + world coordinates in mm.
            max_rmse_mm:    Quality threshold — raises ValueError if exceeded
                            and raise_on_poor_fit is True (default: warn only).

        Returns:
            dict with transform matrix, rmse, and quality assessment.
        """
        segments = self._segments.get(session_id, [])
        if not segments:
            raise ValueError(
                f"No segments found for session {session_id}. Run run_segmentation() first."
            )

        transform = compute_rigid_transform(control_points)
        quality = validate_transform_quality(transform, max_rmse_mm=max_rmse_mm)
        apply_transform(segments, transform)

        self._transforms[session_id] = transform

        self.audit.append(
            AuditEvent(
                session_id=session_id,
                event_type=AuditEventType.COORDINATE_ALIGNED,
                actor="orchestrator",
                detail={
                    "n_control_points": len(control_points),
                    "rmse_mm": round(transform.rmse_mm, 3),
                    "quality_passed": quality["passed"],
                    "quality_message": quality["message"],
                },
            )
        )
        self._push_progress(
            session_id,
            "coordinate_aligned",
            {
                "rmse_mm": round(transform.rmse_mm, 3),
                "passed": quality["passed"],
            },
        )

        logger.info(
            "coordinates_aligned",
            session_id=session_id[:8],
            rmse_mm=round(transform.rmse_mm, 3),
            passed=quality["passed"],
        )
        return {
            "transform": transform.as_dict(),
            "quality": quality,
            "segments_updated": len(segments),
        }

    # ── Step 3: Classification ─────────────────────────────────────────────────

    def classify_and_prepare(
        self, session_id: str, segments: list[GeometrySegment]
    ) -> list[ElementInstruction]:
        """Classify all segments and create element instructions."""
        prepared: list[tuple[ElementInstruction, GeometrySegment, ElementType]] = []
        self._push_progress(session_id, "classification_start", {"total": len(segments)})
        segments_raw = list(segments)

        # Merge fragments into BIM-ready elements
        segments = merge_segments(segments_raw)
        dropped_unknown = 0
        dropped_degenerate = 0
        dropped_reasons: dict[str, int] = {}

        for idx, segment in enumerate(segments):
            viable, reason = _is_viable_for_instruction(segment)
            if not viable:
                dropped_degenerate += 1
                dropped_reasons[reason or "unknown"] = (
                    dropped_reasons.get(reason or "unknown", 0) + 1
                )
                continue

            element_type, discipline, safety = classify_segment(segment)

            # Skip non-BIM artifacts and unknown classifications.
            if element_type == ElementType.UNKNOWN:
                dropped_unknown += 1
                continue

            self.audit.append(
                AuditEvent(
                    session_id=session_id,
                    event_type=AuditEventType.SEGMENT_CLASSIFIED,
                    zone_id=segment.zone_id,
                    segment_id=segment.segment_id,
                    actor="classifier",
                    detail={
                        "element_type": element_type.value,
                        "discipline": discipline.value,
                        "safety_category": safety.safety_category.value,
                        "reason": safety.classification_reason,
                        "confidence": segment.confidence,
                    },
                )
            )

            instruction = ElementInstruction(
                segment_id=segment.segment_id,
                zone_id=segment.zone_id,
                element_type=element_type,
                discipline=discipline,
                safety_category=safety.safety_category,
                dsear_zone=safety.dsear_zone,
                bounding_box=segment.bounding_box,
                centroid=segment.centroid,
                parameters=_geometry_params(segment),
                # revit_family_hint / revit_type_hint left None — Revit bridge
                # resolves via family_map.json then category fallback
            )
            prepared.append((instruction, segment, element_type))

            if (idx + 1) % 10 == 0:
                self._push_progress(
                    session_id,
                    "classification_progress",
                    {"classified": idx + 1, "total": len(segments)},
                )

        prepared = _reduce_wall_segmentation(prepared)
        prepared.sort(key=lambda item: _instruction_processing_key(item[2], item[1]))
        instructions = [item[0] for item in prepared]
        self._instructions[session_id] = instructions
        self._push_progress(
            session_id,
            "classification_done",
            {
                "total": len(instructions),
                "dropped_unknown": dropped_unknown,
                "dropped_degenerate": dropped_degenerate,
                "sc1": sum(1 for i in instructions if i.safety_category == SafetyCategory.SC1),
                "sc2": sum(1 for i in instructions if i.safety_category == SafetyCategory.SC2),
                "sc3": sum(1 for i in instructions if i.safety_category == SafetyCategory.SC3),
                "ns": sum(1 for i in instructions if i.safety_category == SafetyCategory.NS),
            },
        )
        logger.info(
            "classification_done",
            session_id=session_id[:8],
            total=len(instructions),
            dropped_unknown=dropped_unknown,
            dropped_degenerate=dropped_degenerate,
            dropped_degenerate_reasons=dropped_reasons,
            sc1=sum(1 for i in instructions if i.safety_category == SafetyCategory.SC1),
            sc2=sum(1 for i in instructions if i.safety_category == SafetyCategory.SC2),
            sc3=sum(1 for i in instructions if i.safety_category == SafetyCategory.SC3),
            ns=sum(1 for i in instructions if i.safety_category == SafetyCategory.NS),
        )
        return instructions

    # ── Step 4: Clash detection ────────────────────────────────────────────────

    def run_clash_detection(self, session_id: str, tolerance_mm: float = 20.0) -> list[dict]:
        """Run BB clash detection across all prepared instructions.

        Call after classify_and_prepare() and before execute_instructions().
        Hard clashes are auto-raised as CRITICAL NCRs.
        Soft clashes are auto-raised as MAJOR NCRs.

        Returns a list of clash dicts for the API response.
        """

        instructions = self._instructions.get(session_id, [])
        clashes = detect_clashes(instructions, tolerance_mm=tolerance_mm)
        session = self.sessions[session_id]
        session.total_clashes = len(clashes)
        self._clashes[session_id] = [c.model_dump(mode="json") for c in clashes]

        for clash in clashes:
            self.audit.append(
                AuditEvent(
                    session_id=session_id,
                    event_type=AuditEventType.CLASH_DETECTED,
                    zone_id=clash.zone_id,
                    actor="clash_detector",
                    detail={
                        "clash_id": clash.clash_id,
                        "element_a": clash.element_a_id,
                        "element_b": clash.element_b_id,
                        "clash_type": clash.clash_type,
                        "clearance_mm": clash.clearance_mm,
                    },
                )
            )

        self._push_progress(
            session_id,
            "clash_detection_done",
            {
                "total": len(clashes),
                "hard": sum(1 for c in clashes if c.clash_type == "hard"),
                "soft": sum(1 for c in clashes if c.clash_type == "soft"),
            },
        )

        logger.info(
            "clash_detection_done",
            session_id=session_id[:8],
            clashes=len(clashes),
        )
        return self._clashes[session_id]

    # ── Step 5: Execute ────────────────────────────────────────────────────────

    def execute_instructions(
        self,
        session_id: str,
        ncr_service=None,
    ) -> dict:
        """Execute all instructions through safety gate → Revit dispatch.

        Args:
            session_id:  Active session.
            ncr_service: Optional NCRService — if supplied, low-confidence
                        elements automatically raise a MINOR NCR after creation.
        """
        instructions = self._instructions.get(session_id, [])
        session = self.sessions[session_id]

        created = []
        blocked = []
        pending_approval = []
        errors = []

        # Build a fast lookup: segment_id → confidence (from stored segments)
        confidence_map: dict[str, float] = {
            s.segment_id: s.confidence for s in self._segments.get(session_id, [])
        }

        self._push_progress(session_id, "execution_start", {"total": len(instructions)})

        for instruction in instructions:
            try:
                gate = self.safety_gate.check_instruction(instruction, session_id)

                if gate is not None:
                    # SC2 — needs approval
                    pending_approval.append(
                        {
                            "gate_id": gate.gate_id,
                            "segment_id": instruction.segment_id,
                            "element_type": instruction.element_type.value,
                            "safety_category": instruction.safety_category.value,
                        }
                    )
                    continue

                # Safe to create — dispatch to Revit bridge (or mock)
                result = self._dispatch_to_revit(instruction, session_id)
                if result.success:
                    created.append(result)
                    session.total_elements_created += 1

                    # Auto-raise NCR for low-confidence elements
                    confidence = confidence_map.get(instruction.segment_id, 1.0)
                    if ncr_service and confidence < _LOW_CONFIDENCE_THRESHOLD:
                        ncr = ncr_service.raise_ncr(
                            session_id=session_id,
                            element_id=result.element_id or instruction.segment_id,
                            description=(
                                f"Low classification confidence ({confidence:.0%}) for "
                                f"{instruction.element_type.value}. Manual review required."
                            ),
                            severity="MINOR",
                            raised_by="orchestrator",
                        )
                        session.total_ncrs += 1
                        logger.warning(
                            "ncr_auto_raised_low_confidence",
                            segment_id=instruction.segment_id[:8],
                            confidence=round(confidence, 2),
                            ncr_id=ncr.ncr_id[:8],
                        )
                else:
                    errors.append(result)

            except SafetyCategoryViolationError as e:
                blocked.append(
                    {
                        "segment_id": instruction.segment_id,
                        "element_type": instruction.element_type.value,
                        "reason": str(e),
                    }
                )

        if not pending_approval and not blocked:
            session.state = SessionState.COMPLETED
            self.audit.append(
                AuditEvent(
                    session_id=session_id,
                    event_type=AuditEventType.SESSION_COMPLETED,
                    actor="orchestrator",
                    detail={
                        "elements_created": len(created),
                        "blocked": len(blocked),
                        "errors": len(errors),
                    },
                )
            )

        summary = {
            "session_id": session_id,
            "created": len(created),
            "blocked": len(blocked),
            "pending_approval": len(pending_approval),
            "errors": len(errors),
            "pending_gates": pending_approval,
            "blocked_details": blocked,
            "clashes": len(self._clashes.get(session_id, [])),
        }
        self._push_progress(
            session_id,
            "execution_done",
            {k: v for k, v in summary.items() if k not in ("pending_gates", "blocked_details")},
        )
        logger.info(
            "execution_complete",
            **{k: v for k, v in summary.items() if k not in ("pending_gates", "blocked_details")},
        )
        return summary

    # ── Dispatch ───────────────────────────────────────────────────────────────

    def _dispatch_to_revit(self, instruction: ElementInstruction, session_id: str) -> ActionResult:
        """Dispatch element creation to Revit bridge. Falls back to mock in PoV."""
        import time

        import httpx

        start = time.monotonic()

        if self._bridge_unreachable:
            # Fast fallback when bridge is known to be unreachable
            element_id = f"mock-{instruction.instruction_id[:8]}"
        else:
            try:
                with httpx.Client(timeout=10.0) as client:
                    resp = client.post(
                        f"{self.revit_bridge_url}/revit-actions",
                        json=instruction.model_dump(mode="json"),
                    )
                    resp.raise_for_status()
                    data = resp.json()
                    element_id = data.get("element_id", instruction.instruction_id)
            except Exception:
                self._bridge_unreachable = True
                logger.warning(
                    "revit_bridge_unreachable_mock_active",
                    url=f"{self.revit_bridge_url}/revit-actions",
                    instruction_id=instruction.instruction_id[:8],
                    note="Element was NOT created in Revit. Start the Revit bridge to connect.",
                )
                element_id = f"mock-{instruction.instruction_id[:8]}"

        duration = (time.monotonic() - start) * 1000

        self.audit.append(
            AuditEvent(
                session_id=session_id,
                event_type=AuditEventType.ELEMENT_CREATED,
                zone_id=instruction.zone_id,
                segment_id=instruction.segment_id,
                element_id=element_id,
                actor="orchestrator",
                detail={
                    "element_type": instruction.element_type.value,
                    "discipline": instruction.discipline.value,
                    "safety_category": instruction.safety_category.value,
                    "approver_upn": instruction.approver_upn,
                },
            )
        )

        return ActionResult(
            success=True,
            element_id=element_id,
            instruction_id=instruction.instruction_id,
            duration_ms=duration,
        )

    # ── Progress streaming ─────────────────────────────────────────────────────

    def _record_provenance(
        self,
        session_id: str,
        segments: list,
        input_hash: str | None,
        input_source: str | None,
    ) -> None:
        """Compute output_hash + capture the NQA-1 reproducibility record.

        Called from run_segmentation() and from the ingest-elements path so
        every session that has segments also has a provenance record.
        """
        from datetime import datetime, timezone

        from agent.determinism import (
            ProvenanceRecord,
            compute_output_hash,
            resolve_seed,
        )
        from agent.software_identity import get_software_identity

        # Seed that was applied — pulled from any segment's tags, falling
        # back to resolve_seed() if tags were stripped (e.g. ingest path).
        applied_seed = resolve_seed()
        for s in segments:
            tag = getattr(s, "tags", {}) or {}
            if "random_seed" in tag:
                applied_seed = int(tag["random_seed"])
                break

        record = ProvenanceRecord(
            session_id=session_id,
            random_seed=applied_seed,
            software_identity_hash=get_software_identity().identity_hash,
            input_hash=input_hash,
            input_source=input_source,
            output_hash=compute_output_hash(segments),
            segment_count=len(segments),
            generated_at_utc=datetime.now(timezone.utc).isoformat(),
        )
        self._provenance[session_id] = record.to_dict()

        logger.info(
            "provenance_recorded",
            session_id=session_id[:8],
            seed=applied_seed,
            output_hash=record.output_hash[:12],
            segment_count=len(segments),
        )

    def _push_progress(self, session_id: str, stage: str, data: dict) -> None:
        """Push a progress event to the session's progress buffer."""
        import json
        from datetime import datetime, timezone

        event = json.dumps(
            {
                "stage": stage,
                "ts": datetime.now(timezone.utc).isoformat(),
                **data,
            }
        )
        self._progress.setdefault(session_id, []).append(event)

    def stream_progress(self, session_id: str) -> Generator[str, None, None]:
        """Generator that yields SSE-formatted progress events for a session.

        Drains the progress buffer on each call — events are not repeated.
        Yields an empty heartbeat if no new events are available.
        """
        events = self._progress.get(session_id, [])
        if events:
            for event in events:
                yield f"data: {event}\n\n"
            self._progress[session_id] = []
        else:
            yield 'data: {"stage":"heartbeat"}\n\n'

    # ── Sprint 3.5: Multi-scan ICP Registration ───────────────────────────────

    def register_scans(
        self,
        session_id: str,
        file_paths: list[Path],
        voxel_size: float = 0.05,
    ) -> StationRegistrationResult:
        """Register multiple scan stations into one merged point cloud via ICP.

        The merged cloud is stored in mm in self._merged_clouds[session_id].
        Call run_segmentation(session_id, preloaded_points=...) afterwards
        to run the full segmentation pipeline on the merged cloud.

        Args:
            session_id:  Must be an existing session.
            file_paths:  Ordered list of scan file paths (>=2).
            voxel_size:  ICP voxel size in metres (default 5 cm).

        Returns:
            StationRegistrationResult stored on the session.
        """
        if session_id not in self.sessions:
            raise KeyError(f"Session {session_id!r} not found")

        self._push_progress(
            session_id,
            "registration_start",
            {
                "stations": len(file_paths),
            },
        )

        merged_pcd, pair_results = register_all_scans(file_paths, voxel_size=voxel_size)
        merged_mm = merged_pcd_to_numpy_mm(merged_pcd)
        g_rmse = global_rmse_mm(pair_results)

        # Build Pydantic result
        from agent.models import StationRegistrationDetail

        stations_detail = [
            StationRegistrationDetail(
                station_index=r.station_index,
                source_file=r.source_file,
                transform_4x4=r.transform.tolist(),
                rmse_mm=r.rmse,
                inlier_ratio=r.inlier_ratio,
            )
            for r in pair_results
        ]

        result = StationRegistrationResult(
            session_id=session_id,
            station_count=len(pair_results),
            stations=stations_detail,
            global_rmse_mm=g_rmse,
            merged_point_count=len(merged_mm),
        )

        self._registration[session_id] = result
        self._merged_clouds[session_id] = merged_mm

        self.audit.append(
            AuditEvent(
                session_id=session_id,
                event_type=AuditEventType.SCANS_REGISTERED,
                actor="orchestrator",
                detail={
                    "stations": len(pair_results),
                    "global_rmse_mm": round(g_rmse, 3),
                    "merged_points": len(merged_mm),
                },
            )
        )
        self._push_progress(
            session_id,
            "registration_done",
            {
                "stations": len(pair_results),
                "global_rmse_mm": round(g_rmse, 3),
                "merged_points": len(merged_mm),
            },
        )
        logger.info(
            "scans_registered",
            session_id=session_id[:8],
            stations=len(pair_results),
            global_rmse_mm=round(g_rmse, 3),
        )
        return result

    # ── Sprint 3.5: Deviation Report ──────────────────────────────────────────

    def generate_deviation_report(
        self,
        session_id: str,
        ifc_path: Path,
        matched_threshold_mm: float = 25.0,
        shifted_threshold_mm: float = 150.0,
    ) -> DeviationReport:
        """Compare captured scan segments against a reference IFC design model.

        Args:
            session_id:           Must be an existing session with scan segments.
            ifc_path:             Path to the reference IFC file.
            matched_threshold_mm: Distance below which a match is MATCHED.
            shifted_threshold_mm: Distance above which a match is MISSING.

        Returns:
            DeviationReport stored on the session.

        Raises:
            KeyError:  If session_id not found.
            ValueError: If no scan segments exist yet.
        """
        if session_id not in self.sessions:
            raise KeyError(f"Session {session_id!r} not found")

        segments = self._segments.get(session_id, [])
        if not segments:
            raise ValueError(
                f"No scan segments found for session {session_id!r}. Run segmentation first."
            )

        self._push_progress(
            session_id,
            "deviation_report_start",
            {
                "ifc_file": ifc_path.name,
            },
        )

        ifc_elements = extract_ifc_elements(ifc_path)
        report = compute_deviation_report(
            session_id=session_id,
            ifc_elements=ifc_elements,
            scan_segments=segments,
            matched_threshold_mm=matched_threshold_mm,
            shifted_threshold_mm=shifted_threshold_mm,
        )

        self._deviation_reports[session_id] = report

        self.audit.append(
            AuditEvent(
                session_id=session_id,
                event_type=AuditEventType.DEVIATION_REPORT_GENERATED,
                actor="orchestrator",
                detail={
                    "ifc_elements": report.ifc_element_count,
                    "scan_segments": report.scan_segment_count,
                    "matched": report.matched_count,
                    "shifted": report.shifted_count,
                    "missing": report.missing_count,
                    "extra": report.extra_count,
                },
            )
        )
        self._push_progress(
            session_id,
            "deviation_report_done",
            {
                "matched": report.matched_count,
                "shifted": report.shifted_count,
                "missing": report.missing_count,
                "extra": report.extra_count,
            },
        )
        return report

    # ── Sprint 3.5: Floor Plan Slice Export ───────────────────────────────────

    def generate_floorplan(
        self,
        session_id: str,
        elevation_m: float = 1.0,
        thickness_m: float = 0.2,
        fmt: str = "svg",
    ) -> bytes:
        """Slice the point cloud / segments at elevation and export 2D floor plan.

        Prefers the merged point cloud from multi-scan registration when
        available; falls back to bounding-box approximation from segments.

        Args:
            session_id:  Must be an existing session with segments or merged cloud.
            elevation_m: Slice elevation in metres ABOVE the floor (min Z).
            thickness_m: Slice band thickness in metres.
            fmt:         "svg" or "dxf".

        Returns:
            Floor plan as bytes (SVG or DXF).

        Raises:
            KeyError:  If session_id not found.
            ValueError: If no data available to slice.
        """
        if session_id not in self.sessions:
            raise KeyError(f"Session {session_id!r} not found")

        segments = self._segments.get(session_id, [])
        merged_pts = self._merged_clouds.get(session_id)

        if not segments and merged_pts is None:
            raise ValueError(
                f"No scan data for session {session_id!r}. "
                "Run segmentation or register_scans first."
            )

        elevation_mm = elevation_m * 1000.0
        thickness_mm = thickness_m * 1000.0

        # Determine floor Z
        if merged_pts is not None and len(merged_pts) > 0:
            floor_z_mm = float(merged_pts[:, 2].min())
        elif segments:
            floor_z_mm = min(s.bounding_box.min_z for s in segments)
        else:
            floor_z_mm = 0.0

        slice_elev_mm = floor_z_mm + elevation_mm

        # Get XY points in the slice band
        if merged_pts is not None and len(merged_pts) > 0:
            pts_xy = slice_point_cloud(merged_pts, slice_elev_mm, thickness_mm)
        else:
            pts_xy = np.empty((0, 2), dtype=np.float64)

        # Fall back to segment approximation if no merged cloud or empty slice
        if pts_xy.shape[0] == 0 and segments:
            pts_xy = slice_from_segments(segments, slice_elev_mm, thickness_mm)

        lines = ransac_2d_lines(pts_xy) if pts_xy.shape[0] >= 2 else []

        if fmt == "dxf":
            data = render_dxf(lines, pts_xy, elevation_m=elevation_m)
        else:
            data = render_svg(lines, pts_xy).encode("utf-8")

        self.audit.append(
            AuditEvent(
                session_id=session_id,
                event_type=AuditEventType.FLOORPLAN_EXPORTED,
                actor="orchestrator",
                detail={
                    "format": fmt,
                    "elevation_m": elevation_m,
                    "thickness_m": thickness_m,
                    "lines_fitted": len(lines),
                    "slice_pts": pts_xy.shape[0],
                },
            )
        )
        self._push_progress(
            session_id,
            "floorplan_exported",
            {
                "format": fmt,
                "lines": len(lines),
            },
        )
        return data

    # ── Session summary ────────────────────────────────────────────────────────

    def get_session_summary(self, session_id: str) -> dict:
        """Get full session summary including audit stats."""
        session = self.sessions.get(session_id)
        if not session:
            return {"error": "session not found"}

        audit_stats = self.audit.get_stats(session_id)
        chain = self.audit.verify_chain(session_id)

        transform = self._transforms.get(session_id)

        return {
            "session": session.model_dump(mode="json"),
            "audit": audit_stats,
            "chain_integrity": chain,
            "pending_gates": [
                g.model_dump(mode="json") for g in self.safety_gate.get_pending_gates(session_id)
            ],
            "clashes": self._clashes.get(session_id, []),
            "coordinate_aligned": transform is not None,
            "transform_rmse_mm": round(transform.rmse_mm, 3) if transform else None,
        }
