"""
deviation_check.py
──────────────────
Validates that a created BIM element's dimensions match its source geometry
segment within a defined tolerance (default 10 mm).

For the PoV this runs fully in-process:
  • The "as-built" dimensions come from the ElementInstruction bounding box
    (same data sent to Revit).
  • The "as-designed" dimensions come from the originating GeometrySegment.
  • In Stage 2+, Revit query results would be diffed against the segment.

Public API
──────────
  check(instruction, segment)          → ValidationResult
  check_batch(pairs)                   → list[ValidationResult]
  deviation_report(results)            → dict  (summary stats)
"""

from __future__ import annotations

import math
from dataclasses import dataclass

from agent.models import (
    BoundingBox,
    ElementInstruction,
    GeometrySegment,
    ValidationResult,
)

# Default tolerance per ISO 19650 / BIM Level 2 survey accuracy requirement
DEFAULT_TOLERANCE_MM: float = 10.0


# ── Bounding-box dimension helpers ────────────────────────────────────────────


@dataclass(frozen=True)
class BBoxDimensions:
    """Width (X), depth (Y) and height (Z) of a bounding box in millimetres."""

    width_mm: float
    depth_mm: float
    height_mm: float

    @property
    def volume_mm3(self) -> float:
        return self.width_mm * self.depth_mm * self.height_mm

    @property
    def diagonal_mm(self) -> float:
        return math.sqrt(self.width_mm**2 + self.depth_mm**2 + self.height_mm**2)


def _dims(bb: BoundingBox) -> BBoxDimensions:
    return BBoxDimensions(
        width_mm=abs(bb.max_x - bb.min_x),
        depth_mm=abs(bb.max_y - bb.min_y),
        height_mm=abs(bb.max_z - bb.min_z),
    )


# ── Core deviation logic ──────────────────────────────────────────────────────


def _rms_deviation(a: BBoxDimensions, b: BBoxDimensions) -> float:
    """Root-mean-square deviation across W/D/H dimensions (mm)."""
    diffs = [
        (a.width_mm - b.width_mm) ** 2,
        (a.depth_mm - b.depth_mm) ** 2,
        (a.height_mm - b.height_mm) ** 2,
    ]
    return math.sqrt(sum(diffs) / len(diffs))


def _centroid_deviation(instr: ElementInstruction, seg: GeometrySegment) -> float:
    """Euclidean distance between instruction centroid and segment centroid (mm)."""
    c_i = instr.centroid
    c_s = seg.centroid
    return math.sqrt((c_i.x - c_s.x) ** 2 + (c_i.y - c_s.y) ** 2 + (c_i.z - c_s.z) ** 2)


# ── Public API ────────────────────────────────────────────────────────────────


def check(
    instruction: ElementInstruction,
    segment: GeometrySegment,
    tolerance_mm: float = DEFAULT_TOLERANCE_MM,
    element_id: str | None = None,
) -> ValidationResult:
    """
    Validate that the ElementInstruction dimensions match the source segment.

    Deviation is the larger of:
      • RMS bounding-box dimension diff (W/D/H)
      • Centroid displacement

    Both must be ≤ tolerance_mm for the result to pass.

    Parameters
    ----------
    instruction   : ElementInstruction to validate
    segment       : Source GeometrySegment from the point cloud
    tolerance_mm  : Maximum allowable deviation (default 10.0 mm)
    element_id    : Optional Revit element ID string (for traceability)
    """
    dims_instr = _dims(instruction.bounding_box)
    dims_seg = _dims(segment.bounding_box)

    rms_dev = _rms_deviation(dims_instr, dims_seg)
    centroid_dev = _centroid_deviation(instruction, segment)

    # Worst-case deviation is the dominant metric
    deviation_mm = max(rms_dev, centroid_dev)
    passed = deviation_mm <= tolerance_mm

    return ValidationResult(
        element_id=element_id or f"instr-{instruction.instruction_id[:8]}",
        segment_id=segment.segment_id,
        deviation_mm=round(deviation_mm, 2),
        passed=passed,
        tolerance_mm=tolerance_mm,
    )


def check_batch(
    pairs: list[tuple[ElementInstruction, GeometrySegment]],
    tolerance_mm: float = DEFAULT_TOLERANCE_MM,
) -> list[ValidationResult]:
    """
    Validate a list of (instruction, segment) pairs.

    Returns one ValidationResult per pair, preserving input order.
    """
    return [check(instr, seg, tolerance_mm=tolerance_mm) for instr, seg in pairs]


# ── Report summary ────────────────────────────────────────────────────────────


def deviation_report(results: list[ValidationResult]) -> dict:
    """
    Summarise a list of ValidationResults into a stats dict suitable for
    logging or report generation.

    Returns
    -------
    {
        total:          int,
        passed:         int,
        failed:         int,
        pass_rate_pct:  float,
        max_dev_mm:     float,
        mean_dev_mm:    float,
        tolerance_mm:   float,
        failed_elements: list[str],  # element_ids that failed
    }
    """
    if not results:
        return {
            "total": 0,
            "passed": 0,
            "failed": 0,
            "pass_rate_pct": 0.0,
            "max_dev_mm": 0.0,
            "mean_dev_mm": 0.0,
            "tolerance_mm": DEFAULT_TOLERANCE_MM,
            "failed_elements": [],
        }

    passed = [r for r in results if r.passed]
    failed = [r for r in results if not r.passed]
    devs = [r.deviation_mm for r in results]

    return {
        "total": len(results),
        "passed": len(passed),
        "failed": len(failed),
        "pass_rate_pct": round(len(passed) / len(results) * 100, 1),
        "max_dev_mm": round(max(devs), 2),
        "mean_dev_mm": round(sum(devs) / len(devs), 2),
        "tolerance_mm": results[0].tolerance_mm,
        "failed_elements": [r.element_id for r in failed],
    }
