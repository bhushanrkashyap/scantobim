"""Scan vs. design deviation analysis for Scan-to-BIM.

Sprint 3.5 — Feature 2: Deviation Report

Compares captured scan segments (as-built) against a reference IFC model
(design intent) and classifies each element as:

    MATCHED  — within tolerance (<25 mm)
    SHIFTED  — measurable offset (25–150 mm) — possible rework required
    MISSING  — IFC element with no nearby scan segment (>150 mm)
    EXTRA    — scan segment with no corresponding IFC element

IFC coordinate note:
    IfcSIUnit METRE is the IFC standard. Extracted world-frame centroids
    are multiplied by 1000 to convert metres → mm before comparison with
    GeometrySegment centroids (which are always in mm).
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import structlog

from agent.models import (
    DeviationReport,
    DeviationStatus,
    ElementDeviationRecord,
    GeometrySegment,
)

logger = structlog.get_logger()

# ── Thresholds (mm) ───────────────────────────────────────────────────────────

DEFAULT_MATCHED_MM = 25.0
DEFAULT_SHIFTED_MM = 150.0


# ── Internal data container ───────────────────────────────────────────────────


@dataclass
class IFCElementSummary:
    """Lightweight representation of a single IFC product."""

    global_id: str
    name: str | None
    ifc_class: str
    centroid_mm: tuple[float, float, float]
    bbox_mm: tuple[float, float, float, float, float, float] | None = None


# ── Public API ────────────────────────────────────────────────────────────────


def extract_ifc_elements(ifc_path: Path) -> list[IFCElementSummary]:
    """Parse an IFC file and extract world-frame centroids for all IfcProducts.

    Uses ifcopenshell.util.placement.get_local_placement to resolve nested
    placement chains — returns the full world-frame 4×4 transform matrix.

    The translation column [:3, 3] of that matrix is the world-frame position.
    IFC METRE units are multiplied by 1000 to give mm, consistent with the
    rest of the codebase.

    Args:
        ifc_path: Path to an IFC 2x3 or IFC4 file.

    Returns:
        List of IFCElementSummary objects (may be empty if no placed products).

    Raises:
        ImportError: If ifcopenshell is not installed.
        RuntimeError: If the file cannot be opened.
    """
    try:
        import ifcopenshell
        import ifcopenshell.util.placement as ifc_placement
    except ImportError as exc:
        raise ImportError(
            "ifcopenshell is required for deviation analysis. "
            "Install with: pip install ifcopenshell"
        ) from exc

    try:
        ifc_file = ifcopenshell.open(str(ifc_path))
    except Exception as exc:
        raise RuntimeError(f"Failed to open IFC file {ifc_path.name}: {exc}") from exc

    # Determine unit scale factor (some IFC files use MILLIMETRE)
    scale = _ifc_length_scale(ifc_file)  # multiplier to get metres

    elements: list[IFCElementSummary] = []

    for product in ifc_file.by_type("IfcProduct"):
        # Skip spatial containers — they don't represent physical elements
        if product.is_a("IfcSpatialElement") or product.is_a("IfcAnnotation"):
            continue
        if product.ObjectPlacement is None:
            continue

        try:
            matrix = ifc_placement.get_local_placement(product.ObjectPlacement)
            # matrix[:3, 3] = world-frame origin in IFC length units
            pos_m = matrix[:3, 3] * scale  # → metres
            cx, cy, cz = (float(v) * 1000.0 for v in pos_m)  # → mm

            bbox_mm = _extract_bbox_mm(product, scale)

            elements.append(
                IFCElementSummary(
                    global_id=product.GlobalId,
                    name=getattr(product, "Name", None),
                    ifc_class=product.is_a(),
                    centroid_mm=(cx, cy, cz),
                    bbox_mm=bbox_mm,
                )
            )
        except Exception as exc:  # noqa: BLE001
            logger.debug(
                "ifc_element_skip",
                global_id=getattr(product, "GlobalId", "?"),
                reason=str(exc),
            )
            continue

    logger.info("ifc_elements_extracted", count=len(elements), file=ifc_path.name)
    return elements


def compute_deviation_report(
    session_id: str,
    ifc_elements: list[IFCElementSummary],
    scan_segments: list[GeometrySegment],
    matched_threshold_mm: float = DEFAULT_MATCHED_MM,
    shifted_threshold_mm: float = DEFAULT_SHIFTED_MM,
) -> DeviationReport:
    """Compare IFC design elements against captured scan segments.

    Uses a KDTree over scan segment centroids for O(log n) nearest-neighbour
    queries per IFC element.

    Args:
        session_id:           Session identifier.
        ifc_elements:         Design-intent elements from the IFC file.
        scan_segments:        As-built segments from the scan pipeline.
        matched_threshold_mm: Distance below which a match is MATCHED.
        shifted_threshold_mm: Distance above which a match is MISSING.

    Returns:
        DeviationReport with per-element records and summary counts.
    """
    from scipy.spatial import KDTree

    records: list[ElementDeviationRecord] = []
    matched_seg_ids: set[str] = set()

    if scan_segments:
        seg_centroids = np.array(
            [[s.centroid.x, s.centroid.y, s.centroid.z] for s in scan_segments],
            dtype=np.float64,
        )
        seg_tree = KDTree(seg_centroids)
    else:
        seg_centroids = None
        seg_tree = None

    # Build IFC KDTree once (used in EXTRA detection loop below)
    if ifc_elements:
        ifc_centroids = np.array([list(e.centroid_mm) for e in ifc_elements], dtype=np.float64)
        ifc_tree = KDTree(ifc_centroids)
    else:
        ifc_centroids = None
        ifc_tree = None

    # ── Per IFC element ───────────────────────────────────────────────────────
    for elem in ifc_elements:
        ifc_pt = np.array(elem.centroid_mm, dtype=np.float64)

        if seg_tree is None:
            status = DeviationStatus.MISSING
            dist_mm = None
            matched_seg = None
        else:
            nearest_dist_mm, idx = seg_tree.query(ifc_pt)
            dist_mm = float(nearest_dist_mm)

            if dist_mm < matched_threshold_mm:
                status = DeviationStatus.MATCHED
            elif dist_mm < shifted_threshold_mm:
                status = DeviationStatus.SHIFTED
            else:
                status = DeviationStatus.MISSING
                dist_mm = None  # no meaningful match
                idx = None

            matched_seg = scan_segments[idx] if idx is not None else None
            if matched_seg is not None and status != DeviationStatus.MISSING:
                matched_seg_ids.add(matched_seg.segment_id)

        scan_centroid = (
            [matched_seg.centroid.x, matched_seg.centroid.y, matched_seg.centroid.z]
            if matched_seg
            else None
        )

        records.append(
            ElementDeviationRecord(
                ifc_global_id=elem.global_id,
                ifc_class=elem.ifc_class,
                ifc_name=elem.name,
                scan_segment_id=matched_seg.segment_id if matched_seg else None,
                distance_mm=dist_mm,
                status=status,
                ifc_centroid_mm=list(elem.centroid_mm),
                scan_centroid_mm=scan_centroid,
            )
        )

    # ── EXTRA: scan segments with no IFC match ────────────────────────────────
    # ifc_tree is built once above — O(n log n) total, not O(n²)
    for seg in scan_segments:
        if seg.segment_id in matched_seg_ids:
            continue

        if ifc_tree is not None:
            # Check distance from this segment to nearest IFC element
            seg_pt = np.array(
                [seg.centroid.x, seg.centroid.y, seg.centroid.z],
                dtype=np.float64,
            )
            dist_to_ifc, _ = ifc_tree.query(seg_pt)
            if dist_to_ifc >= shifted_threshold_mm:
                records.append(
                    ElementDeviationRecord(
                        ifc_global_id=None,
                        ifc_class=None,
                        ifc_name=None,
                        scan_segment_id=seg.segment_id,
                        distance_mm=float(dist_to_ifc),
                        status=DeviationStatus.EXTRA,
                        ifc_centroid_mm=None,
                        scan_centroid_mm=[seg.centroid.x, seg.centroid.y, seg.centroid.z],
                    )
                )
        else:
            # No IFC elements at all — every scan segment is EXTRA
            records.append(
                ElementDeviationRecord(
                    ifc_global_id=None,
                    ifc_class=None,
                    ifc_name=None,
                    scan_segment_id=seg.segment_id,
                    distance_mm=None,
                    status=DeviationStatus.EXTRA,
                    ifc_centroid_mm=None,
                    scan_centroid_mm=[seg.centroid.x, seg.centroid.y, seg.centroid.z],
                )
            )

    # ── Summary counts ────────────────────────────────────────────────────────
    matched_count = sum(1 for r in records if r.status == DeviationStatus.MATCHED)
    shifted_count = sum(1 for r in records if r.status == DeviationStatus.SHIFTED)
    missing_count = sum(1 for r in records if r.status == DeviationStatus.MISSING)
    extra_count = sum(1 for r in records if r.status == DeviationStatus.EXTRA)

    logger.info(
        "deviation_report_computed",
        session_id=session_id,
        ifc_elements=len(ifc_elements),
        scan_segments=len(scan_segments),
        matched=matched_count,
        shifted=shifted_count,
        missing=missing_count,
        extra=extra_count,
    )

    return DeviationReport(
        session_id=session_id,
        ifc_element_count=len(ifc_elements),
        scan_segment_count=len(scan_segments),
        matched_count=matched_count,
        shifted_count=shifted_count,
        missing_count=missing_count,
        extra_count=extra_count,
        matched_threshold_mm=matched_threshold_mm,
        shifted_threshold_mm=shifted_threshold_mm,
        records=records,
    )


# ── Private helpers ───────────────────────────────────────────────────────────


def _ifc_length_scale(ifc_file: object) -> float:
    """Return the scale factor to convert IFC length units to metres.

    Most files use METRE (scale=1.0). Older files may use MILLIMETRE (scale=0.001).
    """
    try:
        import ifcopenshell.util.unit as ifc_unit

        prefix, unit_type, name = ifc_unit.get_project_unit(ifc_file, "LENGTHUNIT")
        if name == "MILLIMETRE" or (prefix and "MILLI" in str(prefix)):
            return 0.001
        if name == "FOOT":
            return 0.3048
        if name == "INCH":
            return 0.0254
        return 1.0
    except Exception:  # noqa: BLE001
        return 1.0  # default: metres


def _extract_bbox_mm(
    product: object,
    scale: float,
) -> tuple[float, float, float, float, float, float] | None:
    """Try to extract an IfcBoundingBox from an IFC product representation.

    Returns (min_x, min_y, min_z, max_x, max_y, max_z) in mm, or None.
    """
    try:
        if not hasattr(product, "Representation") or product.Representation is None:
            return None

        for rep in product.Representation.Representations:
            for item in rep.Items:
                if item.is_a("IfcBoundingBox"):
                    corner = item.Corner
                    ox = float(corner.Coordinates[0]) * scale * 1000
                    oy = float(corner.Coordinates[1]) * scale * 1000
                    oz = float(corner.Coordinates[2]) * scale * 1000
                    sx = float(item.XDim) * scale * 1000
                    sy = float(item.YDim) * scale * 1000
                    sz = float(item.ZDim) * scale * 1000
                    return (ox, oy, oz, ox + sx, oy + sy, oz + sz)
    except Exception:  # noqa: BLE001
        pass
    return None
