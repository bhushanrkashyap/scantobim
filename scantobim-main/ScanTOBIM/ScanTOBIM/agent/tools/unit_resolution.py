"""Unit Resolution for Point Cloud Input — Phase 2 Spatial Foundation.

Determines the coordinate units of input point cloud files through a
combination of file metadata inspection and geometric plausibility analysis.

Supported formats: E57, LAS, LAZ, PLY, PCD

Returns a formal UnitResolution result with one of four statuses:
    VERIFIED   — explicit metadata confirms units
    INFERRED   — geometric plausibility + weak metadata agreement
    AMBIGUOUS  — conflicting or insufficient evidence; no scaling applied
    INVALID    — metadata present but contradictory or corrupt

The source file is NEVER modified.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import structlog

from agent.tools.coordinate_system import (
    UNIT_AMBIGUOUS,
    UNIT_INFERRED,
    UNIT_VERIFIED,
)

logger = structlog.get_logger()

# ── Typical building envelope ranges for plausibility checks ──────────────
# A building scan in metres typically spans 1–500 m per axis.
# In millimetres the same scan spans 1000–500000 mm.
# In feet the same scan spans ~3–1600 ft.
_METER_DIAG_MIN: float = 1.0
_METER_DIAG_MAX: float = 2000.0
_MM_DIAG_MIN: float = 1000.0
_MM_DIAG_MAX: float = 2_000_000.0
_FEET_DIAG_MIN: float = 3.0
_FEET_DIAG_MAX: float = 6500.0


@dataclass(frozen=True)
class UnitResolution:
    """Result of unit resolution analysis for one point cloud."""

    detected_unit: str  # "meters", "millimeters", "feet", "unknown"
    status: str  # UNIT_VERIFIED / UNIT_INFERRED / UNIT_AMBIGUOUS / UNIT_INVALID
    scale_to_meters: float  # Conversion factor: source_value * scale = metres
    evidence: tuple[str, ...] = ()  # Provenance trail
    confidence_score: float = 0.0  # 0.0–1.0

    def to_dict(self) -> dict:
        return {
            "detected_unit": self.detected_unit,
            "status": self.status,
            "scale_to_meters": self.scale_to_meters,
            "evidence": list(self.evidence),
            "confidence_score": round(self.confidence_score, 3),
        }


def resolve_units(
    points: np.ndarray | None = None,
    file_path: Path | None = None,
    file_format: str | None = None,
    header_meta: dict | None = None,
) -> UnitResolution:
    """Determine point cloud units from metadata and geometric analysis.

    Priority order:
        1. Explicit STB_INPUT_UNIT environment variable override → VERIFIED
        2. File metadata inspection (LAS header, E57 metadata) → VERIFIED or INFERRED
        3. Geometric plausibility (coordinate range analysis) → INFERRED
        4. If ambiguity cannot be resolved → AMBIGUOUS (no scaling)

    Args:
        points: Optional (N, 3) array of coordinates for geometric analysis.
        file_path: Optional path for format-specific metadata reading.
        file_format: Optional explicit format string ("las", "e57", "ply", "pcd").
        header_meta: Optional pre-read header metadata dict.

    Returns:
        UnitResolution with unit, status, scale, evidence, and confidence.
    """
    evidence: list[str] = []
    fmt = file_format or ""
    if file_path is not None:
        fmt = fmt or file_path.suffix.lower().lstrip(".")

    # ── 1. Explicit environment override ──────────────────────────────────
    env_unit = os.environ.get("STB_INPUT_UNIT", "").strip().lower()
    if env_unit:
        unit, scale = _parse_unit_string(env_unit)
        if unit != "unknown":
            evidence.append(f"STB_INPUT_UNIT='{env_unit}' → {unit}")
            logger.info("unit_resolved_env", unit=unit, scale=scale)
            return UnitResolution(
                detected_unit=unit,
                status=UNIT_VERIFIED,
                scale_to_meters=scale,
                evidence=tuple(evidence),
                confidence_score=1.0,
            )
        evidence.append(f"STB_INPUT_UNIT='{env_unit}' unrecognized")

    # ── 2. File metadata inspection ───────────────────────────────────────
    meta_unit, meta_scale, meta_evidence = _inspect_file_metadata(
        file_path, fmt, header_meta
    )
    evidence.extend(meta_evidence)

    # ── 3. Geometric plausibility ─────────────────────────────────────────
    geo_unit, geo_scale, geo_confidence, geo_evidence = _geometric_plausibility(
        points
    )
    evidence.extend(geo_evidence)

    # ── 4. Combine verdicts ───────────────────────────────────────────────

    # Both metadata and geometry agree
    if meta_unit != "unknown" and geo_unit != "unknown":
        if meta_unit == geo_unit:
            evidence.append(
                f"Metadata ({meta_unit}) and geometry ({geo_unit}) AGREE"
            )
            return UnitResolution(
                detected_unit=meta_unit,
                status=UNIT_VERIFIED,
                scale_to_meters=meta_scale,
                evidence=tuple(evidence),
                confidence_score=min(1.0, 0.5 + geo_confidence),
            )
        else:
            evidence.append(
                f"Metadata ({meta_unit}) and geometry ({geo_unit}) DISAGREE"
            )
            logger.warning(
                "unit_resolution_conflict",
                metadata_unit=meta_unit,
                geometry_unit=geo_unit,
            )
            return UnitResolution(
                detected_unit="unknown",
                status=UNIT_AMBIGUOUS,
                scale_to_meters=1.0,
                evidence=tuple(evidence),
                confidence_score=0.3,
            )

    # Only metadata
    if meta_unit != "unknown":
        evidence.append(f"Using metadata-only unit: {meta_unit}")
        return UnitResolution(
            detected_unit=meta_unit,
            status=UNIT_INFERRED,
            scale_to_meters=meta_scale,
            evidence=tuple(evidence),
            confidence_score=0.7,
        )

    # Only geometry
    if geo_unit != "unknown":
        evidence.append(f"Using geometry-only unit: {geo_unit}")
        return UnitResolution(
            detected_unit=geo_unit,
            status=UNIT_INFERRED,
            scale_to_meters=geo_scale,
            evidence=tuple(evidence),
            confidence_score=geo_confidence,
        )

    # Neither
    evidence.append("Insufficient evidence for unit determination")
    logger.warning("unit_resolution_ambiguous", evidence=evidence)
    return UnitResolution(
        detected_unit="unknown",
        status=UNIT_AMBIGUOUS,
        scale_to_meters=1.0,
        evidence=tuple(evidence),
        confidence_score=0.0,
    )


# ── Internal Helpers ──────────────────────────────────────────────────────────


def _parse_unit_string(s: str) -> tuple[str, float]:
    """Parse a unit string and return (canonical_name, scale_to_meters)."""
    s = s.strip().lower()
    if s in {"m", "meter", "meters", "metre", "metres"}:
        return "meters", 1.0
    if s in {"mm", "millimeter", "millimeters", "millimetre", "millimetres"}:
        return "millimeters", 0.001
    if s in {"ft", "feet", "foot"}:
        return "feet", 0.3048
    if s in {"cm", "centimeter", "centimeters"}:
        return "centimeters", 0.01
    if s in {"in", "inch", "inches"}:
        return "inches", 0.0254
    return "unknown", 1.0


def _inspect_file_metadata(
    file_path: Path | None,
    fmt: str,
    header_meta: dict | None,
) -> tuple[str, float, list[str]]:
    """Inspect file-format-specific metadata for unit information.

    Returns:
        (unit, scale_to_meters, evidence_list)
    """
    evidence: list[str] = []

    if file_path is None and header_meta is None:
        return "unknown", 1.0, ["No file path or header metadata available"]

    # Pre-supplied header metadata
    if header_meta:
        unit_hint = header_meta.get("units", header_meta.get("unit", ""))
        if unit_hint:
            u, s = _parse_unit_string(str(unit_hint))
            if u != "unknown":
                evidence.append(f"header_meta.units='{unit_hint}' → {u}")
                return u, s, evidence
            evidence.append(f"header_meta.units='{unit_hint}' unrecognized")

    # Format-specific metadata reading
    if fmt in ("las", "laz"):
        return _inspect_las_metadata(file_path, evidence)
    if fmt == "e57":
        return _inspect_e57_metadata(file_path, evidence)

    evidence.append(f"No format-specific metadata reader for '{fmt}'")
    return "unknown", 1.0, evidence


def _inspect_las_metadata(
    file_path: Path | None,
    evidence: list[str],
) -> tuple[str, float, list[str]]:
    """Read LAS/LAZ header for unit information."""
    if file_path is None or not file_path.exists():
        evidence.append("LAS file not available for header inspection")
        return "unknown", 1.0, evidence

    try:
        import laspy

        with laspy.open(str(file_path)) as reader:
            header = reader.header
            # LAS 1.4 may have CRS WKT in VLRs that declare UNIT
            for vlr in header.vlrs:
                key = getattr(vlr, "record_id", None)
                desc = getattr(vlr, "description", "")
                if key == 2112 or "WKT" in str(desc).upper():
                    try:
                        wkt_data = vlr.record_data
                        if isinstance(wkt_data, (bytes, bytearray)):
                            wkt_str = wkt_data.decode("utf-8", errors="replace")
                        else:
                            wkt_str = str(wkt_data)
                        if "UNIT" in wkt_str.upper():
                            if "metre" in wkt_str.lower() or "meter" in wkt_str.lower():
                                evidence.append(
                                    "LAS VLR WKT declares UNIT=metre"
                                )
                                return "meters", 1.0, evidence
                            if "foot" in wkt_str.lower() or "feet" in wkt_str.lower():
                                evidence.append(
                                    "LAS VLR WKT declares UNIT=foot"
                                )
                                return "feet", 0.3048, evidence
                    except (AttributeError, ValueError, KeyError):
                        logger.debug("las_vlr_parse_skipped")
                        continue

            # Check scale factors — very small scales (< 0.01) hint at metres
            scales = [header.x_scale, header.y_scale, header.z_scale]
            evidence.append(f"LAS header scales: {scales}")
            if all(0.0001 <= abs(s) <= 0.01 for s in scales if s != 0):
                evidence.append("LAS scales suggest metre-precision data")
                # Not definitive enough to VERIFY
    except ImportError:
        evidence.append("laspy not installed; cannot inspect LAS header")
    except (OSError, RuntimeError, ValueError) as exc:
        evidence.append(f"LAS header inspection error: {exc}")

    return "unknown", 1.0, evidence


def _inspect_e57_metadata(
    file_path: Path | None,
    evidence: list[str],
) -> tuple[str, float, list[str]]:
    """Read E57 metadata for unit information."""
    if file_path is None or not file_path.exists():
        evidence.append("E57 file not available for metadata inspection")
        return "unknown", 1.0, evidence

    try:
        import pye57

        e57 = pye57.E57(str(file_path))
        # E57 files store data in the unit defined by the scanner.
        # Check scan header for coordinateMetadata or cartesianBounds.
        for i in range(e57.scan_count):
            header = e57.get_header(i)
            # Some E57 writers set a 'coordinateMetadata' field
            if hasattr(header, "coordinate_metadata"):
                meta_str = str(header.coordinate_metadata)
                if meta_str:
                    evidence.append(f"E57 coordinateMetadata: {meta_str[:200]}")
            # Check cartesian bounds — building-scale in metres is typically < 500
            try:
                bounds = {
                    "x_min": header.x_minimum,
                    "x_max": header.x_maximum,
                    "y_min": header.y_minimum,
                    "y_max": header.y_maximum,
                    "z_min": header.z_minimum,
                    "z_max": header.z_maximum,
                }
                diag = np.sqrt(
                    (bounds["x_max"] - bounds["x_min"]) ** 2
                    + (bounds["y_max"] - bounds["y_min"]) ** 2
                    + (bounds["z_max"] - bounds["z_min"]) ** 2
                )
                evidence.append(f"E57 cartesian bounds diagonal: {diag:.1f}")
                if _METER_DIAG_MIN <= diag <= _METER_DIAG_MAX:
                    evidence.append("E57 bounds consistent with metres")
                    return "meters", 1.0, evidence
                if _MM_DIAG_MIN <= diag <= _MM_DIAG_MAX:
                    evidence.append("E57 bounds consistent with millimetres")
                    return "millimeters", 0.001, evidence
            except (AttributeError, TypeError):
                pass
            break  # Only check first scan
    except ImportError:
        evidence.append("pye57 not installed; cannot inspect E57 metadata")
    except (OSError, RuntimeError, ValueError) as exc:
        evidence.append(f"E57 metadata inspection error: {exc}")

    return "unknown", 1.0, evidence


def _geometric_plausibility(
    points: np.ndarray | None,
) -> tuple[str, float, float, list[str]]:
    """Analyse coordinate ranges for geometric plausibility.

    Returns:
        (unit, scale_to_meters, confidence, evidence_list)
    """
    evidence: list[str] = []

    if points is None or len(points) < 10:
        evidence.append("Insufficient points for geometric plausibility analysis")
        return "unknown", 1.0, 0.0, evidence

    pts = np.asarray(points, dtype=float)
    if pts.ndim != 2 or pts.shape[1] < 3:
        evidence.append("Points array not (N, 3)")
        return "unknown", 1.0, 0.0, evidence

    pts3 = pts[:, :3]
    mins = pts3.min(axis=0)
    maxs = pts3.max(axis=0)
    spans = maxs - mins
    diag = float(np.linalg.norm(spans))

    evidence.append(f"Coordinate spans: [{spans[0]:.1f}, {spans[1]:.1f}, {spans[2]:.1f}]")
    evidence.append(f"Bounding diagonal: {diag:.1f}")

    # Check metre range
    in_meter_range = _METER_DIAG_MIN <= diag <= _METER_DIAG_MAX
    # Check mm range
    in_mm_range = _MM_DIAG_MIN <= diag <= _MM_DIAG_MAX
    # Check feet range
    in_feet_range = _FEET_DIAG_MIN <= diag <= _FEET_DIAG_MAX

    # Additional plausibility: Z range for buildings
    z_span = float(spans[2])
    z_plausible_m = 0.5 <= z_span <= 200.0  # 0.5m to 200m tall
    z_plausible_mm = 500.0 <= z_span <= 200_000.0
    z_plausible_ft = 1.5 <= z_span <= 650.0

    # Score each hypothesis
    scores: dict[str, float] = {}

    if in_meter_range and z_plausible_m:
        scores["meters"] = 0.75
        evidence.append("Consistent with METRES (diagonal + Z range)")
    elif in_meter_range:
        scores["meters"] = 0.5
        evidence.append("Diagonal consistent with metres but Z range unusual")

    if in_mm_range and z_plausible_mm:
        scores["millimeters"] = 0.75
        evidence.append("Consistent with MILLIMETRES (diagonal + Z range)")
    elif in_mm_range:
        scores["millimeters"] = 0.5
        evidence.append("Diagonal consistent with millimetres but Z range unusual")

    if in_feet_range and z_plausible_ft:
        scores["feet"] = 0.6
        evidence.append("Consistent with FEET (diagonal + Z range)")

    if not scores:
        evidence.append("No unit hypothesis meets plausibility criteria")
        return "unknown", 1.0, 0.0, evidence

    # If only one hypothesis is plausible → use it
    if len(scores) == 1:
        unit = next(iter(scores))
        conf = scores[unit]
        _, scale = _parse_unit_string(unit)
        evidence.append(f"Single plausible unit: {unit} (confidence={conf:.2f})")
        return unit, scale, conf, evidence

    # Multiple hypotheses — check if one dominates
    best_unit = max(scores, key=scores.get)  # type: ignore[arg-type]
    best_score = scores[best_unit]
    second_best = sorted(scores.values(), reverse=True)[1] if len(scores) > 1 else 0.0
    margin = best_score - second_best

    if margin >= 0.15:
        _, scale = _parse_unit_string(best_unit)
        evidence.append(
            f"Best hypothesis: {best_unit} (score={best_score:.2f}, margin={margin:.2f})"
        )
        return best_unit, scale, best_score * 0.8, evidence

    # Ambiguous — multiple equally plausible
    evidence.append(f"Ambiguous: scores={scores}")
    return "unknown", 1.0, 0.2, evidence
