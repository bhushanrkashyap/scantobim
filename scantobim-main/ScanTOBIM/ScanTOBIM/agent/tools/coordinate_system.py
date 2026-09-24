"""Authoritative Coordinate System & Transformation for ScanTOBIM.

Phase 2 — Spatial Foundation

Unifies coordinate mappings:
  Source Scan Coordinates [source_units]
  → Canonical Normalized Coordinates [metres]
  → BIM Model Coordinates [mm]
  → Revit Internal Coordinates [feet]

Contract:
  canonical = T_forward(source)
  source    ≈ T_inverse(canonical)   within documented numerical tolerance

Ensures every BIM element retains provenance metadata:
  - source_coordinate_system
  - target_coordinate_system
  - transform_id
  - unit_status
  - orientation_confidence
  - registration_status
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Union

import numpy as np

# ── Canonical Unit Conversion Constants ──────────────────────────────────────
# These are the ONLY definitions of these constants in the codebase.
# All other modules must import from here.
MM_PER_METER: float = 1000.0
METERS_PER_MM: float = 0.001
FEET_PER_METER: float = 3.280839895013123
METERS_PER_FOOT: float = 0.3048
MM_PER_FOOT: float = 304.8
FEET_PER_MM: float = 1.0 / 304.8


# ── Unit / Orientation / Registration Status Constants ────────────────────────

UNIT_VERIFIED: str = "UNIT_VERIFIED"
UNIT_INFERRED: str = "UNIT_INFERRED"
UNIT_AMBIGUOUS: str = "UNIT_AMBIGUOUS"
UNIT_INVALID: str = "UNIT_INVALID"

ORIENTATION_VERIFIED: str = "ORIENTATION_VERIFIED"
ORIENTATION_INFERRED: str = "ORIENTATION_INFERRED"
ORIENTATION_AMBIGUOUS: str = "ORIENTATION_AMBIGUOUS"

REGISTRATION_NOT_REQUIRED: str = "REGISTRATION_NOT_REQUIRED"
REGISTRATION_REQUIRED: str = "REGISTRATION_REQUIRED"
REGISTRATION_SUCCESS: str = "REGISTRATION_SUCCESS"
REGISTRATION_FAILED: str = "REGISTRATION_FAILED"

TRANSFORM_VALIDATED: str = "TRANSFORM_VALIDATED"
TRANSFORM_INVALID: str = "TRANSFORM_INVALID"


# ── Numerical Tolerance ──────────────────────────────────────────────────────
# Documented tolerance for round-trip source → canonical → source.
ROUND_TRIP_TOLERANCE_M: float = 1e-6  # 0.001 mm — sub-millimetre precision


def _rotation_matrix_z(deg: float) -> np.ndarray:
    """Build a 3×3 rotation matrix around the Z-axis."""
    rad = math.radians(deg)
    c, s = math.cos(rad), math.sin(rad)
    return np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]], dtype=float)


@dataclass
class AuthoritativeTransform:
    """Single authoritative transformation container governing the entire pipeline.

    Phase 2 upgrade: supports full 4×4 rigid transform with SO(3) rotation,
    scale, forward/inverse, unit status, orientation status, and registration status.

    The forward transform is:
        canonical = scale * R @ (source - origin_offset_m)

    The inverse transform is:
        source = R^T @ (canonical / scale) + origin_offset_m
    """

    transform_id: str = "TRANSFORM_SCANTOBIM_V2"
    source_coordinate_system: str = "SCAN_SOURCE"
    target_coordinate_system: str = "CANONICAL_METRIC"

    # ── Spatial parameters ────────────────────────────────────────────────
    origin_offset_m: np.ndarray = field(
        default_factory=lambda: np.zeros(3, dtype=float)
    )
    rotation_matrix: np.ndarray = field(
        default_factory=lambda: np.eye(3, dtype=float)
    )
    scale_source_to_canonical: float = 1.0

    # Legacy field — kept for backward compatibility, superseded by rotation_matrix
    rotation_deg_z: float = 0.0

    # ── Axis estimation ───────────────────────────────────────────────────
    up_axis: np.ndarray = field(
        default_factory=lambda: np.array([0.0, 0.0, 1.0], dtype=float)
    )
    up_axis_estimated: np.ndarray = field(
        default_factory=lambda: np.array([0.0, 0.0, 1.0], dtype=float)
    )
    horizontal_axes: np.ndarray = field(
        default_factory=lambda: np.array(
            [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]], dtype=float
        )
    )

    # ── Unit tracking ─────────────────────────────────────────────────────
    source_units: str = "unknown"
    canonical_units: str = "meters"

    # ── Confidence / status ───────────────────────────────────────────────
    unit_confidence: str = UNIT_AMBIGUOUS
    orientation_confidence: str = ORIENTATION_AMBIGUOUS
    registration_status: str = REGISTRATION_NOT_REQUIRED

    # ── Source provenance ─────────────────────────────────────────────────
    source_bounds_m: np.ndarray = field(
        default_factory=lambda: np.zeros((2, 3), dtype=float)
    )
    canonical_bounds_m: np.ndarray = field(
        default_factory=lambda: np.zeros((2, 3), dtype=float)
    )

    # ── 4×4 Transform Matrices ────────────────────────────────────────────

    def source_to_canonical_matrix(self) -> np.ndarray:
        """Build the forward 4×4 transform: canonical = scale * R @ (source - offset).

        Returns:
            (4, 4) float64 matrix.
        """
        R = self._effective_rotation()
        s = self.scale_source_to_canonical
        T = np.eye(4, dtype=float)
        T[0:3, 0:3] = s * R
        T[0:3, 3] = -s * (R @ self.origin_offset_m)
        return T

    def canonical_to_source_matrix(self) -> np.ndarray:
        """Build the inverse 4×4 transform: source = R^T @ (canonical / scale) + offset.

        Returns:
            (4, 4) float64 matrix.
        """
        R = self._effective_rotation()
        s = self.scale_source_to_canonical
        T_inv = np.eye(4, dtype=float)
        T_inv[0:3, 0:3] = R.T / s
        T_inv[0:3, 3] = self.origin_offset_m
        return T_inv

    def transform_points(
        self,
        pts: np.ndarray,
        direction: str = "forward",
    ) -> np.ndarray:
        """Batch transform (N, 3) points.

        Args:
            pts: (N, 3) array of 3D points.
            direction: "forward" (source→canonical) or "inverse" (canonical→source).

        Returns:
            (N, 3) transformed points.
        """
        pts = np.asarray(pts, dtype=float)
        if pts.ndim == 1:
            pts = pts.reshape(1, 3)
        T = (
            self.source_to_canonical_matrix()
            if direction == "forward"
            else self.canonical_to_source_matrix()
        )
        # Apply as homogeneous: [R|t] @ [x; y; z; 1]
        ones = np.ones((len(pts), 1), dtype=float)
        pts_h = np.hstack([pts, ones])  # (N, 4)
        result_h = (T @ pts_h.T).T  # (N, 4)
        return result_h[:, :3]

    def round_trip_error(self, pts_m: np.ndarray) -> dict[str, float]:
        """Compute round-trip error: source → canonical → source.

        Args:
            pts_m: (N, 3) source coordinates.

        Returns:
            {"max_error": float, "mean_error": float, "rmse": float}
        """
        pts_m = np.asarray(pts_m, dtype=float)
        canonical = self.transform_points(pts_m, direction="forward")
        reconstructed = self.transform_points(canonical, direction="inverse")
        errors = np.linalg.norm(reconstructed - pts_m, axis=1)
        return {
            "max_error": float(np.max(errors)),
            "mean_error": float(np.mean(errors)),
            "rmse": float(np.sqrt(np.mean(errors**2))),
        }

    def validate(self) -> dict[str, Any]:
        """Validate the transform: orthogonality, determinant, finite values, consistency.

        Returns:
            {"valid": bool, "checks": dict, "errors": list[str]}
        """
        R = self._effective_rotation()
        errors: list[str] = []
        checks: dict[str, Any] = {}

        # 1. Rotation orthogonality: R^T @ R ≈ I
        orth_err = float(np.max(np.abs(R.T @ R - np.eye(3))))
        checks["orthogonality_max_error"] = orth_err
        if orth_err > 1e-6:
            errors.append(f"Rotation matrix not orthogonal: max error {orth_err:.2e}")

        # 2. Determinant: det(R) ≈ +1 (proper rotation, not reflection)
        det_r = float(np.linalg.det(R))
        checks["rotation_determinant"] = det_r
        if abs(det_r - 1.0) > 1e-6:
            errors.append(f"Rotation determinant {det_r:.6f} != 1.0")

        # 3. Finite values
        T_fwd = self.source_to_canonical_matrix()
        checks["all_finite"] = bool(np.all(np.isfinite(T_fwd)))
        if not checks["all_finite"]:
            errors.append("Transform contains non-finite values (NaN/Inf)")

        # 4. Scale positive
        checks["scale"] = self.scale_source_to_canonical
        if self.scale_source_to_canonical <= 0:
            errors.append(f"Scale factor {self.scale_source_to_canonical} must be positive")

        # 5. Up-axis normalized
        up_norm = float(np.linalg.norm(self.up_axis_estimated))
        checks["up_axis_norm"] = up_norm
        if abs(up_norm - 1.0) > 1e-4:
            errors.append(f"Up-axis not normalized: |up| = {up_norm:.4f}")

        # 6. Round-trip on synthetic test point
        test_pts = np.array(
            [[1.0, 2.0, 3.0], [-5.0, 10.0, -1.0], [100.0, 200.0, 50.0]],
            dtype=float,
        )
        rt = self.round_trip_error(test_pts)
        checks["round_trip_max_error"] = rt["max_error"]
        if rt["max_error"] > ROUND_TRIP_TOLERANCE_M:
            errors.append(
                f"Round-trip error {rt['max_error']:.2e} exceeds tolerance "
                f"{ROUND_TRIP_TOLERANCE_M:.2e}"
            )

        checks["transform_status"] = (
            TRANSFORM_VALIDATED if len(errors) == 0 else TRANSFORM_INVALID
        )

        return {"valid": len(errors) == 0, "checks": checks, "errors": errors}

    # ── Legacy methods (backward compatible) ──────────────────────────────

    def e57_to_bim_mm(
        self,
        xyz_m: Union[np.ndarray, list[float], tuple[float, float, float]],
    ) -> tuple[float, float, float]:
        """Transform coordinates from source metres to BIM millimetres."""
        arr = np.asarray(xyz_m, dtype=float) - self.origin_offset_m
        R = self._effective_rotation()
        arr = R @ arr
        arr = arr * self.scale_source_to_canonical
        return (
            round(float(arr[0] * MM_PER_METER), 1),
            round(float(arr[1] * MM_PER_METER), 1),
            round(float(arr[2] * MM_PER_METER), 1),
        )

    def bim_mm_to_e57_m(
        self,
        xyz_mm: Union[np.ndarray, list[float], tuple[float, float, float]],
    ) -> tuple[float, float, float]:
        """Transform coordinates from BIM millimetres back to source metres."""
        arr = np.asarray(xyz_mm, dtype=float) * METERS_PER_MM
        arr = arr / self.scale_source_to_canonical
        R = self._effective_rotation()
        arr = R.T @ arr
        res = arr + self.origin_offset_m
        return (
            round(float(res[0]), 4),
            round(float(res[1]), 4),
            round(float(res[2]), 4),
        )

    def bim_mm_to_revit_feet(
        self,
        xyz_mm: Union[np.ndarray, list[float], tuple[float, float, float]],
    ) -> tuple[float, float, float]:
        """Transform coordinates from BIM millimetres to Revit internal feet."""
        arr = np.asarray(xyz_mm, dtype=float)
        return (
            float(arr[0] * FEET_PER_MM),
            float(arr[1] * FEET_PER_MM),
            float(arr[2] * FEET_PER_MM),
        )

    def e57_to_revit_feet(
        self,
        xyz_m: Union[np.ndarray, list[float], tuple[float, float, float]],
    ) -> tuple[float, float, float]:
        """Transform coordinates directly from source metres to Revit internal feet."""
        bim_mm = self.e57_to_bim_mm(xyz_m)
        return self.bim_mm_to_revit_feet(bim_mm)

    def stamp_segment(self, segment: dict) -> dict:
        """Stamp segment tags with authoritative coordinate provenance."""
        tags = segment.setdefault("tags", {})
        tags["source_coordinate_system"] = self.source_coordinate_system
        tags["target_coordinate_system"] = self.target_coordinate_system
        tags["transform_id"] = self.transform_id
        tags["coordinate_units"] = "millimeters"
        tags["unit_status"] = self.unit_confidence
        tags["orientation_confidence"] = self.orientation_confidence
        tags["registration_status"] = self.registration_status
        return segment

    def to_dict(self) -> dict[str, Any]:
        """Return serialized transform specification for sidecar JSON."""
        T_fwd = self.source_to_canonical_matrix()
        T_inv = self.canonical_to_source_matrix()
        return {
            "transform_id": self.transform_id,
            "source_coordinate_system": self.source_coordinate_system,
            "target_coordinate_system": self.target_coordinate_system,
            "origin_offset_m": [round(float(v), 6) for v in self.origin_offset_m],
            "rotation_deg_z": round(self.rotation_deg_z, 4),
            "rotation_matrix_3x3": self._effective_rotation().tolist(),
            "scale_source_to_canonical": self.scale_source_to_canonical,
            "source_units": self.source_units,
            "canonical_units": self.canonical_units,
            "unit_status": self.unit_confidence,
            "up_axis": [round(float(v), 6) for v in self.up_axis_estimated],
            "horizontal_axes": self.horizontal_axes.tolist(),
            "orientation_confidence": self.orientation_confidence,
            "registration_status": self.registration_status,
            "source_bounds_m": self.source_bounds_m.tolist(),
            "canonical_bounds_m": self.canonical_bounds_m.tolist(),
            "scale_factor_to_mm": MM_PER_METER,
            "scale_factor_to_feet": FEET_PER_METER,
            "transform_matrix_4x4": T_fwd.tolist(),
            "inverse_transform_4x4": T_inv.tolist(),
        }

    # ── Internal helpers ──────────────────────────────────────────────────

    def _effective_rotation(self) -> np.ndarray:
        """Return the effective rotation matrix.

        Uses `rotation_matrix` if it has been set (is not identity);
        otherwise falls back to legacy `rotation_deg_z`.
        """
        if not np.allclose(self.rotation_matrix, np.eye(3), atol=1e-10):
            return self.rotation_matrix
        if abs(self.rotation_deg_z) > 1e-4:
            return _rotation_matrix_z(self.rotation_deg_z)
        return np.eye(3, dtype=float)


# Global default authoritative transform singleton
GLOBAL_TRANSFORM = AuthoritativeTransform()
