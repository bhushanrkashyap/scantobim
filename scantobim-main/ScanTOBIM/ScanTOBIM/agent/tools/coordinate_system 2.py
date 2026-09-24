"""Authoritative Coordinate System & Transformation for ScanTOBIM.

Unifies coordinate mappings:
  E57 Scan Coordinates [m]
  → Local Normalized Coordinates [m]
  → BIM Model Coordinates [mm]
  → Revit Internal Coordinates [feet]

Ensures every BIM element retains provenance metadata:
  - source_coordinate_system
  - target_coordinate_system
  - transform_id
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np

MM_PER_METER: float = 1000.0
METERS_PER_MM: float = 0.001
FEET_PER_METER: float = 3.280839895013123
METERS_PER_FOOT: float = 0.3048
MM_PER_FOOT: float = 304.8
FEET_PER_MM: float = 1.0 / 304.8


@dataclass
class AuthoritativeTransform:
    """Single authoritative transformation container governing the entire pipeline."""

    transform_id: str = "TRANSFORM_E57_TO_BIM_REVIT_2025_V1"
    source_coordinate_system: str = "E57_LOCAL_METRIC"
    target_coordinate_system: str = "REVIT_INTERNAL_FEET"
    origin_offset_m: np.ndarray = field(default_factory=lambda: np.zeros(3, dtype=float))
    rotation_deg_z: float = 0.0
    up_axis: np.ndarray = field(default_factory=lambda: np.array([0.0, 0.0, 1.0], dtype=float))

    def e57_to_bim_mm(
        self,
        xyz_m: Union[np.ndarray, list[float], tuple[float, float, float]],
    ) -> tuple[float, float, float]:
        """Transform coordinates from E57 metres to BIM millimetres."""
        arr = np.asarray(xyz_m, dtype=float) - self.origin_offset_m
        if abs(self.rotation_deg_z) > 1e-4:
            rad = math.radians(self.rotation_deg_z)
            cos_r, sin_r = math.cos(rad), math.sin(rad)
            x_rot = arr[0] * cos_r - arr[1] * sin_r
            y_rot = arr[0] * sin_r + arr[1] * cos_r
            arr = np.array([x_rot, y_rot, arr[2]])
        return (
            round(float(arr[0] * MM_PER_METER), 1),
            round(float(arr[1] * MM_PER_METER), 1),
            round(float(arr[2] * MM_PER_METER), 1),
        )

    def bim_mm_to_e57_m(
        self,
        xyz_mm: Union[np.ndarray, list[float], tuple[float, float, float]],
    ) -> tuple[float, float, float]:
        """Transform coordinates from BIM millimetres back to E57 metres."""
        arr = np.asarray(xyz_mm, dtype=float) * METERS_PER_MM
        if abs(self.rotation_deg_z) > 1e-4:
            rad = math.radians(-self.rotation_deg_z)
            cos_r, sin_r = math.cos(rad), math.sin(rad)
            x_rot = arr[0] * cos_r - arr[1] * sin_r
            y_rot = arr[0] * sin_r + arr[1] * cos_r
            arr = np.array([x_rot, y_rot, arr[2]])
        res = arr + self.origin_offset_m
        return (round(float(res[0]), 4), round(float(res[1]), 4), round(float(res[2]), 4))

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
        """Transform coordinates directly from E57 metres to Revit internal feet."""
        bim_mm = self.e57_to_bim_mm(xyz_m)
        return self.bim_mm_to_revit_feet(bim_mm)

    def stamp_segment(self, segment: dict) -> dict:
        """Stamp segment tags with authoritative coordinate provenance."""
        tags = segment.setdefault("tags", {})
        tags["source_coordinate_system"] = self.source_coordinate_system
        tags["target_coordinate_system"] = self.target_coordinate_system
        tags["transform_id"] = self.transform_id
        tags["coordinate_units"] = "millimeters"
        return segment

    def to_dict(self) -> dict[str, Any]:
        """Return serialized transform specification."""
        t_matrix = np.eye(4, dtype=float)
        t_matrix[0:3, 3] = -self.origin_offset_m
        return {
            "transform_id": self.transform_id,
            "source_coordinate_system": self.source_coordinate_system,
            "target_coordinate_system": self.target_coordinate_system,
            "origin_offset_m": [round(float(v), 4) for v in self.origin_offset_m],
            "rotation_deg_z": round(self.rotation_deg_z, 4),
            "scale_factor_to_mm": MM_PER_METER,
            "scale_factor_to_feet": FEET_PER_METER,
            "transform_matrix_4x4": t_matrix.tolist(),
        }


# Global default authoritative transform singleton
GLOBAL_TRANSFORM = AuthoritativeTransform()
