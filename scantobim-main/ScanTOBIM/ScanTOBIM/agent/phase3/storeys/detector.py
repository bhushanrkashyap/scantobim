"""Storey Discovery & Level Understanding Module — Phase 3.

Engineering Rules:
- Uses Phase 2 canonical coordinates.
- Automatically identifies structural building levels from:
    1. Horizontal surfaces and slab plane normals (|n_z| > 0.85)
    2. Vertical point density profiles (Z-histogram kernel density estimation)
    3. Prominent floor slabs / ceiling candidates
- Zero scene-specific hardcoding:
    - No fixed floor count
    - No fixed floor-to-floor spacing
    - No hardcoded absolute elevations
- Outputs: StoreyCandidate list with elevation, supporting points, confidence, supporting surfaces.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np
from scipy.signal import find_peaks


@dataclass
class StoreyCandidate:
    """Represents a data-derived structural building level."""

    storey_id: str
    storey_index: int
    elevation_m: float
    height_m: float
    supporting_points_count: int
    confidence: float
    supporting_surfaces: list[str] = field(default_factory=list)
    z_min_m: float = 0.0
    z_max_m: float = 0.0

    def to_dict(self) -> dict[str, Any]:
        return {
            "storey_id": self.storey_id,
            "storey_index": self.storey_index,
            "elevation_m": round(self.elevation_m, 4),
            "height_m": round(self.height_m, 4),
            "supporting_points_count": self.supporting_points_count,
            "confidence": round(self.confidence, 4),
            "supporting_surfaces": self.supporting_surfaces,
            "z_bounds_m": [round(self.z_min_m, 4), round(self.z_max_m, 4)],
        }


def detect_storeys_from_point_cloud(
    points: np.ndarray,
    normals: np.ndarray | None = None,
    bin_size_m: float = 0.05,
    min_floor_clearance_m: float = 2.0,
) -> list[StoreyCandidate]:
    """Discover storeys automatically from canonical Z-coordinates and planar normals."""
    if len(points) < 50:
        return []

    z_vals = points[:, 2]
    z_min, z_max = float(np.min(z_vals)), float(np.max(z_vals))
    total_span = z_max - z_min

    if total_span < min_floor_clearance_m:
        cand = StoreyCandidate(
            storey_id="LEVEL_00",
            storey_index=0,
            elevation_m=z_min,
            height_m=total_span,
            supporting_points_count=len(points),
            confidence=0.92,
            supporting_surfaces=["HORIZONTAL_GROUND_SLICE"],
            z_min_m=z_min,
            z_max_m=z_max,
        )
        return [cand]

    # Compute Z-histogram across canonical elevation
    bins = np.arange(z_min, z_max + bin_size_m, bin_size_m)
    hist, bin_edges = np.histogram(z_vals, bins=bins)
    bin_centers = 0.5 * (bin_edges[:-1] + bin_edges[1:])

    # If normals are provided, weight horizontal normals (|n_z| > 0.85)
    if normals is not None and len(normals) == len(points):
        horiz_mask = np.abs(normals[:, 2]) > 0.85
        if np.sum(horiz_mask) > 30:
            hist_h, _ = np.histogram(z_vals[horiz_mask], bins=bins)
            hist = hist + 3 * hist_h

    # Zero-pad histogram to detect edge/boundary slab peaks at bottom and top of scan
    padded_hist = np.pad(hist, (1, 1), mode="constant", constant_values=0)
    prominence = max(np.max(hist) * 0.25, np.mean(hist) * 1.5, 8.0)
    distance_bins = max(1, int((min_floor_clearance_m * 0.8) / bin_size_m))

    padded_peaks, _ = find_peaks(padded_hist, distance=distance_bins, prominence=prominence)
    peak_indices = [p - 1 for p in padded_peaks if 0 <= (p - 1) < len(bin_centers)]

    if len(peak_indices) == 0:
        return [
            StoreyCandidate(
                storey_id="LEVEL_00",
                storey_index=0,
                elevation_m=z_min,
                height_m=total_span,
                supporting_points_count=len(points),
                confidence=0.85,
                supporting_surfaces=["BASE_POINT_SUPPORT"],
                z_min_m=z_min,
                z_max_m=z_max,
            )
        ]

    elevations = sorted([float(bin_centers[idx]) for idx in peak_indices])

    # If first peak is well above z_min (> 1.8m), prepend z_min
    if elevations[0] - z_min > 1.8:
        elevations.insert(0, z_min)

    storeys: list[StoreyCandidate] = []
    for idx, elev in enumerate(elevations):
        if idx + 1 < len(elevations):
            h = elevations[idx + 1] - elev
            z_top = elevations[idx + 1]
        else:
            h = max(z_max - elev, 2.7)
            z_top = z_max

        mask = (z_vals >= elev - 0.1) & (z_vals <= z_top + 0.1)
        pt_count = int(np.sum(mask))

        storeys.append(
            StoreyCandidate(
                storey_id=f"LEVEL_{idx:02d}",
                storey_index=idx,
                elevation_m=elev,
                height_m=h,
                supporting_points_count=pt_count,
                confidence=0.94 if idx < len(peak_indices) else 0.88,
                supporting_surfaces=[f"HISTOGRAM_PEAK_{bin_size_m*1000:.0f}mm"],
                z_min_m=elev,
                z_max_m=z_top,
            )
        )

    return storeys
