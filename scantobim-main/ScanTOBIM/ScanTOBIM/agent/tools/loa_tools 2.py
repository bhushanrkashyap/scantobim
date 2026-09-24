"""USIBD Level of Accuracy (LOA) v3.1 classification.

Reference: USIBD Level of Accuracy Specification v3.1 (January 2025)
           https://usibd.org/product/loa-version-3-1/

Each BIM element carries:
  • Measured Accuracy (M-LOA): how precisely the scan captured the element
    (scanner spec + registration residual).
  • Represented Accuracy (R-LOA): how precisely the BIM element matches the
    underlying scan points (point-to-fit residual σ).
  • Effective LOA: the worse (less accurate) of M-LOA and R-LOA — this is what
    downstream consumers must use when deciding what the model can and cannot
    be trusted for.

LOA tiers (upper tolerance bound, one-sided):
  LOA10 = ±50 mm    (schematic)
  LOA20 = ±15 mm    (design intent)
  LOA30 = ±5 mm     (construction-ready)
  LOA40 = ±1 mm     (precision / fabrication)
  LOA50 = ±0.1 mm   (metrology)

USIBD v3.1 change: tolerance is expressed at 2σ (95 % confidence) rather than
peak error, so σ_tol = tolerance / 2.
"""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass
from enum import Enum

import numpy as np

# ── Tier definitions ─────────────────────────────────────────────────────────


class LOATier(str, Enum):
    LOA10 = "LOA10"  # ±50 mm
    LOA20 = "LOA20"  # ±15 mm
    LOA30 = "LOA30"  # ±5 mm
    LOA40 = "LOA40"  # ±1 mm
    LOA50 = "LOA50"  # ±0.1 mm


# (tier, upper_tolerance_mm) ordered from most accurate to least
_TIER_TOLERANCES_MM: list[tuple[LOATier, float]] = [
    (LOATier.LOA50, 0.1),
    (LOATier.LOA40, 1.0),
    (LOATier.LOA30, 5.0),
    (LOATier.LOA20, 15.0),
    (LOATier.LOA10, 50.0),
]


def tolerance_for_tier(tier: LOATier) -> float:
    """Return the 2σ upper tolerance in mm for a given LOA tier."""
    for t, tol in _TIER_TOLERANCES_MM:
        if t == tier:
            return tol
    raise ValueError(f"Unknown LOA tier: {tier}")


def classify_loa_tier(sigma_mm: float) -> LOATier:
    """Map a 1σ residual (in mm) to the smallest LOA tier that contains it.

    USIBD v3.1 declares tolerances at 2σ (95 % confidence); so a tier with
    upper tolerance T covers any σ ≤ T/2.

    Anything beyond LOA10 (σ > 25 mm → 2σ > 50 mm) is still reported as LOA10
    with a warning — callers should treat this as "below LOA" and likely raise
    an NCR.
    """
    if sigma_mm < 0:
        raise ValueError(f"sigma_mm must be non-negative, got {sigma_mm}")

    two_sigma = 2.0 * sigma_mm
    for tier, tol in _TIER_TOLERANCES_MM:  # best → worst
        if two_sigma <= tol:
            return tier
    return LOATier.LOA10  # fell off the bottom — still return LOA10


def worse_tier(a: LOATier, b: LOATier) -> LOATier:
    """Return the less accurate (looser) of two LOA tiers."""
    order = [t for t, _ in _TIER_TOLERANCES_MM]  # LOA50 → LOA10 (best→worst)
    ia = order.index(a)
    ib = order.index(b)
    # larger index = worse accuracy
    return order[max(ia, ib)]


# ── Residual computation ──────────────────────────────────────────────────────


def plane_sigma_mm(
    points_m: np.ndarray,
    plane_coeffs: tuple[float, float, float, float],
) -> float:
    """RMS of perpendicular point-to-plane distance in mm.

    plane_coeffs = (a, b, c, d) with a²+b²+c²=1 and plane: ax+by+cz+d=0.
    Points are in metres; returned σ in mm.

    Uses RMS (not std-of-absolute) so the value matches the noise σ for
    zero-mean Gaussian residuals. This is the metric USIBD v3.1 expects.
    """
    a, b, c, d = plane_coeffs
    norm = np.sqrt(a * a + b * b + c * c)
    if norm == 0:
        return float("inf")
    signed_distances_m = (points_m @ np.array([a, b, c]) + d) / norm
    return float(np.sqrt(np.mean(signed_distances_m**2)) * 1000.0)


def cylinder_sigma_mm(
    points_m: np.ndarray,
    axis_point_m: np.ndarray,
    axis_direction: np.ndarray,
    radius_m: float,
) -> float:
    """RMS of radial residual (distance_to_axis − radius) in mm."""
    axis_direction = axis_direction / np.linalg.norm(axis_direction)
    rel = points_m - axis_point_m
    # project onto axis, subtract, take norm of perpendicular component
    proj = rel @ axis_direction
    perp = rel - np.outer(proj, axis_direction)
    radii_m = np.linalg.norm(perp, axis=1)
    signed_residuals_m = radii_m - radius_m
    return float(np.sqrt(np.mean(signed_residuals_m**2)) * 1000.0)


def aabb_sigma_mm(points_m: np.ndarray) -> float:
    """RMS of distance from each point to the nearest AABB face (mm).

    Used when no explicit fit (plane/cylinder) is available. This is a rough
    proxy: it captures how well-contained the points are by the box. For a
    genuine box-shaped element this σ is small; for an irregular cluster the
    σ will be larger and the classifier will return a looser tier.
    """
    if len(points_m) == 0:
        return float("inf")

    mins = points_m.min(axis=0)
    maxs = points_m.max(axis=0)

    # distance to nearest face along each axis
    d_min = points_m - mins  # (N, 3) non-neg
    d_max = maxs - points_m  # (N, 3) non-neg
    nearest = np.minimum(d_min, d_max).min(axis=1)  # (N,) nearest face
    return float(np.sqrt(np.mean(nearest**2)) * 1000.0)


# ── Statement aggregation ─────────────────────────────────────────────────────

# Default scanner accuracy at 10 m range (mm).  Overridden by caller when the
# actual scanner model is known.
DEFAULT_SCANNER_ACCURACY_MM = 5.0


@dataclass(frozen=True)
class LOAStatement:
    """USIBD v3.1 LOA declaration attached to a single element."""

    segment_id: str
    measured_sigma_mm: float  # scanner + registration
    measured_tier: LOATier
    represented_sigma_mm: float  # BIM element vs scan points
    represented_tier: LOATier
    effective_tier: LOATier  # worse of M and R
    standard_version: str = "USIBD LOA v3.1"

    def to_dict(self) -> dict:
        return {
            "segment_id": self.segment_id,
            "measured_sigma_mm": round(self.measured_sigma_mm, 3),
            "measured_tier": self.measured_tier.value,
            "represented_sigma_mm": round(self.represented_sigma_mm, 3),
            "represented_tier": self.represented_tier.value,
            "effective_tier": self.effective_tier.value,
            "standard_version": self.standard_version,
        }


def build_loa_statement(
    segment_id: str,
    represented_sigma_mm: float,
    measured_sigma_mm: float = DEFAULT_SCANNER_ACCURACY_MM,
) -> LOAStatement:
    """Produce a full LOA statement from the two σ inputs."""
    m_tier = classify_loa_tier(measured_sigma_mm)
    r_tier = classify_loa_tier(represented_sigma_mm)
    e_tier = worse_tier(m_tier, r_tier)

    return LOAStatement(
        segment_id=segment_id,
        measured_sigma_mm=measured_sigma_mm,
        measured_tier=m_tier,
        represented_sigma_mm=represented_sigma_mm,
        represented_tier=r_tier,
        effective_tier=e_tier,
    )


# ── Session-wide aggregation ──────────────────────────────────────────────────


def summarise_tier_distribution(
    statements: Iterable[LOAStatement],
) -> dict[str, int]:
    """Count how many elements fall into each effective LOA tier.

    Useful for dashboard reporting: "34 elements at LOA30, 12 at LOA20, 2 at LOA10".
    """
    counts: dict[str, int] = {t.value: 0 for t, _ in _TIER_TOLERANCES_MM}
    for s in statements:
        counts[s.effective_tier.value] += 1
    return counts
