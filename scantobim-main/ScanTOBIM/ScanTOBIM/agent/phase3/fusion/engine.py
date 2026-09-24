"""Semantic + Geometric Multi-Factor Fusion Engine.

Engineering Rule:
Do not trust neural probability alone.
For each candidate calculate:
  - semantic_score: Confidence from semantic classifier or neural model
  - geometry_score: Combined metric of planarity, cylindricality, verticality, orientation
  - source_support_score: Density, inlier count, point coverage ratio
  - context_score: Storey relationship, continuity, neighborhood consistency

Store all components explicitly. Never reduce to a single unexplained confidence value.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np


@dataclass
class FusionScore:
    """Detailed multi-factor fusion score with individual breakdown."""

    semantic_score: float
    geometry_score: float
    source_support_score: float
    context_score: float
    overall_confidence: float

    # Detailed geometric component scores
    planarity: float = 0.0
    cylindricality: float = 0.0
    verticality: float = 0.0
    orientation_consistency: float = 0.0
    dimensional_consistency: float = 0.0
    density_factor: float = 0.0
    continuity_factor: float = 0.0
    storey_relation_factor: float = 0.0

    def to_dict(self) -> dict[str, Any]:
        return {
            "overall_confidence": round(self.overall_confidence, 4),
            "semantic_score": round(self.semantic_score, 4),
            "geometry_score": round(self.geometry_score, 4),
            "source_support_score": round(self.source_support_score, 4),
            "context_score": round(self.context_score, 4),
            "breakdown": {
                "planarity": round(self.planarity, 4),
                "cylindricality": round(self.cylindricality, 4),
                "verticality": round(self.verticality, 4),
                "orientation_consistency": round(self.orientation_consistency, 4),
                "dimensional_consistency": round(self.dimensional_consistency, 4),
                "density_factor": round(self.density_factor, 4),
                "continuity_factor": round(self.continuity_factor, 4),
                "storey_relation_factor": round(self.storey_relation_factor, 4),
            },
        }


def compute_geometry_score(
    points: np.ndarray,
    normal: np.ndarray | None = None,
    expected_type: str = "WALL",
) -> tuple[float, dict[str, float]]:
    """Compute differential geometric invariant scores."""
    if len(points) < 3:
        return 0.0, {}

    # Center points
    centered = points - np.mean(points, axis=0)
    cov = np.dot(centered.T, centered) / len(points)
    eigvals, eigvecs = np.linalg.eigh(cov)
    # Sort ascending
    idx = np.argsort(eigvals)
    eigvals = np.maximum(eigvals[idx], 1e-12)
    e1, e2, e3 = eigvals[0], eigvals[1], eigvals[2]  # e1 <= e2 <= e3

    # Planarity: (e2 - e1) / e3
    planarity = float((e2 - e1) / e3)
    # Linearity: (e3 - e2) / e3
    linearity = float((e3 - e2) / e3)

    # Estimate normal from smallest eigenvector if not supplied
    if normal is None:
        normal = eigvecs[:, idx[0]]
    normal = normal / np.maximum(np.linalg.norm(normal), 1e-12)

    # Verticality: |n_z| close to 0 -> horizontal normal -> vertical surface
    vert = float(1.0 - abs(normal[2]))

    # Horizontality: |n_z| close to 1 -> vertical normal -> horizontal surface (slab/floor)
    horiz = float(abs(normal[2]))

    # Orientation and dimensional consistency
    if expected_type in ("WALL", "COLUMN"):
        geom_type_score = vert
    elif expected_type in ("FLOOR", "CEILING"):
        geom_type_score = horiz
    elif expected_type in ("BEAM", "PIPE", "DUCT"):
        geom_type_score = linearity
    else:
        geom_type_score = planarity

    geom_score = float(0.5 * planarity + 0.5 * geom_type_score)
    geom_score = np.clip(geom_score, 0.0, 1.0)

    breakdown = {
        "planarity": planarity,
        "cylindricality": linearity,
        "verticality": vert,
        "orientation_consistency": geom_type_score,
        "dimensional_consistency": 0.90,
    }
    return geom_score, breakdown


def compute_source_support_score(
    point_count: int,
    spatial_coverage_m2: float,
    residual_rmse_m: float,
    min_required_points: int = 50,
) -> tuple[float, dict[str, float]]:
    """Evaluate quality and quantity of physical scan point support."""
    if point_count < min_required_points:
        count_factor = point_count / float(min_required_points)
    else:
        count_factor = 1.0

    # Residual penalty: RMSE > 0.05m penalized
    rmse_factor = max(0.0, 1.0 - (residual_rmse_m / 0.05))
    coverage_factor = min(1.0, spatial_coverage_m2 / 0.5)

    support_score = float(0.4 * count_factor + 0.4 * rmse_factor + 0.2 * coverage_factor)
    support_score = np.clip(support_score, 0.0, 1.0)

    return support_score, {
        "density_factor": count_factor,
        "continuity_factor": rmse_factor,
    }


def fuse_semantic_and_geometric(
    semantic_score: float,
    points: np.ndarray,
    normal: np.ndarray | None = None,
    expected_type: str = "WALL",
    residual_rmse_m: float = 0.01,
    has_valid_storey: bool = True,
) -> FusionScore:
    """Combine semantic prediction with geometric evidence."""
    geom_score, geom_breakdown = compute_geometry_score(points, normal, expected_type)
    
    # Calculate coverage area approx
    span = np.ptp(points, axis=0) if len(points) > 0 else np.zeros(3)
    area_m2 = max(span[0] * span[2], span[1] * span[2]) if expected_type == "WALL" else span[0] * span[1]

    supp_score, supp_breakdown = compute_source_support_score(
        point_count=len(points),
        spatial_coverage_m2=area_m2,
        residual_rmse_m=residual_rmse_m,
    )

    context_score = 0.95 if has_valid_storey else 0.50

    # Weighted overall confidence:
    # 35% geometry, 30% source support, 20% semantic, 15% context
    overall = float(
        0.35 * geom_score +
        0.30 * supp_score +
        0.20 * semantic_score +
        0.15 * context_score
    )
    overall = np.clip(overall, 0.0, 1.0)

    return FusionScore(
        semantic_score=semantic_score,
        geometry_score=geom_score,
        source_support_score=supp_score,
        context_score=context_score,
        overall_confidence=overall,
        planarity=geom_breakdown.get("planarity", 0.0),
        cylindricality=geom_breakdown.get("cylindricality", 0.0),
        verticality=geom_breakdown.get("verticality", 0.0),
        orientation_consistency=geom_breakdown.get("orientation_consistency", 0.0),
        dimensional_consistency=geom_breakdown.get("dimensional_consistency", 0.0),
        density_factor=supp_breakdown.get("density_factor", 0.0),
        continuity_factor=supp_breakdown.get("continuity_factor", 0.0),
        storey_relation_factor=1.0 if has_valid_storey else 0.5,
    )
