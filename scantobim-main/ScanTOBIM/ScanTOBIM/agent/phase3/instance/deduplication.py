"""Instance Deduplication & Spatial NMS Engine.

Engineering Rules:
Two candidates representing the same physical object must not survive independently.
Evaluate:
  - spatial overlap (AABB / oriented bounding intersection over union)
  - point overlap (source index intersection / union)
  - centroid distance
  - orientation alignment
  - dimensions & semantic class
  - source support
Retain the stronger evidence-backed candidate. Record suppressed candidates.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from agent.phase3.instance.adapter import InstanceMask


@dataclass
class DeduplicationResult:
    """Record of deduplicated active instances and suppressed candidates."""

    active_instances: list[InstanceMask]
    suppressed_candidates: list[dict[str, Any]] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "active_count": len(self.active_instances),
            "suppressed_count": len(self.suppressed_candidates),
            "suppressed_details": self.suppressed_candidates,
        }


def compute_point_iou(indices_a: np.ndarray, indices_b: np.ndarray) -> float:
    """Intersection-over-Union of source point indices."""
    if len(indices_a) == 0 or len(indices_b) == 0:
        return 0.0
    s_a = set(indices_a)
    s_b = set(indices_b)
    intersection = len(s_a.intersection(s_b))
    union = len(s_a.union(s_b))
    return intersection / union if union > 0 else 0.0


def compute_bbox_overlap(
    min_a: np.ndarray, max_a: np.ndarray, min_b: np.ndarray, max_b: np.ndarray
) -> float:
    """Axis-aligned 3D bounding box IoU."""
    inter_min = np.maximum(min_a, min_b)
    inter_max = np.minimum(max_a, max_b)
    inter_dims = np.maximum(inter_max - inter_min, 0.0)
    inter_vol = float(inter_dims[0] * inter_dims[1] * inter_dims[2])

    vol_a = float(np.prod(np.maximum(max_a - min_a, 0.0)))
    vol_b = float(np.prod(np.maximum(max_b - min_b, 0.0)))
    union_vol = vol_a + vol_b - inter_vol

    return inter_vol / union_vol if union_vol > 1e-9 else 0.0


def deduplicate_instances(
    instances: list[InstanceMask],
    point_iou_thresh: float = 0.35,
    spatial_iou_thresh: float = 0.50,
    centroid_dist_thresh_m: float = 0.30,
) -> DeduplicationResult:
    """Perform Non-Maximum Suppression (NMS) and deduplication across candidate instances.

    Sorts candidates descending by point count & confidence, greedily retaining the strongest.
    """
    if not instances:
        return DeduplicationResult(active_instances=[], suppressed_candidates=[])

    sorted_instances = sorted(
        instances,
        key=lambda inst: inst.confidence * math.log(max(inst.point_count, 2)),
        reverse=True,
    )

    kept: list[InstanceMask] = []
    suppressed: list[dict[str, Any]] = []

    for cand in sorted_instances:
        is_duplicate = False
        duplicate_reason = ""
        suppressing_id = ""

        for active in kept:
            if active.semantic_class != cand.semantic_class:
                continue

            # 1. Point IoU
            p_iou = compute_point_iou(active.point_indices, cand.point_indices)
            if p_iou > point_iou_thresh:
                is_duplicate = True
                duplicate_reason = f"Point IoU {p_iou:.3f} > {point_iou_thresh}"
                suppressing_id = active.instance_id
                break

            # 2. Centroid distance
            dist = float(np.linalg.norm(active.centroid_m - cand.centroid_m))
            if dist < centroid_dist_thresh_m:
                # 3. Spatial BBox IoU
                s_iou = compute_bbox_overlap(
                    active.bounding_box_min_m, active.bounding_box_max_m,
                    cand.bounding_box_min_m, cand.bounding_box_max_m,
                )
                if s_iou > spatial_iou_thresh:
                    is_duplicate = True
                    duplicate_reason = f"Centroid dist {dist:.3f}m < {centroid_dist_thresh_m}m & BBox IoU {s_iou:.3f}"
                    suppressing_id = active.instance_id
                    break

        if is_duplicate:
            suppressed.append({
                "candidate_id": cand.instance_id,
                "suppressed_by": suppressing_id,
                "semantic_class": cand.semantic_class,
                "reason": duplicate_reason,
                "points": cand.point_count,
            })
        else:
            kept.append(cand)

    return DeduplicationResult(
        active_instances=kept,
        suppressed_candidates=suppressed,
    )
