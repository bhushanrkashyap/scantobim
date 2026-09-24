"""Point Cloud Annotation Tool & Dataset Generator for Phase 3C.

Provides a lightweight, CPU-friendly annotation workflow:
  - Spatial region selection (disjoint bounding boxes)
  - Point and cluster selection
  - Instance identifier assignment
  - Semantic label and superclass assignment
  - UNKNOWN and REVIEW status marking
  - Save, resume, and export with point-level provenance back to authentic E57 point IDs
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
from scipy.spatial import cKDTree

from agent.phase3.annotation.schema import (
    AnnotationStatus,
    GroundTruthDataset,
    InstanceGroundTruth,
    PointAnnotation,
    RegionMetadata,
    SplitType,
)
from agent.phase3.geometry import compute_geometric_features
from agent.phase3.taxonomy import (
    HIERARCHICAL_TAXONOMY,
    LEVEL1_CATEGORIES,
    get_category_for_class,
)

# Canonical 19-class label space for Phase 3C multi-class learning
PHASE_3C_LABEL_SPACE: list[str] = [
    "UNKNOWN",      # 0
    "WALL",         # 1
    "FLOOR",        # 2
    "SLAB",         # 3
    "CEILING",      # 4
    "COLUMN",       # 5
    "BEAM",         # 6
    "PIPE",         # 7
    "DUCT",         # 8
    "CABLE_TRAY",   # 9
    "CONDUIT",      # 10
    "VALVE",        # 11
    "FITTING",      # 12
    "EQUIPMENT",    # 13
    "DOOR",         # 14
    "WINDOW",       # 15
    "STAIR",        # 16
    "RAILING",      # 17
    "OPENING",      # 18
]

CLASS_NAME_TO_ID: dict[str, int] = {name: idx for idx, name in enumerate(PHASE_3C_LABEL_SPACE)}
ID_TO_CLASS_NAME: dict[int, str] = {idx: name for idx, name in enumerate(PHASE_3C_LABEL_SPACE)}


class PointCloudAnnotator:
    """Manages annotation tasks, spatial region carving, and ground truth creation."""

    def __init__(self, source_name: str = "point cloud data") -> None:
        self.source_name = source_name

    def extract_spatial_region(
        self,
        points: np.ndarray,
        point_ids: np.ndarray,
        bbox_min: list[float] | np.ndarray,
        bbox_max: list[float] | np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Extract a spatial block with original point indices.

        Returns:
            sub_points: (M, 3) coordinates
            sub_point_ids: (M,) original E57 indices
            mask: (N,) boolean selection mask
        """
        b_min = np.asarray(bbox_min, dtype=np.float32)
        b_max = np.asarray(bbox_max, dtype=np.float32)

        in_x = (points[:, 0] >= b_min[0]) & (points[:, 0] <= b_max[0])
        in_y = (points[:, 1] >= b_min[1]) & (points[:, 1] <= b_max[1])
        in_z = (points[:, 2] >= b_min[2]) & (points[:, 2] <= b_max[2])
        mask = in_x & in_y & in_z

        sub_pts = points[mask]
        sub_ids = point_ids[mask]
        return sub_pts, sub_ids, mask

    def build_ground_truth_from_proposals(
        self,
        region_id: str,
        split: SplitType,
        points: np.ndarray,
        point_ids: np.ndarray,
        instances_spec: list[dict[str, Any]],
        notes: str = "",
    ) -> GroundTruthDataset:
        """Create an annotated ground truth dataset slice from verified instance specifications.

        Each spec in instances_spec contains:
          - object_id: str
          - semantic_class: str (e.g. WALL, PIPE, COLUMN, UNKNOWN)
          - point_indices: list[int] (indices relative to points)
          - review_status: VALIDATED | REVIEW | UNCERTAIN
        """
        N = len(points)
        semantic_labels = np.full(N, "UNKNOWN", dtype=object)
        class_ids = np.zeros(N, dtype=np.int64)
        instance_ids = np.full(N, -1, dtype=np.int64)

        instances: list[InstanceGroundTruth] = []
        classes_present: set[str] = set()

        for inst_idx, spec in enumerate(instances_spec):
            obj_id = spec["object_id"]
            sem_cls = spec["semantic_class"].upper()
            indices = np.asarray(spec["point_indices"], dtype=int)
            rev_stat = spec.get("review_status", AnnotationStatus.VALIDATED.value)

            if len(indices) == 0:
                continue

            c_id = CLASS_NAME_TO_ID.get(sem_cls, 0)
            super_cls = get_category_for_class(sem_cls)

            semantic_labels[indices] = sem_cls
            class_ids[indices] = c_id
            instance_ids[indices] = inst_idx
            classes_present.add(sem_cls)

            inst_pts = points[indices]
            centroid = np.mean(inst_pts, axis=0).tolist()
            b_min = np.min(inst_pts, axis=0).tolist()
            b_max = np.max(inst_pts, axis=0).tolist()
            extents = (np.max(inst_pts, axis=0) - np.min(inst_pts, axis=0)).tolist()

            src_ids = [int(point_ids[i]) for i in indices]

            instances.append(
                InstanceGroundTruth(
                    object_id=obj_id,
                    instance_id=inst_idx,
                    semantic_class=sem_cls,
                    superclass=super_cls,
                    point_ids=src_ids,
                    centroid=centroid,
                    bbox_min=b_min,
                    bbox_max=b_max,
                    obb_extents=extents,
                    review_status=rev_stat,
                )
            )

        b_min_all = np.min(points, axis=0).tolist() if N > 0 else [0.0, 0.0, 0.0]
        b_max_all = np.max(points, axis=0).tolist() if N > 0 else [0.0, 0.0, 0.0]

        meta = RegionMetadata(
            region_id=region_id,
            split=split.value,
            spatial_bounds_min=b_min_all,
            spatial_bounds_max=b_max_all,
            point_count=N,
            instance_count=len(instances),
            classes_present=sorted(list(classes_present)),
            source_e57=self.source_name,
            created_at=datetime.now(timezone.utc).isoformat(),
            notes=notes,
        )

        return GroundTruthDataset(
            region_id=region_id,
            split=split,
            points=points,
            point_ids=point_ids,
            semantic_labels=semantic_labels,
            class_ids=class_ids,
            instance_ids=instance_ids,
            instances=instances,
            metadata=meta,
        )
