"""Point Cloud Annotation Schema for Phase 3C Ground-Truth Dataset.

Provides schema definitions, data containers, and serialization for:
  - Point-level annotations (point_id, object_id, semantic_class, superclass, instance_id, etc.)
  - Instance-level annotations (object_id, instance_id, semantic_class, point_ids, obb, dimensions)
  - Dataset partitions (TRAIN, VALIDATION, TEST, REAL_DEPLOYMENT)
  - Provenance tracking back to authentic ASTM E57 point indices
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any

import numpy as np


class SplitType(str, Enum):
    """Dataset partition splits."""
    TRAIN = "TRAIN"
    VALIDATION = "VALIDATION"
    TEST = "TEST"
    REAL_DEPLOYMENT = "REAL_DEPLOYMENT"


class AnnotationStatus(str, Enum):
    """Annotation verification status."""
    VALIDATED = "VALIDATED"
    REVIEW = "REVIEW"
    UNCERTAIN = "UNCERTAIN"


@dataclass
class PointAnnotation:
    """Ground-truth annotation for an individual point."""
    point_id: int
    object_id: str
    semantic_class: str
    superclass: str
    instance_id: int
    annotation_status: str = AnnotationStatus.VALIDATED.value
    annotator: str = "expert_annotator"
    source_cloud: str = "point cloud data"


@dataclass
class InstanceGroundTruth:
    """Instance-level ground truth object."""
    object_id: str
    instance_id: int
    semantic_class: str
    superclass: str
    point_ids: list[int]
    centroid: list[float]
    bbox_min: list[float]
    bbox_max: list[float]
    obb_extents: list[float]
    review_status: str = AnnotationStatus.VALIDATED.value


@dataclass
class RegionMetadata:
    """Metadata for an annotated spatial partition/block."""
    region_id: str
    split: str
    spatial_bounds_min: list[float]
    spatial_bounds_max: list[float]
    point_count: int
    instance_count: int
    classes_present: list[str]
    source_e57: str
    created_at: str
    notes: str = ""


@dataclass
class GroundTruthDataset:
    """Container holding a ground-truth dataset slice with points, classes, and instances."""
    region_id: str
    split: SplitType
    points: np.ndarray  # (N, 3) float32 coordinates
    point_ids: np.ndarray  # (N,) int64 source indices
    semantic_labels: np.ndarray  # (N,) string semantic class names
    class_ids: np.ndarray  # (N,) int class indices
    instance_ids: np.ndarray  # (N,) int instance identifiers
    instances: list[InstanceGroundTruth]
    metadata: RegionMetadata

    def to_dict(self) -> dict[str, Any]:
        """Convert metadata and summary to JSON-serializable dictionary."""
        return {
            "region_id": self.region_id,
            "split": self.split.value,
            "point_count": len(self.points),
            "instance_count": len(self.instances),
            "classes_present": self.metadata.classes_present,
            "spatial_bounds_min": self.metadata.spatial_bounds_min,
            "spatial_bounds_max": self.metadata.spatial_bounds_max,
            "source_e57": self.metadata.source_e57,
        }

    def save_to_disk(self, base_dir: Path | str) -> None:
        """Persist ground truth annotations, point coordinates, and metadata to disk."""
        out_path = Path(base_dir)
        out_path.mkdir(parents=True, exist_ok=True)

        # 1. Save points, point_ids, class_ids, instance_ids as binary numpy archive
        np.savez_compressed(
            out_path / f"{self.region_id}_data.npz",
            points=self.points,
            point_ids=self.point_ids,
            class_ids=self.class_ids,
            instance_ids=self.instance_ids,
        )

        # 2. Save instances as JSON
        instances_json = [asdict(inst) for inst in self.instances]
        with open(out_path / f"{self.region_id}_instances.json", "w", encoding="utf-8") as f:
            json.dump(instances_json, f, indent=2)

        # 3. Save region metadata as JSON
        with open(out_path / f"{self.region_id}_metadata.json", "w", encoding="utf-8") as f:
            json.dump(asdict(self.metadata), f, indent=2)
