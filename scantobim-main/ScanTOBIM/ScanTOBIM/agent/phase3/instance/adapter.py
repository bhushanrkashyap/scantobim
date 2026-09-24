"""Instance Segmentation Adapter Base Interface for Phase 3.

Contract:
- load(): Load model architecture & weights.
- health_check(): Validate operational readiness.
- segment(points, semantic_labels): Partition point cloud into discrete instances.
- metadata(): Operational details, status, parameters.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from agent.phase3.status import UNAVAILABLE


@dataclass
class InstanceMask:
    """Represents a discrete segmented object instance."""

    instance_id: str
    semantic_class: str
    point_indices: np.ndarray  # Indices into the input point array
    confidence: float
    centroid_m: np.ndarray
    bounding_box_min_m: np.ndarray
    bounding_box_max_m: np.ndarray
    point_count: int
    # Phase 3B Section 14 Instance Fields
    oriented_bounding_box: dict[str, Any] = field(default_factory=dict)
    principal_axes: list[list[float]] = field(default_factory=list)
    dimensions: dict[str, float] = field(default_factory=dict)
    surface_area_estimate: float = 0.0
    volume_estimate_if_valid: float | None = None
    semantic_candidates: dict[str, float] = field(default_factory=dict)
    semantic_probability_distribution: dict[str, float] = field(default_factory=dict)
    evidence: dict[str, Any] = field(default_factory=dict)
    parent_instance: str | None = None
    child_instances: list[str] = field(default_factory=list)
    topology_relations: list[dict[str, Any]] = field(default_factory=list)
    source_point_ids: list[int] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        res = {
            "instance_id": self.instance_id,
            "semantic_class": self.semantic_class,
            "point_count": self.point_count,
            "confidence": round(self.confidence, 4),
            "centroid_m": [round(float(c), 4) for c in self.centroid_m],
            "bbox_min_m": [round(float(c), 4) for c in self.bounding_box_min_m],
            "bbox_max_m": [round(float(c), 4) for c in self.bounding_box_max_m],
        }
        if self.oriented_bounding_box:
            res["oriented_bounding_box"] = self.oriented_bounding_box
        if self.principal_axes:
            res["principal_axes"] = self.principal_axes
        if self.dimensions:
            res["dimensions"] = self.dimensions
        if self.surface_area_estimate > 0:
            res["surface_area_estimate"] = round(self.surface_area_estimate, 4)
        if self.volume_estimate_if_valid is not None:
            res["volume_estimate_if_valid"] = round(self.volume_estimate_if_valid, 6)
        if self.semantic_candidates:
            res["semantic_candidates"] = self.semantic_candidates
        if self.semantic_probability_distribution:
            res["semantic_probability_distribution"] = self.semantic_probability_distribution
        if self.evidence:
            res["evidence"] = self.evidence
        if self.parent_instance is not None:
            res["parent_instance"] = self.parent_instance
        if self.child_instances:
            res["child_instances"] = self.child_instances
        if self.topology_relations:
            res["topology_relations"] = self.topology_relations
        if self.source_point_ids:
            res["source_point_ids_count"] = len(self.source_point_ids)
        return res


@dataclass
class InstanceSegmentationResult:
    """Output contract for instance segmentation."""

    instances: list[InstanceMask]
    status: str
    model_name: str
    runtime_s: float = 0.0
    diagnostics: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "model_name": self.model_name,
            "status": self.status,
            "instance_count": len(self.instances),
            "runtime_s": round(self.runtime_s, 4),
            "instances": [inst.to_dict() for inst in self.instances],
            "diagnostics": self.diagnostics,
        }


class InstanceSegmentationAdapter(ABC):
    """Abstract base class for instance segmentation models."""

    def __init__(self, model_name: str, checkpoint_path: str | None = None) -> None:
        self.model_name = model_name
        self.checkpoint_path = checkpoint_path
        self._status: str = UNAVAILABLE
        self._loaded: bool = False

    @property
    def status(self) -> str:
        return self._status

    @property
    def is_loaded(self) -> bool:
        return self._loaded

    @abstractmethod
    def load(self) -> bool:
        pass

    @abstractmethod
    def health_check(self) -> bool:
        pass

    @abstractmethod
    def segment(
        self,
        points: np.ndarray,
        semantic_labels: list[str],
        normals: np.ndarray | None = None,
    ) -> InstanceSegmentationResult:
        pass

    @abstractmethod
    def metadata(self) -> dict[str, Any]:
        pass
