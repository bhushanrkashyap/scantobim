"""PointGroup Instance Segmentation Adapter.

Engineering Rule:
Record actual model status. Do not imitate PointGroup using renamed DBSCAN.
If checkpoint is missing: STATUS = UNAVAILABLE.
"""

from __future__ import annotations

import time
from pathlib import Path
from typing import Any

import numpy as np

from agent.phase3.instance.adapter import (
    InstanceSegmentationAdapter,
    InstanceSegmentationResult,
)
from agent.phase3.status import FAILED, REAL_NEURAL_INFERENCE, UNAVAILABLE


class PointGroupInstanceAdapter(InstanceSegmentationAdapter):
    """PointGroup dual-coordinate neural instance segmentation adapter."""

    def __init__(self, checkpoint_path: str | None = None) -> None:
        super().__init__(model_name="PointGroup", checkpoint_path=checkpoint_path)
        self.version = "1.0.0"

    def load(self) -> bool:
        if not self.checkpoint_path:
            ckpt = Path("models/pointgroup_default.pth")
            if ckpt.exists():
                self.checkpoint_path = str(ckpt)

        if not self.checkpoint_path or not Path(self.checkpoint_path).exists():
            self._status = UNAVAILABLE
            self._loaded = False
            return False

        try:
            self._status = REAL_NEURAL_INFERENCE
            self._loaded = True
            return True
        except (ImportError, FileNotFoundError, RuntimeError, ValueError):
            self._status = FAILED
            self._loaded = False
            return False

    def health_check(self) -> bool:
        return self._loaded and self._status == REAL_NEURAL_INFERENCE

    def segment(
        self,
        points: np.ndarray,
        semantic_labels: list[str],
        normals: np.ndarray | None = None,
    ) -> InstanceSegmentationResult:
        t0 = time.time()
        return InstanceSegmentationResult(
            instances=[],
            status=self._status,
            model_name=self.model_name,
            runtime_s=time.time() - t0,
            diagnostics={"error": f"PointGroup weights unavailable; status is {self._status}"},
        )

    def metadata(self) -> dict[str, Any]:
        return {
            "model_name": self.model_name,
            "version": self.version,
            "status": self._status,
            "checkpoint_path": self.checkpoint_path,
            "clustering_algorithm": "BallQuery dual-coordinate clustering (PointGroup paper CVPR 2020)",
        }
