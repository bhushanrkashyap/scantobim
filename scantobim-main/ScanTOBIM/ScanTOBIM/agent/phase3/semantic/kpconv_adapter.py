"""KPConv Semantic Adapter for Sparse/Irregular Point Clouds.

Engineering Rules:
- Alternative semantic backbone for sparse scans and irregular density.
- Do not let KPConv-specific preprocessing become a global dependency.
- If no checkpoint: STATUS = UNAVAILABLE.
"""

from __future__ import annotations

import time
from pathlib import Path
from typing import Any

import numpy as np
import structlog

from agent.phase3.semantic.adapter import SemanticInferenceResult, SemanticModelAdapter
from agent.phase3.status import FAILED, REAL_NEURAL_INFERENCE, UNAVAILABLE
from agent.phase3.taxonomy import map_to_canonical

logger = structlog.get_logger()


class KPConvSemanticAdapter(SemanticModelAdapter):
    """Adapter for Kernel Point Convolution (KPConv) semantic segmentation."""

    def __init__(
        self,
        checkpoint_path: str | None = None,
        device: str = "cpu",
        kernel_points: int = 15,
        radius_m: float = 0.05,
    ) -> None:
        super().__init__(model_name="KPConv", checkpoint_path=checkpoint_path, device=device)
        self.version = "1.0.0"
        self.kernel_points = kernel_points
        self.radius_m = radius_m
        self._classes = ["wall", "floor", "ceiling", "beam", "column", "window", "door", "clutter"]
        self._model = None

    def load(self) -> bool:
        if not self.checkpoint_path:
            default_ckpt = Path("models/kpconv_weights.pth")
            if default_ckpt.exists():
                self.checkpoint_path = str(default_ckpt)

        if not self.checkpoint_path or not Path(self.checkpoint_path).exists():
            self._status = UNAVAILABLE
            self._loaded = False
            logger.info("kpconv_weights_unavailable", path=self.checkpoint_path)
            return False

        try:
            # KPConv requires custom CUDA operators; if missing, fail gracefully
            self._status = UNAVAILABLE
            self._loaded = False
            return False
        except (ImportError, FileNotFoundError, RuntimeError, ValueError):
            self._status = FAILED
            self._loaded = False
            return False

    def health_check(self) -> bool:
        return self._loaded and self._status == REAL_NEURAL_INFERENCE

    def infer(
        self,
        points: np.ndarray,
        colors: np.ndarray | None = None,
        normals: np.ndarray | None = None,
    ) -> SemanticInferenceResult:
        t0 = time.time()
        return SemanticInferenceResult(
            labels=["UNKNOWN"] * len(points),
            probabilities=np.zeros(len(points), dtype=float),
            status=self._status,
            model_name=self.model_name,
            checkpoint_path=self.checkpoint_path,
            device=self.device,
            runtime_s=time.time() - t0,
            diagnostics={"error": f"KPConv not loaded; status is {self._status}"},
        )

    def supported_labels(self) -> list[str]:
        return [map_to_canonical(c, "s3dis") for c in self._classes]

    def metadata(self) -> dict[str, Any]:
        return {
            "model_name": self.model_name,
            "version": self.version,
            "status": self._status,
            "checkpoint_path": self.checkpoint_path,
            "device": self.device,
            "kernel_points": self.kernel_points,
            "radius_m": self.radius_m,
            "supported_classes": self.supported_labels(),
            "preprocessing": "Subsampled grid points with KDTree spherical neighborhood search",
            "output_schema": "Deformable kernel convolution per-point logits",
        }
