"""Semantic Model Adapter Base Interface for Phase 3.

Contract:
- load(): Load model architecture & weights.
- health_check(): Validate operational readiness.
- infer(points, ...): Run semantic inference.
- supported_labels(): Expose supported canonical taxonomy classes.
- metadata(): Expose model details, version, checkpoint, device, status.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from agent.phase3.status import (
    UNAVAILABLE,
)


@dataclass
class SemanticInferenceResult:
    """Output contract for semantic inference."""

    labels: list[str]  # Canonical labels for each input point/cluster
    probabilities: np.ndarray  # (N,) or (N, C) probabilities
    status: str  # REAL_NEURAL_INFERENCE, GEOMETRIC_ADAPTER, etc.
    model_name: str
    checkpoint_path: str | None = None
    device: str = "cpu"
    runtime_s: float = 0.0
    point_indices: np.ndarray | None = None
    diagnostics: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "model_name": self.model_name,
            "status": self.status,
            "checkpoint_path": self.checkpoint_path,
            "device": self.device,
            "runtime_s": round(self.runtime_s, 4),
            "points_classified": len(self.labels),
            "diagnostics": self.diagnostics,
        }


class SemanticModelAdapter(ABC):
    """Abstract base class for all semantic model adapters."""

    def __init__(
        self,
        model_name: str,
        checkpoint_path: str | None = None,
        device: str = "cpu",
    ) -> None:
        self.model_name = model_name
        self.checkpoint_path = checkpoint_path
        self.device = device
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
        """Attempt to load model and checkpoint. Sets status accordingly."""

    @abstractmethod
    def health_check(self) -> bool:
        """Verify model readiness for inference."""

    @abstractmethod
    def infer(
        self,
        points: np.ndarray,
        colors: np.ndarray | None = None,
        normals: np.ndarray | None = None,
    ) -> SemanticInferenceResult:
        """Execute semantic inference on canonical coordinates."""

    @abstractmethod
    def supported_labels(self) -> list[str]:
        """List canonical labels this adapter can predict."""

    @abstractmethod
    def metadata(self) -> dict[str, Any]:
        """Expose operational metadata."""
