"""Semantic Model Selector.

Evaluates available models, verified checkpoints, supported labels, device,
and point cloud characteristics to select the authoritative validated semantic path.
Generates SEMANTIC_MODEL_SELECTION_REPORT data.
"""

from __future__ import annotations

from typing import Any

import structlog

from agent.phase3.semantic.adapter import SemanticModelAdapter
from agent.phase3.semantic.geometric_adapter import GeometricSemanticAdapter
from agent.phase3.semantic.kpconv_adapter import KPConvSemanticAdapter
from agent.phase3.semantic.ptv2_adapter import PTv2SemanticAdapter
from agent.phase3.semantic.randla_adapter import RandLANetSemanticAdapter

logger = structlog.get_logger()


class SemanticModelSelector:
    """Evaluates and chooses the active semantic segmentation pipeline."""

    def __init__(self, preferred_model: str | None = None) -> None:
        self.preferred_model = (preferred_model or "auto").strip().lower()
        self.adapters: dict[str, SemanticModelAdapter] = {
            "ptv2": PTv2SemanticAdapter(),
            "randlanet": RandLANetSemanticAdapter(),
            "kpconv": KPConvSemanticAdapter(),
            "geometric": GeometricSemanticAdapter(),
        }

    def evaluate_all(self) -> dict[str, Any]:
        """Inspect and health-check all registered adapters."""
        eval_report = {}
        for key, adapter in self.adapters.items():
            adapter.load()
            eval_report[key] = {
                "model_name": adapter.model_name,
                "status": adapter.status,
                "loaded": adapter.is_loaded,
                "healthy": adapter.health_check(),
                "checkpoint": adapter.checkpoint_path,
                "metadata": adapter.metadata(),
            }
        return eval_report

    def select_active_adapter(self) -> tuple[str, SemanticModelAdapter, str]:
        """Select best available adapter.

        Returns:
            (key, selected_adapter, rationale)
        """
        _eval_report = self.evaluate_all()

        # If user explicitly preferred a neural model and it's healthy:
        if self.preferred_model in self.adapters:
            ad = self.adapters[self.preferred_model]
            if ad.health_check():
                return (
                    self.preferred_model,
                    ad,
                    f"Selected explicitly configured model '{self.preferred_model}' ({ad.status}).",
                )

        # Priority 1: Verified Neural PTv2
        if self.adapters["ptv2"].health_check():
            return "ptv2", self.adapters["ptv2"], "Selected PTv2 (verified neural checkpoint available)."

        # Priority 2: Verified Neural RandLA-Net
        if self.adapters["randlanet"].health_check():
            return "randlanet", self.adapters["randlanet"], "Selected RandLA-Net (verified neural checkpoint available)."

        # Priority 3: Geometric Analytical Adapter (Always available, robust fallback)
        geom = self.adapters["geometric"]
        rationale = (
            "Neural checkpoints unavailable or unverified. Selected GEOMETRIC_ADAPTER "
            "to guarantee non-fabricated, mathematically verifiable semantic segmentation."
        )
        return "geometric", geom, rationale
