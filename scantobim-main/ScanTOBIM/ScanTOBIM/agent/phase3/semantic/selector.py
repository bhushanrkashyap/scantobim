"""Semantic Model Selector — Phase 3A CPU Production.

Uses NeuralModelRegistry.best_production_semantic_model() for adapter selection.

Priority (enforced by registry):
  1. RandLA-Net with verified trained checkpoint → REAL_NEURAL_INFERENCE
  2. PTv2 with verified trained checkpoint → REAL_NEURAL_INFERENCE
  3. Geometric → GEOMETRIC_ADAPTER (always production-ready)

TEST_ONLY_NEURAL_FORWARD models are EXCLUDED from production selection.
Random-initialized models MUST NOT reach the production pipeline.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import structlog

from agent.phase3.neural.registry import NeuralModelRegistry
from agent.phase3.semantic.adapter import SemanticModelAdapter

if TYPE_CHECKING:
    pass

logger = structlog.get_logger()


class SemanticModelSelector:
    """Evaluates and chooses the active semantic segmentation pipeline.

    Delegates all selection logic to NeuralModelRegistry to enforce
    the production-readiness contract.
    """

    def __init__(self, preferred_model: str | None = None) -> None:
        self.preferred_model = (preferred_model or "auto").strip().lower()
        self._registry = NeuralModelRegistry()
        self._registry.register_defaults()
        self._evaluation_report: dict[str, Any] = {}

    def evaluate_all(self) -> dict[str, Any]:
        """Inspect and health-check all registered adapters."""
        self._evaluation_report = self._registry.evaluate_all()
        return self._evaluation_report

    def select_active_adapter(self) -> tuple[str, SemanticModelAdapter, str]:
        """Select best production-ready adapter.

        Uses best_production_semantic_model() which enforces:
          - trained checkpoint (not random-init)
          - CPU compatibility
          - successful validation
          - REAL_NEURAL_INFERENCE status

        Returns:
            (key, selected_adapter, rationale)
        """
        preferred = None if self.preferred_model == "auto" else self.preferred_model
        key, adapter, rationale = self._registry.best_production_semantic_model(
            preferred_key=preferred,
        )
        logger.info(
            "semantic_model_selected",
            key=key,
            status=adapter.status,
            production_neural=self._registry.production_neural_models(),
        )
        return key, adapter, rationale

    def production_neural_models(self) -> list[str]:
        """Return keys of healthy, trained neural models."""
        return self._registry.production_neural_models()

    @property
    def adapters(self) -> dict[str, "SemanticModelAdapter"]:
        """Backward-compatible view of registered adapters keyed by model name."""
        if not self._registry._evaluated:
            self._registry.evaluate_all()
        return {key: entry.adapter for key, entry in self._registry._entries.items()}
