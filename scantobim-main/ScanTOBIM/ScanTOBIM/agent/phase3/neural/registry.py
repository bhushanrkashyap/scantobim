"""Dynamic Neural Model Registry — Phase 3A CPU Production.

Production Contract:
  - best_production_semantic_model() returns ONLY models satisfying ALL:
      1. Implementation available
      2. Valid trained checkpoint (not random-init)
      3. CPU compatible (no CUDA)
      4. Successful load (all validation gates passed)
      5. health_check() == True
      6. status == REAL_NEURAL_INFERENCE
  - any_neural_available() is DEPRECATED for production use.
    Replace with best_production_semantic_model() which enforces trained checkpoint.
  - TEST_ONLY_NEURAL_FORWARD models are NOT production candidates.
  - Geometric adapter is always the fallback (PRODUCTION_READY=True but not neural).
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any

import structlog

from agent.phase3.semantic.adapter import SemanticModelAdapter
from agent.phase3.semantic.geometric_adapter import GeometricSemanticAdapter
from agent.phase3.semantic.kpconv_adapter import KPConvSemanticAdapter
from agent.phase3.semantic.ptv2_adapter import PTv2SemanticAdapter
from agent.phase3.semantic.randla_adapter import RandLANetSemanticAdapter
from agent.phase3.status import (
    GEOMETRIC_ADAPTER,
    NO_COMPATIBLE_CHECKPOINT,
    REAL_NEURAL_INFERENCE,
    TEST_ONLY_NEURAL_FORWARD,
    is_production_ready,
    is_real_neural,
)

logger = structlog.get_logger()


@dataclass
class ModelRegistryEntry:
    """One registered model's identity and runtime state."""

    key: str
    display_name: str
    model_type: str           # "neural_semantic", "geometric_semantic"
    adapter: SemanticModelAdapter
    priority: int             # Lower = higher priority
    cpu_compatible: bool      # Can run without CUDA
    requires_checkpoint: bool # Needs a trained checkpoint file
    loaded: bool = False
    healthy: bool = False
    production_ready: bool = False
    load_time_s: float = 0.0
    load_error: str | None = None
    diagnostics: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "key": self.key,
            "display_name": self.display_name,
            "model_type": self.model_type,
            "priority": self.priority,
            "cpu_compatible": self.cpu_compatible,
            "requires_checkpoint": self.requires_checkpoint,
            "status": self.adapter.status,
            "loaded": self.loaded,
            "healthy": self.healthy,
            "production_ready": self.production_ready,
            "load_time_s": round(self.load_time_s, 4),
            "load_error": self.load_error,
            "checkpoint": self.adapter.checkpoint_path,
            "diagnostics": self.diagnostics,
        }


class NeuralModelRegistry:
    """Dynamic registry for all semantic models with production-readiness tracking.

    Key method: best_production_semantic_model()
      Enforces: trained checkpoint + CPU + validated load + REAL_NEURAL_INFERENCE.

    Deprecated (do NOT use for production decisions):
      any_neural_available() — does not enforce trained checkpoint.
    """

    def __init__(self) -> None:
        self._entries: dict[str, ModelRegistryEntry] = {}
        self._evaluated: bool = False
        self._evaluation_report: dict[str, Any] = {}

    def register_defaults(self) -> None:
        """Register the canonical set of semantic models."""
        from pathlib import Path

        multiclass_path = Path("models/randlanet_multiclass_cpu.pth")
        self._entries = {}

        if multiclass_path.exists():
            self._entries["randlanet_multiclass"] = ModelRegistryEntry(
                key="randlanet_multiclass",
                display_name="RandLA-Net Multi-Class (19-Class, 14D Features — Authentic E57 Trained)",
                model_type="neural_semantic",
                adapter=RandLANetSemanticAdapter(checkpoint_path=str(multiclass_path)),
                priority=1,
                cpu_compatible=True,
                requires_checkpoint=True,
            )

        self._entries["randlanet"] = ModelRegistryEntry(
            key="randlanet",
            display_name="RandLA-Net Architectural (2-Class, Synthetic Trained Baseline)",
            model_type="neural_semantic",
            adapter=RandLANetSemanticAdapter(checkpoint_path="models/randlanet_architectural_cpu.pth"),
            priority=2,
            cpu_compatible=True,
            requires_checkpoint=True,
        )

        self._entries["ptv2"] = ModelRegistryEntry(
            key="ptv2",
            display_name="Point Transformer V2 (checkpoint required, S3DIS)",
            model_type="neural_semantic",
            adapter=PTv2SemanticAdapter(),
            priority=3,
            cpu_compatible=True,  # Architecture can run on CPU if checkpoint available
            requires_checkpoint=True,
        )

        self._entries["kpconv"] = ModelRegistryEntry(
            key="kpconv",
            display_name="KPConv (CUDA custom operators — CPU_UNAVAILABLE)",
            model_type="neural_semantic",
            adapter=KPConvSemanticAdapter(),
            priority=4,
            cpu_compatible=False,  # Requires CUDA custom ops
            requires_checkpoint=True,
        )

        self._entries["geometric"] = ModelRegistryEntry(
            key="geometric",
            display_name="Geometric Analytical Adapter (deterministic CPU fallback)",
            model_type="geometric_semantic",
            adapter=GeometricSemanticAdapter(),
            priority=99,
            cpu_compatible=True,
            requires_checkpoint=False,
        )

    def evaluate_all(self) -> dict[str, Any]:
        """Load and health-check every registered model. Returns full evaluation report."""
        if not self._entries:
            self.register_defaults()

        results: dict[str, Any] = {}
        for key, entry in self._entries.items():
            # Skip non-CPU-compatible models
            if not entry.cpu_compatible:
                entry.loaded = False
                entry.healthy = False
                entry.production_ready = False
                entry.load_error = "CPU_UNAVAILABLE: requires CUDA custom operators"
                results[key] = entry.to_dict()
                logger.info(
                    "registry_model_skipped_cpu_unavailable",
                    key=key,
                )
                continue

            t0 = time.time()
            try:
                loaded = entry.adapter.load()
                healthy = entry.adapter.health_check()
                prod_ready = is_production_ready(entry.adapter.status)
                # Additional check: production_ready requires REAL_NEURAL_INFERENCE or GEOMETRIC_ADAPTER
                # TEST_ONLY_NEURAL_FORWARD is NOT production-ready
                if entry.adapter.status == TEST_ONLY_NEURAL_FORWARD:
                    prod_ready = False
                    healthy = False
                    logger.warning(
                        "registry_model_test_only",
                        key=key,
                        note="TEST_ONLY_NEURAL_FORWARD — not production-ready (no trained checkpoint).",
                    )

                entry.loaded = loaded
                entry.healthy = healthy
                entry.production_ready = prod_ready
                entry.load_time_s = time.time() - t0
                entry.diagnostics = entry.adapter.metadata()

                logger.info(
                    "registry_model_evaluated",
                    key=key,
                    status=entry.adapter.status,
                    healthy=healthy,
                    production_ready=prod_ready,
                )
            except Exception as exc:  # noqa: BLE001
                entry.loaded = False
                entry.healthy = False
                entry.production_ready = False
                entry.load_error = str(exc)
                entry.load_time_s = time.time() - t0
                logger.warning("registry_model_evaluate_error", key=key, error=str(exc))

            results[key] = entry.to_dict()

        self._evaluated = True
        self._evaluation_report = {
            "schema_version": "3a.2",
            "evaluated_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "models": results,
            "production_ready_count": sum(
                1 for e in self._entries.values() if e.production_ready
            ),
            "real_neural_count": sum(
                1 for e in self._entries.values()
                if e.adapter.status == REAL_NEURAL_INFERENCE
            ),
        }
        return self._evaluation_report

    def best_production_semantic_model(
        self,
        preferred_key: str | None = None,
    ) -> tuple[str, SemanticModelAdapter, str]:
        """Select the best model satisfying ALL production criteria.

        Production criteria:
          1. implementation_available (adapter loaded without import error)
          2. checkpoint_valid (checkpoint passes all 9 validation gates)
          3. cpu_compatible (no CUDA required)
          4. model_loaded (state dict loads)
          5. real_forward (CPU inference verified)
          6. output_validated (trained-not-random confirmed)
          7. status == REAL_NEURAL_INFERENCE

        Falls back to geometric adapter when no neural model meets all criteria.
        NEVER falls back to TEST_ONLY_NEURAL_FORWARD for production.

        Returns:
            (key, adapter, rationale)
        """
        if not self._evaluated:
            self.evaluate_all()

        # Honour explicit preference if it is production-ready neural
        if preferred_key and preferred_key in self._entries:
            entry = self._entries[preferred_key]
            if (
                entry.production_ready
                and entry.healthy
                and is_real_neural(entry.adapter.status)
            ):
                return (
                    preferred_key,
                    entry.adapter,
                    f"User-preferred production model '{preferred_key}': "
                    f"status={entry.adapter.status}, trained checkpoint verified.",
                )

        # Sort by priority; pick first production-ready NEURAL model
        sorted_entries = sorted(self._entries.values(), key=lambda e: e.priority)

        for entry in sorted_entries:
            if (
                entry.production_ready
                and entry.healthy
                and is_real_neural(entry.adapter.status)
            ):
                rationale = (
                    f"Selected '{entry.key}' ({entry.display_name}): "
                    f"priority={entry.priority}, status={entry.adapter.status}, "
                    f"trained_checkpoint=verified, cpu_compatible=True."
                )
                return entry.key, entry.adapter, rationale

        # No production neural model available → geometric fallback
        geom_entry = self._entries.get("geometric")
        if geom_entry is None or not geom_entry.production_ready:
            # Ensure geometric is loaded
            geom_adapter = GeometricSemanticAdapter()
            geom_adapter.load()
            self._entries["geometric"] = ModelRegistryEntry(
                key="geometric",
                display_name="Geometric Analytical Adapter",
                model_type="geometric_semantic",
                adapter=geom_adapter,
                priority=99,
                cpu_compatible=True,
                requires_checkpoint=False,
                loaded=True,
                healthy=True,
                production_ready=True,
            )
            geom_entry = self._entries["geometric"]

        # Report why neural failed
        neural_statuses = {
            k: e.adapter.status
            for k, e in self._entries.items()
            if e.model_type == "neural_semantic"
        }
        rationale = (
            "No production-ready neural semantic model available. "
            f"Neural model statuses: {neural_statuses}. "
            "Falling back to GEOMETRIC_ADAPTER (production-ready, deterministic, CPU). "
            "Acquire a trained checkpoint to enable neural inference."
        )
        return "geometric", geom_entry.adapter, rationale

    def select_best_adapter(
        self,
        preferred_key: str | None = None,
        require_neural: bool = False,
    ) -> tuple[str, SemanticModelAdapter, str]:
        """Alias for best_production_semantic_model() for backward compatibility.

        NOTE: Does NOT lower the production bar. TEST_ONLY_NEURAL_FORWARD
        models are still excluded from production selection.
        """
        return self.best_production_semantic_model(preferred_key=preferred_key)

    def get_evaluation_report(self) -> dict[str, Any]:
        if not self._evaluated:
            self.evaluate_all()
        return self._evaluation_report

    def production_neural_models(self) -> list[str]:
        """Return keys of models that are production-ready AND real neural."""
        if not self._evaluated:
            self.evaluate_all()
        return [
            key
            for key, entry in self._entries.items()
            if entry.production_ready and is_real_neural(entry.adapter.status)
        ]

    def any_neural_available(self) -> bool:
        """DEPRECATED: does not enforce trained checkpoint.

        Use production_neural_models() for production decisions.
        Returns True if any adapter reports REAL_NEURAL_INFERENCE (including
        possibly-untrained models). Not authoritative for production.
        """
        logger.warning(
            "deprecated_any_neural_available_called",
            note="Use production_neural_models() for production decisions.",
        )
        if not self._evaluated:
            self.evaluate_all()
        return len(self.production_neural_models()) > 0

    def get_entry(self, key: str) -> ModelRegistryEntry | None:
        """Return the registry entry for a model key."""
        if not self._entries:
            self.register_defaults()
        return self._entries.get(key)

    def get_primary_model(self) -> ModelRegistryEntry | None:
        """Return the highest priority registry entry."""
        if not self._entries:
            self.register_defaults()
        sorted_entries = sorted(self._entries.values(), key=lambda e: e.priority)
        return sorted_entries[0] if sorted_entries else None


def get_production_neural_model(preferred_key: str | None = None) -> SemanticModelAdapter | None:
    """Convenience function to get the highest-priority validated production model adapter."""
    registry = NeuralModelRegistry()
    _key, adapter, _rationale = registry.best_production_semantic_model(preferred_key=preferred_key)
    return adapter

