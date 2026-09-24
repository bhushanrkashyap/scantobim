"""Phase 3 — Semantic Understanding + Instance Segmentation + Architectural Reconstruction."""

from agent.phase3.open_world_pipeline import (
    Phase3BExecutionResult,
    run_phase3b_pipeline,
)
from agent.phase3.pipeline import Phase3ReconstructionResult, run_phase3_reconstruction
from agent.phase3.status import (
    ALGORITHMIC_ADAPTER,
    FAILED,
    GEOMETRIC_ADAPTER,
    REAL_NEURAL_INFERENCE,
    UNAVAILABLE,
)
from agent.phase3.taxonomy import CANONICAL_LABELS, map_to_canonical

__all__ = [
    "ALGORITHMIC_ADAPTER",
    "CANONICAL_LABELS",
    "FAILED",
    "GEOMETRIC_ADAPTER",
    "REAL_NEURAL_INFERENCE",
    "UNAVAILABLE",
    "Phase3ReconstructionResult",
    "Phase3BExecutionResult",
    "map_to_canonical",
    "run_phase3_reconstruction",
    "run_phase3b_pipeline",
]
