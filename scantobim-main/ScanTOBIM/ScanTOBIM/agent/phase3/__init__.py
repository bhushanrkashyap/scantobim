"""Phase 3 — Semantic Understanding + Instance Segmentation + Architectural Reconstruction."""

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
    "map_to_canonical",
    "run_phase3_reconstruction",
]
