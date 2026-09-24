"""Model execution and adapter status constants for Phase 3 / Phase 3A CPU Production.

Engineering Rules — Phase 3A CPU Production:
  - REAL_NEURAL_INFERENCE: Only when real trained checkpoint + successful load + CPU forward pass.
  - TEST_ONLY_NEURAL_FORWARD: Architecture loads and forward pass works, but NO trained checkpoint.
    A random-initialized model MUST use this status. It is NEVER production-ready.
  - CPU_UNAVAILABLE: Model requires CUDA/GPU operators not available on this CPU-only machine.
  - NO_COMPATIBLE_CHECKPOINT: Architecture available but no matching trained checkpoint found.
  - GEOMETRIC_ADAPTER: Geometric/analytical path — always available, deterministic.
  - ALGORITHMIC_ADAPTER: Rule-based or statistical adapter without neural weights.
  - UNAVAILABLE: Model cannot be loaded for any reason.
  - FAILED: Model load or inference raised an unrecoverable error.

Production-readiness gate:
  PRODUCTION_READY = True  iff  status == REAL_NEURAL_INFERENCE

Never report a geometric or random-init implementation as a neural model.
Never promote TEST_ONLY_NEURAL_FORWARD to REAL_NEURAL_INFERENCE.
"""

from __future__ import annotations

# ── Core inference statuses ───────────────────────────────────────────────────

REAL_NEURAL_INFERENCE: str = "REAL_NEURAL_INFERENCE"
"""Real trained checkpoint + CPU forward pass. PRODUCTION_READY = True."""

TEST_ONLY_NEURAL_FORWARD: str = "TEST_ONLY_NEURAL_FORWARD"
"""Architecture works, forward pass executes, but model has NO trained checkpoint.
Random-initialized weights. MUST NOT be used in production semantic pipeline.
PRODUCTION_READY = False."""

GEOMETRIC_ADAPTER: str = "GEOMETRIC_ADAPTER"
"""Geometric / analytical inference — no neural weights. Always available."""

ALGORITHMIC_ADAPTER: str = "ALGORITHMIC_ADAPTER"
"""Rule-based or statistical inference — no neural weights."""

# ── Availability statuses ─────────────────────────────────────────────────────

UNAVAILABLE: str = "UNAVAILABLE"
"""Model cannot be loaded — missing dependency, missing file, or import failure."""

CPU_UNAVAILABLE: str = "CPU_UNAVAILABLE"
"""Model requires CUDA / GPU operators not available on this CPU-only system."""

NO_COMPATIBLE_CHECKPOINT: str = "NO_COMPATIBLE_CHECKPOINT"
"""Architecture is available and CPU-compatible, but no matching trained checkpoint exists.
PRODUCTION_READY = False. Must NOT fall back to random-weight inference."""

FAILED: str = "FAILED"
"""Model load or inference raised an unrecoverable error."""

# ── All valid statuses (for validation) ───────────────────────────────────────

ALL_MODEL_STATUSES: frozenset[str] = frozenset({
    REAL_NEURAL_INFERENCE,
    TEST_ONLY_NEURAL_FORWARD,
    GEOMETRIC_ADAPTER,
    ALGORITHMIC_ADAPTER,
    UNAVAILABLE,
    CPU_UNAVAILABLE,
    NO_COMPATIBLE_CHECKPOINT,
    FAILED,
})

# ── Production-ready gate ─────────────────────────────────────────────────────

PRODUCTION_READY_STATUSES: frozenset[str] = frozenset({
    REAL_NEURAL_INFERENCE,
    GEOMETRIC_ADAPTER,   # geometric is always production-ready as fallback
    ALGORITHMIC_ADAPTER,
})


def is_production_ready(status: str) -> bool:
    """Return True only for statuses that are production-safe.

    TEST_ONLY_NEURAL_FORWARD is explicitly excluded.
    """
    return status in PRODUCTION_READY_STATUSES


def is_real_neural(status: str) -> bool:
    """Return True only for verified, trained neural inference."""
    return status == REAL_NEURAL_INFERENCE
