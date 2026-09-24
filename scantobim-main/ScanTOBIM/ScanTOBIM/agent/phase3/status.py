"""Model execution and adapter status constants for Phase 3.

Engineering Rule:
Every model must be labelled as exactly one of:
  - REAL_NEURAL_INFERENCE
  - GEOMETRIC_ADAPTER
  - ALGORITHMIC_ADAPTER
  - UNAVAILABLE
  - FAILED

Never report a geometric implementation as a neural model.
"""

from __future__ import annotations

REAL_NEURAL_INFERENCE: str = "REAL_NEURAL_INFERENCE"
GEOMETRIC_ADAPTER: str = "GEOMETRIC_ADAPTER"
ALGORITHMIC_ADAPTER: str = "ALGORITHMIC_ADAPTER"
UNAVAILABLE: str = "UNAVAILABLE"
FAILED: str = "FAILED"

ALL_MODEL_STATUSES = {
    REAL_NEURAL_INFERENCE,
    GEOMETRIC_ADAPTER,
    ALGORITHMIC_ADAPTER,
    UNAVAILABLE,
    FAILED,
}
