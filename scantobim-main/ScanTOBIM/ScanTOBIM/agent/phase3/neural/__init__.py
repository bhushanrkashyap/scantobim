# agent/phase3/neural/__init__.py
"""Phase 3A Neural Integration — Dynamic Model Registry and Comparison Reporting."""

from agent.phase3.neural.comparison_reporter import SemanticModelComparisonReport
from agent.phase3.neural.registry import NeuralModelRegistry

__all__ = [
    "NeuralModelRegistry",
    "SemanticModelComparisonReport",
]
