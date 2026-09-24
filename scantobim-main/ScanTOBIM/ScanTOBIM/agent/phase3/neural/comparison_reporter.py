"""Semantic Model Comparison Reporter — Phase 3A.

Produces:
  - SEMANTIC_MODEL_COMPARISON.json: Side-by-side per-model evaluation results.
  - AI_EXECUTION_PROVENANCE.json: Source-indexed provenance for each neural execution.

Rules:
  - Never fabricate execution details.
  - A model appears in the neural section ONLY if it reported REAL_NEURAL_INFERENCE.
  - Geometric results appear in a separate geometric section.
"""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any

import numpy as np
import structlog

from agent.phase3.semantic.adapter import SemanticInferenceResult
from agent.phase3.status import REAL_NEURAL_INFERENCE

logger = structlog.get_logger()


def _ts() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


class SemanticModelComparisonReport:
    """Accumulates per-model inference results and writes comparison + provenance files."""

    def __init__(self, source_file: str = "unknown") -> None:
        self.source_file = source_file
        self.created_at = _ts()
        self._records: list[dict[str, Any]] = []

    def record(
        self,
        model_key: str,
        result: SemanticInferenceResult,
        points: np.ndarray,
        rationale: str = "",
    ) -> None:
        """Record one model's inference result for reporting."""
        n = len(points)
        wall_count = sum(1 for lbl in result.labels if lbl == "WALL")
        label_distribution: dict[str, int] = {}
        for lbl in result.labels:
            label_distribution[lbl] = label_distribution.get(lbl, 0) + 1

        record: dict[str, Any] = {
            "model_key": model_key,
            "model_name": result.model_name,
            "status": result.status,
            "is_neural": result.status == REAL_NEURAL_INFERENCE,
            "checkpoint_path": result.checkpoint_path,
            "device": result.device,
            "runtime_s": round(result.runtime_s, 4),
            "total_points": n,
            "wall_points": wall_count,
            "wall_fraction": round(wall_count / max(n, 1), 4),
            "label_distribution": label_distribution,
            "rationale": rationale,
            "diagnostics": result.diagnostics,
        }
        self._records.append(record)
        logger.info(
            "comparison_report_recorded",
            model_key=model_key,
            status=result.status,
            wall_points=wall_count,
        )

    def to_comparison_dict(self) -> dict[str, Any]:
        neural_results = [r for r in self._records if r["is_neural"]]
        geometric_results = [r for r in self._records if not r["is_neural"]]
        return {
            "schema_version": "3a.1",
            "report_type": "SEMANTIC_MODEL_COMPARISON",
            "source_file": self.source_file,
            "created_at": self.created_at,
            "neural_models_evaluated": len(neural_results),
            "geometric_models_evaluated": len(geometric_results),
            "phase3a_pass": len(neural_results) > 0,
            "neural_results": neural_results,
            "geometric_results": geometric_results,
        }

    def to_provenance_dict(
        self,
        selected_model_key: str,
        selected_result: SemanticInferenceResult,
        points: np.ndarray,
    ) -> dict[str, Any]:
        """Build a source-provenance record for the selected (active) model's inference."""
        n = len(points)
        bbox: dict[str, float] = {}
        if n > 0:
            bbox = {
                "x_min": float(points[:, 0].min()),
                "x_max": float(points[:, 0].max()),
                "y_min": float(points[:, 1].min()),
                "y_max": float(points[:, 1].max()),
                "z_min": float(points[:, 2].min()),
                "z_max": float(points[:, 2].max()),
            }

        return {
            "schema_version": "3a.1",
            "report_type": "AI_EXECUTION_PROVENANCE",
            "source_file": self.source_file,
            "created_at": self.created_at,
            "selected_model": {
                "key": selected_model_key,
                "name": selected_result.model_name,
                "status": selected_result.status,
                "is_real_neural_inference": selected_result.status == REAL_NEURAL_INFERENCE,
                "checkpoint_path": selected_result.checkpoint_path,
                "device": selected_result.device,
                "runtime_s": round(selected_result.runtime_s, 4),
            },
            "source_point_cloud": {
                "total_points": n,
                "bounding_box_m": bbox,
            },
            "semantic_output": {
                "total_labels": len(selected_result.labels),
                "wall_points": sum(1 for lbl in selected_result.labels if lbl == "WALL"),
                "has_source_indices": selected_result.point_indices is not None,
            },
            "diagnostics": selected_result.diagnostics,
            "all_models_evaluated": self._records,
        }

    def write_reports(
        self,
        output_dir: Path,
        selected_model_key: str,
        selected_result: SemanticInferenceResult,
        points: np.ndarray,
    ) -> tuple[Path, Path]:
        """Write both JSON report files and return their paths."""
        output_dir.mkdir(parents=True, exist_ok=True)

        comparison_path = output_dir / "SEMANTIC_MODEL_COMPARISON.json"
        provenance_path = output_dir / "AI_EXECUTION_PROVENANCE.json"

        comparison_path.write_text(
            json.dumps(self.to_comparison_dict(), indent=2, default=str), encoding="utf-8"
        )
        provenance_path.write_text(
            json.dumps(
                self.to_provenance_dict(selected_model_key, selected_result, points),
                indent=2,
                default=str,
            ),
            encoding="utf-8",
        )

        logger.info(
            "comparison_reports_written",
            comparison=str(comparison_path),
            provenance=str(provenance_path),
        )
        return comparison_path, provenance_path
