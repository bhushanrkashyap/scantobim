"""Semantic Segmentation Quality Metrics & Evaluation Engine.

Computes comprehensive ground-truth-based evaluation metrics:
  - Overall Accuracy (OA)
  - Macro Precision, Recall, F1
  - Mean Intersection over Union (mIoU)
  - Per-class Precision, Recall, F1, IoU
  - Confusion Matrix
  - Unknown class precision, recall, and rejection statistics
  - False Known Rate (classifying UNKNOWN as a known class)

Strict Engineering Rule:
  Metrics are NEVER computed from predictions alone. Ground truth is required.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any

import numpy as np

from agent.phase3.annotation.annotator import PHASE_3C_LABEL_SPACE


@dataclass
class ClassMetric:
    class_id: int
    class_name: str
    tp: int
    fp: int
    fn: int
    precision: float
    recall: float
    f1: float
    iou: float
    support: int


@dataclass
class SemanticEvaluationMetrics:
    split_name: str
    total_points: int
    overall_accuracy: float
    macro_precision: float
    macro_recall: float
    macro_f1: float
    mean_iou: float
    unknown_precision: float
    unknown_recall: float
    unknown_rejection_rate: float
    false_known_rate: float  # GT is UNKNOWN but predicted as known
    per_class_metrics: dict[str, ClassMetric]
    confusion_matrix: list[list[int]]
    classes_evaluated: list[str]

    def to_dict(self) -> dict[str, Any]:
        """Convert metrics to JSON-serializable dictionary."""
        return {
            "split_name": self.split_name,
            "total_points": self.total_points,
            "overall_accuracy": round(float(self.overall_accuracy), 4),
            "macro_precision": round(float(self.macro_precision), 4),
            "macro_recall": round(float(self.macro_recall), 4),
            "macro_f1": round(float(self.macro_f1), 4),
            "mean_iou": round(float(self.mean_iou), 4),
            "unknown_precision": round(float(self.unknown_precision), 4),
            "unknown_recall": round(float(self.unknown_recall), 4),
            "unknown_rejection_rate": round(float(self.unknown_rejection_rate), 4),
            "false_known_rate": round(float(self.false_known_rate), 4),
            "classes_evaluated": self.classes_evaluated,
            "per_class": {
                k: asdict(v) for k, v in self.per_class_metrics.items()
            },
            "confusion_matrix": self.confusion_matrix,
        }


def compute_semantic_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    split_name: str = "VALIDATION",
    label_space: list[str] = PHASE_3C_LABEL_SPACE,
) -> SemanticEvaluationMetrics:
    """Compute rigorous multi-class semantic metrics against ground truth.

    Args:
        y_true: (N,) ground-truth class IDs [0..C-1]
        y_pred: (N,) predicted class IDs [0..C-1]
        split_name: Name of split (TRAIN, VALIDATION, TEST)
        label_space: List of class names
    """
    assert len(y_true) == len(y_pred), f"Size mismatch: {len(y_true)} vs {len(y_pred)}"
    N = len(y_true)
    if N == 0:
        return SemanticEvaluationMetrics(
            split_name=split_name,
            total_points=0,
            overall_accuracy=0.0,
            macro_precision=0.0,
            macro_recall=0.0,
            macro_f1=0.0,
            mean_iou=0.0,
            unknown_precision=0.0,
            unknown_recall=0.0,
            unknown_rejection_rate=0.0,
            false_known_rate=0.0,
            per_class_metrics={},
            confusion_matrix=[],
            classes_evaluated=[],
        )

    num_classes = len(label_space)
    conf_mat = np.zeros((num_classes, num_classes), dtype=np.int64)

    # Valid indices within range
    valid_mask = (y_true >= 0) & (y_true < num_classes) & (y_pred >= 0) & (y_pred < num_classes)
    y_t_valid = y_true[valid_mask]
    y_p_valid = y_pred[valid_mask]

    for t, p in zip(y_t_valid, y_p_valid):
        conf_mat[t, p] += 1

    overall_accuracy = float(np.trace(conf_mat) / max(N, 1))

    per_class: dict[str, ClassMetric] = {}
    f1_list: list[float] = []
    prec_list: list[float] = []
    rec_list: list[float] = []
    iou_list: list[float] = []
    eval_classes: list[str] = []

    for c_id in range(num_classes):
        c_name = label_space[c_id]
        support = int(np.sum(conf_mat[c_id, :]))
        pred_count = int(np.sum(conf_mat[:, c_id]))
        tp = int(conf_mat[c_id, c_id])
        fp = int(pred_count - tp)
        fn = int(support - tp)

        precision = float(tp / (tp + fp)) if (tp + fp) > 0 else 0.0
        recall = float(tp / (tp + fn)) if (tp + fn) > 0 else 0.0
        f1 = float(2.0 * precision * recall / (precision + recall)) if (precision + recall) > 0 else 0.0
        union = tp + fp + fn
        iou = float(tp / union) if union > 0 else 0.0

        cm = ClassMetric(
            class_id=c_id,
            class_name=c_name,
            tp=tp,
            fp=fp,
            fn=fn,
            precision=round(precision, 4),
            recall=round(recall, 4),
            f1=round(f1, 4),
            iou=round(iou, 4),
            support=support,
        )
        per_class[c_name] = cm

        # Only average over classes with actual support or predictions
        if support > 0 or pred_count > 0:
            eval_classes.append(c_name)
            prec_list.append(precision)
            rec_list.append(recall)
            f1_list.append(f1)
            iou_list.append(iou)

    macro_precision = float(np.mean(prec_list)) if prec_list else 0.0
    macro_recall = float(np.mean(rec_list)) if rec_list else 0.0
    macro_f1 = float(np.mean(f1_list)) if f1_list else 0.0
    mean_iou = float(np.mean(iou_list)) if iou_list else 0.0

    # Unknown metrics (Class ID 0 = UNKNOWN)
    unknown_tp = per_class["UNKNOWN"].tp if "UNKNOWN" in per_class else 0
    unknown_support = per_class["UNKNOWN"].support if "UNKNOWN" in per_class else 0
    unknown_prec = per_class["UNKNOWN"].precision if "UNKNOWN" in per_class else 0.0
    unknown_rec = per_class["UNKNOWN"].recall if "UNKNOWN" in per_class else 0.0
    unknown_rejection = float(unknown_tp / max(unknown_support, 1))

    # False known rate: GT is UNKNOWN (class 0), but predicted as any class > 0
    false_known_count = int(np.sum(conf_mat[0, 1:])) if num_classes > 1 else 0
    false_known_rate = float(false_known_count / max(unknown_support, 1))

    return SemanticEvaluationMetrics(
        split_name=split_name,
        total_points=N,
        overall_accuracy=round(overall_accuracy, 4),
        macro_precision=round(macro_precision, 4),
        macro_recall=round(macro_recall, 4),
        macro_f1=round(macro_f1, 4),
        mean_iou=round(mean_iou, 4),
        unknown_precision=round(unknown_prec, 4),
        unknown_recall=round(unknown_rec, 4),
        unknown_rejection_rate=round(unknown_rejection, 4),
        false_known_rate=round(false_known_rate, 4),
        per_class_metrics=per_class,
        confusion_matrix=conf_mat.tolist(),
        classes_evaluated=eval_classes,
    )
