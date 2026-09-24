"""Instance Segmentation and Object Detection Evaluation Engine.

Computes standard instance-level quality metrics against ground-truth annotations:
  - Instance Precision, Recall, F1 (IoU >= threshold)
  - Mean Instance IoU (mIoU_inst)
  - Proposal Recall (class-agnostic object discovery rate)
  - Duplicate Rate (multiple proposals matching the same GT instance)
  - Fragmentation Rate (GT instance split into multiple fragments)
  - Merge Rate (single proposal bridging multiple distinct GT instances)

Engineering Rule:
  Metrics are NEVER synthesized or estimated from confidence.
  They are strictly computed from point-level intersection over union (IoU)
  between predicted instances and ground-truth instances.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any

import numpy as np

from agent.phase3.annotation.schema import InstanceGroundTruth


@dataclass
class InstanceMatch:
    gt_object_id: str
    pred_object_id: str
    gt_class: str
    pred_class: str
    iou: float
    is_class_match: bool
    is_true_positive: bool


@dataclass
class InstanceEvaluationMetrics:
    split_name: str
    num_gt_instances: int
    num_pred_instances: int
    iou_threshold: float
    true_positives: int
    false_positives: int
    false_negatives: int
    instance_precision: float
    instance_recall: float
    instance_f1: float
    mean_instance_iou: float
    proposal_recall: float       # Class-agnostic discovery rate
    duplicate_rate: float        # Multiple proposals for one GT object
    fragmentation_rate: float    # GT instance broken into fragments
    merge_rate: float            # Multiple GT objects merged into one proposal
    matches: list[InstanceMatch] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        """Convert metrics to JSON dictionary."""
        return {
            "split_name": self.split_name,
            "num_gt_instances": self.num_gt_instances,
            "num_pred_instances": self.num_pred_instances,
            "iou_threshold": self.iou_threshold,
            "true_positives": self.true_positives,
            "false_positives": self.false_positives,
            "false_negatives": self.false_negatives,
            "instance_precision": round(float(self.instance_precision), 4),
            "instance_recall": round(float(self.instance_recall), 4),
            "instance_f1": round(float(self.instance_f1), 4),
            "mean_instance_iou": round(float(self.mean_instance_iou), 4),
            "proposal_recall": round(float(self.proposal_recall), 4),
            "duplicate_rate": round(float(self.duplicate_rate), 4),
            "fragmentation_rate": round(float(self.fragmentation_rate), 4),
            "merge_rate": round(float(self.merge_rate), 4),
        }


def evaluate_instances(
    gt_instances: list[InstanceGroundTruth],
    pred_instances: list[dict[str, Any]],
    split_name: str = "VALIDATION",
    iou_threshold: float = 0.50,
) -> InstanceEvaluationMetrics:
    """Evaluate predicted instances against ground-truth instances using exact point-set IoU.

    Args:
        gt_instances: List of ground-truth instances with `point_ids` (set of E57 point IDs).
        pred_instances: List of predicted instances with `source_point_ids` (or `point_ids`).
        split_name: Split identifier (VALIDATION / TEST).
        iou_threshold: Minimum IoU required for a True Positive match (default: 0.50).
    """
    N_gt = len(gt_instances)
    N_pred = len(pred_instances)

    if N_gt == 0:
        return InstanceEvaluationMetrics(
            split_name=split_name,
            num_gt_instances=0,
            num_pred_instances=N_pred,
            iou_threshold=iou_threshold,
            true_positives=0,
            false_positives=N_pred,
            false_negatives=0,
            instance_precision=0.0,
            instance_recall=0.0,
            instance_f1=0.0,
            mean_instance_iou=0.0,
            proposal_recall=0.0,
            duplicate_rate=0.0,
            fragmentation_rate=0.0,
            merge_rate=0.0,
        )

    # Convert point ID lists to sets
    gt_sets = [set(inst.point_ids) for inst in gt_instances]
    pred_sets = [
        set(p.get("source_point_ids") or p.get("point_ids") or p.get("point_indices", []))
        for p in pred_instances
    ]

    # Compute pairwise IoU matrix [N_pred, N_gt]
    iou_matrix = np.zeros((N_pred, N_gt), dtype=np.float32)
    for p_idx, p_set in enumerate(pred_sets):
        if not p_set:
            continue
        for g_idx, g_set in enumerate(gt_sets):
            intersection = len(p_set & g_set)
            if intersection > 0:
                union = len(p_set | g_set)
                iou_matrix[p_idx, g_idx] = float(intersection / union)

    # 1. Class-agnostic proposal recall: GT instance has at least one proposal with IoU >= 0.30
    gt_proposal_matches = (iou_matrix >= 0.30).sum(axis=0)
    proposal_recall = float(np.sum(gt_proposal_matches > 0) / N_gt)

    # 2. Duplicate rate: GT instance matched by > 1 proposal with IoU >= 0.30
    duplicate_count = int(np.sum(gt_proposal_matches > 1))
    duplicate_rate = float(duplicate_count / N_gt)

    # 3. Merge rate: proposal overlaps significantly (>= 20% points) with >= 2 distinct GT instances
    merge_count = 0
    for p_idx, p_set in enumerate(pred_sets):
        if not p_set:
            continue
        overlapping_gts = 0
        for g_idx, g_set in enumerate(gt_sets):
            overlap = len(p_set & g_set)
            if overlap >= 0.20 * min(len(p_set), len(g_set)) and overlap >= 10:
                overlapping_gts += 1
        if overlapping_gts >= 2:
            merge_count += 1
    merge_rate = float(merge_count / max(N_pred, 1))

    # 4. Fragmentation rate: GT instance whose points are split among >= 2 proposals
    frag_count = 0
    for g_idx, g_set in enumerate(gt_sets):
        contributing_proposals = 0
        for p_idx, p_set in enumerate(pred_sets):
            overlap = len(p_set & g_set)
            if overlap >= 0.15 * len(g_set) and overlap >= 10:
                contributing_proposals += 1
        if contributing_proposals >= 2:
            frag_count += 1
    fragmentation_rate = float(frag_count / N_gt)

    # 5. Greedy matching for Instance Precision, Recall, F1 (IoU >= threshold)
    # Sort match pairs by IoU descending
    pairs: list[tuple[float, int, int]] = []
    for p_idx in range(N_pred):
        for g_idx in range(N_gt):
            if iou_matrix[p_idx, g_idx] >= iou_threshold:
                pairs.append((float(iou_matrix[p_idx, g_idx]), p_idx, g_idx))
    pairs.sort(reverse=True, key=lambda x: x[0])

    matched_preds: set[int] = set()
    matched_gts: set[int] = set()
    matches: list[InstanceMatch] = []
    tp_count = 0

    for iou_val, p_idx, g_idx in pairs:
        if p_idx in matched_preds or g_idx in matched_gts:
            continue

        gt_inst = gt_instances[g_idx]
        pred_inst = pred_instances[p_idx]

        gt_cls = str(gt_inst.semantic_class).upper()
        pred_cls = str(pred_inst.get("semantic_class") or pred_inst.get("semantic_label") or "UNKNOWN").upper()

        # Check semantic class match (UNKNOWN matches UNKNOWN, WALL matches WALL, etc.)
        class_match = (gt_cls == pred_cls)
        is_tp = class_match

        if is_tp:
            tp_count += 1
            matched_preds.add(p_idx)
            matched_gts.add(g_idx)

        matches.append(
            InstanceMatch(
                gt_object_id=gt_inst.object_id,
                pred_object_id=pred_inst.get("object_id", f"pred_{p_idx}"),
                gt_class=gt_cls,
                pred_class=pred_cls,
                iou=round(iou_val, 4),
                is_class_match=class_match,
                is_true_positive=is_tp,
            )
        )

    fp_count = N_pred - tp_count
    fn_count = N_gt - tp_count

    precision = float(tp_count / (tp_count + fp_count)) if (tp_count + fp_count) > 0 else 0.0
    recall = float(tp_count / (tp_count + fn_count)) if (tp_count + fn_count) > 0 else 0.0
    f1 = float(2.0 * precision * recall / (precision + recall)) if (precision + recall) > 0 else 0.0
    mean_iou = float(np.mean([m.iou for m in matches])) if matches else 0.0

    return InstanceEvaluationMetrics(
        split_name=split_name,
        num_gt_instances=N_gt,
        num_pred_instances=N_pred,
        iou_threshold=iou_threshold,
        true_positives=tp_count,
        false_positives=fp_count,
        false_negatives=fn_count,
        instance_precision=round(precision, 4),
        instance_recall=round(recall, 4),
        instance_f1=round(f1, 4),
        mean_instance_iou=round(mean_iou, 4),
        proposal_recall=round(proposal_recall, 4),
        duplicate_rate=round(duplicate_rate, 4),
        fragmentation_rate=round(fragmentation_rate, 4),
        merge_rate=round(merge_rate, 4),
        matches=matches,
    )
