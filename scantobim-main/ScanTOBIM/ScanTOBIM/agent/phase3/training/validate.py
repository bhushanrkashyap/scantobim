"""Evaluation and Validation Loop for Multi-Class Semantic Segmentation."""

from __future__ import annotations

import time
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F

from agent.phase3.annotation.annotator import PHASE_3C_LABEL_SPACE
from agent.phase3.geometry import FEATURE_SCHEMA_14D, compute_geometric_features
from agent.phase3.training.metrics import (
    SemanticEvaluationMetrics,
    compute_semantic_metrics,
)


def evaluate_model_on_point_cloud(
    model: torch.nn.Module,
    points: np.ndarray,
    ground_truth_class_ids: np.ndarray,
    features: np.ndarray | None = None,
    block_size: int = 4096,
    split_name: str = "VALIDATION",
    label_space: list[str] = PHASE_3C_LABEL_SPACE,
) -> tuple[SemanticEvaluationMetrics, np.ndarray, np.ndarray]:
    """Run full-scene CPU evaluation using spatial blocks and compute metrics against ground truth.

    Returns:
        metrics: SemanticEvaluationMetrics
        pred_class_ids: (N,) predicted integer classes
        pred_probs: (N, C) predicted probabilities
    """
    model.eval()
    device = torch.device("cpu")
    model.to(device)

    N = len(points)
    num_classes = len(label_space)

    if features is None:
        feat_set = compute_geometric_features(points)
        features = feat_set.to_feature_matrix(points)

    pred_logits_accum = np.zeros((N, num_classes), dtype=np.float32)
    pred_counts = np.zeros(N, dtype=np.int32)

    # Spatial chunking: stride over points
    step = block_size // 2
    for start in range(0, N, max(1, step)):
        end = min(start + block_size, N)
        if end - start < 16:
            continue

        b_pts = points[start:end]
        b_feats = features[start:end]

        pts_t = torch.from_numpy(b_pts).unsqueeze(0).float().to(device)  # (1, B, 3)
        feats_t = torch.from_numpy(b_feats.T).unsqueeze(0).float().to(device)  # (1, 14, B)

        with torch.no_grad():
            # RandLANet forward: points (1, B, 3), features (1, C, B) -> logits (1, num_classes, B)
            logits = model(pts_t, feats_t)
            probs = F.softmax(logits, dim=1).squeeze(0).cpu().numpy().T  # (B, num_classes)

        pred_logits_accum[start:end] += probs
        pred_counts[start:end] += 1

    # Average probabilities
    valid_mask = pred_counts > 0
    pred_logits_accum[valid_mask] /= pred_counts[valid_mask, None]
    # For any unvisited points, default to uniform
    pred_logits_accum[~valid_mask] = 1.0 / num_classes

    pred_classes = np.argmax(pred_logits_accum, axis=1)

    metrics = compute_semantic_metrics(
        y_true=ground_truth_class_ids,
        y_pred=pred_classes,
        split_name=split_name,
        label_space=label_space,
    )

    return metrics, pred_classes, pred_logits_accum
