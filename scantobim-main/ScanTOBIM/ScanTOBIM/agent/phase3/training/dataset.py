"""PyTorch Dataset and Data Loading for Multi-Class RandLA-Net Training.

Loads ground-truth spatial partitions, extracts scale-aware 14D geometric features,
applies deterministic augmentations, and provides class imbalance weighting.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch.utils.data import Dataset

from agent.phase3.annotation.annotator import PHASE_3C_LABEL_SPACE
from agent.phase3.geometry import FEATURE_SCHEMA_14D, compute_geometric_features
from agent.phase3.training.augment import PointCloudAugmentor


class Phase3CDataset(Dataset):
    """Dataset for training/evaluating multi-class point cloud neural networks on CPU."""

    def __init__(
        self,
        npz_paths: list[Path | str],
        block_size: int = 2048,
        num_blocks: int = 100,
        augmentor: PointCloudAugmentor | None = None,
        precompute_features: bool = True,
        seed: int = 42,
    ) -> None:
        self.block_size = block_size
        self.num_blocks = num_blocks
        self.augmentor = augmentor
        self.rng = np.random.RandomState(seed)

        # 1. Ingest all specified NPZ partitions
        all_pts: list[np.ndarray] = []
        all_ids: list[np.ndarray] = []
        all_classes: list[np.ndarray] = []

        for p in npz_paths:
            path = Path(p)
            assert path.exists(), f"Partition not found: {path}"
            data = np.load(path)
            all_pts.append(data["points"])
            all_ids.append(data["point_ids"])
            all_classes.append(data["class_ids"])

        self.points = np.vstack(all_pts).astype(np.float32)
        self.point_ids = np.concatenate(all_ids).astype(np.int64)
        self.class_ids = np.concatenate(all_classes).astype(np.int64)
        self.total_points = len(self.points)

        # 2. Compute scale-aware 14D features
        if precompute_features and self.total_points > 0:
            feat_set = compute_geometric_features(self.points)
            self.features = feat_set.to_feature_matrix(self.points)
        else:
            self.features = np.zeros((self.total_points, len(FEATURE_SCHEMA_14D)), dtype=np.float32)

        # 3. Compute class weights (inverse class frequency)
        self.class_weights = self._compute_class_weights()

    def _compute_class_weights(self) -> torch.Tensor:
        """Compute smoothed inverse-frequency class weights for cross-entropy."""
        num_classes = len(PHASE_3C_LABEL_SPACE)
        counts = np.bincount(self.class_ids, minlength=num_classes)
        # Smooth with sqrt and median
        total = float(np.sum(counts))
        weights = np.zeros(num_classes, dtype=np.float32)
        for c in range(num_classes):
            if counts[c] > 0:
                weights[c] = float(total / (num_classes * counts[c]))
            else:
                weights[c] = 0.0

        # Normalise and clamp extreme weights
        non_zero = weights[weights > 0]
        if len(non_zero) > 0:
            med = float(np.median(non_zero))
            weights = np.clip(weights, 0.2 * med, 15.0 * med)
            weights /= np.mean(weights[weights > 0])
        else:
            weights = np.ones(num_classes, dtype=np.float32)

        return torch.from_numpy(weights).float()

    def __len__(self) -> int:
        return self.num_blocks

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Sample a local spatial block of points.

        Returns:
            points: (block_size, 3) float tensor
            features: (14, block_size) float tensor
            labels: (block_size,) long tensor
        """
        N = self.total_points
        if N <= self.block_size:
            sample_idx = np.arange(N)
            if N < self.block_size:
                pad = self.rng.choice(N, self.block_size - N, replace=True)
                sample_idx = np.concatenate([sample_idx, pad])
        else:
            # Pick a center point and sample nearest neighbors
            center_idx = self.rng.randint(0, N)
            center = self.points[center_idx]
            dist_sq = np.sum((self.points - center) ** 2, axis=1)
            sample_idx = np.argpartition(dist_sq, self.block_size)[:self.block_size]

        block_pts = self.points[sample_idx]
        block_feats = self.features[sample_idx]
        block_lbls = self.class_ids[sample_idx]

        # Apply augmentation if enabled
        if self.augmentor is not None:
            block_pts, block_feats = self.augmentor(block_pts, block_feats)

        # Transpose features to (C, N) expected by 1D Conv in RandLANet
        pts_t = torch.from_numpy(block_pts).float()
        feats_t = torch.from_numpy(block_feats.T).float()
        lbls_t = torch.from_numpy(block_lbls).long()

        return pts_t, feats_t, lbls_t
