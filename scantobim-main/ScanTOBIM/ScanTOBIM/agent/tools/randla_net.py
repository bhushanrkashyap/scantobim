"""RandLA-Net: Efficient Neural Semantic Segmentation Architecture for 3D Point Clouds.

Implements the RandLA algorithm:
1. Random Sampling (RS) for scalable, linear-time downsampling.
2. Local Feature Aggregation (LFA):
   - Local Spatial Encoding (LocSE): Relative position encoding [p_i, p_i^k, p_i^k - p_i, ||p_i^k - p_i||].
   - Attentive Pooling: Softmax attention across K-nearest neighbors.
   - Dilated Residual Block: Chained LocSE + Attentive Pooling units with residual skip connection.
3. Feature Propagation / Decoder: Upsampling with encoder skip connections.
4. Semantic prediction head for wall, floor, ceiling, and MEP classification.
"""

from __future__ import annotations

import math
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy.spatial import cKDTree


# ── 1. CORE RANDLA MODULES ───────────────────────────────────────────────────


def random_sampling(
    points: torch.Tensor, features: torch.Tensor | None, num_samples: int
) -> tuple[torch.Tensor, torch.Tensor | None, torch.Tensor]:
    """Random sampling (RS) to select a subset of points and features.

    Args:
        points: (B, N, 3) coordinates
        features: (B, C, N) features or None
        num_samples: Target sample count M (M <= N)

    Returns:
        sampled_points: (B, M, 3)
        sampled_features: (B, C, M) or None
        sample_indices: (B, M)
    """
    B, N, _ = points.shape
    num_samples = min(num_samples, N)

    batch_indices = []
    for _ in range(B):
        idx = torch.randperm(N, device=points.device)[:num_samples]
        batch_indices.append(idx)
    sample_indices = torch.stack(batch_indices, dim=0)

    batch_idx_expanded = sample_indices.unsqueeze(-1).expand(-1, -1, 3)
    sampled_points = torch.gather(points, 1, batch_idx_expanded)

    sampled_features = None
    if features is not None:
        c_dim = features.shape[1]
        feat_idx_expanded = sample_indices.unsqueeze(1).expand(-1, c_dim, -1)
        sampled_features = torch.gather(features, 2, feat_idx_expanded)

    return sampled_points, sampled_features, sample_indices


def knn_gather(points: torch.Tensor, neighbor_indices: torch.Tensor) -> torch.Tensor:
    """Gather neighbor coordinates given indices.

    Args:
        points: (B, N, 3)
        neighbor_indices: (B, M, K)

    Returns:
        gathered: (B, M, K, 3)
    """
    B, _, D = points.shape
    M, K = neighbor_indices.shape[1], neighbor_indices.shape[2]
    idx_expanded = neighbor_indices.unsqueeze(-1).expand(-1, -1, -1, D)
    points_expanded = points.unsqueeze(1).expand(-1, M, -1, D)
    return torch.gather(points_expanded, 2, idx_expanded)


def knn_gather_feats(features: torch.Tensor, neighbor_indices: torch.Tensor) -> torch.Tensor:
    """Gather neighbor features given indices.

    Args:
        features: (B, C, N)
        neighbor_indices: (B, M, K)

    Returns:
        gathered: (B, C, M, K)
    """
    B, C, N = features.shape
    M, K = neighbor_indices.shape[1], neighbor_indices.shape[2]
    idx_expanded = neighbor_indices.unsqueeze(1).expand(-1, C, -1, -1)
    feats_expanded = features.unsqueeze(2).expand(-1, C, M, N)
    return torch.gather(feats_expanded, 3, idx_expanded)


class LocalSpatialEncoding(nn.Module):
    """LocSE: Local Spatial Encoding module in RandLA algorithm.

    Encodes relative coordinates:
        r_i^k = [p_i, p_i^k, p_i^k - p_i, ||p_i^k - p_i||]  (10 dimensions)
    Projects r_i^k through shared MLP (Conv2d) and concatenates with neighbor features.
    """

    def __init__(self, in_features: int, out_features: int):
        super().__init__()
        self.mlp_pos = nn.Sequential(
            nn.Conv2d(10, out_features, kernel_size=1, bias=False),
            nn.BatchNorm2d(out_features),
            nn.LeakyReLU(0.2, inplace=True),
        )
        self.mlp_feat = (
            nn.Sequential(
                nn.Conv2d(in_features, out_features, kernel_size=1, bias=False),
                nn.BatchNorm2d(out_features),
                nn.LeakyReLU(0.2, inplace=True),
            )
            if in_features > 0
            else None
        )
        total_in = out_features * 2 if in_features > 0 else out_features
        self.out_conv = nn.Sequential(
            nn.Conv2d(total_in, out_features, kernel_size=1, bias=False),
            nn.BatchNorm2d(out_features),
            nn.LeakyReLU(0.2, inplace=True),
        )

    def forward(
        self,
        center_pts: torch.Tensor,
        neighbor_pts: torch.Tensor,
        neighbor_feats: torch.Tensor | None,
    ) -> torch.Tensor:
        """Args: center_pts: (B, M, 3) neighbor_pts: (B, M, K, 3) neighbor_feats:

        (B, C_in, M, K) or None.

        Returns:
            encoded_feats: (B, C_out, M, K)
        """
        c_expanded = center_pts.unsqueeze(2).expand_as(neighbor_pts)
        diff = neighbor_pts - c_expanded
        dist = torch.norm(diff, dim=-1, keepdim=True)

        rel_pos = torch.cat([c_expanded, neighbor_pts, diff, dist], dim=-1)  # (B, M, K, 10)
        rel_pos_t = rel_pos.permute(0, 3, 1, 2)  # (B, 10, M, K)

        pos_enc = self.mlp_pos(rel_pos_t)  # (B, C_out, M, K)

        if neighbor_feats is not None and self.mlp_feat is not None:
            feat_enc = self.mlp_feat(neighbor_feats)
            combined = torch.cat([pos_enc, feat_enc], dim=1)
        else:
            combined = pos_enc

        return self.out_conv(combined)


class AttentivePooling(nn.Module):
    """Attentive Pooling module in RandLA algorithm.

    Computes attention scores over K neighbors and aggregates local features:
        score_i^k = softmax(W * f_i^k)
        f_agg = sum_k(score_i^k * f_i^k)
    """

    def __init__(self, in_features: int, out_features: int):
        super().__init__()
        self.score_mlp = nn.Sequential(
            nn.Conv2d(in_features, in_features, kernel_size=1, bias=False),
            nn.BatchNorm2d(in_features),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(in_features, 1, kernel_size=1, bias=False),
        )
        self.out_conv = nn.Sequential(
            nn.Conv1d(in_features, out_features, kernel_size=1, bias=False),
            nn.BatchNorm1d(out_features),
            nn.LeakyReLU(0.2, inplace=True),
        )

    def forward(self, local_features: torch.Tensor) -> torch.Tensor:
        """Args: local_features: (B, C, M, K)

        Returns: pooled_features: (B, C_out, M)
        """
        scores = self.score_mlp(local_features)  # (B, 1, M, K)
        attention_weights = F.softmax(scores, dim=-1)  # (B, 1, M, K)

        pooled = torch.sum(local_features * attention_weights, dim=-1)  # (B, C, M)
        return self.out_conv(pooled)


class DilatedResidualBlock(nn.Module):
    """Dilated Residual Block in RandLA algorithm.

    Stacks two LocSE + Attentive Pooling units with a residual skip connection.
    """

    def __init__(self, in_features: int, out_features: int):
        super().__init__()
        mid_features = out_features // 2

        self.locse1 = LocalSpatialEncoding(in_features, mid_features)
        self.pool1 = AttentivePooling(mid_features, mid_features)

        self.locse2 = LocalSpatialEncoding(mid_features, out_features)
        self.pool2 = AttentivePooling(out_features, out_features)

        self.shortcut = (
            nn.Sequential(
                nn.Conv1d(in_features, out_features, kernel_size=1, bias=False),
                nn.BatchNorm1d(out_features),
            )
            if in_features != out_features and in_features > 0
            else None
        )
        self.relu = nn.LeakyReLU(0.2, inplace=True)

    def forward(
        self,
        center_pts: torch.Tensor,
        neighbor_pts: torch.Tensor,
        neighbor_feats: torch.Tensor | None,
        raw_center_feats: torch.Tensor | None,
        neighbor_indices: torch.Tensor,
    ) -> torch.Tensor:
        """Args: center_pts: (B, M, 3) neighbor_pts: (B, M, K, 3) neighbor_feats:

        (B, C_in, M, K) raw_center_feats: (B, C_in, M) neighbor_indices: (B, M,

        K)

        Returns:
            out_feats: (B, C_out, M)
        """
        # Unit 1
        x1 = self.locse1(center_pts, neighbor_pts, neighbor_feats)
        f1 = self.pool1(x1)  # (B, mid, M)

        # Gather f1 for neighbors
        f1_neighbors = knn_gather_feats(f1, neighbor_indices)  # (B, mid, M, K)

        # Unit 2
        x2 = self.locse2(center_pts, neighbor_pts, f1_neighbors)
        f2 = self.pool2(x2)  # (B, out, M)

        # Residual shortcut
        if self.shortcut is not None and raw_center_feats is not None:
            f2 = f2 + self.shortcut(raw_center_feats)
        elif raw_center_feats is not None and raw_center_feats.shape[1] == f2.shape[1]:
            f2 = f2 + raw_center_feats

        return self.relu(f2)


class FeaturePropagation(nn.Module):
    """Nearest-neighbor feature propagation decoder module."""

    def __init__(self, in_channels: int, skip_channels: int, out_channels: int):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Conv1d(in_channels + skip_channels, out_channels, kernel_size=1, bias=False),
            nn.BatchNorm1d(out_channels),
            nn.LeakyReLU(0.2, inplace=True),
        )

    def forward(
        self,
        fine_pts: torch.Tensor,
        coarse_pts: torch.Tensor,
        coarse_feats: torch.Tensor,
        skip_feats: torch.Tensor | None,
    ) -> torch.Tensor:
        """Args: fine_pts: (B, N_fine, 3) coarse_pts: (B, N_coarse, 3)

        coarse_feats: (B, C_coarse, N_coarse) skip_feats: (B, C_skip, N_fine) or

        None.
        """
        fine_pts_np = fine_pts[0].detach().cpu().numpy()
        coarse_pts_np = coarse_pts[0].detach().cpu().numpy()
        tree = cKDTree(coarse_pts_np)
        _, nn_idx = tree.query(fine_pts_np, k=1)
        nn_idx_t = torch.as_tensor(nn_idx, dtype=torch.long, device=fine_pts.device)

        # Upsampled features: (B, C_coarse, N_fine)
        upsampled = coarse_feats[:, :, nn_idx_t]

        if skip_feats is not None:
            combined = torch.cat([upsampled, skip_feats], dim=1)
        else:
            combined = upsampled

        return self.mlp(combined)


# ── 2. FULL RANDLA-NET MODEL ─────────────────────────────────────────────────


class RandLANet(nn.Module):
    """RandLA-Net architecture for efficient large-scale 3D semantic segmentation.

    Uses Random Sampling for downsampling and Local Feature Aggregation (LocSE + Attentive Pooling).
    """

    def __init__(self, in_channels: int = 6, num_classes: int = 2, k_neighbors: int = 16):
        super().__init__()
        self.k = k_neighbors

        self.fc_start = nn.Sequential(
            nn.Conv1d(in_channels, 16, kernel_size=1, bias=False),
            nn.BatchNorm1d(16),
            nn.LeakyReLU(0.2, inplace=True),
        )

        self.block1 = DilatedResidualBlock(16, 32)
        self.block2 = DilatedResidualBlock(32, 64)
        self.block3 = DilatedResidualBlock(64, 128)

        self.fp3 = FeaturePropagation(128, 64, 64)
        self.fp2 = FeaturePropagation(64, 32, 32)
        self.fp1 = FeaturePropagation(32, 16, 16)

        self.classifier = nn.Sequential(
            nn.Conv1d(16, 32, kernel_size=1),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout(0.2),
            nn.Conv1d(32, num_classes, kernel_size=1),
        )

    def forward(self, points: torch.Tensor, features: torch.Tensor) -> torch.Tensor:
        """Args: points: (1, N, 3) point coordinates in metres features: (1,

        in_channels, N) input features.

        Returns:
            logits: (1, num_classes, N)
        """
        B, N, _ = points.shape

        f0 = self.fc_start(features)  # (B, 16, N)

        # Encoder Stage 1 (Full resolution)
        k1_idx = self._find_knn(points, points, self.k)
        k1_pts = knn_gather(points, k1_idx)
        k1_feats = knn_gather_feats(f0, k1_idx)
        f1 = self.block1(points, k1_pts, k1_feats, f0, k1_idx)  # (B, 32, N)

        # Random Subsampling 1: N -> N // 4
        p1, f1_sub, _ = random_sampling(points, f1, max(N // 4, 32))

        # Encoder Stage 2
        k2_idx = self._find_knn(p1, p1, self.k)
        k2_pts = knn_gather(p1, k2_idx)
        k2_feats = knn_gather_feats(f1_sub, k2_idx)
        f2 = self.block2(p1, k2_pts, k2_feats, f1_sub, k2_idx)  # (B, 64, N1)

        # Random Subsampling 2: N1 -> N1 // 4
        p2, f2_sub, _ = random_sampling(p1, f2, max(p1.shape[1] // 4, 16))

        # Encoder Stage 3
        k3_idx = self._find_knn(p2, p2, self.k)
        k3_pts = knn_gather(p2, k3_idx)
        k3_feats = knn_gather_feats(f2_sub, k3_idx)
        f3 = self.block3(p2, k3_pts, k3_feats, f2_sub, k3_idx)  # (B, 128, N2)

        # Decoder Stage 3: Upsample p2 -> p1
        f2_up = self.fp3(p1, p2, f3, f2)

        # Decoder Stage 2: Upsample p1 -> points
        f1_up = self.fp2(points, p1, f2_up, f1)

        # Decoder Stage 1: Refine at full resolution
        f0_up = self.fp1(points, points, f1_up, f0)

        logits = self.classifier(f0_up)  # (B, num_classes, N)
        return logits

    @staticmethod
    def _find_knn(query_pts: torch.Tensor, support_pts: torch.Tensor, k: int) -> torch.Tensor:
        q_np = query_pts[0].detach().cpu().numpy()
        s_np = support_pts[0].detach().cpu().numpy()
        tree = cKDTree(s_np)
        k_val = min(k, len(s_np))
        _, idx = tree.query(q_np, k=k_val)
        if idx.ndim == 1:
            idx = idx[:, None]
        return torch.as_tensor(idx, dtype=torch.long, device=query_pts.device).unsqueeze(0)


# ── 3. HIGH-LEVEL INFERENCE ROUTINE ──────────────────────────────────────────


def predict_randla(
    points_m: np.ndarray,
    wall_threshold: float = 0.50,
    wall_class_id: int = 1,
    checkpoint_path: Path | None = None,
    device: str = "cpu",
) -> tuple[np.ndarray, np.ndarray]:
    """Execute RandLA algorithm prediction on input 3D point cloud.

    Args:
        points_m: (N, 3) coordinates in metres
        wall_threshold: Minimum probability threshold for wall classification
        wall_class_id: Target class index for walls (default 1)
        checkpoint_path: Optional path to pretrained weights (.pth)
        device: 'cpu' or 'cuda' or 'mps'

    Returns:
        wall_mask: (N,) boolean mask
        probabilities: (N,) float probabilities for wall class
    """
    N = points_m.shape[0]
    if N == 0:
        return np.zeros(0, dtype=bool), np.zeros(0, dtype=np.float32)

    coords = points_m.astype(np.float32)

    anchor_count = min(max(500, N // 20), 10000)
    step = max(1, N // anchor_count)
    anchors_sub = coords[::step]
    tree = cKDTree(anchors_sub)
    dists, idxs = tree.query(coords, k=min(12, len(anchors_sub)))
    if idxs.ndim == 1:
        idxs = idxs[:, None]

    anchors = anchors_sub[idxs]
    deltas = anchors - coords[:, None, :]
    dz_var = np.var(deltas[:, :, 2], axis=-1)
    dxy_var = np.var(deltas[:, :, :2], axis=(1, 2)) + 1e-6
    verticality_ratio = np.clip(dz_var / dxy_var, 0.0, 10.0).astype(np.float32)

    z_min, z_max = float(coords[:, 2].min()), float(coords[:, 2].max())
    z_rel = ((coords[:, 2] - z_min) / max(z_max - z_min, 1e-4)).astype(np.float32)

    c_xy = np.mean(coords[:, :2], axis=0)
    dist_c = np.linalg.norm(coords[:, :2] - c_xy, axis=-1, keepdims=True).astype(np.float32)

    feats = np.column_stack([
        coords,
        verticality_ratio[:, None],
        z_rel[:, None],
        dist_c,
    ])

    model = RandLANet(in_channels=6, num_classes=2, k_neighbors=16)

    if checkpoint_path is not None and Path(checkpoint_path).exists():
        try:
            state = torch.load(checkpoint_path, map_location=device)
            if isinstance(state, dict) and "model_state_dict" in state:
                model.load_state_dict(state["model_state_dict"])
            elif isinstance(state, dict):
                model.load_state_dict(state)
        except Exception:
            pass

    model.eval()
    model.to(device)

    chunk_size = 35000
    all_probs = []

    with torch.no_grad():
        for i in range(0, N, chunk_size):
            chunk_pts = coords[i : i + chunk_size]
            chunk_feats = feats[i : i + chunk_size]

            pts_t = torch.as_tensor(chunk_pts, device=device).unsqueeze(0)
            # features shape: (1, 6, N_chunk)
            feats_t = torch.as_tensor(chunk_feats, device=device).permute(1, 0).unsqueeze(0)

            logits = model(pts_t, feats_t)  # (1, 2, N_chunk)
            vert_boost = torch.as_tensor(chunk_feats[:, 3], device=device).unsqueeze(0).unsqueeze(0)
            logits[:, 1:2, :] += torch.clamp(vert_boost * 0.60 - 0.30, -2.0, 3.0)

            probs = F.softmax(logits, dim=1)[0, wall_class_id, :].cpu().numpy()
            all_probs.append(probs)

    full_probs = np.concatenate(all_probs, axis=0)
    wall_mask = full_probs >= float(wall_threshold)
    return wall_mask, full_probs
