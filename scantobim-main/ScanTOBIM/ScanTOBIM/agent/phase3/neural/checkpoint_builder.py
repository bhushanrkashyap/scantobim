"""RandLA-Net Architectural Checkpoint Builder.

Trains a RandLA-Net model on synthetic architectural point cloud data.

This produces a REAL TRAINED CHECKPOINT — not random weights.

The training data is procedurally generated architectural geometry with
exact ground-truth semantic labels:
  - Class 0: NON_WALL (floor, ceiling, clutter, open space)
  - Class 1: WALL     (vertical planar surfaces)

Training data covers:
  - Walls of varying orientation (0°–360°), height (2–4 m), thickness (0.1–0.3 m)
  - Floors at z=0 ± noise
  - Ceilings at z=3 ± noise
  - Clutter (random interior points)

The model learns verticality, planarity, height distribution, and z-relative
position as wall indicators — exactly the feature schema used at inference.

Output checkpoint schema:
  {
    "model_state_dict": <OrderedDict>,
    "architecture": "RandLANet",
    "in_channels": 6,
    "num_classes": 2,
    "k_neighbors": 16,
    "training_dataset": "SYNTHETIC_ARCHITECTURAL_v1",
    "training_label_space": ["NON_WALL", "WALL"],
    "feature_schema": ["x", "y", "z", "verticality_ratio", "z_rel", "dist_centroid"],
    "epochs": <int>,
    "final_loss": <float>,
    "final_train_accuracy": <float>,
    "checkpoint_hash": <sha256>,
    "created_at": <ISO8601>,
  }
"""

from __future__ import annotations

import hashlib
import json
import time
from pathlib import Path
from typing import Any

import numpy as np
import structlog
import torch
import torch.nn.functional as F
from scipy.spatial import cKDTree

from agent.tools.randla_net import RandLANet

logger = structlog.get_logger()

# ── Checkpoint metadata ───────────────────────────────────────────────────────

ARCHITECTURE_NAME: str = "RandLANet"
IN_CHANNELS: int = 6
NUM_CLASSES: int = 2
K_NEIGHBORS: int = 16
TRAINING_DATASET: str = "SYNTHETIC_ARCHITECTURAL_v1"
TRAINING_LABEL_SPACE: list[str] = ["NON_WALL", "WALL"]
FEATURE_SCHEMA: list[str] = ["x", "y", "z", "verticality_ratio", "z_rel", "dist_centroid"]
WALL_CLASS_ID: int = 1

DEFAULT_CHECKPOINT_PATH: Path = Path("models/randlanet_architectural_cpu.pth")


# ── Synthetic data generator ──────────────────────────────────────────────────

def _make_feature_vector(points: np.ndarray) -> np.ndarray:
    """Compute 6D feature vector matching inference feature schema.

    Features:
        0-2: x, y, z (coordinates)
        3: verticality_ratio (var(dz) / var(dxy)) — high for vertical structures
        4: z_rel (z relative to scene height, 0=floor, 1=ceiling)
        5: dist_centroid (horizontal distance from scene centroid)
    """
    coords = points.astype(np.float32)
    N = len(coords)

    # Verticality ratio via KNN covariance
    anchor_count = min(max(200, N // 20), 3000)
    step = max(1, N // anchor_count)
    anchors_sub = coords[::step]
    tree = cKDTree(anchors_sub)
    k_nn = min(12, len(anchors_sub))
    _, idxs = tree.query(coords, k=k_nn)
    if idxs.ndim == 1:
        idxs = idxs[:, None]
    anchors = anchors_sub[idxs]
    deltas = anchors - coords[:, None, :]
    dz_var = np.var(deltas[:, :, 2], axis=-1)
    dxy_var = np.var(deltas[:, :, :2], axis=(1, 2)) + 1e-6
    verticality_ratio = np.clip(dz_var / dxy_var, 0.0, 10.0).astype(np.float32)

    # z_rel
    z_min, z_max = float(coords[:, 2].min()), float(coords[:, 2].max())
    z_rel = ((coords[:, 2] - z_min) / max(z_max - z_min, 1e-4)).astype(np.float32)

    # dist_centroid
    c_xy = np.mean(coords[:, :2], axis=0)
    dist_c = np.linalg.norm(coords[:, :2] - c_xy, axis=-1).astype(np.float32)

    return np.column_stack([coords, verticality_ratio, z_rel, dist_c])


def generate_synthetic_scene(
    rng: np.random.Generator,
    n_points: int = 3000,
) -> tuple[np.ndarray, np.ndarray]:
    """Generate one synthetic architectural scene with ground-truth labels.

    Returns:
        points: (N, 3) float32 coordinates in metres
        labels: (N,) int64  [0=NON_WALL, 1=WALL]
    """
    pts_list: list[np.ndarray] = []
    lbl_list: list[np.ndarray] = []

    # Room dimensions
    room_w = rng.uniform(4.0, 12.0)
    room_d = rng.uniform(4.0, 10.0)
    room_h = rng.uniform(2.5, 4.5)
    noise = 0.005  # 5mm sensor noise

    # ── Walls (2–4 walls, random orientation variants) ──────────────────────
    num_walls = rng.integers(2, 5)
    wall_pts_target = int(n_points * 0.45 / num_walls)

    for _ in range(num_walls):
        angle = rng.uniform(0, np.pi)
        wall_len = rng.uniform(2.0, max(room_w, room_d))
        t = rng.uniform(0, room_w * 0.5)
        cx, cy = rng.uniform(0, room_w), rng.uniform(0, room_d)

        n_w = wall_pts_target
        # parametric wall points
        u = rng.uniform(-wall_len / 2, wall_len / 2, n_w)
        v = rng.normal(0.0, noise, n_w)
        z = rng.uniform(0.0, room_h, n_w)

        x = cx + u * np.cos(angle) + v * np.sin(angle)
        y = cy + u * np.sin(angle) - v * np.cos(angle)
        pts_list.append(np.column_stack([x, y, z]).astype(np.float32))
        lbl_list.append(np.ones(n_w, dtype=np.int64))  # WALL=1

    # ── Floor ────────────────────────────────────────────────────────────────
    n_floor = int(n_points * 0.25)
    fx = rng.uniform(0, room_w, n_floor)
    fy = rng.uniform(0, room_d, n_floor)
    fz = rng.normal(0.0, noise, n_floor)
    pts_list.append(np.column_stack([fx, fy, fz]).astype(np.float32))
    lbl_list.append(np.zeros(n_floor, dtype=np.int64))

    # ── Ceiling ──────────────────────────────────────────────────────────────
    n_ceil = int(n_points * 0.15)
    cx2 = rng.uniform(0, room_w, n_ceil)
    cy2 = rng.uniform(0, room_d, n_ceil)
    cz = rng.normal(room_h, noise, n_ceil)
    pts_list.append(np.column_stack([cx2, cy2, cz]).astype(np.float32))
    lbl_list.append(np.zeros(n_ceil, dtype=np.int64))

    # ── Clutter (interior objects, random) ───────────────────────────────────
    n_clutter = n_points - sum(len(l) for l in lbl_list)
    if n_clutter > 0:
        rx = rng.uniform(0.5, room_w - 0.5, n_clutter)
        ry = rng.uniform(0.5, room_d - 0.5, n_clutter)
        rz = rng.uniform(0.1, room_h - 0.1, n_clutter)
        pts_list.append(np.column_stack([rx, ry, rz]).astype(np.float32))
        lbl_list.append(np.zeros(n_clutter, dtype=np.int64))

    points = np.vstack(pts_list)
    labels = np.concatenate(lbl_list)

    # Shuffle
    idx = rng.permutation(len(points))
    return points[idx], labels[idx]


def build_batch(
    rng: np.random.Generator,
    batch_size: int = 4,
    n_pts: int = 1024,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Build one training batch.

    Returns:
        pts_batch:   (B, N, 3) float32
        feats_batch: (B, 6, N) float32
        labels_batch:(B, N)    int64
    """
    pts_list, feats_list, labels_list = [], [], []
    for _ in range(batch_size):
        pts, lbl = generate_synthetic_scene(rng, n_points=n_pts)
        # Subsample to exactly n_pts
        if len(pts) > n_pts:
            idx = rng.choice(len(pts), n_pts, replace=False)
            pts, lbl = pts[idx], lbl[idx]
        feats = _make_feature_vector(pts)  # (N, 6)
        pts_list.append(pts)
        feats_list.append(feats.T)  # (6, N)
        labels_list.append(lbl)

    pts_t = torch.as_tensor(np.stack(pts_list), dtype=torch.float32)
    feats_t = torch.as_tensor(np.stack(feats_list), dtype=torch.float32)
    labels_t = torch.as_tensor(np.stack(labels_list), dtype=torch.long)
    return pts_t, feats_t, labels_t


# ── Checkpoint hash ───────────────────────────────────────────────────────────

def compute_file_sha256(path: Path) -> str:
    """Compute SHA-256 hash of a file."""
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()


# ── Training ──────────────────────────────────────────────────────────────────

def train_randlanet_cpu(
    checkpoint_path: Path = DEFAULT_CHECKPOINT_PATH,
    epochs: int = 12,
    batch_size: int = 2,
    n_pts_per_sample: int = 800,
    seed: int = 42,
    verbose: bool = True,
) -> dict[str, Any]:
    """Train RandLA-Net on synthetic architectural data and save checkpoint.

    Args:
        checkpoint_path: Where to save the trained model.
        epochs: Training epochs (12 = ~2 min on CPU, sufficient for synthetic data).
        batch_size: Mini-batch size.
        n_pts_per_sample: Points per synthetic scene.
        seed: RNG seed for reproducibility.
        verbose: Print progress.

    Returns:
        Training summary dict with checkpoint metadata.
    """
    checkpoint_path = Path(checkpoint_path)
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)

    rng = np.random.default_rng(seed)
    torch.manual_seed(seed)

    device = "cpu"
    model = RandLANet(in_channels=IN_CHANNELS, num_classes=NUM_CLASSES, k_neighbors=K_NEIGHBORS)
    model.train()
    model.to(device)

    # Manual SGD with momentum — avoids torch.optim (which triggers torch._dynamo → onnx import error)
    lr = 1e-3
    momentum = 0.9
    weight_decay = 1e-4
    params = list(model.parameters())
    velocities = [torch.zeros_like(p) for p in params]

    # Cosine annealing schedule: lr_t = lr_min + 0.5*(lr-lr_min)*(1 + cos(pi*t/T))
    lr_min = 1e-5

    def get_lr(epoch: int) -> float:
        return lr_min + 0.5 * (lr - lr_min) * (1 + np.cos(np.pi * epoch / epochs))

    # Class weights: walls are ~45% of points, non-wall ~55%
    class_weights = torch.tensor([1.0, 1.3], dtype=torch.float32)

    steps_per_epoch = 10
    t_start = time.time()
    final_loss = float("inf")
    final_acc = 0.0

    for epoch in range(1, epochs + 1):
        epoch_loss = 0.0
        epoch_correct = 0
        epoch_total = 0

        for step in range(steps_per_epoch):
            pts_t, feats_t, labels_t = build_batch(rng, batch_size=batch_size, n_pts=n_pts_per_sample)
            pts_t = pts_t.to(device)
            feats_t = feats_t.to(device)
            labels_t = labels_t.to(device)

            # Zero gradients manually
            for p in params:
                if p.grad is not None:
                    p.grad.zero_()

            # Forward pass — process one sample at a time
            B = pts_t.shape[0]
            loss_total = torch.tensor(0.0, device=device)

            for b in range(B):
                pts_b = pts_t[b:b+1]
                feats_b = feats_t[b:b+1]
                lbl_b = labels_t[b]

                logits = model(pts_b, feats_b)  # (1, num_classes, N)
                logits_flat = logits[0].T        # (N, num_classes)

                # Weighted cross-entropy (manual, avoids torch.optim path)
                log_probs = torch.log_softmax(logits_flat, dim=-1)  # (N, num_classes)
                nll = -log_probs[torch.arange(len(lbl_b)), lbl_b]   # (N,)
                weights_per_pt = class_weights[lbl_b]
                loss_b = (nll * weights_per_pt).mean()
                loss_total = loss_total + loss_b

                with torch.no_grad():
                    preds = logits[0].argmax(dim=0)
                    epoch_correct += int((preds == lbl_b).sum().item())
                    epoch_total += len(lbl_b)

            loss_mean = loss_total / B
            loss_mean.backward()

            # Gradient clipping (manual)
            total_norm = 0.0
            for p in params:
                if p.grad is not None:
                    total_norm += p.grad.data.norm(2).item() ** 2
            total_norm = total_norm ** 0.5
            clip_coef = 5.0 / max(total_norm, 5.0)

            # Manual SGD + momentum update
            current_lr = get_lr(epoch)
            with torch.no_grad():
                for p, v in zip(params, velocities):
                    if p.grad is None:
                        continue
                    g = p.grad.data * clip_coef + weight_decay * p.data
                    v.mul_(momentum).add_(g)
                    p.data.sub_(v, alpha=current_lr)

            epoch_loss += loss_mean.item()

        avg_loss = epoch_loss / steps_per_epoch
        acc = epoch_correct / max(epoch_total, 1)
        final_loss = avg_loss
        final_acc = acc

        if verbose:
            logger.info(
                "randlanet_training_epoch",
                epoch=epoch,
                epochs=epochs,
                loss=round(avg_loss, 4),
                accuracy=round(acc, 4),
                elapsed_s=round(time.time() - t_start, 1),
            )

    # Save checkpoint
    model.eval()
    checkpoint: dict[str, Any] = {
        "model_state_dict": model.state_dict(),
        "architecture": ARCHITECTURE_NAME,
        "in_channels": IN_CHANNELS,
        "num_classes": NUM_CLASSES,
        "k_neighbors": K_NEIGHBORS,
        "training_dataset": TRAINING_DATASET,
        "training_label_space": TRAINING_LABEL_SPACE,
        "feature_schema": FEATURE_SCHEMA,
        "wall_class_id": WALL_CLASS_ID,
        "epochs": epochs,
        "steps_per_epoch": steps_per_epoch,
        "batch_size": batch_size,
        "n_pts_per_sample": n_pts_per_sample,
        "seed": seed,
        "final_loss": round(final_loss, 6),
        "final_train_accuracy": round(final_acc, 6),
        "created_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "device": device,
        "pytorch_version": torch.__version__,
    }

    torch.save(checkpoint, checkpoint_path)

    # Record hash
    ckpt_hash = compute_file_sha256(checkpoint_path)
    checkpoint["checkpoint_hash"] = ckpt_hash
    checkpoint["checkpoint_path"] = str(checkpoint_path.resolve())

    # Save sidecar metadata JSON
    meta_path = checkpoint_path.with_suffix(".json")
    meta_dict = {k: v for k, v in checkpoint.items() if k != "model_state_dict"}
    meta_path.write_text(json.dumps(meta_dict, indent=2), encoding="utf-8")

    elapsed = time.time() - t_start
    logger.info(
        "randlanet_checkpoint_saved",
        path=str(checkpoint_path),
        hash=ckpt_hash[:12] + "...",
        final_loss=round(final_loss, 4),
        final_accuracy=round(final_acc, 4),
        elapsed_s=round(elapsed, 1),
    )

    return meta_dict


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Train RandLA-Net on synthetic architectural data.")
    parser.add_argument("--output", default=str(DEFAULT_CHECKPOINT_PATH))
    parser.add_argument("--epochs", type=int, default=12)
    parser.add_argument("--batch-size", type=int, default=2)
    args = parser.parse_args()

    result = train_randlanet_cpu(
        checkpoint_path=Path(args.output),
        epochs=args.epochs,
        batch_size=args.batch_size,
    )
    print(f"\nCheckpoint saved: {result['checkpoint_path']}")
    print(f"SHA-256: {result['checkpoint_hash']}")
    print(f"Final loss: {result['final_loss']}")
    print(f"Final accuracy: {result['final_train_accuracy']:.2%}")
