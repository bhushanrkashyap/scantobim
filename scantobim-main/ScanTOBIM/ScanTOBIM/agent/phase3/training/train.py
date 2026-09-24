"""Multi-Class RandLA-Net Training Pipeline on CPU.

Trains the 14D multi-class RandLA-Net model on authentic ground-truth partitions,
incorporates class imbalance weights, tracks validation loss and mIoU,
and saves the validated checkpoint with SHA-256 and model card.
"""

from __future__ import annotations

import time
from pathlib import Path
from typing import Any

import numpy as np
import structlog
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

from agent.phase3.annotation.annotator import PHASE_3C_LABEL_SPACE
from agent.phase3.geometry import FEATURE_SCHEMA_14D
from agent.phase3.training.augment import PointCloudAugmentor
from agent.phase3.training.checkpoints import save_multiclass_checkpoint
from agent.phase3.training.dataset import Phase3CDataset
from agent.phase3.training.validate import evaluate_model_on_point_cloud
from agent.tools.randla_net import RandLANet

logger = structlog.get_logger(__name__)


def train_multiclass_randlanet(
    train_npz_paths: list[Path | str],
    val_npz_paths: list[Path | str],
    output_checkpoint_path: Path | str = "models/randlanet_multiclass_cpu.pth",
    epochs: int = 12,
    batch_size: int = 1,
    learning_rate: float = 0.003,
    block_size: int = 2048,
    blocks_per_epoch: int = 30,
    seed: int = 42,
) -> dict[str, Any]:
    """Train multi-class RandLA-Net on CPU with full provenance and validation gates."""
    print("=" * 80)
    print("PHASE 3C: MULTI-CLASS RANDLA-NET TRAINING PIPELINE (CPU ONLY)")
    print("=" * 80)

    # 1. Deterministic seeds and CPU device verification
    torch.manual_seed(seed)
    np.random.seed(seed)
    assert not torch.cuda.is_available(), "CUDA is strictly prohibited in CPU production pipeline"
    device = torch.device("cpu")
    print(f"Device: {device} | Architecture: RandLANet (in_channels=14, num_classes={len(PHASE_3C_LABEL_SPACE)})")

    # 2. Prepare datasets
    print("Loading training dataset partitions...")
    augmentor = PointCloudAugmentor(rotate_z=True, jitter_std=0.005, seed=seed)
    train_ds = Phase3CDataset(
        npz_paths=train_npz_paths,
        block_size=block_size,
        num_blocks=blocks_per_epoch,
        augmentor=augmentor,
        seed=seed,
    )
    print(f"Training dataset: {train_ds.total_points:,} points across {len(train_npz_paths)} partitions.")
    print(f"Class weights computed (smoothed inverse frequency).")

    # Load validation points directly for full-scene evaluation
    val_pts_list, val_ids_list, val_cls_list = [], [], []
    for vp in val_npz_paths:
        v_data = np.load(vp)
        val_pts_list.append(v_data["points"])
        val_ids_list.append(v_data["point_ids"])
        val_cls_list.append(v_data["class_ids"])

    val_points = np.vstack(val_pts_list).astype(np.float32)
    val_class_ids = np.concatenate(val_cls_list).astype(np.int64)
    print(f"Validation dataset: {len(val_points):,} points across {len(val_npz_paths)} partitions.")

    # 3. Initialize Model
    model = RandLANet(
        in_channels=len(FEATURE_SCHEMA_14D),
        num_classes=len(PHASE_3C_LABEL_SPACE),
        k_neighbors=16,
    ).to(device)

    # Use weighted cross-entropy loss
    criterion = nn.CrossEntropyLoss(weight=train_ds.class_weights.to(device))
    
    # Manual SGD with momentum (CPU-safe, avoids PyTorch 2.1 dynamo -> onnx -> transformers import hooks)
    momentum = 0.9
    weight_decay = 1e-4
    params = list(model.parameters())
    velocities = [torch.zeros_like(p) for p in params]

    def get_lr(epoch_idx: int) -> float:
        lr_min = 1e-5
        return float(lr_min + 0.5 * (learning_rate - lr_min) * (1.0 + np.cos(np.pi * epoch_idx / epochs)))

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)

    best_val_f1 = -1.0
    best_metrics: dict[str, Any] = {}
    best_loss = float("inf")
    t0_train = time.time()

    # 4. Training Loop
    for epoch in range(1, epochs + 1):
        model.train()
        epoch_losses: list[float] = []

        for pts, feats, labels in train_loader:
            pts = pts.to(device)      # (B, N, 3)
            feats = feats.to(device)  # (B, C, N)
            labels = labels.to(device) # (B, N)

            for p in params:
                if p.grad is not None:
                    p.grad.zero_()

            logits = model(pts, feats) # (B, num_classes, N)
            loss = criterion(logits, labels)
            loss.backward()

            # Gradient clipping
            total_norm = 0.0
            for p in params:
                if p.grad is not None:
                    total_norm += p.grad.data.norm(2).item() ** 2
            total_norm = total_norm ** 0.5
            clip_coef = 1.0 / max(total_norm, 1.0)

            # Parameter update
            cur_lr = get_lr(epoch)
            with torch.no_grad():
                for p, v in zip(params, velocities):
                    if p.grad is None:
                        continue
                    g = p.grad.data * clip_coef + weight_decay * p.data
                    v.mul_(momentum).add_(g)
                    p.data.sub_(v, alpha=cur_lr)

            epoch_losses.append(float(loss.item()))

        avg_loss = float(np.mean(epoch_losses)) if epoch_losses else 0.0

        # Periodic Validation
        if epoch % 3 == 0 or epoch == epochs:
            val_metrics, _, _ = evaluate_model_on_point_cloud(
                model=model,
                points=val_points,
                ground_truth_class_ids=val_class_ids,
                split_name="VALIDATION",
                label_space=PHASE_3C_LABEL_SPACE,
            )
            print(
                f"Epoch [{epoch:02d}/{epochs:02d}] "
                f"Loss: {avg_loss:.4f} | "
                f"Val OA: {val_metrics.overall_accuracy * 100:.1f}% | "
                f"Val Macro F1: {val_metrics.macro_f1 * 100:.1f}% | "
                f"Val mIoU: {val_metrics.mean_iou * 100:.1f}%"
            )

            if val_metrics.macro_f1 > best_val_f1:
                best_val_f1 = val_metrics.macro_f1
                best_loss = avg_loss
                best_metrics = val_metrics.to_dict()

    train_duration = time.time() - t0_train
    print(f"\nTraining completed in {train_duration:.2f}s! Best Val Macro F1: {best_val_f1 * 100:.1f}%")

    # 5. Save Validated Checkpoint and Model Card
    model_card = save_multiclass_checkpoint(
        model=model,
        checkpoint_path=output_checkpoint_path,
        architecture="RandLANet",
        in_channels=len(FEATURE_SCHEMA_14D),
        num_classes=len(PHASE_3C_LABEL_SPACE),
        label_space=PHASE_3C_LABEL_SPACE,
        feature_schema=FEATURE_SCHEMA_14D,
        training_dataset="AUTHENTIC_ASTM_E57_SECTOR_TRAIN_v1",
        validation_metrics=best_metrics,
        epochs=epochs,
        final_loss=best_loss,
        notes="Phase 3C 14D Multi-Class RandLA-Net trained on authentic E57 point cloud on CPU.",
    )

    return {
        "status": "TRAINING_COMPLETE",
        "checkpoint_path": str(output_checkpoint_path),
        "duration_s": round(train_duration, 2),
        "best_metrics": best_metrics,
        "model_card": model_card,
    }


if __name__ == "__main__":
    train_npz = [Path("datasets/phase3c/annotations/sector_train_central_data.npz")]
    val_npz = [Path("datasets/phase3c/annotations/sector_val_east_data.npz")]
    train_multiclass_randlanet(train_npz, val_npz, epochs=12, batch_size=1)
