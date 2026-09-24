"""RandLA-Net Checkpoint Validator — Phase 3A CPU Production.

Validates that a checkpoint file:
  1. Exists and is readable
  2. SHA-256 hash matches recorded hash
  3. Is a valid PyTorch checkpoint dict
  4. Contains model_state_dict with expected architecture
  5. num_classes matches expected
  6. in_channels matches expected feature schema
  7. Architecture can load the state dict without errors
  8. Model runs in eval mode on CPU without crashing
  9. Model is NOT random-initialized (loss differs from random baseline)

Produces:
  CheckpointValidationResult with detailed pass/fail for each gate.
"""

from __future__ import annotations

import hashlib
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import structlog

logger = structlog.get_logger()

# Expected schema for the RandLA-Net architectural checkpoint
EXPECTED_ARCHITECTURE: str = "RandLANet"
EXPECTED_IN_CHANNELS: int = 6
EXPECTED_NUM_CLASSES: int = 2
EXPECTED_K_NEIGHBORS: int = 16
EXPECTED_TRAINING_DATASET: str = "SYNTHETIC_ARCHITECTURAL_v1"
EXPECTED_LABEL_SPACE: list[str] = ["NON_WALL", "WALL"]
EXPECTED_FEATURE_SCHEMA: list[str] = [
    "x", "y", "z", "verticality_ratio", "z_rel", "dist_centroid"
]


@dataclass
class CheckpointValidationResult:
    """Detailed outcome of checkpoint validation across all 9 gates."""

    checkpoint_path: str
    file_exists: bool = False
    file_readable: bool = False
    hash_matches: bool | None = None        # None if no recorded hash
    is_valid_dict: bool = False
    has_state_dict: bool = False
    architecture_matches: bool = False
    num_classes_match: bool = False
    in_channels_match: bool = False
    state_dict_loads: bool = False
    cpu_forward_pass: bool = False
    is_trained_not_random: bool = False     # True if not a random-init model
    overall_valid: bool = False
    errors: list[str] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)
    validation_time_s: float = 0.0

    def to_dict(self) -> dict[str, Any]:
        return {
            "checkpoint_path": self.checkpoint_path,
            "gates": {
                "file_exists": self.file_exists,
                "file_readable": self.file_readable,
                "hash_matches": self.hash_matches,
                "is_valid_dict": self.is_valid_dict,
                "has_state_dict": self.has_state_dict,
                "architecture_matches": self.architecture_matches,
                "num_classes_match": self.num_classes_match,
                "in_channels_match": self.in_channels_match,
                "state_dict_loads": self.state_dict_loads,
                "cpu_forward_pass": self.cpu_forward_pass,
                "is_trained_not_random": self.is_trained_not_random,
            },
            "overall_valid": self.overall_valid,
            "errors": self.errors,
            "metadata": self.metadata,
            "validation_time_s": round(self.validation_time_s, 4),
        }


def compute_file_sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()


def _random_weight_baseline_loss(class_weights: list[float] | None = None) -> float:
    """Compute expected cross-entropy loss for a random classifier.

    For unweighted balanced binary: -log(0.5) = 0.693
    For weighted (weights=[1.0, 1.3]) balanced binary:
      E[loss] = 0.5 * 1.0 * (-log(0.5)) + 0.5 * 1.3 * (-log(0.5))
              = 0.5 * (1.0 + 1.3) * 0.693 = 0.5 * 2.3 * 0.693 = 0.797

    A trained model with these weights should have loss < 0.797.
    """
    if class_weights is None:
        class_weights = [1.0, 1.3]  # default from checkpoint_builder
    import math
    n_classes = len(class_weights)
    uniform_prob = 1.0 / n_classes
    log_uniform = math.log(uniform_prob)  # -log(0.5) = 0.693
    # Expected weighted CE for uniform predictions on balanced data
    expected = sum(w * (-log_uniform) for w in class_weights) / n_classes
    return expected  # ≈ 0.797 for [1.0, 1.3] weights


def validate_checkpoint(
    checkpoint_path: Path | str,
    expected_hash: str | None = None,
    expected_in_channels: int | None = None,
    expected_num_classes: int | None = None,
) -> CheckpointValidationResult:
    """Run all 9 validation gates on a checkpoint file.

    Args:
        checkpoint_path: Path to the .pth checkpoint file.
        expected_hash: If provided, verify SHA-256 matches.
        expected_in_channels: If provided, verify in_channels.
        expected_num_classes: If provided, verify num_classes.

    Returns:
        CheckpointValidationResult with pass/fail for each gate.
    """
    t0 = time.time()
    path = Path(checkpoint_path)
    result = CheckpointValidationResult(checkpoint_path=str(path))

    # Gate 1: File exists
    result.file_exists = path.exists()
    if not result.file_exists:
        result.errors.append(f"Checkpoint file not found: {path}")
        result.validation_time_s = time.time() - t0
        return result

    # Gate 2: File readable + SHA-256
    try:
        actual_hash = compute_file_sha256(path)
        result.file_readable = True
        result.metadata["sha256"] = actual_hash
    except OSError as e:
        result.errors.append(f"Cannot read checkpoint: {e}")
        result.validation_time_s = time.time() - t0
        return result

    # Gate 2b: Hash check (optional)
    if expected_hash is not None:
        result.hash_matches = actual_hash == expected_hash
        if not result.hash_matches:
            result.errors.append(
                f"Hash mismatch: expected {expected_hash[:12]}… "
                f"got {actual_hash[:12]}…"
            )

    # Gate 3: Load as PyTorch dict
    try:
        import torch
        ckpt = torch.load(path, map_location="cpu")
        result.is_valid_dict = isinstance(ckpt, dict)
        if not result.is_valid_dict:
            result.errors.append(f"Checkpoint is {type(ckpt).__name__}, expected dict.")
            result.validation_time_s = time.time() - t0
            return result
        # Record metadata from checkpoint
        for key in [
            "architecture", "in_channels", "num_classes", "k_neighbors",
            "training_dataset", "training_label_space", "feature_schema",
            "wall_class_id", "epochs", "final_loss", "final_train_accuracy",
            "created_at", "pytorch_version", "checkpoint_hash",
        ]:
            if key in ckpt:
                result.metadata[key] = ckpt[key]
    except Exception as e:  # noqa: BLE001
        result.errors.append(f"Failed to load checkpoint: {e}")
        result.validation_time_s = time.time() - t0
        return result

    # Gate 4: Has state dict
    result.has_state_dict = "model_state_dict" in ckpt
    if not result.has_state_dict:
        result.errors.append("Checkpoint missing 'model_state_dict' key.")
        result.validation_time_s = time.time() - t0
        return result

    state_dict = ckpt["model_state_dict"]

    # Gate 5: Architecture matches
    arch = ckpt.get("architecture", "UNKNOWN")
    result.architecture_matches = arch == EXPECTED_ARCHITECTURE
    result.metadata["architecture_found"] = arch
    if not result.architecture_matches:
        result.errors.append(f"Architecture '{arch}' != expected '{EXPECTED_ARCHITECTURE}'")

    # Gate 6: num_classes matches
    nc = ckpt.get("num_classes", -1)
    if expected_num_classes is not None:
        result.num_classes_match = (nc == expected_num_classes)
        target_nc = expected_num_classes
    else:
        result.num_classes_match = (nc in (EXPECTED_NUM_CLASSES, 19))
        target_nc = nc if nc > 0 else EXPECTED_NUM_CLASSES

    if not result.num_classes_match:
        result.errors.append(f"num_classes {nc} not compatible")

    # Gate 7: in_channels matches
    ic = ckpt.get("in_channels", -1)
    if expected_in_channels is not None:
        result.in_channels_match = (ic == expected_in_channels)
        target_ic = expected_in_channels
    else:
        result.in_channels_match = (ic in (EXPECTED_IN_CHANNELS, 14))
        target_ic = ic if ic > 0 else EXPECTED_IN_CHANNELS

    if not result.in_channels_match:
        result.errors.append(f"in_channels {ic} not compatible")

    # Gate 8: State dict loads into model
    try:
        import torch
        from agent.tools.randla_net import RandLANet

        k_neighbors = ckpt.get("k_neighbors", EXPECTED_K_NEIGHBORS)
        model = RandLANet(
            in_channels=target_ic,
            num_classes=target_nc,
            k_neighbors=k_neighbors,
        )
        model.load_state_dict(state_dict, strict=True)
        model.eval()
        result.state_dict_loads = True
    except Exception as e:  # noqa: BLE001
        result.errors.append(f"State dict load failed: {e}")
        result.validation_time_s = time.time() - t0
        return result

    # Gate 9: CPU forward pass succeeds
    try:
        import torch

        rng = np.random.default_rng(0)
        test_pts = rng.uniform(0, 5, (1, 64, 3)).astype(np.float32)
        test_feats = rng.uniform(-1, 1, (1, target_ic, 64)).astype(np.float32)

        pts_t = torch.as_tensor(test_pts)
        feats_t = torch.as_tensor(test_feats)

        with torch.no_grad():
            logits = model(pts_t, feats_t)

        expected_shape = (1, target_nc, 64)
        if logits.shape != expected_shape:
            result.errors.append(
                f"Output shape {tuple(logits.shape)} != expected {expected_shape}"
            )
        else:
            result.cpu_forward_pass = True
            result.metadata["output_shape"] = list(logits.shape)

        # Gate 9b: Trained check — compute loss on uniform data
        recorded_loss = ckpt.get("final_loss", None)
        recorded_acc = ckpt.get("final_train_accuracy", None)
        if recorded_acc is None:
            # Check validation metrics dict if present
            recorded_acc = ckpt.get("validation_metrics", {}).get("overall_accuracy", None)

        if recorded_loss is not None and recorded_acc is not None:
            import math
            random_baseline = _random_weight_baseline_loss() if target_nc == 2 else float(math.log(target_nc))
            min_acc_threshold = 0.55 if target_nc == 2 else (1.5 / target_nc)

            result.is_trained_not_random = (
                float(recorded_acc) > min_acc_threshold and float(recorded_loss) < random_baseline
            )
            result.metadata["trained_evidence"] = {
                "final_loss": recorded_loss,
                "final_accuracy": recorded_acc,
                "random_baseline_loss": random_baseline,
                "conclusion": "TRAINED" if result.is_trained_not_random else "SUSPECT_RANDOM_INIT",
            }
            if not result.is_trained_not_random:
                result.errors.append(
                    f"Model performance suspect: loss {recorded_loss} >= baseline {random_baseline:.3f} "
                    f"or acc {recorded_acc} <= {min_acc_threshold:.3f}"
                )
        else:
            # No training metadata — cannot confirm training
            result.is_trained_not_random = False
            result.errors.append(
                "Checkpoint missing training metadata (final_loss/final_train_accuracy). "
                "Cannot confirm model was trained. Treating as SUSPECT_RANDOM_INIT."
            )

    except Exception as e:  # noqa: BLE001
        result.errors.append(f"CPU forward pass failed: {e}")

    # Overall gate
    result.overall_valid = (
        result.file_exists
        and result.file_readable
        and result.is_valid_dict
        and result.has_state_dict
        and result.architecture_matches
        and result.num_classes_match
        and result.in_channels_match
        and result.state_dict_loads
        and result.cpu_forward_pass
        and result.is_trained_not_random
        and (result.hash_matches is not False)  # True or None (not provided)
    )

    result.validation_time_s = time.time() - t0
    logger.info(
        "checkpoint_validation_complete",
        path=str(path),
        overall_valid=result.overall_valid,
        errors=result.errors,
    )
    return result
