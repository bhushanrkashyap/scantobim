"""RandLA-Net Semantic Adapter — CPU Production Path (Phase 3A).

Production Rules:
  - STATUS = REAL_NEURAL_INFERENCE only if trained checkpoint passes ALL validation gates.
  - STATUS = TEST_ONLY_NEURAL_FORWARD if architecture loads but no trained checkpoint exists.
    Random-initialized models MUST use TEST_ONLY_NEURAL_FORWARD. Never REAL_NEURAL_INFERENCE.
  - STATUS = NO_COMPATIBLE_CHECKPOINT if no compatible checkpoint is found.
  - STATUS = UNAVAILABLE if PyTorch is not installed.
  - PRODUCTION_READY = True ONLY for REAL_NEURAL_INFERENCE.

Neural Architecture:
  RandLANet (Random Sampling + Local Feature Aggregation)
  in_channels=6, num_classes=2, k_neighbors=16

Feature Schema (6D, required to match checkpoint training):
  [x, y, z, verticality_ratio, z_rel, dist_centroid]

Chunking:
  Adaptive: chunk_size derived from available RAM + model footprint.
  Overlap: 15% spatial overlap with probability averaging reconciliation.

Supported ScanTOBIM classes:
  WALL — direct from neural output (class_id=1)
  NON_WALL (class_id=0) → FLOOR/CEILING/COLUMN via geometric post-processing.

Unsupported (must NOT be claimed):
  PIPE, DUCT, CABLE_TRAY, VALVE, EQUIPMENT, BEAM, DOOR, WINDOW, STAIR.
"""

from __future__ import annotations

import time
from pathlib import Path
from typing import Any

import numpy as np
import structlog
import torch
import torch.nn.functional as F

from agent.phase3.neural.adaptive_chunker import (
    generate_adaptive_chunks,
    reconcile_chunk_predictions,
)
from agent.phase3.neural.checkpoint_validator import (
    EXPECTED_IN_CHANNELS,
    EXPECTED_K_NEIGHBORS,
    EXPECTED_NUM_CLASSES,
    validate_checkpoint,
)
from agent.phase3.neural.cuda_guard import assert_tensor_on_cpu, get_production_device
from agent.phase3.semantic.adapter import SemanticInferenceResult, SemanticModelAdapter
from agent.phase3.status import (
    FAILED,
    NO_COMPATIBLE_CHECKPOINT,
    REAL_NEURAL_INFERENCE,
    TEST_ONLY_NEURAL_FORWARD,
    UNAVAILABLE,
    is_production_ready,
)
from agent.phase3.taxonomy import map_to_canonical

logger = structlog.get_logger()

# Production-default checkpoint path
_DEFAULT_CHECKPOINT = Path("models/randlanet_architectural_cpu.pth")

# Semantic label definitions
_WALL_CLASS_ID: int = 1
_NON_WALL_CLASS_ID: int = 0
_WALL_THRESHOLD: float = 0.50
_TRAINING_DATASET: str = "SYNTHETIC_ARCHITECTURAL_v1"
_TRAINING_LABEL_SPACE: list[str] = ["NON_WALL", "WALL"]
_FEATURE_SCHEMA: list[str] = ["x", "y", "z", "verticality_ratio", "z_rel", "dist_centroid"]

# Supported ScanTOBIM classes — MUST match training
_SUPPORTED_SCANTOBIM_CLASSES: list[str] = ["WALL"]
_UNSUPPORTED_SCANTOBIM_CLASSES: list[str] = [
    "FLOOR", "CEILING", "COLUMN", "BEAM", "DOOR", "WINDOW",
    "STAIR", "PIPE", "DUCT", "CABLE_TRAY", "VALVE", "EQUIPMENT",
]


def _compute_feature_vector(points: np.ndarray) -> np.ndarray:
    """Compute the 6D feature vector matching the checkpoint training schema.

    Feature schema (MUST match checkpoint_builder.py exactly):
      [x, y, z, verticality_ratio, z_rel, dist_centroid]

    Missing attributes (RGB, intensity, normals) are NOT fabricated.
    Only geometrically derivable features are used.
    """
    from scipy.spatial import cKDTree

    coords = points.astype(np.float32)
    N = len(coords)

    if N == 0:
        return np.zeros((0, EXPECTED_IN_CHANNELS), dtype=np.float32)

    # Verticality ratio: var(dz) / var(dxy) — high for vertical structures (walls)
    anchor_count = min(max(200, N // 20), 5000)
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

    # z_rel: normalised height position (0=floor, 1=ceiling)
    z_min = float(coords[:, 2].min())
    z_max = float(coords[:, 2].max())
    z_rel = ((coords[:, 2] - z_min) / max(z_max - z_min, 1e-4)).astype(np.float32)

    # dist_centroid: horizontal distance from centroid
    c_xy = np.mean(coords[:, :2], axis=0)
    dist_c = np.linalg.norm(coords[:, :2] - c_xy, axis=-1).astype(np.float32)

    return np.column_stack([coords, verticality_ratio, z_rel, dist_c])


class RandLANetSemanticAdapter(SemanticModelAdapter):
    """Production CPU semantic adapter using RandLA-Net.

    Lifecycle:
      load() → validates checkpoint → sets status
      health_check() → True only if REAL_NEURAL_INFERENCE
      infer() → chunked CPU forward pass with overlap reconciliation

    Status semantics:
      REAL_NEURAL_INFERENCE  — trained checkpoint + validated + CPU forward.
      TEST_ONLY_NEURAL_FORWARD — architecture works but NO trained checkpoint.
      NO_COMPATIBLE_CHECKPOINT — no matching checkpoint file found.
      UNAVAILABLE            — PyTorch import failed.
      FAILED                 — unexpected error during load or inference.
    """

    def __init__(
        self,
        checkpoint_path: str | None = None,
        device: str = "cpu",
        overlap_fraction: float = 0.15,
    ) -> None:
        super().__init__(
            model_name="RandLA-Net",
            checkpoint_path=checkpoint_path,
            device=get_production_device(),  # Always CPU
        )
        self.version = "2.0.0"
        self.overlap_fraction = overlap_fraction
        self._model: Any = None
        self._validation_result: Any = None
        self._is_trained: bool = False
        self._production_ready: bool = False
        self._checkpoint_hash: str | None = None
        self._checkpoint_metadata: dict[str, Any] = {}

    def load(self) -> bool:
        """Validate checkpoint and load model.

        Returns True if REAL_NEURAL_INFERENCE (trained checkpoint loaded).
        Returns False for all other statuses.
        """
        # 1. PyTorch availability
        try:
            import torch
            from agent.tools.randla_net import RandLANet  # noqa: F401
        except ImportError as exc:
            logger.warning("randlanet_torch_unavailable", error=str(exc))
            self._status = UNAVAILABLE
            self._loaded = False
            self._production_ready = False
            return False

        # 2. Resolve checkpoint path
        ckpt_path: Path | None = None
        if self.checkpoint_path:
            ckpt_path = Path(self.checkpoint_path)
        elif _DEFAULT_CHECKPOINT.exists():
            ckpt_path = _DEFAULT_CHECKPOINT
            self.checkpoint_path = str(ckpt_path)

        # 3. No checkpoint found
        if ckpt_path is None or not ckpt_path.exists():
            logger.info(
                "randlanet_no_checkpoint",
                searched=str(ckpt_path or _DEFAULT_CHECKPOINT),
                note=(
                    "No trained checkpoint available. "
                    "Status=NO_COMPATIBLE_CHECKPOINT. "
                    "Production inference requires a real trained checkpoint."
                ),
            )
            self._status = NO_COMPATIBLE_CHECKPOINT
            self._loaded = False
            self._production_ready = False
            return False

        # 4. Validate checkpoint through all 9 gates
        val = validate_checkpoint(ckpt_path)
        self._validation_result = val
        self._checkpoint_metadata = val.metadata

        if not val.overall_valid:
            # Checkpoint exists but failed validation
            if val.state_dict_loads and val.cpu_forward_pass and not val.is_trained_not_random:
                # Architecture works but model appears untrained
                logger.warning(
                    "randlanet_checkpoint_appears_untrained",
                    path=str(ckpt_path),
                    errors=val.errors,
                    note="Status=TEST_ONLY_NEURAL_FORWARD. Not production-ready.",
                )
                self._status = TEST_ONLY_NEURAL_FORWARD
            else:
                logger.warning(
                    "randlanet_checkpoint_validation_failed",
                    path=str(ckpt_path),
                    errors=val.errors,
                )
                self._status = FAILED
            self._loaded = False
            self._production_ready = False
            return False

        # 5. Load verified model
        try:
            import torch
            from agent.tools.randla_net import RandLANet

            ckpt = torch.load(ckpt_path, map_location="cpu")
            k = ckpt.get("k_neighbors", EXPECTED_K_NEIGHBORS)
            in_c = ckpt.get("in_channels", EXPECTED_IN_CHANNELS)
            num_c = ckpt.get("num_classes", EXPECTED_NUM_CLASSES)
            model = RandLANet(
                in_channels=in_c,
                num_classes=num_c,
                k_neighbors=k,
            )
            model.load_state_dict(ckpt["model_state_dict"], strict=True)
            model.eval()

            # Production guard: ensure model is on CPU
            assert_tensor_on_cpu(model, name="RandLANet production model")

            self._model = model
            self._is_trained = True
            self._production_ready = True
            self._checkpoint_hash = val.metadata.get("sha256")
            self._status = REAL_NEURAL_INFERENCE
            self._loaded = True

            logger.info(
                "randlanet_production_ready",
                checkpoint=str(ckpt_path),
                hash=str(self._checkpoint_hash)[:12] + "...",
                final_loss=ckpt.get("final_loss"),
                final_accuracy=ckpt.get("final_train_accuracy"),
                training_dataset=ckpt.get("training_dataset"),
            )
            return True

        except Exception as exc:  # noqa: BLE001
            logger.exception("randlanet_load_error", error=str(exc))
            self._status = FAILED
            self._loaded = False
            self._production_ready = False
            return False

    def health_check(self) -> bool:
        """Return True ONLY if production-ready with trained checkpoint."""
        return (
            self._loaded
            and self._status == REAL_NEURAL_INFERENCE
            and self._production_ready
            and self._is_trained
        )

    def infer(
        self,
        points: np.ndarray,
        colors: np.ndarray | None = None,
        normals: np.ndarray | None = None,
    ) -> SemanticInferenceResult:
        """Execute CPU-only RandLA-Net inference with adaptive chunking.

        Feature schema: [x, y, z, verticality_ratio, z_rel, dist_centroid]
        Colors/intensity/normals: NOT used (not fabricated if absent).
        Chunking: adaptive based on available RAM.
        Overlap: 15% with probability-averaging reconciliation.
        Source indices: preserved throughout.

        Returns:
            SemanticInferenceResult with REAL_NEURAL_INFERENCE status (if production-ready)
            or the current UNAVAILABLE/FAILED/TEST_ONLY_NEURAL_FORWARD status.
        """
        t0 = time.time()
        N = len(points)

        # Guard: not production-ready
        if not self.health_check():
            return SemanticInferenceResult(
                labels=["UNKNOWN"] * N,
                probabilities=np.zeros(N, dtype=float),
                status=self._status,
                model_name=self.model_name,
                checkpoint_path=self.checkpoint_path,
                device=self.device,
                runtime_s=time.time() - t0,
                diagnostics={
                    "error": f"Model not production-ready; status={self._status}",
                    "production_ready": False,
                    "is_trained": self._is_trained,
                },
            )

        if N == 0:
            return SemanticInferenceResult(
                labels=[],
                probabilities=np.zeros(0, dtype=float),
                status=REAL_NEURAL_INFERENCE,
                model_name=self.model_name,
                checkpoint_path=self.checkpoint_path,
                device=self.device,
                runtime_s=0.0,
                point_indices=np.zeros(0, dtype=np.int64),
                diagnostics={"points": 0, "production_ready": True},
            )

        try:
            import torch

            in_c = self._checkpoint_metadata.get("in_channels", EXPECTED_IN_CHANNELS)
            num_c = self._checkpoint_metadata.get("num_classes", EXPECTED_NUM_CLASSES)

            # 1. Compute features matching training schema
            if in_c == 14:
                from agent.phase3.geometry import compute_geometric_features
                feat_set = compute_geometric_features(points)
                features = feat_set.to_feature_matrix(points)
            else:
                features = _compute_feature_vector(points)  # (N, 6)

            # 2. Generate adaptive chunks with overlap
            chunks = generate_adaptive_chunks(
                points=points,
                features=features,
                overlap_fraction=self.overlap_fraction,
            )

            # 3. Per-chunk neural inference
            chunk_predictions: list[dict[str, Any]] = []
            model = self._model

            with torch.no_grad():
                for chunk in chunks:
                    chunk_pts = chunk.points.astype(np.float32)     # (M, 3)
                    chunk_feats = chunk.features.astype(np.float32)  # (M, in_c)
                    M = len(chunk_pts)

                    pts_t = torch.as_tensor(chunk_pts).unsqueeze(0)   # (1, M, 3)
                    feats_t = torch.as_tensor(chunk_feats).T.unsqueeze(0)  # (1, in_c, M)

                    # CPU guard
                    assert_tensor_on_cpu(pts_t, "chunk_pts")
                    assert_tensor_on_cpu(feats_t, "chunk_feats")

                    logits = model(pts_t, feats_t)  # (1, num_c, M)
                    probs = F.softmax(logits, dim=1)[0].T.cpu().numpy()  # (M, num_c)

                    chunk_predictions.append({
                        "chunk_id": chunk.chunk_id,
                        "source_indices": chunk.source_indices,
                        "class_probs": probs,
                    })

            # 4. Overlap reconciliation — probability averaging
            labels_int, probs_avg = reconcile_chunk_predictions(
                total_points=N,
                chunk_predictions=chunk_predictions,
                num_classes=num_c,
            )

            # 5. Map to canonical labels
            if num_c == 19:
                from agent.phase3.annotation.annotator import PHASE_3C_LABEL_SPACE
                canonical_labels: list[str] = [
                    PHASE_3C_LABEL_SPACE[lbl] if lbl < len(PHASE_3C_LABEL_SPACE) else "UNKNOWN"
                    for lbl in labels_int
                ]
                point_probs = np.max(probs_avg, axis=1)
                wall_count = int((np.array(canonical_labels) == "WALL").sum())
            else:
                canonical_labels = [
                    "WALL" if lbl == _WALL_CLASS_ID else "UNKNOWN"
                    for lbl in labels_int
                ]
                wall_count = int((labels_int == _WALL_CLASS_ID).sum())
                point_probs = probs_avg[:, _WALL_CLASS_ID]

            # Confidence distribution stats
            conf_dist = {
                "min": float(point_probs.min()),
                "max": float(point_probs.max()),
                "mean": float(point_probs.mean()),
                "wall_fraction": round(wall_count / max(N, 1), 4),
            }

            return SemanticInferenceResult(
                labels=canonical_labels,
                probabilities=point_probs,
                status=REAL_NEURAL_INFERENCE,
                model_name=self.model_name,
                checkpoint_path=self.checkpoint_path,
                device=self.device,
                runtime_s=time.time() - t0,
                point_indices=np.arange(N, dtype=np.int64),
                diagnostics={
                    "total_points": N,
                    "wall_points": wall_count,
                    "num_chunks": len(chunks),
                    "overlap_fraction": self.overlap_fraction,
                    "checkpoint_hash": self._checkpoint_hash,
                    "training_dataset": self._checkpoint_metadata.get("training_dataset"),
                    "feature_schema": _FEATURE_SCHEMA,
                    "colors_used": False,    # Colors not in feature schema
                    "normals_used": False,   # Normals not in feature schema
                    "production_ready": True,
                    "confidence_distribution": conf_dist,
                },
            )

        except Exception as exc:  # noqa: BLE001
            logger.exception("randlanet_inference_error", error=str(exc))
            self._status = FAILED
            return SemanticInferenceResult(
                labels=["UNKNOWN"] * N,
                probabilities=np.zeros(N, dtype=float),
                status=FAILED,
                model_name=self.model_name,
                checkpoint_path=self.checkpoint_path,
                device=self.device,
                runtime_s=time.time() - t0,
                diagnostics={"error": str(exc), "production_ready": False},
            )

    def supported_labels(self) -> list[str]:
        """Return ONLY the classes this model was trained to predict."""
        return list(_SUPPORTED_SCANTOBIM_CLASSES)

    def unsupported_labels(self) -> list[str]:
        """Return classes this model MUST NOT be used to classify."""
        return list(_UNSUPPORTED_SCANTOBIM_CLASSES)

    def metadata(self) -> dict[str, Any]:
        return {
            "model_name": self.model_name,
            "version": self.version,
            "status": self._status,
            "production_ready": self._production_ready,
            "is_trained": self._is_trained,
            "checkpoint_path": self.checkpoint_path,
            "checkpoint_hash": self._checkpoint_hash,
            "device": self.device,
            "architecture": "RandLANet (Random Sampling + LFA encoder/decoder)",
            "in_channels": EXPECTED_IN_CHANNELS,
            "num_classes": EXPECTED_NUM_CLASSES,
            "k_neighbors": EXPECTED_K_NEIGHBORS,
            "training_dataset": self._checkpoint_metadata.get(
                "training_dataset", "NONE" if not self._is_trained else _TRAINING_DATASET
            ),
            "training_label_space": _TRAINING_LABEL_SPACE,
            "supported_scan_to_bim_classes": _SUPPORTED_SCANTOBIM_CLASSES,
            "unsupported_scan_to_bim_classes": _UNSUPPORTED_SCANTOBIM_CLASSES,
            "feature_schema": _FEATURE_SCHEMA,
            "overlap_fraction": self.overlap_fraction,
            "inference_engine": "agent.tools.randla_net.RandLANet",
            "cpu_only": True,
            "cuda_required": False,
            "checkpoint_validation": (
                self._validation_result.to_dict()
                if self._validation_result is not None
                else None
            ),
        }
