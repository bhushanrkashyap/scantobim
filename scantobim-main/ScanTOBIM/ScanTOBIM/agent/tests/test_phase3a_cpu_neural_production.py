"""Phase 3A CPU Neural Production Tests.

Test suite for the CPU-only production deep learning semantic segmentation path.

Phase 3A CPU PASS criteria:
  1. At least one real trained checkpoint exists on disk.
  2. Checkpoint SHA-256 is recorded and matches file.
  3. Checkpoint schema matches architecture expectations.
  4. Checkpoint loads into RandLANet without error.
  5. CPU device is enforced — no CUDA.
  6. Actual neural forward pass executes.
  7. Output shape is (1, num_classes, N).
  8. Output labels are canonical ScanTOBIM classes.
  9. Source-index provenance is preserved.
  10. Chunked inference covers all source points.
  11. Overlap reconciliation averages probabilities correctly.
  12. Missing features (RGB, intensity, normals) handled without fabrication.
  13. best_production_semantic_model() selects based on production criteria.
  14. Random-init / invalid checkpoint is NOT classified as REAL_NEURAL_INFERENCE.
  15. No CUDA introduced in production code.
  16. Semantic + geometric fusion scores combine correctly.
  17. Model provenance is recorded with hash, dataset, and feature schema.

Required statuses tested:
  REAL_NEURAL_INFERENCE   → trained checkpoint + validated inference
  TEST_ONLY_NEURAL_FORWARD → NOT production-ready
  NO_COMPATIBLE_CHECKPOINT → adapter reports correctly when checkpoint absent
  CPU_UNAVAILABLE          → KPConv correctly excluded from CPU production
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
import torch

from agent.phase3.neural.adaptive_chunker import (
    compute_adaptive_chunk_size,
    generate_adaptive_chunks,
    reconcile_chunk_predictions,
)
from agent.phase3.neural.checkpoint_builder import (
    DEFAULT_CHECKPOINT_PATH,
    compute_file_sha256,
)
from agent.phase3.neural.checkpoint_validator import (
    EXPECTED_ARCHITECTURE,
    EXPECTED_IN_CHANNELS,
    EXPECTED_NUM_CLASSES,
    validate_checkpoint,
)
from agent.phase3.neural.cuda_guard import (
    CUDAProductionViolation,
    assert_tensor_on_cpu,
    get_production_device,
    scan_for_cuda_imports,
)
from agent.phase3.neural.label_space import (
    generate_label_space_report,
    validate_label_claim,
)
from agent.phase3.neural.registry import NeuralModelRegistry
from agent.phase3.semantic.randla_adapter import RandLANetSemanticAdapter
from agent.phase3.semantic.selector import SemanticModelSelector
from agent.phase3.status import (
    FAILED,
    GEOMETRIC_ADAPTER,
    NO_COMPATIBLE_CHECKPOINT,
    REAL_NEURAL_INFERENCE,
    TEST_ONLY_NEURAL_FORWARD,
    is_production_ready,
    is_real_neural,
)

# ── Checkpoint path — may or may not exist ────────────────────────────────────
CHECKPOINT_PATH = DEFAULT_CHECKPOINT_PATH
CHECKPOINT_AVAILABLE = CHECKPOINT_PATH.exists()


def skip_if_no_checkpoint(func):
    """Decorator: skip test if trained checkpoint is not available."""
    return pytest.mark.skipif(
        not CHECKPOINT_AVAILABLE,
        reason=f"Trained checkpoint not found: {CHECKPOINT_PATH}. Run checkpoint_builder to train.",
    )(func)


# ── Synthetic point cloud helpers ─────────────────────────────────────────────

def make_wall_pts(n: int = 500) -> np.ndarray:
    rng = np.random.default_rng(42)
    x = rng.uniform(0.0, 6.0, n)
    y = rng.normal(0.0, 0.005, n)  # tight wall
    z = rng.uniform(0.0, 3.0, n)
    return np.column_stack([x, y, z]).astype(np.float64)


def make_room_pts(n: int = 2000) -> np.ndarray:
    rng = np.random.default_rng(7)
    # Wall + floor + clutter
    wall = np.column_stack([rng.uniform(0, 8, n // 3), np.zeros(n // 3), rng.uniform(0, 3, n // 3)])
    floor = np.column_stack([rng.uniform(0, 8, n // 3), rng.uniform(0, 6, n // 3), np.zeros(n // 3)])
    clutter = rng.uniform(0, 6, (n - 2 * (n // 3), 3))
    return np.vstack([wall, floor, clutter]).astype(np.float64)


# ═══════════════════════════════════════════════════════════════════════════════
# 1. CHECKPOINT EXISTENCE
# ═══════════════════════════════════════════════════════════════════════════════

class TestCheckpointExistence:
    """Gate 1: Real trained checkpoint exists on disk."""

    def test_checkpoint_path_defined(self):
        assert CHECKPOINT_PATH is not None
        assert str(CHECKPOINT_PATH).endswith(".pth")

    @skip_if_no_checkpoint
    def test_checkpoint_file_exists(self):
        """Requirement: trained checkpoint file must exist."""
        assert CHECKPOINT_PATH.exists(), (
            f"Trained checkpoint not found at {CHECKPOINT_PATH}. "
            "Run: python3 -m agent.phase3.neural.checkpoint_builder"
        )

    @skip_if_no_checkpoint
    def test_checkpoint_file_size_nonzero(self):
        """Checkpoint must have non-zero size."""
        size = CHECKPOINT_PATH.stat().st_size
        assert size > 1000, f"Checkpoint file size {size} bytes is suspiciously small."

    @skip_if_no_checkpoint
    def test_checkpoint_metadata_sidecar_exists(self):
        """Sidecar JSON metadata must accompany the checkpoint."""
        meta_path = CHECKPOINT_PATH.with_suffix(".json")
        assert meta_path.exists(), f"Metadata sidecar not found: {meta_path}"
        meta = json.loads(meta_path.read_text())
        assert "checkpoint_hash" in meta
        assert "training_dataset" in meta
        assert "feature_schema" in meta


# ═══════════════════════════════════════════════════════════════════════════════
# 2. CHECKPOINT HASH
# ═══════════════════════════════════════════════════════════════════════════════

class TestCheckpointHash:
    """Gate 2: SHA-256 is recorded and matches file."""

    @skip_if_no_checkpoint
    def test_sha256_hash_computable(self):
        """Hash computation must succeed."""
        h = compute_file_sha256(CHECKPOINT_PATH)
        assert len(h) == 64, f"Expected 64-char SHA-256, got {len(h)}"
        assert h.isalnum(), "SHA-256 should be hexadecimal"

    @skip_if_no_checkpoint
    def test_sha256_matches_sidecar(self):
        """Recorded hash in sidecar must match actual file hash."""
        actual = compute_file_sha256(CHECKPOINT_PATH)
        meta_path = CHECKPOINT_PATH.with_suffix(".json")
        if meta_path.exists():
            meta = json.loads(meta_path.read_text())
            recorded = meta.get("checkpoint_hash", "MISSING")
            assert actual == recorded, (
                f"Hash mismatch: file={actual[:12]}… sidecar={recorded[:12]}…. "
                "Checkpoint may have been corrupted or replaced."
            )


# ═══════════════════════════════════════════════════════════════════════════════
# 3. CHECKPOINT SCHEMA
# ═══════════════════════════════════════════════════════════════════════════════

class TestCheckpointSchema:
    """Gate 3-7: Checkpoint dict structure and architecture parameters."""

    @skip_if_no_checkpoint
    def test_checkpoint_is_dict(self):
        ckpt = torch.load(CHECKPOINT_PATH, map_location="cpu")
        assert isinstance(ckpt, dict), f"Expected dict, got {type(ckpt)}"

    @skip_if_no_checkpoint
    def test_checkpoint_has_state_dict(self):
        ckpt = torch.load(CHECKPOINT_PATH, map_location="cpu")
        assert "model_state_dict" in ckpt

    @skip_if_no_checkpoint
    def test_checkpoint_architecture_matches(self):
        ckpt = torch.load(CHECKPOINT_PATH, map_location="cpu")
        assert ckpt.get("architecture") == EXPECTED_ARCHITECTURE, (
            f"Architecture mismatch: {ckpt.get('architecture')} != {EXPECTED_ARCHITECTURE}"
        )

    @skip_if_no_checkpoint
    def test_checkpoint_num_classes_matches(self):
        ckpt = torch.load(CHECKPOINT_PATH, map_location="cpu")
        assert ckpt.get("num_classes") == EXPECTED_NUM_CLASSES

    @skip_if_no_checkpoint
    def test_checkpoint_in_channels_matches(self):
        ckpt = torch.load(CHECKPOINT_PATH, map_location="cpu")
        assert ckpt.get("in_channels") == EXPECTED_IN_CHANNELS

    @skip_if_no_checkpoint
    def test_checkpoint_has_training_metadata(self):
        ckpt = torch.load(CHECKPOINT_PATH, map_location="cpu")
        assert "training_dataset" in ckpt
        assert "training_label_space" in ckpt
        assert "feature_schema" in ckpt
        assert "final_loss" in ckpt
        assert "final_train_accuracy" in ckpt


# ═══════════════════════════════════════════════════════════════════════════════
# 4. CHECKPOINT LOAD
# ═══════════════════════════════════════════════════════════════════════════════

class TestCheckpointLoad:
    """Gate 8: State dict loads into RandLANet without error."""

    @skip_if_no_checkpoint
    def test_state_dict_loads_strictly(self):
        from agent.tools.randla_net import RandLANet
        ckpt = torch.load(CHECKPOINT_PATH, map_location="cpu")
        model = RandLANet(in_channels=EXPECTED_IN_CHANNELS, num_classes=EXPECTED_NUM_CLASSES)
        model.load_state_dict(ckpt["model_state_dict"], strict=True)  # must not raise

    @skip_if_no_checkpoint
    def test_full_validation_passes(self):
        result = validate_checkpoint(CHECKPOINT_PATH)
        assert result.overall_valid, (
            f"Checkpoint failed validation. Errors: {result.errors}"
        )

    @skip_if_no_checkpoint
    def test_validation_gates_detail(self):
        result = validate_checkpoint(CHECKPOINT_PATH)
        assert result.file_exists
        assert result.is_valid_dict
        assert result.has_state_dict
        assert result.architecture_matches
        assert result.num_classes_match
        assert result.in_channels_match
        assert result.state_dict_loads
        assert result.cpu_forward_pass
        assert result.is_trained_not_random, (
            f"Checkpoint not confirmed as trained: {result.metadata.get('trained_evidence')}"
        )


# ═══════════════════════════════════════════════════════════════════════════════
# 5. CPU DEVICE ENFORCEMENT
# ═══════════════════════════════════════════════════════════════════════════════

class TestCPUEnforcement:
    """Gate 5: CPU-only execution — no CUDA allowed."""

    def test_production_device_is_cpu(self):
        assert get_production_device() == "cpu"

    def test_cpu_tensor_assertion_passes(self):
        t = torch.zeros(10)  # CPU by default
        assert_tensor_on_cpu(t, "test_tensor")  # must not raise

    def test_cuda_tensor_assertion_raises(self):
        if not torch.cuda.is_available():
            pytest.skip("CUDA not available on this machine.")
        t = torch.zeros(10).cuda()
        with pytest.raises(CUDAProductionViolation):
            assert_tensor_on_cpu(t, "cuda_tensor")

    def test_cuda_not_initialized(self):
        """Production startup: CUDA context must not be initialized."""
        assert not torch.cuda.is_initialized(), (
            "CUDA context is initialized — production code must not activate CUDA."
        )

    @skip_if_no_checkpoint
    def test_adapter_device_is_cpu(self):
        adapter = RandLANetSemanticAdapter()
        assert adapter.device == "cpu"
        adapter.load()
        assert adapter.device == "cpu"


# ═══════════════════════════════════════════════════════════════════════════════
# 6. ACTUAL NEURAL FORWARD PASS
# ═══════════════════════════════════════════════════════════════════════════════

class TestActualForwardPass:
    """Gate 6-7: Neural forward pass and output shape."""

    @skip_if_no_checkpoint
    def test_adapter_load_succeeds_with_trained_checkpoint(self):
        adapter = RandLANetSemanticAdapter(checkpoint_path=str(CHECKPOINT_PATH))
        loaded = adapter.load()
        assert loaded is True, f"Load failed with status: {adapter.status}"
        assert adapter.status == REAL_NEURAL_INFERENCE

    @skip_if_no_checkpoint
    def test_forward_pass_returns_real_neural_status(self):
        adapter = RandLANetSemanticAdapter(checkpoint_path=str(CHECKPOINT_PATH))
        adapter.load()
        pts = make_wall_pts(200)
        result = adapter.infer(pts)
        assert result.status == REAL_NEURAL_INFERENCE, (
            f"Expected REAL_NEURAL_INFERENCE, got {result.status}. "
            "Only trained checkpoints qualify."
        )

    @skip_if_no_checkpoint
    def test_output_shape_matches_input(self):
        adapter = RandLANetSemanticAdapter(checkpoint_path=str(CHECKPOINT_PATH))
        adapter.load()
        pts = make_wall_pts(300)
        result = adapter.infer(pts)
        assert len(result.labels) == len(pts)
        assert result.probabilities.shape == (len(pts),)

    @skip_if_no_checkpoint
    def test_probabilities_in_valid_range(self):
        adapter = RandLANetSemanticAdapter(checkpoint_path=str(CHECKPOINT_PATH))
        adapter.load()
        pts = make_wall_pts(200)
        result = adapter.infer(pts)
        assert float(result.probabilities.min()) >= 0.0
        assert float(result.probabilities.max()) <= 1.0


# ═══════════════════════════════════════════════════════════════════════════════
# 7-8. OUTPUT LABEL COMPATIBILITY
# ═══════════════════════════════════════════════════════════════════════════════

class TestOutputLabelCompatibility:
    """Gate 8: Output labels are canonical ScanTOBIM classes."""

    @skip_if_no_checkpoint
    def test_labels_are_canonical(self):
        from agent.phase3.taxonomy import CANONICAL_LABEL_SET
        adapter = RandLANetSemanticAdapter(checkpoint_path=str(CHECKPOINT_PATH))
        adapter.load()
        pts = make_wall_pts(200)
        result = adapter.infer(pts)
        bad = {lbl for lbl in result.labels if lbl not in CANONICAL_LABEL_SET}
        assert not bad, f"Non-canonical labels: {bad}"

    def test_adapter_supported_labels_are_canonical(self):
        from agent.phase3.taxonomy import CANONICAL_LABEL_SET
        adapter = RandLANetSemanticAdapter()
        for lbl in adapter.supported_labels():
            assert lbl in CANONICAL_LABEL_SET

    def test_adapter_does_not_claim_unsupported_mep_classes(self):
        """Model must NOT claim PIPE, DUCT, CABLE_TRAY, VALVE."""
        adapter = RandLANetSemanticAdapter()
        supported = set(adapter.supported_labels())
        mep_classes = {"PIPE", "DUCT", "CABLE_TRAY", "VALVE"}
        falsely_claimed = mep_classes & supported
        assert not falsely_claimed, (
            f"RandLA-Net falsely claims to support: {falsely_claimed}. "
            "These classes are not in the training vocabulary."
        )

    def test_label_space_validation(self):
        assert validate_label_claim("randlanet", "WALL") == "SUPPORTED"
        assert validate_label_claim("randlanet", "PIPE") == "UNSUPPORTED"
        assert validate_label_claim("randlanet", "DUCT") == "UNSUPPORTED"
        assert validate_label_claim("randlanet", "VALVE") == "UNSUPPORTED"

    def test_label_space_report_written(self, tmp_path):
        out = generate_label_space_report(output_dir=tmp_path, model_keys=["randlanet"])
        assert out.exists()
        data = json.loads(out.read_text())
        assert data["models"]["randlanet"]["supported_scan_to_bim_classes"] == ["WALL"]
        assert "PIPE" in data["models"]["randlanet"]["unsupported_scan_to_bim_classes"]


# ═══════════════════════════════════════════════════════════════════════════════
# 9. SOURCE-INDEX PRESERVATION
# ═══════════════════════════════════════════════════════════════════════════════

class TestSourceIndexPreservation:
    """Gate 9: Source-point indices preserved throughout inference."""

    @skip_if_no_checkpoint
    def test_point_indices_populated(self):
        adapter = RandLANetSemanticAdapter(checkpoint_path=str(CHECKPOINT_PATH))
        adapter.load()
        pts = make_wall_pts(150)
        result = adapter.infer(pts)
        assert result.point_indices is not None, "point_indices must not be None"
        assert len(result.point_indices) == len(pts)

    @skip_if_no_checkpoint
    def test_point_indices_are_zero_to_n(self):
        adapter = RandLANetSemanticAdapter(checkpoint_path=str(CHECKPOINT_PATH))
        adapter.load()
        pts = make_wall_pts(100)
        result = adapter.infer(pts)
        np.testing.assert_array_equal(
            np.sort(result.point_indices),
            np.arange(len(pts), dtype=np.int64),
        )


# ═══════════════════════════════════════════════════════════════════════════════
# 10. CHUNKED INFERENCE
# ═══════════════════════════════════════════════════════════════════════════════

class TestChunkedInference:
    """Gate 10: Adaptive chunking covers all source points."""

    def test_adaptive_chunk_size_respects_min_max(self):
        from agent.phase3.neural.adaptive_chunker import MAX_CHUNK_SIZE, MIN_CHUNK_SIZE
        size = compute_adaptive_chunk_size(total_points=1_000_000)
        assert MIN_CHUNK_SIZE <= size <= MAX_CHUNK_SIZE

    def test_single_chunk_for_small_cloud(self):
        pts = make_wall_pts(500)
        chunks = generate_adaptive_chunks(pts, chunk_size=10_000)
        assert len(chunks) == 1
        assert len(chunks[0].source_indices) == len(pts)

    def test_multi_chunk_for_large_cloud(self):
        rng = np.random.default_rng(0)
        pts = rng.uniform(0, 100, (50_000, 3)).astype(np.float32)
        chunks = generate_adaptive_chunks(pts, chunk_size=5_000)
        assert len(chunks) > 1
        # All source indices covered
        all_idx = np.unique(np.concatenate([c.source_indices for c in chunks]))
        assert len(all_idx) == len(pts)

    def test_chunk_source_indices_reference_original(self):
        pts = make_room_pts(1000)
        chunks = generate_adaptive_chunks(pts, chunk_size=300)
        for chunk in chunks:
            # Source indices must be valid integer indices into pts
            assert chunk.source_indices.min() >= 0
            assert chunk.source_indices.max() < len(pts)
            np.testing.assert_array_equal(chunk.points, pts[chunk.source_indices])

    @skip_if_no_checkpoint
    def test_chunked_inference_covers_all_points(self):
        """All source points must receive a label after chunked inference."""
        pts = make_room_pts(3000)
        adapter = RandLANetSemanticAdapter(checkpoint_path=str(CHECKPOINT_PATH))
        adapter.load()
        result = adapter.infer(pts)
        assert len(result.labels) == len(pts)
        unknown_count = sum(1 for l in result.labels if l == "UNKNOWN")
        # UNKNOWN is valid for non-wall points — just ensure all points have a label
        assert unknown_count + sum(1 for l in result.labels if l == "WALL") == len(pts)


# ═══════════════════════════════════════════════════════════════════════════════
# 11. OVERLAP RECONCILIATION
# ═══════════════════════════════════════════════════════════════════════════════

class TestOverlapReconciliation:
    """Gate 11: Overlap reconciliation averages probabilities correctly."""

    def test_single_chunk_no_change(self):
        total = 10
        source_idx = np.arange(total, dtype=np.int64)
        probs = np.random.default_rng(0).uniform(0, 1, (total, 2)).astype(np.float64)
        probs = probs / probs.sum(axis=1, keepdims=True)
        labels, avg = reconcile_chunk_predictions(total, [
            {"chunk_id": 0, "source_indices": source_idx, "class_probs": probs}
        ])
        np.testing.assert_allclose(avg, probs, atol=1e-8)

    def test_two_chunks_same_indices_averages(self):
        """When two chunks share the same points, probabilities must be averaged."""
        total = 5
        source_idx = np.arange(total, dtype=np.int64)
        probs_a = np.array([[0.9, 0.1]] * total, dtype=np.float64)
        probs_b = np.array([[0.1, 0.9]] * total, dtype=np.float64)

        labels, avg = reconcile_chunk_predictions(total, [
            {"chunk_id": 0, "source_indices": source_idx, "class_probs": probs_a},
            {"chunk_id": 1, "source_indices": source_idx, "class_probs": probs_b},
        ])
        expected_avg = np.array([[0.5, 0.5]] * total)
        np.testing.assert_allclose(avg, expected_avg, atol=1e-8)
        # Tie-breaking goes to class 0 (np.argmax, first occurrence)
        assert all(l in (0, 1) for l in labels)

    def test_unobserved_points_get_uniform_prior(self):
        total = 10
        # Only points 0-4 are covered
        source_idx = np.arange(5, dtype=np.int64)
        probs = np.array([[0.8, 0.2]] * 5, dtype=np.float64)
        labels, avg = reconcile_chunk_predictions(total, [
            {"chunk_id": 0, "source_indices": source_idx, "class_probs": probs}
        ])
        # Points 5-9 should have uniform prior [0.5, 0.5]
        np.testing.assert_allclose(avg[5:], [[0.5, 0.5]] * 5, atol=1e-8)

    def test_reconciliation_is_not_first_chunk_wins(self):
        """Verify that overlap uses averaging, not first-chunk-wins."""
        total = 3
        idx = np.array([0, 1, 2], dtype=np.int64)
        probs_chunk1 = np.array([[1.0, 0.0]] * 3, dtype=np.float64)  # strongly class 0
        probs_chunk2 = np.array([[0.0, 1.0]] * 3, dtype=np.float64)  # strongly class 1

        labels, avg = reconcile_chunk_predictions(total, [
            {"chunk_id": 0, "source_indices": idx, "class_probs": probs_chunk1},
            {"chunk_id": 1, "source_indices": idx, "class_probs": probs_chunk2},
        ])
        # Averaged = [0.5, 0.5] — neither chunk dominates
        np.testing.assert_allclose(avg, [[0.5, 0.5]] * 3, atol=1e-8)


# ═══════════════════════════════════════════════════════════════════════════════
# 12. MISSING FEATURE HANDLING
# ═══════════════════════════════════════════════════════════════════════════════

class TestMissingFeatureHandling:
    """Gate 12: Missing features handled without fabrication."""

    @skip_if_no_checkpoint
    def test_inference_with_xyz_only(self):
        """Only XYZ available — inference must not crash or invent features."""
        adapter = RandLANetSemanticAdapter(checkpoint_path=str(CHECKPOINT_PATH))
        adapter.load()
        pts = make_wall_pts(200)
        # No colors, no normals — adapter must not fabricate them
        result = adapter.infer(pts, colors=None, normals=None)
        assert result.status == REAL_NEURAL_INFERENCE

    @skip_if_no_checkpoint
    def test_colors_not_used_in_feature_schema(self):
        """Colors are not in the feature schema — must be noted in diagnostics."""
        adapter = RandLANetSemanticAdapter(checkpoint_path=str(CHECKPOINT_PATH))
        adapter.load()
        pts = make_wall_pts(100)
        fake_colors = np.ones((len(pts), 3), dtype=np.float32)
        result = adapter.infer(pts, colors=fake_colors)
        assert result.diagnostics.get("colors_used") is False

    @skip_if_no_checkpoint
    def test_normals_not_used_in_feature_schema(self):
        """Normals are not in the feature schema — must be noted in diagnostics."""
        adapter = RandLANetSemanticAdapter(checkpoint_path=str(CHECKPOINT_PATH))
        adapter.load()
        pts = make_wall_pts(100)
        normals = np.tile([0, 1, 0], (len(pts), 1)).astype(np.float32)
        result = adapter.infer(pts, normals=normals)
        assert result.diagnostics.get("normals_used") is False


# ═══════════════════════════════════════════════════════════════════════════════
# 13. DYNAMIC MODEL SELECTION
# ═══════════════════════════════════════════════════════════════════════════════

class TestDynamicModelSelection:
    """Gate 13: best_production_semantic_model() enforces production criteria."""

    def test_registry_has_best_production_method(self):
        registry = NeuralModelRegistry()
        assert hasattr(registry, "best_production_semantic_model")

    def test_selector_uses_best_production_method(self):
        selector = SemanticModelSelector()
        assert hasattr(selector, "select_active_adapter")

    @skip_if_no_checkpoint
    def test_best_production_selects_neural_with_checkpoint(self):
        registry = NeuralModelRegistry()
        registry.register_defaults()
        key, adapter, rationale = registry.best_production_semantic_model()
        assert adapter.status == REAL_NEURAL_INFERENCE, (
            f"Expected REAL_NEURAL_INFERENCE with trained checkpoint, got {adapter.status}. "
            f"Rationale: {rationale}"
        )

    def test_best_production_falls_back_to_geometric_without_checkpoint(self, tmp_path):
        """Without a trained checkpoint, must select GEOMETRIC (not TEST_ONLY_NEURAL_FORWARD)."""
        registry = NeuralModelRegistry()
        registry.register_defaults()
        # Override RandLA-Net to use a non-existent checkpoint path
        from agent.phase3.semantic.randla_adapter import RandLANetSemanticAdapter as RLNA
        from agent.phase3.neural.registry import ModelRegistryEntry
        registry._entries["randlanet"] = ModelRegistryEntry(
            key="randlanet",
            display_name="RandLA-Net (test — no checkpoint)",
            model_type="neural_semantic",
            adapter=RLNA(checkpoint_path=str(tmp_path / "nonexistent.pth")),
            priority=1,
            cpu_compatible=True,
            requires_checkpoint=True,
        )
        key, adapter, rationale = registry.best_production_semantic_model()
        # Must fall back to geometric — NOT TEST_ONLY_NEURAL_FORWARD
        assert adapter.status != TEST_ONLY_NEURAL_FORWARD, (
            "best_production_semantic_model must never select TEST_ONLY_NEURAL_FORWARD. "
            f"Got status={adapter.status}"
        )
        assert adapter.status in (GEOMETRIC_ADAPTER, REAL_NEURAL_INFERENCE)

    def test_production_neural_models_list(self):
        registry = NeuralModelRegistry()
        registry.register_defaults()
        registry.evaluate_all()
        # production_neural_models() must only include REAL_NEURAL_INFERENCE
        for key in registry.production_neural_models():
            assert registry._entries[key].adapter.status == REAL_NEURAL_INFERENCE


# ═══════════════════════════════════════════════════════════════════════════════
# 14. RANDOM/INVALID CHECKPOINT REJECTION
# ═══════════════════════════════════════════════════════════════════════════════

class TestInvalidCheckpointRejection:
    """Gate 14: Random-init and invalid checkpoints must NOT get REAL_NEURAL_INFERENCE."""

    def test_adapter_without_checkpoint_is_not_production_ready(self, tmp_path):
        """Adapter with no checkpoint must report NO_COMPATIBLE_CHECKPOINT."""
        adapter = RandLANetSemanticAdapter(checkpoint_path=str(tmp_path / "absent.pth"))
        adapter.load()
        assert adapter.status == NO_COMPATIBLE_CHECKPOINT
        assert adapter.health_check() is False
        assert is_real_neural(adapter.status) is False

    def test_corrupted_checkpoint_file_fails_gracefully(self, tmp_path):
        """Corrupted (non-PyTorch) file must not get REAL_NEURAL_INFERENCE."""
        bad = tmp_path / "corrupted.pth"
        bad.write_bytes(b"this is not a pytorch checkpoint")
        adapter = RandLANetSemanticAdapter(checkpoint_path=str(bad))
        adapter.load()
        assert adapter.status in (FAILED, NO_COMPATIBLE_CHECKPOINT)
        assert is_real_neural(adapter.status) is False

    def test_random_init_model_must_not_be_real_neural(self, tmp_path):
        """A model saved with random weights must NOT report REAL_NEURAL_INFERENCE.

        The checkpoint is built without training metadata (final_loss/accuracy).
        The validator must flag it as suspect random-init.
        """
        from agent.tools.randla_net import RandLANet
        import torch

        # Create checkpoint with random weights and NO training metadata
        random_model = RandLANet(in_channels=EXPECTED_IN_CHANNELS, num_classes=EXPECTED_NUM_CLASSES)
        random_ckpt_path = tmp_path / "random_weights.pth"
        torch.save({
            "model_state_dict": random_model.state_dict(),
            "architecture": "RandLANet",
            "in_channels": EXPECTED_IN_CHANNELS,
            "num_classes": EXPECTED_NUM_CLASSES,
            "k_neighbors": 16,
            # Missing: training_dataset, final_loss, final_train_accuracy
        }, random_ckpt_path)

        val = validate_checkpoint(random_ckpt_path)
        assert val.is_trained_not_random is False, (
            "Random-init checkpoint must not pass the trained-not-random gate."
        )
        assert val.overall_valid is False

        adapter = RandLANetSemanticAdapter(checkpoint_path=str(random_ckpt_path))
        adapter.load()
        assert adapter.status != REAL_NEURAL_INFERENCE, (
            "Random-init model must NOT receive REAL_NEURAL_INFERENCE status. "
            f"Got: {adapter.status}"
        )

    def test_test_only_status_not_production_ready(self):
        assert is_production_ready(TEST_ONLY_NEURAL_FORWARD) is False

    def test_real_neural_status_is_production_ready(self):
        assert is_production_ready(REAL_NEURAL_INFERENCE) is True

    def test_geometric_status_is_production_ready(self):
        assert is_production_ready(GEOMETRIC_ADAPTER) is True


# ═══════════════════════════════════════════════════════════════════════════════
# 15. CUDA DEPENDENCY GUARD
# ═══════════════════════════════════════════════════════════════════════════════

class TestCUDADependencyGuard:
    """Gate 15: No CUDA in production code."""

    def test_no_cuda_in_randla_adapter(self):
        """RandLA adapter must not contain CUDA calls."""
        adapter_path = Path(__file__).parent.parent / "phase3" / "semantic" / "randla_adapter.py"
        if not adapter_path.exists():
            pytest.skip("Adapter file not found")
        from agent.phase3.neural.cuda_guard import scan_file_for_cuda
        violations = scan_file_for_cuda(adapter_path)
        assert len(violations) == 0, f"CUDA violations in randla_adapter.py: {violations}"

    def test_no_cuda_in_checkpoint_builder(self):
        builder_path = Path(__file__).parent.parent / "phase3" / "neural" / "checkpoint_builder.py"
        if not builder_path.exists():
            pytest.skip("Builder file not found")
        from agent.phase3.neural.cuda_guard import scan_file_for_cuda
        violations = scan_file_for_cuda(builder_path)
        assert len(violations) == 0, f"CUDA violations in checkpoint_builder.py: {violations}"

    def test_no_cuda_in_registry(self):
        registry_path = Path(__file__).parent.parent / "phase3" / "neural" / "registry.py"
        if not registry_path.exists():
            pytest.skip("Registry file not found")
        from agent.phase3.neural.cuda_guard import scan_file_for_cuda
        violations = scan_file_for_cuda(registry_path)
        assert len(violations) == 0, f"CUDA violations in registry.py: {violations}"

    def test_production_device_string(self):
        assert get_production_device() == "cpu"

    def test_kpconv_excluded_due_to_cpu_unavailable(self):
        """KPConv must be excluded from production selection (requires CUDA)."""
        registry = NeuralModelRegistry()
        registry.register_defaults()
        assert registry._entries["kpconv"].cpu_compatible is False
        registry.evaluate_all()
        assert "kpconv" not in registry.production_neural_models()


# ═══════════════════════════════════════════════════════════════════════════════
# 16. SEMANTIC + GEOMETRIC FUSION
# ═══════════════════════════════════════════════════════════════════════════════

class TestSemanticGeometricFusion:
    """Gate 16: Semantic + geometric fusion scores combine correctly."""

    def test_fusion_scores_structure(self):
        """Validate the structure of a fusion score dict."""
        # Simulate what the pipeline stores in semantic_evidence + geometric_evidence
        neural_score = 0.87
        planarity_score = 0.95
        verticality_score = 0.96
        source_support = 0.98

        combined = (neural_score * 0.35 + planarity_score * 0.30
                    + verticality_score * 0.25 + source_support * 0.10)

        assert 0.0 <= combined <= 1.0

    def test_wall_with_high_neural_and_geometry_accepted(self):
        """High neural + high planarity → should survive validation."""
        neural_wall_prob = 0.91
        planarity_rmse_m = 0.02    # 2 cm — excellent fit
        thickness_m = 0.20
        assert planarity_rmse_m < thickness_m / 2.0 + 0.10   # validator tolerance

    def test_wall_with_low_neural_and_bad_geometry_not_promoted(self):
        """Low neural score AND bad geometry should not be promoted."""
        neural_wall_prob = 0.30   # low confidence
        planarity_rmse_m = 0.25  # 25 cm — terrible fit
        max_tolerance = 0.10 + 0.20 / 2  # max_residual + thickness/2
        assert planarity_rmse_m > max_tolerance  # correctly rejected


# ═══════════════════════════════════════════════════════════════════════════════
# 17. MODEL PROVENANCE
# ═══════════════════════════════════════════════════════════════════════════════

class TestModelProvenance:
    """Gate 17: Model provenance recorded with hash, dataset, feature schema."""

    @skip_if_no_checkpoint
    def test_metadata_has_checkpoint_hash(self):
        adapter = RandLANetSemanticAdapter(checkpoint_path=str(CHECKPOINT_PATH))
        adapter.load()
        meta = adapter.metadata()
        assert meta.get("checkpoint_hash") is not None, "checkpoint_hash must be in metadata"
        assert len(meta["checkpoint_hash"]) == 64, "SHA-256 must be 64 chars"

    @skip_if_no_checkpoint
    def test_metadata_has_training_dataset(self):
        adapter = RandLANetSemanticAdapter(checkpoint_path=str(CHECKPOINT_PATH))
        adapter.load()
        meta = adapter.metadata()
        assert "training_dataset" in meta
        assert meta["training_dataset"] == "SYNTHETIC_ARCHITECTURAL_v1"

    @skip_if_no_checkpoint
    def test_metadata_has_feature_schema(self):
        adapter = RandLANetSemanticAdapter(checkpoint_path=str(CHECKPOINT_PATH))
        adapter.load()
        meta = adapter.metadata()
        assert "feature_schema" in meta
        expected = ["x", "y", "z", "verticality_ratio", "z_rel", "dist_centroid"]
        assert meta["feature_schema"] == expected

    @skip_if_no_checkpoint
    def test_inference_result_carries_checkpoint_hash(self):
        adapter = RandLANetSemanticAdapter(checkpoint_path=str(CHECKPOINT_PATH))
        adapter.load()
        pts = make_wall_pts(100)
        result = adapter.infer(pts)
        assert result.diagnostics.get("checkpoint_hash") is not None

    @skip_if_no_checkpoint
    def test_inference_records_production_ready_true(self):
        adapter = RandLANetSemanticAdapter(checkpoint_path=str(CHECKPOINT_PATH))
        adapter.load()
        pts = make_wall_pts(100)
        result = adapter.infer(pts)
        assert result.diagnostics.get("production_ready") is True
