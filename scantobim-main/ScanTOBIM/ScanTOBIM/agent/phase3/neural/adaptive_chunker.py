"""Adaptive Chunking + Overlap Reconciliation for Large-Scale CPU Inference.

Engineering Rules:
  - Chunk size derived from available RAM, model footprint, point density.
  - A fixed `35000` is only a starting parameter, never a hardcoded universal.
  - When a source point appears in multiple overlapping chunks, combine predictions
    via probability averaging (not first-chunk-wins).
  - Source indices are always preserved through the full pipeline.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import structlog

logger = structlog.get_logger()

# Default adaptive chunk parameters
DEFAULT_CHUNK_SIZE: int = 35_000          # starting point
MIN_CHUNK_SIZE: int = 2_000              # never go below this
MAX_CHUNK_SIZE: int = 100_000            # never exceed this (RAM limit)
DEFAULT_OVERLAP_FRACTION: float = 0.15   # 15% overlap between chunks
MODEL_BYTES_PER_POINT: float = 6 * 4 + 16 * 4 + 32 * 4  # rough RandLA-Net activation estimate


# ── Adaptive chunk size ───────────────────────────────────────────────────────

def compute_adaptive_chunk_size(
    total_points: int,
    available_ram_bytes: int | None = None,
    in_channels: int = 6,
    hidden_dim: int = 32,
    safety_factor: float = 0.25,
) -> int:
    """Compute a safe chunk size based on available RAM and model footprint.

    Args:
        total_points: Total number of input points.
        available_ram_bytes: Available RAM in bytes. If None, probes psutil.
        in_channels: Feature dimensionality.
        hidden_dim: Approximate hidden dimension of the model.
        safety_factor: Fraction of available RAM to use (conservative).

    Returns:
        Chunk size in number of points.
    """
    if available_ram_bytes is None:
        try:
            import psutil
            available_ram_bytes = psutil.virtual_memory().available
        except ImportError:
            available_ram_bytes = 2 * 1024 ** 3  # conservative 2 GB fallback

    # Bytes per point: input feats + intermediate activations (rough)
    bytes_per_point = (in_channels + hidden_dim * 4) * 4  # float32
    safe_bytes = available_ram_bytes * safety_factor
    computed = int(safe_bytes / max(bytes_per_point, 1))

    chunk_size = max(MIN_CHUNK_SIZE, min(computed, MAX_CHUNK_SIZE, total_points))

    logger.debug(
        "adaptive_chunk_size_computed",
        total_points=total_points,
        available_ram_gb=round(available_ram_bytes / 1e9, 2),
        computed_chunk=computed,
        final_chunk=chunk_size,
    )
    return chunk_size


# ── Spatial chunk generator ───────────────────────────────────────────────────

@dataclass
class PointCloudChunk:
    """One spatial chunk of a point cloud with source index tracking."""

    chunk_id: int
    source_indices: np.ndarray      # (M,) indices into the original N-point cloud
    points: np.ndarray              # (M, 3) coordinates
    features: np.ndarray | None     # (M, C) or None


def generate_adaptive_chunks(
    points: np.ndarray,
    features: np.ndarray | None = None,
    chunk_size: int | None = None,
    overlap_fraction: float = DEFAULT_OVERLAP_FRACTION,
    available_ram_bytes: int | None = None,
) -> list[PointCloudChunk]:
    """Partition point cloud into overlapping spatial chunks.

    Uses a Z-column (XY-grid) decomposition to produce spatially coherent
    chunks that respect local structure better than random subsets.

    Args:
        points: (N, 3) coordinates.
        features: (N, C) optional pre-computed features.
        chunk_size: Target points per chunk. Computed adaptively if None.
        overlap_fraction: How much neighbouring chunks overlap.
        available_ram_bytes: RAM available for adaptive sizing.

    Returns:
        List of PointCloudChunk objects (each with source_indices).
    """
    N = len(points)
    if N == 0:
        return []

    if chunk_size is None:
        chunk_size = compute_adaptive_chunk_size(N, available_ram_bytes=available_ram_bytes)

    if N <= chunk_size:
        # Single chunk — no splitting needed
        return [PointCloudChunk(
            chunk_id=0,
            source_indices=np.arange(N, dtype=np.int64),
            points=points,
            features=features,
        )]

    # XY-grid decomposition
    mins = points[:, :2].min(axis=0)
    maxs = points[:, :2].max(axis=0)
    extents = maxs - mins

    # Target cells per dimension
    n_cells = max(1, int(math.ceil(N / chunk_size)))
    cells_per_dim = max(1, int(math.ceil(n_cells ** 0.5)))
    cell_size = extents / cells_per_dim
    overlap_m = extents * overlap_fraction / cells_per_dim

    chunks: list[PointCloudChunk] = []
    chunk_id = 0

    for cx in range(cells_per_dim):
        x_min = mins[0] + cx * cell_size[0] - overlap_m[0]
        x_max = mins[0] + (cx + 1) * cell_size[0] + overlap_m[0]

        for cy in range(cells_per_dim):
            y_min = mins[1] + cy * cell_size[1] - overlap_m[1]
            y_max = mins[1] + (cy + 1) * cell_size[1] + overlap_m[1]

            mask = (
                (points[:, 0] >= x_min) & (points[:, 0] <= x_max) &
                (points[:, 1] >= y_min) & (points[:, 1] <= y_max)
            )
            idx = np.where(mask)[0]
            if len(idx) == 0:
                continue

            chunk_pts = points[idx]
            chunk_feats = features[idx] if features is not None else None

            chunks.append(PointCloudChunk(
                chunk_id=chunk_id,
                source_indices=idx.astype(np.int64),
                points=chunk_pts,
                features=chunk_feats,
            ))
            chunk_id += 1

    logger.debug(
        "chunks_generated",
        total_points=N,
        chunk_size=chunk_size,
        num_chunks=len(chunks),
        overlap_fraction=overlap_fraction,
    )
    return chunks


# ── Overlap reconciliation ────────────────────────────────────────────────────

@dataclass
class ReconciliationRecord:
    """Tracking record for one source point across overlapping chunks."""

    source_point_index: int
    chunk_ids: list[int] = field(default_factory=list)
    class_probability_accumulator: np.ndarray | None = None  # (num_classes,)
    observation_count: int = 0

    def add_observation(self, chunk_id: int, class_probs: np.ndarray) -> None:
        self.chunk_ids.append(chunk_id)
        if self.class_probability_accumulator is None:
            self.class_probability_accumulator = class_probs.copy().astype(np.float64)
        else:
            self.class_probability_accumulator += class_probs.astype(np.float64)
        self.observation_count += 1

    def final_probabilities(self) -> np.ndarray:
        if self.class_probability_accumulator is None or self.observation_count == 0:
            return np.zeros(2, dtype=np.float64)
        return self.class_probability_accumulator / self.observation_count

    def final_label(self) -> int:
        return int(np.argmax(self.final_probabilities()))


def reconcile_chunk_predictions(
    total_points: int,
    chunk_predictions: list[dict[str, Any]],
    num_classes: int = 2,
) -> tuple[np.ndarray, np.ndarray]:
    """Merge overlapping chunk predictions via probability averaging.

    For each source point that appears in multiple chunks, average the
    class probability vectors across all chunks.

    Args:
        total_points: Total source cloud size.
        chunk_predictions: List of dicts:
            {
                "chunk_id": int,
                "source_indices": np.ndarray (M,),
                "class_probs": np.ndarray (M, num_classes),
            }
        num_classes: Number of semantic classes.

    Returns:
        labels: (N,) int64 final labels for each source point.
        probs:  (N, num_classes) float64 averaged probabilities.
    """
    # Accumulators
    prob_sum = np.zeros((total_points, num_classes), dtype=np.float64)
    obs_count = np.zeros(total_points, dtype=np.int32)

    for pred in chunk_predictions:
        src_idx = pred["source_indices"]   # (M,)
        cls_prob = pred["class_probs"]     # (M, num_classes)
        np.add.at(prob_sum, src_idx, cls_prob)
        np.add.at(obs_count, src_idx, 1)

    # Average where observed; leave zeros for unobserved points
    observed = obs_count > 0
    prob_avg = np.zeros((total_points, num_classes), dtype=np.float64)
    prob_avg[observed] = prob_sum[observed] / obs_count[observed, np.newaxis]

    # For unobserved points: uniform distribution
    if not np.all(observed):
        n_unobserved = int((~observed).sum())
        logger.warning(
            "unobserved_source_points",
            count=n_unobserved,
            note="Points outside all chunk boundaries; using uniform prior.",
        )
        prob_avg[~observed] = 1.0 / num_classes

    labels = prob_avg.argmax(axis=1).astype(np.int64)

    # Coverage stats
    coverage_pct = float(observed.sum()) / total_points * 100
    overlap_pts = int((obs_count > 1).sum())
    logger.info(
        "chunk_reconciliation_complete",
        total_points=total_points,
        coverage_pct=round(coverage_pct, 2),
        overlap_points=overlap_pts,
        chunks_merged=len(chunk_predictions),
    )

    return labels, prob_avg
