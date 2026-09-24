"""determinism.py — pin every RNG + content hash utilities for reproducible runs.

NQA-1 Subpart 2.7 § 401 requires that V&V evidence be reproducible: an
independent reviewer must be able to re-execute the pipeline on the same
inputs and obtain the same outputs. Two obstacles without this module:

  1. RANSAC, DBSCAN, and synthetic-data generation all draw random samples.
     Without pinned seeds, two runs of the same scan produce slightly
     different segment counts, bounding boxes, and confidences.
  2. UUIDs for segment_id are created with uuid4(), so even a byte-for-byte
     identical geometry stream gets a different identifier every run.

This module provides:

  set_global_seed(seed)       Seeds random, numpy, Open3D in one call.
  deterministic_segment_id()  uuid5 derived from session+shape+centroid+bbox.
  compute_input_hash(path)    SHA-256 of the raw scan file.
  compute_output_hash(segs)   SHA-256 of canonical JSON of GeometrySegments.
  ProvenanceRecord            Dataclass holding seed + hashes + identity.

The default seed is 42, overridable via the STB_RANDOM_SEED env var so a
reviewer can ask for a specific seed to match an earlier V&V run.
"""

from __future__ import annotations

import hashlib
import json
import os
import random
import uuid
from dataclasses import asdict, dataclass
from pathlib import Path

# Namespace UUID for deriving deterministic segment IDs.  This UUID is
# project-specific — do not change it, or all prior segment_ids would
# renumber and break cross-run traceability.
_STB_NAMESPACE_UUID = uuid.UUID("8a2b4c6d-1e3f-5a7b-9c0d-e1f2a3b4c5d6")

DEFAULT_SEED = 42


# ── RNG seeding ──────────────────────────────────────────────────────────────


def resolve_seed(seed: int | None = None) -> int:
    """Pick a seed for this run.

    Precedence: explicit argument > STB_RANDOM_SEED env var > DEFAULT_SEED.
    """
    if seed is not None:
        return int(seed)
    env = os.environ.get("STB_RANDOM_SEED")
    if env is not None:
        try:
            return int(env)
        except ValueError:
            pass
    return DEFAULT_SEED


def set_global_seed(seed: int | None = None) -> int:
    """Seed every RNG the pipeline uses. Returns the seed that was applied.

    Call once at the start of each segmentation run. Safe to call multiple
    times — idempotent for the same seed.
    """
    s = resolve_seed(seed)
    random.seed(s)
    try:
        import numpy as np

        np.random.seed(s)  # legacy global RNG
    except ImportError:
        pass
    try:
        import open3d as o3d

        o3d.utility.random.seed(s)  # Open3D's C++ RNG
    except ImportError:
        pass
    except AttributeError:
        # older Open3D versions don't expose utility.random
        pass
    return s


# ── Deterministic segment IDs ────────────────────────────────────────────────


def deterministic_segment_id(
    session_id: str,
    shape: str,
    centroid_mm: tuple[float, float, float],
    bbox_mm: tuple[float, float, float, float, float, float],
) -> str:
    """Derive a stable segment_id from the session + geometry signature.

    Rounded coordinates (0.1 mm) so tiny float noise doesn't change the ID.
    Uses uuid5 under the project namespace — guaranteed valid UUID format
    so any existing uuid-field validators keep working.
    """
    cx, cy, cz = centroid_mm
    x0, y0, z0, x1, y1, z1 = bbox_mm
    key = (
        f"{session_id}|{shape}|"
        f"{round(cx, 1)}|{round(cy, 1)}|{round(cz, 1)}|"
        f"{round(x0, 1)}|{round(y0, 1)}|{round(z0, 1)}|"
        f"{round(x1, 1)}|{round(y1, 1)}|{round(z1, 1)}"
    )
    return str(uuid.uuid5(_STB_NAMESPACE_UUID, key))


# ── Content hashing ──────────────────────────────────────────────────────────


def compute_input_hash(path: Path, chunk_size: int = 1 << 20) -> str:
    """SHA-256 of a scan file. Streams so memory use is bounded."""
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while True:
            block = f.read(chunk_size)
            if not block:
                break
            h.update(block)
    return h.hexdigest()


def _canonical_segment_dict(seg) -> dict:
    """Canonical dict-form of a GeometrySegment for stable hashing.

    Excludes fields that don't affect semantic output: scan_station_id,
    source_file, and the segment_id itself (the id is derivable).
    Includes tags sorted by key so insertion order doesn't affect the hash.
    """
    bb = seg.bounding_box
    c = seg.centroid
    tags = dict(sorted(seg.tags.items())) if seg.tags else {}
    d = {
        "zone_id": seg.zone_id,
        "shape": seg.shape.value if hasattr(seg.shape, "value") else str(seg.shape),
        "centroid": [round(c.x, 3), round(c.y, 3), round(c.z, 3)],
        "bbox": [
            round(bb.min_x, 3),
            round(bb.min_y, 3),
            round(bb.min_z, 3),
            round(bb.max_x, 3),
            round(bb.max_y, 3),
            round(bb.max_z, 3),
        ],
        "point_count": int(seg.point_count),
        "confidence": round(float(seg.confidence), 4),
        "tags": tags,
    }
    if seg.normal is not None:
        n = seg.normal
        d["normal"] = [round(n.x, 4), round(n.y, 4), round(n.z, 4)]
    return d


def compute_output_hash(segments: list) -> str:
    """SHA-256 over a canonical JSON serialisation of all segments.

    Segments are sorted by (zone_id, shape, centroid) before hashing so the
    order in which detection stages produced them doesn't affect the result.
    """
    canonical = [_canonical_segment_dict(s) for s in segments]
    canonical.sort(
        key=lambda d: (
            d["zone_id"],
            d["shape"],
            tuple(d["centroid"]),
        )
    )
    payload = json.dumps(canonical, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


# ── Provenance record ────────────────────────────────────────────────────────


@dataclass(frozen=True)
class ProvenanceRecord:
    """Everything a reviewer needs to reproduce a session's output."""

    session_id: str
    random_seed: int
    software_identity_hash: str
    input_hash: str | None  # SHA-256 of source scan file
    input_source: str | None  # filename or "synthetic" | "plugin"
    output_hash: str  # SHA-256 of canonical segments
    segment_count: int
    generated_at_utc: str

    def to_dict(self) -> dict:
        return asdict(self)
