"""Orientation Estimation — Phase 2 Spatial Foundation.

Estimates the vertical axis and horizontal frame from point cloud geometry
without assuming a fixed up-direction.

Methods (in priority order):
  1. Surface normal distribution analysis
  2. Dominant horizontal plane (floor/slab) detection
  3. PCA eigenvalue analysis
  4. Configurable default fallback with AMBIGUOUS status

The estimated vertical axis is always normalized.
If evidence is insufficient, ORIENTATION_AMBIGUOUS is reported — the pipeline
never silently forces Z-up.
"""

from __future__ import annotations

import os

import numpy as np
import structlog

from agent.tools.coordinate_system import (
    ORIENTATION_AMBIGUOUS,
    ORIENTATION_INFERRED,
    ORIENTATION_VERIFIED,
)

logger = structlog.get_logger()

# ── Configuration ─────────────────────────────────────────────────────────────

# Minimum cosine similarity between independent estimates for agreement
_CONSENSUS_THRESHOLD: float = 0.95

# Maximum number of points to use for normal estimation (performance guard)
_MAX_POINTS_FOR_ORIENTATION: int = 100_000

# Minimum number of points for reliable estimation
_MIN_POINTS_FOR_ORIENTATION: int = 500


def estimate_vertical_axis(
    points: np.ndarray,
    normals: np.ndarray | None = None,
) -> tuple[np.ndarray, str, dict]:
    """Estimate the vertical (up) axis from point cloud geometry.

    Combines up to three independent methods and reports consensus.

    Args:
        points: (N, 3) array of 3D points (any consistent unit).
        normals: Optional (N, 3) array of surface normals. If None,
                 only PCA-based estimation is available.

    Returns:
        (up_axis, confidence, evidence)
        up_axis:     normalized (3,) unit vector
        confidence:  ORIENTATION_VERIFIED / INFERRED / AMBIGUOUS
        evidence:    dict with per-method results
    """
    pts = np.asarray(points, dtype=float)
    evidence: dict = {}

    if len(pts) < _MIN_POINTS_FOR_ORIENTATION:
        logger.warning(
            "orientation_insufficient_points",
            n_points=len(pts),
            min_required=_MIN_POINTS_FOR_ORIENTATION,
        )
        default_up = _configured_default()
        evidence["reason"] = f"Only {len(pts)} points (need {_MIN_POINTS_FOR_ORIENTATION})"
        return default_up, ORIENTATION_AMBIGUOUS, evidence

    # Subsample for performance
    if len(pts) > _MAX_POINTS_FOR_ORIENTATION:
        idx = np.random.default_rng(42).choice(
            len(pts), _MAX_POINTS_FOR_ORIENTATION, replace=False
        )
        pts_sub = pts[idx]
        normals_sub = normals[idx] if normals is not None else None
    else:
        pts_sub = pts
        normals_sub = normals

    estimates: list[tuple[np.ndarray, float, str]] = []  # (axis, weight, method)

    # ── Method 1: Surface normal distribution ─────────────────────────────
    if normals_sub is not None and len(normals_sub) > 100:
        axis_norm, weight_norm, ev_norm = _estimate_from_normals(normals_sub)
        estimates.append((axis_norm, weight_norm, "normal_distribution"))
        evidence["normal_distribution"] = ev_norm

    # ── Method 2: Dominant horizontal plane detection ─────────────────────
    if normals_sub is not None and len(normals_sub) > 100:
        axis_plane, weight_plane, ev_plane = _estimate_from_dominant_planes(
            pts_sub, normals_sub
        )
        if weight_plane > 0.1:
            estimates.append((axis_plane, weight_plane, "dominant_plane"))
            evidence["dominant_plane"] = ev_plane

    # ── Method 3: PCA (smallest eigenvalue direction) ─────────────────────
    axis_pca, weight_pca, ev_pca = _estimate_from_pca(pts_sub)
    if weight_pca > 0.1:
        estimates.append((axis_pca, weight_pca, "pca"))
        evidence["pca"] = ev_pca

    if not estimates:
        default_up = _configured_default()
        evidence["reason"] = "No estimation method produced a result"
        return default_up, ORIENTATION_AMBIGUOUS, evidence

    # ── Consensus ─────────────────────────────────────────────────────────
    # Weight-average the estimates, checking for agreement
    best_axis, best_weight, best_method = max(estimates, key=lambda x: x[1])

    # Ensure consistent sign (flip if necessary so Z-component is positive or largest)
    best_axis = _canonical_sign(best_axis)

    if len(estimates) >= 2:
        # Check agreement between the best and other estimates
        agreement_count = 0
        for axis, weight, method in estimates:
            axis = _canonical_sign(axis)
            cosine = abs(float(np.dot(best_axis, axis)))
            if cosine >= _CONSENSUS_THRESHOLD:
                agreement_count += 1

        if agreement_count == len(estimates):
            # All methods agree
            confidence = ORIENTATION_VERIFIED if len(estimates) >= 2 else ORIENTATION_INFERRED
            evidence["consensus"] = f"All {len(estimates)} methods agree"
        elif agreement_count >= 2:
            confidence = ORIENTATION_INFERRED
            evidence["consensus"] = f"{agreement_count}/{len(estimates)} methods agree"
        else:
            confidence = ORIENTATION_AMBIGUOUS
            evidence["consensus"] = "Methods disagree"
    else:
        confidence = ORIENTATION_INFERRED
        evidence["consensus"] = f"Single method: {best_method}"

    evidence["selected_method"] = best_method
    evidence["up_axis"] = best_axis.tolist()

    logger.info(
        "vertical_axis_estimated",
        up_axis=best_axis.tolist(),
        confidence=confidence,
        method=best_method,
    )
    return best_axis, confidence, evidence


def estimate_horizontal_frame(
    up_axis: np.ndarray,
    points: np.ndarray | None = None,
    normals: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray, str]:
    """Derive two orthogonal horizontal axes from the estimated vertical axis.

    If points and normals are provided, attempts to find dominant wall
    directions to establish a scene-aligned horizontal frame. Otherwise,
    generates an arbitrary orthogonal frame.

    Args:
        up_axis: (3,) normalized vertical axis.
        points: Optional (N, 3) points for wall direction analysis.
        normals: Optional (N, 3) normals for wall direction analysis.

    Returns:
        (h1, h2, confidence) where h1, h2 are (3,) unit vectors
        orthogonal to up_axis and to each other.
    """
    up = np.asarray(up_axis, dtype=float)
    up = up / np.linalg.norm(up)

    if points is not None and normals is not None and len(normals) > 50:
        h1, h2, conf = _horizontal_from_wall_directions(up, points, normals)
        if conf != ORIENTATION_AMBIGUOUS:
            return h1, h2, conf

    # Arbitrary orthogonal frame via Gram-Schmidt
    h1, h2 = _arbitrary_horizontal_frame(up)
    return h1, h2, ORIENTATION_AMBIGUOUS


def build_canonical_frame(
    up_axis: np.ndarray,
    h1: np.ndarray,
    h2: np.ndarray,
) -> np.ndarray:
    """Build a 3×3 rotation matrix from source to canonical frame.

    The canonical frame is defined as:
        - Column 0 = h1 (horizontal axis 1)
        - Column 1 = h2 (horizontal axis 2)
        - Column 2 = up  (vertical axis)

    The rotation matrix R satisfies: canonical = R @ source

    Args:
        up_axis: (3,) vertical axis.
        h1: (3,) first horizontal axis.
        h2: (3,) second horizontal axis.

    Returns:
        (3, 3) rotation matrix.
    """
    up = np.asarray(up_axis, dtype=float)
    up = up / np.linalg.norm(up)
    e1 = np.asarray(h1, dtype=float)
    e1 = e1 / np.linalg.norm(e1)
    e2 = np.asarray(h2, dtype=float)
    e2 = e2 / np.linalg.norm(e2)

    # R maps source to canonical: rows are the new basis in source coords
    R = np.array([e1, e2, up], dtype=float)

    # Verify orthogonality
    orth_err = float(np.max(np.abs(R @ R.T - np.eye(3))))
    if orth_err > 1e-6:
        # Force orthogonality via Gram-Schmidt
        e1 = e1 - np.dot(e1, up) * up
        e1 = e1 / np.linalg.norm(e1)
        e2 = np.cross(up, e1)
        R = np.array([e1, e2, up], dtype=float)

    # Ensure proper rotation (det = +1)
    if np.linalg.det(R) < 0:
        R[1] = -R[1]

    return R


# ── Internal Estimation Methods ───────────────────────────────────────────────


def _estimate_from_normals(
    normals: np.ndarray,
) -> tuple[np.ndarray, float, dict]:
    """Estimate vertical axis from surface normal distribution.

    The vertical axis has a bimodal normal distribution: many normals are
    nearly parallel to it (floors/ceilings) or nearly perpendicular (walls).
    We look for the axis where |n·axis| has the strongest bimodal signal.
    """
    n = np.asarray(normals, dtype=float)
    norms = np.linalg.norm(n, axis=1)
    valid = norms > 1e-6
    n = n[valid]
    n = n / norms[valid, None]

    best_axis = np.array([0.0, 0.0, 1.0])
    best_score = 0.0

    # Test each cardinal axis and several intermediate angles
    candidate_axes = [
        np.array([1.0, 0.0, 0.0]),
        np.array([0.0, 1.0, 0.0]),
        np.array([0.0, 0.0, 1.0]),
    ]

    for axis in candidate_axes:
        # Compute |n · axis| for all normals
        projections = np.abs(n @ axis)
        # A good vertical axis has many normals near 0 (walls) and near 1 (floors)
        # Score = proportion near 0 (< 0.25) × proportion near 1 (> 0.85)
        near_zero = np.mean(projections < 0.25)
        near_one = np.mean(projections > 0.85)
        score = near_zero * near_one

        if score > best_score:
            best_score = score
            # Refine the axis using normals that are nearly parallel to it
            parallel_mask = projections > 0.85
            if np.sum(parallel_mask) >= 3:
                # Average the nearly-parallel normals
                parallel_n = n[parallel_mask]
                # Make signs consistent
                signs = np.sign(parallel_n @ axis)
                signs[signs == 0] = 1.0
                parallel_n = parallel_n * signs[:, None]
                refined = parallel_n.mean(axis=0)
                refined_norm = np.linalg.norm(refined)
                if refined_norm > 1e-6:
                    best_axis = refined / refined_norm
                else:
                    best_axis = axis.copy()
            else:
                best_axis = axis.copy()

    weight = min(best_score * 5.0, 1.0)  # Scale to [0, 1]
    ev = {
        "score": round(best_score, 4),
        "weight": round(weight, 4),
        "axis": best_axis.tolist(),
    }
    return best_axis, weight, ev


def _estimate_from_dominant_planes(
    points: np.ndarray,
    normals: np.ndarray,
) -> tuple[np.ndarray, float, dict]:
    """Estimate vertical from dominant horizontal planes (floor/ceiling candidates).

    Large horizontal planes have normals closely aligned with the vertical axis.
    """
    n = np.asarray(normals, dtype=float)
    norms = np.linalg.norm(n, axis=1)
    valid = norms > 1e-6
    n_valid = n[valid] / norms[valid, None]

    # Find the dominant normal direction using histogram of the Z-components
    # For each candidate up-axis, count how many normals are near-parallel
    best_axis = np.array([0.0, 0.0, 1.0])
    best_count = 0

    for axis in [
        np.array([1.0, 0.0, 0.0]),
        np.array([0.0, 1.0, 0.0]),
        np.array([0.0, 0.0, 1.0]),
    ]:
        projections = np.abs(n_valid @ axis)
        near_parallel = projections > 0.85
        count = int(np.sum(near_parallel))
        if count > best_count:
            best_count = count
            # Refine axis from near-parallel normals
            if count >= 5:
                pn = n_valid[near_parallel]
                signs = np.sign(pn @ axis)
                signs[signs == 0] = 1.0
                pn = pn * signs[:, None]
                refined = pn.mean(axis=0)
                rn = np.linalg.norm(refined)
                best_axis = refined / rn if rn > 1e-6 else axis.copy()
            else:
                best_axis = axis.copy()

    total_points = len(n_valid)
    ratio = best_count / max(total_points, 1)
    weight = min(ratio * 3.0, 1.0)  # At least ~33% should be floor/ceiling

    ev = {
        "horizontal_plane_points": best_count,
        "total_points": total_points,
        "ratio": round(ratio, 4),
        "weight": round(weight, 4),
        "axis": best_axis.tolist(),
    }
    return best_axis, weight, ev


def _estimate_from_pca(
    points: np.ndarray,
) -> tuple[np.ndarray, float, dict]:
    """Estimate vertical axis from PCA of point distribution.

    For a building scan, the smallest principal component typically corresponds
    to the vertical direction (buildings are wider/longer than they are tall),
    but this depends on the building's proportions.
    """
    pts = np.asarray(points, dtype=float)
    centroid = pts.mean(axis=0)
    centered = pts - centroid

    try:
        cov = np.cov(centered.T)  # (3, 3)
        eigenvalues, eigenvectors = np.linalg.eigh(cov)

        # Sort by eigenvalue (ascending)
        sort_idx = np.argsort(eigenvalues)
        eigenvalues = eigenvalues[sort_idx]
        eigenvectors = eigenvectors[:, sort_idx]

        # The smallest eigenvalue direction is a candidate for vertical
        # But only if the eigenvalue ratios suggest a planar distribution
        ratios = eigenvalues / max(eigenvalues[-1], 1e-12)
        ev = {
            "eigenvalue_ratios": [round(float(r), 4) for r in ratios],
            "smallest_eigenvector": eigenvectors[:, 0].tolist(),
        }

        # If the smallest eigenvalue is significantly smaller than the others,
        # the point cloud is planar → smallest eigenvector ≈ vertical
        if ratios[0] < 0.3:  # Significant planarity
            axis = eigenvectors[:, 0].copy()
            weight = min((0.3 - ratios[0]) * 3.0, 0.7)
            ev["weight"] = round(weight, 4)
            ev["axis"] = axis.tolist()
            return axis, weight, ev
        else:
            ev["weight"] = 0.0
            ev["reason"] = "Point cloud not sufficiently planar for PCA vertical estimation"
            return np.array([0.0, 0.0, 1.0]), 0.0, ev

    except np.linalg.LinAlgError:
        return (
            np.array([0.0, 0.0, 1.0]),
            0.0,
            {"error": "PCA computation failed"},
        )


def _horizontal_from_wall_directions(
    up: np.ndarray,
    points: np.ndarray,
    normals: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, str]:
    """Find dominant wall directions to establish horizontal frame.

    Projects wall normals onto the horizontal plane and finds dominant directions
    via circular histogram.
    """
    n = np.asarray(normals, dtype=float)
    norms = np.linalg.norm(n, axis=1)
    valid = norms > 1e-6
    n_valid = n[valid] / norms[valid, None]

    # Select vertical planes (walls): |n·up| < 0.25
    projections = np.abs(n_valid @ up)
    wall_mask = projections < 0.25
    wall_normals = n_valid[wall_mask]

    if len(wall_normals) < 20:
        h1, h2 = _arbitrary_horizontal_frame(up)
        return h1, h2, ORIENTATION_AMBIGUOUS

    # Project wall normals onto horizontal plane
    horizontal = wall_normals - np.outer(wall_normals @ up, up)
    h_norms = np.linalg.norm(horizontal, axis=1)
    valid_h = h_norms > 1e-6
    horizontal = horizontal[valid_h] / h_norms[valid_h, None]

    if len(horizontal) < 10:
        h1, h2 = _arbitrary_horizontal_frame(up)
        return h1, h2, ORIENTATION_AMBIGUOUS

    # Compute angles in [0, 180°) — wall normals are sign-ambiguous
    angles = np.arctan2(horizontal[:, 1], horizontal[:, 0]) % np.pi
    angles_deg = np.degrees(angles)

    # Histogram to find dominant direction
    hist, edges = np.histogram(angles_deg, bins=36, range=(0, 180))
    peak_bin = np.argmax(hist)
    peak_angle = (edges[peak_bin] + edges[peak_bin + 1]) / 2.0

    # Dominant horizontal axis 1
    rad1 = np.radians(peak_angle)
    h1 = np.array([np.cos(rad1), np.sin(rad1), 0.0])
    # Remove any vertical component
    h1 = h1 - np.dot(h1, up) * up
    h1 = h1 / np.linalg.norm(h1)

    # h2 = up × h1
    h2 = np.cross(up, h1)
    h2 = h2 / np.linalg.norm(h2)

    return h1, h2, ORIENTATION_INFERRED


def _arbitrary_horizontal_frame(
    up: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Generate an arbitrary orthogonal horizontal frame via Gram-Schmidt."""
    up = up / np.linalg.norm(up)

    # Pick a vector not parallel to up
    if abs(up[0]) < 0.9:
        seed = np.array([1.0, 0.0, 0.0])
    else:
        seed = np.array([0.0, 1.0, 0.0])

    h1 = seed - np.dot(seed, up) * up
    h1 = h1 / np.linalg.norm(h1)
    h2 = np.cross(up, h1)
    h2 = h2 / np.linalg.norm(h2)
    return h1, h2


def _canonical_sign(axis: np.ndarray) -> np.ndarray:
    """Ensure consistent sign: prefer the direction with largest positive component."""
    max_idx = int(np.argmax(np.abs(axis)))
    if axis[max_idx] < 0:
        return -axis
    return axis


def _configured_default() -> np.ndarray:
    """Return the configured default up-axis from environment or Z."""
    up_name = os.environ.get("STB_UP_AXIS", "z").strip().lower()
    mapping = {
        "x": np.array([1.0, 0.0, 0.0]),
        "y": np.array([0.0, 1.0, 0.0]),
        "z": np.array([0.0, 0.0, 1.0]),
    }
    return mapping.get(up_name, mapping["z"])
