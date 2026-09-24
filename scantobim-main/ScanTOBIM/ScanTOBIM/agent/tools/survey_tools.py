"""Survey coordinate alignment tools.

Computes a rigid transform (rotation + translation, no scale) from scan-frame
to world / project coordinates using a minimum of 3 control point pairs via
the Kabsch / SVD algorithm.

Also provides a pre-segmentation scan quality gate to reject poor scans before
wasting compute on bad data.
"""

from __future__ import annotations

import hashlib
import hmac
from dataclasses import dataclass

import numpy as np
import structlog

from agent.models import (
    BoundingBox,
    GeometrySegment,
    Point3D,
    ScanQualityResult,
    SurveyControlPoint,
)

logger = structlog.get_logger()


# ── Rigid Transform ────────────────────────────────────────────────────────────


@dataclass
class RigidTransform:
    """3-D rigid transform: world_pt = R @ scan_pt + t  (coordinates in mm).

    R:       (3, 3) orthonormal rotation matrix
    t:       (3,)   translation vector in mm
    rmse_mm: root-mean-square residual of control points after fitting
    """

    R: np.ndarray
    t: np.ndarray
    rmse_mm: float

    def as_dict(self) -> dict:
        return {
            "R": self.R.tolist(),
            "t": self.t.tolist(),
            "rmse_mm": round(self.rmse_mm, 3),
        }


def compute_rigid_transform(
    control_points: list[SurveyControlPoint],
) -> RigidTransform:
    """Compute rigid transform from ≥3 survey control point pairs.

    Uses the Kabsch / SVD algorithm — finds the rotation R and translation t
    that minimises the RMSE between the transformed scan points and the
    corresponding world-frame points.

    Raises ValueError for fewer than 3 points or degenerate (collinear) geometry.
    """
    if len(control_points) < 3:
        raise ValueError(f"At least 3 control point pairs required, got {len(control_points)}")

    P = np.array([[cp.scan_x, cp.scan_y, cp.scan_z] for cp in control_points], dtype=float)
    Q = np.array([[cp.world_x, cp.world_y, cp.world_z] for cp in control_points], dtype=float)

    # Centre both point clouds
    p_c = P.mean(axis=0)
    q_c = Q.mean(axis=0)
    Pc = P - p_c
    Qc = Q - q_c

    # Kabsch: covariance → SVD → optimal rotation
    H = Pc.T @ Qc  # (3, 3)
    U, _S, Vt = np.linalg.svd(H)

    # Correct for reflection (ensure det(R) = +1)
    d = np.linalg.det(Vt.T @ U.T)
    D = np.diag([1.0, 1.0, d])
    R = Vt.T @ D @ U.T  # (3, 3) rotation

    # Translation in world frame
    t = q_c - R @ p_c  # (3,)

    # RMSE residual
    transformed = (R @ P.T).T + t
    rmse = float(np.sqrt(((transformed - Q) ** 2).sum(axis=1).mean()))

    if rmse > 50.0:  # warn if >50 mm fit error — likely bad control points
        logger.warning(
            "survey_transform_high_residual",
            rmse_mm=round(rmse, 1),
            note="Check control point coordinates — RMSE > 50 mm is unusually high",
        )

    logger.info(
        "rigid_transform_computed",
        n_control_points=len(control_points),
        rmse_mm=round(rmse, 3),
    )
    return RigidTransform(R=R, t=t, rmse_mm=rmse)


# ── Apply transform to segments ────────────────────────────────────────────────


def _transform_point(pt: Point3D, R: np.ndarray, t: np.ndarray) -> Point3D:
    v = R @ np.array([pt.x, pt.y, pt.z]) + t
    return Point3D(x=float(v[0]), y=float(v[1]), z=float(v[2]))


def _transform_bbox(bb: BoundingBox, R: np.ndarray, t: np.ndarray) -> BoundingBox:
    """Transform all 8 AABB corners and return the new world-frame AABB."""
    corners = np.array(
        [
            [bb.min_x, bb.min_y, bb.min_z],
            [bb.max_x, bb.min_y, bb.min_z],
            [bb.min_x, bb.max_y, bb.min_z],
            [bb.max_x, bb.max_y, bb.min_z],
            [bb.min_x, bb.min_y, bb.max_z],
            [bb.max_x, bb.min_y, bb.max_z],
            [bb.min_x, bb.max_y, bb.max_z],
            [bb.max_x, bb.max_y, bb.max_z],
        ]
    )
    tc = (R @ corners.T).T + t  # (8, 3) transformed corners
    return BoundingBox(
        min_x=float(tc[:, 0].min()),
        max_x=float(tc[:, 0].max()),
        min_y=float(tc[:, 1].min()),
        max_y=float(tc[:, 1].max()),
        min_z=float(tc[:, 2].min()),
        max_z=float(tc[:, 2].max()),
    )


def apply_transform(
    segments: list[GeometrySegment],
    transform: RigidTransform,
) -> list[GeometrySegment]:
    """Apply rigid transform to all segment centroids, bounding boxes and normals.

    Modifies segments in place (GeometrySegment has validate_assignment=True, not frozen).
    Returns the same list for chaining.
    """
    R, t = transform.R, transform.t

    for seg in segments:
        seg.centroid = _transform_point(seg.centroid, R, t)
        seg.bounding_box = _transform_bbox(seg.bounding_box, R, t)
        if seg.normal:
            # Normals transform by rotation only — no translation
            n = R @ np.array([seg.normal.x, seg.normal.y, seg.normal.z])
            seg.normal = Point3D(x=float(n[0]), y=float(n[1]), z=float(n[2]))

    logger.info(
        "transform_applied_to_segments",
        segment_count=len(segments),
        rmse_mm=round(transform.rmse_mm, 3),
    )
    return segments


def validate_transform_quality(
    transform: RigidTransform,
    max_rmse_mm: float = 5.0,
) -> dict:
    """Check whether the transform fit quality meets the project tolerance.

    Returns a result dict suitable for inclusion in the audit event detail.
    """
    passed = transform.rmse_mm <= max_rmse_mm
    return {
        "passed": passed,
        "rmse_mm": round(transform.rmse_mm, 3),
        "max_allowed_mm": max_rmse_mm,
        "message": (
            "Transform quality OK"
            if passed
            else (
                f"RMSE {transform.rmse_mm:.1f} mm exceeds {max_rmse_mm} mm tolerance. "
                "Re-survey control points or increase max_rmse_mm if acceptable."
            )
        ),
    }


def build_transform_hmac(transform: RigidTransform, secret: str = "scantobim") -> str:
    """Produce an HMAC-SHA256 fingerprint for the transform so it can be referenced
    in the audit chain without storing the full matrix every time."""
    payload = f"{transform.rmse_mm}|{transform.R.tobytes().hex()}|{transform.t.tobytes().hex()}"
    return hmac.new(secret.encode(), payload.encode(), hashlib.sha256).hexdigest()


# ── Scan quality gate ──────────────────────────────────────────────────────────


def scan_quality_check(
    point_count: int,
    area_m2: float,
    noise_pts: int = 0,
    min_density_pts_per_m2: float = 100.0,
    max_noise_ratio: float = 0.10,
) -> ScanQualityResult:
    """Pre-segmentation quality gate — call before running any segmentation.

    Rejects scans that are too sparse, too noisy, or too small to produce
    reliable results, saving expensive compute on bad input data.

    Args:
        point_count:             Total raw points after loading.
        area_m2:                 Approximate scan footprint area in m².
        noise_pts:               Points flagged as noise by SOR (optional).
        min_density_pts_per_m2:  Minimum acceptable point density.
        max_noise_ratio:         Maximum ratio noise_pts / point_count.

    Returns:
        ScanQualityResult with passed flag, density stats and error/warning lists.
    """
    errors: list[str] = []
    warnings: list[str] = []

    if point_count < 1_000:
        errors.append(f"Only {point_count} points — minimum 1 000 required for segmentation.")

    density = point_count / max(area_m2, 0.001)
    if density < min_density_pts_per_m2:
        errors.append(
            f"Point density {density:.0f} pts/m² is below the minimum "
            f"{min_density_pts_per_m2:.0f} pts/m². Re-scan at closer range or "
            "reduce voxel downsampling size."
        )

    noise_ratio = noise_pts / max(point_count, 1)
    if noise_ratio > max_noise_ratio:
        warnings.append(
            f"Noise ratio {noise_ratio:.1%} exceeds {max_noise_ratio:.1%}. "
            "Consider scanning in better conditions, using a higher laser power "
            "setting, or applying additional SOR filtering."
        )

    if area_m2 < 1.0:
        warnings.append(
            f"Scan footprint {area_m2:.1f} m² is very small. "
            "Check that the scan covers the intended zone."
        )

    passed = len(errors) == 0

    logger.info(
        "scan_quality_checked",
        passed=passed,
        point_count=point_count,
        density_pts_m2=round(density, 1),
        noise_ratio=round(noise_ratio, 4),
        errors=len(errors),
        warnings=len(warnings),
    )

    return ScanQualityResult(
        passed=passed,
        point_count=point_count,
        area_m2=area_m2,
        density_pts_per_m2=round(density, 1),
        min_required_density=min_density_pts_per_m2,
        noise_ratio=round(noise_ratio, 4),
        warnings=warnings,
        errors=errors,
    )
