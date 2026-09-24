"""A-Scan2BIM Wall Candidate Generation & Geometric Reasoning Adapter.

Adapted from:
- A-Scan2BIM code/learn/backend.py
- A-Scan2BIM code/learn/models/corner_models.py (CornerEnum)
- A-Scan2BIM code/learn/models/edge_full_models.py (EdgeEnum)

Engineering Provenance & Checkpoint Verification:
- HEAT corner detector: NOT EXECUTING (checkpoint.pth unavailable in repo; requires quickstart.zip)
- Edge classifier: NOT EXECUTING (neural weights unavailable)
- Next-wall prediction: NOT EXECUTING (beam search weights unavailable)
- Metric-learning model: NOT EXECUTING (weights unavailable)
- Candidate wall enumeration: ACTIVELY EXECUTING (algorithmic corner-edge reasoning using
  Shi-Tomasi eigenvalue corner responses and point-cloud line corridor support counting).
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import cv2
import numpy as np
import structlog

logger = structlog.get_logger()


@dataclass
class AScan2BimStatus:
    """Explicit audit record of which A-Scan2BIM components are executing."""

    heat_corner_detector: bool = False
    edge_classifier: bool = False
    next_wall_prediction: bool = False
    metric_learning_model: bool = False
    candidate_wall_enumeration: bool = True
    reasoning_mode: str = "ALGORITHMIC_CORNER_EDGE_REASONING"

    def to_dict(self) -> dict[str, Any]:
        return {
            "heat_corner_detector_executed": self.heat_corner_detector,
            "edge_classifier_executed": self.edge_classifier,
            "next_wall_prediction_executed": self.next_wall_prediction,
            "metric_learning_model_executed": self.metric_learning_model,
            "candidate_wall_enumeration_executed": self.candidate_wall_enumeration,
            "reasoning_mode": self.reasoning_mode,
            "checkpoint_required": "quickstart.zip: ckpts/corner/checkpoint.pth",
        }


@dataclass
class WallCandidateProposal:
    """Represents an A-Scan2BIM candidate wall edge proposal."""

    candidate_id: str
    storey_id: str
    start_pt_m: tuple[float, float]
    end_pt_m: tuple[float, float]
    length_m: float
    angle_deg: float
    confidence: float
    inlier_support: int
    source: str = "ascan2bim_corner_edge"

    def to_dict(self) -> dict[str, Any]:
        return {
            "candidate_id": self.candidate_id,
            "storey_id": self.storey_id,
            "start_pt_m": [round(c, 4) for c in self.start_pt_m],
            "end_pt_m": [round(c, 4) for c in self.end_pt_m],
            "length_m": round(self.length_m, 3),
            "angle_deg": round(self.angle_deg, 2),
            "confidence": round(self.confidence, 3),
            "inlier_support": self.inlier_support,
            "source": self.source,
        }


class AScan2BimWallAdapter:
    """Adapter executing A-Scan2BIM wall candidate reasoning."""

    def __init__(self, min_wall_length_m: float = 1.0, raster_res_m: float = 0.10):
        self.min_wall_length_m = min_wall_length_m
        self.raster_res_m = raster_res_m
        self.status = AScan2BimStatus()
        logger.info(
            "ascan2bim_adapter_initialized",
            min_length=min_wall_length_m,
            mode=self.status.reasoning_mode,
        )

    def generate_wall_candidates(
        self,
        points_xyz: np.ndarray,
        storey_id: str,
        base_z_m: float,
        top_z_m: float,
    ) -> list[WallCandidateProposal]:
        """Generate ranked candidate wall edges for a storey using corner-edge reasoning."""
        mask_z = (points_xyz[:, 2] >= base_z_m) & (points_xyz[:, 2] <= top_z_m)
        pts = points_xyz[mask_z]
        if len(pts) < 100:
            return []

        pts_xy = pts[:, :2]
        min_xy = pts_xy.min(axis=0)
        max_xy = pts_xy.max(axis=0)
        span = max_xy - min_xy
        if np.any(span < self.min_wall_length_m):
            return []

        # 2D Density rasterization
        nx = max(10, int(np.ceil(span[0] / self.raster_res_m)))
        ny = max(10, int(np.ceil(span[1] / self.raster_res_m)))
        raster = np.zeros((ny, nx), dtype=np.uint8)

        xi = np.clip(((pts_xy[:, 0] - min_xy[0]) / self.raster_res_m).astype(int), 0, nx - 1)
        yi = np.clip(((pts_xy[:, 1] - min_xy[1]) / self.raster_res_m).astype(int), 0, ny - 1)
        coords, counts = np.unique(np.column_stack([yi, xi]), axis=0, return_counts=True)
        for (y, x), c in zip(coords, counts):
            raster[y, x] = min(255, int(c * 15))

        # Morphological closing along wall corridors
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
        dilated = cv2.dilate(raster, kernel, iterations=1)

        # Corner and linear feature candidate detection (Shi-Tomasi + Hough transform)
        corners = cv2.goodFeaturesToTrack(
            dilated,
            maxCorners=60,
            qualityLevel=0.01,
            minDistance=int(self.min_wall_length_m / self.raster_res_m * 0.5),
            blockSize=3,
        )

        detected_corners: list[tuple[float, float]] = []
        if corners is not None:
            for pt in corners:
                cx_px, cy_px = pt.ravel()
                wx = float(min_xy[0] + cx_px * self.raster_res_m)
                wy = float(min_xy[1] + cy_px * self.raster_res_m)
                detected_corners.append((wx, wy))

        # Linear wall segment endpoints via probabilistic Hough transform
        h_lines = cv2.HoughLinesP(
            dilated,
            1,
            np.pi / 180,
            threshold=10,
            minLineLength=int(self.min_wall_length_m / self.raster_res_m),
            maxLineGap=4,
        )
        if h_lines is not None:
            for hl in h_lines:
                x1, y1, x2, y2 = hl[0]
                detected_corners.append((float(min_xy[0] + x1 * self.raster_res_m), float(min_xy[1] + y1 * self.raster_res_m)))
                detected_corners.append((float(min_xy[0] + x2 * self.raster_res_m), float(min_xy[1] + y2 * self.raster_res_m)))

        # Candidate edge enumeration connecting plausible corner pairs
        proposals: list[WallCandidateProposal] = []
        n_c = len(detected_corners)
        prop_idx = 0

        for i in range(n_c):
            p1 = np.array(detected_corners[i])
            for j in range(i + 1, n_c):
                p2 = np.array(detected_corners[j])
                dist = float(np.linalg.norm(p2 - p1))

                if dist < self.min_wall_length_m or dist > 25.0:
                    continue

                angle_rad = math.atan2(p2[1] - p1[1], p2[0] - p1[0])
                angle_deg = (math.degrees(angle_rad)) % 180.0

                # Manhattan alignment preference (near 0°, 90°, 180°)
                is_axis_aligned = (
                    abs(angle_deg - 0.0) < 15.0
                    or abs(angle_deg - 90.0) < 15.0
                    or abs(angle_deg - 180.0) < 15.0
                )

                # Measure point support in the line corridor (+- 0.25 m)
                v = (p2 - p1) / dist
                n_vec = np.array([-v[1], v[0]])

                diff = pts_xy - p1
                proj = diff[:, 0] * v[0] + diff[:, 1] * v[1]
                perp = np.abs(diff[:, 0] * n_vec[0] + diff[:, 1] * n_vec[1])

                corridor_inliers = int(np.sum((proj >= 0.0) & (proj <= dist) & (perp <= 0.25)))
                coverage_ratio = corridor_inliers / max(dist * 20.0, 1.0)

                if corridor_inliers >= 25 and coverage_ratio >= 0.35:
                    prop_idx += 1
                    conf = min(0.65 + 0.20 * min(coverage_ratio, 1.0) + (0.10 if is_axis_aligned else 0.0), 0.95)
                    proposals.append(
                        WallCandidateProposal(
                            candidate_id=f"ascan_{storey_id}_{prop_idx}",
                            storey_id=storey_id,
                            start_pt_m=(float(p1[0]), float(p1[1])),
                            end_pt_m=(float(p2[0]), float(p2[1])),
                            length_m=dist,
                            angle_deg=angle_deg,
                            confidence=conf,
                            inlier_support=corridor_inliers,
                        )
                    )

        proposals.sort(key=lambda p: p.confidence, reverse=True)
        return proposals
