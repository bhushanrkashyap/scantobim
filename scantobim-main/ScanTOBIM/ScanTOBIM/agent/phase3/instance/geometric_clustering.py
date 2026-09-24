"""Geometric Instance Segmentation Fallback Engine.

Engineering Rules:
When no compatible neural instance model is available:
Use connected components, Euclidean clustering, normal continuity, plane similarity,
and spatial proximity.
All parameters must be derived or resolution-aware. Zero dataset-specific values.
Status: ALGORITHMIC_ADAPTER.
"""

from __future__ import annotations

import time
from typing import Any

import numpy as np
from scipy.spatial import cKDTree

from agent.phase3.instance.adapter import (
    InstanceMask,
    InstanceSegmentationAdapter,
    InstanceSegmentationResult,
)
from agent.phase3.status import ALGORITHMIC_ADAPTER


class GeometricInstanceAdapter(InstanceSegmentationAdapter):
    """Deterministic geometric instance clustering based on spatial continuity & normal alignment."""

    def __init__(
        self,
        cluster_distance_tol_m: float = 0.25,
        normal_similarity_deg: float = 40.0,
        min_cluster_points: int = 25,
    ) -> None:
        super().__init__(model_name="GeometricInstanceClusterer", checkpoint_path=None)
        self.version = "2.0.0"
        self.cluster_distance_tol_m = cluster_distance_tol_m
        self.normal_similarity_cos = float(np.cos(np.radians(normal_similarity_deg)))
        self.min_cluster_points = min_cluster_points
        self._status = ALGORITHMIC_ADAPTER
        self._loaded = True

    def load(self) -> bool:
        self._status = ALGORITHMIC_ADAPTER
        self._loaded = True
        return True

    def health_check(self) -> bool:
        return True

    def segment(
        self,
        points: np.ndarray,
        semantic_labels: list[str],
        normals: np.ndarray | None = None,
    ) -> InstanceSegmentationResult:
        t0 = time.time()
        N = len(points)
        if N == 0:
            return InstanceSegmentationResult(
                instances=[],
                status=ALGORITHMIC_ADAPTER,
                model_name=self.model_name,
                runtime_s=0.0,
            )

        # Adaptive cluster tolerance based on average nearest neighbor point spacing
        if N > 10:
            full_tree = cKDTree(points)
            sample_sub = points[::max(1, N // 200)]
            dists, _ = full_tree.query(sample_sub, k=min(2, N))
            if dists.ndim > 1 and dists.shape[1] > 1:
                median_spacing = float(np.median(dists[:, 1]))
                adaptive_tol = max(self.cluster_distance_tol_m, median_spacing * 2.2)
            else:
                adaptive_tol = self.cluster_distance_tol_m
        else:
            adaptive_tol = self.cluster_distance_tol_m

        # Group indices by semantic class first
        by_class: dict[str, list[int]] = {}
        for idx, lbl in enumerate(semantic_labels):
            if lbl not in by_class:
                by_class[lbl] = []
            by_class[lbl].append(idx)

        instances: list[InstanceMask] = []
        inst_counter = 1

        for cls_name, cls_indices in by_class.items():
            if cls_name == "UNKNOWN" or len(cls_indices) < self.min_cluster_points:
                continue

            cls_idx_arr = np.array(cls_indices, dtype=int)
            sub_points = points[cls_idx_arr]

            tree = cKDTree(sub_points)
            visited = np.zeros(len(sub_points), dtype=bool)

            for i in range(len(sub_points)):
                if visited[i]:
                    continue

                cluster = []
                queue = [i]
                visited[i] = True

                while queue:
                    curr = queue.pop(0)
                    cluster.append(curr)

                    neighbors = tree.query_ball_point(sub_points[curr], adaptive_tol)
                    for nbr in neighbors:
                        if not visited[nbr]:
                            if normals is not None:
                                n1 = normals[cls_idx_arr[curr]]
                                n2 = normals[cls_idx_arr[nbr]]
                                if abs(np.dot(n1, n2)) < self.normal_similarity_cos:
                                    continue

                            visited[nbr] = True
                            queue.append(nbr)

                if len(cluster) >= self.min_cluster_points:
                    orig_indices = cls_idx_arr[cluster]
                    c_pts = points[orig_indices]
                    centroid = np.mean(c_pts, axis=0)
                    bbox_min = np.min(c_pts, axis=0)
                    bbox_max = np.max(c_pts, axis=0)

                    inst_id = f"{cls_name}_{inst_counter:03d}"
                    inst_counter += 1

                    instances.append(
                        InstanceMask(
                            instance_id=inst_id,
                            semantic_class=cls_name,
                            point_indices=orig_indices,
                            confidence=0.90,
                            centroid_m=centroid,
                            bounding_box_min_m=bbox_min,
                            bounding_box_max_m=bbox_max,
                            point_count=len(orig_indices),
                        )
                    )

        return InstanceSegmentationResult(
            instances=instances,
            status=ALGORITHMIC_ADAPTER,
            model_name=self.model_name,
            runtime_s=time.time() - t0,
            diagnostics={
                "adaptive_tolerance_m": round(adaptive_tol, 4),
                "total_instances": len(instances),
                "by_class": {
                    cls: sum(1 for inst in instances if inst.semantic_class == cls)
                    for cls in {inst.semantic_class for inst in instances}
                },
            },
        )

    def metadata(self) -> dict[str, Any]:
        return {
            "model_name": self.model_name,
            "version": self.version,
            "status": self._status,
            "cluster_distance_tol_m": self.cluster_distance_tol_m,
            "normal_similarity_cos": self.normal_similarity_cos,
            "min_cluster_points": self.min_cluster_points,
            "algorithm": "Density-adaptive KDTree Euclidean region growing with normal continuity",
        }
