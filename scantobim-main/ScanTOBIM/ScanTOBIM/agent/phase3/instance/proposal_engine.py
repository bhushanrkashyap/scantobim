"""Class-Agnostic 3D Object Proposal Engine for Scan-to-BIM.

Phase 3B Mandatory Requirement (Section 13 & 14):
- Semantic segmentation alone is NOT object detection.
- Answers: "Where are the distinct physical objects?" BEFORE "What is each object?"
- Discovers object instances even when semantic class is unknown or outside training vocabulary.
- Uses:
    * adaptive spatial clustering (data-derived tolerance, NOT a single fixed DBSCAN epsilon)
    * local normal continuity and surface smoothness
    * curvature transitions and boundary detection
    * Euclidean clustering & connected components
    * multi-scale merging for coplanar / co-axial fragments
    * PCA-based Oriented Bounding Boxes (OBB) & AABB
    * strict source provenance tracking back to input points

CPU ONLY — NO CUDA OPERATORS.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any
import numpy as np
from scipy.spatial import cKDTree
import structlog

from agent.phase3.geometry.features import (
    compute_geometric_features,
    estimate_multi_scale_radii,
)
from agent.phase3.instance.adapter import InstanceMask, InstanceSegmentationResult
from agent.phase3.status import ALGORITHMIC_ADAPTER

logger = structlog.get_logger(__name__)


@dataclass
class ObjectProposal:
    """Class-agnostic physical 3D object candidate."""
    proposal_id: str
    point_indices: np.ndarray
    point_count: int
    centroid_m: np.ndarray
    bbox_min_m: np.ndarray
    bbox_max_m: np.ndarray
    obb_center_m: np.ndarray
    obb_extents_m: np.ndarray
    obb_rotation_matrix: np.ndarray
    principal_axes: np.ndarray
    surface_area_m2: float
    volume_m3: float
    geometric_signature: dict[str, float]
    initial_shape_type: str  # "PLANAR_VERTICAL", "PLANAR_HORIZONTAL", "CYLINDRICAL", "ELONGATED", "COMPACT", "AMBIGUOUS"
    confidence: float
    source_point_ids: list[int] = field(default_factory=list)

    def to_instance_mask(self, semantic_label: str = "UNKNOWN_OBJECT") -> InstanceMask:
        """Convert class-agnostic proposal into full InstanceMask."""
        return InstanceMask(
            instance_id=self.proposal_id,
            semantic_class=semantic_label,
            point_indices=self.point_indices,
            confidence=self.confidence,
            centroid_m=self.centroid_m,
            bounding_box_min_m=self.bbox_min_m,
            bounding_box_max_m=self.bbox_max_m,
            point_count=self.point_count,
            oriented_bounding_box={
                "center_m": [round(float(c), 4) for c in self.obb_center_m],
                "extents_m": [round(float(e), 4) for e in self.obb_extents_m],
                "rotation_matrix": [[round(float(val), 4) for val in row] for row in self.obb_rotation_matrix],
            },
            principal_axes=[[round(float(val), 4) for val in row] for row in self.principal_axes],
            dimensions={
                "length_m": round(float(self.obb_extents_m[2]), 4),
                "width_m": round(float(self.obb_extents_m[1]), 4),
                "height_or_thickness_m": round(float(self.obb_extents_m[0]), 4),
            },
            surface_area_estimate=self.surface_area_m2,
            volume_estimate_if_valid=self.volume_m3,
            semantic_candidates={semantic_label: self.confidence},
            semantic_probability_distribution={semantic_label: self.confidence},
            evidence={
                "shape_type": self.initial_shape_type,
                "geometric_signature": self.geometric_signature,
            },
            source_point_ids=self.source_point_ids,
        )


class ClassAgnosticProposalEngine:
    """Discovers physical 3D object proposals from geometry and point topology."""

    def __init__(
        self,
        min_proposal_points: int = 15,
        normal_smoothness_deg: float = 35.0,
        coplanar_merge_dist_m: float = 0.15,
    ) -> None:
        self.min_proposal_points = min_proposal_points
        self.normal_smoothness_cos = float(np.cos(np.radians(normal_smoothness_deg)))
        self.coplanar_merge_dist_m = coplanar_merge_dist_m

    def generate_proposals(
        self,
        points: np.ndarray,
        normals: np.ndarray | None = None,
        source_indices: np.ndarray | None = None,
    ) -> list[ObjectProposal]:
        """Generate class-agnostic physical object proposals.

        Args:
            points: (N, 3) float32 coordinates.
            normals: (N, 3) optional precomputed normals.
            source_indices: Optional original point index mapping (preserves E57 provenance).
        """
        N = len(points)
        if N < self.min_proposal_points:
            return []

        t0 = time.time()
        radii = estimate_multi_scale_radii(points)
        adaptive_tol = radii.local_radius_m

        # Compute or use features
        features = compute_geometric_features(
            points,
            k_neighbors=min(24, N),
            precomputed_radii=radii,
        )
        use_normals = normals if normals is not None else features.normals

        tree = cKDTree(points)
        visited = np.zeros(N, dtype=bool)
        raw_clusters: list[list[int]] = []

        # 1. Surface-continuity Region Growing (Superpoint / Surface clustering)
        # Sort points by planarity descending: start growing from cleanest planar seeds
        seed_order = np.argsort(-features.planarity)

        for seed_idx in seed_order:
            if visited[seed_idx]:
                continue

            # Only initiate smooth growing if point has reasonable surface definition
            is_seed_planar = bool(features.planarity[seed_idx] > 0.35)
            is_seed_tubular = bool(features.linearity[seed_idx] > 0.45)
            if not is_seed_planar and not is_seed_tubular:
                continue

            current_cluster: list[int] = [seed_idx]
            visited[seed_idx] = True
            queue: list[int] = [seed_idx]

            while queue:
                curr = queue.pop(0)
                curr_normal = use_normals[curr]

                # Adaptive neighbor query
                nbr_indices = tree.query_ball_point(points[curr], r=adaptive_tol * 1.5)
                for nbr in nbr_indices:
                    if visited[nbr]:
                        continue

                    # Boundary condition: prevent planar wall surface from absorbing tubular/high-curvature points
                    if is_seed_planar and features.planarity[nbr] < 0.25 and features.linearity[nbr] > 0.40:
                        continue
                    # Boundary condition: prevent tubular pipe surface from absorbing flat wall points
                    if is_seed_tubular and features.planarity[nbr] > 0.40 and features.linearity[nbr] < 0.35:
                        continue

                    # Check normal alignment (smooth surface continuity)
                    dot_norm = abs(float(np.dot(curr_normal, use_normals[nbr])))
                    if dot_norm >= self.normal_smoothness_cos:
                        visited[nbr] = True
                        current_cluster.append(nbr)
                        queue.append(nbr)

            if len(current_cluster) >= 3:
                raw_clusters.append(current_cluster)

        # 2. Residual Euclidean Clustering for Non-Planar Objects (pipes, equipment, valves, clutter)
        unvisited_indices = np.where(~visited)[0]
        if len(unvisited_indices) >= 5:
            unvisited_pts = points[unvisited_indices]
            unvisited_tree = cKDTree(unvisited_pts)
            unv_visited = np.zeros(len(unvisited_indices), dtype=bool)

            for u_idx in range(len(unvisited_indices)):
                if unv_visited[u_idx]:
                    continue

                cluster_u: list[int] = [u_idx]
                unv_visited[u_idx] = True
                u_queue = [u_idx]

                while u_queue:
                    curr_u = u_queue.pop(0)
                    nbrs = unvisited_tree.query_ball_point(unvisited_pts[curr_u], r=adaptive_tol * 2.0)
                    for nbr_u in nbrs:
                        if not unv_visited[nbr_u]:
                            unv_visited[nbr_u] = True
                            cluster_u.append(nbr_u)
                            u_queue.append(nbr_u)

                if len(cluster_u) >= 5:
                    mapped_cluster = [int(unvisited_indices[i]) for i in cluster_u]
                    raw_clusters.append(mapped_cluster)

        # 3. Multi-scale Coplanar Fragment Merging
        merged_clusters = self._merge_coplanar_proposals(points, use_normals, raw_clusters)

        # 4. Construct Object Proposals with OBB and Geometric Signatures
        proposals: list[ObjectProposal] = []
        for prop_idx, cluster in enumerate(merged_clusters):
            c_indices = np.array(cluster, dtype=int)
            c_points = points[c_indices]
            if len(c_points) < self.min_proposal_points:
                continue

            # Centroid
            centroid = np.mean(c_points, axis=0)

            # AABB
            bbox_min = np.min(c_points, axis=0)
            bbox_max = np.max(c_points, axis=0)

            # PCA & Oriented Bounding Box (OBB)
            centered = c_points - centroid
            cov = np.dot(centered.T, centered) / float(len(c_points))
            eigvals, eigvecs = np.linalg.eigh(cov)
            sort_idx = np.argsort(eigvals)
            sorted_eigs = eigvals[sort_idx]
            rot_matrix = eigvecs[:, sort_idx]  # Columns are v0, v1, v2

            # Project points onto principal axes
            projs = np.dot(centered, rot_matrix)
            proj_min = np.min(projs, axis=0)
            proj_max = np.max(projs, axis=0)
            obb_extents = np.maximum(proj_max - proj_min, 1e-4)
            obb_center = centroid + np.dot(rot_matrix, (proj_min + proj_max) / 2.0)

            # Surface area & volume
            dx, dy, dz = obb_extents[0], obb_extents[1], obb_extents[2]
            surface_area = 2.0 * (dx * dy + dy * dz + dx * dz)
            volume = float(dx * dy * dz)

            # Geometric signatures from cluster points
            c_planarity = float(np.mean(features.planarity[c_indices]))
            c_linearity = float(np.mean(features.linearity[c_indices]))
            c_scattering = float(np.mean(features.scattering[c_indices]))
            c_verticality = float(np.mean(features.verticality[c_indices]))
            c_horizontality = float(np.mean(features.horizontality[c_indices]))
            c_cylindricality = float(np.mean(features.cylindricality[c_indices]))

            # Classify initial geometric shape
            if c_planarity > 0.45 and c_verticality > 0.60:
                shape_type = "PLANAR_VERTICAL"  # Likely Wall or vertical panel
            elif c_planarity > 0.45 and c_horizontality > 0.60:
                shape_type = "PLANAR_HORIZONTAL"  # Likely Floor, Ceiling, Slab
            elif c_cylindricality > 0.40 or (c_linearity > 0.50 and min(dx, dy) < 0.35):
                shape_type = "CYLINDRICAL"  # Likely Pipe, Column, Conduit, or Duct
            elif c_linearity > 0.50:
                shape_type = "ELONGATED"  # Likely Beam or Cable Tray
            elif c_scattering > 0.25:
                shape_type = "COMPACT"  # Likely Equipment, Valve, Furniture, Fitting
            else:
                shape_type = "AMBIGUOUS"

            # Compute confidence from geometric coherence
            shape_coherence = max(c_planarity, c_linearity, c_cylindricality, c_scattering)
            density_support = min(1.0, len(c_points) / 50.0)
            prop_confidence = float(np.clip(0.5 * shape_coherence + 0.5 * density_support, 0.2, 0.99))

            # Provenance
            if source_indices is not None:
                src_ids = [int(source_indices[i]) for i in c_indices]
            else:
                src_ids = [int(i) for i in c_indices]

            proposal = ObjectProposal(
                proposal_id=f"prop_{prop_idx + 1:04d}",
                point_indices=c_indices,
                point_count=len(c_indices),
                centroid_m=centroid,
                bbox_min_m=bbox_min,
                bbox_max_m=bbox_max,
                obb_center_m=obb_center,
                obb_extents_m=obb_extents,
                obb_rotation_matrix=rot_matrix,
                principal_axes=rot_matrix.T,
                surface_area_m2=float(surface_area),
                volume_m3=float(volume),
                geometric_signature={
                    "planarity": round(c_planarity, 4),
                    "linearity": round(c_linearity, 4),
                    "scattering": round(c_scattering, 4),
                    "verticality": round(c_verticality, 4),
                    "horizontality": round(c_horizontality, 4),
                    "cylindricality": round(c_cylindricality, 4),
                },
                initial_shape_type=shape_type,
                confidence=round(prop_confidence, 4),
                source_point_ids=src_ids,
            )
            proposals.append(proposal)

        logger.info(
            "class_agnostic_proposals_generated",
            input_points=N,
            proposals_count=len(proposals),
            duration_s=round(time.time() - t0, 4),
        )
        return proposals

    def _merge_coplanar_proposals(
        self,
        points: np.ndarray,
        normals: np.ndarray,
        clusters: list[list[int]],
    ) -> list[list[int]]:
        """Merge adjacent clusters that share coplanar geometry and proximity."""
        if len(clusters) <= 1:
            return clusters

        # Compute plane normal and offset for each cluster
        cluster_planes: list[tuple[np.ndarray, float, np.ndarray]] = []
        valid_clusters: list[list[int]] = []

        for c in clusters:
            if len(c) < 3:
                continue
            c_pts = points[c]
            c_ctr = np.mean(c_pts, axis=0)
            c_norm = np.mean(normals[c], axis=0)
            norm_len = np.linalg.norm(c_norm)
            if norm_len > 1e-6:
                c_norm /= norm_len
            else:
                c_norm = np.array([0.0, 0.0, 1.0])
            d = float(np.dot(c_norm, c_ctr))
            cluster_planes.append((c_norm, d, c_ctr))
            valid_clusters.append(c)

        merged: list[list[int]] = []
        used = np.zeros(len(valid_clusters), dtype=bool)

        for i in range(len(valid_clusters)):
            if used[i]:
                continue
            cur_merged = list(valid_clusters[i])
            used[i] = True
            n_i, d_i, ctr_i = cluster_planes[i]

            for j in range(i + 1, len(valid_clusters)):
                if used[j]:
                    continue
                n_j, d_j, ctr_j = cluster_planes[j]

                # Check normal parallelism
                norm_dot = abs(float(np.dot(n_i, n_j)))
                if norm_dot > 0.95:  # within ~18 degrees
                    # Check plane distance offset
                    plane_offset = abs(d_i - d_j)
                    if plane_offset < self.coplanar_merge_dist_m:
                        # Check spatial proximity between centroids
                        centroid_dist = np.linalg.norm(ctr_i - ctr_j)
                        if centroid_dist < 3.0:  # reasonable building fragment span
                            cur_merged.extend(valid_clusters[j])
                            used[j] = True

            merged.append(cur_merged)

        return [c for c in merged if len(c) >= self.min_proposal_points]
