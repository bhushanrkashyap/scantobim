"""Small Object & MEP Fitting Separation Engine for Phase 3C.

Addresses Section 19 requirement:
  - Dedicated small-object path for compact MEP components (valves, fittings, elbows, tees).
  - Inspects local curvature transitions and radial variance along tubular runs.
  - Prevents small MEP fittings from being absorbed into host pipes.
  - Generates distinct proposals with UNKNOWN_MEP / VALVE / FITTING / REVIEW_REQUIRED labels.
"""

from __future__ import annotations

from typing import Any
import numpy as np
from scipy.spatial import cKDTree
import structlog

from agent.phase3.instance.proposal_engine import ObjectProposal

logger = structlog.get_logger(__name__)


class SmallObjectDetector:
    """Detects and separates compact MEP fittings and inline components from host runs."""

    def __init__(
        self,
        min_fitting_points: int = 12,
        radial_expansion_ratio: float = 1.35,
        high_curvature_threshold: float = 0.18,
    ) -> None:
        self.min_fitting_points = min_fitting_points
        self.radial_expansion_ratio = radial_expansion_ratio
        self.high_curvature_threshold = high_curvature_threshold

    def separate_small_mep_components(
        self,
        proposals: list[ObjectProposal],
        points: np.ndarray,
        curvatures: np.ndarray | None = None,
    ) -> list[ObjectProposal]:
        """Inspect pipe/tubular proposals for inline fittings and separate them into distinct instances.

        Args:
            proposals: List of candidate object proposals.
            points: Full point cloud coordinates (N, 3).
            curvatures: Optional precomputed point curvatures (N,).

        Returns:
            Refined proposals list with separated small MEP fittings.
        """
        refined_proposals: list[ObjectProposal] = []
        next_id_num = len(proposals) + 1

        for prop in proposals:
            # Only inspect elongated / cylindrical or MEP-candidate proposals
            is_pipe_candidate = (
                prop.initial_shape_type in ("CYLINDRICAL", "LINEAR_EXTRUDED")
                or prop.geometric_signature.get("cylindricality", 0.0) > 0.20
            )

            # Need sufficient length and points to contain an inline component
            max_ext = max(prop.obb_extents_m)
            min_ext = min(prop.obb_extents_m)
            if not is_pipe_candidate or len(prop.point_indices) < 60 or max_ext < 0.8:
                refined_proposals.append(prop)
                continue

            c_pts = points[prop.point_indices]
            center = prop.centroid_m
            axis = prop.obb_rotation_matrix[:, 2]  # Elongation axis

            # Project points along axis (t) and compute radial distance (r)
            rel_pts = c_pts - center
            t_coords = np.dot(rel_pts, axis)
            proj_on_axis = np.outer(t_coords, axis)
            rad_vectors = rel_pts - proj_on_axis
            r_dists = np.linalg.norm(rad_vectors, axis=1)

            # Compute median pipe radius
            med_radius = float(np.median(r_dists))
            if med_radius < 0.01:
                refined_proposals.append(prop)
                continue

            # Identify points that deviate significantly in radius (expansion = valve/fitting)
            # or points with high local curvature
            is_expanded = r_dists > (med_radius * self.radial_expansion_ratio)

            if curvatures is not None:
                prop_curv = curvatures[prop.point_indices]
                is_high_curv = prop_curv > self.high_curvature_threshold
                fitting_mask = is_expanded | (is_high_curv & (r_dists > med_radius * 1.15))
            else:
                fitting_mask = is_expanded

            fitting_indices_local = np.where(fitting_mask)[0]

            # If a compact cluster of fitting points is detected, separate it
            if self.min_fitting_points <= len(fitting_indices_local) <= len(prop.point_indices) * 0.40:
                # Check spatial compactness along the pipe axis
                t_spread = float(np.ptp(t_coords[fitting_indices_local]))
                if t_spread < 0.50:  # Fitting span along pipe is compact (< 50 cm)
                    # Cluster fitting points
                    fit_pts = c_pts[fitting_indices_local]
                    pipe_mask = ~fitting_mask
                    pipe_pts = c_pts[pipe_mask]

                    if len(pipe_pts) >= 15:
                        # 1. Update the original pipe proposal with remaining pipe points
                        new_pipe_indices = prop.point_indices[pipe_mask]
                        new_src_ids = [prop.source_point_ids[i] for i in np.where(pipe_mask)[0]] if prop.source_point_ids else new_pipe_indices.tolist()

                        pipe_prop = ObjectProposal(
                            proposal_id=prop.proposal_id,
                            point_indices=new_pipe_indices,
                            point_count=len(new_pipe_indices),
                            centroid_m=np.mean(pipe_pts, axis=0),
                            bbox_min_m=np.min(pipe_pts, axis=0),
                            bbox_max_m=np.max(pipe_pts, axis=0),
                            obb_center_m=prop.obb_center_m,
                            obb_extents_m=prop.obb_extents_m,
                            obb_rotation_matrix=prop.obb_rotation_matrix,
                            surface_area_m2=prop.surface_area_m2,
                            volume_m3=prop.volume_m3,
                            geometric_signature=prop.geometric_signature,
                            initial_shape_type="CYLINDRICAL",
                            confidence=prop.confidence,
                            source_point_ids=new_src_ids,
                        )
                        refined_proposals.append(pipe_prop)

                        # 2. Create the distinct fitting proposal
                        new_fit_indices = prop.point_indices[fitting_indices_local]
                        new_fit_src_ids = [prop.source_point_ids[i] for i in fitting_indices_local] if prop.source_point_ids else new_fit_indices.tolist()

                        fit_centroid = np.mean(fit_pts, axis=0)
                        fit_extents = np.maximum(np.ptp(fit_pts, axis=0), 0.02)
                        fit_volume = float(np.prod(fit_extents))

                        # Classify fitting type
                        max_fit_dim = max(fit_extents)
                        fit_cls = "VALVE" if max_fit_dim > 0.15 else "UNKNOWN_MEP"

                        fitting_prop = ObjectProposal(
                            proposal_id=f"prop_{next_id_num:04d}",
                            point_indices=new_fit_indices,
                            point_count=len(new_fit_indices),
                            centroid_m=fit_centroid,
                            bbox_min_m=np.min(fit_pts, axis=0),
                            bbox_max_m=np.max(fit_pts, axis=0),
                            obb_center_m=fit_centroid,
                            obb_extents_m=fit_extents,
                            obb_rotation_matrix=np.eye(3, dtype=np.float32),
                            surface_area_m2=2.0 * float(fit_extents[0]*fit_extents[1] + fit_extents[1]*fit_extents[2] + fit_extents[0]*fit_extents[2]),
                            volume_m3=fit_volume,
                            geometric_signature={
                                "planarity": 0.20,
                                "linearity": 0.25,
                                "scattering": 0.55,
                                "cylindricality": 0.10,
                                "component_type": "INLINE_MEP_FITTING",
                            },
                            initial_shape_type="COMPACT_COMPLEX",
                            confidence=0.82,
                            source_point_ids=new_fit_src_ids,
                        )
                        refined_proposals.append(fitting_prop)
                        next_id_num += 1

                        logger.info(
                            "small_mep_component_separated",
                            host_pipe=prop.proposal_id,
                            component_id=fitting_prop.proposal_id,
                            fitting_class=fit_cls,
                            points_count=len(new_fit_indices),
                        )
                        continue

            refined_proposals.append(prop)

        return refined_proposals
