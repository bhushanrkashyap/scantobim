"""Topological Reasoning Graph Engine for Scan-to-BIM.

Phase 3B Core Requirement (Section 19):
- Builds spatial and physical relationship graph:
    * NODE: Discovered object instance
    * EDGE: Physical / spatial relationship
        - DOOR / WINDOW --HOSTED_BY--> WALL
        - PIPE --CONNECTED_TO--> VALVE / FITTING / EQUIPMENT
        - COLUMN --SUPPORTED_BY--> FLOOR / SLAB
        - WALL --INTERSECTS--> FLOOR / SLAB
        - DUCT --CONNECTED_TO--> HVAC_EQUIPMENT
        - CABLE_TRAY --SUPPORTED_BY--> STRUCTURE
- Uses topology feedback loop to reinforce or disambiguate classifications:
    * E.g. inline cylindrical component between two pipes -> strong evidence for VALVE / FITTING.
    * E.g. vertical element supported by floor and reaching ceiling -> strong evidence for COLUMN.

CPU ONLY — DETERMINISTIC — NO HARDCODED DATASET ASSUMPTIONS.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any
import numpy as np
import structlog

logger = structlog.get_logger(__name__)


@dataclass
class TopologyRelation:
    """Directed edge in the physical relationship graph."""
    source_id: str
    target_id: str
    relation_type: str  # HOSTED_BY, CONNECTED_TO, SUPPORTED_BY, INTERSECTS, ADJACENT_TO
    confidence: float
    distance_m: float
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "source_id": self.source_id,
            "target_id": self.target_id,
            "relation_type": self.relation_type,
            "confidence": round(float(self.confidence), 4),
            "distance_m": round(float(self.distance_m), 4),
            "metadata": self.metadata,
        }


@dataclass
class TopologyNode:
    """Node in the scene topology graph representing an object instance."""
    object_id: str
    semantic_label: str
    centroid_m: np.ndarray
    bbox_min_m: np.ndarray
    bbox_max_m: np.ndarray
    obb_extents_m: np.ndarray
    point_count: int
    elevation_bottom_m: float
    elevation_top_m: float
    in_edges: list[TopologyRelation] = field(default_factory=list)
    out_edges: list[TopologyRelation] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "object_id": self.object_id,
            "semantic_label": self.semantic_label,
            "centroid_m": [round(float(c), 4) for c in self.centroid_m],
            "bbox_min_m": [round(float(c), 4) for c in self.bbox_min_m],
            "bbox_max_m": [round(float(c), 4) for c in self.bbox_max_m],
            "extents_m": [round(float(c), 4) for c in self.obb_extents_m],
            "point_count": self.point_count,
            "relationships_out": [e.to_dict() for e in self.out_edges],
            "relationships_in": [e.to_dict() for e in self.in_edges],
        }


class TopologyGraph:
    """Manages scene objects, spatial relations, and topological classification reasoning."""

    def __init__(self, connection_tolerance_m: float = 0.25) -> None:
        self.nodes: dict[str, TopologyNode] = {}
        self.edges: list[TopologyRelation] = []
        self.connection_tol_m = connection_tolerance_m

    def add_object(
        self,
        object_id: str,
        semantic_label: str,
        centroid_m: np.ndarray,
        bbox_min_m: np.ndarray,
        bbox_max_m: np.ndarray,
        obb_extents_m: np.ndarray | None = None,
        point_count: int = 0,
    ) -> TopologyNode:
        """Add an object instance as a node in the graph."""
        extents = obb_extents_m if obb_extents_m is not None else (bbox_max_m - bbox_min_m)
        node = TopologyNode(
            object_id=object_id,
            semantic_label=semantic_label,
            centroid_m=centroid_m,
            bbox_min_m=bbox_min_m,
            bbox_max_m=bbox_max_m,
            obb_extents_m=extents,
            point_count=point_count,
            elevation_bottom_m=float(bbox_min_m[2]),
            elevation_top_m=float(bbox_max_m[2]),
        )
        self.nodes[object_id] = node
        return node

    def add_relation(
        self,
        source_id: str,
        target_id: str,
        relation_type: str,
        confidence: float,
        distance_m: float,
        metadata: dict[str, Any] | None = None,
    ) -> TopologyRelation | None:
        """Add a directed physical relationship edge between two objects."""
        if source_id not in self.nodes or target_id not in self.nodes:
            return None

        rel = TopologyRelation(
            source_id=source_id,
            target_id=target_id,
            relation_type=relation_type,
            confidence=confidence,
            distance_m=distance_m,
            metadata=metadata or {},
        )
        self.edges.append(rel)
        self.nodes[source_id].out_edges.append(rel)
        self.nodes[target_id].in_edges.append(rel)
        return rel

    def build_spatial_topology(self) -> None:
        """Analyze all pairs of objects and discover physical relationships."""
        obj_ids = list(self.nodes.keys())
        N = len(obj_ids)

        for i in range(N):
            id_a = obj_ids[i]
            node_a = self.nodes[id_a]

            for j in range(i + 1, N):
                id_b = obj_ids[j]
                node_b = self.nodes[id_b]

                # Center distance
                dist_centers = float(np.linalg.norm(node_a.centroid_m - node_b.centroid_m))

                # 1. Structural Support: Column/Wall Supported by Floor/Slab
                if node_b.semantic_label in ("FLOOR", "SLAB"):
                    floor_top = node_b.elevation_top_m
                    if abs(node_a.elevation_bottom_m - floor_top) < self.connection_tol_m:
                        if node_a.semantic_label in ("COLUMN", "WALL"):
                            self.add_relation(
                                source_id=id_a,
                                target_id=id_b,
                                relation_type="SUPPORTED_BY",
                                confidence=0.92,
                                distance_m=abs(node_a.elevation_bottom_m - floor_top),
                            )
                elif node_a.semantic_label in ("FLOOR", "SLAB"):
                    floor_top = node_a.elevation_top_m
                    if abs(node_b.elevation_bottom_m - floor_top) < self.connection_tol_m:
                        if node_b.semantic_label in ("COLUMN", "WALL"):
                            self.add_relation(
                                source_id=id_b,
                                target_id=id_a,
                                relation_type="SUPPORTED_BY",
                                confidence=0.92,
                                distance_m=abs(node_b.elevation_bottom_m - floor_top),
                            )

                # 2. Host Wall Relationship: Door / Window Hosted by Wall
                if node_b.semantic_label == "WALL" and node_a.semantic_label in ("DOOR", "WINDOW", "OPENING"):
                    if self._is_inside_bbox(node_a, node_b, margin_m=0.20):
                        self.add_relation(
                            source_id=id_a,
                            target_id=id_b,
                            relation_type="HOSTED_BY",
                            confidence=0.95,
                            distance_m=dist_centers,
                        )
                elif node_a.semantic_label == "WALL" and node_b.semantic_label in ("DOOR", "WINDOW", "OPENING"):
                    if self._is_inside_bbox(node_b, node_a, margin_m=0.20):
                        self.add_relation(
                            source_id=id_b,
                            target_id=id_a,
                            relation_type="HOSTED_BY",
                            confidence=0.95,
                            distance_m=dist_centers,
                        )

                # 3. MEP Connectivity: Pipe <-> Pipe, Pipe <-> Valve, Pipe <-> Equipment, Duct <-> Equipment
                mep_types = ("PIPE", "VALVE", "FITTING", "EQUIPMENT", "DUCT", "PUMP", "HVAC_COMPONENT")
                if node_a.semantic_label in mep_types and node_b.semantic_label in mep_types:
                    # Bounding box clearance
                    clearance = self._bbox_clearance(node_a, node_b)
                    if clearance <= self.connection_tol_m:
                        self.add_relation(
                            source_id=id_a,
                            target_id=id_b,
                            relation_type="CONNECTED_TO",
                            confidence=0.88,
                            distance_m=clearance,
                        )
                        self.add_relation(
                            source_id=id_b,
                            target_id=id_a,
                            relation_type="CONNECTED_TO",
                            confidence=0.88,
                            distance_m=clearance,
                        )

                # 4. Wall <-> Floor Intersections
                if (node_a.semantic_label == "WALL" and node_b.semantic_label in ("FLOOR", "SLAB")) or \
                   (node_b.semantic_label == "WALL" and node_a.semantic_label in ("FLOOR", "SLAB")):
                    wall_node = node_a if node_a.semantic_label == "WALL" else node_b
                    slab_node = node_b if node_a.semantic_label == "WALL" else node_a
                    if wall_node.elevation_bottom_m <= slab_node.elevation_top_m + 0.10:
                        self.add_relation(
                            source_id=wall_node.object_id,
                            target_id=slab_node.object_id,
                            relation_type="INTERSECTS",
                            confidence=0.90,
                            distance_m=0.0,
                        )

    def get_topology_reinforcement(self, object_id: str) -> tuple[float, list[str]]:
        """Compute topological confidence boost and reason codes for an object."""
        if object_id not in self.nodes:
            return 0.0, []

        node = self.nodes[object_id]
        boost = 0.0
        reasons: list[str] = []

        for edge in node.out_edges:
            if edge.relation_type == "HOSTED_BY":
                boost += 0.30
                reasons.append("HOST_WALL_CONFIRMED")
            elif edge.relation_type == "CONNECTED_TO":
                target = self.nodes.get(edge.target_id)
                if target and target.semantic_label == "PIPE":
                    boost += 0.25
                    reasons.append("PIPE_CONNECTION_CONFIRMED")
                elif target and target.semantic_label in ("EQUIPMENT", "HVAC_COMPONENT"):
                    boost += 0.20
                    reasons.append("MEP_EQUIPMENT_CONNECTION_CONFIRMED")
            elif edge.relation_type == "SUPPORTED_BY":
                boost += 0.20
                reasons.append("FOUNDATION_SUPPORT_CONFIRMED")

        return float(min(0.40, boost)), reasons

    def to_dict(self) -> dict[str, Any]:
        """Serialize topology graph for reports and export."""
        return {
            "node_count": len(self.nodes),
            "edge_count": len(self.edges),
            "nodes": {k: v.to_dict() for k, v in self.nodes.items()},
            "edges": [e.to_dict() for e in self.edges],
        }

    def _is_inside_bbox(self, child: TopologyNode, parent: TopologyNode, margin_m: float = 0.10) -> bool:
        """Check if child centroid lies inside parent bounding box extended by margin."""
        c = child.centroid_m
        p_min = parent.bbox_min_m - margin_m
        p_max = parent.bbox_max_m + margin_m
        return bool(np.all(c >= p_min) and np.all(c <= p_max))

    def _bbox_clearance(self, a: TopologyNode, b: TopologyNode) -> float:
        """Calculate minimum clearance distance between two 3D bounding boxes."""
        dx = max(0.0, float(max(a.bbox_min_m[0] - b.bbox_max_m[0], b.bbox_min_m[0] - a.bbox_max_m[0])))
        dy = max(0.0, float(max(a.bbox_min_m[1] - b.bbox_max_m[1], b.bbox_min_m[1] - a.bbox_max_m[1])))
        dz = max(0.0, float(max(a.bbox_min_m[2] - b.bbox_max_m[2], b.bbox_min_m[2] - a.bbox_max_m[2])))
        return float(np.sqrt(dx * dx + dy * dy + dz * dz))
