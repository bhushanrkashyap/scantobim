"""Segment classifier — assigns element type, discipline, and safety category.

Two modes:
  1. Rule-based (default) — fast, deterministic, no API cost
  2. GPT-4o (optional)   — richer reasoning, requires OPENAI_API_KEY
"""

from __future__ import annotations

import os
import uuid
from dataclasses import dataclass, field

import numpy as np
import structlog

from agent.models import (
    BoundingBox,
    Discipline,
    DSEARZone,
    ElementType,
    GeometrySegment,
    Point3D,
    SafetyCategory,
    SafetyClassification,
    SegmentShape,
)
from agent.tools.geometry_tools import build_mep_graph

logger = structlog.get_logger()


_MEP_LINEAR_TYPES = {
    ElementType.PIPE,
    ElementType.CONDUIT,
    ElementType.DUCT,
    ElementType.CABLE_TRAY,
    ElementType.DRAINAGE,
}

_MEP_TERMINAL_TYPES = {
    ElementType.ELECTRICAL_PANEL,
    ElementType.JUNCTION_BOX,
}

_ELECTRICAL_CARRIER_TYPES = {
    ElementType.CONDUIT,
    ElementType.CABLE_TRAY,
    ElementType.PIPE,
}


@dataclass(slots=True)
class BIMNode:
    """Scene-graph node wrapping a classified GeometrySegment."""

    segment: GeometrySegment
    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    type: ElementType | None = None
    host: str | None = None
    children: list[str] = field(default_factory=list)
    connections: list[str] = field(default_factory=list)
    parametric: dict[str, float] = field(default_factory=dict)
    confidence: float = 1.0

    @property
    def relationships(self) -> dict[str, object]:
        # Backward-compatible projection for existing call sites.
        return {
            "host": self.host,
            "contains": self.children,
            "connected_to": self.connections,
        }


def bb_volume(bb) -> float:
    dx = float(bb.max_x - bb.min_x)
    dy = float(bb.max_y - bb.min_y)
    dz = float(bb.max_z - bb.min_z)
    return max(dx, 0.0) * max(dy, 0.0) * max(dz, 0.0)


def bb_overlap(a, b) -> float:
    dx = max(0.0, min(float(a.max_x), float(b.max_x)) - max(float(a.min_x), float(b.min_x)))
    dy = max(0.0, min(float(a.max_y), float(b.max_y)) - max(float(a.min_y), float(b.min_y)))
    dz = max(0.0, min(float(a.max_z), float(b.max_z)) - max(float(a.min_z), float(b.min_z)))
    return dx * dy * dz


def bb_contained(inner, outer) -> bool:
    return (
        inner.min_x >= outer.min_x
        and inner.max_x <= outer.max_x
        and inner.min_y >= outer.min_y
        and inner.max_y <= outer.max_y
        and inner.min_z >= outer.min_z
        and inner.max_z <= outer.max_z
    )


def centroid(seg: GeometrySegment) -> np.ndarray:
    bb = seg.bounding_box
    return np.array(
        [
            (bb.min_x + bb.max_x) / 3.0,
            (bb.min_y + bb.max_y) / 2.0,
            (bb.min_z + bb.max_z) / 2.0,
        ],
        dtype=float,
    )


def distance(a: GeometrySegment, b: GeometrySegment) -> float:
    return float(np.linalg.norm(centroid(a) - centroid(b)))


def _tag_float(tags: dict, key: str) -> float | None:
    try:
        value = tags.get(key)
        if value is None:
            return None
        value_f = float(value)
        if np.isnan(value_f) or np.isinf(value_f):
            return None
        return value_f
    except (TypeError, ValueError):
        return None


def _populate_cable_tray_tags(seg: GeometrySegment) -> None:
    """Populate cable-tray tags from the segment extents using 3D principal axis fitting.

    Derives true 3D start/end points, width, and height without locking to X or Y.
    """
    tags = getattr(seg, "tags", None)
    if tags is None:
        tags = {}
        if hasattr(seg, "tags"):
            seg.tags = tags
    bb = seg.bounding_box
    dims = np.array([bb.max_x - bb.min_x, bb.max_y - bb.min_y, bb.max_z - bb.min_z], dtype=float)
    if np.any(dims <= 0.0):
        return

    pts = getattr(seg, "inlier_points", None)
    if pts is None:
        pts = getattr(seg, "points", None)
    if pts is not None and len(pts) >= 10:
        pts_arr = np.asarray(pts, dtype=float)
        mean_pt = np.mean(pts_arr, axis=0)
        centered = pts_arr - mean_pt
        cov = np.cov(centered, rowvar=False)
        eigenvalues, eigenvectors = np.linalg.eigh(cov)
        sort_idx = np.argsort(eigenvalues)[::-1]
        v_primary = eigenvectors[:, sort_idx[0]]
        v_secondary = eigenvectors[:, sort_idx[1]]
        v_tertiary = eigenvectors[:, sort_idx[2]]

        # Reject near-vertical runs for horizontal trays
        if abs(v_primary[2]) > 0.85:
            return

        proj_primary = np.dot(centered, v_primary)
        p_min = float(np.percentile(proj_primary, 2))
        p_max = float(np.percentile(proj_primary, 98))
        half_len = max(0.5 * (p_max - p_min), 100.0)

        proj_secondary = np.dot(centered, v_secondary)
        proj_tertiary = np.dot(centered, v_tertiary)
        span2 = float(np.percentile(proj_secondary, 95) - np.percentile(proj_secondary, 5))
        span3 = float(np.percentile(proj_tertiary, 95) - np.percentile(proj_tertiary, 5))

        width_mm = float(max(span2, span3))
        height_mm = float(min(span2, span3))
        axis_vec = v_primary / np.linalg.norm(v_primary)
        centroid = mean_pt
        tags["axis_fitting_method"] = "3d_pca"
    else:
        # Fallback to horizontal planar bounding box principal direction
        dx = dims[0]
        dy = dims[1]
        dz = dims[2]

        if dz > max(dx, dy) * 1.5:
            # Vertical element, not a horizontal cable tray
            return

        # Determine horizontal angle
        if dx >= dy:
            axis_vec = np.array([1.0, 0.0, 0.0], dtype=float)
            half_len = dx / 2.0
            width_mm = float(dy)
            height_mm = float(dz)
        else:
            axis_vec = np.array([0.0, 1.0, 0.0], dtype=float)
            half_len = dy / 2.0
            width_mm = float(dx)
            height_mm = float(dz)

        centroid = np.array([seg.centroid.x, seg.centroid.y, seg.centroid.z], dtype=float)
        tags["axis_fitting_method"] = "bbox_principal"

    # Architectural clamping for standard cable tray sizes
    width_mm = float(np.clip(width_mm, 150.0, 900.0))
    height_mm = float(np.clip(height_mm, 50.0, 200.0))
    tray_thickness_mm = float(np.clip(height_mm * 0.25, 20.0, 60.0))

    tags["width_mm"] = width_mm
    tags["height_mm"] = height_mm
    tags["tray_thickness_mm"] = tray_thickness_mm

    tags["cyl_axis_x"] = float(axis_vec[0])
    tags["cyl_axis_y"] = float(axis_vec[1])
    tags["cyl_axis_z"] = float(axis_vec[2])

    start = centroid - axis_vec * half_len
    end = centroid + axis_vec * half_len
    tags["cyl_start_x_mm"] = float(start[0])
    tags["cyl_start_y_mm"] = float(start[1])
    tags["cyl_start_z_mm"] = float(start[2])
    tags["cyl_end_x_mm"] = float(end[0])
    tags["cyl_end_y_mm"] = float(end[1])
    tags["cyl_end_z_mm"] = float(end[2])

    if hasattr(seg, "tags"):
        seg.tags = tags



def _preferred_pipe_radius_mm(seg: GeometrySegment, dx: float, dy: float, dz: float) -> float:
    tags = getattr(seg, "tags", {}) or {}
    fitted_radius = _tag_float(tags, "fitted_radius_mm")
    if fitted_radius is not None and fitted_radius > 0.0:
        return float(fitted_radius)

    tagged_diameter = _tag_float(tags, "diameter_mm")
    if tagged_diameter is not None and tagged_diameter > 0.0:
        return float(tagged_diameter / 2.0)

    tagged_radius = _tag_float(tags, "radius_mm")
    if tagged_radius is not None and tagged_radius > 0.0:
        return float(tagged_radius)

    return float(max(min(dx, dy, dz) / 2.0, 1.0))


def _linear_length_mm(seg: GeometrySegment, dx: float, dy: float, dz: float) -> float:
    tags = getattr(seg, "tags", {}) or {}
    trace_len = _tag_float(tags, "trace_length_mm")
    if trace_len is not None and trace_len > 0.0:
        return float(trace_len)

    endpoint_keys = (
        "cyl_start_x_mm",
        "cyl_start_y_mm",
        "cyl_start_z_mm",
        "cyl_end_x_mm",
        "cyl_end_y_mm",
        "cyl_end_z_mm",
    )
    endpoint_vals = [_tag_float(tags, key) for key in endpoint_keys]
    if all(v is not None for v in endpoint_vals):
        sx, sy, sz, ex, ey, ez = endpoint_vals
        return float(np.linalg.norm(np.array([ex - sx, ey - sy, ez - sz], dtype=float)))

    metre_endpoint_keys = (
        "_cyl_start_x_m",
        "_cyl_start_y_m",
        "_cyl_start_z_m",
        "_cyl_end_x_m",
        "_cyl_end_y_m",
        "_cyl_end_z_m",
    )
    metre_endpoint_vals = [_tag_float(tags, key) for key in metre_endpoint_keys]
    if all(v is not None for v in metre_endpoint_vals):
        sx, sy, sz, ex, ey, ez = metre_endpoint_vals
        return float(np.linalg.norm(np.array([ex - sx, ey - sy, ez - sz], dtype=float)) * 1000.0)

    return float(max(dx, dy, dz))


def is_parallel(seg_a: GeometrySegment, seg_b: GeometrySegment, tol: float = 0.95) -> bool:
    if seg_a.normal is None or seg_b.normal is None:
        return False

    na = np.array([seg_a.normal.x, seg_a.normal.y, seg_a.normal.z], dtype=float)
    nb = np.array([seg_b.normal.x, seg_b.normal.y, seg_b.normal.z], dtype=float)
    na_n = np.linalg.norm(na)
    nb_n = np.linalg.norm(nb)
    if na_n < 1e-9 or nb_n < 1e-9:
        return False
    dot = abs(float(np.dot(na / na_n, nb / nb_n)))
    return dot > tol


def build_scene_graph(segments: list[GeometrySegment]) -> dict[str, BIMNode]:
    """Build a BIM scene graph from segmented geometry."""
    nodes: dict[str, BIMNode] = {}

    for seg in segments:
        node = BIMNode(segment=seg)
        element_type, _discipline, safety = classify_segment(seg)
        node.type = element_type
        node.confidence = float(min(max(safety.confidence, 0.0), 1.0))
        nodes[node.id] = node

    _attach_topology(nodes)
    return nodes


def _attach_topology(nodes: dict[str, BIMNode]) -> None:
    node_list = list(nodes.values())

    for i in range(len(node_list)):
        a = node_list[i]
        for j in range(i + 1, len(node_list)):
            b = node_list[j]

            # Door/window hosted by wall when opening lies in wall envelope.
            if a.type in {ElementType.DOOR, ElementType.WINDOW} and b.type == ElementType.WALL:
                if bb_contained(a.segment.bounding_box, b.segment.bounding_box):
                    a.host = b.id
                    if a.id not in b.children:
                        b.children.append(a.id)
            if b.type in {ElementType.DOOR, ElementType.WINDOW} and a.type == ElementType.WALL:
                if bb_contained(b.segment.bounding_box, a.segment.bounding_box):
                    b.host = a.id
                    if b.id not in a.children:
                        a.children.append(b.id)

            if a.type in _MEP_TERMINAL_TYPES and b.type == ElementType.WALL:
                if bb_contained(a.segment.bounding_box, b.segment.bounding_box):
                    a.host = b.id
                    if a.id not in b.children:
                        b.children.append(a.id)
            if b.type in _MEP_TERMINAL_TYPES and a.type == ElementType.WALL:
                if bb_contained(b.segment.bounding_box, a.segment.bounding_box):
                    b.host = a.id
                    if b.id not in a.children:
                        a.children.append(b.id)

            # Pipe connectivity — enhanced with adaptive distance thresholds
            if a.type in _MEP_LINEAR_TYPES and b.type in _MEP_LINEAR_TYPES:
                a_rad = float(a.segment.tags.get("radius_mm", 50.0))
                b_rad = float(b.segment.tags.get("radius_mm", 50.0))
                avg_radius_mm = (a_rad + b_rad) / 2.0
                connection_radius_mm = max(500.0, avg_radius_mm * 5.0 + 200.0)

                horiz_dist = distance(a.segment, b.segment)
                z_a_mid = (
                    float(a.segment.bounding_box.min_z) + float(a.segment.bounding_box.max_z)
                ) / 2.0
                z_b_mid = (
                    float(b.segment.bounding_box.min_z) + float(b.segment.bounding_box.max_z)
                ) / 2.0
                z_sep = abs(z_a_mid - z_b_mid)

                if horiz_dist < connection_radius_mm and z_sep < 300.0:
                    if b.id not in a.connections:
                        a.connections.append(b.id)
                    if a.id not in b.connections:
                        b.connections.append(a.id)


def reconstruct_parametric(node: BIMNode) -> None:
    seg = node.segment
    bb = seg.bounding_box

    dx = float(bb.max_x - bb.min_x)
    dy = float(bb.max_y - bb.min_y)
    dz = float(bb.max_z - bb.min_z)

    if node.type == ElementType.WALL:
        # Respect the thickness computed by the segmentation pipeline
        # (stage-2 face pairing / angle-aware fallback). The AABB short side
        # min(dx, dy) is ONLY a valid thickness for axis-aligned walls: for
        # angled walls (e.g. 46°) the AABB is the diagonal envelope of the
        # wall plane and min(dx, dy) can be metres thick — never a thickness.
        thickness_mm: float | None = None
        raw_thickness = seg.tags.get("wall_thickness_mm")
        if raw_thickness is not None:
            try:
                thickness_mm = float(raw_thickness)
            except (TypeError, ValueError):
                thickness_mm = None

        if thickness_mm is None or thickness_mm <= 0.0:
            angle = float(seg.tags.get("wall_angle_deg", 0.0))
            off_axis = min(angle % 90.0, 90.0 - (angle % 90.0))
            if off_axis <= 5.0:
                thickness_mm = float(min(dx, dy))
            else:
                from agent.tools.detection_config import DEFAULT_WALL_THICKNESS_MM

                thickness_mm = float(DEFAULT_WALL_THICKNESS_MM)
            seg.tags["wall_thickness_mm"] = thickness_mm

        node.parametric = {
            "height_mm": dz,
            "thickness_mm": thickness_mm,
            "length_mm": max(dx, dy),
        }
    elif node.type in {ElementType.PIPE, ElementType.CONDUIT, ElementType.DRAINAGE}:
        radius_mm = _preferred_pipe_radius_mm(seg, dx, dy, dz)
        wall_mm = float(min(max(radius_mm * 0.24, 3.0), 25.0))
        node.parametric = {
            "radius_mm": radius_mm,
            "length_mm": _linear_length_mm(seg, dx, dy, dz),
            "pipe_wall_thickness_mm": wall_mm,
        }
        seg.tags["radius_mm"] = radius_mm
        seg.tags["pipe_wall_thickness_mm"] = wall_mm
    elif node.type in _MEP_TERMINAL_TYPES:
        shell_mm = float(min(max(min(dx, dy, dz) * 0.10, 20.0), 120.0))
        node.parametric = {
            "width_mm": dx,
            "depth_mm": dy,
            "height_mm": dz,
            "shell_thickness_mm": shell_mm,
        }
        seg.tags["shell_thickness_mm"] = shell_mm
    elif node.type == ElementType.FLOOR:
        node.parametric = {
            "thickness_mm": dz,
            "length_mm": dx,
            "width_mm": dy,
        }


def _connect_terminals_to_mep(nodes: dict[str, BIMNode]) -> None:
    """Connect MEP terminals (valves, fittings) to their carrier lines (pipes, ducts).

    Enhanced to:
    - Search across multiple levels (vertical stacks)
    - Use adaptive connection distance based on carrier size
    - Prefer closest carrier within range
    """
    carriers = [node for node in nodes.values() if node.type in _MEP_LINEAR_TYPES]
    terminals = [node for node in nodes.values() if node.type in _MEP_TERMINAL_TYPES]

    for terminal in terminals:
        preferred = [node for node in carriers if node.type in _ELECTRICAL_CARRIER_TYPES]
        candidates = preferred if preferred else carriers

        nearest: BIMNode | None = None
        nearest_dist = float("inf")

        for carrier in candidates:
            # Get carrier dimensions for adaptive connection radius
            carrier_rad = float(carrier.segment.tags.get("radius_mm", 50.0))
            # Terminals connect up to 3× carrier radius away + 150mm tolerance
            max_conn_dist = carrier_rad * 3.0 + 150.0

            # Horizontal (XY) distance
            horiz_dist = distance(terminal.segment, carrier.segment)

            # Vertical (Z) separation - allow wider search across levels
            z_t_mid = (
                float(terminal.segment.bounding_box.min_z)
                + float(terminal.segment.bounding_box.max_z)
            ) / 2.0
            z_c_mid = (
                float(carrier.segment.bounding_box.min_z)
                + float(carrier.segment.bounding_box.max_z)
            ) / 2.0
            z_sep = abs(z_t_mid - z_c_mid)

            # Accept if within XY distance AND Z separation allows level-to-level connection
            if horiz_dist <= max_conn_dist and z_sep <= 400.0:
                # Use combined metric: horiz + reduced Z weight (Z is less important for selection)
                combined_dist = horiz_dist + z_sep * 0.3
                if combined_dist < nearest_dist:
                    nearest = carrier
                    nearest_dist = combined_dist

        if nearest is None:
            continue
        if nearest.id not in terminal.connections:
            terminal.connections.append(nearest.id)
        if terminal.id not in nearest.connections:
            nearest.connections.append(terminal.id)


def reconstruct(node: BIMNode) -> None:
    """V2-friendly alias."""
    reconstruct_parametric(node)


def merge_score(a: GeometrySegment, b: GeometrySegment) -> float:
    return 0.5 * (1.0 if is_parallel(a, b) else 0.0) + 0.5 * (
        1.0 - min(distance(a, b) / 2000.0, 1.0)
    )


def merge_segments(segments: list[GeometrySegment]) -> list[GeometrySegment]:
    """Greedy global merge pass to reduce over-fragmentation."""
    merged: list[GeometrySegment] = []
    used: set[int] = set()

    for i, a in enumerate(segments):
        if i in used:
            continue

        group = [a]
        used.add(i)

        for j in range(i + 1, len(segments)):
            if j in used:
                continue
            b = segments[j]
            if merge_score(a, b) > 0.75:
                group.append(b)
                used.add(j)

        merged.append(_merge_group(group))

    return merged


def enforce_rules(nodes: dict[str, BIMNode]) -> None:
    """Apply coarse topology sanity checks and down-weight confidence on violations."""
    for node in nodes.values():
        if node.type == ElementType.DOOR and not node.host:
            node.confidence *= 0.4
        if node.type == ElementType.PIPE and not node.connections:
            node.confidence *= 0.6


def _has_locked_storey_span(seg: GeometrySegment) -> bool:
    tags = getattr(seg, "tags", {}) or {}
    bim_tags = tags.get("bim", {}) if isinstance(tags.get("bim", {}), dict) else {}
    return bool(
        bim_tags.get("auto_level_applied")
        or "reconstructed_storey_base_mm" in tags
        or "reconstructed_storey_top_mm" in tags
    )


def align_axes(segments: list[GeometrySegment], floor_step_mm: float = 3000.0) -> None:
    """Snap segment elevations to approximate building floor bands."""
    if floor_step_mm <= 0.0:
        return

    for seg in segments:
        if _has_locked_storey_span(seg):
            continue

        bb = seg.bounding_box
        z_mean = (bb.min_z + bb.max_z) / 2.0
        snapped = round(z_mean / floor_step_mm) * floor_step_mm
        shift = snapped - z_mean
        new_min_z = float(bb.min_z + shift)
        new_max_z = float(bb.max_z + shift)
        seg.bounding_box = bb.model_copy(
            update={
                "min_z": new_min_z,
                "max_z": new_max_z,
            }
        )
        seg.centroid.z += shift


def run_bim_pipeline_v2(segments: list[GeometrySegment]) -> dict[str, BIMNode]:
    """Grounded structural BIM pipeline for topology and parametrics."""
    align_axes(segments)
    enable_global_merge = os.environ.get("STB_ENABLE_BIM_GLOBAL_MERGE", "0").strip().lower() in {
        "1",
        "true",
        "yes",
        "on",
    }
    merged = merge_segments(segments) if enable_global_merge else segments
    nodes = build_scene_graph(merged)
    for node in nodes.values():
        reconstruct_parametric(node)
    enforce_rules(nodes)

    mep_nodes = [n for n in nodes.values() if n.type in _MEP_LINEAR_TYPES]
    if mep_nodes:
        mep_segments = []
        id_by_idx: dict[int, str] = {}
        for idx, node in enumerate(mep_nodes):
            seg = node.segment
            id_by_idx[idx] = node.id
            mep_segments.append(
                {
                    "shape": seg.shape,
                    "centroid": np.array(
                        [
                            seg.centroid.x / 1000.0,
                            seg.centroid.y / 1000.0,
                            seg.centroid.z / 1000.0,
                        ]
                    ),
                }
            )

        mep_graph = build_mep_graph(mep_segments)
        if mep_graph is not None:
            for a, b in mep_graph.edges():
                src = nodes[id_by_idx[int(a)]]
                dst = nodes[id_by_idx[int(b)]]
                if dst.id not in src.connections:
                    src.connections.append(dst.id)
                if src.id not in dst.connections:
                    dst.connections.append(src.id)

    _connect_terminals_to_mep(nodes)

    return nodes


def run_bim_pipeline(segments: list[GeometrySegment]) -> dict[str, BIMNode]:
    """Backward-compatible alias for pipeline callers."""
    return run_bim_pipeline_v2(segments)


def global_merge(segments: list[GeometrySegment]) -> list[GeometrySegment]:
    """Backward-compatible alias for existing merge callers."""
    return merge_segments(segments)


def _is_inside(inner: GeometrySegment, outer: GeometrySegment, tol_mm: float = 150.0) -> bool:
    a = inner.bounding_box
    b = outer.bounding_box
    return (
        a.min_x >= (b.min_x - tol_mm)
        and a.max_x <= (b.max_x + tol_mm)
        and a.min_y >= (b.min_y - tol_mm)
        and a.max_y <= (b.max_y + tol_mm)
        and a.min_z >= (b.min_z - tol_mm)
        and a.max_z <= (b.max_z + tol_mm)
    )


def _close_enough(a: GeometrySegment, b: GeometrySegment, threshold_mm: float = 300.0) -> bool:
    pa = np.array([a.centroid.x, a.centroid.y, a.centroid.z], dtype=float)
    pb = np.array([b.centroid.x, b.centroid.y, b.centroid.z], dtype=float)
    return float(np.linalg.norm(pa - pb)) <= threshold_mm


def _parallel(a: GeometrySegment, b: GeometrySegment, cos_tol: float = 0.95) -> bool:
    if a.normal is not None and b.normal is not None:
        na = np.array([a.normal.x, a.normal.y, a.normal.z], dtype=float)
        nb = np.array([b.normal.x, b.normal.y, b.normal.z], dtype=float)
        na_norm = np.linalg.norm(na)
        nb_norm = np.linalg.norm(nb)
        if na_norm > 1e-9 and nb_norm > 1e-9:
            return abs(float(np.dot(na / na_norm, nb / nb_norm))) >= cos_tol

    axa = np.array(
        [
            float(a.tags.get("cyl_axis_x", 0.0)),
            float(a.tags.get("cyl_axis_y", 0.0)),
            float(a.tags.get("cyl_axis_z", 0.0)),
        ],
        dtype=float,
    )
    axb = np.array(
        [
            float(b.tags.get("cyl_axis_x", 0.0)),
            float(b.tags.get("cyl_axis_y", 0.0)),
            float(b.tags.get("cyl_axis_z", 0.0)),
        ],
        dtype=float,
    )
    na_norm = np.linalg.norm(axa)
    nb_norm = np.linalg.norm(axb)
    if na_norm > 1e-9 and nb_norm > 1e-9:
        return abs(float(np.dot(axa / na_norm, axb / nb_norm))) >= cos_tol

    return False


def _distance(a: GeometrySegment, b: GeometrySegment) -> float:
    """AABB-to-AABB Euclidean gap distance in mm."""
    aa = a.bounding_box
    bb = b.bounding_box

    dx = max(0.0, max(aa.min_x, bb.min_x) - min(aa.max_x, bb.max_x))
    dy = max(0.0, max(aa.min_y, bb.min_y) - min(aa.max_y, bb.max_y))
    dz = max(0.0, max(aa.min_z, bb.min_z) - min(aa.max_z, bb.max_z))
    return float(np.sqrt(dx * dx + dy * dy + dz * dz))


def _merge_group(group: list[GeometrySegment]) -> GeometrySegment:
    if len(group) == 1:
        return group[0]

    first = group[0]
    min_x = min(s.bounding_box.min_x for s in group)
    min_y = min(s.bounding_box.min_y for s in group)
    min_z = min(s.bounding_box.min_z for s in group)
    max_x = max(s.bounding_box.max_x for s in group)
    max_y = max(s.bounding_box.max_y for s in group)
    max_z = max(s.bounding_box.max_z for s in group)

    total_points = sum(max(int(s.point_count), 1) for s in group)
    cx = sum(s.centroid.x * max(int(s.point_count), 1) for s in group) / total_points
    cy = sum(s.centroid.y * max(int(s.point_count), 1) for s in group) / total_points
    cz = sum(s.centroid.z * max(int(s.point_count), 1) for s in group) / total_points

    merged_tags: dict = {}
    for seg in group:
        merged_tags.update(seg.tags)
    merged_tags["merged_group_size"] = len(group)

    return first.model_copy(
        update={
            "segment_id": str(uuid.uuid4()),
            "bounding_box": BoundingBox(
                min_x=min_x,
                min_y=min_y,
                min_z=min_z,
                max_x=max_x,
                max_y=max_y,
                max_z=max_z,
            ),
            "centroid": Point3D(x=cx, y=cy, z=cz),
            "point_count": total_points,
            "confidence": max(float(s.confidence) for s in group),
            "tags": merged_tags,
        }
    )


def _env_flag(name: str, default: str = "0") -> bool:
    return os.environ.get(name, default).strip().lower() in {"1", "true", "yes", "on"}


def _is_pipe_like_linear_candidate(
    segment: GeometrySegment,
    *,
    major: float,
    minor1: float,
    minor2: float,
    has_trace_color: bool,
    has_cyl_trace: bool,
) -> bool:
    """Return True when a long, slender segment is more consistent with a pipe run than a tray."""
    if major < 800.0:
        return False

    tags = getattr(segment, "tags", {}) or {}
    trace_evidence = has_trace_color or has_cyl_trace or bool(_tag_float(tags, "trace_length_mm"))

    elongation = major / max(minor2, 1.0)
    cross_section_ratio = max(minor1, minor2) / max(min(minor1, minor2), 1.0)
    cross_section_span = max(minor1, minor2) / max(major, 1.0)
    cross_section_extent = max(minor1, minor2)
    cross_section_min = min(minor1, minor2)

    axis_tags_present = any(k in tags for k in ("cyl_axis_x", "cyl_axis_y", "cyl_axis_z"))
    radius_tags_present = any(
        _tag_float(tags, key) is not None
        for key in ("fitted_radius_mm", "radius_mm", "diameter_mm")
    )

    if elongation < 2.5:
        return False
    if cross_section_ratio > 3.0:
        return False
    if cross_section_span > 0.25:
        return False

    if trace_evidence or axis_tags_present or radius_tags_present:
        return True

    # Very elongated, compact boxes with a near-square/rectangular profile are
    # more consistent with pipe runs than cable trays when no tray-specific
    # evidence is present.
    return (
        major >= 2500.0
        and cross_section_min <= 250.0
        and cross_section_extent <= 450.0
        and elongation >= 5.0
        and cross_section_ratio <= 1.8
    )


def _is_cable_tray_candidate(
    segment: GeometrySegment,
    *,
    width: float,
    depth: float,
    height: float,
    minor1: float,
    minor2: float,
    major: float,
    face_ratio: float,
    centroid_z: float,
    dominant_axis_vertical: bool,
    has_trace_color: bool,
    has_cyl_trace: bool,
) -> bool:
    """Return True when a box-like segment behaves like a cable tray.

    The rule remains geometry-led and rejects line-like, pipe-like boxes that
    only resemble trays because they are very elongated.
    """
    if segment.shape != SegmentShape.BOX or major < 1200.0:
        return False

    if dominant_axis_vertical:
        return False

    elongation = major / max(minor2, 1.0)
    flatness = minor1 / max(minor2, 1.0)
    span_ratio = max(width, depth) / max(height, 1.0)

    floor_datum_mm = 0.0
    if segment.tags:
        for tag_key in ("floor_z_mm", "storey_base_z_mm", "level_elevation_mm", "host_wall_base_z_mm"):
            if tag_key in segment.tags:
                try:
                    floor_datum_mm = float(segment.tags[tag_key])
                    break
                except (ValueError, TypeError):
                    pass
    rel_z = centroid_z - floor_datum_mm

    if minor1 > 320.0 or minor2 > 700.0:
        return False
    if span_ratio < 2.0:
        return False
    if rel_z > 4500.0:
        return False
    if rel_z < 800.0:
        return False
    if height < 120.0:
        return False
    if max(width, depth) < 240.0:
        return False
    if height > max(width, depth) * 0.65:
        return False

    if _is_pipe_like_linear_candidate(
        segment,
        major=major,
        minor1=minor1,
        minor2=minor2,
        has_trace_color=has_trace_color,
        has_cyl_trace=has_cyl_trace,
    ):
        return False

    if flatness >= 0.75:
        return False

    score = 0.0
    score += 1.0 if elongation >= 2.2 else 0.0
    score += 1.0 if 0.20 <= flatness <= 0.60 else 0.0
    score += 0.75 if minor1 <= 320.0 and minor2 <= 700.0 else 0.0
    score += 0.5 if span_ratio >= 2.2 else 0.0
    score += 0.5 if rel_z < 3500.0 else 0.0
    score += 0.5 if has_trace_color or has_cyl_trace else 0.0
    score += 0.25 if face_ratio >= 1.8 else 0.0
    score += 0.25 if segment.point_count >= 24 else 0.0

    return score >= 2.5


# ── Rule-Based Classifier ─────────────────────────────────────────────────────


def classify_element_type(segment: GeometrySegment) -> ElementType:
    """Infer BIM element type from segment geometry.

    Priority within each shape branch is top-to-bottom — more specific rules first.
    All spatial values from bounding_box and centroid are in millimetres.
    Additional discriminators from the scan pipeline are stored in segment.tags.
    """
    # Anti-artifact filter: degenerate scan fragments should not become elements.
    # Keep synthetic/derived VOIDs with zero points (used for openings).
    if getattr(segment, "point_count", 0) <= 0 and segment.shape != SegmentShape.VOID:
        return ElementType.UNKNOWN

    bb = segment.bounding_box
    width = bb.max_x - bb.min_x  # mm
    depth = bb.max_y - bb.min_y  # mm
    height = bb.max_z - bb.min_z  # mm

    # Reject near-zero geometry extents to avoid creating invalid disjoint artifacts.
    if max(width, depth, height) < 50.0:
        return ElementType.UNKNOWN

    centroid_z = segment.centroid.z  # mm

    floor_datum_mm = 0.0
    if segment.tags:
        for tag_key in ("floor_z_mm", "storey_base_z_mm", "level_elevation_mm", "host_wall_base_z_mm"):
            if tag_key in segment.tags:
                try:
                    floor_datum_mm = float(segment.tags[tag_key])
                    break
                except (ValueError, TypeError):
                    pass
    rel_z = centroid_z - floor_datum_mm

    def _stair_steps_from_tags() -> int:
        raw = segment.tags.get("stair_steps", 0) if segment.tags else 0
        try:
            return max(int(float(raw)), 0)
        except (TypeError, ValueError):
            return 0

    def _is_steep_stair_candidate() -> bool:
        horizontal_run = max(width, depth, 1.0)
        slope_ratio = height / horizontal_run
        # Ramps are typically shallower; classify as stair when slope and rise
        # are both clearly stair-like even if stair_steps tagging is missing.
        return height >= 700.0 and slope_ratio >= 0.35

    # Stair recovery guard: noisy scans may fragment stairs into BOX segments.
    # When step evidence exists, prefer STAIR so it is not lost as generic beam/box.
    stair_steps = _stair_steps_from_tags()

    if stair_steps >= 2 and segment.shape in {
        SegmentShape.PLANE_SLOPED,
        SegmentShape.BOX,
        SegmentShape.PLANE_VERTICAL,
    }:
        return ElementType.STAIR

    # ── VALVE_CANDIDATE ───────────────────────────────────────────────────────
    if segment.shape == SegmentShape.VALVE_CANDIDATE:
        # Optional tracing mode: preserve inline fragments as pipes so routing
        # remains continuous when scans under-segment valves/fittings.
        if _env_flag("STB_PIPE_TRACE_MODE", "0"):
            return ElementType.PIPE

        dims_sorted = sorted([width, depth, height])
        minor1, minor2, major = dims_sorted
        # Safety Relief Valve: tiny upright body (<150mm), usually taller than wide
        if major < 160 and minor2 < 100 and height > width and height > depth:
            element_type = ElementType.SAFETY_RELIEF_VALVE
        # Expansion joint: short cylinder on pipe axis (horizontal, very short)
        elif major < 200 and height <= width:
            element_type = ElementType.EXPANSION_JOINT
        # Strainer: medium-small compact body (100–300mm)
        elif minor1 < 120 and minor2 < 200 and major < 320:
            element_type = ElementType.STRAINER
        # Deluge valve: large body on fire main
        elif major > 450:
            element_type = ElementType.DELUGE_VALVE
        else:
            element_type = ElementType.VALVE

    # ── BIM SEMANTIC CATEGORIES ─────────────────────────────────────────────
    # Replace primitive shapes with BIM categories using geometry, dimensions, elevation, normals, connectivity, and RGB
    elif segment.shape in [
        SegmentShape.VOID,
        SegmentShape.PLANE_VERTICAL,
        SegmentShape.PLANE_HORIZONTAL,
        SegmentShape.PLANE_SLOPED,
    ]:
        # --- Wall, Floor, Ceiling, Stair, Ramp, Railing, Door, Window, Opening ---
        z_extent = bb.max_z - bb.min_z
        xy_area = width * depth  # mm²
        wall_thickness_mm = float(segment.tags.get("wall_thickness_mm", 0))
        n_steps = _stair_steps_from_tags()
        density_per_m2 = segment.point_count / max((width / 1000.0) * (depth / 1000.0), 0.01)
        centroid_z = segment.centroid.z
        # --- Wall ---
        if segment.shape == SegmentShape.PLANE_VERTICAL:
            if height < 600 and height > 80:
                element_type = ElementType.BUND_WALL
            else:
                footprint_max = max(width, depth)
                footprint_min = min(width, depth, 1.0)
                aspect_ratio_fp = footprint_max / footprint_min
                if footprint_max < 800 and aspect_ratio_fp < 3.0:
                    element_type = ElementType.COLUMN
                else:
                    element_type = ElementType.WALL
        # --- Floor, Ceiling, Grating ---
        elif segment.shape == SegmentShape.PLANE_HORIZONTAL:
            if density_per_m2 < 20.0 and xy_area > 250_000:
                element_type = ElementType.GRATING
            elif segment.normal and segment.normal.z < -0.30:
                element_type = ElementType.CEILING
            elif segment.normal and segment.normal.z > 0.30:
                element_type = ElementType.FLOOR
            elif rel_z > 1800:
                element_type = ElementType.CEILING
            else:
                element_type = ElementType.FLOOR
        # --- Stair, Ramp ---
        elif segment.shape == SegmentShape.PLANE_SLOPED:
            element_type = (
                ElementType.STAIR
                if (n_steps >= 2 or _is_steep_stair_candidate())
                else ElementType.RAMP
            )
        # --- Door, Window, Opening ---
        elif segment.shape == SegmentShape.VOID:
            # Synthetic openings can legitimately have zero points; classify them
            # from geometry rather than dropping them as invalid artifacts.
            if wall_thickness_mm > 800:
                element_type = ElementType.CONTAINMENT_PENETRATION
            elif wall_thickness_mm > 600:
                element_type = ElementType.PENETRATION_SEAL
            elif z_extent < 400 and xy_area > 90_000:
                length_to_width = max(width, depth) / max(min(width, depth), 1.0)
                if length_to_width > 5:
                    element_type = ElementType.TRENCH
                elif xy_area < 600_000:
                    element_type = ElementType.HATCH
                else:
                    # Filter noisy/placeholder voids that carry no point support.
                    # These are frequently produced as "hatch/void" artifacts and
                    # should not become slab openings.
                    if getattr(segment, "point_count", 0) <= 0:
                        element_type = ElementType.UNKNOWN
                    else:
                        element_type = ElementType.SLAB_OPENING

            else:
                # Discriminate Door vs Window using sill height relative to floor/wall datum
                sill_mm = None
                if segment.tags and "opening_sill_mm" in segment.tags:
                    try:
                        sill_mm = float(segment.tags["opening_sill_mm"])
                    except (ValueError, TypeError):
                        sill_mm = None
                if sill_mm is None:
                    sill_mm = bb.min_z - floor_datum_mm

                if sill_mm < 300.0:
                    element_type = ElementType.DOOR
                else:
                    element_type = ElementType.WINDOW

    elif segment.shape == SegmentShape.CYLINDER:
        dims_sorted = sorted([width, depth, height])  # [minor1, minor2, major] mm
        minor1_d, minor2_d, major_d = dims_sorted
        diameter_mm = (minor1_d + minor2_d) / 2
        aspect_ratio = major_d / max(diameter_mm, 1.0)
        all_small = major_d < 180  # very small dims under 180 mm → micro-fixture
        color_hex = (
            str(segment.tags.get("trace_color_hex") or segment.dominant_color or "")
        ).upper()

        is_vertical = False
        if segment.tags:
            is_vertical = abs(float(segment.tags.get("cyl_axis_z", 0.0))) > 0.7
        if segment.normal:
            is_vertical = is_vertical or abs(segment.normal.z) > 0.7

        if is_vertical and diameter_mm > 400.0 and major_d > 1000.0:
            element_type = ElementType.TANK
            return element_type

        try:
            r = int(color_hex[1:3], 16) if len(color_hex) == 7 else 0
            g = int(color_hex[3:5], 16) if len(color_hex) == 7 else 0
            b = int(color_hex[5:7], 16) if len(color_hex) == 7 else 0
        except (ValueError, IndexError):
            r = g = b = 0

        is_red = r > 170 and g < 120 and b < 120

        # ── Nuclear primary circuit vessels (check FIRST — SC1) ──────────────
        # Steam generator: very large cylinder (> 10 m tall, > 1.5 m diameter)
        if diameter_mm > 1500 and major_d > 10_000:
            element_type = ElementType.STEAM_GENERATOR
        # Pressurizer: tall narrow cylinder (aspect ratio > 7, height > 5 m)
        elif 400 < diameter_mm < 1600 and aspect_ratio > 7 and major_d > 5000:
            element_type = ElementType.PRESSURIZER
        # Heat exchanger: medium cylinder, moderate aspect ratio 3–7
        elif 200 < diameter_mm < 1500 and 3.0 < aspect_ratio <= 7.0 and major_d > 1000:
            element_type = ElementType.HEAT_EXCHANGER
        # Pressure vessel: stocky large cylinder (aspect ratio 1–3, vol > ~1 m³)
        elif 300 < diameter_mm < 2500 and 1.0 <= aspect_ratio <= 3.0 and major_d > 1000:
            bb_vol_m3 = (width / 1000) * (depth / 1000) * (height / 1000)
            if bb_vol_m3 > 1.0:
                element_type = ElementType.PRESSURE_VESSEL
            else:
                element_type = ElementType.TANK
        # ── Seismic isolator: wide flat disc under equipment ──────────────────
        elif diameter_mm > 250 and major_d < 150 and rel_z < 500:
            element_type = ElementType.SEISMIC_ISOLATOR
        # ── Portable fire extinguisher: short upright cylinder at body height ──
        elif (
            80 <= diameter_mm <= 250
            and 300 <= major_d <= 900
            and 700 <= rel_z <= 1800
            or (
                is_red
                and 70 <= diameter_mm <= 260
                and 250 <= major_d <= 1000
                and major_d > diameter_mm
            )
        ):
            element_type = ElementType.FIRE_EXTINGUISHER
        # ── Micro-fixtures ────────────────────────────────────────────────────
        elif major_d >= 120 and aspect_ratio >= 1.6 and 30 <= diameter_mm <= 500:
            # Slender cylinders are usually pipe-like runs even when short.
            element_type = ElementType.PIPE
        elif all_small:
            if rel_z > 2400 and major_d < 80:
                element_type = ElementType.SMOKE_DETECTOR  # tiny ceiling disc
            elif 600 < rel_z < 900 and diameter_mm > 40:
                element_type = ElementType.FIRE_HYDRANT  # standpipe at hose height
            elif rel_z > 2200:
                element_type = ElementType.SPRINKLER  # ceiling fire head
            elif rel_z < 200 and aspect_ratio < 1.8:
                element_type = ElementType.DRAINAGE  # squat floor drain-like
            else:
                element_type = ElementType.VALVE  # small cylindrical fixture
        # ── Standard piping (with color heuristics) ───────────────────────────
        elif diameter_mm > 1200:
            element_type = ElementType.TANK
        elif diameter_mm < 60:
            element_type = ElementType.CONDUIT
        else:
            # Color-based sub-classification for pipes
            # Simple color bucket logic
            try:
                is_blue = b > 160 and r < 140
                is_yellow = r > 170 and g > 150 and b < 130
                is_green = g > 150 and r < 140

                # Color only selects service class after CYLINDER geometry has
                # already decided this is a pipe-like object.
                if is_blue or is_yellow or is_green:
                    element_type = ElementType.PIPE
                else:
                    element_type = ElementType.PIPE
            except (ValueError, IndexError):
                element_type = ElementType.PIPE

    # ── BOX ──────────────────────────────────────────────────────────────────
    elif segment.shape == SegmentShape.BOX:
        dims_sorted = sorted([width, depth, height])  # [minor1, minor2, major] mm
        minor1, minor2, major = dims_sorted
        face_ratio = major / max(minor2, 1.0)
        # Is the dominant (longest) dimension vertical?
        dominant_axis_vertical = (height > width) and (height > depth)

        color_hex = (
            str(segment.tags.get("trace_color_hex") or segment.dominant_color or "")
        ).upper()

        try:
            r = int(color_hex[1:3], 16) if len(color_hex) == 7 else 0
            g = int(color_hex[3:5], 16) if len(color_hex) == 7 else 0
            b = int(color_hex[5:7], 16) if len(color_hex) == 7 else 0
        except (ValueError, IndexError):
            r = g = b = 0

        is_yellow = r > 170 and g > 150 and b < 120
        is_red = r > 180 and g < 120 and b < 120
        is_blue = b > 160 and r < 120 and g < 160
        is_green = g > 150 and r < 130 and b < 140
        is_cyan = g > 140 and b > 140 and r < 120
        is_orange = r > 180 and 90 < g < 190 and b < 100
        has_trace_color = bool(segment.tags.get("trace_color_hex"))
        has_cyl_trace = any(
            k in segment.tags
            for k in ("cyl_start_x_mm", "cyl_end_x_mm", "trace_axis_z", "trace_length_mm")
        )

        extinguisher_like = (
            dominant_axis_vertical
            and 700 <= centroid_z <= 1800
            and 90 < minor1 < 300
            and minor2 < 380
            and 300 < major < 1100
        )

        # Pipe recovery: elongated slim BOX fragments with trace evidence should
        # remain pipes, not be downgraded to beam/generic due noisy shape fitting.
        if _is_pipe_like_linear_candidate(
            segment,
            major=major,
            minor1=minor1,
            minor2=minor2,
            has_trace_color=has_trace_color,
            has_cyl_trace=has_cyl_trace,
        ):
            return ElementType.PIPE

        # Avoid forcing semantics on tiny sparse fragments; these are frequently
        # residual noise from scan edges. Keep likely extinguishers for safety.
        if segment.point_count < 12 and major < 500 and not extinguisher_like:
            return ElementType.UNKNOWN

        tray_candidate = _is_cable_tray_candidate(
            segment,
            width=width,
            depth=depth,
            height=height,
            minor1=minor1,
            minor2=minor2,
            major=major,
            face_ratio=face_ratio,
            centroid_z=centroid_z,
            dominant_axis_vertical=dominant_axis_vertical,
            has_trace_color=has_trace_color,
            has_cyl_trace=has_cyl_trace,
        )

        if tray_candidate:
            _populate_cable_tray_tags(segment)

        # Waste bin / floor container: compact floor-level box, often painted red/yellow.
        # Keep this before panel/cabinet rules so bins are not mis-routed to MEP trays.
        if tray_candidate:
            element_type = ElementType.CABLE_TRAY

        elif (
            (is_yellow or is_red)
            and 220 <= minor1 <= 900
            and 220 <= minor2 <= 900
            and 400 <= major <= 1400
            and rel_z < 1200
            and not (
                dominant_axis_vertical
                and 100 < minor1 < 250
                and minor2 < 350
                and 350 < major < 900
                and 700 <= rel_z <= 1800
            )
        ):
            element_type = ElementType.GENERIC_MODEL

        # ── Large-scale nuclear / industrial equipment (highest priority) ─────
        # Overhead crane: very high centroid, very long horizontal span
        elif rel_z > 4000 and major > 5000 and not dominant_axis_vertical:
            element_type = ElementType.OVERHEAD_CRANE
        # Emergency diesel generator: very large box (all dims >> 800mm)
        elif minor1 > 800 and minor2 > 1000 and major > 3000:
            element_type = ElementType.EMERGENCY_DIESEL_GENERATOR
        # Transformer: large electrical box, floor or plinth-mounted
        elif minor1 > 500 and minor2 > 600 and 800 < major < 4000 and rel_z < 2500:
            element_type = ElementType.TRANSFORMER
        # Compressor: elongated floor-mounted mechanical unit
        elif minor1 > 350 and minor2 > 450 and major > 1500 and rel_z < 1500:
            element_type = ElementType.COMPRESSOR
        # Switchgear / MCC: tall narrow floor-standing cabinet row
        elif minor1 < 700 and minor2 > 400 and major > 1500 and 800 <= rel_z <= 2000:
            element_type = ElementType.SWITCHGEAR
        # UPS / battery rack: long low profile row of cabinets
        elif 200 < minor1 < 700 and 200 < minor2 < 700 and major > 2000 and rel_z < 1200:
            element_type = ElementType.UPS_SYSTEM
        # ── Vertical elongated cluster → ladder ───────────────────────────────
        elif dominant_axis_vertical and height > 2000 and minor1 < 600 and minor2 < 600:
            element_type = ElementType.LADDER
        # ── Fire extinguisher cabinet/body: compact upright object at body height ─
        elif extinguisher_like:
            element_type = ElementType.FIRE_EXTINGUISHER
        # ── Color-traced linear runs (explicit service paint tracing) ───────
        elif has_trace_color and major > 700 and face_ratio > 3.0 and minor1 < 260:
            if is_cyan:
                element_type = ElementType.DUCT
            elif is_green:
                element_type = ElementType.CONDUIT
            elif is_orange:
                element_type = ElementType.CABLE_TRAY
            elif is_blue or is_yellow:
                element_type = ElementType.PIPE
            else:
                element_type = ElementType.PIPE
        # ── Long narrow runs should not be pulled into monitor/panel buckets ──
        elif major > 700 and face_ratio > 3.0 and minor1 < 220:
            if minor2 <= 120:
                element_type = ElementType.CONDUIT
            elif rel_z > 2400 and minor1 < 180 and minor2 < 280:
                element_type = ElementType.LIGHTING_FITTING
            else:
                element_type = ElementType.BEAM
        # ── Radiation monitor: compact wall/ceiling box ───────────────────────
        elif (
            minor1 < 120
            and minor2 < 220
            and 80 < major < 420
            and 800 < rel_z < 2600
            and face_ratio < 2.2
            and segment.point_count >= 20
        ):
            element_type = ElementType.RADIATION_MONITOR
        # ── Fire alarm panel: thin wall-mounted box ───────────────────────────
        elif minor1 < 100 and 150 < minor2 < 600 and major < 700 and 1000 < rel_z < 2200:
            element_type = ElementType.FIRE_ALARM_PANEL
        # ── Junction box: tiny wall enclosure ────────────────────────────────
        elif minor1 < 100 and minor2 < 300 and major < 300:
            element_type = ElementType.JUNCTION_BOX
        # ── Pipe support: small bracket/hanger, usually slender but not tray-sized ─
        elif minor1 < 120 and 250 < major < 1000 and face_ratio > 2.5 and 500 < rel_z < 3500:
            element_type = ElementType.PIPE_SUPPORT
        # ── Lighting fitting: small elongated near ceiling ────────────────────
        elif rel_z > 2200 and minor1 < 200 and 200 < major < 2000:
            element_type = ElementType.LIGHTING_FITTING
        # ── Low civil elements ────────────────────────────────────────────────
        elif rel_z < 150 and 350 < major < 1400 and not dominant_axis_vertical:
            element_type = ElementType.KERB
        elif rel_z < 150 and minor1 < 220 and face_ratio < 2.2 and major < 600:
            element_type = ElementType.DRAINAGE
        # ── Architectural elements at body height ─────────────────────────────
        elif 800 <= rel_z <= 1200 and minor1 < 200:
            element_type = ElementType.RAILING
        elif minor1 < 250 and minor2 > 400 and face_ratio < 3 and 1000 <= rel_z <= 2000:
            element_type = ElementType.ELECTRICAL_PANEL
        elif (
            major / max(minor2, 1.0) >= 2.5
            and not dominant_axis_vertical
            and (rel_z > 1800 or (segment.normal is not None and abs(segment.normal.z) < 0.2))
        ):
            element_type = ElementType.BEAM
        elif (
            minor1 > 200
            and minor2 > 250
            and major > 1000
            and rel_z > 1500
            and (is_cyan or segment.tags.get("mep_system") == "hvac" or "duct" in segment.tags or bool(segment.tags.get("hollow")))
        ):
            element_type = ElementType.DUCT
        elif minor1 > 400 and minor2 > 400:
            element_type = ElementType.PUMP if rel_z < 1000 else ElementType.HVAC_EQUIPMENT
        else:
            element_type = ElementType.GENERIC_MODEL

    # ── PLANE_SLOPED ─────────────────────────────────────────────────────────
    elif segment.shape == SegmentShape.PLANE_SLOPED:
        # D3: stair_steps tag populated by count_stair_steps() in detect_planes()
        n_steps = _stair_steps_from_tags()
        element_type = (
            ElementType.STAIR if (n_steps >= 2 or _is_steep_stair_candidate()) else ElementType.RAMP
        )

    # ── PLANE_HORIZONTAL ─────────────────────────────────────────────────────
    elif segment.shape == SegmentShape.PLANE_HORIZONTAL:
        # Grating: sparse point density (scanner sees through open-steel mesh)
        bb_area_m2 = max((width / 1000.0) * (depth / 1000.0), 0.01)
        density_per_m2 = segment.point_count / bb_area_m2
        thin_dim_mm = min(width, depth, height)
        if density_per_m2 < 120.0 and bb_area_m2 > 0.40 and thin_dim_mm < 300.0:
            element_type = ElementType.GRATING
        elif segment.normal and segment.normal.z < -0.30:
            element_type = ElementType.CEILING
        elif segment.normal and segment.normal.z > 0.30:
            element_type = ElementType.FLOOR
        elif rel_z > 1800:
            element_type = ElementType.CEILING
        else:
            element_type = ElementType.FLOOR

    # ── PLANE_VERTICAL ───────────────────v────────────────────────────────────
    elif segment.shape == SegmentShape.PLANE_VERTICAL:
        # Bund wall: very low vertical plane (containment bund around tanks)
        # NOTE: Use point_count/density to reduce misclassification from sparse noise.
        # height/width/depth are in mm.
        wall_thickness_mm = float(segment.tags.get("wall_thickness_mm", 0))
        density_per_m2 = segment.point_count / max((width / 1000.0) * (depth / 1000.0), 0.01)

        if height < 600 and height > 80:
            # require a minimum density so tiny random vertical patches don’t become bund walls
            element_type = ElementType.BUND_WALL if density_per_m2 >= 5.0 else ElementType.WALL
        else:
            footprint_max = max(width, depth)
            footprint_min = max(min(width, depth), 1.0)
            aspect_ratio_fp = footprint_max / footprint_min

            # Columns: compact verticals, higher point density
            if footprint_max < 800 and aspect_ratio_fp < 3.0 and density_per_m2 >= 8.0:
                element_type = ElementType.COLUMN
            else:
                element_type = ElementType.WALL

        # If the segment is explicitly tagged as a wall thickness penetration seal candidate,
        # keep as wall-derived element to avoid collapsing to column.
        if wall_thickness_mm and wall_thickness_mm > 800 and height > 400:
            element_type = ElementType.WALL

    # ── CURVED_STRUCTURAL_SURFACE ───────────────────────────────────────────
    elif segment.shape == SegmentShape.CURVED_STRUCTURAL_SURFACE:
        element_type = ElementType.CURVED_STRUCTURAL_SURFACE

    # ── DEFAULT / UNKNOWN ───────────────────────────────────────────────────
    else:
        # Check if color gives a hint
        color = (segment.dominant_color or "").upper()
        if color == "#FF0000":
            element_type = ElementType.FIRE_ALARM_PANEL
        elif color == "#FFFF00":  # Yellow
            # Do not force yellow to pipe here; shape rules decide first.
            element_type = ElementType.DRAINAGE
        else:
            element_type = ElementType.WALL

    logger.debug(
        "element_type_classified",
        segment_id=segment.segment_id[:8],
        shape=segment.shape.value,
        width_mm=round(width),
        depth_mm=round(depth),
        height_mm=round(height),
        centroid_z_mm=round(centroid_z),
        point_count=segment.point_count,
        element_type=element_type.value,
    )
    return element_type


def classify_discipline(element_type: ElementType) -> Discipline:
    """Map element type to discipline."""
    mapping = {
        # Structural
        ElementType.WALL: Discipline.STRUCTURAL,
        ElementType.FLOOR: Discipline.STRUCTURAL,
        ElementType.CEILING: Discipline.ARCHITECTURAL,
        ElementType.COLUMN: Discipline.STRUCTURAL,
        ElementType.BEAM: Discipline.STRUCTURAL,
        ElementType.OVERHEAD_CRANE: Discipline.STRUCTURAL,
        ElementType.CURVED_STRUCTURAL_SURFACE: Discipline.STRUCTURAL,
        ElementType.SEISMIC_ISOLATOR: Discipline.STRUCTURAL,
        ElementType.CONTAINMENT_PENETRATION: Discipline.STRUCTURAL,
        ElementType.PENETRATION_SEAL: Discipline.STRUCTURAL,
        ElementType.SLAB_OPENING: Discipline.STRUCTURAL,
        # Architectural / Access
        ElementType.STAIR: Discipline.ARCHITECTURAL,
        ElementType.RAMP: Discipline.ARCHITECTURAL,
        ElementType.DOOR: Discipline.ARCHITECTURAL,
        ElementType.WINDOW: Discipline.ARCHITECTURAL,
        ElementType.RAILING: Discipline.ARCHITECTURAL,
        ElementType.LADDER: Discipline.ARCHITECTURAL,
        ElementType.GRATING: Discipline.ARCHITECTURAL,
        ElementType.HATCH: Discipline.ARCHITECTURAL,
        # Civil
        ElementType.TRENCH: Discipline.CIVIL,
        ElementType.BUND_WALL: Discipline.CIVIL,
        ElementType.KERB: Discipline.CIVIL,
        # MEP — Piping / Inline
        ElementType.PIPE: Discipline.MEP,
        ElementType.CONDUIT: Discipline.MEP,
        ElementType.DUCT: Discipline.MEP,
        ElementType.CABLE_TRAY: Discipline.MEP,
        ElementType.HVAC_EQUIPMENT: Discipline.MEP,
        ElementType.TANK: Discipline.MEP,
        ElementType.PUMP: Discipline.MEP,
        ElementType.VALVE: Discipline.MEP,
        ElementType.SAFETY_RELIEF_VALVE: Discipline.MEP,
        ElementType.STRAINER: Discipline.MEP,
        ElementType.EXPANSION_JOINT: Discipline.MEP,
        ElementType.PIPE_SUPPORT: Discipline.MEP,
        ElementType.DRAINAGE: Discipline.MEP,
        ElementType.ELECTRICAL_PANEL: Discipline.MEP,
        # MEP — Vessels / Equipment
        ElementType.PRESSURE_VESSEL: Discipline.MEP,
        ElementType.HEAT_EXCHANGER: Discipline.MEP,
        ElementType.COMPRESSOR: Discipline.MEP,
        ElementType.PRESSURIZER: Discipline.MEP,
        ElementType.STEAM_GENERATOR: Discipline.MEP,
        ElementType.EMERGENCY_DIESEL_GENERATOR: Discipline.MEP,
        ElementType.RADIATION_MONITOR: Discipline.MEP,
        # Electrical (classified under MEP discipline)
        ElementType.TRANSFORMER: Discipline.MEP,
        ElementType.SWITCHGEAR: Discipline.MEP,
        ElementType.UPS_SYSTEM: Discipline.MEP,
        ElementType.JUNCTION_BOX: Discipline.MEP,
        ElementType.LIGHTING_FITTING: Discipline.MEP,
        # Fire / Life Safety
        ElementType.SPRINKLER: Discipline.FIRE_PROTECTION,
        ElementType.FIRE_HYDRANT: Discipline.FIRE_PROTECTION,
        ElementType.FIRE_EXTINGUISHER: Discipline.FIRE_PROTECTION,
        ElementType.DELUGE_VALVE: Discipline.FIRE_PROTECTION,
        ElementType.FIRE_ALARM_PANEL: Discipline.FIRE_PROTECTION,
        ElementType.SMOKE_DETECTOR: Discipline.FIRE_PROTECTION,
        ElementType.FIRE_DAMPER: Discipline.FIRE_PROTECTION,
    }
    return mapping.get(element_type, Discipline.STRUCTURAL)


def classify_safety_rule_based(
    segment: GeometrySegment,
    element_type: ElementType,
    zone_safety_override: SafetyCategory | None = None,
) -> SafetyClassification:
    """Rule-based safety classification.

    Heuristics for PoV:
    - Columns with height > 3000mm and confidence < 0.75 → SC2 (structural risk)
    - Pipes in certain zones → SC3 (MEP safety-relevant)
    - Everything else → NS

    In production, this would use a trained model + site-specific rules.
    """
    if zone_safety_override:
        return SafetyClassification(
            segment_id=segment.segment_id,
            safety_category=zone_safety_override,
            dsear_zone=DSEARZone.NONE,
            classification_reason=f"Zone-level override: {zone_safety_override.value}",
            confidence=1.0,
            classified_by="rule_engine",
        )

    bb = segment.bounding_box
    height = bb.max_z - bb.min_z
    width = bb.max_x - bb.min_x

    # Very large walls (potential containment/shielding) → SC1
    # Use area-normalised point density so sparse real-scan data triggers correctly.
    # 5 pts/m² is a conservative lower bound for any usable RANSAC plane fit.
    bb_area_sqm = max((width / 1000.0) * (height / 1000.0), 0.01)  # mm² → m²
    point_density = segment.point_count / bb_area_sqm  # pts/m²
    if element_type == ElementType.WALL and height > 3500 and width > 8000 and point_density >= 5.0:
        return SafetyClassification(
            segment_id=segment.segment_id,
            safety_category=SafetyCategory.SC1,
            dsear_zone=DSEARZone.NONE,
            classification_reason=(
                f"Large structural wall {width:.0f}x{height:.0f}mm "
                f"(density {point_density:.1f} pts/m²) — "
                f"potential containment/radiation shielding. Requires manual design."
            ),
            confidence=segment.confidence,
            classified_by="rule_engine",
        )

    # Doors: fire/blast doors in nuclear/industrial facilities → SC2
    if element_type == ElementType.DOOR:
        return SafetyClassification(
            segment_id=segment.segment_id,
            safety_category=SafetyCategory.SC2,
            dsear_zone=DSEARZone.NONE,
            classification_reason=(
                "Door opening detected — potential fire door or blast door. "
                "Requires engineer classification before placement."
            ),
            confidence=segment.confidence,
            classified_by="rule_engine",
        )

    # Windows: non-safety by default unless in classified zones
    if element_type == ElementType.WINDOW:
        return SafetyClassification(
            segment_id=segment.segment_id,
            safety_category=SafetyCategory.NS,
            dsear_zone=DSEARZone.NONE,
            classification_reason="Window opening — no safety flags detected",
            confidence=segment.confidence,
            classified_by="rule_engine",
        )

    # Railings: life-safety (fall protection) → SC3
    if element_type == ElementType.RAILING:
        return SafetyClassification(
            segment_id=segment.segment_id,
            safety_category=SafetyCategory.SC3,
            dsear_zone=DSEARZone.NONE,
            classification_reason="Railing — fall-protection element, flagged for life-safety review",
            confidence=segment.confidence,
            classified_by="rule_engine",
        )

    # Structural beams with low confidence → SC2
    # Beams are safety-related in nuclear/critical facilities (collapse risk)
    if element_type == ElementType.BEAM and segment.confidence < 0.75:
        return SafetyClassification(
            segment_id=segment.segment_id,
            safety_category=SafetyCategory.SC2,
            dsear_zone=DSEARZone.NONE,
            classification_reason=(
                f"Structural beam with low confidence ({segment.confidence:.2f}) — "
                f"requires engineer verification before placement"
            ),
            confidence=segment.confidence,
            classified_by="rule_engine",
        )

    # Stairs → SC3 (life safety element — fire egress)
    if element_type == ElementType.STAIR:
        return SafetyClassification(
            segment_id=segment.segment_id,
            safety_category=SafetyCategory.SC3,
            dsear_zone=DSEARZone.NONE,
            classification_reason="Stair segment — flagged for egress/life safety review",
            confidence=segment.confidence,
            classified_by="rule_engine",
        )

    # Ramps → NS (accessibility element, non-safety by default)
    if element_type == ElementType.RAMP:
        return SafetyClassification(
            segment_id=segment.segment_id,
            safety_category=SafetyCategory.NS,
            dsear_zone=DSEARZone.NONE,
            classification_reason="Ramp segment — smooth sloped surface, no stair treads detected",
            confidence=segment.confidence,
            classified_by="rule_engine",
        )

    # Tall structural columns with low confidence → SC2
    if element_type == ElementType.COLUMN and height > 3000 and segment.confidence < 0.75:
        return SafetyClassification(
            segment_id=segment.segment_id,
            safety_category=SafetyCategory.SC2,
            dsear_zone=DSEARZone.NONE,
            classification_reason=(
                f"Structural column height={height:.0f}mm with low confidence "
                f"({segment.confidence:.2f}) requires engineer review"
            ),
            confidence=segment.confidence,
            classified_by="rule_engine",
        )

    # Large load-bearing walls → SC3
    if element_type == ElementType.WALL and height > 3200 and segment.point_count > 3000:
        return SafetyClassification(
            segment_id=segment.segment_id,
            safety_category=SafetyCategory.SC3,
            dsear_zone=DSEARZone.NONE,
            classification_reason=(
                f"Large wall height={height:.0f}mm, high density ({segment.point_count} pts), "
                f"flagged for load-bearing review"
            ),
            confidence=segment.confidence,
            classified_by="rule_engine",
        )

    # Tanks: large vessels (volume > ~5 000 L ≈ 5 m³) → SC2 (pressure/containment risk)
    if element_type == ElementType.TANK:
        bb_volume_m3 = (width / 1000) * (height / 1000) * ((bb.max_y - bb.min_y) / 1000)
        if False:  # bb_volume_m3 > 5.0:
            return SafetyClassification(
                segment_id=segment.segment_id,
                safety_category=SafetyCategory.SC2,
                dsear_zone=DSEARZone.ZONE_1,
                classification_reason=(
                    f"Large vessel {bb_volume_m3:.1f} m³ — potential pressure/hazardous "
                    f"storage. Requires engineer sign-off."
                ),
                confidence=segment.confidence,
                classified_by="rule_engine",
            )
        return SafetyClassification(
            segment_id=segment.segment_id,
            safety_category=SafetyCategory.SC3,
            dsear_zone=DSEARZone.ZONE_2,
            classification_reason=f"Tank/vessel {bb_volume_m3:.1f} m³ — MEP safety review",
            confidence=segment.confidence,
            classified_by="rule_engine",
        )

    # HVAC equipment and pumps → SC3
    if element_type in (ElementType.HVAC_EQUIPMENT, ElementType.PUMP):
        return SafetyClassification(
            segment_id=segment.segment_id,
            safety_category=SafetyCategory.SC3,
            dsear_zone=DSEARZone.NONE,
            classification_reason=f"{element_type.value} — flagged for MEP system review",
            confidence=segment.confidence,
            classified_by="rule_engine",
        )

    # Ducts → SC3 (smoke/fire damper potential in nuclear facilities)
    if element_type == ElementType.DUCT:
        return SafetyClassification(
            segment_id=segment.segment_id,
            safety_category=SafetyCategory.SC3,
            dsear_zone=DSEARZone.NONE,
            classification_reason="Duct segment — check for fire/smoke damper requirements",
            confidence=segment.confidence,
            classified_by="rule_engine",
        )

    # Cable trays and conduits → NS (electrical containment, non-safety default)
    if element_type in (ElementType.CABLE_TRAY, ElementType.CONDUIT):
        return SafetyClassification(
            segment_id=segment.segment_id,
            safety_category=SafetyCategory.NS,
            dsear_zone=DSEARZone.NONE,
            classification_reason=f"{element_type.value} — electrical containment, no safety flags",
            confidence=segment.confidence,
            classified_by="rule_engine",
        )

    # Pipes → SC3 (potential DSEAR relevance)
    if element_type == ElementType.PIPE:
        return SafetyClassification(
            segment_id=segment.segment_id,
            safety_category=SafetyCategory.SC3,
            dsear_zone=DSEARZone.ZONE_2,
            classification_reason="Pipe segment — flagged for MEP safety review",
            confidence=segment.confidence,
            classified_by="rule_engine",
        )

    # Valves: inline process valves in nuclear/industrial facilities → SC2
    # (isolation and control valves affect process safety systems)
    if element_type == ElementType.VALVE:
        return SafetyClassification(
            segment_id=segment.segment_id,
            safety_category=SafetyCategory.SC2,
            dsear_zone=DSEARZone.ZONE_1,
            classification_reason=(
                "Inline valve on process pipe — potential isolation/control valve. "
                "Requires P&ID cross-check and engineer approval."
            ),
            confidence=segment.confidence,
            classified_by="rule_engine",
        )

    # Sprinklers → SC2 (active fire suppression — safety-related)
    if element_type == ElementType.SPRINKLER:
        return SafetyClassification(
            segment_id=segment.segment_id,
            safety_category=SafetyCategory.SC2,
            dsear_zone=DSEARZone.NONE,
            classification_reason=(
                "Sprinkler head detected — active fire suppression system. "
                "Requires fire engineer approval before placement."
            ),
            confidence=segment.confidence,
            classified_by="rule_engine",
        )

    # Slab openings → SC2 (structural penetration — affects slab integrity)
    if element_type == ElementType.SLAB_OPENING:
        return SafetyClassification(
            segment_id=segment.segment_id,
            safety_category=SafetyCategory.SC2,
            dsear_zone=DSEARZone.NONE,
            classification_reason=(
                "Slab opening/penetration detected — structural integrity impact. "
                "Requires structural engineer sign-off."
            ),
            confidence=segment.confidence,
            classified_by="rule_engine",
        )

    # Electrical panels → SC3 (electrical safety)
    if element_type == ElementType.ELECTRICAL_PANEL:
        return SafetyClassification(
            segment_id=segment.segment_id,
            safety_category=SafetyCategory.SC3,
            dsear_zone=DSEARZone.NONE,
            classification_reason="Electrical panel — flagged for electrical safety review",
            confidence=segment.confidence,
            classified_by="rule_engine",
        )

    # Kerbs and drainage → NS (civil/drainage — non-safety by default)
    if element_type in (ElementType.KERB, ElementType.DRAINAGE):
        return SafetyClassification(
            segment_id=segment.segment_id,
            safety_category=SafetyCategory.NS,
            dsear_zone=DSEARZone.NONE,
            classification_reason=f"{element_type.value} — civil element, no safety flags",
            confidence=segment.confidence,
            classified_by="rule_engine",
        )

    # ── SC1 — Nuclear primary-circuit / safety-critical equipment ─────────────
    _SC1_TYPES = (
        ElementType.CONTAINMENT_PENETRATION,
        ElementType.PRESSURIZER,
        ElementType.STEAM_GENERATOR,
        ElementType.EMERGENCY_DIESEL_GENERATOR,
        ElementType.SAFETY_RELIEF_VALVE,
    )
    if element_type in _SC1_TYPES:
        return SafetyClassification(
            segment_id=segment.segment_id,
            safety_category=SafetyCategory.SC1,
            dsear_zone=DSEARZone.NONE,
            classification_reason=(
                f"{element_type.value} — safety-critical nuclear/process element. "
                f"NEVER auto-create. Requires nuclear engineer sign-off."
            ),
            confidence=segment.confidence,
            classified_by="rule_engine",
        )

    # ── SC2 — Safety-related equipment requiring human approval ───────────────
    _SC2_TYPES = (
        ElementType.PRESSURE_VESSEL,
        ElementType.HEAT_EXCHANGER,
        ElementType.TRANSFORMER,
        ElementType.UPS_SYSTEM,
        ElementType.FIRE_HYDRANT,
        ElementType.DELUGE_VALVE,
        ElementType.FIRE_ALARM_PANEL,
        ElementType.PENETRATION_SEAL,
        ElementType.OVERHEAD_CRANE,
        ElementType.COMPRESSOR,
        ElementType.SEISMIC_ISOLATOR,
        ElementType.RADIATION_MONITOR,
    )
    if element_type in _SC2_TYPES:
        return SafetyClassification(
            segment_id=segment.segment_id,
            safety_category=SafetyCategory.SC2,
            dsear_zone=DSEARZone.ZONE_1
            if element_type
            in (
                ElementType.PRESSURE_VESSEL,
                ElementType.HEAT_EXCHANGER,
                ElementType.COMPRESSOR,
            )
            else DSEARZone.NONE,
            classification_reason=(
                f"{element_type.value} — safety-related element. "
                f"Requires engineer approval before Revit placement."
            ),
            confidence=segment.confidence,
            classified_by="rule_engine",
        )

    # ── SC3 — Safety-relevant, flagged but auto-proceed ────────────────────────
    _SC3_TYPES = (
        ElementType.SWITCHGEAR,
        ElementType.FIRE_DAMPER,
        ElementType.STRAINER,
        ElementType.EXPANSION_JOINT,
        ElementType.BUND_WALL,
        ElementType.FIRE_EXTINGUISHER,
        ElementType.SMOKE_DETECTOR,
    )
    if element_type in _SC3_TYPES:
        return SafetyClassification(
            segment_id=segment.segment_id,
            safety_category=SafetyCategory.SC3,
            dsear_zone=DSEARZone.NONE,
            classification_reason=(
                f"{element_type.value} — flagged for safety review, auto-proceed"
            ),
            confidence=segment.confidence,
            classified_by="rule_engine",
        )

    # ── NS — Non-safety civil/access/minor electrical elements ────────────────
    _NS_TYPES = (
        ElementType.GRATING,
        ElementType.LADDER,
        ElementType.TRENCH,
        ElementType.HATCH,
        ElementType.JUNCTION_BOX,
        ElementType.LIGHTING_FITTING,
        ElementType.PIPE_SUPPORT,
    )
    if element_type in _NS_TYPES:
        return SafetyClassification(
            segment_id=segment.segment_id,
            safety_category=SafetyCategory.NS,
            dsear_zone=DSEARZone.NONE,
            classification_reason=f"{element_type.value} — no safety flags, auto-proceed",
            confidence=segment.confidence,
            classified_by="rule_engine",
        )

    # Default: non-safety
    return SafetyClassification(
        segment_id=segment.segment_id,
        safety_category=SafetyCategory.NS,
        dsear_zone=DSEARZone.NONE,
        classification_reason="No safety flags detected — auto-proceed",
        confidence=segment.confidence,
        classified_by="rule_engine",
    )


# ── GPT-4o Classifier (Optional) ──────────────────────────────────────────────


async def classify_safety_gpt4o(
    segment: GeometrySegment,
    element_type: ElementType,
) -> SafetyClassification:
    """Use GPT-4o to classify safety category with richer reasoning."""
    try:
        from openai import AsyncOpenAI
    except ImportError:
        logger.warning("openai not installed, falling back to rule-based")
        return classify_safety_rule_based(segment, element_type)

    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        logger.warning("OPENAI_API_KEY not set, falling back to rule-based")
        return classify_safety_rule_based(segment, element_type)

    client = AsyncOpenAI(api_key=api_key)
    model = os.getenv("OPENAI_MODEL", "gpt-4o")

    prompt = f"""Classify the safety category of this BIM element for a nuclear/critical infrastructure facility.

Element type: {element_type.value}
Shape: {segment.shape.value}
Height (mm): {segment.bounding_box.max_z - segment.bounding_box.min_z:.0f}
Width (mm): {segment.bounding_box.max_x - segment.bounding_box.min_x:.0f}
Point count: {segment.point_count}
Detection confidence: {segment.confidence:.2f}

Safety categories:
- SC1: Safety-critical — containment, reactor building structure, radiation shielding. NEVER auto-create.
- SC2: Safety-related — load-bearing near safety zones, fire barriers, DSEAR boundaries. Requires human approval.
- SC3: Safety-relevant — structural elements near safety systems, MEP in classified areas. Flagged but auto-proceed.
- NS: Non-safety — standard elements with no safety implications.

Respond with JSON only:
{{"safety_category": "SC1|SC2|SC3|NS", "dsear_zone": "Zone0|Zone1|Zone2|None", "reason": "brief explanation"}}"""

    try:
        response = await client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": prompt}],
            response_format={"type": "json_object"},
            temperature=0.1,
            max_tokens=200,
        )

        import json

        result = json.loads(response.choices[0].message.content)

        return SafetyClassification(
            segment_id=segment.segment_id,
            safety_category=SafetyCategory(result["safety_category"]),
            dsear_zone=DSEARZone(result.get("dsear_zone", "None")),
            classification_reason=result.get("reason", "GPT-4o classification"),
            confidence=segment.confidence,
            classified_by="gpt-4o",
        )
    except Exception as e:
        logger.error("gpt4o_classification_failed", error=str(e))
        return classify_safety_rule_based(segment, element_type)


# ── Main Classifier Entry Point ────────────────────────────────────────────────


def classify_segment(
    segment: GeometrySegment,
    zone_safety_override: SafetyCategory | None = None,
) -> tuple[ElementType, Discipline, SafetyClassification]:
    """Full classification pipeline: element type → discipline → safety."""
    element_type = classify_element_type(segment)

    # Prefer core BIM component outputs by default so downstream authoring tools
    # receive standard wall/opening classes instead of STB-specific subtypes.
    if _env_flag("STB_NORMALIZE_CORE_BIM_TYPES", "1"):
        if element_type == ElementType.BUND_WALL:
            element_type = ElementType.WALL
        elif element_type in {ElementType.HATCH, ElementType.TRENCH}:
            element_type = ElementType.SLAB_OPENING

    discipline = classify_discipline(element_type)
    safety = classify_safety_rule_based(segment, element_type, zone_safety_override)

    logger.info(
        "segment_classified",
        segment_id=segment.segment_id[:8],
        element_type=element_type.value,
        discipline=discipline.value,
        safety=safety.safety_category.value,
        reason=safety.classification_reason[:60],
    )

    return element_type, discipline, safety
