from __future__ import annotations

from bisect import bisect_right
from collections import defaultdict
from dataclasses import dataclass
from uuid import uuid4

import numpy as np

from agent.classifier import _populate_cable_tray_tags
from agent.models import ElementType, GeometrySegment, SegmentShape

_WALL_LIKE_TYPES: tuple[ElementType, ...] = (
    ElementType.WALL,
    ElementType.BUND_WALL,
)

_STAIR_TYPES: tuple[ElementType, ...] = (
    ElementType.STAIR,
    ElementType.RAMP,
)


@dataclass(slots=True)
class ReconstructedNode:
    id: str
    segment: GeometrySegment
    type: ElementType
    host: str | None
    children: list[str]
    connections: list[str]
    parametric: dict[str, float]
    confidence: float


def _is_type(seg: GeometrySegment, element_type: ElementType) -> bool:
    et = seg.element_type
    if isinstance(et, ElementType):
        return et == element_type
    if isinstance(et, str):
        return et.lower() == element_type.value
    return False


def _is_wall_like(seg: GeometrySegment) -> bool:
    return any(_is_type(seg, element_type) for element_type in _WALL_LIKE_TYPES)


def _is_stair_like(seg: GeometrySegment) -> bool:
    return any(_is_type(seg, element_type) for element_type in _STAIR_TYPES)


def _segment_center(seg: GeometrySegment) -> np.ndarray:
    bb = seg.bounding_box
    return np.array(
        [
            (bb.min_x + bb.max_x) / 2.0,
            (bb.min_y + bb.max_y) / 2.0,
            (bb.min_z + bb.max_z) / 2.0,
        ],
        dtype=float,
    )


def _segment_dims(seg: GeometrySegment) -> tuple[float, float, float]:
    bb = seg.bounding_box
    return (
        float(bb.max_x - bb.min_x),
        float(bb.max_y - bb.min_y),
        float(bb.max_z - bb.min_z),
    )


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


def _wall_axis_vector(seg: GeometrySegment) -> np.ndarray:
    """Return the canonical 2D wall axis unit vector for a segment."""
    from agent.tools.geometry_tools import canonicalize_wall_axis

    tags = getattr(seg, "tags", {}) or {}
    try:
        ax = float(tags.get("wall_axis_x", 0.0))
        ay = float(tags.get("wall_axis_y", 0.0))
        if abs(ax) > 1e-6 or abs(ay) > 1e-6:
            return canonicalize_wall_axis([ax, ay])
    except (TypeError, ValueError):
        pass

    if seg.normal:
        nx = float(seg.normal.x)
        ny = float(seg.normal.y)
        if abs(nx) > 1e-6 or abs(ny) > 1e-6:
            return canonicalize_wall_axis([-ny, nx])
    return np.array([1.0, 0.0], dtype=float)


def _wall_axis(seg: GeometrySegment) -> str:
    """Return wall orientation identifier preserving arbitrary angles."""
    from agent.tools.geometry_tools import wall_angle_degrees

    vec = _wall_axis_vector(seg)
    angle = wall_angle_degrees(vec)
    if abs(vec[0]) >= 0.98:
        return "X"
    if abs(vec[1]) >= 0.98:
        return "Y"
    return f"ANG_{round(angle, 1)}"


def _shell_thickness_mm(seg: GeometrySegment) -> float:
    dims = sorted(max(v, 1.0) for v in _segment_dims(seg))
    return float(np.clip(dims[0] * 0.10, 20.0, 120.0))


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


def _pipe_wall_thickness_mm(radius_mm: float) -> float:
    radius = max(radius_mm, 0.0)
    return float(np.clip(radius * 0.24, 3.0, 25.0))


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


def _detail_parametric(seg: GeometrySegment, element_type: ElementType) -> dict[str, float]:
    dx, dy, dz = _segment_dims(seg)
    params: dict[str, float] = {}

    if element_type in {ElementType.PIPE, ElementType.CONDUIT, ElementType.DRAINAGE}:
        radius_mm = _preferred_pipe_radius_mm(seg, dx, dy, dz)
        params = {
            "radius_mm": radius_mm,
            "length_mm": _linear_length_mm(seg, dx, dy, dz),
            "pipe_wall_thickness_mm": _pipe_wall_thickness_mm(radius_mm),
        }
    elif element_type == ElementType.CABLE_TRAY:
        tags = getattr(seg, "tags", {}) or {}
        if not any(k in tags for k in ("width_mm", "height_mm", "tray_thickness_mm", "cyl_axis_x")):
            _populate_cable_tray_tags(seg)
            tags = getattr(seg, "tags", {}) or {}
        width_mm = _tag_float(tags, "width_mm")
        height_mm = _tag_float(tags, "height_mm")
        tray_thickness_mm = _tag_float(tags, "tray_thickness_mm")
        params = {
            "length_mm": _linear_length_mm(seg, dx, dy, dz),
            "width_mm": float(width_mm)
            if width_mm is not None and width_mm > 0.0
            else float(max(dx, dy)),
            "height_mm": float(height_mm)
            if height_mm is not None and height_mm > 0.0
            else float(min(dx, dy, dz)),
            "tray_thickness_mm": float(tray_thickness_mm)
            if tray_thickness_mm is not None and tray_thickness_mm > 0.0
            else float(min(dx, dy, dz)),
        }
    elif element_type in {ElementType.ELECTRICAL_PANEL, ElementType.JUNCTION_BOX}:
        params = {
            "width_mm": float(dx),
            "depth_mm": float(dy),
            "height_mm": float(dz),
            "shell_thickness_mm": _shell_thickness_mm(seg),
        }

    for key, value in params.items():
        seg.tags[key] = value

    return params


def distance(a: GeometrySegment, b: GeometrySegment) -> float:
    return float(np.linalg.norm(_segment_center(a) - _segment_center(b)))


def is_inside(inner: GeometrySegment, outer: GeometrySegment, tol_mm: float = 100.0) -> bool:
    ia = inner.bounding_box
    ob = outer.bounding_box
    return (
        ia.min_x >= (ob.min_x - tol_mm)
        and ia.max_x <= (ob.max_x + tol_mm)
        and ia.min_y >= (ob.min_y - tol_mm)
        and ia.max_y <= (ob.max_y + tol_mm)
        and ia.min_z >= (ob.min_z - tol_mm)
        and ia.max_z <= (ob.max_z + tol_mm)
    )


def _axis_overlap(min_a: float, max_a: float, min_b: float, max_b: float) -> float:
    return max(0.0, min(max_a, max_b) - max(min_a, min_b))


def _is_within_wall_envelope(
    inner: GeometrySegment,
    wall: GeometrySegment,
    *,
    run_margin_mm: float,
    depth_margin_mm: float,
    z_margin_mm: float,
) -> bool:
    ib = inner.bounding_box
    wb = wall.bounding_box
    axis = _wall_axis(wall)

    if axis == "X":
        run_ok = ib.max_x >= (wb.min_x - run_margin_mm) and ib.min_x <= (wb.max_x + run_margin_mm)
        depth_ok = ib.min_y >= (wb.min_y - depth_margin_mm) and ib.max_y <= (
            wb.max_y + depth_margin_mm
        )
    elif axis == "Y":
        run_ok = ib.max_y >= (wb.min_y - run_margin_mm) and ib.min_y <= (wb.max_y + run_margin_mm)
        depth_ok = ib.min_x >= (wb.min_x - depth_margin_mm) and ib.max_x <= (
            wb.max_x + depth_margin_mm
        )
    else:
        vec = _wall_axis_vector(wall)
        norm_xy = np.array([-vec[1], vec[0]], dtype=float)
        wc = np.array([wall.centroid.x, wall.centroid.y], dtype=float)
        ic = np.array([inner.centroid.x, inner.centroid.y], dtype=float)
        d_run = abs(float(np.dot(ic - wc, vec)))
        d_perp = abs(float(np.dot(ic - wc, norm_xy)))
        w_len = float(wall.tags.get("wall_length_mm", max(wb.max_x - wb.min_x, wb.max_y - wb.min_y)))
        w_thick = float(wall.tags.get("wall_thickness_mm", 200.0))
        run_ok = d_run <= (w_len / 2.0 + run_margin_mm)
        depth_ok = d_perp <= (w_thick / 2.0 + depth_margin_mm)

    z_ok = ib.max_z >= (wb.min_z - z_margin_mm) and ib.min_z <= (wb.max_z + z_margin_mm)
    return run_ok and depth_ok and z_ok


def _is_component_embedded_in_wall(component: GeometrySegment, wall: GeometrySegment) -> bool:
    """Decide if a component is embedded *within* the wall thickness envelope.

    This is intentionally stricter than raw bbox-inside checks to prevent
    left/right leakage where components appear on both faces.

    Side-of-wall is handled by requiring the component centroid to be close
    to one of the wall faces, not merely the wall centerline.
    """
    cb = component.bounding_box
    wb = wall.bounding_box
    axis = _wall_axis(wall)

    if axis == "X":
        # Wall runs along X; thickness is Y
        wall_thickness = max(float(wb.max_y - wb.min_y), 1.0)
        comp_depth = float(cb.max_y - cb.min_y)
        wall_center = (wb.min_y + wb.max_y) / 2.0
        comp_center = (cb.min_y + cb.max_y) / 2.0
        wall_face_dist = min(abs(cb.min_y - wb.min_y), abs(cb.max_y - wb.max_y))
        run_overlap = _axis_overlap(cb.min_x, cb.max_x, wb.min_x, wb.max_x)
        run_span = max(min(cb.max_x - cb.min_x, wb.max_x - wb.min_x), 1.0)
    else:
        # Wall runs along Y; thickness is X
        wall_thickness = max(float(wb.max_x - wb.min_x), 1.0)
        comp_depth = float(cb.max_x - cb.min_x)
        wall_center = (wb.min_x + wb.max_x) / 2.0
        comp_center = (cb.min_x + cb.max_x) / 2.0
        wall_face_dist = min(abs(cb.min_x - wb.min_x), abs(cb.max_x - wb.max_x))
        run_overlap = _axis_overlap(cb.min_y, cb.max_y, wb.min_y, wb.max_y)
        run_span = max(min(cb.max_y - cb.min_y, wb.max_y - wb.min_y), 1.0)

    # Require meaningful run overlap
    if run_overlap < 0.50 * run_span:
        return False

    # Component should be near one wall face, not centered inside the wall.
    if wall_face_dist > max(60.0, wall_thickness * 0.35):
        return False

    center_offset = abs(comp_center - wall_center)
    if abs(center_offset - wall_thickness / 2.0) > max(50.0, wall_thickness * 0.35):
        return False

    # Also ensure component depth is not wildly larger than the wall thickness
    if comp_depth > wall_thickness * 1.35:
        return False

    return _is_within_wall_envelope(
        component,
        wall,
        run_margin_mm=120.0,
        depth_margin_mm=40.0,
        z_margin_mm=120.0,
    )


def _is_linear_component_adjacent_to_wall(
    component: GeometrySegment,
    wall: GeometrySegment,
    *,
    face_margin_mm: float = 250.0,
    run_margin_mm: float = 250.0,
    z_margin_mm: float = 220.0,
) -> bool:
    """Return True when a linear MEP run is plausibly hosted by a wall face."""
    cb = component.bounding_box
    wb = wall.bounding_box
    axis = _wall_axis(wall)

    if axis == "X":
        face_dist = min(abs(cb.min_y - wb.min_y), abs(cb.max_y - wb.max_y))
        run_ok = cb.max_x >= (wb.min_x - run_margin_mm) and cb.min_x <= (wb.max_x + run_margin_mm)
    else:
        face_dist = min(abs(cb.min_x - wb.min_x), abs(cb.max_x - wb.max_x))
        run_ok = cb.max_y >= (wb.min_y - run_margin_mm) and cb.min_y <= (wb.max_y + run_margin_mm)

    z_ok = cb.max_z >= (wb.min_z - z_margin_mm) and cb.min_z <= (wb.max_z + z_margin_mm)
    return face_dist <= face_margin_mm and run_ok and z_ok


def _is_door_aligned_to_wall_opening(door: GeometrySegment, wall: GeometrySegment) -> bool:
    db = door.bounding_box
    wb = wall.bounding_box
    axis = _wall_axis(wall)

    if axis == "X":
        wall_center = (wb.min_y + wb.max_y) / 2.0
        door_center = (db.min_y + db.max_y) / 2.0
        wall_thickness = max(float(wb.max_y - wb.min_y), 1.0)
        run_overlap = _axis_overlap(db.min_x, db.max_x, wb.min_x, wb.max_x)
        run_span = max(min(db.max_x - db.min_x, wb.max_x - wb.min_x), 1.0)
    else:
        wall_center = (wb.min_x + wb.max_x) / 2.0
        door_center = (db.min_x + db.max_x) / 2.0
        wall_thickness = max(float(wb.max_x - wb.min_x), 1.0)
        run_overlap = _axis_overlap(db.min_y, db.max_y, wb.min_y, wb.max_y)
        run_span = max(min(db.max_y - db.min_y, wb.max_y - wb.min_y), 1.0)

    if run_overlap < 0.30 * run_span:
        return False

    if abs(door_center - wall_center) > max(120.0, wall_thickness / 2.0 + 80.0):
        return False

    z_overlap = _axis_overlap(db.min_z, db.max_z, wb.min_z, wb.max_z)
    z_span = max(min(db.max_z - db.min_z, wb.max_z - wb.min_z), 1.0)
    if z_overlap < 0.20 * z_span:
        return False

    return _is_within_wall_envelope(
        door,
        wall,
        run_margin_mm=200.0,
        depth_margin_mm=max(180.0, wall_thickness),
        z_margin_mm=180.0,
    )


def detect_levels(segments: list[GeometrySegment], bins: str | int = "auto") -> list[float]:
    """Infer structural storey levels from horizontal plane detections (mm).

    Uses Z-centroid clustering across horizontal plane detections produced
    by Stage 2 (floors, ceilings, gratings, horizontal slabs).
    """
    if not segments:
        return [0.0, 3000.0]

    horiz_z: list[float] = []
    weights: list[int] = []

    for seg in segments:
        shape_val = getattr(seg.shape, "value", seg.shape)
        et_val = getattr(seg.element_type, "value", seg.element_type)
        is_horiz = (
            shape_val in (SegmentShape.PLANE_HORIZONTAL, "plane_horizontal")
            or str(et_val).lower() in ("floor", "ceiling", "grating")
        )
        if is_horiz:
            zc = float(seg.centroid.z)
            pts = max(int(getattr(seg, "point_count", 1)), 1)
            horiz_z.append(zc)
            weights.append(pts)

    if not horiz_z:
        # Fallback: estimate from bounding box min/max if no horizontal planes detected
        z_vals = [float(s.bounding_box.min_z) for s in segments] + [
            float(s.bounding_box.max_z) for s in segments
        ]
        if not z_vals:
            return [0.0, 3000.0]
        z_min = float(np.min(z_vals))
        z_max = float(np.max(z_vals))
        if z_max - z_min < 1000.0:
            return [z_min, z_min + 3000.0]
        levels = [z_min]
        curr = z_min + 3000.0
        while curr < z_max + 750.0:
            levels.append(curr)
            curr += 3000.0
        if z_max > levels[-1] + 1000.0:
            levels.append(z_max)
        return sorted({round(float(L), 1) for L in levels})

    # Cluster horizontal plane Z-centroids with 350mm bucket tolerance
    sorted_pairs = sorted(zip(horiz_z, weights), key=lambda p: p[0])
    clusters: list[list[tuple[float, int]]] = []
    curr_cluster: list[tuple[float, int]] = [sorted_pairs[0]]

    for zc, w in sorted_pairs[1:]:
        if zc - curr_cluster[-1][0] <= 350.0:
            curr_cluster.append((zc, w))
        else:
            clusters.append(curr_cluster)
            curr_cluster = [(zc, w)]
    clusters.append(curr_cluster)

    # Compute weighted average Z for each cluster
    clustered_levels: list[float] = []
    for cl in clusters:
        total_w = sum(w for _, w in cl)
        mean_z = sum(zc * w for zc, w in cl) / max(total_w, 1)
        clustered_levels.append(mean_z)

    # Filter levels that are too close to each other (< 1200mm)
    distinct_levels: list[float] = []
    for L in sorted(clustered_levels):
        if not distinct_levels:
            distinct_levels.append(L)
        elif L - distinct_levels[-1] >= 1200.0:
            distinct_levels.append(L)

    # Ensure building base extent is captured
    all_min_z = float(min(s.bounding_box.min_z for s in segments))
    all_max_z = float(max(s.bounding_box.max_z for s in segments))

    if distinct_levels and distinct_levels[0] - all_min_z > 2000.0:
        distinct_levels.insert(0, all_min_z)

    if distinct_levels and all_max_z - distinct_levels[-1] > 2000.0:
        distinct_levels.append(all_max_z)

    if len(distinct_levels) < 2:
        base = distinct_levels[0] if distinct_levels else 0.0
        distinct_levels = [base, base + 3000.0]

    return sorted({round(float(L), 1) for L in distinct_levels})


def _level_span_pairs(
    levels: list[float], fallback_span_mm: float = 3000.0
) -> list[tuple[float, float]]:
    if not levels:
        return [(0.0, fallback_span_mm)]
    if len(levels) == 1:
        return [(levels[0], levels[0] + fallback_span_mm)]
    return [(float(a), float(b)) for a, b in zip(levels[:-1], levels[1:]) if b > a]


def _segment_storey_index(seg: GeometrySegment, levels: list[float]) -> int:
    if len(levels) < 2:
        return 0
    bb = seg.bounding_box
    spans = _level_span_pairs(levels)

    if not spans:
        zc = float((bb.min_z + bb.max_z) / 2.0)
        idx = bisect_right(levels, zc) - 1
        return max(0, min(idx, len(levels) - 2))

    overlaps: list[float] = []
    for z0, z1 in spans:
        overlap = max(0.0, min(float(bb.max_z), float(z1)) - max(float(bb.min_z), float(z0)))
        overlaps.append(float(overlap))

    best_overlap = max(overlaps)
    if best_overlap > 0.0:
        best_idxs = [i for i, ov in enumerate(overlaps) if abs(ov - best_overlap) <= 1e-6]
        if len(best_idxs) == 1:
            return best_idxs[0]

        zc = float((bb.min_z + bb.max_z) / 2.0)
        centroid_idx = bisect_right(levels, zc) - 1
        centroid_idx = max(0, min(centroid_idx, len(spans) - 1))
        if centroid_idx in best_idxs:
            return centroid_idx
        return best_idxs[0]

    zc = float((bb.min_z + bb.max_z) / 2.0)
    idx = bisect_right(levels, zc) - 1
    return max(0, min(idx, len(levels) - 2))


# -----------------------------------------------------------------------------
# 1. WALL RECONSTRUCTION (AXIS-BASED)
# -----------------------------------------------------------------------------


def reconstruct_walls(segments: list[GeometrySegment]) -> list[GeometrySegment]:
    walls: list[GeometrySegment] = []
    wall_groups: dict[str, list[GeometrySegment]] = defaultdict(list)

    for seg in segments:
        if not _is_wall_like(seg):
            continue

        nx = abs(seg.normal.x) if seg.normal else 0.0
        ny = abs(seg.normal.y) if seg.normal else 0.0

        axis = "Y" if nx > ny else "X"
        wall_groups[axis].append(seg)

    for group in wall_groups.values():
        clusters = cluster_by_distance(group, threshold=200.0)
        for cluster in clusters:
            walls.append(merge_wall_cluster(cluster))

    return walls


def reconstruct_walls_between_levels(
    segments: list[GeometrySegment],
    level0: float,
    level1: float,
    cluster_threshold_mm: float = 300.0,
    min_inter_wall_gap_mm: float | None = None,
) -> list[GeometrySegment]:
    """Merge wall fragments for a single storey and force vertical span to that storey."""
    level_walls = [s for s in segments if _is_wall_like(s)]
    if not level_walls:
        return []

    if min_inter_wall_gap_mm is None:
        thicknesses = [
            _wall_thickness(seg, _wall_axis(seg))
            for seg in level_walls
            if _wall_thickness(seg, _wall_axis(seg)) > 1.0
        ]
        if thicknesses:
            median_thickness = float(np.median(thicknesses))
            min_inter_wall_gap_mm = float(np.clip(median_thickness * 0.55, 80.0, 420.0))
        else:
            min_inter_wall_gap_mm = 120.0

    n = len(level_walls)
    parent = list(range(n))

    def find(i: int) -> int:
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    def union(i: int, j: int):
        ri, rj = find(i), find(j)
        if ri != rj:
            parent[rj] = ri

    for i in range(n):
        w1 = level_walls[i]
        ax1 = _wall_axis_vector(w1)
        n1 = np.array([-ax1[1], ax1[0]], dtype=float)
        c1 = np.array([w1.centroid.x, w1.centroid.y], dtype=float)
        span1 = _wall_run_span(w1)

        for j in range(i + 1, n):
            w2 = level_walls[j]
            ax2 = _wall_axis_vector(w2)
            # Angle diff <= 10 deg
            if abs(float(np.dot(ax1, ax2))) < 0.984:
                continue
            # Perpendicular offset <= 120 mm
            c2 = np.array([w2.centroid.x, w2.centroid.y], dtype=float)
            perp_offset = abs(float(np.dot(c2 - c1, n1)))
            if perp_offset > 120.0:
                continue
            # Along-axis run gap <= 1200 mm
            span2 = _wall_run_span(w2)
            if span1[1] < span2[0]:
                gap = span2[0] - span1[1]
            elif span2[1] < span1[0]:
                gap = span1[0] - span2[1]
            else:
                gap = 0.0
            if gap <= 1200.0:
                union(i, j)

    clusters_map: dict[int, list[GeometrySegment]] = defaultdict(list)
    for i in range(n):
        clusters_map[find(i)].append(level_walls[i])

    grouped: list[GeometrySegment] = []
    for cluster in clusters_map.values():
        grouped.append(_merge_wall_cluster_between_levels(cluster, level0, level1))

    if len(grouped) >= 2:
        grouped = _collapse_overlapping_parallel_walls(
            grouped,
            axis="",
            min_inter_wall_gap_mm=min_inter_wall_gap_mm,
        )

    return grouped


def _synthesize_missing_enclosure_walls(
    grouped_walls: list[GeometrySegment],
    source_walls: list[GeometrySegment],
    *,
    level0: float,
    level1: float,
) -> list[GeometrySegment]:
    return grouped_walls

    # Only synthesize from segmentation-driven wall axis metadata; this avoids
    # overbuilding in synthetic/unit-test cases without robust scan orientation.
    tagged_sources = 0
    for seg in source_walls:
        tags = getattr(seg, "tags", {}) or {}
        if "wall_axis_x" in tags and "wall_axis_y" in tags:
            tagged_sources += 1
    # Require majority of walls to be tagged before synthesis
    if tagged_sources < len(source_walls) * 0.60:
        return grouped_walls

    axes = {_wall_axis(seg) for seg in grouped_walls}
    if axes == {"X", "Y"}:
        return grouped_walls

    min_x = min(float(s.bounding_box.min_x) for s in source_walls)
    max_x = max(float(s.bounding_box.max_x) for s in source_walls)
    min_y = min(float(s.bounding_box.min_y) for s in source_walls)
    max_y = max(float(s.bounding_box.max_y) for s in source_walls)
    if min_x >= max_x or min_y >= max_y:
        return grouped_walls
    if (max_x - min_x) < 300.0 or (max_y - min_y) < 300.0:
        return grouped_walls

    thickness_samples: list[float] = []
    for seg in grouped_walls:
        axis = _wall_axis(seg)
        thickness_samples.append(_wall_thickness(seg, axis))
    wall_thickness_mm = float(
        np.clip(np.median(thickness_samples) if thickness_samples else 180.0, 120.0, 280.0)
    )

    synthetic: list[GeometrySegment] = []
    synthetic: list[GeometrySegment] = []
    if axes == {"Y"}:
        inward_push = float(np.clip(wall_thickness_mm * 0.5, 40.0, wall_thickness_mm * 0.8))
        for boundary_y, face in ((min_y, "min"), (max_y, "max")):
            offset = inward_push if face == "min" else -inward_push
            position = float(boundary_y + offset)
            base = grouped_walls[0].model_copy(deep=True)
            base.segment_id = str(uuid4())
            bb = base.bounding_box
            bb.min_x = float(min_x)
            bb.max_x = float(max_x)
            bb.min_y = float(position - wall_thickness_mm / 2.0)
            bb.max_y = float(position + wall_thickness_mm / 2.0)
            bb.min_z = float(level0)
            bb.max_z = float(level1)
            base.centroid.x = float((bb.min_x + bb.max_x) / 2.0)
            base.centroid.y = float((bb.min_y + bb.max_y) / 2.0)
            base.centroid.z = float((bb.min_z + bb.max_z) / 2.0)
            base.tags["reconstructed_wall"] = True
            base.tags["reconstructed_synth_enclosure"] = True
            base.tags["wall_axis"] = "x"
            base.tags["wall_axis_x"] = 1.0
            base.tags["wall_axis_y"] = 0.0
            base.tags["wall_thickness_mm"] = wall_thickness_mm
            base.tags["reconstructed_inward_push_mm"] = inward_push
            if _wall_bbox_has_minimum_dimensions(base):
                synthetic.append(base)
    elif axes == {"X"}:
        inward_push = float(np.clip(wall_thickness_mm * 0.5, 40.0, wall_thickness_mm * 0.8))
        for boundary_x, face in ((min_x, "min"), (max_x, "max")):
            offset = inward_push if face == "min" else -inward_push
            position = float(boundary_x + offset)
            base = grouped_walls[0].model_copy(deep=True)
            base.segment_id = str(uuid4())
            bb = base.bounding_box
            bb.min_x = float(position - wall_thickness_mm / 2.0)
            bb.max_x = float(position + wall_thickness_mm / 2.0)
            bb.min_y = float(min_y)
            bb.max_y = float(max_y)
            bb.min_z = float(level0)
            bb.max_z = float(level1)
            base.centroid.x = float((bb.min_x + bb.max_x) / 2.0)
            base.centroid.y = float((bb.min_y + bb.max_y) / 2.0)
            base.centroid.z = float((bb.min_z + bb.max_z) / 2.0)
            base.tags["reconstructed_wall"] = True
            base.tags["reconstructed_synth_enclosure"] = True
            base.tags["wall_axis"] = "y"
            base.tags["wall_axis_x"] = 0.0
            base.tags["wall_axis_y"] = 1.0
            base.tags["wall_thickness_mm"] = wall_thickness_mm
            base.tags["reconstructed_inward_push_mm"] = inward_push
            if _wall_bbox_has_minimum_dimensions(base):
                synthetic.append(base)

    if not synthetic:
        return grouped_walls
    return grouped_walls + synthetic


def _wall_run_span(seg: GeometrySegment, axis: str = "") -> tuple[float, float]:
    tags = getattr(seg, "tags", {}) or {}
    axis_vec = _wall_axis_vector(seg)
    if "wall_start_x_mm" in tags and "wall_end_x_mm" in tags:
        p0 = np.array([float(tags["wall_start_x_mm"]), float(tags["wall_start_y_mm"])], dtype=float)
        p1 = np.array([float(tags["wall_end_x_mm"]), float(tags["wall_end_y_mm"])], dtype=float)
        s0 = float(np.dot(p0, axis_vec))
        s1 = float(np.dot(p1, axis_vec))
        return min(s0, s1), max(s0, s1)
    bb = seg.bounding_box
    corners = np.array([
        [bb.min_x, bb.min_y],
        [bb.min_x, bb.max_y],
        [bb.max_x, bb.min_y],
        [bb.max_x, bb.max_y],
    ], dtype=float)
    projs = corners @ axis_vec
    return float(projs.min()), float(projs.max())


def _wall_centerline(seg: GeometrySegment, axis: str = "") -> float:
    axis_vec = _wall_axis_vector(seg)
    norm_vec = np.array([-axis_vec[1], axis_vec[0]], dtype=float)
    tags = getattr(seg, "tags", {}) or {}
    if "wall_start_x_mm" in tags and "wall_start_y_mm" in tags:
        p0 = np.array([float(tags["wall_start_x_mm"]), float(tags["wall_start_y_mm"])], dtype=float)
        return float(np.dot(p0, norm_vec))
    bb = seg.bounding_box
    c = np.array([(bb.min_x + bb.max_x) / 2.0, (bb.min_y + bb.max_y) / 2.0], dtype=float)
    return float(np.dot(c, norm_vec))


def _wall_thickness(seg: GeometrySegment, axis: str = "X") -> float:
    tags = getattr(seg, "tags", {}) or {}
    t_tag = _tag_float(tags, "wall_thickness_mm")
    if t_tag is not None and t_tag > 0.0:
        return float(np.clip(t_tag, 100.0, 600.0))
    bb = seg.bounding_box
    if axis == "X":
        return float(np.clip(bb.max_y - bb.min_y, 100.0, 600.0))
    elif axis == "Y":
        return float(np.clip(bb.max_x - bb.min_x, 100.0, 600.0))
    return float(np.clip(min(bb.max_x - bb.min_x, bb.max_y - bb.min_y), 100.0, 600.0))


def _span_overlap_ratio(a: tuple[float, float], b: tuple[float, float]) -> float:
    inter = max(0.0, min(a[1], b[1]) - max(a[0], b[0]))
    if inter <= 0.0:
        return 0.0
    al = max(a[1] - a[0], 1.0)
    bl = max(b[1] - b[0], 1.0)
    return float(inter / min(al, bl))


def _wall_bbox_has_minimum_dimensions(seg: GeometrySegment, min_dim_mm: float = 10.0) -> bool:
    bb = seg.bounding_box
    return (
        float(bb.max_x - bb.min_x) >= min_dim_mm
        and float(bb.max_y - bb.min_y) >= min_dim_mm
        and float(bb.max_z - bb.min_z) >= min_dim_mm
    )


def _ensure_minimum_bbox_dimensions(
    seg: GeometrySegment, min_dim_mm: float = 10.0
) -> GeometrySegment:
    bb = seg.bounding_box
    dx = float(bb.max_x - bb.min_x)
    dy = float(bb.max_y - bb.min_y)
    dz = float(bb.max_z - bb.min_z)

    if dx < min_dim_mm:
        center_x = float((bb.min_x + bb.max_x) / 2.0)
        bb.min_x = float(center_x - min_dim_mm / 2.0)
        bb.max_x = float(center_x + min_dim_mm / 2.0)
    if dy < min_dim_mm:
        center_y = float((bb.min_y + bb.max_y) / 2.0)
        bb.min_y = float(center_y - min_dim_mm / 2.0)
        bb.max_y = float(center_y + min_dim_mm / 2.0)
    if dz < min_dim_mm:
        center_z = float((bb.min_z + bb.max_z) / 2.0)
        bb.min_z = float(center_z - min_dim_mm / 2.0)
        bb.max_z = float(center_z + min_dim_mm / 2.0)

    seg.centroid.x = float((bb.min_x + bb.max_x) / 2.0)
    seg.centroid.y = float((bb.min_y + bb.max_y) / 2.0)
    seg.centroid.z = float((bb.min_z + bb.max_z) / 2.0)
    return seg


def _merge_parallel_walls(
    primary: GeometrySegment, secondary: GeometrySegment, axis: str = ""
) -> GeometrySegment:
    merged = primary.model_copy(deep=True)
    bb = merged.bounding_box
    pbb = primary.bounding_box
    sbb = secondary.bounding_box

    run_min = min(_wall_run_span(primary, axis)[0], _wall_run_span(secondary, axis)[0])
    run_max = max(_wall_run_span(primary, axis)[1], _wall_run_span(secondary, axis)[1])

    p_pts = max(int(primary.point_count), 1)
    s_pts = max(int(secondary.point_count), 1)
    total_pts = p_pts + s_pts
    center = (
        _wall_centerline(primary, axis) * p_pts + _wall_centerline(secondary, axis) * s_pts
    ) / total_pts
    thickness = max(_wall_thickness(primary, axis), _wall_thickness(secondary, axis))

    axis_vec = _wall_axis_vector(primary)
    norm_vec = np.array([-axis_vec[1], axis_vec[0]], dtype=float)
    from agent.tools.geometry_tools import wall_angle_degrees

    angle_deg = wall_angle_degrees(axis_vec)
    mid_run = (run_min + run_max) / 2.0
    length = max(run_max - run_min, 100.0)
    c_xy = axis_vec * mid_run + norm_vec * center

    half_l = length / 2.0
    half_t = thickness / 2.0
    start_pt = c_xy - axis_vec * half_l
    end_pt = c_xy + axis_vec * half_l

    corners = np.array([
        c_xy - axis_vec * half_l - norm_vec * half_t,
        c_xy + axis_vec * half_l - norm_vec * half_t,
        c_xy + axis_vec * half_l + norm_vec * half_t,
        c_xy - axis_vec * half_l + norm_vec * half_t,
    ], dtype=float)

    bb.min_x = float(np.min(corners[:, 0]))
    bb.max_x = float(np.max(corners[:, 0]))
    bb.min_y = float(np.min(corners[:, 1]))
    bb.max_y = float(np.max(corners[:, 1]))

    merged.centroid.x = float(c_xy[0])
    merged.centroid.y = float(c_xy[1])
    merged.tags["wall_start_x_mm"] = round(float(start_pt[0]), 1)
    merged.tags["wall_start_y_mm"] = round(float(start_pt[1]), 1)
    merged.tags["wall_end_x_mm"] = round(float(end_pt[0]), 1)
    merged.tags["wall_end_y_mm"] = round(float(end_pt[1]), 1)
    merged.tags["wall_length_mm"] = round(float(length), 1)
    merged.tags["wall_axis_x"] = round(float(axis_vec[0]), 6)
    merged.tags["wall_axis_y"] = round(float(axis_vec[1]), 6)
    merged.tags["wall_angle_deg"] = round(float(angle_deg), 2)
    merged.tags["wall_thickness_mm"] = round(float(thickness), 1)

    bb.min_z = float(min(pbb.min_z, sbb.min_z))
    bb.max_z = float(max(pbb.max_z, sbb.max_z))
    merged.centroid.x = float((bb.min_x + bb.max_x) / 2.0)
    merged.centroid.y = float((bb.min_y + bb.max_y) / 2.0)
    merged.centroid.z = float((bb.min_z + bb.max_z) / 2.0)
    merged.point_count = int(primary.point_count + secondary.point_count)

    merged.tags["reconstructed_cluster_size"] = int(
        primary.tags.get("reconstructed_cluster_size", 1)
        + secondary.tags.get("reconstructed_cluster_size", 1)
    )
    merged.tags["wall_thickness_mm"] = float(thickness)
    merged.tags["source_min_z_mm"] = float(
        min(
            primary.tags.get("source_min_z_mm", pbb.min_z),
            secondary.tags.get("source_min_z_mm", sbb.min_z),
        )
    )
    merged.tags["source_max_z_mm"] = float(
        max(
            primary.tags.get("source_max_z_mm", pbb.max_z),
            secondary.tags.get("source_max_z_mm", sbb.max_z),
        )
    )
    primary_sources = primary.tags.get("reconstructed_from_segment_ids") or [primary.segment_id]
    secondary_sources = secondary.tags.get("reconstructed_from_segment_ids") or [
        secondary.segment_id
    ]
    merged.tags["reconstructed_from_segment_ids"] = sorted(
        {str(sid) for sid in [*primary_sources, *secondary_sources] if sid}
    )
    merged.tags["reconstructed_overlap_merged"] = True
    _ensure_minimum_bbox_dimensions(merged)
    return merged


def _collapse_overlapping_parallel_walls(
    walls: list[GeometrySegment],
    axis: str,
    min_inter_wall_gap_mm: float,
) -> list[GeometrySegment]:
    if len(walls) < 2:
        return walls

    ordered = sorted(walls, key=lambda seg: _wall_centerline(seg, axis))
    collapsed: list[GeometrySegment] = []

    for wall in ordered:
        if not collapsed:
            collapsed.append(wall)
            continue

        prev = collapsed[-1]
        overlap = _span_overlap_ratio(_wall_run_span(prev, axis), _wall_run_span(wall, axis))
        # Require stronger run overlap evidence before treating walls as candidates
        # for collapse. This helps prevent flattening distinct parallel walls into
        # a single thin plane when scans only partially cover surfaces.
        if overlap < 0.70:
            collapsed.append(wall)
            continue

        prev_src_min_z = float(prev.tags.get("source_min_z_mm", prev.bounding_box.min_z))
        prev_src_max_z = float(prev.tags.get("source_max_z_mm", prev.bounding_box.max_z))
        curr_src_min_z = float(wall.tags.get("source_min_z_mm", wall.bounding_box.min_z))
        curr_src_max_z = float(wall.tags.get("source_max_z_mm", wall.bounding_box.max_z))
        z_inter = max(
            0.0, min(prev_src_max_z, curr_src_max_z) - max(prev_src_min_z, curr_src_min_z)
        )
        z_min_span = max(1.0, min(prev_src_max_z - prev_src_min_z, curr_src_max_z - curr_src_min_z))
        if (z_inter / z_min_span) < 0.30:
            collapsed.append(wall)
            continue

        c_prev = _wall_centerline(prev, axis)
        c_curr = _wall_centerline(wall, axis)
        t_prev = _wall_thickness(prev, axis)
        t_curr = _wall_thickness(wall, axis)
        face_gap = abs(c_curr - c_prev) - (t_prev + t_curr) / 2.0

        # Prefer to bound walls (merge) when they are very close. Only merge
        # when the face gap is tiny relative to the expected inter-wall gap to
        # avoid collapsing real cavities into single walls.
        if face_gap >= min_inter_wall_gap_mm * 0.5:
            collapsed.append(wall)
            continue

        # Preserve cavity depth between parallel walls when they are clearly distinct
        # but too tight; separate them instead of collapsing into one wall.
        # This threshold is tuned to preserve thin wall separations seen in real BIM.
        # Preserve cavity depth between parallel walls when they are clearly distinct
        # but still tight; allow a small separation to be preserved rather than
        # merging into a single nominal wall.
        if face_gap >= min_inter_wall_gap_mm * 0.15:
            shifted = wall.model_copy(deep=True)
            shift = max(min_inter_wall_gap_mm * 0.1, face_gap * 0.5)
            direction = 1.0 if c_curr >= c_prev else -1.0
            axis_vec = _wall_axis_vector(wall)
            norm_vec = np.array([-axis_vec[1], axis_vec[0]], dtype=float)
            shift_vec = direction * shift * norm_vec
            bb = shifted.bounding_box
            bb.min_x = float(bb.min_x + shift_vec[0])
            bb.max_x = float(bb.max_x + shift_vec[0])
            bb.min_y = float(bb.min_y + shift_vec[1])
            bb.max_y = float(bb.max_y + shift_vec[1])
            shifted.centroid.x = float(shifted.centroid.x + shift_vec[0])
            shifted.centroid.y = float(shifted.centroid.y + shift_vec[1])
            if "wall_start_x_mm" in shifted.tags:
                shifted.tags["wall_start_x_mm"] = round(shifted.tags["wall_start_x_mm"] + float(shift_vec[0]), 1)
                shifted.tags["wall_start_y_mm"] = round(shifted.tags["wall_start_y_mm"] + float(shift_vec[1]), 1)
                shifted.tags["wall_end_x_mm"] = round(shifted.tags["wall_end_x_mm"] + float(shift_vec[0]), 1)
                shifted.tags["wall_end_y_mm"] = round(shifted.tags["wall_end_y_mm"] + float(shift_vec[1]), 1)
            shifted.tags["reconstructed_depth_separated"] = True
            collapsed.append(shifted)
            continue

        # Merge near-duplicate or too-tight parallel walls to avoid Revit overlap
        # and achieve proper bounding without gaps.
        if prev.point_count >= wall.point_count:
            collapsed[-1] = _merge_parallel_walls(prev, wall, axis)
        else:
            collapsed[-1] = _merge_parallel_walls(wall, prev, axis)

    return collapsed


def _merge_wall_cluster_between_levels(
    cluster: list[GeometrySegment],
    level0: float,
    level1: float,
    axis: str = "",
) -> GeometrySegment:
    primary = max(cluster, key=lambda s: int(getattr(s, "point_count", 0)))
    axis_vec = _wall_axis_vector(primary)
    norm_vec = np.array([-axis_vec[1], axis_vec[0]], dtype=float)
    from agent.tools.geometry_tools import wall_angle_degrees

    angle_deg = wall_angle_degrees(axis_vec)

    # Thickness inference
    thickness_samples: list[float] = []
    for seg in cluster:
        tag_t = _tag_float(seg.tags, "wall_thickness_mm")
        if tag_t is not None and 80.0 <= tag_t <= 800.0:
            thickness_samples.append(tag_t)
        else:
            dx, dy, _ = _segment_dims(seg)
            sampled_t = min(dx, dy)
            if 80.0 <= sampled_t <= 800.0:
                thickness_samples.append(sampled_t)

    inferred_thickness = float(np.median(thickness_samples)) if thickness_samples else 200.0
    wall_thickness_mm = float(np.clip(inferred_thickness, 100.0, 600.0))

    # Along-axis run and perpendicular center
    all_runs: list[tuple[float, float]] = []
    center_samples: list[float] = []

    for seg in cluster:
        r0, r1 = _wall_run_span(seg, axis)
        all_runs.append((r0, r1))
        center_samples.append(_wall_centerline(seg, axis))

    run_min = min(r[0] for r in all_runs)
    run_max = max(r[1] for r in all_runs)
    length_mm = max(run_max - run_min, 100.0)
    center_d = float(np.median(center_samples)) if center_samples else 0.0

    mid_run = (run_min + run_max) / 2.0
    c_xy = axis_vec * mid_run + norm_vec * center_d

    half_l = length_mm / 2.0
    half_t = wall_thickness_mm / 2.0
    start_pt = c_xy - axis_vec * half_l
    end_pt = c_xy + axis_vec * half_l

    corners = np.array([
        c_xy - axis_vec * half_l - norm_vec * half_t,
        c_xy + axis_vec * half_l - norm_vec * half_t,
        c_xy + axis_vec * half_l + norm_vec * half_t,
        c_xy - axis_vec * half_l + norm_vec * half_t,
    ], dtype=float)

    base = primary.model_copy(deep=True)
    base.segment_id = str(uuid4())
    bb = base.bounding_box
    bb.min_x = float(np.min(corners[:, 0]))
    bb.max_x = float(np.max(corners[:, 0]))
    bb.min_y = float(np.min(corners[:, 1]))
    bb.max_y = float(np.max(corners[:, 1]))
    bb.min_z = float(level0)
    bb.max_z = float(level1)

    base.centroid.x = float(c_xy[0])
    base.centroid.y = float(c_xy[1])
    base.centroid.z = float((level0 + level1) / 2.0)
    base.point_count = sum(s.point_count for s in cluster)

    tags = base.tags
    tags["reconstructed_wall"] = True
    tags["reconstructed_cluster_size"] = len(cluster)
    tags["reconstructed_storey_base_mm"] = float(level0)
    tags["reconstructed_storey_top_mm"] = float(level1)
    tags["wall_axis"] = _wall_axis(primary).lower()
    tags["wall_axis_x"] = round(float(axis_vec[0]), 6)
    tags["wall_axis_y"] = round(float(axis_vec[1]), 6)
    tags["wall_angle_deg"] = round(float(angle_deg), 2)
    tags["wall_start_x_mm"] = round(float(start_pt[0]), 1)
    tags["wall_start_y_mm"] = round(float(start_pt[1]), 1)
    tags["wall_end_x_mm"] = round(float(end_pt[0]), 1)
    tags["wall_end_y_mm"] = round(float(end_pt[1]), 1)
    tags["wall_length_mm"] = round(float(length_mm), 1)
    tags["wall_thickness_mm"] = round(float(wall_thickness_mm), 1)
    tags["source_min_z_mm"] = float(min(s.bounding_box.min_z for s in cluster))
    tags["source_max_z_mm"] = float(max(s.bounding_box.max_z for s in cluster))
    tags["reconstructed_from_segment_ids"] = [
        str(seg.segment_id) for seg in cluster if getattr(seg, "segment_id", None)
    ]
    _ensure_minimum_bbox_dimensions(base)
    return base


def cluster_by_distance(
    segments: list[GeometrySegment],
    threshold: float,
) -> list[list[GeometrySegment]]:
    clusters: list[list[GeometrySegment]] = []
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
            if distance(a, b) < threshold:
                group.append(b)
                used.add(j)

        clusters.append(group)

    return clusters


def merge_wall_cluster(cluster: list[GeometrySegment]) -> GeometrySegment:
    min_x = min(s.bounding_box.min_x for s in cluster)
    max_x = max(s.bounding_box.max_x for s in cluster)
    min_y = min(s.bounding_box.min_y for s in cluster)
    max_y = max(s.bounding_box.max_y for s in cluster)
    min_z = min(s.bounding_box.min_z for s in cluster)
    max_z = max(s.bounding_box.max_z for s in cluster)

    base = cluster[0].model_copy()

    base.bounding_box.min_x = min_x
    base.bounding_box.max_x = max_x
    base.bounding_box.min_y = min_y
    base.bounding_box.max_y = max_y
    base.bounding_box.min_z = min_z
    base.bounding_box.max_z = max_z

    base.point_count = sum(s.point_count for s in cluster)
    base.tags["reconstructed_wall"] = True
    base.tags["reconstructed_cluster_size"] = len(cluster)

    return base


def _merge_floor(cluster: list[GeometrySegment]) -> GeometrySegment:
    min_x = min(s.bounding_box.min_x for s in cluster)
    max_x = max(s.bounding_box.max_x for s in cluster)
    min_y = min(s.bounding_box.min_y for s in cluster)
    max_y = max(s.bounding_box.max_y for s in cluster)
    min_z = min(s.bounding_box.min_z for s in cluster)
    max_z = max(s.bounding_box.max_z for s in cluster)

    base = cluster[0].model_copy(deep=True)
    bb = base.bounding_box
    bb.min_x = float(min_x)
    bb.max_x = float(max_x)
    bb.min_y = float(min_y)
    bb.max_y = float(max_y)
    bb.min_z = float(min_z)
    bb.max_z = float(max_z)

    base.centroid.x = float((bb.min_x + bb.max_x) / 2.0)
    base.centroid.y = float((bb.min_y + bb.max_y) / 2.0)
    base.centroid.z = float((bb.min_z + bb.max_z) / 2.0)
    base.point_count = sum(s.point_count for s in cluster)
    base.tags["reconstructed_floor"] = True
    base.tags["reconstructed_cluster_size"] = len(cluster)
    base.tags["reconstructed_from_segment_ids"] = [
        str(seg.segment_id) for seg in cluster if getattr(seg, "segment_id", None)
    ]

    return base


def merge_floor(cluster: list[GeometrySegment]) -> GeometrySegment:
    return _merge_floor(cluster)


# -----------------------------------------------------------------------------
# 2. FLOOR CONSOLIDATION
# -----------------------------------------------------------------------------


def reconstruct_floors(segments: list[GeometrySegment]) -> list[GeometrySegment]:
    floors: list[GeometrySegment] = []
    levels: dict[int, list[GeometrySegment]] = defaultdict(list)

    for seg in segments:
        if not _is_type(seg, ElementType.FLOOR):
            continue

        z = (seg.bounding_box.min_z + seg.bounding_box.max_z) / 2.0
        key = round(z / 500.0)
        levels[key].append(seg)

    for group in levels.values():
        floors.append(_merge_floor(group))

    return floors


def reconstruct_floors_at_level(
    segments: list[GeometrySegment],
    level0: float,
    thickness_mm: float = 150.0,
) -> list[GeometrySegment]:
    from scipy.spatial import ConvexHull

    floor_segments = [s for s in segments if _is_type(s, ElementType.FLOOR)]
    if not floor_segments:
        return []

    pts_2d_list: list[np.ndarray] = []
    for s in floor_segments:
        raw_pts = getattr(s, "points", None)
        if raw_pts is not None and len(raw_pts) > 0:
            arr = np.asarray(raw_pts, dtype=float)
            if arr.ndim == 2 and arr.shape[1] >= 2:
                pts_2d_list.append(arr[:, :2])
        bb = s.bounding_box
        corners = np.array([
            [bb.min_x, bb.min_y],
            [bb.max_x, bb.min_y],
            [bb.max_x, bb.max_y],
            [bb.min_x, bb.max_y],
        ], dtype=float)
        pts_2d_list.append(corners)

    all_pts_2d = np.concatenate(pts_2d_list, axis=0)

    if len(all_pts_2d) >= 3:
        try:
            hull = ConvexHull(all_pts_2d)
            hull_vertices = all_pts_2d[hull.vertices]
            polygon = [[round(float(pt[0]), 1), round(float(pt[1]), 1)] for pt in hull_vertices]
            if polygon and polygon[0] != polygon[-1]:
                polygon.append(polygon[0])
        except Exception:
            min_x = float(all_pts_2d[:, 0].min())
            max_x = float(all_pts_2d[:, 0].max())
            min_y = float(all_pts_2d[:, 1].min())
            max_y = float(all_pts_2d[:, 1].max())
            polygon = [
                [round(min_x, 1), round(min_y, 1)],
                [round(max_x, 1), round(min_y, 1)],
                [round(max_x, 1), round(max_y, 1)],
                [round(min_x, 1), round(max_y, 1)],
                [round(min_x, 1), round(min_y, 1)],
            ]
    else:
        min_x = min(f.bounding_box.min_x for f in floor_segments)
        max_x = max(f.bounding_box.max_x for f in floor_segments)
        min_y = min(f.bounding_box.min_y for f in floor_segments)
        max_y = max(f.bounding_box.max_y for f in floor_segments)
        polygon = [
            [round(min_x, 1), round(min_y, 1)],
            [round(max_x, 1), round(min_y, 1)],
            [round(max_x, 1), round(max_y, 1)],
            [round(min_x, 1), round(max_y, 1)],
            [round(min_x, 1), round(min_y, 1)],
        ]

    poly_arr = np.array(polygon[:-1], dtype=float)
    if len(poly_arr) >= 2:
        c_poly = np.mean(poly_arr, axis=0)
        centered = poly_arr - c_poly
        cov = centered.T @ centered
        eigvals, eigvecs = np.linalg.eigh(cov)
        dominant_axis = eigvecs[:, 1]
    else:
        dominant_axis = np.array([1.0, 0.0], dtype=float)

    if dominant_axis[0] < 0:
        dominant_axis = -dominant_axis
    axis_x, axis_y = float(dominant_axis[0]), float(dominant_axis[1])
    norm_x, norm_y = -axis_y, axis_x

    u_projs = poly_arr @ np.array([axis_x, axis_y])
    v_projs = poly_arr @ np.array([norm_x, norm_y])

    u_min, u_max = float(u_projs.min()), float(u_projs.max())
    v_min, v_max = float(v_projs.min()), float(v_projs.max())

    floor_len = max(u_max - u_min, 100.0)
    floor_wid = max(v_max - v_min, 100.0)
    center_u = (u_min + u_max) / 2.0
    center_v = (v_min + v_max) / 2.0

    c_floor_xy = np.array([axis_x, axis_y]) * center_u + np.array([norm_x, norm_y]) * center_v

    min_x = float(all_pts_2d[:, 0].min())
    max_x = float(all_pts_2d[:, 0].max())
    min_y = float(all_pts_2d[:, 1].min())
    max_y = float(all_pts_2d[:, 1].max())

    base = floor_segments[0].model_copy(deep=True)
    bb = base.bounding_box
    bb.min_x = float(min_x)
    bb.max_x = float(max_x)
    bb.min_y = float(min_y)
    bb.max_y = float(max_y)
    bb.min_z = float(level0)
    bb.max_z = float(level0 + thickness_mm)

    base.centroid.x = float(c_floor_xy[0])
    base.centroid.y = float(c_floor_xy[1])
    base.centroid.z = float((level0 + level0 + thickness_mm) / 2.0)
    base.tags["reconstructed_floor"] = True
    base.tags["reconstructed_floor_polygon"] = True
    base.tags["reconstructed_storey_base_mm"] = float(level0)
    base.tags["boundary_polygon"] = polygon
    base.tags["profile_points"] = polygon
    base.tags["floor_axis_x"] = round(axis_x, 6)
    base.tags["floor_axis_y"] = round(axis_y, 6)
    base.tags["floor_length_mm"] = round(floor_len, 1)
    base.tags["floor_width_mm"] = round(floor_wid, 1)
    base.tags["floor_center_x_mm"] = round(float(c_floor_xy[0]), 1)
    base.tags["floor_center_y_mm"] = round(float(c_floor_xy[1]), 1)
    base.tags["reconstructed_from_segment_ids"] = [
        str(seg.segment_id) for seg in floor_segments if getattr(seg, "segment_id", None)
    ]
    return [base]


def reconstruct_ceilings_at_level(
    floor_segments: list[GeometrySegment],
    ceiling_segments: list[GeometrySegment],
    level_top: float,
    thickness_mm: float = 120.0,
) -> list[GeometrySegment]:
    if not ceiling_segments:
        return []
    source = ceiling_segments

    min_x = min(seg.bounding_box.min_x for seg in source)
    max_x = max(seg.bounding_box.max_x for seg in source)
    min_y = min(seg.bounding_box.min_y for seg in source)
    max_y = max(seg.bounding_box.max_y for seg in source)

    base = source[0].model_copy(deep=True)
    bb = base.bounding_box
    bb.min_x = float(min_x)
    bb.max_x = float(max_x)
    bb.min_y = float(min_y)
    bb.max_y = float(max_y)
    bb.max_z = float(level_top)
    bb.min_z = float(level_top - thickness_mm)

    base.centroid.x = float((bb.min_x + bb.max_x) / 2.0)
    base.centroid.y = float((bb.min_y + bb.max_y) / 2.0)
    base.centroid.z = float((bb.min_z + bb.max_z) / 2.0)
    base.element_type = ElementType.CEILING
    base.tags["reconstructed_ceiling"] = True
    base.tags["reconstructed_storey_top_mm"] = float(level_top)
    base.tags["reconstructed_ceiling_inferred"] = bool(
        not ceiling_segments and bool(floor_segments)
    )
    base.tags["reconstructed_from_segment_ids"] = [
        str(seg.segment_id) for seg in source if getattr(seg, "segment_id", None)
    ]
    return [base]


def reconstruct_stairs_between_levels(
    segments: list[GeometrySegment],
    level0: float,
    level1: float,
) -> list[GeometrySegment]:
    """Reconstruct stairs/ramps to span between storey levels.

    Fixes: avoid duplicate/staked stairs by deduplicating groups and using
    consistent XY extents from the most confident representative group.
    """
    stair_segments = [s for s in segments if _is_stair_like(s)]
    if not stair_segments:
        return []

    reconstructed: list[GeometrySegment] = []

    # Group stairs by horizontal proximity and approximate orientation.
    stair_groups: dict[tuple, list[GeometrySegment]] = defaultdict(list)

    for seg in stair_segments:
        bb = seg.bounding_box
        cx = (bb.min_x + bb.max_x) * 0.5
        cy = (bb.min_y + bb.max_y) * 0.5

        # Quantize to 300mm grid for tighter grouping.
        grid_x = round(cx / 300.0) * 300.0
        grid_y = round(cy / 300.0) * 300.0

        dx = float(bb.max_x - bb.min_x)
        dy = float(bb.max_y - bb.min_y)
        orient = "X" if dx >= dy else "Y"

        stair_groups[(grid_x, grid_y, orient)].append(seg)

    for group in stair_groups.values():
        if not group:
            continue

        # Deduplicate: keep the most confident representative for XY anchoring.
        rep = max(group, key=lambda s: float(getattr(s, "confidence", 0.0)))

        # Extents come from the whole group to preserve stair footprint size.
        min_x = min(float(s.bounding_box.min_x) for s in group)
        max_x = max(float(s.bounding_box.max_x) for s in group)
        min_y = min(float(s.bounding_box.min_y) for s in group)
        max_y = max(float(s.bounding_box.max_y) for s in group)

        center_x = float((min_x + max_x) / 2.0)
        center_y = float((min_y + max_y) / 2.0)

        base = rep.model_copy(deep=True)
        bb = base.bounding_box

        half_x = (max_x - min_x) / 2.0
        half_y = (max_y - min_y) / 2.0

        bb.min_x = center_x - half_x
        bb.max_x = center_x + half_x
        bb.min_y = center_y - half_y
        bb.max_y = center_y + half_y
        bb.min_z = float(level0)
        bb.max_z = float(level1)

        base.centroid.x = float(center_x)
        base.centroid.y = float(center_y)
        base.centroid.z = float((level0 + level1) / 2.0)
        base.point_count = sum(s.point_count for s in group)

        base.tags["reconstructed_stair"] = True
        base.tags["reconstructed_storey_base_mm"] = float(level0)
        base.tags["reconstructed_storey_top_mm"] = float(level1)
        base.tags["base_level_mm"] = float(level0)
        base.tags["top_level_mm"] = float(level1)
        base.tags["rise_mm"] = float(level1 - level0)
        base.tags["reconstructed_from_segment_ids"] = [
            str(seg.segment_id) for seg in group if getattr(seg, "segment_id", None)
        ]
        base.tags["reconstructed_stair_span_levels"] = True

        reconstructed.append(base)

    return reconstructed


# -----------------------------------------------------------------------------
# 3. OPENING (DOOR, WINDOW, VOID) HOSTING
# -----------------------------------------------------------------------------


def attach_doors_to_walls(
    walls: list[GeometrySegment],
    doors: list[GeometrySegment],
) -> list[GeometrySegment]:
    for door in doors:
        for wall in walls:
            if is_inside(door, wall):
                wall.tags.setdefault("voids", [])
                wall.tags["voids"].append(door.segment_id)

    return walls


def attach_openings(
    walls: list[ReconstructedNode], segments: list[GeometrySegment]
) -> list[ReconstructedNode]:
    """Host doors, windows, and opening voids on containing walls.

    Matches each opening's 3D position against wall polylines, validates that the
    opening centroid lies within the wall's thickness envelope and length span,
    computes wall-local insertion coordinates (insertion_u_mm, insertion_v_mm,
    sill height), and establishes host-wall references on both opening and wall.
    """
    from agent.tools.geometry_tools import wall_angle_degrees

    opening_types = {
        ElementType.DOOR,
        ElementType.WINDOW,
        ElementType.SLAB_OPENING,
    }
    openings: list[GeometrySegment] = []
    for s in segments:
        et_val = getattr(s.element_type, "value", s.element_type)
        et_str = str(et_val or "").lower()
        shape_str = str(getattr(s, "shape", "") or "").lower()
        if (
            s.element_type in opening_types
            or any(k in et_str for k in ("door", "window", "opening", "void"))
            or shape_str in ("void", "opening")
        ):
            openings.append(s)

    nodes: list[ReconstructedNode] = []

    for op in openings:
        et = op.element_type if isinstance(op.element_type, ElementType) else ElementType.DOOR
        et_str = str(getattr(et, "value", et)).lower()
        if "window" in et_str:
            target_type = ElementType.WINDOW
        elif "slab_opening" in et_str:
            target_type = ElementType.SLAB_OPENING
        else:
            target_type = ElementType.DOOR

        op_c = np.array([op.centroid.x, op.centroid.y, op.centroid.z], dtype=float)
        ob = op.bounding_box

        best_wall: ReconstructedNode | None = None
        best_perp_dist = float("inf")
        best_u = 0.0
        best_v = 0.0
        best_axis = np.array([1.0, 0.0], dtype=float)

        for w_node in walls:
            w = w_node.segment
            wb = w.bounding_box

            # Z overlap check with 200mm margin
            z_overlap = min(float(ob.max_z), float(wb.max_z)) - max(float(ob.min_z), float(wb.min_z))
            if z_overlap < -200.0:
                continue

            axis_vec = _wall_axis_vector(w)
            norm_vec = np.array([-axis_vec[1], axis_vec[0]], dtype=float)

            w_tags = getattr(w, "tags", {}) or {}
            if "wall_start_x_mm" in w_tags and "wall_start_y_mm" in w_tags:
                start_xy = np.array([float(w_tags["wall_start_x_mm"]), float(w_tags["wall_start_y_mm"])], dtype=float)
            else:
                center_xy = np.array([w.centroid.x, w.centroid.y], dtype=float)
                w_len = float(w_tags.get("wall_length_mm", max(wb.max_x - wb.min_x, wb.max_y - wb.min_y)))
                start_xy = center_xy - axis_vec * (w_len / 2.0)

            wall_len = float(w_tags.get("wall_length_mm", max(wb.max_x - wb.min_x, wb.max_y - wb.min_y)))
            wall_thick = float(w_tags.get("wall_thickness_mm", 200.0))

            delta = op_c[:2] - start_xy
            u_dist = float(np.dot(delta, axis_vec))

            center_xy = np.array([w.centroid.x, w.centroid.y], dtype=float)
            v_dist = float(np.dot(op_c[:2] - center_xy, norm_vec))

            # Validate opening lies within wall's thickness envelope and length span
            if -200.0 <= u_dist <= (wall_len + 200.0):
                if abs(v_dist) <= (wall_thick / 2.0 + 200.0):
                    if abs(v_dist) < best_perp_dist:
                        best_perp_dist = abs(v_dist)
                        best_wall = w_node
                        best_u = u_dist
                        best_v = v_dist
                        best_axis = axis_vec

        enriched_op = op.model_copy(deep=True)
        enriched_op.element_type = target_type

        node = ReconstructedNode(
            id=enriched_op.segment_id,
            segment=enriched_op,
            type=target_type,
            host=best_wall.id if best_wall else None,
            children=[],
            connections=[],
            parametric={
                "width": float(max(ob.max_x - ob.min_x, ob.max_y - ob.min_y)),
                "height": float(ob.max_z - ob.min_z),
                "insertion_u_mm": round(best_u, 1),
                "insertion_v_mm": round(best_v, 1),
            },
            confidence=float(enriched_op.confidence),
        )

        if best_wall is not None:
            if node.id not in best_wall.children:
                best_wall.children.append(node.id)
            enriched_op.tags["host_wall_id"] = str(best_wall.id)
            enriched_op.tags["insertion_u_mm"] = round(best_u, 1)
            enriched_op.tags["insertion_v_mm"] = round(best_v, 1)
            enriched_op.tags["wall_axis_x"] = round(float(best_axis[0]), 6)
            enriched_op.tags["wall_axis_y"] = round(float(best_axis[1]), 6)
            enriched_op.tags["wall_angle_deg"] = round(float(wall_angle_degrees(best_axis)), 2)
            enriched_op.tags["sill_height_mm"] = round(float(max(0.0, ob.min_z - best_wall.segment.bounding_box.min_z)), 1)
            enriched_op.tags["opening_height_mm"] = round(float(ob.max_z - ob.min_z), 1)
            enriched_op.tags["opening_width_mm"] = round(float(max(ob.max_x - ob.min_x, ob.max_y - ob.min_y)), 1)

            w_seg = best_wall.segment
            w_seg.tags.setdefault("hosted_opening_ids", [])
            if str(node.id) not in w_seg.tags["hosted_opening_ids"]:
                w_seg.tags["hosted_opening_ids"].append(str(node.id))
            w_seg.tags.setdefault("voids", [])
            if str(node.id) not in w_seg.tags["voids"]:
                w_seg.tags["voids"].append(str(node.id))

        nodes.append(node)

    return nodes


def attach_doors(
    walls: list[ReconstructedNode], segments: list[GeometrySegment]
) -> list[ReconstructedNode]:
    """Compatibility wrapper delegating to attach_openings."""
    return attach_openings(walls, segments)


def attach_detail_nodes(
    walls: list[ReconstructedNode],
    floors: list[ReconstructedNode],
    segments: list[GeometrySegment],
) -> list[ReconstructedNode]:
    detail_nodes: list[ReconstructedNode] = []
    skipped = {
        ElementType.WALL,
        ElementType.FLOOR,
        ElementType.CEILING,
        ElementType.DOOR,
        ElementType.WINDOW,
        ElementType.SLAB_OPENING,
        ElementType.STAIR,
    }
    linear_types = {
        ElementType.PIPE,
        ElementType.CONDUIT,
        ElementType.DUCT,
        ElementType.CABLE_TRAY,
        ElementType.DRAINAGE,
    }
    terminal_types = {ElementType.ELECTRICAL_PANEL, ElementType.JUNCTION_BOX}

    for seg in segments:
        et = seg.element_type
        if not isinstance(et, ElementType) or et in skipped:
            continue

        enriched = seg.model_copy(deep=True)
        node = ReconstructedNode(
            id=enriched.segment_id,
            segment=enriched,
            type=et,
            host=None,
            children=[],
            connections=[],
            parametric=_detail_parametric(enriched, et),
            confidence=float(enriched.confidence),
        )

        if et in {ElementType.WINDOW, *terminal_types}:
            for wall in walls:
                if _is_component_embedded_in_wall(enriched, wall.segment):
                    node.host = wall.id
                    if node.id not in wall.children:
                        wall.children.append(node.id)
                    break

        if et == ElementType.CABLE_TRAY:
            for wall in walls:
                if _is_linear_component_adjacent_to_wall(enriched, wall.segment):
                    node.host = wall.id
                    if node.id not in wall.children:
                        wall.children.append(node.id)
                    break

        if et == ElementType.SLAB_OPENING:
            for floor in floors:
                if is_inside(enriched, floor.segment, tol_mm=200.0):
                    node.host = floor.id
                    if node.id not in floor.children:
                        floor.children.append(node.id)
                    break

        detail_nodes.append(node)

    linear_nodes = [node for node in detail_nodes if node.type in linear_types]
    terminal_nodes = [node for node in detail_nodes if node.type in terminal_types]

    for idx, src in enumerate(linear_nodes):
        for dst in linear_nodes[idx + 1 :]:
            if distance(src.segment, dst.segment) > 1200.0:
                continue
            if dst.id not in src.connections:
                src.connections.append(dst.id)
            if src.id not in dst.connections:
                dst.connections.append(src.id)

    for terminal in terminal_nodes:
        nearest: ReconstructedNode | None = None
        nearest_dist = float("inf")
        for carrier in linear_nodes:
            d = distance(terminal.segment, carrier.segment)
            if d < nearest_dist and d <= 1500.0:
                nearest = carrier
                nearest_dist = d
        if nearest is None:
            continue
        if nearest.id not in terminal.connections:
            terminal.connections.append(nearest.id)
        if terminal.id not in nearest.connections:
            nearest.connections.append(terminal.id)

    return detail_nodes


def run_auto_bim_pipeline(segments: list[GeometrySegment]) -> dict[str, ReconstructedNode]:
    """Auto-detect levels and generate level-spanning walls + hosted doors + stairs."""
    levels = detect_levels(segments)
    spans = _level_span_pairs(levels)

    wall_segments = [s for s in segments if _is_wall_like(s)]
    floor_segments = [s for s in segments if _is_type(s, ElementType.FLOOR)]
    ceiling_segments = [s for s in segments if _is_type(s, ElementType.CEILING)]
    stair_segments = [s for s in segments if _is_stair_like(s)]

    walls: list[ReconstructedNode] = []
    floors: list[ReconstructedNode] = []
    ceilings: list[ReconstructedNode] = []
    stairs: list[ReconstructedNode] = []

    for idx, (z0, z1) in enumerate(spans):
        storey_h = max(float(z1 - z0), 100.0)
        min_overlap = min(400.0, storey_h * 0.20)
        storey_walls: list[GeometrySegment] = []
        for s in wall_segments:
            sz_min = float(s.tags.get("source_min_z_mm", s.bounding_box.min_z))
            sz_max = float(s.tags.get("source_max_z_mm", s.bounding_box.max_z))
            overlap = min(sz_max, z1) - max(sz_min, z0)
            if overlap >= min_overlap:
                s_copy = s.model_copy(deep=True)
                s_copy.bounding_box.min_z = float(z0)
                s_copy.bounding_box.max_z = float(z1)
                storey_walls.append(s_copy)
        for seg in reconstruct_walls_between_levels(storey_walls, z0, z1):
            bb = seg.bounding_box
            walls.append(
                ReconstructedNode(
                    id=seg.segment_id,
                    segment=seg,
                    type=ElementType.WALL,
                    host=None,
                    children=[],
                    connections=[],
                    parametric={
                        "height": float(z1 - z0),
                        "length": float(max(bb.max_x - bb.min_x, bb.max_y - bb.min_y)),
                        "thickness": float(min(bb.max_x - bb.min_x, bb.max_y - bb.min_y)),
                    },
                    confidence=float(seg.confidence),
                )
            )

        # Floors are reconstructed from storey-local candidates when available, or from building footprint
        storey_floors = [s for s in floor_segments if _segment_storey_index(s, levels) == idx]
        if not storey_floors and floor_segments:
            storey_floors = floor_segments
        for seg in reconstruct_floors_at_level(storey_floors, z0):
            floors.append(
                ReconstructedNode(
                    id=seg.segment_id,
                    segment=seg,
                    type=ElementType.FLOOR,
                    host=None,
                    children=[],
                    connections=[],
                    parametric={
                        "thickness": float(seg.bounding_box.max_z - seg.bounding_box.min_z),
                        "length": float(seg.bounding_box.max_x - seg.bounding_box.min_x),
                        "width": float(seg.bounding_box.max_y - seg.bounding_box.min_y),
                    },
                    confidence=float(seg.confidence),
                )
            )

        storey_ceilings = [s for s in ceiling_segments if _segment_storey_index(s, levels) == idx]
        for seg in reconstruct_ceilings_at_level(storey_floors, storey_ceilings, z1):
            ceilings.append(
                ReconstructedNode(
                    id=seg.segment_id,
                    segment=seg,
                    type=ElementType.CEILING,
                    host=None,
                    children=[],
                    connections=[],
                    parametric={
                        "thickness": float(seg.bounding_box.max_z - seg.bounding_box.min_z),
                        "length": float(seg.bounding_box.max_x - seg.bounding_box.min_x),
                        "width": float(seg.bounding_box.max_y - seg.bounding_box.min_y),
                    },
                    confidence=float(seg.confidence),
                )
            )

    # Stairs span between levels, not contained within a single storey
    if stair_segments:
        if len(spans) >= 2:
            stair_levels: dict[int, list[GeometrySegment]] = defaultdict(list)
            for seg in stair_segments:
                storey_idx = _segment_storey_index(seg, levels)
                if storey_idx >= len(spans) - 1:
                    storey_idx = len(spans) - 2
                if storey_idx >= 0:
                    stair_levels[storey_idx].append(seg)

            for storey_idx, stair_group in stair_levels.items():
                z0, z1 = spans[storey_idx]
                for stair_seg in reconstruct_stairs_between_levels(stair_group, z0, z1):
                    bb = stair_seg.bounding_box
                    stairs.append(
                        ReconstructedNode(
                            id=stair_seg.segment_id,
                            segment=stair_seg,
                            type=ElementType.STAIR,
                            host=None,
                            children=[],
                            connections=[],
                            parametric={
                                "rise": float(z1 - z0),
                                "run_length": float(max(bb.max_x - bb.min_x, bb.max_y - bb.min_y)),
                                "run_width": float(min(bb.max_x - bb.min_x, bb.max_y - bb.min_y)),
                                "base_level_mm": float(z0),
                                "top_level_mm": float(z1),
                            },
                            confidence=float(stair_seg.confidence),
                        )
                    )
        elif spans:
            z0, z1 = spans[0]
            for stair_seg in reconstruct_stairs_between_levels(stair_segments, z0, z1):
                bb = stair_seg.bounding_box
                stairs.append(
                    ReconstructedNode(
                        id=stair_seg.segment_id,
                        segment=stair_seg,
                        type=ElementType.STAIR,
                        host=None,
                        children=[],
                        connections=[],
                        parametric={
                            "rise": float(z1 - z0),
                            "run_length": float(max(bb.max_x - bb.min_x, bb.max_y - bb.min_y)),
                            "run_width": float(min(bb.max_x - bb.min_x, bb.max_y - bb.min_y)),
                            "base_level_mm": float(z0),
                            "top_level_mm": float(z1),
                        },
                        confidence=float(stair_seg.confidence),
                    )
                )

    openings = attach_openings(walls, segments)
    details = attach_detail_nodes(walls, floors, segments)
    all_nodes = walls + floors + ceilings + stairs + openings + details
    return {n.id: n for n in all_nodes}


# -----------------------------------------------------------------------------
# 4. FINAL CLEAN PIPELINE
# -----------------------------------------------------------------------------


def run_clean_bim(segments: list[GeometrySegment]) -> list[GeometrySegment]:
    walls = reconstruct_walls(segments)
    floors = reconstruct_floors(segments)
    doors = [s for s in segments if _is_type(s, ElementType.DOOR)]
    walls = attach_doors_to_walls(walls, doors)
    return walls + floors
