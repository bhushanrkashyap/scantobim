from __future__ import annotations

from collections import defaultdict

from agent.models import SegmentShape

MIN_POINT_COUNT = 1
MERGE_IOU_THRESHOLD = 0.65
MERGE_CONTAINMENT_THRESHOLD = 0.90
MERGE_ADJACENT_GAP_MM = 120.0
ALWAYS_KEEP_SHAPES = {SegmentShape.VOID, SegmentShape.VALVE_CANDIDATE}
# Preserve detail for internal/opening candidates; merging these tends to collapse
# many real components into a few oversized blocks.
NO_MERGE_SHAPES = {
    SegmentShape.BOX,
    SegmentShape.CYLINDER,
    SegmentShape.VOID,
    SegmentShape.VALVE_CANDIDATE,
}


def bbox_overlap(b1, b2) -> bool:
    return not (
        b1.max_x < b2.min_x
        or b1.min_x > b2.max_x
        or b1.max_y < b2.min_y
        or b1.min_y > b2.max_y
        or b1.max_z < b2.min_z
        or b1.min_z > b2.max_z
    )


def same_level(z1: float, z2: float, tol: float = 150.0) -> bool:
    return abs(z1 - z2) < tol


def _level_tol_for_shape(shape) -> float:
    try:
        return 300.0 if getattr(shape, "value", "") == "plane_sloped" else 150.0
    except Exception:
        return 150.0


def _bbox_volume(bb) -> float:
    width = max(bb.max_x - bb.min_x, 0.0)
    depth = max(bb.max_y - bb.min_y, 0.0)
    height = max(bb.max_z - bb.min_z, 0.0)
    return width * depth * height


def _bbox_iou(b1, b2) -> float:
    inter_x = max(0.0, min(b1.max_x, b2.max_x) - max(b1.min_x, b2.min_x))
    inter_y = max(0.0, min(b1.max_y, b2.max_y) - max(b1.min_y, b2.min_y))
    inter_z = max(0.0, min(b1.max_z, b2.max_z) - max(b1.min_z, b2.min_z))
    inter_volume = inter_x * inter_y * inter_z

    if inter_volume <= 0.0:
        return 0.0

    union = _bbox_volume(b1) + _bbox_volume(b2) - inter_volume
    if union <= 0.0:
        return 0.0

    return inter_volume / union


def _bbox_containment(small, large) -> float:
    small_volume = _bbox_volume(small)
    if small_volume <= 0.0:
        return 0.0

    inter_x = max(0.0, min(small.max_x, large.max_x) - max(small.min_x, large.min_x))
    inter_y = max(0.0, min(small.max_y, large.max_y) - max(small.min_y, large.min_y))
    inter_z = max(0.0, min(small.max_z, large.max_z) - max(small.min_z, large.min_z))
    inter_volume = inter_x * inter_y * inter_z
    return inter_volume / small_volume


def _axis_overlap(min_a: float, max_a: float, min_b: float, max_b: float) -> float:
    return max(0.0, min(max_a, max_b) - max(min_a, min_b))


def _axis_gap(min_a: float, max_a: float, min_b: float, max_b: float) -> float:
    if max_a < min_b:
        return min_b - max_a
    if max_b < min_a:
        return min_a - max_b
    return 0.0


def _is_near_adjoining(a, b) -> bool:
    shape = getattr(a, "shape", None)
    if shape not in {
        SegmentShape.PLANE_VERTICAL,
        SegmentShape.PLANE_HORIZONTAL,
        SegmentShape.PLANE_SLOPED,
    }:
        return False

    a_bb = a.bounding_box
    b_bb = b.bounding_box

    x_ol = _axis_overlap(a_bb.min_x, a_bb.max_x, b_bb.min_x, b_bb.max_x)
    y_ol = _axis_overlap(a_bb.min_y, a_bb.max_y, b_bb.min_y, b_bb.max_y)
    z_ol = _axis_overlap(a_bb.min_z, a_bb.max_z, b_bb.min_z, b_bb.max_z)
    x_gap = _axis_gap(a_bb.min_x, a_bb.max_x, b_bb.min_x, b_bb.max_x)
    y_gap = _axis_gap(a_bb.min_y, a_bb.max_y, b_bb.min_y, b_bb.max_y)
    z_gap = _axis_gap(a_bb.min_z, a_bb.max_z, b_bb.min_z, b_bb.max_z)

    # Vertical planes: merge coplanar strips with a small horizontal gap.
    if shape == SegmentShape.PLANE_VERTICAL:
        return z_ol > 300.0 and (
            (x_ol > 300.0 and y_gap <= MERGE_ADJACENT_GAP_MM)
            or (y_ol > 300.0 and x_gap <= MERGE_ADJACENT_GAP_MM)
        )

    # Horizontal/sloped planes: merge slab/floor fragments on same elevation.
    return z_gap <= 80.0 and (
        (x_ol > 300.0 and y_gap <= MERGE_ADJACENT_GAP_MM)
        or (y_ol > 300.0 and x_gap <= MERGE_ADJACENT_GAP_MM)
        or (x_ol > 300.0 and y_ol > 300.0)
    )


def _should_merge(a, b, level_tol: float) -> bool:
    shape = getattr(a, "shape", None)
    if shape != getattr(b, "shape", None):
        return False

    if shape in NO_MERGE_SHAPES:
        return False

    if not same_level(a.centroid.z, b.centroid.z, tol=level_tol):
        return False

    if not bbox_overlap(a.bounding_box, b.bounding_box):
        return False

    iou = _bbox_iou(a.bounding_box, b.bounding_box)
    if iou >= MERGE_IOU_THRESHOLD:
        return True

    if _bbox_volume(a.bounding_box) <= _bbox_volume(b.bounding_box):
        containment = _bbox_containment(a.bounding_box, b.bounding_box)
    else:
        containment = _bbox_containment(b.bounding_box, a.bounding_box)

    if containment >= MERGE_CONTAINMENT_THRESHOLD:
        return True

    return _is_near_adjoining(a, b)


def _should_keep_segment(segment) -> bool:
    shape = getattr(segment, "shape", None)
    if shape in ALWAYS_KEEP_SHAPES:
        return True

    if shape is None:
        return False

    bb = getattr(segment, "bounding_box", None)
    if bb is None:
        return False

    if getattr(segment, "point_count", 0) >= MIN_POINT_COUNT:
        return True
    return False


def merge_segments(segments):
    """Merge only duplicate-like segments and keep valid low-count elements.

    The merge pass is intentionally conservative so synthetic and dense scenes
    retain distinct elements instead of collapsing into a few large boxes.
    """

    if not segments:
        return []

    segments = [s for s in segments if _should_keep_segment(s)]
    if not segments:
        return []

    grouped = defaultdict(list)
    for s in segments:
        grouped[s.shape].append(s)

    merged = []

    for shape, group in grouped.items():
        temp = []
        level_tol = _level_tol_for_shape(shape)

        for seg in group:
            merged_flag = False

            for m in temp:
                if _should_merge(seg, m, level_tol):
                    m.bounding_box.min_x = min(m.bounding_box.min_x, seg.bounding_box.min_x)
                    m.bounding_box.max_x = max(m.bounding_box.max_x, seg.bounding_box.max_x)
                    m.bounding_box.min_y = min(m.bounding_box.min_y, seg.bounding_box.min_y)
                    m.bounding_box.max_y = max(m.bounding_box.max_y, seg.bounding_box.max_y)
                    m.bounding_box.min_z = min(m.bounding_box.min_z, seg.bounding_box.min_z)
                    m.bounding_box.max_z = max(m.bounding_box.max_z, seg.bounding_box.max_z)
                    m.point_count += seg.point_count
                    m.confidence = max(m.confidence, seg.confidence)
                    merged_flag = True
                    break

            if not merged_flag:
                temp.append(seg)

        merged.extend(temp)

    return [s for s in merged if _should_keep_segment(s)]
