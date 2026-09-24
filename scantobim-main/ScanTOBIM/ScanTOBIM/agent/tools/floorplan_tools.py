"""2D floor plan slice extraction and export for Scan-to-BIM.

Sprint 3.5 — Feature 3: Floor Plan Slice Export

Slices a point cloud (or falls back to segment bounding boxes) at a
configurable elevation, fits 2D wall/column lines using iterative RANSAC,
and renders the result as SVG (zero external dependencies) or DXF (ezdxf).

All coordinates in this module are in MILLIMETRES unless the argument name
contains _m or is documented otherwise.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np
import structlog

from agent.models import GeometrySegment, SegmentShape

logger = structlog.get_logger()

# ── Constants ─────────────────────────────────────────────────────────────────

_RANSAC_DISTANCE_THRESHOLD_MM = 50.0
_RANSAC_MIN_INLIERS = 20
_RANSAC_N_ITERATIONS = 300
_RANSAC_MAX_LINES = 100
_CIRCLE_APPROX_POINTS = 12  # polygon points for cylinder cross-section


# ── Data containers ───────────────────────────────────────────────────────────


@dataclass
class LineSegment2D:
    """A 2D line segment fitted by RANSAC."""

    x0: float
    y0: float
    x1: float
    y1: float
    inlier_count: int
    length_mm: float = field(init=False)

    def __post_init__(self) -> None:
        dx = self.x1 - self.x0
        dy = self.y1 - self.y0
        self.length_mm = math.sqrt(dx * dx + dy * dy)


# ── Public API ────────────────────────────────────────────────────────────────


def slice_point_cloud(
    points_mm: np.ndarray,
    elevation_mm: float,
    thickness_mm: float = 200.0,
) -> np.ndarray:
    """Extract XY points within a horizontal Z-band.

    Args:
        points_mm:    (N, 3) array of XYZ points in mm.
        elevation_mm: Centre Z of the slice band in mm.
        thickness_mm: Total band height; half-thickness above and below.

    Returns:
        (M, 2) array of XY points inside the band.
    """
    if points_mm.ndim != 2 or points_mm.shape[1] < 3:
        return np.empty((0, 2), dtype=np.float64)

    half = thickness_mm / 2.0
    z_min = elevation_mm - half
    z_max = elevation_mm + half

    mask = (points_mm[:, 2] >= z_min) & (points_mm[:, 2] <= z_max)
    return points_mm[mask, :2].astype(np.float64)


def slice_from_segments(
    segments: list[GeometrySegment],
    elevation_mm: float,
    thickness_mm: float = 200.0,
) -> np.ndarray:
    """Approximate a Z-band slice from segment bounding boxes.

    Used when no merged point cloud is available.

    Strategy per segment shape:
    - PLANE_VERTICAL  → 4 corner XY points of the face
    - CYLINDER        → circle approximated as N polygon points
    - Others          → 4 XY corners of the bounding box footprint

    Only segments whose Z range overlaps [elevation_mm ± thickness_mm/2]
    are included.

    Returns:
        (M, 2) XY array in mm.
    """
    half = thickness_mm / 2.0
    z_low = elevation_mm - half
    z_high = elevation_mm + half

    xy_pts: list[np.ndarray] = []

    for seg in segments:
        bb = seg.bounding_box
        # Skip if segment Z range doesn't overlap the slice band
        if bb.max_z < z_low or bb.min_z > z_high:
            continue

        if seg.shape == SegmentShape.PLANE_VERTICAL:
            # Four XY corners of the vertical face
            pts = np.array(
                [
                    [bb.min_x, bb.min_y],
                    [bb.max_x, bb.min_y],
                    [bb.max_x, bb.max_y],
                    [bb.min_x, bb.max_y],
                ],
                dtype=np.float64,
            )

        elif seg.shape == SegmentShape.CYLINDER:
            # Circle at centroid XY, radius from bounding box
            cx = seg.centroid.x
            cy = seg.centroid.y
            r = (bb.max_x - bb.min_x + bb.max_y - bb.min_y) / 4.0
            angles = np.linspace(0, 2 * math.pi, _CIRCLE_APPROX_POINTS, endpoint=False)
            pts = np.column_stack(
                [
                    cx + r * np.cos(angles),
                    cy + r * np.sin(angles),
                ]
            )

        else:
            # Bounding box XY footprint
            pts = np.array(
                [
                    [bb.min_x, bb.min_y],
                    [bb.max_x, bb.min_y],
                    [bb.max_x, bb.max_y],
                    [bb.min_x, bb.max_y],
                ],
                dtype=np.float64,
            )

        xy_pts.append(pts)

    if not xy_pts:
        return np.empty((0, 2), dtype=np.float64)

    return np.vstack(xy_pts)


def ransac_2d_lines(
    pts_xy: np.ndarray,
    min_inliers: int = _RANSAC_MIN_INLIERS,
    n_iterations: int = _RANSAC_N_ITERATIONS,
    distance_threshold_mm: float = _RANSAC_DISTANCE_THRESHOLD_MM,
    max_lines: int = _RANSAC_MAX_LINES,
) -> list[LineSegment2D]:
    """Iterative 2D RANSAC line fitting.

    Each iteration:
      1. Sample 2 random points to define a line.
      2. Count inliers (points within distance_threshold_mm).
      3. Keep the best line if it beats the previous best.
    After all iterations, refine inliers by least-squares and remove them
    before the next line extraction pass.

    Args:
        pts_xy:               (M, 2) XY array in mm.
        min_inliers:          Minimum inlier count to accept a line.
        n_iterations:         RANSAC iterations per line.
        distance_threshold_mm: Point-to-line distance threshold.
        max_lines:            Hard cap on lines extracted.

    Returns:
        List of LineSegment2D, longest lines first.
    """
    if pts_xy.shape[0] < 2:
        return []

    rng = np.random.default_rng(42)
    lines: list[LineSegment2D] = []
    remaining = pts_xy.copy()

    while len(remaining) >= max(min_inliers, 2) and len(lines) < max_lines:
        best_inliers: np.ndarray | None = None
        best_count = 0

        for _ in range(n_iterations):
            if len(remaining) < 2:
                break
            idxs = rng.choice(len(remaining), size=2, replace=False)
            p1, p2 = remaining[idxs[0]], remaining[idxs[1]]

            # Line equation: ax + by + c = 0, normalised so a²+b²=1
            dx = p2[0] - p1[0]
            dy = p2[1] - p1[1]
            length = math.sqrt(dx * dx + dy * dy)
            if length < 1e-9:
                continue

            a = -dy / length
            b = dx / length
            c = -(a * p1[0] + b * p1[1])

            dist = np.abs(a * remaining[:, 0] + b * remaining[:, 1] + c)
            inlier_mask = dist < distance_threshold_mm
            inlier_count = int(inlier_mask.sum())

            if inlier_count > best_count:
                best_count = inlier_count
                best_inliers = inlier_mask

        if best_inliers is None or best_count < min_inliers:
            break

        # Refine: fit least-squares line through all inliers
        inlier_pts = remaining[best_inliers]
        seg = _fit_line_segment(inlier_pts)
        if seg is not None:
            lines.append(seg)

        # Remove inliers for next iteration
        remaining = remaining[~best_inliers]

    lines.sort(key=lambda l: l.length_mm, reverse=True)
    logger.debug("ransac_2d_lines_done", lines=len(lines), remaining_pts=len(remaining))
    return lines


def render_svg(
    lines: list[LineSegment2D],
    pts_xy: np.ndarray,
    width_px: int = 1200,
    height_px: int = 900,
    margin_px: int = 40,
) -> str:
    """Render floor plan lines and raw slice points as an SVG string.

    Pure Python — no matplotlib, no Pillow, no external dependencies.

    Args:
        lines:      Fitted wall/column line segments.
        pts_xy:     Raw slice points (grey dots, semi-transparent).
        width_px:   SVG canvas width in pixels.
        height_px:  SVG canvas height in pixels.
        margin_px:  Canvas margin in pixels.

    Returns:
        Complete SVG document as a Python string.
    """
    # Compute bounding box of all geometry
    all_pts = pts_xy.copy() if pts_xy.shape[0] > 0 else np.array([[0, 0]], dtype=float)
    if lines:
        line_pts = np.array([[l.x0, l.y0] for l in lines] + [[l.x1, l.y1] for l in lines])
        all_pts = np.vstack([all_pts, line_pts]) if pts_xy.shape[0] > 0 else line_pts

    if all_pts.shape[0] == 0:
        return _empty_svg(width_px, height_px, "No points in slice band")

    min_x, min_y = all_pts.min(axis=0)
    max_x, max_y = all_pts.max(axis=0)
    span_x = max_x - min_x or 1.0
    span_y = max_y - min_y or 1.0

    draw_w = width_px - 2 * margin_px
    draw_h = height_px - 2 * margin_px
    scale = min(draw_w / span_x, draw_h / span_y)

    def to_px(x: float, y: float) -> tuple[float, float]:
        # Flip Y so north is up
        px = margin_px + (x - min_x) * scale
        py = height_px - margin_px - (y - min_y) * scale
        return round(px, 2), round(py, 2)

    parts: list[str] = []
    parts.append(
        f'<svg xmlns="http://www.w3.org/2000/svg" '
        f'width="{width_px}" height="{height_px}" '
        f'viewBox="0 0 {width_px} {height_px}">'
    )
    parts.append(f'<rect width="{width_px}" height="{height_px}" fill="#1a1a2e"/>')

    # Raw slice points — grey, semi-transparent, radius 1px
    if pts_xy.shape[0] > 0:
        # Subsample if too many points (SVG gets large)
        max_dot_pts = 20_000
        if len(pts_xy) > max_dot_pts:
            rng = np.random.default_rng(0)
            idxs = rng.choice(len(pts_xy), max_dot_pts, replace=False)
            dot_pts = pts_xy[idxs]
        else:
            dot_pts = pts_xy

        # Batch as a single <g> for performance
        parts.append('<g fill="#aaaaaa" opacity="0.3">')
        for pt in dot_pts:
            cx, cy = to_px(float(pt[0]), float(pt[1]))
            parts.append(f'<circle cx="{cx}" cy="{cy}" r="1"/>')
        parts.append("</g>")

    # Fitted lines — bright blue, stroke-width 2
    if lines:
        parts.append('<g stroke="#4fc3f7" stroke-width="2" stroke-linecap="round">')
        for ln in lines:
            x0, y0 = to_px(ln.x0, ln.y0)
            x1, y1 = to_px(ln.x1, ln.y1)
            parts.append(f'<line x1="{x0}" y1="{y0}" x2="{x1}" y2="{y1}"/>')
        parts.append("</g>")

    # Legend
    parts.append(
        f'<text x="{margin_px}" y="{height_px - 10}" '
        f'font-family="monospace" font-size="11" fill="#888">'
        f"ScanToBIM Floor Plan · {len(lines)} lines · {len(pts_xy):,} pts"
        f"</text>"
    )
    parts.append("</svg>")

    return "\n".join(parts)


def render_dxf(
    lines: list[LineSegment2D],
    pts_xy: np.ndarray,
    elevation_m: float = 1.0,
) -> bytes:
    """Render floor plan as a DXF R2010 file.

    Layers:
        POINTS  — raw slice points as POINT entities
        WALLS   — fitted line segments as LINE entities
        METADATA — single TEXT entity with session info

    Args:
        lines:       Fitted wall/column line segments (mm coordinates).
        pts_xy:      Raw slice points (mm coordinates).
        elevation_m: Slice elevation in metres (used in metadata text).

    Returns:
        DXF file as bytes.

    Raises:
        ImportError: If ezdxf is not installed.
    """
    try:
        import ezdxf
    except ImportError as exc:
        raise ImportError(
            "ezdxf is required for DXF export. Install with: pip install ezdxf"
        ) from exc

    doc = ezdxf.new(dxfversion="R2010")
    msp = doc.modelspace()

    # Create layers
    doc.layers.add("POINTS", dxfattribs={"color": 8})  # grey
    doc.layers.add("WALLS", dxfattribs={"color": 5})  # blue
    doc.layers.add("METADATA", dxfattribs={"color": 2})  # yellow

    # WALLS layer — fitted lines (convert mm → m for DXF world units)
    for ln in lines:
        msp.add_line(
            start=(ln.x0 / 1000, ln.y0 / 1000),
            end=(ln.x1 / 1000, ln.y1 / 1000),
            dxfattribs={"layer": "WALLS"},
        )

    # POINTS layer — raw slice points (subsample to keep file size manageable)
    max_pts = 50_000
    if len(pts_xy) > max_pts:
        rng = np.random.default_rng(0)
        idxs = rng.choice(len(pts_xy), max_pts, replace=False)
        export_pts = pts_xy[idxs]
    else:
        export_pts = pts_xy

    for pt in export_pts:
        msp.add_point(
            location=(float(pt[0]) / 1000, float(pt[1]) / 1000),
            dxfattribs={"layer": "POINTS"},
        )

    # METADATA text
    msp.add_text(
        f"ScanToBIM Floor Plan | Elevation: {elevation_m:.2f}m | "
        f"Lines: {len(lines)} | Points: {len(pts_xy):,}",
        dxfattribs={"layer": "METADATA", "height": 0.3},
    )

    import io as _io

    buf = _io.StringIO()
    doc.write(buf)
    return buf.getvalue().encode("utf-8")


# ── Private helpers ───────────────────────────────────────────────────────────


def _fit_line_segment(pts: np.ndarray) -> LineSegment2D | None:
    """Fit a line to a set of 2D points and return the endpoint segment.

    Projects all points onto the PCA principal axis and returns the extreme
    projected points as the segment endpoints.
    """
    if len(pts) < 2:
        return None

    centroid = pts.mean(axis=0)
    centred = pts - centroid
    try:
        _, _, Vt = np.linalg.svd(centred, full_matrices=False)
    except np.linalg.LinAlgError:
        return None

    direction = Vt[0]  # principal axis (unit vector)

    # Project each point onto the direction
    projections = centred @ direction
    t_min, t_max = projections.min(), projections.max()

    p0 = centroid + t_min * direction
    p1 = centroid + t_max * direction

    return LineSegment2D(
        x0=float(p0[0]),
        y0=float(p0[1]),
        x1=float(p1[0]),
        y1=float(p1[1]),
        inlier_count=len(pts),
    )


def _empty_svg(width_px: int, height_px: int, message: str) -> str:
    """Return a valid SVG with a single explanatory text message."""
    cx = width_px // 2
    cy = height_px // 2
    return (
        f'<svg xmlns="http://www.w3.org/2000/svg" '
        f'width="{width_px}" height="{height_px}">'
        f'<rect width="{width_px}" height="{height_px}" fill="#1a1a2e"/>'
        f'<text x="{cx}" y="{cy}" text-anchor="middle" '
        f'font-family="monospace" font-size="16" fill="#888">'
        f"{message}</text>"
        f"</svg>"
    )
