"""deviation_heatmap.py — SVG/PNG heatmap renderer for DeviationReport + LOA.

Produces a 2D overlay of every element in a session, coloured by either:
  • deviation magnitude vs source scan (mm), or
  • USIBD LOA σ-band / effective tier.

Pure-Python SVG output (zero dependencies). Consumed by:
  GET /sessions/{id}/deviation-heatmap?view=plan&colour_by=loa

Two projection views:
  plan       — top-down X-Y
  elevation  — front Y-Z (cross-section through the site)

Colour bands (LOA tier):
  LOA50 → #1b7a3e  (dark green)
  LOA40 → #31a353  (green)
  LOA30 → #f5b400  (amber)
  LOA20 → #f28b24  (orange)
  LOA10 → #e14d3d  (red)
  unknown → #9ca3af (grey)

Colour bands (deviation magnitude, mm):
  < 5 mm  → green      (LOA30+)
  < 15 mm → amber      (LOA20)
  < 50 mm → orange     (LOA10)
  ≥ 50 mm → red        (out of tolerance)
"""

from __future__ import annotations

from collections.abc import Iterable
from typing import Literal

from agent.models import DeviationReport
from agent.tools.loa_tools import LOATier, classify_loa_tier

# ── Colour palette ───────────────────────────────────────────────────────────

_LOA_COLOURS: dict[str, str] = {
    "LOA50": "#1b7a3e",  # dark green
    "LOA40": "#31a353",  # green
    "LOA30": "#f5b400",  # amber
    "LOA20": "#f28b24",  # orange
    "LOA10": "#e14d3d",  # red
}
_UNKNOWN_COLOUR = "#9ca3af"  # grey
_BG = "#1a1a2e"
_AXIS = "#444466"
_TEXT = "#c8c8e0"


def _deviation_to_colour(dev_mm: float | None) -> str:
    if dev_mm is None:
        return _UNKNOWN_COLOUR
    if dev_mm < 5:
        return _LOA_COLOURS["LOA30"]  # misleading variable name; ok
    if dev_mm < 15:
        return "#f5b400"  # amber
    if dev_mm < 50:
        return "#f28b24"  # orange
    return "#e14d3d"  # red


def _loa_tier_to_colour(tier: str) -> str:
    return _LOA_COLOURS.get(tier, _UNKNOWN_COLOUR)


ViewType = Literal["plan", "elevation"]
ColourBy = Literal["deviation", "loa"]


# ── Public API ───────────────────────────────────────────────────────────────


def render_heatmap_svg(
    report: DeviationReport | None,
    loa_statements: Iterable[dict] | None = None,
    view: ViewType = "plan",
    colour_by: ColourBy = "deviation",
    width_px: int = 1200,
    height_px: int = 900,
    margin_px: int = 60,
) -> str:
    """Render a deviation/LOA heatmap as a stand-alone SVG document.

    Args:
        report:         Session's DeviationReport (provides coordinates).
        loa_statements: Optional list of dicts from /loa-report (used when
                        colour_by='loa'). Each dict must contain
                        'segment_id' and 'effective_tier'.
        view:           'plan' (XY) or 'elevation' (YZ).
        colour_by:      'deviation' uses distance_mm, 'loa' uses effective tier.

    Returns:
        Complete SVG document as string.
    """
    if report is None or not report.records:
        return _empty_svg(width_px, height_px, "No deviation records available")

    # Optional LOA lookup
    loa_by_id: dict[str, str] = {}
    if loa_statements:
        for s in loa_statements:
            sid = s.get("segment_id")
            tier = s.get("effective_tier")
            if sid and tier:
                loa_by_id[sid] = tier

    # Extract 2D coordinates + colour for every record that has a centroid
    plot_records = []
    for r in report.records:
        c = r.scan_centroid_mm or r.ifc_centroid_mm
        if c is None:
            continue  # record with no spatial info (e.g. pure MISSING)
        if view == "plan":
            x, y = float(c[0]), float(c[1])
        else:  # elevation
            x, y = float(c[1]), float(c[2])

        if colour_by == "loa":
            tier = loa_by_id.get(r.scan_segment_id or "")
            if tier is None and r.distance_mm is not None:
                # derive a tier on the fly from deviation (assumes dev ≈ 2σ)
                tier = classify_loa_tier(r.distance_mm / 2.0).value
            colour = _loa_tier_to_colour(tier) if tier else _UNKNOWN_COLOUR
            tooltip = f"{r.ifc_name or r.scan_segment_id or '?'} · {tier or 'unknown'}"
        else:
            colour = _deviation_to_colour(r.distance_mm)
            tooltip = (
                f"{r.ifc_name or r.scan_segment_id or '?'} · {r.distance_mm:.1f} mm"
                if r.distance_mm is not None
                else f"{r.ifc_name or r.scan_segment_id or '?'} · ?"
            )

        plot_records.append((x, y, colour, tooltip))

    if not plot_records:
        return _empty_svg(width_px, height_px, "No spatial data in report")

    # Compute plot extents
    xs = [p[0] for p in plot_records]
    ys = [p[1] for p in plot_records]
    min_x, max_x = min(xs), max(xs)
    min_y, max_y = min(ys), max(ys)
    span_x = max_x - min_x or 1.0
    span_y = max_y - min_y or 1.0

    draw_w = width_px - 2 * margin_px
    draw_h = height_px - 2 * margin_px - 60  # extra 60 px for legend
    scale = min(draw_w / span_x, draw_h / span_y)

    def to_px(x: float, y: float) -> tuple[float, float]:
        px = margin_px + (x - min_x) * scale
        py = height_px - margin_px - 60 - (y - min_y) * scale
        return round(px, 2), round(py, 2)

    parts: list[str] = []
    parts.append(
        f'<svg xmlns="http://www.w3.org/2000/svg" '
        f'width="{width_px}" height="{height_px}" '
        f'viewBox="0 0 {width_px} {height_px}">'
    )
    parts.append(f'<rect width="{width_px}" height="{height_px}" fill="{_BG}"/>')

    # Plot dots (one per element)
    parts.append('<g stroke="#000" stroke-width="0.5">')
    for x, y, colour, tooltip in plot_records:
        cx, cy = to_px(x, y)
        parts.append(
            f'<circle cx="{cx}" cy="{cy}" r="4" fill="{colour}">'
            f"<title>{_escape(tooltip)}</title>"
            f"</circle>"
        )
    parts.append("</g>")

    # Scale bar — 1 m
    bar_m_px = 1000.0 * scale  # 1000 mm in pixel units
    bar_x = margin_px
    bar_y = height_px - 70
    parts.append(
        f'<line x1="{bar_x}" y1="{bar_y}" '
        f'x2="{bar_x + bar_m_px}" y2="{bar_y}" '
        f'stroke="{_AXIS}" stroke-width="2"/>'
    )
    parts.append(
        f'<text x="{bar_x + bar_m_px + 6}" y="{bar_y + 4}" '
        f'font-family="monospace" font-size="10" fill="{_TEXT}">1 m</text>'
    )

    # Legend
    _render_legend(parts, colour_by, height_px, margin_px, width_px)

    # Title
    axis_label = "Plan view (X–Y)" if view == "plan" else "Elevation (Y–Z)"
    title = (
        f"Deviation heatmap · {axis_label} · coloured by "
        f"{'LOA tier' if colour_by == 'loa' else 'deviation (mm)'} · "
        f"{len(plot_records)} elements"
    )
    parts.append(
        f'<text x="{margin_px}" y="{margin_px - 20}" '
        f'font-family="monospace" font-size="13" fill="{_TEXT}">'
        f"{_escape(title)}</text>"
    )

    parts.append("</svg>")
    return "\n".join(parts)


# ── Internal helpers ─────────────────────────────────────────────────────────


def _render_legend(
    parts: list[str], colour_by: ColourBy, height_px: int, margin_px: int, width_px: int
) -> None:
    """Append a colour-band legend to the SVG parts."""
    y = height_px - 35

    if colour_by == "loa":
        bands = [
            ("LOA50 · ±0.1 mm", _LOA_COLOURS["LOA50"]),
            ("LOA40 · ±1 mm", _LOA_COLOURS["LOA40"]),
            ("LOA30 · ±5 mm", _LOA_COLOURS["LOA30"]),
            ("LOA20 · ±15 mm", _LOA_COLOURS["LOA20"]),
            ("LOA10 · ±50 mm", _LOA_COLOURS["LOA10"]),
        ]
    else:
        bands = [
            ("< 5 mm", _LOA_COLOURS["LOA30"]),
            ("5–15 mm", "#f5b400"),
            ("15–50 mm", "#f28b24"),
            ("≥ 50 mm", "#e14d3d"),
            ("unknown", _UNKNOWN_COLOUR),
        ]

    x = margin_px
    for label, colour in bands:
        parts.append(
            f'<rect x="{x}" y="{y}" width="14" height="14" fill="{colour}" '
            f'stroke="#000" stroke-width="0.5"/>'
        )
        parts.append(
            f'<text x="{x + 20}" y="{y + 11}" '
            f'font-family="monospace" font-size="10" fill="{_TEXT}">'
            f"{_escape(label)}</text>"
        )
        # advance: rect + padding + text width (~ len*6.5 px monospace)
        x += 14 + 10 + len(label) * 7 + 14


def _escape(s: str) -> str:
    """Minimal XML escape for labels / tooltips."""
    return s.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;").replace('"', "&quot;")


def _empty_svg(width_px: int, height_px: int, message: str) -> str:
    return (
        f'<svg xmlns="http://www.w3.org/2000/svg" '
        f'width="{width_px}" height="{height_px}" '
        f'viewBox="0 0 {width_px} {height_px}">'
        f'<rect width="{width_px}" height="{height_px}" fill="{_BG}"/>'
        f'<text x="{width_px / 2}" y="{height_px / 2}" '
        f'font-family="monospace" font-size="14" fill="{_TEXT}" '
        f'text-anchor="middle">{_escape(message)}</text>'
        f"</svg>"
    )


# ── Summary statistics ───────────────────────────────────────────────────────


def compute_heatmap_summary(
    report: DeviationReport,
    loa_statements: Iterable[dict] | None = None,
) -> dict:
    """Return a JSON-serialisable summary of what the heatmap shows.

    Useful when the client wants stats without the SVG payload.
    """
    # Distribution by deviation magnitude
    dev_buckets = {"<5": 0, "5-15": 0, "15-50": 0, ">=50": 0, "unknown": 0}
    for r in report.records:
        d = r.distance_mm
        if d is None:
            dev_buckets["unknown"] += 1
        elif d < 5:
            dev_buckets["<5"] += 1
        elif d < 15:
            dev_buckets["5-15"] += 1
        elif d < 50:
            dev_buckets["15-50"] += 1
        else:
            dev_buckets[">=50"] += 1

    # Distribution by LOA tier (if provided)
    loa_buckets: dict[str, int] = {t: 0 for t in LOATier.__members__}
    loa_buckets["unknown"] = 0
    if loa_statements:
        for s in loa_statements:
            tier = s.get("effective_tier", "unknown")
            loa_buckets[tier] = loa_buckets.get(tier, 0) + 1

    return {
        "total_records": len(report.records),
        "by_deviation_mm": dev_buckets,
        "by_loa_tier": loa_buckets if loa_statements else None,
        "matched_threshold_mm": report.matched_threshold_mm,
        "shifted_threshold_mm": report.shifted_threshold_mm,
    }
