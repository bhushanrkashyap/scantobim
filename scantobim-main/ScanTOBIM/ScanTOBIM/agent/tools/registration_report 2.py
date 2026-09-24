"""registration_report.py — P2.2c Registration QA report.

Takes a StationRegistrationResult (multi-scan ICP output from Sprint 3.5) and
classifies every station + the session against industry-standard thresholds:

  RICS Measured Surveys of Buildings, 3rd ed. (April 2024) / PAS 128 Quality Levels:
    Category A  — ≤  3 mm RMSE, inlier ratio ≥ 0.90   (survey-grade / engineering)
    Category B  — ≤ 10 mm RMSE, inlier ratio ≥ 0.80   (BIM / design)
    Category C  — ≤ 50 mm RMSE, inlier ratio ≥ 0.60   (general planning)
    FAIL        —  > 50 mm RMSE or inlier ratio < 0.60

The Category is the first tier the station satisfies, walking the order above.
A session's overall category is the worst tier among its stations (same logic
as LOA's "worst effective tier" gate).

Outputs:
  - RegistrationQAReport — structured model (JSON-friendly)
  - render_registration_html() — self-contained HTML for downloadable QA deliverable

Zero external dependencies — pure Python.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from enum import Enum

from agent.models import StationRegistrationResult

# ── Category thresholds ──────────────────────────────────────────────────────


class RICSCategory(str, Enum):
    A = "A"  # ≤ 3 mm
    B = "B"  # ≤ 10 mm
    C = "C"  # ≤ 50 mm
    FAIL = "FAIL"  # out of tolerance


# (category, rmse_max_mm, inlier_min) — evaluated best → worst
_CATEGORY_THRESHOLDS: list[tuple[RICSCategory, float, float]] = [
    (RICSCategory.A, 3.0, 0.90),
    (RICSCategory.B, 10.0, 0.80),
    (RICSCategory.C, 50.0, 0.60),
]


def classify_station(rmse_mm: float, inlier_ratio: float) -> RICSCategory:
    """Return the best RICS category a station satisfies, or FAIL."""
    for cat, rmse_cap, inlier_min in _CATEGORY_THRESHOLDS:
        if rmse_mm <= rmse_cap and inlier_ratio >= inlier_min:
            return cat
    return RICSCategory.FAIL


def worst_category(a: RICSCategory, b: RICSCategory) -> RICSCategory:
    """Return the less accurate (further down the table) of two categories."""
    order = [RICSCategory.A, RICSCategory.B, RICSCategory.C, RICSCategory.FAIL]
    return order[max(order.index(a), order.index(b))]


# ── Report structure ─────────────────────────────────────────────────────────


@dataclass(frozen=True)
class StationQA:
    station_index: int
    source_file: str
    rmse_mm: float
    inlier_ratio: float
    category: RICSCategory
    pass_fail: str  # "pass" | "fail"

    def to_dict(self) -> dict:
        return {
            "station_index": self.station_index,
            "source_file": self.source_file,
            "rmse_mm": round(self.rmse_mm, 3),
            "inlier_ratio": round(self.inlier_ratio, 3),
            "category": self.category.value,
            "pass_fail": self.pass_fail,
        }


@dataclass(frozen=True)
class RegistrationQAReport:
    session_id: str
    generated_at_utc: str
    station_count: int
    stations: list[StationQA]
    global_rmse_mm: float
    overall_category: RICSCategory
    overall_pass_fail: str
    merged_point_count: int
    standard_version: str = "RICS Measured Surveys 3rd ed. (2024)"

    def to_dict(self) -> dict:
        return {
            "session_id": self.session_id,
            "generated_at_utc": self.generated_at_utc,
            "station_count": self.station_count,
            "stations": [s.to_dict() for s in self.stations],
            "global_rmse_mm": round(self.global_rmse_mm, 3),
            "overall_category": self.overall_category.value,
            "overall_pass_fail": self.overall_pass_fail,
            "merged_point_count": self.merged_point_count,
            "standard_version": self.standard_version,
        }


def build_qa_report(result: StationRegistrationResult) -> RegistrationQAReport:
    """Classify every station + the session against RICS/PAS 128 tiers."""
    stations: list[StationQA] = []
    for s in result.stations:
        cat = classify_station(s.rmse_mm, s.inlier_ratio)
        stations.append(
            StationQA(
                station_index=s.station_index,
                source_file=s.source_file,
                rmse_mm=s.rmse_mm,
                inlier_ratio=s.inlier_ratio,
                category=cat,
                pass_fail="pass" if cat != RICSCategory.FAIL else "fail",
            )
        )

    # Overall category = worst of (per-station worst, global-RMSE-derived)
    worst_from_stations = RICSCategory.A
    for s in stations:
        worst_from_stations = worst_category(worst_from_stations, s.category)

    global_cat = classify_station(
        result.global_rmse_mm,
        # Use average inlier ratio when we only have per-station values
        sum(s.inlier_ratio for s in result.stations) / max(len(result.stations), 1),
    )
    overall = worst_category(worst_from_stations, global_cat)

    return RegistrationQAReport(
        session_id=result.session_id,
        generated_at_utc=datetime.now(timezone.utc).isoformat(),
        station_count=result.station_count,
        stations=stations,
        global_rmse_mm=result.global_rmse_mm,
        overall_category=overall,
        overall_pass_fail="pass" if overall != RICSCategory.FAIL else "fail",
        merged_point_count=result.merged_point_count,
    )


# ── HTML rendering ───────────────────────────────────────────────────────────

_CATEGORY_COLOURS: dict[str, str] = {
    "A": "#1b7a3e",  # dark green
    "B": "#31a353",  # green
    "C": "#f5b400",  # amber
    "FAIL": "#e14d3d",  # red
}


def render_registration_html(report: RegistrationQAReport) -> str:
    """Produce a self-contained HTML QA deliverable (printable)."""
    rows = []
    for s in report.stations:
        cat_colour = _CATEGORY_COLOURS[s.category.value]
        rows.append(
            f"<tr>"
            f"<td>{s.station_index}</td>"
            f"<td class='mono'>{_escape(s.source_file)}</td>"
            f"<td class='num'>{s.rmse_mm:.3f}</td>"
            f"<td class='num'>{s.inlier_ratio:.3f}</td>"
            f"<td><span class='tier' style='background:{cat_colour}'>"
            f"Cat {s.category.value}</span></td>"
            f"<td class='{'pass' if s.pass_fail == 'pass' else 'fail'}'>"
            f"{s.pass_fail.upper()}</td>"
            f"</tr>"
        )

    overall_colour = _CATEGORY_COLOURS[report.overall_category.value]

    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<title>Registration QA Report — {_escape(report.session_id[:8])}</title>
<style>
  :root {{
    --bg:#0f0f1a; --card:#1a1a2e; --ink:#e8e8f0; --muted:#9092a8;
    --line:#2a2a42;
  }}
  body {{ margin:0; padding:40px; font-family:-apple-system,system-ui,sans-serif;
         background:var(--bg); color:var(--ink); }}
  .card {{ max-width:960px; margin:0 auto; background:var(--card);
           padding:32px; border-radius:12px; border:1px solid var(--line); }}
  h1 {{ margin:0 0 4px; font-size:22px; }}
  .sub {{ color:var(--muted); font-size:13px; margin-bottom:28px; }}
  .summary {{ display:grid; grid-template-columns:repeat(4, 1fr); gap:14px;
              margin-bottom:28px; }}
  .stat {{ background:var(--bg); border:1px solid var(--line);
          padding:14px; border-radius:8px; }}
  .stat .label {{ color:var(--muted); font-size:11px; text-transform:uppercase;
                  letter-spacing:0.05em; }}
  .stat .value {{ font-size:22px; font-weight:600; margin-top:4px; }}
  table {{ width:100%; border-collapse:collapse; margin-top:8px; }}
  th {{ text-align:left; padding:10px 12px; color:var(--muted);
        font-size:11px; text-transform:uppercase; border-bottom:1px solid var(--line); }}
  td {{ padding:10px 12px; border-bottom:1px solid var(--line); font-size:13px; }}
  td.mono {{ font-family:ui-monospace, SFMono-Regular, Menlo, monospace;
             font-size:12px; color:var(--muted); }}
  td.num  {{ font-family:ui-monospace, SFMono-Regular, Menlo, monospace;
             text-align:right; }}
  .tier   {{ display:inline-block; padding:2px 10px; border-radius:11px;
            font-size:11px; font-weight:600; color:#fff; }}
  .pass {{ color:#4fc97b; font-weight:600; }}
  .fail {{ color:#ff7566; font-weight:600; }}
  .legend {{ margin-top:28px; padding-top:20px; border-top:1px solid var(--line);
             color:var(--muted); font-size:12px; line-height:1.6; }}
  .legend strong {{ color:var(--ink); }}
  .footer {{ margin-top:22px; color:var(--muted); font-size:11px; }}
</style>
</head>
<body>
<div class="card">
  <h1>Registration QA Report</h1>
  <div class="sub">
    Session <span class="mono">{_escape(report.session_id[:8])}</span>
    · Generated {_escape(report.generated_at_utc[:19])}Z
    · {_escape(report.standard_version)}
  </div>

  <div class="summary">
    <div class="stat">
      <div class="label">Stations</div>
      <div class="value">{report.station_count}</div>
    </div>
    <div class="stat">
      <div class="label">Global RMSE</div>
      <div class="value">{report.global_rmse_mm:.2f} mm</div>
    </div>
    <div class="stat">
      <div class="label">Merged points</div>
      <div class="value">{report.merged_point_count:,}</div>
    </div>
    <div class="stat">
      <div class="label">Overall</div>
      <div class="value">
        <span class="tier" style="background:{overall_colour}">
          Cat {report.overall_category.value}
        </span>
      </div>
    </div>
  </div>

  <table>
    <thead><tr>
      <th>#</th><th>Source file</th><th>RMSE (mm)</th>
      <th>Inliers</th><th>Category</th><th>Status</th>
    </tr></thead>
    <tbody>
      {"".join(rows) if rows else '<tr><td colspan="6" class="mono">No stations</td></tr>'}
    </tbody>
  </table>

  <div class="legend">
    <strong>RICS / PAS 128 categories:</strong><br>
    <strong>Cat A</strong> ≤ 3 mm RMSE · ≥ 0.90 inlier — survey-grade, engineering<br>
    <strong>Cat B</strong> ≤ 10 mm RMSE · ≥ 0.80 inlier — BIM / design<br>
    <strong>Cat C</strong> ≤ 50 mm RMSE · ≥ 0.60 inlier — general planning<br>
    <strong>FAIL</strong> beyond Cat C — rescan recommended
  </div>

  <div class="footer">ScanToBIM · Registration QA v1.0</div>
</div>
</body>
</html>
"""
    return html


def _escape(s: str) -> str:
    return s.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;").replace('"', "&quot;")
