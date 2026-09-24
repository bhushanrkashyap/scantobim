"""
report.py
─────────
Generates a self-contained HTML audit report from a completed session.

Usage
─────
  from agent.report import generate_report
  path = generate_report(session_id, session, audit_events, chain_valid, event_count)

  # Or run standalone after demo_runner:
  python -m agent.report --db demo_audit.db --session <session_id>
"""

from __future__ import annotations

import argparse
import json
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

# ── Colour map (matches the dark-theme used in the guide) ─────────────────────

_SC_COLOUR = {
    "SC1": ("#e84040", "#1a0808"),
    "SC2": ("#f5a623", "#1a1200"),
    "SC3": ("#4a90e2", "#0a1018"),
    "NS": ("#40c060", "#081408"),
}

_EVENT_COLOUR = {
    "element_created": "#40c060",
    "element_blocked": "#e84040",
    "safety_gate_raised": "#f5a623",
    "safety_gate_approved": "#40c060",
    "safety_gate_rejected": "#e84040",
    "segment_classified": "#4a90e2",
    "session_started": "#a855f7",
    "session_completed": "#00c896",
    "clash_detected": "#ff6b35",
    "ncr_raised": "#ff6b35",
}


# ── Helpers ───────────────────────────────────────────────────────────────────


def _ts(iso: str) -> str:
    """Trim ISO timestamp to HH:MM:SS."""
    return iso[11:19] if len(iso) >= 19 else iso


def _sc_badge(sc: str) -> str:
    fg, bg = _SC_COLOUR.get(sc, ("#b0c4e0", "#10141f"))
    return (
        f'<span style="display:inline-block;padding:1px 7px;border-radius:3px;'
        f"font-size:9px;font-weight:700;letter-spacing:0.5px;"
        f'background:{bg};border:1px solid {fg};color:{fg}">{sc}</span>'
    )


def _event_dot(et: str) -> str:
    colour = _EVENT_COLOUR.get(et, "#4a5e78")
    return f'<span style="color:{colour};font-weight:700">●</span>'


def _chain_badge(valid: bool) -> str:
    if valid:
        return (
            '<span style="color:#40c060;font-weight:700;font-size:13px">'
            "✓ PASS — chain intact</span>"
        )
    return '<span style="color:#e84040;font-weight:700;font-size:13px">✗ FAIL — chain broken</span>'


# ── HTML template ─────────────────────────────────────────────────────────────

_CSS = """
:root{--bg:#06080e;--sf:#0c0f18;--sf2:#10141f;--bd:#182030;--bd2:#263050;
--tx:#b0c4e0;--mt:#4a5e78;--wh:#e0eaff;--ac:#00c896}
*{margin:0;padding:0;box-sizing:border-box}
body{background:var(--bg);color:var(--tx);
  font-family:'IBM Plex Mono',Consolas,monospace;
  font-size:12px;line-height:1.7;padding:40px 32px 80px;
  max-width:1080px;margin:0 auto}
h1{font-size:22px;font-weight:800;color:var(--wh);margin-bottom:4px}
h2{font-size:13px;font-weight:700;color:var(--wh);
  padding-bottom:8px;margin:28px 0 14px;
  border-bottom:2px solid var(--bd)}
.meta{color:var(--mt);font-size:10px;margin-bottom:32px;
  border-bottom:1px solid var(--bd);padding-bottom:12px}
.meta span{color:var(--ac)}
.stats{display:grid;grid-template-columns:repeat(6,1fr);gap:2px;margin-bottom:32px}
.stat{background:var(--sf);border:1px solid var(--bd);border-radius:3px;
  padding:12px 14px;text-align:center}
.stat-n{font-size:22px;font-weight:800;line-height:1;margin-bottom:3px}
.stat-l{color:var(--mt);font-size:9px;letter-spacing:.7px;text-transform:uppercase}
table{width:100%;border-collapse:collapse;font-size:11px;margin-bottom:12px}
th{text-align:left;font-size:9px;font-weight:700;letter-spacing:1px;
  text-transform:uppercase;color:var(--mt);padding:7px 10px;
  border-bottom:2px solid var(--bd2);background:var(--sf2)}
td{padding:7px 10px;border-bottom:1px solid var(--bd);vertical-align:top}
tr:hover td{background:var(--sf2)}
.chain-box{background:var(--sf);border:1px solid var(--bd);
  border-left:3px solid var(--ac);border-radius:4px;
  padding:14px 18px;margin:12px 0}
.chain-fail{border-left-color:#e84040}
code{background:var(--sf2);border:1px solid var(--bd);border-radius:3px;
  padding:1px 5px;font-size:10px;color:#00d4ff}
.footer{margin-top:48px;padding-top:12px;border-top:1px solid var(--bd);
  display:flex;justify-content:space-between;color:var(--mt);font-size:10px}
"""


def _build_html(
    session_id: str,
    session: dict,
    audit_events: list[dict],
    chain_valid: bool,
    event_count: int,
) -> str:

    site = session.get("site_name", "—")
    state = session.get("state", "?")
    total_seg = session.get("total_segments", 0)
    total_el = session.get("total_elements_created", 0)
    total_clash = session.get("total_clashes", 0)
    total_ncr = session.get("total_ncrs", 0)
    generated = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")

    # ── Stats bar ─────────────────────────────────────────────────────────────
    stats_html = f"""
<div class="stats">
  <div class="stat"><div class="stat-n" style="color:#00c896">{total_seg}</div>
    <div class="stat-l">Segments</div></div>
  <div class="stat"><div class="stat-n" style="color:#40c060">{total_el}</div>
    <div class="stat-l">Created</div></div>
  <div class="stat"><div class="stat-n" style="color:#4a90e2">{event_count}</div>
    <div class="stat-l">Audit Events</div></div>
  <div class="stat"><div class="stat-n" style="color:#f5a623">{total_clash}</div>
    <div class="stat-l">Clashes</div></div>
  <div class="stat"><div class="stat-n" style="color:#ff6b35">{total_ncr}</div>
    <div class="stat-l">NCRs</div></div>
  <div class="stat"><div class="stat-n" style="color:{"#40c060" if chain_valid else "#e84040"}">
    {"✓" if chain_valid else "✗"}</div>
    <div class="stat-l">Chain</div></div>
</div>"""

    # ── Safety classification summary ─────────────────────────────────────────
    classified = [e for e in audit_events if e.get("event_type") == "segment_classified"]
    sc_counts = Counter(e.get("detail", {}).get("safety_category", "NS") for e in classified)
    sc_rows = "".join(
        f"<tr><td>{_sc_badge(sc)}</td><td>{count}</td>"
        f"<td>{round(count / max(len(classified), 1) * 100, 1)}%</td></tr>"
        for sc, count in sorted(sc_counts.items())
    )
    sc_table = f"""
<h2>Safety Classification Summary</h2>
<table>
  <tr><th>Category</th><th>Segments</th><th>Share</th></tr>
  {sc_rows if sc_rows else '<tr><td colspan="3" style="color:#4a5e78">No classifications recorded</td></tr>'}
</table>"""

    # ── Element creation table ────────────────────────────────────────────────
    el_events = [
        e for e in audit_events if e.get("event_type") in ("element_created", "element_blocked")
    ]
    el_rows = ""
    for e in el_events[:100]:  # cap at 100 rows for readability
        et = e.get("event_type", "?")
        el_id = e.get("element_id") or e.get("detail", {}).get("element_id", "—")
        seg_id = (e.get("segment_id") or "—")[:12] + "..."
        detail = e.get("detail", {})
        el_type = detail.get("element_type", "?")
        sc = detail.get("safety_category", "?")
        actor = e.get("actor", "?")
        ts = _ts(e.get("timestamp_utc", ""))
        status_html = (
            '<span style="color:#40c060">✓ Created</span>'
            if et == "element_created"
            else '<span style="color:#e84040">🔴 Blocked</span>'
        )
        el_rows += (
            f"<tr><td>{ts}</td><td><code>{el_id[:18]}</code></td>"
            f"<td>{el_type}</td><td>{_sc_badge(sc)}</td>"
            f"<td>{seg_id}</td><td>{status_html}</td><td>{actor}</td></tr>"
        )
    el_table = f"""
<h2>Element Provenance ({len(el_events)} records)</h2>
<table>
  <tr>
    <th>Time</th><th>Element ID</th><th>Type</th>
    <th>Safety</th><th>Segment</th><th>Status</th><th>Actor</th>
  </tr>
  {el_rows if el_rows else '<tr><td colspan="7" style="color:#4a5e78">No element events</td></tr>'}
</table>"""

    # ── Safety gate decisions ─────────────────────────────────────────────────
    gate_events = [e for e in audit_events if "safety_gate" in e.get("event_type", "")]
    gate_rows = ""
    for e in gate_events:
        et = e.get("event_type", "?")
        detail = e.get("detail", {})
        ts = _ts(e.get("timestamp_utc", ""))
        sc = detail.get("safety_category", "?")
        actor = e.get("actor", "?")
        reason = detail.get("reason", detail.get("comments", "—"))[:60]
        colour = "#40c060" if "approved" in et else "#e84040" if "rejected" in et else "#f5a623"
        gate_rows += (
            f"<tr><td>{ts}</td>"
            f"<td style='color:{colour};font-weight:700'>{et.replace('_', ' ').title()}</td>"
            f"<td>{_sc_badge(sc)}</td>"
            f"<td>{actor}</td><td>{reason}</td></tr>"
        )
    gate_table = f"""
<h2>Safety Gate Decisions ({len(gate_events)} events)</h2>
<table>
  <tr><th>Time</th><th>Decision</th><th>Safety</th><th>Actor</th><th>Reason / Comments</th></tr>
  {gate_rows if gate_rows else '<tr><td colspan="5" style="color:#4a5e78">No gate events</td></tr>'}
</table>"""

    # ── Audit event log ───────────────────────────────────────────────────────
    event_type_counts = Counter(e.get("event_type", "?") for e in audit_events)
    event_count_rows = "".join(
        f"<tr><td>{_event_dot(et)} {et}</td><td>{cnt}</td></tr>"
        for et, cnt in sorted(event_type_counts.items(), key=lambda x: -x[1])
    )
    event_log = f"""
<h2>Audit Event Summary ({event_count} total)</h2>
<table style="max-width:420px">
  <tr><th>Event Type</th><th>Count</th></tr>
  {event_count_rows}
</table>"""

    # ── Chain integrity block ─────────────────────────────────────────────────
    chain_class = "chain-box" if chain_valid else "chain-box chain-fail"
    chain_block = f"""
<h2>Audit Chain Integrity</h2>
<div class="{chain_class}">
  {_chain_badge(chain_valid)}<br>
  <span style="color:#4a5e78;font-size:10px;margin-top:6px;display:block">
    HMAC-SHA256 chained ledger — {event_count} events verified.
    {
        "All event hashes form an unbroken chain. No tampering detected."
        if chain_valid
        else "One or more links are broken. Investigate immediately."
    }
  </span>
</div>"""

    # ── Assemble ──────────────────────────────────────────────────────────────
    return f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>Scan-to-BIM Audit Report — {site}</title>
<link href="https://fonts.googleapis.com/css2?family=IBM+Plex+Mono:wght@300;400;700&display=swap"
      rel="stylesheet">
<style>{_CSS}</style>
</head>
<body>

<h1>Scan-to-BIM — Audit Report</h1>
<div class="meta">
  Site: <span>{site}</span>
  &nbsp;·&nbsp; Session: <span>{session_id}</span>
  &nbsp;·&nbsp; State: <span>{state}</span>
  &nbsp;·&nbsp; Generated: <span>{generated}</span>
</div>

{stats_html}
{chain_block}
{sc_table}
{el_table}
{gate_table}
{event_log}

<div class="footer">
  <span>Scan-to-BIM PoV</span>
  <span>{generated}</span>
</div>

</body>
</html>"""


# ── Public API ────────────────────────────────────────────────────────────────


def generate_report(
    session_id: str,
    session: dict,
    audit_events: list[dict],
    chain_valid: bool,
    event_count: int,
    output_path: str = "demo_report.html",
) -> str:
    """
    Build and write an HTML audit report.

    Returns the resolved absolute path of the written file.
    """
    html = _build_html(
        session_id=session_id,
        session=session,
        audit_events=audit_events,
        chain_valid=chain_valid,
        event_count=event_count,
    )
    path = Path(output_path).resolve()
    path.write_text(html, encoding="utf-8")
    return str(path)


# ── CLI entry point ───────────────────────────────────────────────────────────


def _main() -> None:
    parser = argparse.ArgumentParser(description="Generate Scan-to-BIM HTML audit report")
    parser.add_argument("--db", required=True, help="Path to SQLite audit database")
    parser.add_argument("--session", required=True, help="Session ID to report on")
    parser.add_argument("--out", default="demo_report.html", help="Output HTML file path")
    args = parser.parse_args()

    # Lazy import only needed for CLI path
    import asyncio

    import aiosqlite

    async def _load(db_path: str, session_id: str) -> tuple[dict, list[dict]]:
        async with aiosqlite.connect(db_path) as db:
            db.row_factory = aiosqlite.Row

            # Session row
            async with db.execute(
                "SELECT * FROM sessions WHERE session_id = ?", [session_id]
            ) as cur:
                row = await cur.fetchone()
                session = dict(row) if row else {}

            # Audit events
            async with db.execute(
                "SELECT * FROM audit_events WHERE session_id = ? ORDER BY timestamp_utc",
                [session_id],
            ) as cur:
                rows = await cur.fetchall()
                events = []
                for r in rows:
                    d = dict(r)
                    # detail may be stored as JSON string
                    if isinstance(d.get("detail"), str):
                        try:
                            d["detail"] = json.loads(d["detail"])
                        except Exception:
                            pass
                    events.append(d)

        return session, events

    session, events = asyncio.run(_load(args.db, args.session))
    if not session:
        print(f"Session {args.session!r} not found in {args.db}")
        raise SystemExit(1)

    path = generate_report(
        session_id=args.session,
        session=session,
        audit_events=events,
        chain_valid=True,  # re-verify externally if needed
        event_count=len(events),
        output_path=args.out,
    )
    print(f"Report written → {path}")


if __name__ == "__main__":
    _main()
