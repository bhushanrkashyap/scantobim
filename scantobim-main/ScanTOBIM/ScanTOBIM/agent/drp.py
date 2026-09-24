"""
drp.py  —  P5 Design Record Package generator
───────────────────────────────────────────────
Generates a self-contained, print-ready HTML Design Record Package (DRP)
that satisfies NQA-1 and ISO 19650 documentation requirements.

7 sections:
  1. Cover page       — project, session, standard declarations
  2. Executive summary — element counts, safety breakdown
  3. Safety gate log   — SC1 blocks, SC2 approvals/rejections, approver names
  4. NCR register      — open + resolved NCRs with resolutions
  5. Chain certificate — HMAC-SHA256 integrity PASS/FAIL
  6. Approval signatures — SC2 approvers, timestamps, UPNs
  7. Element provenance — full scan → BIM traceability table

Usage
─────
  from agent.drp import generate_drp
  html = generate_drp(session_id, session, audit_events, ncr_summary,
                       chain_result, cde_state)
  Path("drp.html").write_text(html)
"""

from __future__ import annotations

from collections import Counter
from datetime import datetime, timezone

# ── Shared style ──────────────────────────────────────────────────────────────

_CSS = """
@page { size: A4; margin: 20mm 18mm; }
:root {
  --bg:#fff; --sf:#f4f6fa; --bd:#d0d8e8; --tx:#1a2030;
  --mt:#607090; --ac:#0050c0; --red:#c00020; --grn:#006820;
  --ylw:#b06000; --mono: 'Courier New', monospace;
}
* { margin:0; padding:0; box-sizing:border-box; }
body {
  background:var(--bg); color:var(--tx);
  font-family:'Segoe UI',Arial,sans-serif;
  font-size:10pt; line-height:1.55;
  padding:12mm 14mm;
}
/* Cover */
.cover { page-break-after:always; text-align:center; padding-top:30mm; }
.cover h1 { font-size:22pt; font-weight:800; color:var(--ac); margin-bottom:4mm; }
.cover h2 { font-size:14pt; font-weight:600; color:var(--tx); margin-bottom:8mm; }
.cover .meta-table { display:inline-block; text-align:left; margin-top:10mm; }
.cover .meta-table td { padding:2mm 6mm 2mm 0; }
.cover .meta-table .lbl { color:var(--mt); font-size:9pt; font-weight:600; }
.cover .standards {
  margin-top:14mm; border:1px solid var(--bd);
  padding:4mm 8mm; display:inline-block; text-align:left;
}
.cover .standards li { font-size:9pt; margin-bottom:2mm; }
/* Section headers */
h2.section {
  font-size:13pt; font-weight:700; color:var(--ac);
  border-bottom:2px solid var(--ac);
  padding-bottom:2mm; margin:10mm 0 5mm;
  page-break-after:avoid;
}
h3 { font-size:10.5pt; font-weight:700; margin:6mm 0 3mm; color:var(--tx); }
/* Tables */
table { width:100%; border-collapse:collapse; font-size:9pt; margin-bottom:6mm; }
th {
  background:var(--sf); color:var(--mt);
  text-align:left; font-size:8pt; font-weight:700;
  letter-spacing:.5px; text-transform:uppercase;
  padding:2.5mm 3mm; border-bottom:2px solid var(--bd);
}
td { padding:2mm 3mm; border-bottom:1px solid var(--bd); vertical-align:top; }
tr:nth-child(even) td { background:#fafbfd; }
/* Badges */
.badge {
  display:inline-block; padding:0.5mm 3mm;
  border-radius:3px; font-size:8pt; font-weight:700;
}
.sc1 { background:#ffe8e8; color:var(--red); border:1px solid #ffc0c0; }
.sc2 { background:#fff4e0; color:var(--ylw); border:1px solid #ffd080; }
.sc3 { background:#e8f0ff; color:var(--ac);  border:1px solid #b0c8ff; }
.ns  { background:#e8f8ee; color:var(--grn); border:1px solid #a0d8b0; }
.pass { color:var(--grn); font-weight:700; }
.fail { color:var(--red); font-weight:700; }
.open-ncr { color:var(--red); font-weight:700; }
.resolved-ncr { color:var(--grn); }
/* Cert box */
.cert {
  border:2px solid var(--bd); border-radius:4px;
  padding:5mm 8mm; margin:4mm 0 8mm;
  background:var(--sf);
}
.cert.pass { border-color:var(--grn); }
.cert.fail { border-color:var(--red); }
.cert-title { font-size:12pt; font-weight:700; margin-bottom:2mm; }
/* Stats grid */
.stats { display:flex; gap:4mm; flex-wrap:wrap; margin-bottom:6mm; }
.stat {
  border:1px solid var(--bd); border-radius:4px;
  padding:3mm 5mm; flex:1; min-width:28mm; text-align:center;
}
.stat-n { font-size:18pt; font-weight:800; color:var(--ac); line-height:1; }
.stat-l { font-size:7.5pt; color:var(--mt); text-transform:uppercase;
          letter-spacing:.5px; margin-top:1mm; }
code { font-family:var(--mono); font-size:8pt; color:#303850; }
.footer {
  margin-top:10mm; padding-top:3mm; border-top:1px solid var(--bd);
  font-size:8pt; color:var(--mt);
  display:flex; justify-content:space-between;
}
"""


# ── Helpers ───────────────────────────────────────────────────────────────────


def _badge(sc: str) -> str:
    return f'<span class="badge {sc.lower()}">{sc}</span>'


def _ts(iso: str) -> str:
    return iso[0:19].replace("T", " ") if len(iso) >= 19 else iso


def _pass(v: bool) -> str:
    return '<span class="pass">✓ PASS</span>' if v else '<span class="fail">✗ FAIL</span>'


# ── Section builders ──────────────────────────────────────────────────────────


def _cover(session_id: str, session: dict, cde_state: str, generated: str) -> str:
    site = session.get("site_name", "—")
    state = session.get("state", "?")
    return f"""
<div class="cover">
  <h1>Design Record Package</h1>
  <h2>{site}</h2>
  <table class="meta-table">
    <tr><td class="lbl">Session ID</td><td><code>{session_id}</code></td></tr>
    <tr><td class="lbl">CDE State</td><td><strong>{cde_state}</strong></td></tr>
    <tr><td class="lbl">Session State</td><td>{state}</td></tr>
    <tr><td class="lbl">Generated</td><td>{generated}</td></tr>
    <tr><td class="lbl">Generated by</td><td>ScanToBIM Agent v0.2.0</td></tr>
  </table>
  <div class="standards">
    <strong>Compliance Standards</strong>
    <ul style="margin-top:3mm;padding-left:5mm">
      <li>NQA-1  — Nuclear Quality Assurance Standard</li>
      <li>ISO 19650 — BIM Information Management</li>
      <li>DSEAR  — Dangerous Substances and Explosive Atmospheres Regulations</li>
      <li>IEC 61511 — Functional Safety (SIL classification reference)</li>
    </ul>
  </div>
</div>"""


def _executive_summary(session: dict, audit_events: list[dict]) -> str:
    total_seg = session.get("total_segments", 0)
    total_el = session.get("total_elements_created", 0)
    clashes = session.get("total_clashes", 0)
    ncrs = session.get("total_ncrs", 0)

    classified = [e for e in audit_events if e.get("event_type") == "segment_classified"]
    sc_counts = Counter(e.get("detail", {}).get("safety_category", "NS") for e in classified)
    sc_rows = "".join(
        f"<tr><td>{_badge(sc)}</td><td>{cnt}</td>"
        f"<td>{round(cnt / max(len(classified), 1) * 100, 1)}%</td></tr>"
        for sc, cnt in sorted(sc_counts.items())
    )

    return f"""
<h2 class="section">Executive Summary</h2>
<div class="stats">
  <div class="stat"><div class="stat-n">{total_seg}</div><div class="stat-l">Segments</div></div>
  <div class="stat"><div class="stat-n">{total_el}</div><div class="stat-l">Elements Created</div></div>
  <div class="stat"><div class="stat-n">{clashes}</div><div class="stat-l">Clashes</div></div>
  <div class="stat"><div class="stat-n">{ncrs}</div><div class="stat-l">NCRs</div></div>
  <div class="stat"><div class="stat-n">{len(classified)}</div><div class="stat-l">Classified</div></div>
</div>
<h3>Safety Classification Breakdown</h3>
<table>
  <tr><th>Category</th><th>Segments</th><th>Share</th></tr>
  {sc_rows or "<tr><td colspan='3'>No segments classified</td></tr>"}
</table>"""


def _safety_gate_log(audit_events: list[dict]) -> str:
    gate_events = [
        e
        for e in audit_events
        if e.get("event_type", "").startswith("safety_gate")
        or e.get("event_type") == "element_blocked"
    ]
    rows = ""
    for e in gate_events:
        et = e.get("event_type", "?")
        detail = e.get("detail", {})
        ts = _ts(e.get("timestamp_utc", ""))
        sc = detail.get("safety_category", "?")
        actor = e.get("actor", "?")
        reason = detail.get("reason", detail.get("comments", "—"))[:80]
        colour = (
            "pass" if "approved" in et else "fail" if ("rejected" in et or "blocked" in et) else ""
        )
        label = et.replace("_", " ").title()
        rows += (
            f"<tr><td>{ts}</td>"
            f"<td class='{colour}'>{label}</td>"
            f"<td>{_badge(sc) if sc in ('SC1', 'SC2', 'SC3', 'NS') else sc}</td>"
            f"<td>{actor}</td><td>{reason}</td></tr>"
        )
    return f"""
<h2 class="section">Safety Gate Log</h2>
<table>
  <tr><th>Time</th><th>Event</th><th>Safety</th><th>Actor</th><th>Reason / Comment</th></tr>
  {rows or "<tr><td colspan='5'>No safety gate events recorded</td></tr>"}
</table>"""


def _ncr_register(ncrs: list[dict]) -> str:
    rows = ""
    for n in ncrs:
        status = n.get("status", "OPEN")
        cls = "open-ncr" if status == "OPEN" else "resolved-ncr"
        rows += (
            f"<tr>"
            f"<td><code>{n.get('ncr_id', '?')[:12]}</code></td>"
            f"<td>{n.get('severity', '?')}</td>"
            f"<td class='{cls}'>{status}</td>"
            f"<td>{n.get('description', '—')[:60]}</td>"
            f"<td>{n.get('raised_by', '?')}</td>"
            f"<td>{_ts(n.get('raised_at', ''))}</td>"
            f"<td>{n.get('resolution', '—')[:50]}</td>"
            f"</tr>"
        )
    return f"""
<h2 class="section">NCR Register ({len(ncrs)} records)</h2>
<table>
  <tr><th>NCR ID</th><th>Severity</th><th>Status</th><th>Description</th>
      <th>Raised By</th><th>Date</th><th>Resolution</th></tr>
  {rows or "<tr><td colspan='7' class='pass'>No NCRs raised — compliant</td></tr>"}
</table>"""


def _chain_certificate(chain_result: dict) -> str:
    valid = chain_result.get("valid", False)
    count = chain_result.get("event_count", 0)
    broken = chain_result.get("broken_links", [])
    cls = "pass" if valid else "fail"
    verdict = "INTEGRITY VERIFIED" if valid else "INTEGRITY FAILURE"
    detail = (
        f"All {count} audit events form an unbroken HMAC-SHA256 chain. "
        "No tampering or data loss detected."
        if valid
        else f"{len(broken)} broken link(s) detected in audit chain. "
        "This document may not be accepted for regulatory submission."
    )
    return f"""
<h2 class="section">Audit Chain Integrity Certificate</h2>
<div class="cert {cls}">
  <div class="cert-title {cls}">{_pass(valid)} — {verdict}</div>
  <p style="margin-top:2mm;font-size:9pt">{detail}</p>
  <p style="margin-top:2mm;font-size:8pt;color:var(--mt)">
    Algorithm: HMAC-SHA256 &nbsp;·&nbsp; Events verified: {count}
    &nbsp;·&nbsp; Broken links: {len(broken)}
  </p>
</div>"""


def _approval_signatures(audit_events: list[dict]) -> str:
    approved = [e for e in audit_events if e.get("event_type") == "safety_gate_approved"]
    rows = ""
    for e in approved:
        d = e.get("detail", {})
        actor = e.get("actor", d.get("approver_upn", "?"))
        sc = d.get("safety_category", "SC2")
        ts = _ts(e.get("timestamp_utc", ""))
        seg = (e.get("segment_id") or "?")[:12] + "…"
        rows += (
            f"<tr><td>{ts}</td><td>{_badge(sc)}</td>"
            f"<td>{actor}</td><td><code>{seg}</code></td>"
            f"<td>{d.get('comments', '—')[:60]}</td></tr>"
        )
    return f"""
<h2 class="section">Approval Signatures ({len(approved)} approvals)</h2>
<table>
  <tr><th>Time</th><th>Safety</th><th>Approver UPN</th><th>Segment</th><th>Comments</th></tr>
  {rows or "<tr><td colspan='5'>No SC2 approvals recorded</td></tr>"}
</table>"""


def _element_provenance(audit_events: list[dict]) -> str:
    el_events = [
        e for e in audit_events if e.get("event_type") in ("element_created", "element_blocked")
    ]
    rows = ""
    for e in el_events[:200]:  # cap at 200 for print-readability
        et = e.get("event_type", "?")
        el_id = (e.get("element_id") or "—")[:18]
        seg_id = (e.get("segment_id") or "—")[:12] + "…"
        d = e.get("detail", {})
        el_type = d.get("element_type", "?")
        sc = d.get("safety_category", "?")
        ts = _ts(e.get("timestamp_utc", ""))
        status = "✓ Created" if et == "element_created" else "🔴 Blocked"
        cls = "" if et == "element_created" else "fail"
        rows += (
            f"<tr><td>{ts}</td><td><code>{el_id}</code></td>"
            f"<td>{el_type}</td>"
            f"<td>{_badge(sc) if sc in ('SC1', 'SC2', 'SC3', 'NS') else sc}</td>"
            f"<td><code>{seg_id}</code></td>"
            f"<td class='{cls}'>{status}</td></tr>"
        )
    return f"""
<h2 class="section">Element Provenance ({len(el_events)} records)</h2>
<table>
  <tr><th>Time</th><th>Element ID</th><th>Type</th><th>Safety</th>
      <th>Source Segment</th><th>Status</th></tr>
  {rows or "<tr><td colspan='6'>No element events recorded</td></tr>"}
</table>"""


# ── Public API ────────────────────────────────────────────────────────────────


def generate_drp(
    session_id: str,
    session: dict,
    audit_events: list[dict],
    ncr_records: list[dict],
    chain_result: dict,
    cde_state: str = "PUBLISHED",
    output_path: str | None = None,
) -> str:
    """
    Build a complete Design Record Package HTML string.

    Parameters
    ----------
    session_id    : UUID of the session
    session       : SiteSession dict (from orchestrator.get_session_summary)
    audit_events  : All audit events for the session (list of dicts)
    ncr_records   : All NCR records for the session (list of dicts)
    chain_result  : Result of audit.verify_chain() — {valid, broken_links, event_count}
    cde_state     : Current CDE state string (default "PUBLISHED")
    output_path   : If provided, write HTML to this path and return it

    Returns
    -------
    HTML string (self-contained, no external dependencies except Google Fonts CDN)
    """
    generated = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
    site = session.get("site_name", "Unknown Site")

    body = "\n".join(
        [
            _cover(session_id, session, cde_state, generated),
            _executive_summary(session, audit_events),
            _safety_gate_log(audit_events),
            _ncr_register(ncr_records),
            _chain_certificate(chain_result),
            _approval_signatures(audit_events),
            _element_provenance(audit_events),
        ]
    )

    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<title>DRP — {site}</title>
<link href="https://fonts.googleapis.com/css2?family=IBM+Plex+Mono&display=swap" rel="stylesheet">
<style>{_CSS}</style>
</head>
<body>
{body}
<div class="footer">
  <span>ScanToBIM Agent &middot; Design Record Package &middot; {site}</span>
  <span>{generated}</span>
</div>
</body>
</html>"""

    if output_path:
        from pathlib import Path

        p = Path(output_path)
        p.write_text(html, encoding="utf-8")

    return html
