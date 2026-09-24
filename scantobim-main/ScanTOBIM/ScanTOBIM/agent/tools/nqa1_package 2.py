"""nqa1_package.py — NQA-1 Subpart 2.7 V&V package generator.

Produces a formal Verification & Validation record that an independent
reviewer can use to qualify the software output for nuclear use.

The package has five required parts (NQA-1 § 401-402, § 702):

  1. Software Identification Record
        Product name / version / git SHA / deps / identity hash.
        Proves "which code produced this result".

  2. Requirements Traceability Matrix
        Every functional / safety requirement mapped to source files
        and passing test IDs. Proves "every claim is tested".

  3. Test Results Summary
        Pass / fail / skip counts from the last pytest run (if an
        xUnit result file is available) plus coverage %.

  4. Session Evidence
        Audit trail (HMAC-chained), chain integrity result, open NCRs,
        safety gate decisions. Proves "what the software did on this job".

  5. Signatures
        Placeholder block for Preparer / Verifier / Approver signatures
        per NQA-1 Rev. 2024 Part I § 302.

Output: (html_bytes, json_dict). Wire through /nqa1-package endpoint.
"""

from __future__ import annotations

import html as _html
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import TYPE_CHECKING

from agent.software_identity import get_software_identity
from agent.tools.requirements_matrix import REQUIREMENTS, verify_matrix_coverage

if TYPE_CHECKING:
    from agent.audit import AuditLedger
    from agent.ncr import NCRService
    from agent.orchestrator import ScanToBIMOrchestrator


# ── Test-results parser (pytest xUnit xml, optional) ─────────────────────────


@dataclass(frozen=True)
class TestResultsSummary:
    total: int
    passed: int
    failed: int
    skipped: int
    errors: int
    duration_s: float
    source_file: str | None  # where the xUnit xml was read from, if any

    @property
    def pass_rate(self) -> float:
        return (self.passed / self.total * 100.0) if self.total else 0.0

    def to_dict(self) -> dict:
        return {
            "total": self.total,
            "passed": self.passed,
            "failed": self.failed,
            "skipped": self.skipped,
            "errors": self.errors,
            "duration_s": round(self.duration_s, 2),
            "pass_rate": round(self.pass_rate, 1),
            "source_file": self.source_file,
        }


def parse_xunit(xml_path: Path) -> TestResultsSummary | None:
    """Parse a pytest-generated JUnit XML file into a summary."""
    if not xml_path.exists():
        return None
    try:
        tree = ET.parse(xml_path)
        root = tree.getroot()
        # Can be testsuite or testsuites; aggregate across
        suites = [root] if root.tag == "testsuite" else list(root.findall("testsuite"))
        if not suites:
            return None
        total = failed = errors = skipped = 0
        duration = 0.0
        for s in suites:
            total += int(s.attrib.get("tests", 0))
            failed += int(s.attrib.get("failures", 0))
            errors += int(s.attrib.get("errors", 0))
            skipped += int(s.attrib.get("skipped", 0))
            duration += float(s.attrib.get("time", 0))
        passed = max(total - failed - errors - skipped, 0)
        return TestResultsSummary(
            total=total,
            passed=passed,
            failed=failed,
            skipped=skipped,
            errors=errors,
            duration_s=duration,
            source_file=str(xml_path),
        )
    except (ET.ParseError, ValueError):
        return None


def _find_xunit() -> Path | None:
    """Look for a JUnit xml in the usual CI locations."""
    root = Path(__file__).resolve().parents[2]  # project root
    for candidate in (
        root / "test-results.xml",
        root / "junit.xml",
        root / "agent" / "test-results.xml",
    ):
        if candidate.exists():
            return candidate
    return None


# ── Package builder ──────────────────────────────────────────────────────────


def build_nqa1_package(
    session_id: str,
    orchestrator: ScanToBIMOrchestrator,
    audit: AuditLedger,
    ncr_service: NCRService | None = None,
    signature_service: NQA1SignatureService | None = None,
) -> tuple[str, dict]:
    """Build the NQA-1 V&V package. Returns (html, json_dict)."""
    if session_id not in orchestrator.sessions:
        raise KeyError(f"Unknown session: {session_id}")

    session = orchestrator.sessions[session_id]
    identity = get_software_identity()
    matrix = verify_matrix_coverage()

    # Test results (best-effort — missing xml is not a hard fail)
    xml_path = _find_xunit()
    results = parse_xunit(xml_path) if xml_path else None

    # Session evidence
    events = audit.get_session_events(session_id)
    chain = audit.verify_chain(session_id)
    ncrs = ncr_service.list_session(session_id) if ncr_service else []

    sc_counts = _safety_breakdown(events)
    provenance = orchestrator._provenance.get(session_id)  # may be None

    # P2.4++ signatures — resolved live if the signature service is provided
    if signature_service is not None:
        sig_status = signature_service.status(session_id)
    else:
        sig_status = {
            "session_id": session_id,
            "fully_signed": False,
            "next_required": "preparer",
            "by_role": {
                "preparer": {
                    "signed": False,
                    "signer_upn": None,
                    "signer_name": None,
                    "timestamp_utc": None,
                },
                "verifier": {
                    "signed": False,
                    "signer_upn": None,
                    "signer_name": None,
                    "timestamp_utc": None,
                },
                "approver": {
                    "signed": False,
                    "signer_upn": None,
                    "signer_name": None,
                    "timestamp_utc": None,
                },
            },
            "chain_valid": True,
            "chain_broken": [],
            "signature_count": 0,
        }

    json_doc = {
        "document": "NQA-1 Subpart 2.7 V&V Package",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "session": {
            "session_id": session_id,
            "site_name": getattr(session, "site_name", None),
        },
        "software_identity": identity.to_dict(),
        "matrix_coverage": matrix.to_dict(),
        "requirements": [r.to_dict() for r in REQUIREMENTS],
        "test_results": results.to_dict() if results else None,
        "provenance": provenance,  # ← NQA-1 determinism record
        "session_evidence": {
            "audit_event_count": len(events),
            "chain": {
                "valid": bool(chain.get("valid")),
                "broken_links": chain.get("broken_links", []),
            },
            "safety_breakdown": sc_counts,
            "open_ncr_count": sum(1 for n in ncrs if not getattr(n, "resolved_at", None)),
        },
        "signatures": sig_status,
    }

    html = _render_html(json_doc)
    return html, json_doc


def _safety_breakdown(events) -> dict[str, int]:
    """Count audit events by type for the safety audit trail section."""
    counts: dict[str, int] = {}
    for e in events:
        t = e.event_type.value if hasattr(e.event_type, "value") else str(e.event_type)
        counts[t] = counts.get(t, 0) + 1
    return counts


# ── HTML rendering ───────────────────────────────────────────────────────────

_GRADE_COLOURS = {
    "A": "#e14d3d",  # red — strictest
    "B": "#f28b24",  # orange
    "C": "#f5b400",  # amber
    "D": "#31a353",  # green
}


def _render_html(doc: dict) -> str:
    identity = doc["software_identity"]
    matrix = doc["matrix_coverage"]
    tests = doc["test_results"]
    session = doc["session"]
    evidence = doc["session_evidence"]
    provenance = doc.get("provenance")
    signatures = doc.get("signatures") or {}

    # Requirements rows
    req_rows = []
    for req in doc["requirements"]:
        grade_colour = _GRADE_COLOURS.get(req["grade"], "#9ca3af")
        src_list = "<br>".join(_html.escape(f) for f in req["source_files"])
        test_list = (
            "<br>".join(f"<code>{_html.escape(t)}</code>" for t in req["test_ids"])
            or '<span style="color:#e14d3d">NONE</span>'
        )
        req_rows.append(f"""
            <tr>
              <td class="mono">{_html.escape(req["req_id"])}</td>
              <td><strong>{_html.escape(req["title"])}</strong><br>
                  <span class="muted">{_html.escape(req["description"])}</span><br>
                  <span class="ref">{_html.escape(req["standard_ref"])}</span>
              </td>
              <td><span class="grade" style="background:{grade_colour}">
                    Grade {req["grade"]}</span></td>
              <td class="mono small">{src_list}</td>
              <td class="mono small">{test_list}</td>
            </tr>
        """)

    # Dependency rows
    dep_rows = (
        "".join(
            f"<tr><td class='mono'>{_html.escape(k)}</td>"
            f"<td class='mono'>{_html.escape(v)}</td></tr>"
            for k, v in sorted(identity["dependencies"].items())
        )
        or "<tr><td colspan='2' class='muted'>(none tracked)</td></tr>"
    )

    # Safety audit breakdown
    sc_rows = (
        "".join(
            f"<tr><td class='mono'>{_html.escape(k)}</td><td class='num'>{v}</td></tr>"
            for k, v in sorted(evidence["safety_breakdown"].items())
        )
        or "<tr><td colspan='2' class='muted'>(no events)</td></tr>"
    )

    test_block = _render_test_block(tests)
    coverage_banner = _render_coverage_banner(matrix)
    provenance_block = _render_provenance_block(provenance)
    signature_block = _render_signature_block(signatures)
    chain_status = evidence["chain"]
    chain_colour = "#31a353" if chain_status["valid"] else "#e14d3d"
    chain_label = "VALID" if chain_status["valid"] else "BROKEN"

    tree_colour = "#31a353" if identity["git_tree_clean"] else "#f5b400"
    tree_label = "CLEAN" if identity["git_tree_clean"] else "DIRTY"

    return f"""<!DOCTYPE html>
<html lang="en"><head>
<meta charset="utf-8">
<title>NQA-1 V&amp;V Package — Session {_html.escape(session["session_id"][:8])}</title>
<style>
  :root {{
    --bg:#0f0f1a; --card:#1a1a2e; --ink:#e8e8f0; --muted:#9092a8;
    --line:#2a2a42; --ok:#31a353; --warn:#f5b400; --err:#e14d3d;
  }}
  body {{ margin:0; padding:40px; font-family:-apple-system,system-ui,sans-serif;
         background:var(--bg); color:var(--ink); }}
  .card {{ max-width:1100px; margin:0 auto 24px; background:var(--card);
           padding:28px 32px; border-radius:12px; border:1px solid var(--line); }}
  h1 {{ margin:0 0 4px; font-size:24px; }}
  h2 {{ margin:0 0 14px; font-size:15px; color:var(--muted);
       text-transform:uppercase; letter-spacing:0.06em; }}
  .sub {{ color:var(--muted); font-size:13px; margin-bottom:24px; }}
  .grid2 {{ display:grid; grid-template-columns:repeat(2, 1fr); gap:14px;
           margin:8px 0; }}
  .grid4 {{ display:grid; grid-template-columns:repeat(4, 1fr); gap:14px;
           margin:8px 0; }}
  .stat {{ background:var(--bg); padding:14px; border-radius:8px;
          border:1px solid var(--line); }}
  .stat .label {{ color:var(--muted); font-size:11px;
                 text-transform:uppercase; letter-spacing:0.05em; }}
  .stat .value {{ font-size:20px; font-weight:600; margin-top:4px; }}
  table {{ width:100%; border-collapse:collapse; margin-top:10px; }}
  th {{ text-align:left; padding:10px 12px; color:var(--muted);
        font-size:11px; text-transform:uppercase;
        border-bottom:1px solid var(--line); }}
  td {{ padding:10px 12px; border-bottom:1px solid var(--line);
       font-size:13px; vertical-align:top; }}
  td.mono  {{ font-family:ui-monospace, Menlo, monospace; font-size:12px; }}
  td.small {{ font-size:11px; }}
  td.num   {{ text-align:right; font-family:ui-monospace, Menlo, monospace; }}
  .muted {{ color:var(--muted); }}
  .ref   {{ color:var(--muted); font-size:11px; font-style:italic; }}
  .grade {{ display:inline-block; padding:3px 10px; border-radius:11px;
           color:#fff; font-size:11px; font-weight:600; }}
  .pill  {{ display:inline-block; padding:2px 10px; border-radius:10px;
           color:#fff; font-size:11px; font-weight:600; }}
  .sig {{ display:grid; grid-template-columns:repeat(3, 1fr); gap:14px;
         margin-top:12px; }}
  .sig-box {{ background:var(--bg); border:1px dashed var(--line);
             padding:18px; border-radius:8px; min-height:90px; }}
  .sig-box .role {{ font-size:11px; text-transform:uppercase;
                   letter-spacing:0.06em; color:var(--muted); }}
  .sig-box .line {{ border-bottom:1px solid var(--line); height:24px;
                   margin:18px 0 6px; }}
  code {{ background:#0b0b15; padding:1px 6px; border-radius:3px;
         font-size:11px; }}
</style>
</head><body>

<div class="card">
  <h1>NQA-1 Verification &amp; Validation Package</h1>
  <div class="sub">
    ASME NQA-1 Subpart 2.7 (Quality Assurance Requirements for Computer Software) ·
    Session <span class="mono">{_html.escape(session["session_id"][:8])}</span> ·
    Generated {_html.escape(doc["generated_at"][:19])}Z
  </div>

  <h2>Section 1 — Software Identification</h2>
  <div class="grid4">
    <div class="stat">
      <div class="label">Product</div>
      <div class="value">{_html.escape(identity["product_name"])}
        v{_html.escape(identity["product_version"])}</div>
    </div>
    <div class="stat">
      <div class="label">Git SHA</div>
      <div class="value mono" style="font-size:14px">
        {_html.escape(identity["git_sha"][:12])}</div>
    </div>
    <div class="stat">
      <div class="label">Working tree</div>
      <div class="value"><span class="pill" style="background:{tree_colour}">
        {tree_label}</span></div>
    </div>
    <div class="stat">
      <div class="label">Identity hash</div>
      <div class="value mono" style="font-size:12px">
        {_html.escape(identity["identity_hash"][:16])}…</div>
    </div>
  </div>

  <table>
    <thead><tr><th>Package</th><th>Version</th></tr></thead>
    <tbody>{dep_rows}</tbody>
  </table>

  <p class="muted" style="font-size:11px; margin-top:10px">
    Python {_html.escape(identity["python_version"])} ·
    {_html.escape(identity["platform_label"])} ·
    {"git tag: " + _html.escape(identity["git_tag"]) if identity["git_tag"] else "no git tag"}
  </p>
</div>

<div class="card">
  <h2>Section 2 — Requirements Traceability Matrix</h2>
  {coverage_banner}
  <table>
    <thead><tr>
      <th>ID</th><th>Requirement</th><th>Grade</th>
      <th>Source files</th><th>Verifying tests</th>
    </tr></thead>
    <tbody>{"".join(req_rows)}</tbody>
  </table>
</div>

<div class="card">
  <h2>Section 3 — Test Results</h2>
  {test_block}
</div>

<div class="card">
  <h2>Section 3b — Determinism / Provenance</h2>
  {provenance_block}
</div>

<div class="card">
  <h2>Section 4 — Session Evidence</h2>
  <div class="grid4">
    <div class="stat">
      <div class="label">Audit events</div>
      <div class="value">{evidence["audit_event_count"]}</div>
    </div>
    <div class="stat">
      <div class="label">Chain integrity</div>
      <div class="value"><span class="pill" style="background:{chain_colour}">
        {chain_label}</span></div>
    </div>
    <div class="stat">
      <div class="label">Open NCRs</div>
      <div class="value">{evidence["open_ncr_count"]}</div>
    </div>
    <div class="stat">
      <div class="label">Broken links</div>
      <div class="value">{len(chain_status["broken_links"])}</div>
    </div>
  </div>

  <table>
    <thead><tr><th>Audit event type</th><th>Count</th></tr></thead>
    <tbody>{sc_rows}</tbody>
  </table>
</div>

<div class="card">
  <h2>Section 5 — Signatures</h2>
  <p class="muted" style="font-size:12px; margin-bottom:10px">
    Per NQA-1 Part I Requirement 3 § 302, this package is not valid for
    qualification use until signed by the Preparer, Verifier, and Approver
    below. Signatures are captured via the SC2 HMAC workflow and chained
    to the package hash — any edit to the V&amp;V package after signing
    breaks chain integrity and the package must be re-signed.
  </p>
  {signature_block}
</div>

<p class="muted" style="text-align:center; font-size:11px">
  ScanToBIM NQA-1 V&amp;V Package schema v1.0
</p>

</body></html>
"""


def _render_coverage_banner(matrix: dict) -> str:
    fully = matrix["fully_traced"]
    colour = "#31a353" if fully else "#e14d3d"
    label = "FULLY TRACED" if fully else "COVERAGE GAP"
    missing = matrix["missing_tests"]

    by_grade = matrix["by_grade"]
    grade_pills = " ".join(
        f'<span class="pill" style="background:{_GRADE_COLOURS.get(g, "#9ca3af")}">'
        f"Grade {g}: {c}</span>"
        for g, c in sorted(by_grade.items())
    )

    missing_block = ""
    if missing:
        missing_block = (
            f'<p style="color:#e14d3d; font-size:12px; margin-top:8px">'
            f"Requirements without verifying tests: "
            f"{', '.join(_html.escape(m) for m in missing)}"
            f"</p>"
        )

    return f"""
      <div style="display:flex; gap:10px; align-items:center; margin-bottom:8px">
        <span class="pill" style="background:{colour}">{label}</span>
        <span class="muted" style="font-size:12px">
          {matrix["total_requirements"]} requirements ·
          {len(matrix["standards_covered"])} standards
        </span>
      </div>
      <div style="margin-bottom:8px">{grade_pills}</div>
      {missing_block}
    """


def _render_provenance_block(provenance: dict | None) -> str:
    if provenance is None:
        return (
            '<p class="muted" style="font-size:12px">'
            "No provenance record — run segmentation or ingest-elements "
            "to populate this section.</p>"
        )

    input_hash = provenance.get("input_hash")
    input_line = (
        f"<code>{_html.escape(input_hash[:16])}…</code> "
        f'<span class="muted">({_html.escape(provenance.get("input_source") or "n/a")})</span>'
        if input_hash
        else f'<span class="muted">not captured '
        f"({_html.escape(provenance.get('input_source') or 'unknown source')})</span>"
    )

    return f"""
      <div class="grid4">
        <div class="stat">
          <div class="label">Random seed</div>
          <div class="value mono">{provenance["random_seed"]}</div>
        </div>
        <div class="stat">
          <div class="label">Output hash (SHA-256)</div>
          <div class="value mono" style="font-size:12px">
            {_html.escape(provenance["output_hash"][:16])}…</div>
        </div>
        <div class="stat">
          <div class="label">Segment count</div>
          <div class="value">{provenance["segment_count"]}</div>
        </div>
        <div class="stat">
          <div class="label">Identity hash</div>
          <div class="value mono" style="font-size:12px">
            {_html.escape(provenance["software_identity_hash"][:16])}…</div>
        </div>
      </div>
      <p class="muted" style="font-size:12px; margin-top:10px">
        Input hash: {input_line}<br>
        Generated: <code>{_html.escape(provenance["generated_at_utc"][:19])}Z</code>
      </p>
      <p class="muted" style="font-size:11px; margin-top:6px; font-style:italic">
        To reproduce: run this software version against the same input file
        with <code>STB_RANDOM_SEED={provenance["random_seed"]}</code>. The
        resulting output hash must match the value above.
      </p>
    """


def _render_signature_block(signatures: dict) -> str:
    """Render Section 5 sig grid from the live signature service status."""
    if not signatures or not signatures.get("by_role"):
        return '<p class="muted" style="font-size:12px">No signatures yet.</p>'

    by_role = signatures["by_role"]
    roles = [
        ("preparer", "Preparer"),
        ("verifier", "Verifier (independent)"),
        ("approver", "Approver"),
    ]

    boxes = []
    for key, label in roles:
        sig = by_role.get(key, {})
        if sig.get("signed"):
            upn = _html.escape(sig.get("signer_upn") or "")
            name = _html.escape(sig.get("signer_name") or "")
            ts = _html.escape((sig.get("timestamp_utc") or "")[:19])
            boxes.append(f"""
              <div class="sig-box" style="border-color:#31a353">
                <div class="role">{label}</div>
                <div style="padding:14px 0; font-size:13px; font-weight:600">
                  ✓ {name}</div>
                <div class="muted" style="font-size:11px">
                  <code>{upn}</code> · {ts}Z
                </div>
              </div>
            """)
        else:
            boxes.append(f"""
              <div class="sig-box">
                <div class="role">{label}</div>
                <div class="line"></div>
                <div class="muted" style="font-size:11px">
                  Name · UPN · Date — unsigned</div>
              </div>
            """)

    chain_valid = signatures.get("chain_valid", True)
    fully_signed = signatures.get("fully_signed", False)

    if fully_signed and chain_valid:
        banner = (
            '<p style="color:#31a353; font-weight:600; margin-top:12px; '
            'font-size:13px">✓ Package is fully signed and chain-valid.</p>'
        )
    elif not chain_valid:
        broken_count = len(signatures.get("chain_broken") or [])
        banner = (
            f'<p style="color:#e14d3d; font-weight:600; margin-top:12px; '
            f'font-size:13px">⚠ Chain integrity broken — '
            f"{broken_count} signature(s) fail verification.</p>"
        )
    else:
        next_role = signatures.get("next_required") or "—"
        banner = (
            f'<p class="muted" style="margin-top:12px; font-size:12px">'
            f"Next required signature: <strong>{_html.escape(next_role)}</strong>"
            f"</p>"
        )

    return f'<div class="sig">{"".join(boxes)}</div>{banner}'


def _render_test_block(tests) -> str:
    if tests is None:
        return (
            '<p class="muted" style="font-size:12px">'
            "No xUnit test-results file found. Run "
            "<code>pytest agent/tests/ --junitxml=test-results.xml</code> "
            "before generating the package to include test evidence.</p>"
        )

    pass_colour = "#31a353" if tests["failed"] == 0 and tests["errors"] == 0 else "#e14d3d"
    pass_label = f"{tests['pass_rate']:.1f}% PASS"

    return f"""
      <div class="grid4">
        <div class="stat">
          <div class="label">Status</div>
          <div class="value"><span class="pill"
            style="background:{pass_colour}">{pass_label}</span></div>
        </div>
        <div class="stat">
          <div class="label">Passed</div>
          <div class="value">{tests["passed"]} / {tests["total"]}</div>
        </div>
        <div class="stat">
          <div class="label">Failed / Error</div>
          <div class="value">{tests["failed"]} / {tests["errors"]}</div>
        </div>
        <div class="stat">
          <div class="label">Duration</div>
          <div class="value">{tests["duration_s"]:.1f} s</div>
        </div>
      </div>
      <p class="muted" style="font-size:11px; margin-top:10px">
        Source: <code>{_html.escape(tests["source_file"] or "n/a")}</code>
      </p>
    """
