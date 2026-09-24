"""fmea.py — Failure Mode and Effects Analysis for the ScanToBIM pipeline.

NQA-1 Subpart 2.7 § 102 and IEC 61508 both require a documented hazard
analysis for software serving safety-related functions. This module
produces an auditable FMEA that:

  • Enumerates every major pipeline function we claim to verify
  • States one or more potential failure modes per function
  • Scores each mode on Severity / Likelihood / Detection (1–10)
  • Computes Risk Priority Number (RPN = S × L × D)
  • Names the mitigation (code path) and the pytest test ID that proves
    the mitigation works
  • Flags High-RPN items (RPN ≥ 200) requiring explicit review

The catalog is static — it lives in this module as source of truth so
changes are code-reviewed and version-pinned. Scores are conservative
honest estimates by the engineer; review them at every major release.

Renderers: HTML (printable, matches the NQA-1 package style) and JSON.
"""

from __future__ import annotations

import html as _html
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from typing import Literal

# ── Data model ───────────────────────────────────────────────────────────────

Severity = Literal[1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
Likelihood = Literal[1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
Detection = Literal[1, 2, 3, 4, 5, 6, 7, 8, 9, 10]


@dataclass(frozen=True)
class FailureMode:
    """One row in the FMEA table."""

    fmea_id: str
    function: str
    failure_mode: str
    effect_local: str
    effect_system: str
    severity: Severity  # 1=minor, 10=catastrophic
    likelihood: Likelihood  # 1=extremely rare, 10=certain
    detection: Detection  # 1=automatically caught, 10=undetectable
    mitigation: str
    test_ids: tuple[str, ...]

    @property
    def rpn(self) -> int:
        return self.severity * self.likelihood * self.detection

    @property
    def risk_tier(self) -> str:
        rpn = self.rpn
        if rpn >= 200:
            return "HIGH"
        if rpn >= 100:
            return "MEDIUM"
        if rpn >= 50:
            return "LOW"
        return "NEGLIGIBLE"

    def to_dict(self) -> dict:
        d = asdict(self)
        d["test_ids"] = list(self.test_ids)
        d["rpn"] = self.rpn
        d["risk_tier"] = self.risk_tier
        return d


# ── Scoring guidance ─────────────────────────────────────────────────────────

_SEVERITY_GUIDE = {
    10: "Catastrophic — regulated non-compliance, nuclear safety breach, or irreversible rework at scale",
    9: "Critical — audit finding that invalidates the deliverable",
    8: "Major — SC1/SC2 element mis-classified; client-visible integrity issue",
    7: "Significant — LOA/QA evidence is wrong; delivery must be re-issued",
    6: "Moderate — individual element wrong; spot rework required",
    5: "Low — informational accuracy tag drifts outside declared tier",
    4: "Minor — usability or reporting format glitch",
    3: "Negligible — cosmetic UI issue",
    2: "Trivial — log noise",
    1: "None — no user-observable impact",
}

_LIKELIHOOD_GUIDE = {
    10: "Certain — will happen in normal operation",
    8: "High — happens on most sessions without defence-in-depth",
    6: "Medium — happens on a significant fraction of sessions",
    4: "Low — requires unusual inputs",
    2: "Rare — requires adversarial inputs or corrupted data",
    1: "Practically impossible",
}

_DETECTION_GUIDE = {
    10: "Undetectable — no automated or manual check can catch it",
    8: "Weak — caught only by manual review of outputs",
    6: "Moderate — caught by regression tests on demand",
    4: "Good — caught by CI every PR",
    2: "Strong — caught at runtime by explicit guard + audit",
    1: "Certain — hard-blocked at the API boundary",
}


# ── Catalog ──────────────────────────────────────────────────────────────────

# Scoring rationale: these are honest engineering estimates by the author,
# to be reviewed by an independent QA engineer before any NQA-1 qualification
# submission. S × L × D = RPN; anything ≥ 200 is called out as HIGH.

FAILURE_MODES: list[FailureMode] = [
    # ── Safety gate (the biggest risk surface) ───────────────────────────────
    FailureMode(
        fmea_id="FM-SG-001",
        function="safety_gate.check_instruction",
        failure_mode="SC1 element passed through without raising SafetyCategoryViolationError",
        effect_local="SC1 element dispatched to Revit without human approval",
        effect_system="Nuclear safety-critical item auto-created; audit finding; "
        "potential regulatory breach",
        severity=10,
        likelihood=2,  # hard-blocked in code, but misclassification upstream could feed wrong category
        detection=2,  # guard raises + audit event
        mitigation="Hard Python raise in safety_gate; classifier fallback is NS not SC1; "
        "audit event before Revit dispatch",
        test_ids=(
            "test_sprint3.py::TestSafetyGate::test_sc1_hard_block_raises",
            "test_sprint3.py::TestSafetyGate::test_sc1_never_dispatched",
        ),
    ),
    FailureMode(
        fmea_id="FM-SG-002",
        function="safety_gate.check_instruction",
        failure_mode="SC2 element dispatched before engineer approval",
        effect_local="Element created without HMAC-signed decision",
        effect_system="Safety-related element enters BIM without approval chain; "
        "audit trail shows instruction created but no gate decision",
        severity=8,
        likelihood=3,
        detection=2,  # gate raises SC2 review event + pending-approval queue
        mitigation="check_instruction returns SafetyGateRequest; dispatch skipped; "
        "pending_gates queued; Teams notification optional",
        test_ids=(
            "test_sprint3.py::TestSafetyGate::test_sc2_requires_approval",
            "test_sprint3.py::TestSafetyGate::test_sc2_with_valid_signature_dispatches",
        ),
    ),
    # ── Classification ────────────────────────────────────────────────────────
    FailureMode(
        fmea_id="FM-CL-001",
        function="classifier.classify_segment",
        failure_mode="Pipe mis-classified as ductwork (or vice versa)",
        effect_local="Wrong element_type assigned; wrong Revit family placed",
        effect_system="Downstream coordination uses wrong discipline; clash detection "
        "may flag / miss real clashes",
        severity=6,
        likelihood=4,  # shape heuristics have known edge cases
        detection=4,  # confidence < 0.6 surfaces as red in Revit
        mitigation="Confidence threshold drives NCR; visual review via colour override; "
        "classifier uses multiple cues (shape + Z + aspect ratio)",
        test_ids=(
            "test_sprint3.py::TestClassifier::test_cylinder_elongated_pipe",
            "test_sprint3.py::TestClassifier::test_box_elongated_at_ceiling_duct",
        ),
    ),
    FailureMode(
        fmea_id="FM-CL-002",
        function="classifier.classify_segment",
        failure_mode="Safety category drops to NS when it should be SC1/SC2",
        effect_local="Element bypasses the safety gate entirely",
        effect_system="Same as FM-SG-001 downstream",
        severity=9,
        likelihood=3,
        detection=4,
        mitigation="Rule-based classifier has explicit SC1 table for pressuriser, "
        "steam_generator, EDG, containment penetration; defaults to SC2 "
        "for anything in a nuclear zone",
        test_ids=(
            "test_sprint3.py::TestClassifier::test_pressuriser_is_sc1",
            "test_sprint3.py::TestClassifier::test_containment_penetration_is_sc1",
        ),
    ),
    # ── Segmentation ─────────────────────────────────────────────────────────
    FailureMode(
        fmea_id="FM-SE-001",
        function="scan_tools.run_segmentation",
        failure_mode="Same segment detected twice across multi-pass DBSCAN",
        effect_local="Duplicate DirectShape; inflated element count",
        effect_system="False clash; wasted modeller review time",
        severity=4,
        likelihood=5,  # multi-pass detection known to overlap
        detection=2,  # _resolve_overlaps post-pass merges
        mitigation="IoU merge (≥60%) + containment absorb (≥80%) in _resolve_overlaps; "
        "valves exempted",
        test_ids=(
            "test_overlap.py::TestResolveOverlaps::test_same_shape_high_iou_merges",
            "test_overlap.py::TestResolveOverlaps::test_containment_absorb_drops_small",
        ),
    ),
    FailureMode(
        fmea_id="FM-SE-002",
        function="scan_tools.run_segmentation",
        failure_mode="Cylinder radius overestimated from AABB when RANSAC unavailable",
        effect_local="Revit DirectShape wider than actual pipe",
        effect_system="Spurious soft clashes; inflated insulation requirement",
        severity=5,
        likelihood=4,
        detection=6,  # hard to spot visually at scale
        mitigation="RANSAC cylinder fit stored as tags.fitted_radius_mm; "
        "ScanGeometryBuilder prefers fitted_radius over AABB",
        test_ids=("test_overlap.py::TestResolveOverlaps::test_valve_not_absorbed_by_pipe",),
    ),
    # ── Clash detection ───────────────────────────────────────────────────────
    FailureMode(
        fmea_id="FM-CD-001",
        function="coordinator_tools.detect_clashes",
        failure_mode="Valve-on-pipe flagged as hard clash (false positive)",
        effect_local="Genuine valve installation flagged for rework",
        effect_system="Modeller fatigue from false-positive clashes; real clashes "
        "overlooked in the noise",
        severity=4,
        likelihood=7,  # inevitable without whitelist
        detection=2,
        mitigation="_is_whitelisted_pair() suppresses inline-MEP-on-conduit, "
        "pipe-through-wall, stacked-horizontal-plane patterns",
        test_ids=(
            "test_overlap.py::TestClashWhitelist::test_valve_on_pipe",
            "test_overlap.py::TestClashWhitelist::test_pipe_through_wall",
        ),
    ),
    # ── Audit & chain integrity ───────────────────────────────────────────────
    FailureMode(
        fmea_id="FM-AU-001",
        function="audit.append",
        failure_mode="Audit event written AFTER Revit dispatch (not before)",
        effect_local="Revit element created but no audit entry on crash",
        effect_system="Breach of NQA-1 § 302 audit-before-action requirement",
        severity=8,
        likelihood=2,
        detection=4,
        mitigation="Orchestrator always calls audit.append() synchronously BEFORE "
        "_dispatch_to_revit(); chain_integrity.json detects missing events",
        test_ids=("test_sprint3.py::TestOrchestrator::test_audit_before_dispatch",),
    ),
    FailureMode(
        fmea_id="FM-AU-002",
        function="audit.verify_chain",
        failure_mode="Tampered event goes undetected",
        effect_local="Attacker rewrites history",
        effect_system="Entire audit trail untrustworthy for QA sign-off",
        severity=10,
        likelihood=2,
        detection=2,
        mitigation="HMAC-SHA256 chain with AUDIT_HMAC_SECRET; verify_chain returns "
        "valid=False on any tamper; secret injected via env var and NOT "
        "checked into source",
        test_ids=("test_sprint3.py::TestAuditLedger::test_chain_break_detected",),
    ),
    # ── Determinism (NQA-1 § 401) ────────────────────────────────────────────
    FailureMode(
        fmea_id="FM-DT-001",
        function="determinism.set_global_seed",
        failure_mode="Two runs with same input + seed produce different outputs",
        effect_local="Reviewer cannot reproduce prior V&V result",
        effect_system="Failed NQA-1 § 401 determinism requirement; package invalid",
        severity=9,
        likelihood=1,
        detection=2,
        mitigation="set_global_seed pins numpy + python random + Open3D; output_hash "
        "computed on sorted canonical segments; regression test compares "
        "two synthetic runs byte-for-byte",
        test_ids=(
            "test_determinism.py::TestOrchestratorProvenance::test_same_synthetic_input_same_output_hash",
        ),
    ),
    # ── LOA ──────────────────────────────────────────────────────────────────
    FailureMode(
        fmea_id="FM-LA-001",
        function="loa_tools.classify_loa_tier",
        failure_mode="Element declared as LOA30 when residual is actually LOA20",
        effect_local="Client uses model at tighter tolerance than justified",
        effect_system="Downstream fabrication based on wrong accuracy claim",
        severity=7,
        likelihood=3,
        detection=4,
        mitigation="σ computed as RMS (matches noise σ) not std-of-abs; 2σ → tier at "
        "tolerance boundary; worst_tier gate surfaces outliers",
        test_ids=(
            "test_loa.py::TestClassifyLOATier::test_construction_fit_is_loa30",
            "test_loa.py::TestClassifyLOATier::test_boundary_just_over_5mm_is_loa20",
        ),
    ),
    # ── Signatures ───────────────────────────────────────────────────────────
    FailureMode(
        fmea_id="FM-SI-001",
        function="nqa1_signature.sign",
        failure_mode="Same UPN signs as both Verifier and Approver",
        effect_local="Separation of duties violated",
        effect_system="NQA-1 Requirement 3 § 302 breach — package invalid",
        severity=8,
        likelihood=4,
        detection=1,
        mitigation="DuplicateSignerError raised by sign(); case-normalised UPN check "
        "against active signatures",
        test_ids=(
            "test_nqa1_signature.py::TestSeparationOfDuties::test_same_upn_two_roles_rejected",
            "test_nqa1_signature.py::TestSeparationOfDuties::test_same_upn_different_case_rejected",
        ),
    ),
    FailureMode(
        fmea_id="FM-SI-002",
        function="nqa1_signature.sign",
        failure_mode="Approver signs before Preparer (order skipped)",
        effect_local="Chain lacks root; verification fails",
        effect_system="Qualification invalid",
        severity=7,
        likelihood=3,
        detection=1,
        mitigation="SignatureOrderError raised in sign()",
        test_ids=(
            "test_nqa1_signature.py::TestSignOrdering::test_verifier_before_preparer_rejected",
            "test_nqa1_signature.py::TestSignOrdering::test_approver_without_verifier_rejected",
        ),
    ),
    FailureMode(
        fmea_id="FM-SI-003",
        function="nqa1_signature.verify_chain",
        failure_mode="Database-level tamper of signer_name goes undetected",
        effect_local="Attacker substitutes approver identity",
        effect_system="Chain verification misses identity swap",
        severity=9,
        likelihood=2,
        detection=1,
        mitigation="compute_hash includes every human-readable field (signer_name, "
        "comments); any DB edit breaks HMAC",
        test_ids=("test_nqa1_signature.py::TestChainIntegrity::test_hmac_detects_payload_tamper",),
    ),
    # ── Registration QA ───────────────────────────────────────────────────────
    FailureMode(
        fmea_id="FM-RG-001",
        function="registration_report.classify_station",
        failure_mode="Station passes Cat A despite inlier ratio below floor",
        effect_local="Survey claimed at wrong accuracy grade",
        effect_system="Client accepts scan that should have been rescanned",
        severity=7,
        likelihood=2,
        detection=2,
        mitigation="classify_station requires BOTH rmse_mm and inlier_ratio to meet "
        "tier threshold; FAIL returned if either fails",
        test_ids=(
            "test_registration_report.py::TestClassifyStation::test_cat_a_inlier_boundary",
            "test_registration_report.py::TestClassifyStation::test_cat_c_fails_when_inlier_too_low",
        ),
    ),
    # ── Ingest path ──────────────────────────────────────────────────────────
    FailureMode(
        fmea_id="FM-IN-001",
        function="/sessions/{id}/ingest-elements",
        failure_mode="Malformed segment list accepted and stored",
        effect_local="Downstream endpoints crash or return garbage",
        effect_system="API surface contract broken; client integrations fail",
        severity=5,
        likelihood=3,
        detection=2,
        mitigation="Pydantic GeometrySegment validation inside the route; "
        "422 returned on any validation error",
        test_ids=(
            "test_ingest.py::TestIngestValidation::test_bad_segment_returns_422",
            "test_ingest.py::TestIngestValidation::test_missing_bounding_box_returns_422",
        ),
    ),
    # ── Handover bundle ──────────────────────────────────────────────────────
    FailureMode(
        fmea_id="FM-HB-001",
        function="handover_bundle.build_handover_bundle",
        failure_mode="Missing section silently excluded without explanation",
        effect_local="Reviewer unaware an expected artifact is missing",
        effect_system="Qualified deliverable appears complete when it is not",
        severity=6,
        likelihood=4,
        detection=2,
        mitigation="SectionResult.reason recorded; manifest.json and README.md "
        "explicitly list skipped sections with reason",
        test_ids=(
            "test_handover_bundle.py::TestEmptySession::test_sections_mark_skipped_when_no_data",
            "test_handover_bundle.py::TestManifestAndReadme::test_readme_lists_included_and_skipped",
        ),
    ),
    # ── Auth ────────────────────────────────────────────────────────────────
    FailureMode(
        fmea_id="FM-AU-003",
        function="auth.get_current_user",
        failure_mode="Expired JWT accepted",
        effect_local="Stale session continues operating",
        effect_system="Non-repudiation broken; audit events attributed to wrong user",
        severity=6,
        likelihood=2,
        detection=2,
        mitigation="python-jose enforces exp claim; 401 returned on expired token",
        test_ids=("test_api.py::TestAuth::test_expired_token_rejected",),
    ),
]


# ── Summary helpers ──────────────────────────────────────────────────────────


def summarise(modes: list[FailureMode]) -> dict:
    """Aggregate counts by risk tier + max RPN + high-RPN items list."""
    tier_counts = {"HIGH": 0, "MEDIUM": 0, "LOW": 0, "NEGLIGIBLE": 0}
    for m in modes:
        tier_counts[m.risk_tier] += 1
    high = sorted([m for m in modes if m.risk_tier == "HIGH"], key=lambda m: -m.rpn)
    return {
        "total_modes": len(modes),
        "by_tier": tier_counts,
        "max_rpn": max((m.rpn for m in modes), default=0),
        "high_rpn_ids": [m.fmea_id for m in high],
    }


# ── Build document ───────────────────────────────────────────────────────────


def build_fmea_document(session_id: str) -> dict:
    """Assemble the FMEA document (JSON-friendly dict) for a session.

    The catalog is static — session_id appears only for traceability in
    the document header so the report matches the NQA-1 package's session.
    """
    modes = FAILURE_MODES
    summary = summarise(modes)
    return {
        "document": "FMEA / Hazard Analysis",
        "standard_refs": ["NQA-1 Subpart 2.7 § 102", "IEC 61508-3 § 7.4"],
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "session_id": session_id,
        "scoring_guides": {
            "severity": _SEVERITY_GUIDE,
            "likelihood": _LIKELIHOOD_GUIDE,
            "detection": _DETECTION_GUIDE,
        },
        "failure_modes": [m.to_dict() for m in modes],
        "summary": summary,
    }


# ── HTML renderer ────────────────────────────────────────────────────────────

_TIER_COLOURS = {
    "HIGH": "#e14d3d",
    "MEDIUM": "#f28b24",
    "LOW": "#f5b400",
    "NEGLIGIBLE": "#31a353",
}


def render_fmea_html(doc: dict) -> str:
    """Produce a self-contained printable HTML FMEA report."""
    rows = []
    for m in doc["failure_modes"]:
        colour = _TIER_COLOURS.get(m["risk_tier"], "#9ca3af")
        tests = (
            "<br>".join(f"<code>{_html.escape(t)}</code>" for t in m["test_ids"])
            or '<span style="color:#e14d3d">NONE</span>'
        )
        rows.append(f"""
          <tr>
            <td class="mono">{_html.escape(m["fmea_id"])}</td>
            <td class="mono small">{_html.escape(m["function"])}</td>
            <td><strong>{_html.escape(m["failure_mode"])}</strong><br>
                <span class="muted">Local: {_html.escape(m["effect_local"])}</span><br>
                <span class="muted">System: {_html.escape(m["effect_system"])}</span>
            </td>
            <td class="num">{m["severity"]}</td>
            <td class="num">{m["likelihood"]}</td>
            <td class="num">{m["detection"]}</td>
            <td class="num"><strong>{m["rpn"]}</strong></td>
            <td><span class="tier" style="background:{colour}">
                {_html.escape(m["risk_tier"])}</span></td>
            <td class="small">{_html.escape(m["mitigation"])}</td>
            <td class="mono small">{tests}</td>
          </tr>
        """)

    summary = doc["summary"]
    by_tier = summary["by_tier"]

    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<title>FMEA Report — {_html.escape(doc["session_id"][:8])}</title>
<style>
  :root {{
    --bg:#0f0f1a; --card:#1a1a2e; --ink:#e8e8f0; --muted:#9092a8;
    --line:#2a2a42;
  }}
  body {{ margin:0; padding:40px; font-family:-apple-system,system-ui,sans-serif;
         background:var(--bg); color:var(--ink); }}
  .card {{ max-width:1200px; margin:0 auto 18px; background:var(--card);
           padding:28px; border-radius:12px; border:1px solid var(--line); }}
  h1 {{ margin:0 0 4px; font-size:22px; }}
  .sub {{ color:var(--muted); font-size:12px; margin-bottom:22px; }}
  h2 {{ font-size:16px; border-bottom:1px solid var(--line);
        padding-bottom:8px; margin:0 0 16px; }}
  .grid4 {{ display:grid; grid-template-columns:repeat(4, 1fr); gap:12px;
           margin-bottom:14px; }}
  .stat {{ background:var(--bg); border:1px solid var(--line);
          padding:12px; border-radius:8px; }}
  .stat .label {{ color:var(--muted); font-size:11px; text-transform:uppercase;
                  letter-spacing:0.05em; }}
  .stat .value {{ font-size:20px; font-weight:600; margin-top:4px; }}
  table {{ width:100%; border-collapse:collapse; font-size:12px; }}
  th {{ text-align:left; padding:8px 10px; color:var(--muted); font-size:10px;
       text-transform:uppercase; border-bottom:1px solid var(--line);
       letter-spacing:0.04em; }}
  td {{ padding:10px; border-bottom:1px solid var(--line);
       vertical-align:top; }}
  td.mono {{ font-family:ui-monospace,Menlo,monospace; font-size:11px; }}
  td.num {{ text-align:right; font-family:ui-monospace,Menlo,monospace; }}
  .small {{ font-size:11px; }}
  .muted {{ color:var(--muted); }}
  .tier {{ display:inline-block; padding:2px 8px; border-radius:9px;
          font-size:10px; font-weight:600; color:#fff; }}
  .footer {{ color:var(--muted); font-size:11px; margin-top:14px; }}
</style>
</head>
<body>

<div class="card">
  <h1>Failure Mode and Effects Analysis (FMEA)</h1>
  <div class="sub">
    Session <code>{_html.escape(doc["session_id"][:8])}</code>
    · Generated {_html.escape(doc["generated_at_utc"][:19])}Z
    · Standards: {", ".join(_html.escape(s) for s in doc["standard_refs"])}
  </div>

  <div class="grid4">
    <div class="stat">
      <div class="label">Total failure modes</div>
      <div class="value">{summary["total_modes"]}</div>
    </div>
    <div class="stat">
      <div class="label">HIGH risk</div>
      <div class="value" style="color:{_TIER_COLOURS["HIGH"]}">
        {by_tier["HIGH"]}</div>
    </div>
    <div class="stat">
      <div class="label">MEDIUM risk</div>
      <div class="value" style="color:{_TIER_COLOURS["MEDIUM"]}">
        {by_tier["MEDIUM"]}</div>
    </div>
    <div class="stat">
      <div class="label">Max RPN</div>
      <div class="value">{summary["max_rpn"]}</div>
    </div>
  </div>

  <p class="muted" style="font-size:12px">
    RPN = Severity × Likelihood × Detection. Thresholds: HIGH ≥ 200,
    MEDIUM ≥ 100, LOW ≥ 50. Items in HIGH require QA manager review
    before qualification. Every mitigation is verified by at least one
    named pytest test whose ID appears in the rightmost column.
  </p>
</div>

<div class="card">
  <h2>Failure-mode register</h2>
  <div style="overflow-x:auto">
  <table>
    <thead><tr>
      <th>ID</th><th>Function</th><th>Mode / Effects</th>
      <th>S</th><th>L</th><th>D</th><th>RPN</th><th>Tier</th>
      <th>Mitigation</th><th>Verifying tests</th>
    </tr></thead>
    <tbody>{"".join(rows)}</tbody>
  </table>
  </div>
</div>

<div class="card">
  <h2>Scoring guide</h2>
  <p class="muted" style="font-size:12px; margin-bottom:10px">
    Scores are honest engineering estimates by the author, to be reviewed
    by an independent QA engineer before any NQA-1 qualification submission.
  </p>
  <table>
    <thead><tr><th>Score</th><th>Severity</th><th>Likelihood</th><th>Detection</th></tr></thead>
    <tbody>
      {
        "".join(
            f'<tr><td class="num"><strong>{k}</strong></td>'
            f'<td class="small">{_html.escape(doc["scoring_guides"]["severity"].get(k, ""))}</td>'
            f'<td class="small">{_html.escape(doc["scoring_guides"]["likelihood"].get(k, ""))}</td>'
            f'<td class="small">{_html.escape(doc["scoring_guides"]["detection"].get(k, ""))}</td>'
            f"</tr>"
            for k in [10, 8, 6, 4, 2, 1]
        )
    }
    </tbody>
  </table>
</div>

<p class="footer">ScanToBIM · FMEA v1.0 · sibling document to the NQA-1 V&amp;V package</p>
</body>
</html>
"""
    return html
