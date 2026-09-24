"""
checker_tools.py  —  P3 Checker Agent (READ-ONLY)
──────────────────────────────────────────────────
6 read-only tools for the Checker Agent.

CRITICAL: The Checker Agent MUST NOT create or modify any element.
          These tools are intentionally limited to verification and reporting.
          Any attempt to pass a write operation through checker_tools
          should raise AssertionError at the call site.

Tools:
  1. verify_element_parameters   — check 4 shared params are set
  2. verify_audit_chain          — HMAC integrity check
  3. check_safety_classifications — SC1 never created, SC2 all resolved
  4. list_open_ncrs              — list unresolved NCRs
  5. generate_verification_report — full session check summary
  6. check_cde_prerequisites      — can we transition CDE state?
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import structlog

from agent.models import SafetyCategory

if TYPE_CHECKING:
    from agent.audit import AuditLedger
    from agent.ncr import NCRService
    from agent.orchestrator import ScanToBIMOrchestrator
    from agent.safety_gate import SafetyGateService

logger = structlog.get_logger()


CHECKER_ACTOR = "checker-agent"


# ── Tool 1: Verify element parameters ────────────────────────────────────────


def verify_element_parameters(
    session_id: str,
    audit: AuditLedger,
) -> dict:
    """
    Verifies that every created element has audit events with the 4 required
    compliance parameter values (safety_category, element_type, discipline, zone_id).

    Returns {verified: int, missing_params: list[element_id], pass: bool}
    """
    events = audit.get_session_events(session_id)
    created_events = [e for e in events if e.event_type.value == "element_created"]

    missing = []
    for ev in created_events:
        d = ev.detail
        required = ["element_type", "discipline", "safety_category"]
        if not all(d.get(k) for k in required):
            missing.append(ev.element_id or ev.event_id)

    result = {
        "verified": len(created_events),
        "missing_params": missing,
        "pass": len(missing) == 0,
    }
    logger.info("params_verified", session_id=session_id[:12], **result)
    return result


# ── Tool 2: Verify HMAC audit chain ──────────────────────────────────────────


def verify_audit_chain(
    session_id: str,
    audit: AuditLedger,
) -> dict:
    """
    Runs full HMAC-SHA256 chain verification.
    Returns audit.verify_chain() result dict: {valid, broken_links, event_count}
    """
    result = audit.verify_chain(session_id)
    logger.info(
        "chain_verified",
        session_id=session_id[:12],
        valid=result.get("valid"),
        event_count=result.get("event_count"),
        broken_links=len(result.get("broken_links", [])),
    )
    return result


# ── Tool 3: Check safety classification compliance ────────────────────────────


def check_safety_classifications(
    session_id: str,
    orchestrator: ScanToBIMOrchestrator,
    safety_gate: SafetyGateService,
) -> dict:
    """
    Verifies the three safety rules:
      1. No SC1 element was ever created (zero element_created events with SC1)
      2. All SC2 gates are resolved (approved or rejected — none still pending)
      3. Every SC1 segment has a corresponding element_blocked audit event

    Returns {sc1_never_created, sc2_all_resolved, sc1_blocked_events, issues: list}
    """

    audit = orchestrator.audit
    events = audit.get_session_events(session_id)

    # Rule 1 — no SC1 element_created
    sc1_created = [
        e
        for e in events
        if e.event_type.value == "element_created" and e.detail.get("safety_category") == "SC1"
    ]
    sc1_never_created = len(sc1_created) == 0

    # Rule 2 — all SC2 gates resolved
    pending_gates = safety_gate.get_pending_gates(session_id)
    sc2_pending = [g for g in pending_gates if g.safety_category == SafetyCategory.SC2]
    sc2_all_resolved = len(sc2_pending) == 0

    # Rule 3 — each SC1 segment has a blocked event
    sc1_blocked = [
        e
        for e in events
        if e.event_type.value == "element_blocked" and e.detail.get("safety_category") == "SC1"
    ]

    issues = []
    if not sc1_never_created:
        issues.append(f"SC1 element(s) were created: {[e.element_id for e in sc1_created]}")
    if not sc2_all_resolved:
        issues.append(f"SC2 gates still pending: {[g.gate_id[:8] for g in sc2_pending]}")

    result = {
        "sc1_never_created": sc1_never_created,
        "sc2_all_resolved": sc2_all_resolved,
        "sc1_blocked_events": len(sc1_blocked),
        "sc2_pending_count": len(sc2_pending),
        "issues": issues,
        "pass": sc1_never_created and sc2_all_resolved,
    }
    logger.info("safety_classifications_checked", session_id=session_id[:12], pass_=result["pass"])
    return result


# ── Tool 4: List open NCRs ────────────────────────────────────────────────────


def list_open_ncrs(
    session_id: str,
    ncr_service: NCRService,
) -> dict:
    """
    Returns all open (unresolved) NCRs for the session.
    {open_count: int, ncrs: list[dict]}
    """
    open_ncrs = ncr_service.list_open(session_id)
    return {
        "open_count": len(open_ncrs),
        "ncrs": [n.model_dump() for n in open_ncrs],
    }


# ── Tool 5: Generate full verification report ─────────────────────────────────


def generate_verification_report(
    session_id: str,
    orchestrator: ScanToBIMOrchestrator,
    safety_gate: SafetyGateService,
    ncr_service: NCRService,
) -> dict:
    """
    Runs all checker tools and assembles a single verification report dict.
    Suitable for logging, API response, or DRP section content.

    Returns {session_id, checks: {params, chain, safety, ncrs}, overall_pass}
    """
    audit = orchestrator.audit

    params_check = verify_element_parameters(session_id, audit)
    chain_check = verify_audit_chain(session_id, audit)
    safety_check = check_safety_classifications(session_id, orchestrator, safety_gate)
    ncr_check = list_open_ncrs(session_id, ncr_service)

    overall_pass = (
        params_check["pass"]
        and chain_check.get("valid", False)
        and safety_check["pass"]
        and ncr_check["open_count"] == 0
    )

    report = {
        "session_id": session_id,
        "checks": {
            "element_parameters": params_check,
            "audit_chain": chain_check,
            "safety_compliance": safety_check,
            "ncr_status": ncr_check,
        },
        "overall_pass": overall_pass,
        "actor": CHECKER_ACTOR,
    }

    logger.info(
        "verification_report_generated",
        session_id=session_id[:12],
        overall_pass=overall_pass,
    )
    return report


# ── Tool 6: Check CDE prerequisites ──────────────────────────────────────────


def check_cde_prerequisites(
    session_id: str,
    target_state: str,
    orchestrator: ScanToBIMOrchestrator,
    safety_gate: SafetyGateService,
    ncr_service: NCRService,
) -> dict:
    """
    Pre-flight check before a CDE state transition.

    WIP → SHARED:
      - All instructions have been executed (no pending)

    SHARED → PUBLISHED:
      - No open NCRs
      - All SC2 gates resolved
      - Audit chain valid

    Returns {can_transition, target_state, blockers: list[str]}
    """
    blockers = []
    target_state = target_state.upper()

    instructions = orchestrator._instructions.get(session_id, [])
    results = orchestrator._results.get(session_id, [])
    pending = len(instructions) - len(results)

    if target_state == "SHARED":
        if pending > 0:
            blockers.append(f"{pending} instructions still pending execution")

    elif target_state == "PUBLISHED":
        # 1. Open NCRs
        open_ncrs = ncr_service.list_open(session_id)
        if open_ncrs:
            blockers.append(f"{len(open_ncrs)} open NCR(s) must be resolved before publishing")

        # 2. Pending SC2 gates
        pending_gates = safety_gate.get_pending_gates(session_id)
        sc2_pending = [g for g in pending_gates if g.safety_category == SafetyCategory.SC2]
        if sc2_pending:
            blockers.append(f"{len(sc2_pending)} SC2 gate(s) still await approval")

        # 3. Chain integrity
        chain = orchestrator.audit.verify_chain(session_id)
        if not chain.get("valid", False):
            blockers.append("Audit chain integrity check FAILED — cannot publish")

    else:
        blockers.append(f"Unknown target state: {target_state}. Valid: SHARED, PUBLISHED")

    return {
        "can_transition": len(blockers) == 0,
        "target_state": target_state,
        "blockers": blockers,
    }
