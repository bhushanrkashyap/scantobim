"""
cde_state.py  —  P5 ISO 19650 CDE State Machine
─────────────────────────────────────────────────
Manages the Common Data Environment lifecycle for each session:

    WIP  ──────────────────►  SHARED  ──────────────────►  PUBLISHED
         all elements created          no open NCRs
                                       all SC2 gates resolved
                                       audit chain valid
                                       → auto-generate DRP

Rules are enforced synchronously. Any violation raises CDETransitionError.
Publishing triggers DRP generation and writes a DESIGN_RECORD_ISSUED audit event.
"""

from __future__ import annotations

from datetime import datetime, timezone
from typing import TYPE_CHECKING

import structlog

from agent.models import (
    AuditEvent,
    AuditEventType,
    CDESessionState,
    CDEState,
    CDETransition,
    SafetyCategory,
)

if TYPE_CHECKING:
    from agent.audit import AuditLedger
    from agent.ncr import NCRService
    from agent.orchestrator import ScanToBIMOrchestrator
    from agent.safety_gate import SafetyGateService

logger = structlog.get_logger()


class CDETransitionError(Exception):
    """Raised when a requested CDE transition violates a prerequisite rule."""


class CDEStateService:
    """
    Manages ISO 19650 CDE state for all active sessions.
    One instance is shared across the FastAPI app (same as AuditLedger).
    """

    def __init__(
        self,
        audit: AuditLedger,
        orchestrator: ScanToBIMOrchestrator,
        safety_gate: SafetyGateService,
        ncr_service: NCRService,
    ) -> None:
        self._audit = audit
        self._orchestrator = orchestrator
        self._safety_gate = safety_gate
        self._ncr = ncr_service
        # session_id → CDESessionState
        self._states: dict[str, CDESessionState] = {}

    # ── Public API ────────────────────────────────────────────────────────────

    def get_state(self, session_id: str) -> CDESessionState:
        """Return current CDE state for session. Creates WIP entry if not seen."""
        if session_id not in self._states:
            self._states[session_id] = CDESessionState(session_id=session_id)
        return self._states[session_id]

    def transition(
        self,
        session_id: str,
        target_state: str,
        triggered_by: str = "coordinator",
        notes: str | None = None,
    ) -> CDETransition:
        """
        Attempt a CDE state transition. Validates all prerequisites first.

        Raises CDETransitionError if any prerequisite is not met.
        Returns the CDETransition record on success.
        """
        cde = self.get_state(session_id)
        target = CDEState(target_state.upper())

        self._validate_transition(cde.current_state, target, session_id)

        record = CDETransition(
            session_id=session_id,
            from_state=cde.current_state,
            to_state=target,
            triggered_by=triggered_by,
            notes=notes,
        )
        cde.transitions.append(record)
        cde.current_state = target
        cde.updated_at = datetime.now(timezone.utc)

        # Audit
        self._audit.append(
            AuditEvent(
                session_id=session_id,
                event_type=AuditEventType.CDE_STATE_CHANGED,
                actor=triggered_by,
                detail={
                    "from_state": record.from_state.value,
                    "to_state": target.value,
                    "notes": notes,
                },
            )
        )

        logger.info(
            "cde_transition",
            session_id=session_id[:12],
            from_state=record.from_state.value,
            to_state=target.value,
            by=triggered_by,
        )

        # Publishing auto-generates the DRP
        if target == CDEState.PUBLISHED:
            self._on_publish(session_id, triggered_by)

        return record

    # ── Validation rules ──────────────────────────────────────────────────────

    def _validate_transition(
        self,
        current: CDEState,
        target: CDEState,
        session_id: str,
    ) -> None:
        """Enforce all prerequisite rules. Raises CDETransitionError on failure."""

        # --- Valid transition paths -------------------------------------------
        valid = {
            CDEState.WIP: {CDEState.SHARED},
            CDEState.SHARED: {CDEState.PUBLISHED},
        }
        if target not in valid.get(current, set()):
            raise CDETransitionError(
                f"Invalid transition {current.value} → {target.value}. "
                f"Allowed from {current.value}: "
                f"{[s.value for s in valid.get(current, set())]}"
            )

        # --- WIP → SHARED rules -----------------------------------------------
        if target == CDEState.SHARED:
            self._assert_no_pending_instructions(session_id)

        # --- SHARED → PUBLISHED rules -----------------------------------------
        if target == CDEState.PUBLISHED:
            self._assert_no_open_ncrs(session_id)
            self._assert_sc2_all_resolved(session_id)
            self._assert_chain_valid(session_id)

    def _assert_no_pending_instructions(self, session_id: str) -> None:
        instructions = self._orchestrator._instructions.get(session_id, [])
        results = self._orchestrator._results.get(session_id, [])
        pending = len(instructions) - len(results)
        if pending > 0:
            raise CDETransitionError(
                f"Cannot transition to SHARED: {pending} instruction(s) still pending. "
                "All elements must be created or explicitly blocked before sharing."
            )

    def _assert_no_open_ncrs(self, session_id: str) -> None:
        open_ncrs = self._ncr.list_open(session_id)
        if open_ncrs:
            ids = ", ".join(n.ncr_id[:8] for n in open_ncrs[:5])
            raise CDETransitionError(
                f"Cannot PUBLISH: {len(open_ncrs)} open NCR(s) must be resolved first. "
                f"NCR IDs: {ids}{'...' if len(open_ncrs) > 5 else ''}"
            )

    def _assert_sc2_all_resolved(self, session_id: str) -> None:
        pending_gates = self._safety_gate.get_pending_gates(session_id)
        sc2_pending = [g for g in pending_gates if g.safety_category == SafetyCategory.SC2]
        if sc2_pending:
            raise CDETransitionError(
                f"Cannot PUBLISH: {len(sc2_pending)} SC2 safety gate(s) still await "
                "engineer approval. Approve or reject all SC2 gates before publishing."
            )

    def _assert_chain_valid(self, session_id: str) -> None:
        chain = self._audit.verify_chain(session_id)
        if not chain.get("valid", False):
            broken = len(chain.get("broken_links", []))
            raise CDETransitionError(
                f"Cannot PUBLISH: Audit chain integrity FAILED ({broken} broken link(s)). "
                "Investigate and restore chain integrity before publishing."
            )

    # ── Post-publish hook ─────────────────────────────────────────────────────

    def _on_publish(self, session_id: str, triggered_by: str) -> None:
        """
        Called automatically when a session transitions to PUBLISHED.
        Writes a DESIGN_RECORD_ISSUED audit event.
        DRP HTML is generated on-demand via GET /sessions/{id}/drp.
        """
        self._audit.append(
            AuditEvent(
                session_id=session_id,
                event_type=AuditEventType.DESIGN_RECORD_ISSUED,
                actor=triggered_by,
                detail={
                    "message": "Design Record Package issued upon PUBLISHED state transition.",
                    "cde_state": CDEState.PUBLISHED.value,
                },
            )
        )
        logger.info("drp_issued", session_id=session_id[:12])
