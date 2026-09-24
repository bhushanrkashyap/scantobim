"""Safety gate — enforces SC1 hard block, SC2 approval requirement.

PoV: SC2 approval is via CLI prompt or API endpoint.
Production: Teams Adaptive Cards + Power Automate + HMAC signatures.
"""

from __future__ import annotations

import hashlib
import hmac
from datetime import datetime, timezone

import structlog

from agent.audit import AuditLedger
from agent.models import (
    AuditEvent,
    AuditEventType,
    ElementInstruction,
    SafetyCategory,
    SafetyGateDecision,
    SafetyGateRequest,
    SafetyGateStatus,
)

logger = structlog.get_logger()

# Secret used for HMAC approval signatures.
# Production: load from environment / Azure Key Vault.
_APPROVAL_SECRET = "scantobim-gate-secret"


class SafetyCategoryViolationError(Exception):
    """Raised when an SC1 element creation is attempted."""


class SC2ApprovalRequiredError(Exception):
    """Raised when an SC2 element lacks approval."""

    def __init__(self, gate: SafetyGateRequest):
        self.gate = gate
        super().__init__(f"SC2 approval required for segment {gate.segment_id}")


class SafetyGateService:
    """Enforces safety classification rules before element creation."""

    def __init__(self, audit: AuditLedger):
        self.audit = audit
        self._pending_gates: dict[str, SafetyGateRequest] = {}
        self._decisions: dict[str, SafetyGateDecision] = {}
        # Store the blocked instruction alongside its gate so re-dispatch
        # after approval does not need to search through _instructions.
        self._gate_instructions: dict[str, ElementInstruction] = {}

    # ── Check ──────────────────────────────────────────────────────────────────

    def check_instruction(
        self, instruction: ElementInstruction, session_id: str
    ) -> SafetyGateRequest | None:
        """Check if an instruction is safe to execute.

        Returns None if safe to proceed, or a SafetyGateRequest if approval needed.
        Raises SafetyCategoryViolationError for SC1 — NEVER proceed.
        """
        if instruction.safety_category == SafetyCategory.SC1:
            # HARD BLOCK — log and raise
            self.audit.append(
                AuditEvent(
                    session_id=session_id,
                    event_type=AuditEventType.ELEMENT_BLOCKED,
                    zone_id=instruction.zone_id,
                    segment_id=instruction.segment_id,
                    actor="safety_gate",
                    detail={
                        "reason": "SC1 hard block — safety-critical element cannot be auto-created",
                        "element_type": instruction.element_type.value,
                        "safety_category": "SC1",
                    },
                )
            )
            logger.error(
                "SC1_HARD_BLOCK",
                segment_id=instruction.segment_id[:8],
                element_type=instruction.element_type.value,
            )
            raise SafetyCategoryViolationError(
                f"SC1 element {instruction.segment_id} ({instruction.element_type.value}) "
                f"CANNOT be auto-created. Requires manual design."
            )

        if instruction.safety_category == SafetyCategory.SC2:
            if instruction.approval_signature:
                # Already carries a valid approval — verify it then proceed
                if self._verify_approval_signature(instruction):
                    return None
                # Tampered / invalid signature — treat as unapproved
                logger.error(
                    "SC2_invalid_approval_signature",
                    segment_id=instruction.segment_id[:8],
                )

            # Raise a gate for approval
            gate = SafetyGateRequest(
                segment_id=instruction.segment_id,
                element_type=instruction.element_type,
                safety_category=SafetyCategory.SC2,
                classification_reason="SC2 safety-related element requires engineer approval",
                session_id=session_id,
                zone_id=instruction.zone_id,
            )
            self._pending_gates[gate.gate_id] = gate
            # Store instruction so decide_gate can re-dispatch without scanning
            self._gate_instructions[gate.gate_id] = instruction

            self.audit.append(
                AuditEvent(
                    session_id=session_id,
                    event_type=AuditEventType.SAFETY_GATE_RAISED,
                    zone_id=instruction.zone_id,
                    segment_id=instruction.segment_id,
                    actor="safety_gate",
                    detail={
                        "gate_id": gate.gate_id,
                        "safety_category": "SC2",
                        "element_type": instruction.element_type.value,
                    },
                )
            )
            logger.warning(
                "SC2_gate_raised",
                gate_id=gate.gate_id[:8],
                segment_id=instruction.segment_id[:8],
            )
            return gate

        # SC3 and NS — auto-proceed
        return None

    # ── Decide ─────────────────────────────────────────────────────────────────

    def decide_gate(
        self, decision: SafetyGateDecision, session_id: str
    ) -> tuple[bool, ElementInstruction | None]:
        """Record a gate decision.

        Returns (approved: bool, instruction | None).
        If approved, the returned instruction has approval_signature set and
        is ready for immediate dispatch — caller does not need to search
        _instructions by segment_id.
        """
        gate = self._pending_gates.get(decision.gate_id)
        if not gate:
            raise ValueError(f"Gate {decision.gate_id} not found")

        gate.status = SafetyGateStatus.APPROVED if decision.approved else SafetyGateStatus.REJECTED
        self._decisions[decision.gate_id] = decision

        event_type = (
            AuditEventType.SAFETY_GATE_APPROVED
            if decision.approved
            else AuditEventType.SAFETY_GATE_REJECTED
        )
        self.audit.append(
            AuditEvent(
                session_id=session_id,
                event_type=event_type,
                zone_id=gate.zone_id,
                segment_id=gate.segment_id,
                actor=decision.approver_upn,
                detail={
                    "gate_id": decision.gate_id,
                    "approved": decision.approved,
                    "comments": decision.comments,
                },
            )
        )

        logger.info(
            "gate_decided",
            gate_id=decision.gate_id[:8],
            approved=decision.approved,
            approver=decision.approver_upn,
        )

        if not decision.approved:
            return False, None

        # Build HMAC approval signature and stamp it onto the instruction
        instruction = self._gate_instructions.get(decision.gate_id)
        if instruction:
            sig = self.build_approval_signature(
                gate_id=decision.gate_id,
                approver_upn=decision.approver_upn,
            )
            instruction.approval_signature = sig
            instruction.approver_upn = decision.approver_upn
            instruction.approved_at_utc = decision.decided_at

        return True, instruction

    # ── Lookup helpers ─────────────────────────────────────────────────────────

    def get_pending_gates(self, session_id: str | None = None) -> list[SafetyGateRequest]:
        """Get all pending gates, optionally filtered by session."""
        gates = [g for g in self._pending_gates.values() if g.status == SafetyGateStatus.PENDING]
        if session_id:
            gates = [g for g in gates if g.session_id == session_id]
        return gates

    def get_gate(self, gate_id: str) -> SafetyGateRequest | None:
        return self._pending_gates.get(gate_id)

    def get_gate_instruction(self, gate_id: str) -> ElementInstruction | None:
        """Return the blocked instruction associated with a gate, or None."""
        return self._gate_instructions.get(gate_id)

    # ── HMAC signature helpers ─────────────────────────────────────────────────

    def build_approval_signature(
        self, gate_id: str, approver_upn: str, secret: str = _APPROVAL_SECRET
    ) -> str:
        """Produce an HMAC-SHA256 approval signature for an SC2 gate decision.

        The signature binds the gate_id and approver UPN together so that a
        signature from one gate cannot be reused for a different element.
        """
        payload = f"{gate_id}|{approver_upn}|{datetime.now(timezone.utc).date().isoformat()}"
        return hmac.new(secret.encode(), payload.encode(), hashlib.sha256).hexdigest()

    def _verify_approval_signature(
        self, instruction: ElementInstruction, secret: str = _APPROVAL_SECRET
    ) -> bool:
        """Verify that an instruction's approval_signature is authentic.

        Checks today's date only — signatures expire at midnight UTC so that
        a token from a previous shift cannot be replayed indefinitely.
        """
        if not instruction.approval_signature or not instruction.approver_upn:
            return False

        # Find the gate_id for this instruction
        gate_id = next(
            (
                gid
                for gid, instr in self._gate_instructions.items()
                if instr.segment_id == instruction.segment_id
            ),
            None,
        )
        if not gate_id:
            return False  # no gate was ever raised for this instruction

        expected = self.build_approval_signature(gate_id, instruction.approver_upn, secret)
        # Use hmac.compare_digest to prevent timing attacks
        return hmac.compare_digest(expected, instruction.approval_signature)
