"""
ncr.py  —  P5 Non-Conformance Report service
─────────────────────────────────────────────
Manages NCR lifecycle: raise → investigate → resolve → close.

NCRs are raised when:
  - Element deviation exceeds tolerance (deviation_check.py)
  - Safety parameter mismatch detected (checker_tools.py)
  - Clash is unresolvable (discipline_tools.py)
  - Any manual override attempted on SC1 element

All NCR raises and resolutions are appended to the audit ledger.
"""

from __future__ import annotations

from datetime import datetime, timezone
from typing import TYPE_CHECKING

import structlog

from agent.models import (
    AuditEvent,
    AuditEventType,
    NCRRecord,
    NCRSeverity,
    NCRStatus,
)

if TYPE_CHECKING:
    from agent.audit import AuditLedger

logger = structlog.get_logger()


class NCRNotFoundError(Exception):
    pass


class NCRAlreadyResolvedError(Exception):
    pass


class NCRService:
    """
    In-memory NCR store backed by the audit ledger for persistence.
    One instance shared across the FastAPI app.
    """

    def __init__(self, audit: AuditLedger) -> None:
        self._audit = audit
        # ncr_id → NCRRecord
        self._ncrs: dict[str, NCRRecord] = {}

    # ── Raise ─────────────────────────────────────────────────────────────────

    def raise_ncr(
        self,
        session_id: str,
        element_id: str,
        description: str,
        severity: str,
        raised_by: str = "agent",
    ) -> NCRRecord:
        """
        Create and store a new NCR. Appends NCR_RAISED to audit ledger.

        severity: 'CRITICAL' | 'MAJOR' | 'MINOR'
        """
        ncr = NCRRecord(
            session_id=session_id,
            element_id=element_id,
            description=description,
            severity=NCRSeverity(severity.upper()),
            raised_by=raised_by,
        )
        self._ncrs[ncr.ncr_id] = ncr

        self._audit.append(
            AuditEvent(
                session_id=session_id,
                event_type=AuditEventType.NCR_RAISED,
                element_id=element_id,
                actor=raised_by,
                detail={
                    "ncr_id": ncr.ncr_id,
                    "severity": ncr.severity.value,
                    "description": description[:120],
                },
            )
        )

        logger.warning(
            "ncr_raised",
            ncr_id=ncr.ncr_id[:12],
            element_id=element_id[:12],
            severity=ncr.severity.value,
        )
        return ncr

    # ── Resolve ───────────────────────────────────────────────────────────────

    def resolve_ncr(
        self,
        ncr_id: str,
        resolved_by: str,
        resolution: str,
    ) -> NCRRecord:
        """
        Mark an NCR as resolved. Appends NCR_RESOLVED to audit ledger.

        Raises NCRNotFoundError if ncr_id is unknown.
        Raises NCRAlreadyResolvedError if status is already RESOLVED or CLOSED.
        """
        ncr = self._ncrs.get(ncr_id)
        if not ncr:
            raise NCRNotFoundError(f"NCR {ncr_id!r} not found.")
        if ncr.status != NCRStatus.OPEN:
            raise NCRAlreadyResolvedError(
                f"NCR {ncr_id[:8]} is already {ncr.status.value}. Cannot resolve again."
            )

        ncr.status = NCRStatus.RESOLVED
        ncr.resolved_by = resolved_by
        ncr.resolved_at = datetime.now(timezone.utc)
        ncr.resolution = resolution

        self._audit.append(
            AuditEvent(
                session_id=ncr.session_id,
                event_type=AuditEventType.NCR_RESOLVED,
                element_id=ncr.element_id,
                actor=resolved_by,
                detail={
                    "ncr_id": ncr.ncr_id,
                    "severity": ncr.severity.value,
                    "resolution": resolution[:120],
                },
            )
        )

        logger.info(
            "ncr_resolved",
            ncr_id=ncr_id[:12],
            resolved_by=resolved_by,
        )
        return ncr

    # ── Close ─────────────────────────────────────────────────────────────────

    def close_ncr(self, ncr_id: str, closed_by: str) -> NCRRecord:
        """Close a resolved NCR (admin action — post-publish sign-off)."""
        ncr = self._ncrs.get(ncr_id)
        if not ncr:
            raise NCRNotFoundError(f"NCR {ncr_id!r} not found.")
        if ncr.status == NCRStatus.OPEN:
            raise NCRAlreadyResolvedError("Cannot close an OPEN NCR — resolve it first.")
        ncr.status = NCRStatus.CLOSED
        logger.info("ncr_closed", ncr_id=ncr_id[:12], closed_by=closed_by)
        return ncr

    # ── Queries ───────────────────────────────────────────────────────────────

    def get_ncr(self, ncr_id: str) -> NCRRecord:
        ncr = self._ncrs.get(ncr_id)
        if not ncr:
            raise NCRNotFoundError(f"NCR {ncr_id!r} not found.")
        return ncr

    def list_session(self, session_id: str) -> list[NCRRecord]:
        """All NCRs (any status) for a session, newest first."""
        return sorted(
            [n for n in self._ncrs.values() if n.session_id == session_id],
            key=lambda n: n.raised_at,
            reverse=True,
        )

    def list_open(self, session_id: str) -> list[NCRRecord]:
        """Only OPEN NCRs for a session."""
        return [n for n in self.list_session(session_id) if n.status == NCRStatus.OPEN]

    def summary(self, session_id: str) -> dict:
        """Stats dict for DRP and API responses."""
        all_ncrs = self.list_session(session_id)
        return {
            "total": len(all_ncrs),
            "open": sum(1 for n in all_ncrs if n.status == NCRStatus.OPEN),
            "resolved": sum(1 for n in all_ncrs if n.status == NCRStatus.RESOLVED),
            "closed": sum(1 for n in all_ncrs if n.status == NCRStatus.CLOSED),
            "critical": sum(1 for n in all_ncrs if n.severity == NCRSeverity.CRITICAL),
            "major": sum(1 for n in all_ncrs if n.severity == NCRSeverity.MAJOR),
            "minor": sum(1 for n in all_ncrs if n.severity == NCRSeverity.MINOR),
        }
