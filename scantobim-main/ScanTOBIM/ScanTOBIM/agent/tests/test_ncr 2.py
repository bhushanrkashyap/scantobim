"""Tests for P5 NCR Service — ncr.py"""

from __future__ import annotations

import uuid

import pytest

from agent.audit import AuditLedger
from agent.models import NCRSeverity, NCRStatus
from agent.ncr import NCRAlreadyResolvedError, NCRNotFoundError, NCRService

ELEMENT = "revit-el-abc123"


@pytest.fixture
def svc(tmp_path):
    """Fresh isolated AuditLedger + NCRService per test via tmp_path."""
    audit = AuditLedger(db_path=tmp_path / "test_ncr.db")
    return NCRService(audit), audit


@pytest.fixture
def session():
    """Unique session ID per test — prevents cross-test event leakage."""
    return str(uuid.uuid4())


# ── Raise ─────────────────────────────────────────────────────────────────────


def test_raise_ncr_returns_record(svc, session):
    service, _ = svc
    ncr = service.raise_ncr(session, ELEMENT, "Deviation 18mm > 10mm tolerance", "MAJOR")
    assert ncr.ncr_id
    assert ncr.session_id == session
    assert ncr.element_id == ELEMENT
    assert ncr.severity == NCRSeverity.MAJOR
    assert ncr.status == NCRStatus.OPEN


def test_raise_ncr_critical(svc, session):
    service, _ = svc
    ncr = service.raise_ncr(session, ELEMENT, "SC1 override attempt", "CRITICAL")
    assert ncr.severity == NCRSeverity.CRITICAL


def test_raise_ncr_minor(svc, session):
    service, _ = svc
    ncr = service.raise_ncr(session, ELEMENT, "Minor cosmetic gap", "MINOR")
    assert ncr.severity == NCRSeverity.MINOR


def test_raise_ncr_appends_audit_event(svc, session):
    service, audit = svc
    service.raise_ncr(session, ELEMENT, "Test NCR", "MAJOR")
    events = audit.get_session_events(session)
    ncr_events = [e for e in events if e.event_type.value == "ncr_raised"]
    assert len(ncr_events) == 1


# ── Resolve ───────────────────────────────────────────────────────────────────


def test_resolve_ncr_changes_status(svc, session):
    service, _ = svc
    ncr = service.raise_ncr(session, ELEMENT, "Issue found", "MAJOR")
    resolved = service.resolve_ncr(ncr.ncr_id, "eng@example.com", "Adjusted within tolerance")
    assert resolved.status == NCRStatus.RESOLVED
    assert resolved.resolved_by == "eng@example.com"
    assert resolved.resolution == "Adjusted within tolerance"
    assert resolved.resolved_at is not None


def test_resolve_ncr_appends_audit_event(svc, session):
    service, audit = svc
    ncr = service.raise_ncr(session, ELEMENT, "Test", "MINOR")
    service.resolve_ncr(ncr.ncr_id, "eng@example.com", "Fixed")
    events = audit.get_session_events(session)
    resolved_events = [e for e in events if e.event_type.value == "ncr_resolved"]
    assert len(resolved_events) == 1


def test_resolve_unknown_ncr_raises(svc):
    service, _ = svc
    with pytest.raises(NCRNotFoundError):
        service.resolve_ncr("nonexistent-ncr-id", "eng@example.com", "N/A")


def test_resolve_already_resolved_raises(svc, session):
    service, _ = svc
    ncr = service.raise_ncr(session, ELEMENT, "Test", "MINOR")
    service.resolve_ncr(ncr.ncr_id, "eng@example.com", "Fixed")
    with pytest.raises(NCRAlreadyResolvedError):
        service.resolve_ncr(ncr.ncr_id, "eng@example.com", "Fixed again")


# ── Queries ───────────────────────────────────────────────────────────────────


def test_list_open_returns_only_open(svc, session):
    service, _ = svc
    n1 = service.raise_ncr(session, ELEMENT, "NCR 1", "MAJOR")
    n2 = service.raise_ncr(session, ELEMENT, "NCR 2", "MINOR")
    service.resolve_ncr(n1.ncr_id, "eng@example.com", "Resolved")
    open_ncrs = service.list_open(session)
    assert len(open_ncrs) == 1
    assert open_ncrs[0].ncr_id == n2.ncr_id


def test_list_session_returns_all(svc, session):
    service, _ = svc
    service.raise_ncr(session, ELEMENT, "NCR A", "CRITICAL")
    service.raise_ncr(session, ELEMENT, "NCR B", "MAJOR")
    all_ncrs = service.list_session(session)
    assert len(all_ncrs) == 2


def test_summary_correct_counts(svc, session):
    service, _ = svc
    n1 = service.raise_ncr(session, ELEMENT, "Critical", "CRITICAL")
    service.raise_ncr(session, ELEMENT, "Major", "MAJOR")
    service.raise_ncr(session, ELEMENT, "Minor", "MINOR")
    service.resolve_ncr(n1.ncr_id, "eng@example.com", "Fixed critical")
    summary = service.summary(session)
    assert summary["total"] == 3
    assert summary["open"] == 2
    assert summary["resolved"] == 1
    assert summary["critical"] == 1
    assert summary["major"] == 1
    assert summary["minor"] == 1


def test_get_ncr_by_id(svc, session):
    service, _ = svc
    ncr = service.raise_ncr(session, ELEMENT, "Test", "MAJOR")
    fetched = service.get_ncr(ncr.ncr_id)
    assert fetched.ncr_id == ncr.ncr_id


def test_get_unknown_ncr_raises(svc):
    service, _ = svc
    with pytest.raises(NCRNotFoundError):
        service.get_ncr("bad-id-999")
