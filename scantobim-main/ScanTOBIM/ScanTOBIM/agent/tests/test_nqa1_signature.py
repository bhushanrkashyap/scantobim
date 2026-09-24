"""test_nqa1_signature.py — NQA-1 three-party signature workflow.

Covers:
  • Service happy path: sign preparer/verifier/approver in order
  • Ordering enforcement (verifier before preparer → error)
  • Separation of duties (same UPN twice → error)
  • Duplicate role (sign twice as same role → error)
  • Chain integrity: valid + HMAC tamper detection
  • Revocation preserves row, invalidates fully_signed
  • status() summary shape
  • UPN validation
  • Package hash is stable when signatures change
  • Auto-preparer sign on /nqa1-package GET
  • Endpoints: POST / GET / PATCH revoke
  • Endpoint error codes (404, 409 conflict, 422)
  • NQA-1 HTML rendering of signatures (green tick when signed)
"""

from __future__ import annotations

from pathlib import Path

import pytest
import pytest_asyncio
from httpx import ASGITransport, AsyncClient

from agent.models import NQA1Role, NQA1Signature
from agent.nqa1_signature import (
    DuplicateRoleError,
    DuplicateSignerError,
    NQA1SignatureService,
    SignatureOrderError,
    compute_package_hash,
)

# ── Helpers ──────────────────────────────────────────────────────────────────


@pytest.fixture
def fresh_service(tmp_path: Path) -> NQA1SignatureService:
    return NQA1SignatureService(
        db_path=tmp_path / "sig.db",
        secret="test-secret-not-production",
    )


_PH = "a" * 64  # canonical test package hash


# ═════════════════════════════════════════════════════════════════════════════
# 1. Service happy path + ordering
# ═════════════════════════════════════════════════════════════════════════════


class TestSignOrdering:
    def test_preparer_signs_first(self, fresh_service):
        sig = fresh_service.sign(
            session_id="S1",
            role=NQA1Role.PREPARER,
            signer_upn="prep@org.com",
            signer_name="Alice Preparer",
            package_hash=_PH,
        )
        assert sig.role == NQA1Role.PREPARER
        assert sig.previous_signature_hash is None
        assert sig.signature_hash is not None
        assert len(sig.signature_hash) == 64  # sha256 hex

    def test_three_in_order_succeeds(self, fresh_service):
        fresh_service.sign("S1", NQA1Role.PREPARER, "p@o.com", "P", _PH)
        fresh_service.sign("S1", NQA1Role.VERIFIER, "v@o.com", "V", _PH)
        fresh_service.sign("S1", NQA1Role.APPROVER, "a@o.com", "A", _PH)

        status = fresh_service.status("S1")
        assert status["fully_signed"] is True
        assert status["next_required"] is None
        assert status["signature_count"] == 3

    def test_verifier_before_preparer_rejected(self, fresh_service):
        with pytest.raises(SignatureOrderError):
            fresh_service.sign("S1", NQA1Role.VERIFIER, "v@o.com", "V", _PH)

    def test_approver_without_verifier_rejected(self, fresh_service):
        fresh_service.sign("S1", NQA1Role.PREPARER, "p@o.com", "P", _PH)
        with pytest.raises(SignatureOrderError):
            fresh_service.sign("S1", NQA1Role.APPROVER, "a@o.com", "A", _PH)


class TestSeparationOfDuties:
    def test_same_upn_two_roles_rejected(self, fresh_service):
        fresh_service.sign("S1", NQA1Role.PREPARER, "same@o.com", "Same", _PH)
        with pytest.raises(DuplicateSignerError):
            fresh_service.sign("S1", NQA1Role.VERIFIER, "same@o.com", "Same", _PH)

    def test_same_upn_different_case_rejected(self, fresh_service):
        """UPNs are normalised to lower case before comparison."""
        fresh_service.sign("S1", NQA1Role.PREPARER, "Same@Org.com", "Same", _PH)
        with pytest.raises(DuplicateSignerError):
            fresh_service.sign("S1", NQA1Role.VERIFIER, "SAME@ORG.COM", "Same", _PH)

    def test_different_upns_ok(self, fresh_service):
        fresh_service.sign("S1", NQA1Role.PREPARER, "p@o.com", "P", _PH)
        fresh_service.sign("S1", NQA1Role.VERIFIER, "v@o.com", "V", _PH)
        # no exception


class TestDuplicateRole:
    def test_same_role_twice_rejected(self, fresh_service):
        fresh_service.sign("S1", NQA1Role.PREPARER, "p1@o.com", "P1", _PH)
        with pytest.raises(DuplicateRoleError):
            fresh_service.sign("S1", NQA1Role.PREPARER, "p2@o.com", "P2", _PH)

    def test_sign_after_revoke_ok(self, fresh_service):
        """After revocation, the role slot is free for a fresh signer."""
        s1 = fresh_service.sign("S1", NQA1Role.PREPARER, "p1@o.com", "P1", _PH)
        fresh_service.revoke(s1.signature_id, "admin", "replaced")
        fresh_service.sign("S1", NQA1Role.PREPARER, "p2@o.com", "P2", _PH)
        # no exception


# ═════════════════════════════════════════════════════════════════════════════
# 2. Chain integrity + HMAC
# ═════════════════════════════════════════════════════════════════════════════


class TestChainIntegrity:
    def test_fresh_chain_valid(self, fresh_service):
        fresh_service.sign("S1", NQA1Role.PREPARER, "p@o.com", "P", _PH)
        fresh_service.sign("S1", NQA1Role.VERIFIER, "v@o.com", "V", _PH)
        result = fresh_service.verify_chain("S1")
        assert result["valid"] is True
        assert result["broken_links"] == []
        assert result["signature_count"] == 2

    def test_hmac_detects_payload_tamper(self, fresh_service):
        """Forging a signature_hash in the DB breaks verification."""
        import sqlite3

        fresh_service.sign("S1", NQA1Role.PREPARER, "p@o.com", "P", _PH)

        # Tamper directly in SQLite
        with sqlite3.connect(fresh_service.db_path) as conn:
            conn.execute(
                "UPDATE nqa1_signatures SET signer_name='Attacker' WHERE session_id=?",
                ("S1",),
            )

        result = fresh_service.verify_chain("S1")
        assert result["valid"] is False
        assert any(b["reason"] == "hash_mismatch" for b in result["broken_links"])

    def test_chain_break_when_previous_hash_forged(self, fresh_service):
        """Inserting a signature with a bogus previous_hash breaks the chain."""
        import sqlite3

        fresh_service.sign("S1", NQA1Role.PREPARER, "p@o.com", "P", _PH)

        with sqlite3.connect(fresh_service.db_path) as conn:
            conn.execute(
                "UPDATE nqa1_signatures SET previous_signature_hash='bogus' WHERE session_id=?",
                ("S1",),
            )

        result = fresh_service.verify_chain("S1")
        assert result["valid"] is False

    def test_empty_session_chain_valid(self, fresh_service):
        """No signatures = trivially valid."""
        result = fresh_service.verify_chain("never-signed")
        assert result["valid"] is True
        assert result["signature_count"] == 0


# ═════════════════════════════════════════════════════════════════════════════
# 3. Revocation
# ═════════════════════════════════════════════════════════════════════════════


class TestRevocation:
    def test_revoke_preserves_row(self, fresh_service):
        sig = fresh_service.sign("S1", NQA1Role.PREPARER, "p@o.com", "P", _PH)
        fresh_service.revoke(sig.signature_id, "qa@o.com", "signer left")
        all_sigs = fresh_service.list_session("S1")
        assert len(all_sigs) == 1
        assert all_sigs[0].revoked_at_utc is not None
        assert all_sigs[0].revoked_by == "qa@o.com"
        assert all_sigs[0].revocation_reason == "signer left"

    def test_revoke_drops_fully_signed(self, fresh_service):
        p = fresh_service.sign("S1", NQA1Role.PREPARER, "p@o.com", "P", _PH)
        fresh_service.sign("S1", NQA1Role.VERIFIER, "v@o.com", "V", _PH)
        fresh_service.sign("S1", NQA1Role.APPROVER, "a@o.com", "A", _PH)
        assert fresh_service.status("S1")["fully_signed"] is True

        fresh_service.revoke(p.signature_id, "admin", "audit flagged")
        assert fresh_service.status("S1")["fully_signed"] is False

    def test_revoke_unknown_raises(self, fresh_service):
        with pytest.raises(KeyError):
            fresh_service.revoke("does-not-exist", "admin", "test")


# ═════════════════════════════════════════════════════════════════════════════
# 4. Status + UPN validation
# ═════════════════════════════════════════════════════════════════════════════


class TestStatusAndValidation:
    def test_status_shape(self, fresh_service):
        fresh_service.sign("S1", NQA1Role.PREPARER, "p@o.com", "Alice", _PH)
        s = fresh_service.status("S1")

        assert s["session_id"] == "S1"
        assert s["fully_signed"] is False
        assert s["next_required"] == "verifier"
        assert s["by_role"]["preparer"]["signed"] is True
        assert s["by_role"]["preparer"]["signer_name"] == "Alice"
        assert s["by_role"]["verifier"]["signed"] is False
        assert s["by_role"]["approver"]["signed"] is False
        assert s["chain_valid"] is True

    def test_invalid_upn_rejected_by_model(self):
        with pytest.raises(ValueError):
            NQA1Signature(
                session_id="S1",
                role=NQA1Role.PREPARER,
                signer_upn="not-an-email",
                signer_name="X",
                package_hash=_PH,
            )


# ═════════════════════════════════════════════════════════════════════════════
# 5. compute_package_hash stability
# ═════════════════════════════════════════════════════════════════════════════


class TestPackageHash:
    def test_stable_across_calls(self):
        d = {"a": 1, "b": [1, 2, 3], "c": "x"}
        assert compute_package_hash(d) == compute_package_hash(d)

    def test_signatures_field_ignored(self):
        """Adding signatures to the package must not change the hash —
        otherwise a signature would invalidate itself on read-back."""
        a = {"a": 1, "signatures": {"fully_signed": False}}
        b = {
            "a": 1,
            "signatures": {"fully_signed": True, "by_role": {"preparer": {"signed": True}}},
        }
        assert compute_package_hash(a) == compute_package_hash(b)

    def test_content_changes_change_hash(self):
        assert compute_package_hash({"a": 1}) != compute_package_hash({"a": 2})


# ═════════════════════════════════════════════════════════════════════════════
# 6. Auto-preparer sign on /nqa1-package
# ═════════════════════════════════════════════════════════════════════════════


class TestAutoPreparerSign:
    def test_first_package_read_signs_preparer(self):
        from agent.main import _auto_preparer_sign, nqa1_signatures, orchestrator

        sess = orchestrator.create_session("Auto Prep 1")
        orchestrator.run_segmentation(
            sess.session_id,
            use_synthetic=True,
            zone_id="Z",
        )
        _auto_preparer_sign(sess.session_id)

        active = [
            s for s in nqa1_signatures.list_session(sess.session_id) if s.revoked_at_utc is None
        ]
        assert len(active) == 1
        assert active[0].role == NQA1Role.PREPARER
        assert active[0].signer_upn == "agent@scantobim.ai"

    def test_repeated_package_reads_idempotent(self):
        from agent.main import _auto_preparer_sign, nqa1_signatures, orchestrator

        sess = orchestrator.create_session("Auto Prep 2")
        orchestrator.run_segmentation(
            sess.session_id,
            use_synthetic=True,
            zone_id="Z",
        )
        _auto_preparer_sign(sess.session_id)
        _auto_preparer_sign(sess.session_id)
        _auto_preparer_sign(sess.session_id)

        prep_sigs = [
            s
            for s in nqa1_signatures.list_session(sess.session_id)
            if s.role == NQA1Role.PREPARER and s.revoked_at_utc is None
        ]
        assert len(prep_sigs) == 1


# ═════════════════════════════════════════════════════════════════════════════
# 7. API endpoints
# ═════════════════════════════════════════════════════════════════════════════


@pytest_asyncio.fixture
async def auth_client():
    from agent.main import app

    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as ac:
        resp = await ac.post(
            "/auth/token",
            data={"username": "admin", "password": "admin123"},
        )
        ac.headers["Authorization"] = f"Bearer {resp.json().get('access_token', '')}"
        yield ac


class TestSignEndpoint:
    @pytest.mark.asyncio
    async def test_requires_auth(self):
        from agent.main import app

        async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as ac:
            resp = await ac.post(
                "/sessions/x/nqa1-signatures",
                json={
                    "role": "preparer",
                    "signer_upn": "p@o.com",
                    "signer_name": "P",
                },
            )
            assert resp.status_code == 401

    @pytest.mark.asyncio
    async def test_unknown_session_404(self, auth_client: AsyncClient):
        resp = await auth_client.post(
            "/sessions/does-not-exist/nqa1-signatures",
            json={"role": "preparer", "signer_upn": "p@o.com", "signer_name": "P"},
        )
        assert resp.status_code == 404

    @pytest.mark.asyncio
    async def test_invalid_role_422(self, auth_client: AsyncClient):
        from agent.main import orchestrator

        sess = orchestrator.create_session("Sig EP Role")
        orchestrator.run_segmentation(sess.session_id, use_synthetic=True, zone_id="Z")
        resp = await auth_client.post(
            f"/sessions/{sess.session_id}/nqa1-signatures",
            json={"role": "chief_dragon", "signer_upn": "x@o.com", "signer_name": "X"},
        )
        assert resp.status_code == 422

    @pytest.mark.asyncio
    async def test_sign_verifier_happy_path(self, auth_client: AsyncClient):
        from agent.main import orchestrator

        sess = orchestrator.create_session("Sig EP Happy")
        orchestrator.run_segmentation(sess.session_id, use_synthetic=True, zone_id="Z")

        # Trigger auto-preparer by fetching the package once
        pkg = await auth_client.get(f"/sessions/{sess.session_id}/nqa1-package")
        assert pkg.status_code == 200

        resp = await auth_client.post(
            f"/sessions/{sess.session_id}/nqa1-signatures",
            json={"role": "verifier", "signer_upn": "verif@o.com", "signer_name": "Ver Ifier"},
        )
        assert resp.status_code == 200
        body = resp.json()
        assert body["role"] == "verifier"
        assert body["signature_hash"] is not None

    @pytest.mark.asyncio
    async def test_sign_out_of_order_409(self, auth_client: AsyncClient):
        from agent.main import orchestrator

        sess = orchestrator.create_session("Sig EP Order")
        orchestrator.run_segmentation(sess.session_id, use_synthetic=True, zone_id="Z")

        # Don't fetch package — preparer NOT yet signed
        # Skip straight to verifier → 409
        resp = await auth_client.post(
            f"/sessions/{sess.session_id}/nqa1-signatures",
            json={"role": "approver", "signer_upn": "app@o.com", "signer_name": "A"},
        )
        assert resp.status_code == 409


class TestListEndpoint:
    @pytest.mark.asyncio
    async def test_list_returns_status_and_signatures(self, auth_client: AsyncClient):
        from agent.main import orchestrator

        sess = orchestrator.create_session("Sig EP List")
        orchestrator.run_segmentation(sess.session_id, use_synthetic=True, zone_id="Z")
        await auth_client.get(f"/sessions/{sess.session_id}/nqa1-package")  # auto-preparer

        resp = await auth_client.get(
            f"/sessions/{sess.session_id}/nqa1-signatures",
        )
        assert resp.status_code == 200
        body = resp.json()
        assert "fully_signed" in body
        assert "by_role" in body
        assert "signatures" in body
        assert len(body["signatures"]) >= 1


class TestRevokeEndpoint:
    @pytest.mark.asyncio
    async def test_revoke_unknown_signature_404(self, auth_client: AsyncClient):
        resp = await auth_client.patch(
            "/nqa1-signatures/does-not-exist/revoke",
            json={"reason": "test"},
        )
        assert resp.status_code == 404

    @pytest.mark.asyncio
    async def test_revoke_happy_path(self, auth_client: AsyncClient):
        from agent.main import nqa1_signatures, orchestrator

        sess = orchestrator.create_session("Sig EP Revoke")
        orchestrator.run_segmentation(sess.session_id, use_synthetic=True, zone_id="Z")
        await auth_client.get(f"/sessions/{sess.session_id}/nqa1-package")

        sigs = nqa1_signatures.list_session(sess.session_id)
        sig_id = sigs[0].signature_id

        resp = await auth_client.patch(
            f"/nqa1-signatures/{sig_id}/revoke",
            json={"reason": "regression test"},
        )
        assert resp.status_code == 200
        assert resp.json()["revoked_at_utc"] is not None


# ═════════════════════════════════════════════════════════════════════════════
# 8. HTML rendering
# ═════════════════════════════════════════════════════════════════════════════


class TestNqa1SignatureHtml:
    def test_unsigned_role_shows_line(self):
        from agent.tools.nqa1_package import _render_signature_block

        html = _render_signature_block(
            {
                "fully_signed": False,
                "next_required": "preparer",
                "chain_valid": True,
                "chain_broken": [],
                "signature_count": 0,
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
            }
        )
        assert "Preparer" in html
        assert "unsigned" in html
        assert "Next required signature" in html

    def test_signed_role_shows_tick_and_name(self):
        from agent.tools.nqa1_package import _render_signature_block

        html = _render_signature_block(
            {
                "fully_signed": False,
                "next_required": "verifier",
                "chain_valid": True,
                "chain_broken": [],
                "signature_count": 1,
                "by_role": {
                    "preparer": {
                        "signed": True,
                        "signer_upn": "p@o.com",
                        "signer_name": "Alice",
                        "timestamp_utc": "2026-04-19T12:00:00+00:00",
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
            }
        )
        assert "✓ Alice" in html
        assert "p@o.com" in html

    def test_chain_broken_banner(self):
        from agent.tools.nqa1_package import _render_signature_block

        html = _render_signature_block(
            {
                "fully_signed": False,
                "next_required": "approver",
                "chain_valid": False,
                "chain_broken": [{"index": 0, "reason": "hash_mismatch"}],
                "signature_count": 2,
                "by_role": {
                    "preparer": {
                        "signed": True,
                        "signer_upn": "p@o.com",
                        "signer_name": "P",
                        "timestamp_utc": "2026-01-01T00:00:00+00:00",
                    },
                    "verifier": {
                        "signed": True,
                        "signer_upn": "v@o.com",
                        "signer_name": "V",
                        "timestamp_utc": "2026-01-02T00:00:00+00:00",
                    },
                    "approver": {
                        "signed": False,
                        "signer_upn": None,
                        "signer_name": None,
                        "timestamp_utc": None,
                    },
                },
            }
        )
        assert "Chain integrity broken" in html

    def test_fully_signed_banner(self):
        from agent.tools.nqa1_package import _render_signature_block

        html = _render_signature_block(
            {
                "fully_signed": True,
                "next_required": None,
                "chain_valid": True,
                "chain_broken": [],
                "signature_count": 3,
                "by_role": {
                    r: {
                        "signed": True,
                        "signer_upn": f"{r}@o.com",
                        "signer_name": r.capitalize(),
                        "timestamp_utc": "2026-01-01T00:00:00+00:00",
                    }
                    for r in ("preparer", "verifier", "approver")
                },
            }
        )
        assert "fully signed" in html.lower()
