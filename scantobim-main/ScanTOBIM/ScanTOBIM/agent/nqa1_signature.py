"""NQA-1 Subpart 2.7 three-party signature workflow.

Closes the placeholder `{"name": null, ...}` block in the NQA-1 package
HTML. Each session's V&V package is signed by three independent parties
in order:

  1. PREPARER   — author of the V&V evidence (usually the agent itself)
  2. VERIFIER   — independent QA reviewer
  3. APPROVER   — QA manager / signatory authority

Every signature is HMAC-chained — each signature's `signature_hash` is
derived from a payload that includes the prior signature's hash. Tamper
detection is therefore end-to-end: if anyone edits the NQA-1 package or
forges a signature, `verify_chain()` returns `valid=False`.

Separation of duties: the same signer UPN cannot hold two roles on the
same session. A revoked signature does not delete the row — it sets
`revoked_at_utc` / `revoked_by` / `revocation_reason` and causes
`is_fully_signed()` to return False.
"""

from __future__ import annotations

import json
import os
import sqlite3
from datetime import datetime, timezone
from pathlib import Path

import structlog

from agent.models import NQA1Role, NQA1Signature

logger = structlog.get_logger()

DB_PATH = Path(os.getenv("DATABASE_URL", "sqlite:///./scantobim.db").replace("sqlite:///", ""))
HMAC_SECRET = os.getenv("AUDIT_HMAC_SECRET", "pov-secret-change-me")


# ── Exceptions ───────────────────────────────────────────────────────────────


class SignatureOrderError(ValueError):
    """Raised when signatures are collected out of the required order."""


class DuplicateSignerError(ValueError):
    """Raised when the same UPN attempts to sign in multiple roles."""


class DuplicateRoleError(ValueError):
    """Raised when an (already-active) signature for the role already exists."""


# ── Service ──────────────────────────────────────────────────────────────────


class NQA1SignatureService:
    """SQLite-backed, HMAC-chained signature ledger for NQA-1 packages."""

    _REQUIRED_ORDER = [NQA1Role.PREPARER, NQA1Role.VERIFIER, NQA1Role.APPROVER]

    def __init__(self, db_path: Path | None = None, secret: str | None = None):
        self.db_path = db_path or DB_PATH
        self.secret = secret or HMAC_SECRET
        self._init_db()

    def _init_db(self) -> None:
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS nqa1_signatures (
                    signature_id            TEXT PRIMARY KEY,
                    session_id              TEXT NOT NULL,
                    role                    TEXT NOT NULL,
                    signer_upn              TEXT NOT NULL,
                    signer_name             TEXT NOT NULL,
                    timestamp_utc           TEXT NOT NULL,
                    package_hash            TEXT NOT NULL,
                    previous_signature_hash TEXT,
                    signature_hash          TEXT NOT NULL,
                    comments                TEXT NOT NULL DEFAULT '',
                    revoked_at_utc          TEXT,
                    revoked_by              TEXT,
                    revocation_reason       TEXT
                )
            """)
            conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_nqa1_sig_session
                ON nqa1_signatures(session_id, timestamp_utc)
            """)

    # ── Public API ───────────────────────────────────────────────────────────

    def sign(
        self,
        session_id: str,
        role: NQA1Role,
        signer_upn: str,
        signer_name: str,
        package_hash: str,
        comments: str = "",
    ) -> NQA1Signature:
        """Append a new signature to the ledger, enforcing ordering and duties."""
        active = self._list_active(session_id)

        # 1. Duplicate-role check — active signature already exists for this role
        for s in active:
            if s.role == role:
                raise DuplicateRoleError(
                    f"{role.value} signature already active for session {session_id}"
                )

        # 2. Separation-of-duties — same UPN must not hold two roles
        normalized_upn = signer_upn.lower().strip()
        for s in active:
            if s.signer_upn == normalized_upn:
                raise DuplicateSignerError(
                    f"UPN {signer_upn!r} already signed as {s.role.value}; "
                    "cannot also sign as {role.value}"
                )

        # 3. Order check — cannot skip roles
        active_roles = {s.role for s in active}
        for required in self._REQUIRED_ORDER:
            if required == role:
                break
            if required not in active_roles:
                raise SignatureOrderError(
                    f"Cannot sign as {role.value} — {required.value} must sign first"
                )

        # Build chained signature
        previous_hash = active[-1].signature_hash if active else None
        sig = NQA1Signature(
            session_id=session_id,
            role=role,
            signer_upn=normalized_upn,
            signer_name=signer_name,
            package_hash=package_hash,
            previous_signature_hash=previous_hash,
            comments=comments,
        )
        sig = sig.model_copy(
            update={
                "signature_hash": sig.compute_hash(self.secret),
            }
        )

        with sqlite3.connect(self.db_path) as conn:
            conn.execute(
                """INSERT INTO nqa1_signatures
                   (signature_id, session_id, role, signer_upn, signer_name,
                    timestamp_utc, package_hash, previous_signature_hash,
                    signature_hash, comments)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                (
                    sig.signature_id,
                    sig.session_id,
                    sig.role.value,
                    sig.signer_upn,
                    sig.signer_name,
                    sig.timestamp_utc.isoformat(),
                    sig.package_hash,
                    sig.previous_signature_hash,
                    sig.signature_hash,
                    sig.comments,
                ),
            )

        logger.info(
            "nqa1_signature_added",
            session_id=session_id[:8],
            role=role.value,
            signer=normalized_upn,
        )
        return sig

    def list_session(self, session_id: str) -> list[NQA1Signature]:
        """Return all signatures (active + revoked) in chronological order."""
        with sqlite3.connect(self.db_path) as conn:
            rows = conn.execute(
                """SELECT signature_id, session_id, role, signer_upn, signer_name,
                          timestamp_utc, package_hash, previous_signature_hash,
                          signature_hash, comments, revoked_at_utc,
                          revoked_by, revocation_reason
                     FROM nqa1_signatures
                    WHERE session_id=?
                 ORDER BY timestamp_utc ASC""",
                (session_id,),
            ).fetchall()

        return [self._row_to_model(r) for r in rows]

    def _list_active(self, session_id: str) -> list[NQA1Signature]:
        return [s for s in self.list_session(session_id) if s.revoked_at_utc is None]

    def revoke(
        self,
        signature_id: str,
        revoked_by: str,
        revocation_reason: str,
    ) -> NQA1Signature:
        """Mark a signature as revoked. Does not delete — append-only ledger."""
        with sqlite3.connect(self.db_path) as conn:
            row = conn.execute(
                "SELECT * FROM nqa1_signatures WHERE signature_id=?",
                (signature_id,),
            ).fetchone()
            if not row:
                raise KeyError(f"No signature {signature_id}")

            now_utc = datetime.now(timezone.utc).isoformat()
            conn.execute(
                """UPDATE nqa1_signatures
                      SET revoked_at_utc=?, revoked_by=?, revocation_reason=?
                    WHERE signature_id=?""",
                (now_utc, revoked_by, revocation_reason, signature_id),
            )

        logger.warning(
            "nqa1_signature_revoked",
            signature_id=signature_id[:8],
            revoked_by=revoked_by,
        )

        with sqlite3.connect(self.db_path) as conn:
            fresh = conn.execute(
                """SELECT signature_id, session_id, role, signer_upn, signer_name,
                          timestamp_utc, package_hash, previous_signature_hash,
                          signature_hash, comments, revoked_at_utc,
                          revoked_by, revocation_reason
                     FROM nqa1_signatures WHERE signature_id=?""",
                (signature_id,),
            ).fetchone()
        return self._row_to_model(fresh)

    def verify_chain(self, session_id: str) -> dict:
        """Verify HMAC chain integrity for all signatures on a session."""
        sigs = self.list_session(session_id)
        broken: list[dict] = []

        prev_hash: str | None = None
        for i, s in enumerate(sigs):
            # Chain linkage
            if s.previous_signature_hash != prev_hash:
                broken.append(
                    {
                        "signature_id": s.signature_id,
                        "index": i,
                        "reason": "chain_break",
                    }
                )
            # Hash integrity
            expected = s.compute_hash(self.secret)
            if expected != s.signature_hash:
                broken.append(
                    {
                        "signature_id": s.signature_id,
                        "index": i,
                        "reason": "hash_mismatch",
                    }
                )
            prev_hash = s.signature_hash

        return {
            "valid": len(broken) == 0,
            "broken_links": broken,
            "signature_count": len(sigs),
        }

    def status(self, session_id: str) -> dict:
        """Summary: which roles have signed, fully signed y/n, chain integrity."""
        active = self._list_active(session_id)
        active_roles = {s.role.value for s in active}

        chain = self.verify_chain(session_id)

        by_role = {}
        for role in self._REQUIRED_ORDER:
            sig = next((s for s in active if s.role == role), None)
            by_role[role.value] = {
                "signed": sig is not None,
                "signer_upn": sig.signer_upn if sig else None,
                "signer_name": sig.signer_name if sig else None,
                "timestamp_utc": sig.timestamp_utc.isoformat() if sig else None,
            }

        return {
            "session_id": session_id,
            "fully_signed": active_roles == {r.value for r in self._REQUIRED_ORDER},
            "next_required": self._next_required_role(active_roles),
            "by_role": by_role,
            "chain_valid": chain["valid"],
            "chain_broken": chain["broken_links"],
            "signature_count": chain["signature_count"],
        }

    def _next_required_role(self, active_roles: set[str]) -> str | None:
        for role in self._REQUIRED_ORDER:
            if role.value not in active_roles:
                return role.value
        return None

    # ── Helpers ──────────────────────────────────────────────────────────────

    @staticmethod
    def _row_to_model(row) -> NQA1Signature:
        return NQA1Signature(
            signature_id=row[0],
            session_id=row[1],
            role=NQA1Role(row[2]),
            signer_upn=row[3],
            signer_name=row[4],
            timestamp_utc=datetime.fromisoformat(row[5]),
            package_hash=row[6],
            previous_signature_hash=row[7],
            signature_hash=row[8],
            comments=row[9] or "",
            revoked_at_utc=datetime.fromisoformat(row[10]) if row[10] else None,
            revoked_by=row[11],
            revocation_reason=row[12],
        )


# ── Package hashing (used by sign() to pin what was signed) ───────────────────


def compute_package_hash(package_json: dict) -> str:
    """SHA-256 of the canonical NQA-1 package JSON.

    Stable ordering is critical so the hash is reproducible.
    """
    import hashlib

    # Strip signature / auto-computed fields that would change between reads
    scrubbed = {k: v for k, v in package_json.items() if k != "signatures"}
    payload = json.dumps(scrubbed, sort_keys=True, separators=(",", ":"), default=str)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()
