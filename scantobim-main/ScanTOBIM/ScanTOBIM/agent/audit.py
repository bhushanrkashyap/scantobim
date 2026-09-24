"""Append-only audit ledger with HMAC chain for tamper detection.

PoV uses SQLite. Production would use Cosmos DB with append-only stored procedures.
"""

from __future__ import annotations

import json
import os
import sqlite3
from datetime import datetime
from pathlib import Path

import structlog

from agent.models import AuditEvent, AuditEventType

logger = structlog.get_logger()

DB_PATH = Path(os.getenv("DATABASE_URL", "sqlite:///./scantobim.db").replace("sqlite:///", ""))
HMAC_SECRET = os.getenv("AUDIT_HMAC_SECRET", "pov-secret-change-me")


class AuditLedger:
    """Immutable, HMAC-chained audit ledger."""

    def __init__(self, db_path: Path | None = None, secret: str | None = None):
        self.db_path = db_path or DB_PATH
        self.secret = secret or HMAC_SECRET
        self._last_hash: dict[str, str] = {}  # session_id → last hash
        self._init_db()

    def _init_db(self):
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS audit_events (
                    event_id TEXT PRIMARY KEY,
                    session_id TEXT NOT NULL,
                    event_type TEXT NOT NULL,
                    timestamp_utc TEXT NOT NULL,
                    zone_id TEXT,
                    segment_id TEXT,
                    element_id TEXT,
                    actor TEXT NOT NULL,
                    detail TEXT NOT NULL DEFAULT '{}',
                    previous_hash TEXT,
                    event_hash TEXT NOT NULL
                )
            """)
            conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_audit_session
                ON audit_events(session_id, timestamp_utc)
            """)
            conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_audit_element
                ON audit_events(element_id)
            """)

    def _get_last_hash(self, session_id: str) -> str | None:
        if session_id in self._last_hash:
            return self._last_hash[session_id]
        # Recover from DB on cold start
        with sqlite3.connect(self.db_path) as conn:
            row = conn.execute(
                "SELECT event_hash FROM audit_events WHERE session_id=? ORDER BY timestamp_utc DESC LIMIT 1",
                (session_id,),
            ).fetchone()
        if row:
            self._last_hash[session_id] = row[0]
            return row[0]
        return None

    def append(self, event: AuditEvent) -> AuditEvent:
        """Append an event to the ledger with HMAC chain linking.

        AuditEvent is frozen — use model_copy to build the chained version.
        """
        previous_hash = self._get_last_hash(event.session_id)
        # Build chain: set previous_hash first, then compute + set event_hash
        event = event.model_copy(update={"previous_hash": previous_hash})
        event = event.model_copy(update={"event_hash": event.compute_hash(self.secret)})
        self._last_hash[event.session_id] = event.event_hash

        with sqlite3.connect(self.db_path) as conn:
            conn.execute(
                """INSERT INTO audit_events
                   (event_id, session_id, event_type, timestamp_utc, zone_id,
                    segment_id, element_id, actor, detail, previous_hash, event_hash)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                (
                    event.event_id,
                    event.session_id,
                    event.event_type.value,
                    event.timestamp_utc.isoformat(),
                    event.zone_id,
                    event.segment_id,
                    event.element_id,
                    event.actor,
                    json.dumps(event.detail),
                    event.previous_hash,
                    event.event_hash,
                ),
            )

        logger.info(
            "audit_appended",
            event_type=event.event_type.value,
            session_id=event.session_id[:8],
            element_id=event.element_id,
        )
        return event

    def get_session_events(self, session_id: str) -> list[AuditEvent]:
        """Retrieve all events for a session, ordered by time."""
        with sqlite3.connect(self.db_path) as conn:
            rows = conn.execute(
                "SELECT * FROM audit_events WHERE session_id=? ORDER BY timestamp_utc",
                (session_id,),
            ).fetchall()
        return [self._row_to_event(r) for r in rows]

    def get_element_provenance(self, element_id: str) -> list[AuditEvent]:
        """Get all audit events for a specific element."""
        with sqlite3.connect(self.db_path) as conn:
            rows = conn.execute(
                "SELECT * FROM audit_events WHERE element_id=? ORDER BY timestamp_utc",
                (element_id,),
            ).fetchall()
        return [self._row_to_event(r) for r in rows]

    def verify_chain(self, session_id: str) -> dict:
        """Verify HMAC chain integrity for a session. Returns {valid, broken_links}."""
        events = self.get_session_events(session_id)
        if not events:
            return {"valid": True, "broken_links": [], "event_count": 0}

        broken = []
        for i, event in enumerate(events):
            expected_hash = event.compute_hash(self.secret)
            if event.event_hash != expected_hash:
                broken.append({"event_id": event.event_id, "index": i, "reason": "hash_mismatch"})

            if i > 0 and event.previous_hash != events[i - 1].event_hash:
                broken.append({"event_id": event.event_id, "index": i, "reason": "chain_break"})

        result = {"valid": len(broken) == 0, "broken_links": broken, "event_count": len(events)}
        logger.info("chain_verified", session_id=session_id[:8], **result)
        return result

    def get_stats(self, session_id: str) -> dict:
        """Get audit statistics for a session."""
        with sqlite3.connect(self.db_path) as conn:
            total = conn.execute(
                "SELECT COUNT(*) FROM audit_events WHERE session_id=?", (session_id,)
            ).fetchone()[0]

            by_type = conn.execute(
                "SELECT event_type, COUNT(*) FROM audit_events WHERE session_id=? GROUP BY event_type",
                (session_id,),
            ).fetchall()

        return {
            "session_id": session_id,
            "total_events": total,
            "by_type": {row[0]: row[1] for row in by_type},
        }

    @staticmethod
    def _row_to_event(row: tuple) -> AuditEvent:
        return AuditEvent(
            event_id=row[0],
            session_id=row[1],
            event_type=AuditEventType(row[2]),
            timestamp_utc=datetime.fromisoformat(row[3]),
            zone_id=row[4],
            segment_id=row[5],
            element_id=row[6],
            actor=row[7],
            detail=json.loads(row[8]),
            previous_hash=row[9],
            event_hash=row[10],
        )
