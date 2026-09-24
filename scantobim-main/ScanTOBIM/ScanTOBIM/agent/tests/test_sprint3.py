"""test_sprint3.py — Tests for Sprint 3: Web Frontend backend endpoints.

Tests cover:
  - GET  /sessions              — list all sessions, requires auth
  - POST /sessions/{id}/notify-teams — Teams Adaptive Card, requires TEAMS_WEBHOOK_URL
  - GET  /               — root redirects to /ui/login.html
  - GET  /ui/login.html  — static file served
  - _post_teams_card     — Adaptive Card payload structure (unit test, no HTTP)
"""

from __future__ import annotations

import asyncio
import uuid
from unittest.mock import MagicMock, patch

import pytest

# ── Helpers ───────────────────────────────────────────────────────────────────


def _run(coro):
    return asyncio.get_event_loop().run_until_complete(coro)


@pytest.fixture
def auth_headers(client):
    """Bearer token for admin user."""

    async def _get():
        resp = await client.post(
            "/auth/token",
            data={"username": "admin", "password": "admin123"},
            headers={"Content-Type": "application/x-www-form-urlencoded"},
        )
        assert resp.status_code == 200
        return {"Authorization": f"Bearer {resp.json()['access_token']}"}

    return _run(_get())


@pytest.fixture
def session_id(client):
    """Create a real session and return its ID."""

    async def _create():
        resp = await client.post(
            "/sessions",
            json={"site_name": "Sprint3 Test Site", "use_synthetic": True},
        )
        assert resp.status_code == 200
        return resp.json()["session_id"]

    return _run(_create())


# ─────────────────────────────────────────────────────────────────────────────
# 1. GET /sessions — list all sessions
# ─────────────────────────────────────────────────────────────────────────────


class TestListSessions:
    @pytest.mark.asyncio
    async def test_list_sessions_requires_auth(self, client):
        """401 without token."""
        resp = await client.get("/sessions")
        assert resp.status_code == 401

    @pytest.mark.asyncio
    async def test_list_sessions_returns_list(self, client, auth_headers, session_id):
        """Returns sessions array with total count."""
        resp = await client.get("/sessions", headers=auth_headers)

        assert resp.status_code == 200
        data = resp.json()
        assert "sessions" in data
        assert "total" in data
        assert isinstance(data["sessions"], list)
        assert data["total"] == len(data["sessions"])

    @pytest.mark.asyncio
    async def test_list_sessions_includes_created_session(self, client, auth_headers, session_id):
        """The session we just created appears in the list."""
        resp = await client.get("/sessions", headers=auth_headers)

        assert resp.status_code == 200
        ids = [s["session_id"] for s in resp.json()["sessions"]]
        assert session_id in ids

    @pytest.mark.asyncio
    async def test_list_sessions_has_required_fields(self, client, auth_headers, session_id):
        """Each session object has the fields the frontend needs."""
        resp = await client.get("/sessions", headers=auth_headers)

        assert resp.status_code == 200
        sessions = resp.json()["sessions"]
        assert len(sessions) > 0
        s = next(x for x in sessions if x["session_id"] == session_id)

        assert "session_id" in s
        assert "site_name" in s
        assert "cde_state" in s
        assert "total_elements_created" in s

    @pytest.mark.asyncio
    async def test_list_sessions_sorted_most_recent_first(self, client, auth_headers):
        """Sessions are returned with most-recent first (created_at descending)."""
        # Create two sessions in sequence
        s1 = await client.post("/sessions", json={"site_name": "Alpha", "use_synthetic": True})
        s2 = await client.post("/sessions", json={"site_name": "Beta", "use_synthetic": True})
        id1 = s1.json()["session_id"]
        id2 = s2.json()["session_id"]

        resp = await client.get("/sessions", headers=auth_headers)
        ids = [s["session_id"] for s in resp.json()["sessions"]]

        # Beta (id2) was created later — should appear before Alpha (id1)
        assert ids.index(id2) < ids.index(id1)

    @pytest.mark.asyncio
    async def test_list_sessions_cde_state_is_valid(self, client, auth_headers, session_id):
        """CDE state is one of the known valid values."""
        resp = await client.get("/sessions", headers=auth_headers)
        sessions = resp.json()["sessions"]
        s = next((x for x in sessions if x["session_id"] == session_id), None)
        assert s is not None
        assert s["cde_state"] in ("WIP", "SHARED", "PUBLISHED", "FAILED")


# ─────────────────────────────────────────────────────────────────────────────
# 2. POST /sessions/{id}/notify-teams
# ─────────────────────────────────────────────────────────────────────────────


class TestNotifyTeams:
    @pytest.mark.asyncio
    async def test_notify_teams_requires_auth(self, client, session_id):
        """401 without token."""
        resp = await client.post(f"/sessions/{session_id}/notify-teams")
        assert resp.status_code == 401

    @pytest.mark.asyncio
    async def test_notify_teams_unknown_session_returns_404(self, client, auth_headers):
        """404 for a non-existent session."""
        resp = await client.post(
            "/sessions/no-such-session/notify-teams",
            headers=auth_headers,
        )
        assert resp.status_code == 404

    @pytest.mark.asyncio
    async def test_notify_teams_returns_400_without_webhook_url(
        self, client, auth_headers, session_id
    ):
        """400 when TEAMS_WEBHOOK_URL is not configured."""
        import os

        original = os.environ.pop("TEAMS_WEBHOOK_URL", None)
        try:
            resp = await client.post(
                f"/sessions/{session_id}/notify-teams",
                headers=auth_headers,
            )
            assert resp.status_code == 400
            assert "TEAMS_WEBHOOK_URL" in resp.json()["detail"]
        finally:
            if original:
                os.environ["TEAMS_WEBHOOK_URL"] = original

    @pytest.mark.asyncio
    async def test_notify_teams_returns_zero_sent_when_no_pending_gates(
        self, client, auth_headers, session_id
    ):
        """When no SC2 gates are pending, returns sent=0 gracefully."""
        import os

        from agent.main import safety_gate

        os.environ["TEAMS_WEBHOOK_URL"] = "http://fake-teams-webhook.example.com"
        # Stash and clear pending gates so this session has none
        original_gates = dict(safety_gate._pending_gates)
        safety_gate._pending_gates.clear()
        try:
            resp = await client.post(
                f"/sessions/{session_id}/notify-teams",
                headers=auth_headers,
            )
            assert resp.status_code == 200
            data = resp.json()
            assert data["sent"] == 0
            assert "No pending" in data["message"]
        finally:
            del os.environ["TEAMS_WEBHOOK_URL"]
            safety_gate._pending_gates.update(original_gates)

    @pytest.mark.asyncio
    async def test_notify_teams_calls_post_teams_card_per_gate(
        self, client, auth_headers, session_id
    ):
        """When pending SC2 gates exist, _post_teams_card is called for each."""
        import os

        from agent.main import safety_gate
        from agent.models import SafetyGateRequest, SafetyGateStatus

        # Inject a fake pending SC2 gate for this session
        fake_gate = SafetyGateRequest(
            gate_id=str(uuid.uuid4()),
            segment_id=str(uuid.uuid4()),
            session_id=session_id,
            element_type="pipe",
            safety_category="SC2",
            zone_id="zone-sc2",
            classification_reason="test SC2 gate",
            status=SafetyGateStatus.PENDING,
        )
        safety_gate._pending_gates[fake_gate.gate_id] = fake_gate

        os.environ["TEAMS_WEBHOOK_URL"] = "http://fake-teams-webhook.example.com"
        try:
            with patch("agent.main._post_teams_card", return_value=True) as mock_post:
                resp = await client.post(
                    f"/sessions/{session_id}/notify-teams",
                    headers=auth_headers,
                )

            assert resp.status_code == 200
            data = resp.json()
            assert data["sent"] >= 1
            assert data["total_pending"] >= 1
            assert "gates_url" in data

            # Verify _post_teams_card was called with correct Adaptive Card args
            mock_post.assert_called()
            call_kwargs = mock_post.call_args.kwargs
            assert call_kwargs["webhook_url"] == "http://fake-teams-webhook.example.com"
            assert "approve_url" in call_kwargs
            assert "reject_url" in call_kwargs
            assert fake_gate.gate_id in call_kwargs["approve_url"]
            assert "action=approve" in call_kwargs["approve_url"]
            assert "action=reject" in call_kwargs["reject_url"]

        finally:
            del os.environ["TEAMS_WEBHOOK_URL"]
            safety_gate._pending_gates.pop(fake_gate.gate_id, None)

    @pytest.mark.asyncio
    async def test_notify_teams_approve_url_contains_gate_id(
        self, client, auth_headers, session_id
    ):
        """Deep-link URLs include gate_id so gates.html can auto-open modal."""
        import os

        from agent.main import safety_gate
        from agent.models import SafetyGateRequest, SafetyGateStatus

        fake_gate = SafetyGateRequest(
            gate_id=str(uuid.uuid4()),
            segment_id=str(uuid.uuid4()),
            session_id=session_id,
            element_type="wall",
            safety_category="SC2",
            zone_id="zone-wall",
            classification_reason="test SC2 wall gate",
            status=SafetyGateStatus.PENDING,
        )
        safety_gate._pending_gates[fake_gate.gate_id] = fake_gate
        os.environ["TEAMS_WEBHOOK_URL"] = "http://fake-webhook"

        try:
            with patch("agent.main._post_teams_card", return_value=True) as mock_post:
                await client.post(f"/sessions/{session_id}/notify-teams", headers=auth_headers)

            kw = mock_post.call_args.kwargs
            assert fake_gate.gate_id in kw["approve_url"]
            assert fake_gate.gate_id in kw["reject_url"]
            assert kw["site_name"] == "Sprint3 Test Site"
            assert kw["element_type"] == "wall"
        finally:
            del os.environ["TEAMS_WEBHOOK_URL"]
            safety_gate._pending_gates.pop(fake_gate.gate_id, None)


# ─────────────────────────────────────────────────────────────────────────────
# 3. GET / — root redirect to login
# ─────────────────────────────────────────────────────────────────────────────


class TestRootRedirect:
    @pytest.mark.asyncio
    async def test_root_redirects_to_login(self, client):
        """GET / returns a redirect to /ui/login.html."""
        resp = await client.get("/", follow_redirects=False)
        assert resp.status_code in (301, 302, 307, 308)
        assert "/ui/login.html" in resp.headers.get("location", "")


# ─────────────────────────────────────────────────────────────────────────────
# 4. _post_teams_card unit tests (no network, validates Adaptive Card shape)
# ─────────────────────────────────────────────────────────────────────────────


class TestPostTeamsCard:
    def _call_with_mock(self, status=200):
        """Call _post_teams_card with a mock HTTP response."""
        import urllib.request as _req

        from agent.main import _post_teams_card

        mock_resp = MagicMock()
        mock_resp.status = status

        captured_payload = {}

        def fake_urlopen(request, timeout):
            import json

            captured_payload["data"] = json.loads(request.data)
            return mock_resp

        with patch.object(_req, "urlopen", side_effect=fake_urlopen):
            result = _post_teams_card(
                webhook_url="http://fake",
                title="Test Title",
                site_name="Test Site",
                element_type="pipe",
                zone_id="zone-001",
                gate_id="aabbccdd-1234-5678-abcd-ef1234567890",
                approve_url="http://localhost:8765/ui/gates.html?action=approve",
                reject_url="http://localhost:8765/ui/gates.html?action=reject",
            )
        return result, captured_payload.get("data", {})

    def test_returns_true_on_200(self):
        result, _ = self._call_with_mock(status=200)
        assert result is True

    def test_returns_false_on_non_200(self):
        result, _ = self._call_with_mock(status=400)
        assert result is False

    def test_payload_is_adaptive_card_format(self):
        """Payload must use Teams 'message' type with Adaptive Card attachment."""
        _, payload = self._call_with_mock()
        assert payload.get("type") == "message"
        attachments = payload.get("attachments", [])
        assert len(attachments) == 1
        assert attachments[0]["contentType"] == "application/vnd.microsoft.card.adaptive"

    def test_adaptive_card_has_approve_reject_actions(self):
        """Card must have two Action.OpenUrl actions (Approve + Reject)."""
        _, payload = self._call_with_mock()
        card = payload["attachments"][0]["content"]
        actions = card.get("actions", [])
        assert len(actions) == 2
        types = [a["type"] for a in actions]
        assert all(t == "Action.OpenUrl" for t in types)
        urls = [a["url"] for a in actions]
        assert any("action=approve" in u for u in urls)
        assert any("action=reject" in u for u in urls)

    def test_adaptive_card_contains_fact_set(self):
        """FactSet must include Site, Zone, Element, Gate ID fields."""
        _, payload = self._call_with_mock()
        card = payload["attachments"][0]["content"]
        facts = []
        for block in card.get("body", []):
            if block.get("type") == "FactSet":
                facts = [f["title"] for f in block.get("facts", [])]
                break
        assert "Site" in facts
        assert "Zone" in facts
        assert "Element" in facts
        assert "Gate ID" in facts

    def test_returns_false_on_connection_error(self):
        """Returns False gracefully when Teams webhook is unreachable."""
        import urllib.error

        from agent.main import _post_teams_card

        with patch("urllib.request.urlopen", side_effect=urllib.error.URLError("refused")):
            result = _post_teams_card(
                webhook_url="http://unreachable",
                title="T",
                site_name="S",
                element_type="pipe",
                zone_id="z",
                gate_id="g",
                approve_url="http://a",
                reject_url="http://r",
            )
        assert result is False
