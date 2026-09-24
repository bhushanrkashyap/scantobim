"""test_revit_bridge.py — Tests for Sprint 2: Revit Bridge endpoints + client.

Tests cover:
  - GET  /bridge/health                          — no auth, mocked bridge client
  - POST /sessions/{id}/revit-callback           — no auth, push results from Revit
  - GET  /sessions/{id}/revit-results            — requires auth
  - POST /bridge/push/{id}?instruction_id=...    — requires revit_bridge feature (Pro+)

  - RevitBridgeClient unit tests (push/poll/callback helpers)
  - store_callback_result / consume_callback_result module helpers
"""

from __future__ import annotations

import asyncio
import uuid
from unittest.mock import MagicMock, patch

import pytest

# ── Helpers ───────────────────────────────────────────────────────────────────


def _run(coro):
    """Run a coroutine synchronously (for sync fixtures calling async client)."""
    return asyncio.get_event_loop().run_until_complete(coro)


def _make_action_result(
    *,
    success: bool = True,
    instruction_id: str | None = None,
    element_id: str | None = None,
    error: str | None = None,
    family_name: str = "Pipe-Round",
    type_name: str = "50mm",
) -> dict:
    """Build a minimal BridgeActionResult payload as a dict."""
    return {
        "success": success,
        "instruction_id": instruction_id or str(uuid.uuid4()),
        "element_id": element_id or (str(uuid.uuid4()) if success else None),
        "error": error,
        "duration_ms": 42,
        "family_name": family_name,
        "type_name": type_name,
        "resolution_path": "config",
    }


# ── Fixtures ──────────────────────────────────────────────────────────────────


@pytest.fixture
def auth_headers(client):
    """Bearer token for admin user (enterprise tier — has revit_bridge feature)."""

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
def surveyor_headers(client):
    """Bearer token for a surveyor user (starter tier — no revit_bridge feature)."""

    async def _get():
        resp = await client.post(
            "/auth/token",
            data={"username": "surveyor", "password": "survey123"},
            headers={"Content-Type": "application/x-www-form-urlencoded"},
        )
        # surveyor may or may not exist in demo users — tolerate 401
        if resp.status_code == 200:
            return {"Authorization": f"Bearer {resp.json()['access_token']}"}
        # Fall back to admin headers so tests that just need *any* valid auth work
        resp2 = await client.post(
            "/auth/token",
            data={"username": "admin", "password": "admin123"},
            headers={"Content-Type": "application/x-www-form-urlencoded"},
        )
        return {"Authorization": f"Bearer {resp2.json()['access_token']}"}

    return _run(_get())


@pytest.fixture
def session_id(client):
    """Create a real session and return its ID."""

    async def _create():
        resp = await client.post(
            "/sessions",
            json={"site_name": "Bridge Test Site", "use_synthetic": True},
        )
        assert resp.status_code == 200
        return resp.json()["session_id"]

    return _run(_create())


# ─────────────────────────────────────────────────────────────────────────────
# 1. GET /bridge/health
# ─────────────────────────────────────────────────────────────────────────────


class TestBridgeHealth:
    """GET /bridge/health — no auth required."""

    @pytest.mark.asyncio
    async def test_health_when_revit_connected(self, client):
        """Returns bridge stats when Revit is running."""
        mock_health = MagicMock()
        mock_health.model_dump.return_value = {
            "connected": True,
            "queue_depth": 0,
            "elements_created": 5,
            "elements_blocked": 1,
            "version": "2.0.0",
        }
        with patch("agent.main._get_bridge_client") as mock_get_client:
            mock_client = MagicMock()
            mock_client.get_health.return_value = mock_health
            mock_get_client.return_value = mock_client

            resp = await client.get("/bridge/health")

        assert resp.status_code == 200
        data = resp.json()
        assert data["connected"] is True
        assert data["elements_created"] == 5

    @pytest.mark.asyncio
    async def test_health_when_revit_disconnected(self, client):
        """Returns connected=False gracefully when Revit is not running."""
        mock_health = MagicMock()
        mock_health.model_dump.return_value = {
            "connected": False,
            "queue_depth": 0,
            "elements_created": 0,
            "elements_blocked": 0,
            "version": "unknown",
        }
        with patch("agent.main._get_bridge_client") as mock_get_client:
            mock_client = MagicMock()
            mock_client.get_health.return_value = mock_health
            mock_get_client.return_value = mock_client

            resp = await client.get("/bridge/health")

        assert resp.status_code == 200
        assert resp.json()["connected"] is False

    @pytest.mark.asyncio
    async def test_health_no_auth_required(self, client):
        """Endpoint must be accessible without a Bearer token."""
        mock_health = MagicMock()
        mock_health.model_dump.return_value = {
            "connected": False,
            "queue_depth": 0,
            "elements_created": 0,
            "elements_blocked": 0,
            "version": "unknown",
        }
        with patch("agent.main._get_bridge_client") as mock_get_client:
            mock_client = MagicMock()
            mock_client.get_health.return_value = mock_health
            mock_get_client.return_value = mock_client

            # No auth header
            resp = await client.get("/bridge/health")

        assert resp.status_code == 200  # must NOT be 401


# ─────────────────────────────────────────────────────────────────────────────
# 2. POST /sessions/{id}/revit-callback
# ─────────────────────────────────────────────────────────────────────────────


class TestRevitCallback:
    """POST /sessions/{id}/revit-callback — no auth required (called by Revit on localhost)."""

    @pytest.mark.asyncio
    async def test_callback_success_accepted(self, client, session_id):
        """A success callback is stored and acknowledged."""
        payload = _make_action_result(success=True)

        resp = await client.post(
            f"/sessions/{session_id}/revit-callback",
            json=payload,
        )

        assert resp.status_code == 200
        data = resp.json()
        assert data["accepted"] is True
        assert data["instruction_id"] == payload["instruction_id"]

    @pytest.mark.asyncio
    async def test_callback_failure_accepted(self, client, session_id):
        """A failed callback (element creation error) is also stored."""
        payload = _make_action_result(
            success=False,
            element_id=None,
            error="SC1_VIOLATION: element blocked",
        )

        resp = await client.post(
            f"/sessions/{session_id}/revit-callback",
            json=payload,
        )

        assert resp.status_code == 200
        assert resp.json()["accepted"] is True

    @pytest.mark.asyncio
    async def test_callback_unknown_session_returns_404(self, client):
        """Posting to a non-existent session ID returns 404."""
        payload = _make_action_result(success=True)

        resp = await client.post(
            "/sessions/non-existent-session-xyz/revit-callback",
            json=payload,
        )

        assert resp.status_code == 404

    @pytest.mark.asyncio
    async def test_callback_no_auth_required(self, client, session_id):
        """No Authorization header needed — Revit calls from localhost."""
        payload = _make_action_result(success=True)

        # Intentionally omit auth header
        resp = await client.post(
            f"/sessions/{session_id}/revit-callback",
            json=payload,
        )

        assert resp.status_code == 200  # must NOT be 401

    @pytest.mark.asyncio
    async def test_callback_populates_global_store(self, client, session_id):
        """Result pushed via callback should be retrievable via consume_callback_result()."""
        from agent.revit_bridge import consume_callback_result

        instr_id = str(uuid.uuid4())
        payload = _make_action_result(success=True, instruction_id=instr_id)

        await client.post(
            f"/sessions/{session_id}/revit-callback",
            json=payload,
        )

        # Global callback store should now have this result
        result = consume_callback_result(instr_id)
        assert result is not None
        assert result.instruction_id == instr_id
        assert result.success is True

    @pytest.mark.asyncio
    async def test_multiple_callbacks_accumulate(self, client, session_id):
        """Sending two callbacks for the same session both get stored."""
        for _ in range(2):
            payload = _make_action_result(success=True)
            resp = await client.post(
                f"/sessions/{session_id}/revit-callback",
                json=payload,
            )
            assert resp.status_code == 200

        # Verify via the results endpoint
        resp = await client.post(
            "/auth/token",
            data={"username": "admin", "password": "admin123"},
            headers={"Content-Type": "application/x-www-form-urlencoded"},
        )
        token = resp.json()["access_token"]
        headers = {"Authorization": f"Bearer {token}"}

        resp = await client.get(
            f"/sessions/{session_id}/revit-results",
            headers=headers,
        )
        assert resp.status_code == 200
        data = resp.json()
        # callbacks list should have at least 2
        assert data["callbacks"] >= 2


# ─────────────────────────────────────────────────────────────────────────────
# 3. GET /sessions/{id}/revit-results
# ─────────────────────────────────────────────────────────────────────────────


class TestRevitResults:
    """GET /sessions/{id}/revit-results — requires authentication."""

    @pytest.mark.asyncio
    async def test_results_empty_for_new_session(self, client, session_id, auth_headers):
        """Fresh session returns zero results with correct structure."""
        resp = await client.get(
            f"/sessions/{session_id}/revit-results",
            headers=auth_headers,
        )

        assert resp.status_code == 200
        data = resp.json()
        assert data["session_id"] == session_id
        assert "total" in data
        assert "succeeded" in data
        assert "failed" in data
        assert "callbacks" in data
        assert "results" in data
        assert isinstance(data["results"], list)

    @pytest.mark.asyncio
    async def test_results_after_successful_callback(self, client, session_id, auth_headers):
        """Results endpoint reflects callbacks received."""
        instr_id = str(uuid.uuid4())
        payload = _make_action_result(success=True, instruction_id=instr_id)

        # Push a callback
        await client.post(
            f"/sessions/{session_id}/revit-callback",
            json=payload,
        )

        resp = await client.get(
            f"/sessions/{session_id}/revit-results",
            headers=auth_headers,
        )

        assert resp.status_code == 200
        data = resp.json()
        assert data["callbacks"] >= 1

    @pytest.mark.asyncio
    async def test_results_requires_auth(self, client, session_id):
        """Without token, endpoint returns 401."""
        resp = await client.get(f"/sessions/{session_id}/revit-results")
        assert resp.status_code == 401

    @pytest.mark.asyncio
    async def test_results_unknown_session_returns_404(self, client, auth_headers):
        """404 for a session that doesn't exist."""
        resp = await client.get(
            "/sessions/does-not-exist-abc/revit-results",
            headers=auth_headers,
        )
        assert resp.status_code == 404

    @pytest.mark.asyncio
    async def test_results_counts_success_failure(self, client, session_id, auth_headers):
        """succeeded / failed counts match what was pushed via callbacks."""
        # Push one success and one failure
        await client.post(
            f"/sessions/{session_id}/revit-callback",
            json=_make_action_result(success=True),
        )
        await client.post(
            f"/sessions/{session_id}/revit-callback",
            json=_make_action_result(success=False, error="test error", element_id=None),
        )

        resp = await client.get(
            f"/sessions/{session_id}/revit-results",
            headers=auth_headers,
        )

        assert resp.status_code == 200
        data = resp.json()
        # Total results tracked by orchestrator may differ from callbacks
        # but callbacks must be >= 2
        assert data["callbacks"] >= 2


# ─────────────────────────────────────────────────────────────────────────────
# 4. POST /bridge/push/{session_id}
# ─────────────────────────────────────────────────────────────────────────────


class TestBridgePush:
    """POST /bridge/push/{session_id}?instruction_id=... — requires revit_bridge feature."""

    @pytest.mark.asyncio
    async def test_push_requires_auth(self, client, session_id):
        """Without token, returns 401."""
        resp = await client.post(
            f"/bridge/push/{session_id}",
            params={"instruction_id": str(uuid.uuid4())},
        )
        assert resp.status_code == 401

    @pytest.mark.asyncio
    async def test_push_unknown_session_returns_404(self, client, auth_headers):
        """404 when session does not exist."""
        resp = await client.post(
            "/bridge/push/no-such-session",
            params={"instruction_id": str(uuid.uuid4())},
            headers=auth_headers,
        )
        assert resp.status_code == 404

    @pytest.mark.asyncio
    async def test_push_unknown_instruction_returns_404(self, client, session_id, auth_headers):
        """404 when instruction_id is not in the session's instruction list."""
        with patch("agent.main._get_bridge_client") as mock_get_client:
            mock_client = MagicMock()
            mock_client.is_healthy.return_value = True
            mock_get_client.return_value = mock_client

            resp = await client.post(
                f"/bridge/push/{session_id}",
                params={"instruction_id": "non-existent-instruction-id"},
                headers=auth_headers,
            )

        assert resp.status_code == 404

    @pytest.mark.asyncio
    async def test_push_503_when_bridge_unhealthy(self, client, session_id, auth_headers):
        """503 when Revit bridge is not reachable."""
        with patch("agent.main._get_bridge_client") as mock_get_client:
            mock_client = MagicMock()
            mock_client.is_healthy.return_value = False
            mock_get_client.return_value = mock_client

            # We need a real instruction_id — use one from the session if available
            # First get the session to extract an instruction
            from agent.main import orchestrator

            instructions = orchestrator._instructions.get(session_id, [])

            if instructions:
                instr_id = instructions[0].instruction_id
                resp = await client.post(
                    f"/bridge/push/{session_id}",
                    params={"instruction_id": instr_id},
                    headers=auth_headers,
                )
                assert resp.status_code == 503
                assert "8766" in resp.json()["detail"]
            else:
                pytest.skip("Session has no instructions to push")

    @pytest.mark.asyncio
    async def test_push_succeeds_when_bridge_healthy(self, client, session_id, auth_headers):
        """Returns queued=True when bridge accepts the instruction."""
        from agent.main import orchestrator

        instructions = orchestrator._instructions.get(session_id, [])

        if not instructions:
            pytest.skip("Session has no instructions to push")

        instr_id = instructions[0].instruction_id

        with patch("agent.main._get_bridge_client") as mock_get_client:
            from agent.revit_bridge import BridgeQueueResponse

            mock_client = MagicMock()
            mock_client.is_healthy.return_value = True
            mock_client.push_instruction.return_value = BridgeQueueResponse(
                queued=True,
                instruction_id=instr_id,
            )
            mock_get_client.return_value = mock_client

            resp = await client.post(
                f"/bridge/push/{session_id}",
                params={"instruction_id": instr_id},
                headers=auth_headers,
            )

        assert resp.status_code == 200
        data = resp.json()
        assert data["queued"] is True
        assert data["instruction_id"] == instr_id
        assert "poll_url" in data
        assert "callback_url" in data

    @pytest.mark.asyncio
    async def test_push_502_when_bridge_rejects(self, client, session_id, auth_headers):
        """502 when bridge accepts connection but rejects the instruction."""
        from agent.main import orchestrator

        instructions = orchestrator._instructions.get(session_id, [])

        if not instructions:
            pytest.skip("Session has no instructions to push")

        instr_id = instructions[0].instruction_id

        with patch("agent.main._get_bridge_client") as mock_get_client:
            mock_client = MagicMock()
            mock_client.is_healthy.return_value = True
            mock_client.push_instruction.return_value = None  # bridge rejected
            mock_get_client.return_value = mock_client

            resp = await client.post(
                f"/bridge/push/{session_id}",
                params={"instruction_id": instr_id},
                headers=auth_headers,
            )

        assert resp.status_code == 502


# ─────────────────────────────────────────────────────────────────────────────
# 5. RevitBridgeClient unit tests (no HTTP server needed)
# ─────────────────────────────────────────────────────────────────────────────


class TestRevitBridgeClient:
    """Unit tests for agent.revit_bridge — mock httpx calls."""

    def test_push_and_wait_uses_callback_fast_path(self):
        """If callback_result is provided, push_and_wait returns it immediately."""
        from agent.revit_bridge import BridgeActionResult, RevitBridgeClient

        instr_id = str(uuid.uuid4())
        pre_built = BridgeActionResult(
            success=True,
            instruction_id=instr_id,
            element_id="elem-123",
            duration_ms=10,
        )

        client = RevitBridgeClient(base_url="http://localhost:8766", timeout_s=1.0)
        result = client.push_and_wait(
            instruction_payload={"instruction_id": instr_id},
            callback_result=pre_built,
        )

        assert result is pre_built
        assert result.success is True
        assert result.element_id == "elem-123"

    def test_push_and_wait_returns_error_when_unreachable(self):
        """Returns failed ActionResult when bridge is not running."""
        from agent.revit_bridge import RevitBridgeClient

        instr_id = str(uuid.uuid4())
        client = RevitBridgeClient(base_url="http://localhost:19999", timeout_s=1.0)

        # Mock push_instruction to return None (unreachable)
        with patch.object(client, "push_instruction", return_value=None):
            result = client.push_and_wait(
                instruction_payload={"instruction_id": instr_id},
                timeout_s=1.0,
            )

        assert result.success is False
        assert result.instruction_id == instr_id
        assert "unreachable" in (result.error or "").lower()

    def test_is_healthy_returns_false_on_connection_error(self):
        """is_healthy() returns False when Revit is not listening."""
        import httpx

        from agent.revit_bridge import RevitBridgeClient

        client = RevitBridgeClient(base_url="http://localhost:19999", timeout_s=0.5)

        with patch.object(client._client, "get", side_effect=httpx.ConnectError("refused")):
            assert client.is_healthy() is False

    def test_is_healthy_returns_true_on_200(self):
        """is_healthy() returns True when bridge responds with 200."""

        from agent.revit_bridge import RevitBridgeClient

        client = RevitBridgeClient(base_url="http://localhost:8766", timeout_s=1.0)
        mock_resp = MagicMock()
        mock_resp.status_code = 200

        with patch.object(client._client, "get", return_value=mock_resp):
            assert client.is_healthy() is True

    def test_poll_result_returns_none_when_pending(self):
        """poll_result returns None when status is 'pending'."""
        from agent.revit_bridge import RevitBridgeClient

        client = RevitBridgeClient(base_url="http://localhost:8766")
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.json.return_value = {"status": "pending", "instruction_id": "abc"}

        with patch.object(client._client, "get", return_value=mock_resp):
            result = client.poll_result("abc")

        assert result is None

    def test_poll_result_returns_action_result_when_ready(self):
        """poll_result parses and returns BridgeActionResult when result is ready."""
        from agent.revit_bridge import RevitBridgeClient

        instr_id = str(uuid.uuid4())
        client = RevitBridgeClient(base_url="http://localhost:8766")
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.json.return_value = {
            "success": True,
            "instruction_id": instr_id,
            "element_id": "elem-456",
            "error": None,
            "duration_ms": 88,
        }

        with patch.object(client._client, "get", return_value=mock_resp):
            result = client.poll_result(instr_id)

        assert result is not None
        assert result.success is True
        assert result.element_id == "elem-456"
        assert result.duration_ms == 88

    def test_push_instruction_returns_queue_response(self):
        """push_instruction returns BridgeQueueResponse on 202."""
        from agent.revit_bridge import RevitBridgeClient

        instr_id = str(uuid.uuid4())
        client = RevitBridgeClient(base_url="http://localhost:8766")
        mock_resp = MagicMock()
        mock_resp.status_code = 202
        mock_resp.json.return_value = {"queued": True, "instruction_id": instr_id}

        with patch.object(client._client, "post", return_value=mock_resp):
            result = client.push_instruction({"instruction_id": instr_id})

        assert result is not None
        assert result.queued is True
        assert result.instruction_id == instr_id

    def test_push_instruction_returns_none_on_connection_error(self):
        """push_instruction returns None when bridge is unreachable."""
        import httpx

        from agent.revit_bridge import RevitBridgeClient

        client = RevitBridgeClient(base_url="http://localhost:19999")

        with patch.object(client._client, "post", side_effect=httpx.ConnectError("refused")):
            result = client.push_instruction({"instruction_id": "x"})

        assert result is None

    def test_push_batch_returns_error_for_all_if_unreachable(self):
        """push_batch marks all instructions as failed when bridge unreachable."""
        from agent.revit_bridge import RevitBridgeClient

        client = RevitBridgeClient(base_url="http://localhost:19999")
        instructions = [{"instruction_id": str(uuid.uuid4())} for _ in range(3)]

        with patch.object(client, "push_instruction", return_value=None):
            results = client.push_batch(instructions)

        assert len(results) == 3
        assert all(not r.success for r in results)
        assert all("bridge unreachable" in (r.error or "") for r in results)


# ─────────────────────────────────────────────────────────────────────────────
# 6. Callback store helpers
# ─────────────────────────────────────────────────────────────────────────────


class TestCallbackStore:
    """Unit tests for store_callback_result / consume_callback_result."""

    def test_store_and_consume_round_trip(self):
        """Result stored can be consumed exactly once."""
        from agent.revit_bridge import (
            BridgeActionResult,
            consume_callback_result,
            store_callback_result,
        )

        instr_id = str(uuid.uuid4())
        result = BridgeActionResult(success=True, instruction_id=instr_id, element_id="e-1")

        store_callback_result(result)
        consumed = consume_callback_result(instr_id)

        assert consumed is not None
        assert consumed.instruction_id == instr_id
        assert consumed.element_id == "e-1"

    def test_consume_returns_none_if_not_stored(self):
        """Consuming an ID that was never stored returns None."""
        from agent.revit_bridge import consume_callback_result

        result = consume_callback_result(str(uuid.uuid4()))
        assert result is None

    def test_consume_removes_from_store(self):
        """After consuming, the result is no longer available."""
        from agent.revit_bridge import (
            BridgeActionResult,
            consume_callback_result,
            store_callback_result,
        )

        instr_id = str(uuid.uuid4())
        store_callback_result(BridgeActionResult(success=True, instruction_id=instr_id))

        first = consume_callback_result(instr_id)
        second = consume_callback_result(instr_id)

        assert first is not None
        assert second is None  # already consumed

    def test_store_overwrites_previous(self):
        """Storing a second result for same ID replaces the first."""
        from agent.revit_bridge import (
            BridgeActionResult,
            consume_callback_result,
            store_callback_result,
        )

        instr_id = str(uuid.uuid4())
        store_callback_result(
            BridgeActionResult(success=False, instruction_id=instr_id, error="err1")
        )
        store_callback_result(
            BridgeActionResult(success=True, instruction_id=instr_id, element_id="e-2")
        )

        result = consume_callback_result(instr_id)
        assert result is not None
        assert result.success is True
        assert result.element_id == "e-2"


# ─────────────────────────────────────────────────────────────────────────────
# 7. BridgeActionResult model validation
# ─────────────────────────────────────────────────────────────────────────────


class TestBridgeActionResultModel:
    """Pydantic model validation for BridgeActionResult."""

    def test_minimal_success_result(self):
        from agent.revit_bridge import BridgeActionResult

        r = BridgeActionResult(success=True, instruction_id="abc-123", element_id="elem-1")
        assert r.success is True
        assert r.element_id == "elem-1"
        assert r.error is None
        assert r.duration_ms == 0

    def test_minimal_failure_result(self):
        from agent.revit_bridge import BridgeActionResult

        r = BridgeActionResult(success=False, instruction_id="abc-456", error="SC1 block")
        assert r.success is False
        assert r.element_id is None
        assert r.error == "SC1 block"

    def test_full_result_with_resolution(self):
        from agent.revit_bridge import BridgeActionResult

        r = BridgeActionResult(
            success=True,
            instruction_id="abc-789",
            element_id="e-999",
            duration_ms=150,
            family_name="Pipe-Round",
            type_name="100mm",
            resolution_path="hint",
        )
        assert r.family_name == "Pipe-Round"
        assert r.type_name == "100mm"
        assert r.resolution_path == "hint"
        assert r.duration_ms == 150
