"""Revit Bridge Client — bidirectional push/poll between Python agent and Revit addin.

Architecture:
  Python agent (port 8765) ──POST /revit-actions──▶ Revit bridge (port 8766)
                                                              │
                                                    Revit ExternalEvent queue
                                                              │
                                                    RevitElementFactory.Execute()
                                                              │
  Python agent ◀──POST /revit-callback──────────── AgentBridgeService (writes back)
               ◀──GET  /revit-results/{id}──────── (polling fallback)

Two result delivery modes:
  PUSH (preferred): Revit bridge POSTs ActionResult to POST /sessions/{id}/revit-callback
  POLL (fallback):  wait_for_result() polls GET /revit-results/{id} every 0.5s

Usage in orchestrator:
    from agent.revit_bridge import RevitBridgeClient
    bridge = RevitBridgeClient("http://localhost:8766")
    if bridge.is_healthy():
        result = bridge.push_and_wait(instruction, timeout_s=30)
"""

from __future__ import annotations

import os
import time

import httpx
import structlog

log = structlog.get_logger(__name__)

# ── Config ────────────────────────────────────────────────────────────────────

REVIT_BRIDGE_URL: str = os.environ.get("REVIT_BRIDGE_URL", "http://localhost:8766")
BRIDGE_TIMEOUT_S: float = float(os.environ.get("BRIDGE_TIMEOUT_S", "30"))
BRIDGE_POLL_INTERVAL_S: float = 0.5

# ── Result models (mirrors C# ActionResult) ───────────────────────────────────

from pydantic import BaseModel


class BridgeHealth(BaseModel):
    connected: bool = False
    queue_depth: int = 0
    elements_created: int = 0
    elements_blocked: int = 0
    version: str = "unknown"


class BridgeActionResult(BaseModel):
    """ActionResult received from the Revit bridge."""

    success: bool
    element_id: str | None = None
    instruction_id: str
    error: str | None = None
    duration_ms: int = 0
    # Sprint 2 additions
    family_name: str | None = None  # which family was used
    type_name: str | None = None  # which type was selected
    resolution_path: str | None = None  # "hint" | "config" | "fallback" | "none"


class BridgeQueueResponse(BaseModel):
    queued: bool
    instruction_id: str


# ── Bridge Client ─────────────────────────────────────────────────────────────


class RevitBridgeClient:
    """HTTP client for the Revit bridge running on port 8766.

    Thread-safe: uses httpx sync client with connection pooling.
    All methods return None / empty result on connection failure — the
    orchestrator falls back to mock mode gracefully.
    """

    def __init__(self, base_url: str = REVIT_BRIDGE_URL, timeout_s: float = 5.0):
        self.base_url = base_url.rstrip("/")
        self._timeout = timeout_s
        self._client = httpx.Client(
            base_url=self.base_url,
            timeout=timeout_s,
            headers={"Content-Type": "application/json"},
        )

    def close(self):
        self._client.close()

    def __enter__(self):
        return self

    def __exit__(self, *_):
        self.close()

    # ── Health ────────────────────────────────────────────────────────────────

    def is_healthy(self) -> bool:
        """True if the Revit bridge is reachable and responding."""
        try:
            resp = self._client.get("/health", timeout=2.0)
            return resp.status_code == 200
        except (httpx.ConnectError, httpx.TimeoutException):
            return False

    def get_health(self) -> BridgeHealth:
        """Get full bridge health info. Returns disconnected status on failure."""
        try:
            resp = self._client.get("/health", timeout=2.0)
            if resp.status_code == 200:
                return BridgeHealth(**resp.json())
        except (httpx.ConnectError, httpx.TimeoutException):
            pass
        return BridgeHealth(connected=False)

    # ── Push instruction ──────────────────────────────────────────────────────

    def push_instruction(self, instruction_payload: dict) -> BridgeQueueResponse | None:
        """POST one ElementInstruction to the Revit bridge queue.

        Returns BridgeQueueResponse if accepted, None on connection failure.
        """
        try:
            resp = self._client.post("/revit-actions", json=instruction_payload)
            if resp.status_code in (200, 202):
                data = resp.json()
                return BridgeQueueResponse(
                    queued=data.get("queued", False),
                    instruction_id=data.get("instruction_id", ""),
                )
            log.warning(
                "bridge_push_rejected",
                status=resp.status_code,
                body=resp.text[:200],
            )
            return None
        except (httpx.ConnectError, httpx.TimeoutException) as exc:
            log.warning("bridge_unreachable", error=str(exc))
            return None

    # ── Poll result ───────────────────────────────────────────────────────────

    def poll_result(self, instruction_id: str) -> BridgeActionResult | None:
        """GET the result for one instruction. Returns None if still pending."""
        try:
            resp = self._client.get(f"/revit-results/{instruction_id}")
            if resp.status_code == 200:
                data = resp.json()
                # 202 = still pending from Revit side (not yet processed)
                if data.get("status") == "pending":
                    return None
                return BridgeActionResult(**data)
            if resp.status_code == 202:
                return None  # still in queue
        except (httpx.ConnectError, httpx.TimeoutException):
            pass
        return None

    # ── Push + wait (blocking poll loop) ─────────────────────────────────────

    def push_and_wait(
        self,
        instruction_payload: dict,
        timeout_s: float = BRIDGE_TIMEOUT_S,
        callback_result: BridgeActionResult | None = None,
    ) -> BridgeActionResult:
        """Push an instruction and block until the result arrives.

        If callback_result is provided (result already delivered via callback),
        returns it immediately without polling.

        On timeout or connection failure, returns a failed ActionResult.
        """
        instruction_id = instruction_payload.get("instruction_id", "unknown")

        # Fast path: callback already delivered the result
        if callback_result is not None:
            return callback_result

        # Push the instruction
        queued = self.push_instruction(instruction_payload)
        if queued is None:
            return BridgeActionResult(
                success=False,
                instruction_id=instruction_id,
                error="Revit bridge unreachable — is Revit running with the ScanToBIM addin?",
            )

        # Poll loop
        deadline = time.monotonic() + timeout_s
        while time.monotonic() < deadline:
            result = self.poll_result(instruction_id)
            if result is not None:
                log.info(
                    "bridge_result_received",
                    instruction_id=instruction_id[:8],
                    success=result.success,
                    duration_ms=result.duration_ms,
                )
                return result
            time.sleep(BRIDGE_POLL_INTERVAL_S)

        log.warning("bridge_poll_timeout", instruction_id=instruction_id[:8], timeout_s=timeout_s)
        return BridgeActionResult(
            success=False,
            instruction_id=instruction_id,
            error=f"Revit bridge poll timeout after {timeout_s}s — element may still be queued.",
        )

    # ── Batch push ────────────────────────────────────────────────────────────

    def push_batch(
        self,
        instructions: list[dict],
        progress_callback=None,
    ) -> list[BridgeActionResult]:
        """Push multiple instructions and collect all results.

        progress_callback(done, total) called after each result.
        Non-blocking: fires all instructions then collects results.
        """
        if not instructions:
            return []

        results: dict[str, BridgeActionResult] = {}
        pending: list[str] = []

        # Fire all instructions
        for instr in instructions:
            iid = instr.get("instruction_id", "unknown")
            queued = self.push_instruction(instr)
            if queued:
                pending.append(iid)
            else:
                results[iid] = BridgeActionResult(
                    success=False,
                    instruction_id=iid,
                    error="bridge unreachable",
                )

        # Collect results with shared deadline
        deadline = time.monotonic() + BRIDGE_TIMEOUT_S + len(instructions) * 0.5
        collected: list[str] = []

        while pending and time.monotonic() < deadline:
            for iid in list(pending):
                result = self.poll_result(iid)
                if result is not None:
                    results[iid] = result
                    pending.remove(iid)
                    collected.append(iid)
                    if progress_callback:
                        progress_callback(len(collected), len(instructions))
            if pending:
                time.sleep(BRIDGE_POLL_INTERVAL_S)

        # Timeout any remaining
        for iid in pending:
            results[iid] = BridgeActionResult(
                success=False,
                instruction_id=iid,
                error="bridge poll timeout",
            )

        # Return in original order
        return [results[i.get("instruction_id", "unknown")] for i in instructions]


# ── In-process callback store (populated by POST /revit-callback) ─────────────
# Results pushed by Revit bridge land here before the API response is read.

_callback_results: dict[str, BridgeActionResult] = {}


def store_callback_result(result: BridgeActionResult) -> None:
    """Called by the /revit-callback endpoint to cache the pushed result."""
    _callback_results[result.instruction_id] = result


def consume_callback_result(instruction_id: str) -> BridgeActionResult | None:
    """Pop a callback result if available. Returns None if not yet received."""
    return _callback_results.pop(instruction_id, None)
