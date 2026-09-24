#!/usr/bin/env python3
"""
demo_runner.py
──────────────
Narrated 5-minute Proof-of-Value demo that drives the FastAPI agent
via its REST endpoints.

Usage
─────
  # Terminal 1 — start the agent server
  uvicorn agent.main:app --port 8765

  # Terminal 2 — run the narrated demo
  python demo_runner.py

  # Skip pauses (CI / automated run)
  python demo_runner.py --no-pause

  # Custom server URL
  python demo_runner.py --url http://localhost:8765

Output
──────
  Console narration with colour codes.
  demo_report.html is generated at the end (calls agent/report.py).
"""

from __future__ import annotations

import argparse
import sys
import time

import httpx

# ── Colour helpers ────────────────────────────────────────────────────────────

RESET = "\033[0m"
BOLD = "\033[1m"
GREEN = "\033[92m"
YELLOW = "\033[93m"
RED = "\033[91m"
CYAN = "\033[96m"
DIM = "\033[2m"
ORANGE = "\033[38;5;208m"


def banner(text: str) -> None:
    width = 72
    print(f"\n{CYAN}{'─' * width}{RESET}")
    print(f"{BOLD}{CYAN}  {text}{RESET}")
    print(f"{CYAN}{'─' * width}{RESET}")


def step(n: int, title: str) -> None:
    print(f"\n{BOLD}{ORANGE}  Step {n}/6 — {title}{RESET}")


def ok(msg: str) -> None:
    print(f"  {GREEN}✓{RESET}  {msg}")


def warn(msg: str) -> None:
    print(f"  {YELLOW}⚠{RESET}  {msg}")


def blocked(msg: str) -> None:
    print(f"  {RED}🔴 BLOCKED{RESET}  {msg}")


def pending(msg: str) -> None:
    print(f"  {YELLOW}🟡 PENDING{RESET}  {msg}")


def info(msg: str) -> None:
    print(f"  {DIM}{msg}{RESET}")


def pause(auto: bool) -> None:
    if auto:
        time.sleep(0.4)
    else:
        input(f"  {DIM}[ press Enter to continue ]{RESET}")


# ── HTTP helpers ──────────────────────────────────────────────────────────────


def get(client: httpx.Client, path: str) -> dict:
    r = client.get(path)
    r.raise_for_status()
    return r.json()


def post(client: httpx.Client, path: str, body: dict) -> dict:
    r = client.post(path, json=body)
    r.raise_for_status()
    return r.json()


# ── Demo steps ────────────────────────────────────────────────────────────────


def run_demo(base_url: str, auto: bool) -> None:

    banner("Scan-to-BIM  —  Proof of Value Demo")
    print(f"  {DIM}Agent URL : {base_url}{RESET}")
    print(f"  {DIM}Mode      : {'auto (no pauses)' if auto else 'interactive'}{RESET}")

    client = httpx.Client(base_url=base_url, timeout=30.0)

    # ── Pre-flight ────────────────────────────────────────────────────────────
    step(0, "Health check")
    try:
        health = get(client, "/health")
        ok(
            f"Agent is UP  —  version {health.get('version', '?')}  "
            f"service={health.get('service', '?')}"
        )
    except httpx.ConnectError:
        print(f"\n  {RED}ERROR: Cannot reach {base_url}{RESET}")
        print("  Start the agent first:  uvicorn agent.main:app --port 8765\n")
        sys.exit(1)
    pause(auto)

    # ── Step 1: Create session ────────────────────────────────────────────────
    step(1, "Create session — Demo Nuclear Facility Unit 4")
    info("POST /sessions  {site_name, use_synthetic: true}")
    session = post(
        client,
        "/sessions",
        {
            "site_name": "Demo Nuclear Facility — Unit 4",
            "zone_id": "zone-reactor-001",
            "use_synthetic": True,
        },
    )

    session_id = session["session_id"]
    exec_data = session.get("execution", {})

    ok(f"Session created  →  {session_id}")
    ok(f"Segments found   →  {session.get('segments_found', 0)}")
    print()

    created = exec_data.get("created", 0)
    blocked_ = exec_data.get("blocked", 0)
    pending_ = exec_data.get("pending_approval", 0)
    errors = exec_data.get("errors", 0)

    info(f"  Elements created  :  {created}")
    if blocked_ > 0:
        for b in exec_data.get("blocked_details", []):
            blocked(
                f"SC1  {b.get('element_type', '?'):<8}  "
                f"seg={b.get('segment_id', '?')[:12]}...  "
                f"reason={b.get('reason', '?')[:55]}..."
            )
    if pending_ > 0:
        for g in exec_data.get("pending_gates", []):
            pending(
                f"SC2  {g.get('element_type', '?'):<8}  "
                f"gate={g.get('gate_id', '?')[:12]}..."
            )
    if errors:
        warn(f"Errors: {errors}")
    pause(auto)

    # ── Step 2: Inspect safety classification ─────────────────────────────────
    step(2, "Inspect safety classification")
    info("GET /sessions/{id}/audit  (first 8 segment_classified events)")
    audit = get(client, f"/sessions/{session_id}/audit")
    events = audit if isinstance(audit, list) else audit.get("events", [])

    classified = [e for e in events if e.get("event_type") == "segment_classified"][:8]
    if classified:
        print(f"\n  {'Segment':<14} {'Element':<10} {'Safety':<6}  Actor")
        print(f"  {'─' * 14} {'─' * 10} {'─' * 6}  {'─' * 20}")
        for e in classified:
            detail = e.get("detail", {})
            seg = (e.get("segment_id") or "")[:12]
            el = detail.get("element_type", "?")
            sc = detail.get("safety_category", "?")
            actor = e.get("actor", "classifier")
            colour = (
                RED
                if sc == "SC1"
                else YELLOW
                if sc == "SC2"
                else CYAN
                if sc == "SC3"
                else GREEN
            )
            print(f"  {seg:<14} {el:<10} {colour}{sc:<6}{RESET}  {actor}")
    pause(auto)

    # ── Step 3: Safety gate — SC2 approval ───────────────────────────────────
    step(3, "Safety gate — approve pending SC2 elements")
    info("GET /safety-gates?session_id=...")
    gates_resp = get(client, f"/safety-gates?session_id={session_id}")
    gates = gates_resp if isinstance(gates_resp, list) else gates_resp.get("gates", [])
    pending_gates = [g for g in gates if g.get("status") == "pending"]

    if not pending_gates:
        info(
            "No pending SC2 gates in this session (synthetic data may not include SC2 segments)."
        )
        info(
            "In demo.py (in-process mode) SC1/SC2 segments are injected — run: python demo.py"
        )
    else:
        for gate in pending_gates:
            gate_id = gate["gate_id"]
            el_type = gate.get("element_type", "?")
            sc = gate.get("safety_category", "SC2")
            info(f"Gate {gate_id[:12]}...  type={el_type}  {sc}")
            info(
                "POST /safety-gates/{gate_id}/decision  {approved: true, approver_upn: ...}"
            )

            decision = post(
                client,
                f"/safety-gates/{gate_id}/decision",
                {
                    "approved": True,
                    "approver_upn": "senior.engineer@example.com",
                    "comments": "PoV demo approval — reviewed scan geometry",
                },
            )
            ok(
                f"Gate approved  →  element_id={decision.get('element_id', decision.get('gate_id'))}"
            )
    pause(auto)

    # ── Step 4: Verify audit chain integrity ──────────────────────────────────
    step(4, "Verify audit chain integrity (HMAC-SHA256)")
    info(f"GET /sessions/{session_id[:12]}...  /chain")
    chain = get(client, f"/sessions/{session_id}/chain")

    valid = chain.get("valid", False)
    broken_links = chain.get("broken_links", [])
    event_count = chain.get("event_count", 0)

    if valid:
        ok(f"Chain VALID  —  {event_count} events  —  0 broken links")
        ok("Tamper-proof audit trail confirmed")
    else:
        warn(f"Chain INVALID  —  {len(broken_links)} broken links  —  investigate!")
        for link in broken_links[:3]:
            warn(f"  broken: {link}")
    pause(auto)

    # ── Step 5: Element provenance ────────────────────────────────────────────
    step(5, "Element provenance — trace from scan to BIM")
    info("GET /sessions/{id}/audit  (element_created events)")
    created_events = [e for e in events if e.get("event_type") == "element_created"]

    if created_events:
        sample_event = created_events[0]
        sample_el_id = sample_event.get("element_id") or sample_event.get(
            "detail", {}
        ).get("element_id", "")
        if sample_el_id:
            info(f"GET /elements/{sample_el_id}/provenance")
            try:
                prov = get(client, f"/elements/{sample_el_id}/provenance")
                prov_events = prov if isinstance(prov, list) else prov.get("events", [])
                print(f"\n  Provenance chain for element: {CYAN}{sample_el_id}{RESET}")
                for pe in prov_events:
                    ts = pe.get("timestamp_utc", "")[:19]
                    et = pe.get("event_type", "?")
                    act = pe.get("actor", "?")
                    print(f"    {DIM}{ts}{RESET}  {BOLD}{et:<30}{RESET}  actor={act}")
            except httpx.HTTPStatusError:
                info(
                    f"Provenance endpoint returned non-200 for {sample_el_id} — skipping."
                )
        else:
            info(
                "No element_id in created event detail — provenance not available for this segment."
            )
    else:
        info(
            "No element_created events found (all elements may be blocked or pending)."
        )
    pause(auto)

    # ── Step 6: Session summary ───────────────────────────────────────────────
    step(6, "Final session summary")
    summary = get(client, f"/sessions/{session_id}")

    total_seg = summary.get("total_segments", 0)
    total_el = summary.get("total_elements_created", 0)
    total_clash = summary.get("total_clashes", 0)
    total_ncr = summary.get("total_ncrs", 0)
    state = summary.get("state", "?")

    print()
    print(f"  {'Session ID':<22}  {CYAN}{session_id}{RESET}")
    print(f"  {'Site':<22}  {summary.get('site_name', '?')}")
    print(f"  {'State':<22}  {GREEN if state == 'completed' else YELLOW}{state}{RESET}")
    print(f"  {'Segments found':<22}  {total_seg}")
    print(f"  {'Elements created':<22}  {GREEN}{total_el}{RESET}")
    print(f"  {'Clashes':<22}  {total_clash}")
    print(f"  {'NCRs':<22}  {total_ncr}")
    print(f"  {'Audit events':<22}  {event_count}")
    print(
        f"  {'Chain integrity':<22}  {(GREEN + 'PASS') if valid else (RED + 'FAIL')}{RESET}"
    )
    print()

    # ── Generate HTML report ──────────────────────────────────────────────────
    banner("Generating HTML audit report")
    try:
        from agent.report import generate_report

        report_path = generate_report(
            session_id=session_id,
            session=summary,
            audit_events=events,
            chain_valid=valid,
            event_count=event_count,
        )
        ok(f"Report saved  →  {report_path}")
        info("Open in browser: open demo_report.html")
    except Exception as e:
        warn(f"Report generation skipped: {e}")

    banner("Demo complete")
    print(
        f"  {DIM}All 6 steps completed. Session {session_id[:12]}... recorded.{RESET}\n"
    )


# ── Entry point ───────────────────────────────────────────────────────────────


def main() -> None:
    parser = argparse.ArgumentParser(description="Scan-to-BIM PoV demo runner")
    parser.add_argument(
        "--url",
        default="http://localhost:8765",
        help="Agent base URL (default: http://localhost:8765)",
    )
    parser.add_argument(
        "--no-pause",
        action="store_true",
        help="Skip interactive pauses (for CI / automated runs)",
    )
    args = parser.parse_args()

    run_demo(base_url=args.url, auto=args.no_pause)


if __name__ == "__main__":
    main()
