"""FastAPI application for the Scan-to-BIM agent.

Run: uvicorn agent.main:app --host 0.0.0.0 --port 8765 --reload
"""

from __future__ import annotations

import os
import shutil
import sys
import tempfile
from pathlib import Path

# Ensure pye57 DLL directory is in PATH for E57 support
_venv_site_packages = os.path.join(os.path.dirname(sys.executable), "Lib", "site-packages", "pye57")
if os.path.isdir(_venv_site_packages):
    os.environ["PATH"] = _venv_site_packages + os.pathsep + os.environ["PATH"]

import structlog
from dotenv import load_dotenv
from fastapi import (
    BackgroundTasks,
    Depends,
    FastAPI,
    File,
    HTTPException,
    Query,
    Response,
    UploadFile,
    status,
)
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse, RedirectResponse, StreamingResponse
from fastapi.security import OAuth2PasswordRequestForm
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from agent.audit import AuditLedger
from agent.auth import (
    CurrentUser,
    TokenResponse,
    authenticate_user,
    create_access_token,
    get_current_user,
)
from agent.cde_state import CDEStateService, CDETransitionError
from agent.licensing import LicenseInfo, require_feature
from agent.models import SafetyGateDecision, SurveyControlPoint
from agent.ncr import NCRAlreadyResolvedError, NCRNotFoundError, NCRService
from agent.orchestrator import ScanToBIMOrchestrator
from agent.revit_bridge import (
    BridgeActionResult,
    RevitBridgeClient,
    store_callback_result,
)
from agent.safety_gate import SafetyGateService

load_dotenv()

# ── Paths ───────────────────────────────────────────────────────────────────────
_ROOT = Path(__file__).resolve().parent.parent  # repo root
_FRONTEND = _ROOT / "frontend"

structlog.configure(
    processors=[
        structlog.processors.TimeStamper(fmt="iso"),
        structlog.dev.ConsoleRenderer(),
    ]
)
log = structlog.get_logger(__name__)

app = FastAPI(
    title="Scan-to-BIM Agent",
    description="Enterprise Agentic Scan-to-BIM — NQA-1 / ISO 19650 compliant",
    version="0.3.0",
)

# ── Static UI (Sprint 3 frontend) ─────────────────────────────────────────────
# Serve the dashboard at /ui/*  e.g.  http://localhost:8765/ui/login.html
if _FRONTEND.exists():
    app.mount("/ui", StaticFiles(directory=str(_FRONTEND), html=True), name="ui")

# ── CORS (allow dashboard + Revit bridge) ──────────────────────────────────────
_ALLOWED_ORIGINS = os.environ.get(
    "ALLOWED_ORIGINS",
    "http://localhost:3000,http://localhost:5173,http://localhost:8765",
).split(",")

app.add_middleware(
    CORSMiddleware,
    allow_origins=_ALLOWED_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── Shared state ───────────────────────────────────────────────────────────────

audit = AuditLedger()
safety_gate = SafetyGateService(audit)
orchestrator = ScanToBIMOrchestrator(audit=audit, safety_gate=safety_gate)
ncr_service = NCRService(audit)

# P2.4++ NQA-1 three-party signature workflow
from agent.nqa1_signature import NQA1SignatureService

nqa1_signatures = NQA1SignatureService()
cde_service = CDEStateService(
    audit=audit,
    orchestrator=orchestrator,
    safety_gate=safety_gate,
    ncr_service=ncr_service,
)


# ── Request / Response models ──────────────────────────────────────────────────


class CreateSessionRequest(BaseModel):
    site_name: str
    zone_id: str = "zone-001"
    use_synthetic: bool = True
    scan_file: str | None = None


class GateDecisionRequest(BaseModel):
    approved: bool
    approver_upn: str = "engineer@demo.com"
    comments: str | None = None


class CDETransitionRequest(BaseModel):
    target_state: str  # "SHARED" or "PUBLISHED"
    triggered_by: str = "coordinator"
    notes: str | None = None


class RaiseNCRRequest(BaseModel):
    element_id: str
    description: str
    severity: str = "MAJOR"  # CRITICAL | MAJOR | MINOR
    raised_by: str = "agent"


class ResolveNCRRequest(BaseModel):
    resolved_by: str
    resolution: str


class IngestElementsRequest(BaseModel):
    """Payload from the Revit plugin — pre-processed segments, no raw file needed."""

    segments: list[dict] = Field(
        default_factory=list, description="GeometrySegment dicts from stb-processor"
    )
    scan_metadata: dict = Field(
        default_factory=dict, description="Metadata from processor_cli (file, timing, …)"
    )
    source: str = "revit_plugin"  # "revit_plugin" | "api"


class AlignCoordinatesRequest(BaseModel):
    """Survey control points for coordinate system alignment.

    Provide ≥3 pairs of corresponding scan-frame and world-frame coordinates
    (all in millimetres).  The orchestrator computes the best-fit rigid
    transform (Kabsch/SVD) and applies it to all stored segment positions.
    """

    control_points: list[SurveyControlPoint]
    max_rmse_mm: float = 5.0


class ScanQualityRequest(BaseModel):
    """Parameters for the pre-segmentation scan quality gate."""

    point_count: int
    area_m2: float
    noise_pts: int = 0
    min_density_pts_per_m2: float = 100.0
    raise_on_fail: bool = False


# ── Auth ───────────────────────────────────────────────────────────────────────


@app.post("/auth/token", response_model=TokenResponse, tags=["auth"])
def login(form_data: OAuth2PasswordRequestForm = Depends()):
    """Exchange username + password for a JWT access token.

    Use the returned token as:  Authorization: Bearer <token>
    """
    user = authenticate_user(form_data.username, form_data.password)
    if not user:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="incorrect username or password",
            headers={"WWW-Authenticate": "Bearer"},
        )
    token = create_access_token(
        username=form_data.username,
        role=user["role"],
        tenant_id=user["tenant_id"],
    )
    log.info("login", username=form_data.username, role=user["role"])
    return TokenResponse(
        access_token=token,
        role=user["role"],
        tenant_id=user["tenant_id"],
    )


@app.get("/auth/me", response_model=CurrentUser, tags=["auth"])
def whoami(user: CurrentUser = Depends(get_current_user)):
    """Return the profile of the currently authenticated user."""
    return user


@app.get("/auth/license", response_model=LicenseInfo, tags=["auth"])
def license_info(user: CurrentUser = Depends(get_current_user)):
    """Return feature flags and limits for the current tenant's tier."""
    return LicenseInfo.for_tenant(user.tenant_id)


# ── Health ─────────────────────────────────────────────────────────────────────


@app.get("/", include_in_schema=False)
def root_redirect():
    """Redirect root to the dashboard UI."""
    return RedirectResponse(url="/ui/login.html")


@app.get("/health", tags=["system"])
def health():
    return {"status": "ok", "service": "scan-to-bim-agent", "version": "0.3.0"}


@app.get("/ready", tags=["system"])
def ready():
    """Kubernetes readiness probe — checks audit ledger is reachable."""
    try:
        audit.verify_chain()
        return {"status": "ready"}
    except Exception as exc:
        raise HTTPException(status_code=503, detail=f"audit ledger error: {exc}") from exc


@app.get("/version", tags=["system"])
def version():
    """NQA-1 § 302 Software Identification — the record that pins every
    deliverable to the exact code that produced it.

    Returns product name/version, git SHA, tracked dependency versions,
    and a stable identity_hash that reviewers can cite in V&V records.
    """
    from agent.software_identity import get_software_identity

    return get_software_identity().to_dict()


# ── Sessions ───────────────────────────────────────────────────────────────────


@app.post("/sessions")
def create_session(req: CreateSessionRequest):
    """Create a new session and run the full pipeline."""
    session = orchestrator.create_session(req.site_name)

    # Step 1: Segmentation
    segments = orchestrator.run_segmentation(
        session_id=session.session_id,
        file_path=Path(req.scan_file) if req.scan_file else None,
        zone_id=req.zone_id,
        use_synthetic=req.use_synthetic,
    )

    # Step 2: Classification
    instructions = orchestrator.classify_and_prepare(session.session_id, segments)

    # Step 3: Clash detection (before execution — flag issues before creating elements)
    clashes = orchestrator.run_clash_detection(session.session_id)

    # Step 4: Execute (with safety gates + NCR auto-gen for low-confidence)
    result = orchestrator.execute_instructions(session.session_id, ncr_service=ncr_service)

    return {
        "session_id": session.session_id,
        "segments_found": len(segments),
        "instructions_prepared": len(instructions),
        "clashes_detected": len(clashes),
        "execution": result,
    }


@app.get("/sessions", tags=["sessions"])
def list_sessions(user: CurrentUser = Depends(get_current_user)):
    """List all sessions with summary stats for the dashboard."""
    result = []
    for session_id, session in orchestrator.sessions.items():
        try:
            cde = cde_service.get_state(session_id)
            ncrs = ncr_service.list_session(session_id)
            clashes = orchestrator._clashes.get(session_id, [])
            result.append(
                {
                    "session_id": session_id,
                    "site_name": session.site_name,
                    "zone_id": getattr(session, "zone_id", "zone-001"),
                    "cde_state": cde.current_state.value if cde else "WIP",
                    "total_elements_created": session.total_elements_created,
                    "clash_count": len(clashes),
                    "open_ncrs": sum(1 for n in ncrs if not n.resolved_at),
                    "created_at": session.created_at.isoformat()
                    if hasattr(session.created_at, "isoformat")
                    else str(session.created_at),
                    "scan_source": _ingest_sources.get(session_id, {}).get("source"),
                }
            )
        except Exception:  # noqa: BLE001
            result.append(
                {
                    "session_id": session_id,
                    "site_name": getattr(session, "site_name", session_id),
                    "cde_state": "WIP",
                    "total_elements_created": getattr(session, "total_elements_created", 0),
                }
            )
    # Most-recent first
    result.sort(key=lambda s: s.get("created_at", ""), reverse=True)
    return {"sessions": result, "total": len(result)}


@app.get("/sessions/{session_id}")
def get_session(session_id: str):
    """Get session summary with audit stats, chain integrity and clashes."""
    summary = orchestrator.get_session_summary(session_id)
    if "error" in summary:
        raise HTTPException(status_code=404, detail=summary["error"])
    ingest = _ingest_sources.get(session_id, {})
    if ingest:
        summary["scan_source"] = ingest.get("source")
        summary["scan_metadata"] = ingest.get("metadata", {})
    return summary


@app.get("/sessions/{session_id}/audit")
def get_session_audit(session_id: str):
    """Get full audit trail for a session."""
    events = audit.get_session_events(session_id)
    return {
        "session_id": session_id,
        "event_count": len(events),
        "events": [e.model_dump(mode="json") for e in events],
    }


@app.get("/sessions/{session_id}/chain")
def verify_chain(session_id: str):
    """Verify audit chain integrity for a session."""
    return audit.verify_chain(session_id)


# ── Step 0: Scan quality gate ──────────────────────────────────────────────────


@app.post("/sessions/{session_id}/quality-check")
def scan_quality_check(session_id: str, req: ScanQualityRequest):
    """Pre-segmentation scan quality gate.

    Call this before uploading/processing a real scan file.
    Returns {passed, density, warnings, errors}.
    If raise_on_fail=True, returns HTTP 422 when quality check fails.
    """
    if session_id not in orchestrator.sessions:
        raise HTTPException(status_code=404, detail="Session not found")
    try:
        return orchestrator.check_scan_quality(
            session_id=session_id,
            point_count=req.point_count,
            area_m2=req.area_m2,
            noise_pts=req.noise_pts,
            min_density=req.min_density_pts_per_m2,
            raise_on_fail=req.raise_on_fail,
        )
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc))


# ── Step 2: Coordinate alignment ───────────────────────────────────────────────


@app.post("/sessions/{session_id}/align")
def align_coordinates(session_id: str, req: AlignCoordinatesRequest):
    """Align scan coordinates to project/world CRS.

    Provide ≥3 survey control point pairs (scan mm ↔ world mm).
    Applies a rigid transform (Kabsch/SVD) to all stored segment
    centroids and bounding boxes.

    Returns the transform matrix, RMSE residual, and quality assessment.
    """
    if session_id not in orchestrator.sessions:
        raise HTTPException(status_code=404, detail="Session not found")
    try:
        return orchestrator.align_coordinates(
            session_id=session_id,
            control_points=req.control_points,
            max_rmse_mm=req.max_rmse_mm,
        )
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc))


# ── Clash detection standalone ─────────────────────────────────────────────────


@app.post("/sessions/{session_id}/clash-detection")
def run_clash_detection(
    session_id: str,
    tolerance_mm: float = Query(default=20.0, ge=0.0, description="Soft clash threshold in mm"),
):
    """Run or re-run BB clash detection for a session.

    Returns all detected clashes (hard and soft).
    Hard clashes are also written to the audit trail as CLASH_DETECTED events.
    """
    if session_id not in orchestrator.sessions:
        raise HTTPException(status_code=404, detail="Session not found")
    clashes = orchestrator.run_clash_detection(session_id, tolerance_mm=tolerance_mm)
    return {
        "session_id": session_id,
        "total": len(clashes),
        "hard": sum(1 for c in clashes if c.get("clash_type") == "hard"),
        "soft": sum(1 for c in clashes if c.get("clash_type") == "soft"),
        "clashes": clashes,
    }


# ── Safety gates ───────────────────────────────────────────────────────────────


@app.get("/safety-gates")
def list_safety_gates(session_id: str | None = None):
    """List pending safety gates."""
    gates = safety_gate.get_pending_gates(session_id)
    return {"pending_gates": [g.model_dump(mode="json") for g in gates]}


@app.get("/safety-gates/{gate_id}")
def get_safety_gate(gate_id: str):
    """Get a specific safety gate with its blocked instruction summary."""
    gate = safety_gate.get_gate(gate_id)
    if not gate:
        raise HTTPException(status_code=404, detail="Gate not found")

    instr = safety_gate.get_gate_instruction(gate_id)
    return {
        **gate.model_dump(mode="json"),
        "blocked_element_type": instr.element_type.value if instr else None,
        "blocked_discipline": instr.discipline.value if instr else None,
        "blocked_bounding_box": instr.bounding_box.model_dump() if instr else None,
    }


@app.post("/safety-gates/{gate_id}/decision")
def decide_gate(gate_id: str, req: GateDecisionRequest):
    """Approve or reject a safety gate.

    On approval:
    - Builds an HMAC approval signature
    - Stamps it onto the blocked instruction
    - Immediately re-dispatches the instruction to Revit
    - Updates session element count and audit trail

    On rejection:
    - Logs SAFETY_GATE_REJECTED audit event
    - Element remains unbuilt
    """
    gate = safety_gate.get_gate(gate_id)
    if not gate:
        raise HTTPException(status_code=404, detail="Gate not found")

    decision = SafetyGateDecision(
        gate_id=gate_id,
        approved=req.approved,
        approver_upn=req.approver_upn,
        comments=req.comments,
    )

    approved, instruction = safety_gate.decide_gate(decision, gate.session_id)

    if not approved:
        return {"gate_id": gate_id, "approved": False, "element_created": False}

    # Re-dispatch the approved instruction
    if not instruction:
        raise HTTPException(
            status_code=500, detail="Gate approved but instruction not found — cannot re-dispatch"
        )

    result = orchestrator._dispatch_to_revit(instruction, gate.session_id)

    # Update session element count
    session = orchestrator.sessions.get(gate.session_id)
    if session and result.success:
        session.total_elements_created += 1

    # Auto-raise NCR if confidence was low on this segment
    confidence_map = {
        s.segment_id: s.confidence for s in orchestrator._segments.get(gate.session_id, [])
    }
    confidence = confidence_map.get(instruction.segment_id, 1.0)
    ncr_id = None
    if result.success and confidence < 0.60:
        ncr = ncr_service.raise_ncr(
            session_id=gate.session_id,
            element_id=result.element_id or instruction.segment_id,
            description=(
                f"SC2-approved element with low classification confidence "
                f"({confidence:.0%}). Manual geometry verification recommended."
            ),
            severity="MINOR",
            raised_by=req.approver_upn,
        )
        ncr_id = ncr.ncr_id

    return {
        "gate_id": gate_id,
        "approved": True,
        "element_created": result.success,
        "element_id": result.element_id,
        "approver_upn": req.approver_upn,
        "ncr_raised": ncr_id,
    }


# ── Element provenance ─────────────────────────────────────────────────────────


@app.get("/elements/{element_id}/provenance")
def get_provenance(element_id: str):
    """Get full provenance trail for an element."""
    events = audit.get_element_provenance(element_id)
    return {
        "element_id": element_id,
        "event_count": len(events),
        "events": [e.model_dump(mode="json") for e in events],
    }


# ── IFC export ─────────────────────────────────────────────────────────────────


@app.get("/sessions/{session_id}/export/ifc")
def export_ifc(session_id: str, minimal: bool = Query(default=False)):
    """Export the session's classified elements as an IFC 4 file.

    Returns application/octet-stream — save as <session_id>.ifc.

    Query params:
        minimal: Use the fallback STEP writer (no ifcopenshell required).
                 The result is a valid IFC4 STEP file header with element
                 stubs but not full schema compliance.
    """
    summary = orchestrator.get_session_summary(session_id)
    if "error" in summary:
        raise HTTPException(status_code=404, detail=summary["error"])

    instructions = orchestrator._instructions.get(session_id, [])
    results = orchestrator._results.get(session_id, [])
    session = orchestrator.sessions.get(session_id)
    site_name = session.site_name if session else session_id

    from agent.tools.ifc_export import (
        export_session_to_ifc,
        export_session_to_ifc_minimal,
    )

    try:
        if minimal:
            ifc_bytes = export_session_to_ifc_minimal(session_id, instructions)
        else:
            ifc_bytes = export_session_to_ifc(
                session_id=session_id,
                instructions=instructions,
                results=results,
                site_name=site_name,
            )
    except ImportError:
        # ifcopenshell not installed — fall back to minimal writer automatically
        ifc_bytes = export_session_to_ifc_minimal(session_id, instructions)

    # Log export to audit trail
    from agent.models import AuditEvent, AuditEventType

    audit.append(
        AuditEvent(
            session_id=session_id,
            event_type=AuditEventType.IFC_EXPORTED,
            actor="api",
            detail={
                "elements": len(instructions),
                "size_kb": round(len(ifc_bytes) / 1024, 1),
                "minimal": minimal,
            },
        )
    )

    filename = f"scantobim-{session_id[:8]}.ifc"
    return StreamingResponse(
        iter([ifc_bytes]),
        media_type="application/octet-stream",
        headers={"Content-Disposition": f'attachment; filename="{filename}"'},
    )


# ── Progress streaming (SSE) ───────────────────────────────────────────────────


@app.get("/sessions/{session_id}/progress")
def stream_progress(session_id: str):
    """Server-Sent Events stream for pipeline progress.

    Connect with EventSource in the browser or `curl -N`.
    Each event is a JSON object: {"stage": "...", "ts": "...", ...}

    Events are emitted as they accumulate and drained on each poll.
    Clients should reconnect periodically if events are sparse.
    """
    if session_id not in orchestrator.sessions:
        raise HTTPException(status_code=404, detail="Session not found")

    return StreamingResponse(
        orchestrator.stream_progress(session_id),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no",
        },
    )


# ── CDE State ──────────────────────────────────────────────────────────────────


@app.get("/sessions/{session_id}/cde")
def get_cde_state(session_id: str):
    """Get current ISO 19650 CDE state for a session."""
    cde = cde_service.get_state(session_id)
    return cde.model_dump(mode="json")


@app.post("/sessions/{session_id}/cde/transition")
def cde_transition(session_id: str, req: CDETransitionRequest):
    """Transition session CDE state: WIP → SHARED → PUBLISHED."""
    try:
        transition = cde_service.transition(
            session_id=session_id,
            target_state=req.target_state,
            triggered_by=req.triggered_by,
            notes=req.notes,
        )
        return {
            "success": True,
            "from_state": transition.from_state.value,
            "to_state": transition.to_state.value,
            "at": transition.triggered_at.isoformat(),
        }
    except CDETransitionError as exc:
        raise HTTPException(status_code=422, detail=str(exc))


# ── NCR endpoints ──────────────────────────────────────────────────────────────


@app.post("/sessions/{session_id}/ncrs")
def raise_ncr(session_id: str, req: RaiseNCRRequest):
    """Raise a Non-Conformance Report for an element."""
    ncr = ncr_service.raise_ncr(
        session_id=session_id,
        element_id=req.element_id,
        description=req.description,
        severity=req.severity,
        raised_by=req.raised_by,
    )
    return ncr.model_dump(mode="json")


@app.get("/sessions/{session_id}/ncrs")
def list_ncrs(session_id: str, status: str | None = None):
    """List NCRs for a session. Filter by ?status=OPEN|RESOLVED|CLOSED."""
    records = ncr_service.list_session(session_id)
    if status:
        records = [n for n in records if n.status.value == status.upper()]
    return {
        "session_id": session_id,
        "summary": ncr_service.summary(session_id),
        "ncrs": [n.model_dump(mode="json") for n in records],
    }


@app.patch("/ncrs/{ncr_id}/resolve")
def resolve_ncr(ncr_id: str, req: ResolveNCRRequest):
    """Resolve an open NCR."""
    try:
        ncr = ncr_service.resolve_ncr(
            ncr_id=ncr_id,
            resolved_by=req.resolved_by,
            resolution=req.resolution,
        )
        return ncr.model_dump(mode="json")
    except NCRNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc))
    except NCRAlreadyResolvedError as exc:
        raise HTTPException(status_code=409, detail=str(exc))


# ── Sprint 1: Scan file upload ─────────────────────────────────────────────────

# Cross-platform default: Ubuntu → /tmp/scantobim/scans, Windows → %TEMP%\scantobim\scans.
import tempfile as _tempfile

_DEFAULT_SCAN_DIR = Path(_tempfile.gettempdir()) / "scantobim" / "scans"
_SCAN_DIR = Path(os.environ.get("SCAN_FILES_DIR", str(_DEFAULT_SCAN_DIR)))
_SCAN_DIR.mkdir(parents=True, exist_ok=True)

# In-memory job tracker (replace with DB in Sprint 4)
# job_id → {status, session_id, file, ...}
_jobs: dict[str, dict] = {}
_ingest_sources: dict[str, dict] = {}  # session_id → {source, metadata}


def _run_pipeline_background(session_id: str, file_path: Path, zone_id: str) -> None:
    """Background task: run full pipeline on uploaded file.

    Called by FastAPI BackgroundTasks (no Celery required for dev).
    When Redis is available the worker.process_scan_file Celery task is used instead.
    """

    # Find matching job entry
    job_id = next(
        (jid for jid, j in _jobs.items() if j.get("session_id") == session_id),
        None,
    )

    try:
        if job_id:
            _jobs[job_id]["status"] = "running"

        orchestrator._push_progress(
            session_id,
            "LOAD",
            {
                "message": f"Loading {file_path.name}",
            },
        )

        segments = orchestrator.run_segmentation(
            session_id=session_id,
            file_path=file_path,
            zone_id=zone_id,
            use_synthetic=False,
        )
        orchestrator._push_progress(
            session_id,
            "SEGMENTED",
            {
                "message": f"{len(segments)} segments found",
                "segment_count": len(segments),
            },
        )

        instructions = orchestrator.classify_and_prepare(session_id, segments)
        orchestrator._push_progress(
            session_id,
            "CLASSIFIED",
            {
                "message": f"{len(instructions)} elements classified",
            },
        )

        clashes = orchestrator.run_clash_detection(session_id)
        result = orchestrator.execute_instructions(session_id, ncr_service=ncr_service)

        orchestrator._push_progress(
            session_id,
            "COMPLETE",
            {
                "message": "Pipeline complete",
                "segments": len(segments),
                "clashes": len(clashes),
            },
        )

        if job_id:
            _jobs[job_id].update(
                {
                    "status": "complete",
                    "segments": len(segments),
                    "clashes": len(clashes),
                    "result": result,
                }
            )

    except Exception as exc:  # noqa: BLE001
        log.error("background_pipeline_failed", session_id=session_id, error=str(exc))
        orchestrator._push_progress(
            session_id,
            "ERROR",
            {
                "message": f"Pipeline failed: {exc}",
            },
        )
        if job_id:
            _jobs[job_id].update({"status": "failed", "error": str(exc)})


@app.post("/sessions/{session_id}/upload", tags=["scans"])
async def upload_scan(
    session_id: str,
    background_tasks: BackgroundTasks,
    file: UploadFile = File(..., description="Point cloud file (.e57, .las, .laz, .ply, .xyz)"),
    zone_id: str = Query("zone-001", description="Zone/floor identifier"),
    user: CurrentUser = Depends(require_feature("scan_upload")),
):
    """Upload a real point cloud file and trigger the full pipeline asynchronously.

    Accepted formats: .e57, .las, .laz, .ply, .pcd, .xyz
    Max size: configured per tier (Starter 200 MB, Pro 2 GB, Enterprise 10 GB)

    Returns immediately with a job_id. Poll progress via:
      GET /sessions/{session_id}/progress   (SSE stream)
      GET /jobs/{job_id}                    (status snapshot)
    """
    import uuid

    from agent.tools.scan_tools import SUPPORTED_EXTENSIONS, validate_scan_file

    # Validate session exists
    summary = orchestrator.get_session_summary(session_id)
    if "error" in summary:
        raise HTTPException(status_code=404, detail=f"Session not found: {session_id}")

    # Validate filename / extension before saving
    filename = file.filename or "scan.bin"
    file_ext = Path(filename).suffix.lower()
    if file_ext not in SUPPORTED_EXTENSIONS:
        raise HTTPException(
            status_code=422,
            detail=(
                f"Unsupported file type '{file_ext}'. Accepted: {sorted(SUPPORTED_EXTENSIONS)}"
            ),
        )

    # Save upload to scan directory
    dest_path = _SCAN_DIR / f"{session_id}{file_ext}"
    try:
        content = await file.read()
        dest_path.write_bytes(content)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Failed to save file: {exc}") from exc

    # Quick size + magic-byte validation (no full load yet)
    from agent.licensing import feature_limit

    max_mb = feature_limit(user.tenant_id, "max_scan_size_mb")
    info = validate_scan_file(dest_path, max_size_mb=max_mb)
    if not info.valid:
        dest_path.unlink(missing_ok=True)
        raise HTTPException(status_code=422, detail="; ".join(info.errors))

    # Register job
    job_id = str(uuid.uuid4())
    _jobs[job_id] = {
        "job_id": job_id,
        "session_id": session_id,
        "file": str(dest_path),
        "format": info.format,
        "size_mb": round(info.size_mb, 2),
        "status": "queued",
        "submitted_by": user.username,
    }

    # Try Celery first; fall back to FastAPI background task
    try:
        from agent.worker import process_scan_file

        celery_result = process_scan_file.apply_async(
            args=[session_id, str(dest_path), zone_id],
            task_id=job_id,
        )
        _jobs[job_id]["backend"] = "celery"
        log.info("scan_queued_celery", job_id=job_id, session_id=session_id)
    except Exception:  # noqa: BLE001
        # Celery/Redis not available — run inline as background task
        background_tasks.add_task(_run_pipeline_background, session_id, dest_path, zone_id)
        _jobs[job_id]["backend"] = "fastapi_background"
        log.info("scan_queued_background", job_id=job_id, session_id=session_id)

    return {
        "job_id": job_id,
        "session_id": session_id,
        "file": filename,
        "format": info.format,
        "size_mb": round(info.size_mb, 2),
        "status": "queued",
        "warnings": info.warnings,
        "progress_url": f"/sessions/{session_id}/progress",
        "job_url": f"/jobs/{job_id}",
    }


@app.post("/sessions/{session_id}/ingest-elements", tags=["scans"])
def ingest_elements(
    session_id: str,
    payload: IngestElementsRequest,
    user: CurrentUser = Depends(get_current_user),
):
    """Accept pre-processed segments from the Revit plugin sidecar.

    Runs classifier → safety gate evaluation → clash detection without
    requiring a raw scan file upload.  The existing file-upload endpoint
    (/sessions/{id}/upload) remains available for small files.

    Returns a summary the Revit addin can display in-panel.
    """
    from agent.models import AuditEvent, AuditEventType, GeometrySegment

    if session_id not in orchestrator.sessions:
        raise HTTPException(status_code=404, detail="Session not found")

    # ── Parse segments ─────────────────────────────────────────────────────
    try:
        segments = [GeometrySegment(**s) for s in payload.segments]
    except Exception as exc:
        raise HTTPException(status_code=422, detail=f"Invalid segment data: {exc}") from exc

    if not segments:
        raise HTTPException(status_code=422, detail="segments list must not be empty")

    # ── Store in orchestrator (replaces run_segmentation step) ─────────────
    orchestrator._segments[session_id] = segments
    _ingest_sources[session_id] = {"source": payload.source, "metadata": payload.scan_metadata}
    session = orchestrator.sessions[session_id]
    session.zone_ids = list({s.zone_id for s in segments})
    session.total_segments = len(segments)

    # NQA-1 provenance — input hash reported by the sidecar (if any)
    sidecar_input_hash = payload.scan_metadata.get("input_hash") if payload.scan_metadata else None
    sidecar_source = (
        payload.scan_metadata.get("input_file") if payload.scan_metadata else payload.source
    )
    orchestrator._record_provenance(
        session_id=session_id,
        segments=segments,
        input_hash=sidecar_input_hash,
        input_source=sidecar_source,
    )

    # ── Classify and prepare instructions ──────────────────────────────────
    instructions = orchestrator.classify_and_prepare(session_id, segments)

    # ── Evaluate safety gates (no Revit push — addin already has geometry) ─
    sc1_blocked = 0
    sc2_pending = 0
    for instruction in instructions:
        try:
            gate = safety_gate.check_instruction(instruction, session_id)
            if gate is not None:
                sc2_pending += 1
        except Exception:  # SafetyCategoryViolationError → SC1 hard block
            sc1_blocked += 1

    # ── Clash detection ────────────────────────────────────────────────────
    clashes = orchestrator.run_clash_detection(session_id)

    # ── Audit ──────────────────────────────────────────────────────────────
    audit.append(
        AuditEvent(
            session_id=session_id,
            event_type=AuditEventType.SESSION_STARTED,
            actor=user.username,
            detail={
                "ingest_source": payload.source,
                "segments_received": len(segments),
                "elements_classified": len(instructions),
                "sc1_blocked": sc1_blocked,
                "sc2_pending": sc2_pending,
                "clash_count": len(clashes),
                "scan_metadata": payload.scan_metadata,
            },
        )
    )

    return {
        "session_id": session_id,
        "segments_ingested": len(segments),
        "elements_classified": len(instructions),
        "sc1_blocked": sc1_blocked,
        "gates_pending": sc2_pending,
        "clash_count": len(clashes),
        "source": payload.source,
    }


@app.get("/jobs/{job_id}", tags=["scans"])
def get_job(job_id: str, user: CurrentUser = Depends(get_current_user)):
    """Get the current status of a background scan processing job."""
    job = _jobs.get(job_id)
    if not job:
        raise HTTPException(status_code=404, detail=f"Job not found: {job_id}")

    # If Celery backend, enrich with live Celery status
    if job.get("backend") == "celery":
        try:
            from agent.worker import get_job_status

            celery_info = get_job_status(job_id)
            job = {**job, **celery_info}
        except Exception:  # noqa: BLE001
            pass

    return job


@app.get("/jobs", tags=["scans"])
def list_jobs(
    session_id: str | None = Query(None, description="Filter by session"),
    user: CurrentUser = Depends(get_current_user),
):
    """List all scan processing jobs, optionally filtered by session."""
    jobs = list(_jobs.values())
    if session_id:
        jobs = [j for j in jobs if j.get("session_id") == session_id]
    return {"jobs": jobs, "count": len(jobs)}


# ── Sprint 2: Revit Bridge endpoints ──────────────────────────────────────────

# Per-session callback store: session_id → list[BridgeActionResult]
_bridge_callbacks: dict[str, list[dict]] = {}

# Shared bridge client (singleton, lazy-init)
_bridge_client: RevitBridgeClient | None = None


def _get_bridge_client() -> RevitBridgeClient:
    global _bridge_client
    if _bridge_client is None:
        bridge_url = os.environ.get("REVIT_BRIDGE_URL", "http://localhost:8766")
        _bridge_client = RevitBridgeClient(base_url=bridge_url)
    return _bridge_client


@app.get("/bridge/health", tags=["revit-bridge"])
def bridge_health():
    """Check if the Revit bridge (port 8766) is running and responsive.

    Returns bridge status, queue depth, and element creation stats.
    This endpoint does NOT require authentication so monitoring tools can call it.
    """
    client = _get_bridge_client()
    health = client.get_health()
    return health.model_dump()


@app.post("/sessions/{session_id}/revit-callback", tags=["revit-bridge"])
def revit_callback(session_id: str, result: BridgeActionResult):
    """Receive an ActionResult pushed by the Revit bridge after element creation.

    The Revit addin calls this endpoint after processing each ElementInstruction.
    No authentication required — called from localhost Revit process only.

    The result is stored and immediately merged into the session's action results.
    """
    # Validate session exists
    if session_id not in orchestrator.sessions:
        raise HTTPException(status_code=404, detail=f"Session not found: {session_id}")

    # Store in per-session callback list
    if session_id not in _bridge_callbacks:
        _bridge_callbacks[session_id] = []
    _bridge_callbacks[session_id].append(result.model_dump())

    # Also store in global callback store (for polling path)
    store_callback_result(result)

    # Update orchestrator results if session results list exists
    if session_id in orchestrator._results:
        from agent.models import ActionResult

        action_result = ActionResult(
            success=result.success,
            element_id=result.element_id,
            instruction_id=result.instruction_id,
            error=result.error,
            duration_ms=result.duration_ms,
        )
        orchestrator._results[session_id].append(action_result)
        if result.success:
            session = orchestrator.sessions.get(session_id)
            if session:
                session.total_elements_created += 1

    # Push SSE progress event
    orchestrator._push_progress(
        session_id,
        "REVIT_CALLBACK",
        {
            "instruction_id": result.instruction_id,
            "success": result.success,
            "element_id": result.element_id,
            "error": result.error,
            "family": result.family_name,
            "type": result.type_name,
            "resolution": result.resolution_path,
        },
    )

    log.info(
        "revit_callback_received",
        session_id=session_id[:8],
        instruction_id=result.instruction_id[:8],
        success=result.success,
    )
    return {"accepted": True, "instruction_id": result.instruction_id}


@app.get("/sessions/{session_id}/revit-results", tags=["revit-bridge"])
def get_revit_results(
    session_id: str,
    user: CurrentUser = Depends(get_current_user),
):
    """List all Revit element creation results received via callback for a session."""
    if session_id not in orchestrator.sessions:
        raise HTTPException(status_code=404, detail=f"Session not found: {session_id}")

    callbacks = _bridge_callbacks.get(session_id, [])
    results = orchestrator._results.get(session_id, [])

    total = len(results)
    succeeded = sum(1 for r in results if r.success)
    failed = sum(1 for r in results if not r.success)

    return {
        "session_id": session_id,
        "total": total,
        "succeeded": succeeded,
        "failed": failed,
        "callbacks": len(callbacks),
        "results": [r.model_dump(mode="json") for r in results],
    }


@app.post("/bridge/push/{session_id}", tags=["revit-bridge"])
def push_to_bridge(
    session_id: str,
    instruction_id: str = Query(..., description="Instruction ID to push to Revit"),
    user: CurrentUser = Depends(require_feature("revit_bridge")),
):
    """Manually push a single instruction to the Revit bridge.

    Useful for re-triggering a failed element or testing the bridge connection.
    The instruction must already exist in the session's instruction list.
    """
    if session_id not in orchestrator.sessions:
        raise HTTPException(status_code=404, detail=f"Session not found: {session_id}")

    instructions = orchestrator._instructions.get(session_id, [])
    instruction = next((i for i in instructions if i.instruction_id == instruction_id), None)
    if instruction is None:
        raise HTTPException(status_code=404, detail=f"Instruction not found: {instruction_id}")

    client = _get_bridge_client()
    if not client.is_healthy():
        raise HTTPException(
            status_code=503,
            detail=(
                "Revit bridge is not reachable on port 8766. "
                "Open Revit → ScanToBIM tab → 'Start Agent' to launch the bridge."
            ),
        )

    payload = instruction.model_dump(mode="json")
    queued = client.push_instruction(payload)
    if queued is None:
        raise HTTPException(status_code=502, detail="Bridge rejected the instruction.")

    return {
        "queued": queued.queued,
        "instruction_id": queued.instruction_id,
        "poll_url": f"http://localhost:8766/revit-results/{instruction_id}",
        "callback_url": f"http://localhost:8765/sessions/{session_id}/revit-callback",
    }


# ── Sprint 3: Teams Adaptive Card notifications ────────────────────────────────


def _post_teams_card(
    webhook_url: str,
    title: str,
    site_name: str,
    element_type: str,
    zone_id: str,
    gate_id: str,
    approve_url: str,
    reject_url: str,
) -> bool:
    """Post a Teams Adaptive Card to an Incoming Webhook URL.

    Sends a rich Adaptive Card (1.4) with:
      - Colour-coded SC2 header (amber)
      - Element details (site, zone, type, gate ID)
      - ✓ Approve and ✗ Reject Action.OpenUrl buttons
        which open the ScanToBIM UI with pre-filled gate ID + action,
        so the engineer approves/rejects in one click from the browser.

    Works with any Teams channel Incoming Webhook — no bot required.
    Returns True on HTTP 200 from Teams.
    """
    import json as _json
    import urllib.error as _err
    import urllib.request as _req

    card_payload = {
        "type": "message",
        "attachments": [
            {
                "contentType": "application/vnd.microsoft.card.adaptive",
                "contentUrl": None,
                "content": {
                    "$schema": "http://adaptivecards.io/schemas/adaptive-card.json",
                    "type": "AdaptiveCard",
                    "version": "1.4",
                    "body": [
                        # ── Amber header bar ─────────────────────────────────────
                        {
                            "type": "ColumnSet",
                            "style": "warning",
                            "bleed": True,
                            "columns": [
                                {
                                    "type": "Column",
                                    "width": "stretch",
                                    "items": [
                                        {
                                            "type": "TextBlock",
                                            "text": "⚠ SC2 Approval Required",
                                            "weight": "Bolder",
                                            "size": "Medium",
                                            "color": "Warning",
                                        }
                                    ],
                                }
                            ],
                        },
                        # ── Body ─────────────────────────────────────────────────
                        {
                            "type": "TextBlock",
                            "text": title,
                            "weight": "Bolder",
                            "size": "Default",
                            "wrap": True,
                            "spacing": "Medium",
                        },
                        {
                            "type": "FactSet",
                            "spacing": "Small",
                            "facts": [
                                {"title": "Site", "value": site_name},
                                {"title": "Zone", "value": zone_id or "—"},
                                {
                                    "title": "Element",
                                    "value": element_type.replace("_", " ").title(),
                                },
                                {"title": "Gate ID", "value": gate_id[:8]},
                                {"title": "Standard", "value": "NQA-1 / ISO 19650"},
                            ],
                        },
                        {
                            "type": "TextBlock",
                            "text": "This SC2-classified element is paused until an authorised engineer approves or rejects it. "
                            "Click a button below — your decision is HMAC-signed and audit-logged.",
                            "wrap": True,
                            "size": "Small",
                            "color": "Accent",
                            "spacing": "Medium",
                        },
                    ],
                    "actions": [
                        {
                            "type": "Action.OpenUrl",
                            "title": "✓  Approve",
                            "url": approve_url,
                            "style": "positive",
                        },
                        {
                            "type": "Action.OpenUrl",
                            "title": "✗  Reject",
                            "url": reject_url,
                            "style": "destructive",
                        },
                    ],
                },
            }
        ],
    }

    data = _json.dumps(card_payload).encode("utf-8")
    try:
        r = _req.urlopen(
            _req.Request(
                webhook_url,
                data=data,
                headers={"Content-Type": "application/json"},
                method="POST",
            ),
            timeout=5,
        )
        return r.status == 200
    except _err.URLError as exc:
        log.warning("teams_webhook_failed", error=str(exc))
        return False


@app.post("/sessions/{session_id}/notify-teams", tags=["revit-bridge"])
def notify_teams(
    session_id: str,
    user: CurrentUser = Depends(get_current_user),
):
    """POST a Teams Adaptive Card for all pending SC2 gates in this session.

    Requires TEAMS_WEBHOOK_URL env var to be set. The card links back to
    the Safety Gates review page so the engineer can approve/reject.
    """
    if session_id not in orchestrator.sessions:
        raise HTTPException(status_code=404, detail=f"Session not found: {session_id}")

    webhook_url = os.environ.get("TEAMS_WEBHOOK_URL", "")
    if not webhook_url:
        raise HTTPException(
            status_code=400,
            detail="TEAMS_WEBHOOK_URL is not configured. Set it in .env to enable Teams notifications.",
        )

    # Find pending gates for this session
    from agent.models import SafetyGateStatus

    pending = [
        g
        for g in safety_gate._pending_gates.values()
        if g.session_id == session_id and g.status == SafetyGateStatus.PENDING
    ]

    if not pending:
        return {"sent": 0, "message": "No pending SC2 gates for this session."}

    session = orchestrator.sessions[session_id]
    agent_url = os.environ.get("AGENT_PUBLIC_URL", "http://localhost:8765")

    sent = 0
    for gate in pending:
        zone_id = getattr(gate, "zone_id", "—") or "—"
        title = f"SC2 Approval Required — {gate.element_type.replace('_', ' ').title()}"

        # Deep-link URLs: open gates.html with gate_id + action pre-filled
        # gates.html reads these params and auto-opens the decision modal
        approve_url = (
            f"{agent_url}/ui/gates.html?gate_id={gate.gate_id}&action=approve&session={session_id}"
        )
        reject_url = (
            f"{agent_url}/ui/gates.html?gate_id={gate.gate_id}&action=reject&session={session_id}"
        )

        ok = _post_teams_card(
            webhook_url=webhook_url,
            title=title,
            site_name=session.site_name,
            element_type=gate.element_type,
            zone_id=zone_id,
            gate_id=gate.gate_id,
            approve_url=approve_url,
            reject_url=reject_url,
        )
        if ok:
            sent += 1
        else:
            log.warning("teams_card_failed", gate_id=gate.gate_id[:8])

    gates_url = f"{agent_url}/ui/gates.html?session={session_id}"
    log.info("teams_notification_sent", session_id=session_id[:8], gates_sent=sent)
    return {"sent": sent, "total_pending": len(pending), "gates_url": gates_url}


# ── DRP ────────────────────────────────────────────────────────────────────────


@app.get("/sessions/{session_id}/drp")
def get_drp(session_id: str):
    """Generate and return the Design Record Package as HTML."""
    from agent.drp import generate_drp

    summary = orchestrator.get_session_summary(session_id)
    if "error" in summary:
        raise HTTPException(status_code=404, detail=summary["error"])

    events_resp = audit.get_session_events(session_id)
    events_dict = [e.model_dump(mode="json") for e in events_resp]
    chain = audit.verify_chain(session_id)
    cde = cde_service.get_state(session_id)
    ncr_records = [n.model_dump(mode="json") for n in ncr_service.list_session(session_id)]
    session_data = summary.get("session", summary)

    html = generate_drp(
        session_id=session_id,
        session=session_data,
        audit_events=events_dict,
        ncr_records=ncr_records,
        chain_result=chain,
        cde_state=cde.current_state.value,
    )
    return HTMLResponse(content=html, status_code=200)


# ── Sprint 3.5: Scan Intelligence ─────────────────────────────────────────────


@app.post("/sessions/{session_id}/register-scans", tags=["scan-intelligence"])
async def register_scans(
    session_id: str,
    files: list[UploadFile] = File(..., description="Scan files to register (>=2 required)"),
    voxel_size: float = Query(default=0.05, gt=0.0, description="ICP voxel size in metres"),
    run_pipeline: bool = Query(default=True, description="Auto-run segmentation on merged cloud"),
    user: CurrentUser = Depends(get_current_user),
):
    """Register multiple scan stations into one merged cloud via ICP, then optionally run segmentation."""
    if session_id not in orchestrator.sessions:
        raise HTTPException(status_code=404, detail=f"Session not found: {session_id}")
    if len(files) < 2:
        raise HTTPException(
            status_code=422,
            detail="At least 2 scan files are required for ICP registration.",
        )

    tmp_dir = Path(tempfile.mkdtemp(prefix=f"stb_{session_id[:8]}_"))
    try:
        saved_paths: list[Path] = []
        for i, upload in enumerate(files):
            # Use only the filename stem — strip all directory components to
            # prevent path-traversal attacks (e.g. "../../etc/passwd").
            raw_name = Path(upload.filename or f"scan_{i}.bin").name
            dest = tmp_dir / raw_name
            content = await upload.read()
            dest.write_bytes(content)
            saved_paths.append(dest)

        result = orchestrator.register_scans(session_id, saved_paths, voxel_size=voxel_size)

        seg_count = 0
        if run_pipeline:
            merged_pts = orchestrator._merged_clouds.get(session_id)
            segs = orchestrator.run_segmentation(
                session_id,
                preloaded_points=merged_pts,
                zone_id=orchestrator.sessions[session_id].zone_ids[0]
                if orchestrator.sessions[session_id].zone_ids
                else "zone-001",
            )
            seg_count = len(segs)
    finally:
        shutil.rmtree(tmp_dir, ignore_errors=True)

    return {
        **result.model_dump(mode="json"),
        "segments_after_merge": seg_count,
    }


@app.get("/sessions/{session_id}/provenance", tags=["scan-intelligence"])
def get_provenance(
    session_id: str,
    user: CurrentUser = Depends(get_current_user),
):
    """NQA-1 § 401 determinism record for this session.

    Returns the pinned random seed, software identity hash, input file
    hash (if available), output hash (SHA-256 over canonical segments),
    segment count, and generation timestamp. A reviewer comparing two
    runs of the same input under the same software version should see
    the same output_hash if determinism is preserved.
    """
    if session_id not in orchestrator.sessions:
        raise HTTPException(status_code=404, detail=f"Session not found: {session_id}")

    record = orchestrator._provenance.get(session_id)
    if record is None:
        raise HTTPException(
            status_code=404,
            detail="No provenance record — run segmentation or ingest-elements first.",
        )
    return record


@app.get("/sessions/{session_id}/nqa1-package", tags=["scan-intelligence"])
def get_nqa1_package(
    session_id: str,
    format: str = Query(
        default="json",
        pattern="^(json|html)$",
        description="'json' structured record, 'html' printable V&V package",
    ),
    user: CurrentUser = Depends(get_current_user),
):
    """Download the NQA-1 Subpart 2.7 V&V package for this session.

    Combines software identity, requirements traceability, test results
    (when a JUnit XML exists), audit trail, chain integrity, NCR status,
    and a signature block into a single auditable deliverable per
    ASME NQA-1 Part I § 302 and Subpart 2.7 § 401-402.
    """
    if session_id not in orchestrator.sessions:
        raise HTTPException(status_code=404, detail=f"Session not found: {session_id}")

    from agent.tools.nqa1_package import build_nqa1_package

    try:
        # Auto-sign preparer if not already signed — agent is the author
        # of the V&V evidence; human roles (Verifier, Approver) still gate
        # qualification use.
        _auto_preparer_sign(session_id)

        html, json_doc = build_nqa1_package(
            session_id=session_id,
            orchestrator=orchestrator,
            audit=audit,
            ncr_service=ncr_service,
            signature_service=nqa1_signatures,
        )
    except KeyError as exc:
        raise HTTPException(status_code=404, detail=str(exc))

    if format == "html":
        return Response(content=html, media_type="text/html")
    return json_doc


def _auto_preparer_sign(session_id: str) -> None:
    """Lazily sign as Preparer on first NQA-1 package fetch if unsigned.

    Idempotent: no-op if Preparer is already signed for this session.
    The signing actor is the agent itself — this represents 'the agent
    authored this evidence', not human sign-off. Human Verifier /
    Approver roles must be completed externally.
    """
    from agent.models import NQA1Role
    from agent.nqa1_signature import (
        DuplicateRoleError,
        DuplicateSignerError,
        SignatureOrderError,
        compute_package_hash,
    )
    from agent.tools.nqa1_package import build_nqa1_package

    # Skip if preparer already present (active or revoked)
    for s in nqa1_signatures.list_session(session_id):
        if s.role == NQA1Role.PREPARER and s.revoked_at_utc is None:
            return

    # Build a package with NO signature service so package_hash is stable
    # across auto-sign and subsequent reads
    try:
        _, package_doc = build_nqa1_package(
            session_id=session_id,
            orchestrator=orchestrator,
            audit=audit,
            ncr_service=ncr_service,
            signature_service=None,
        )
    except KeyError:
        return  # session gone or empty — let the outer handler return 404

    package_hash = compute_package_hash(package_doc)

    try:
        nqa1_signatures.sign(
            session_id=session_id,
            role=NQA1Role.PREPARER,
            signer_upn="agent@scantobim.ai",
            signer_name="ScanToBIM Agent (auto-preparer)",
            package_hash=package_hash,
            comments="Auto-signed on first V&V package generation. "
            "Human Verifier and Approver signatures required "
            "before qualification use.",
        )
    except (SignatureOrderError, DuplicateRoleError, DuplicateSignerError):
        pass  # race — another request already signed


# ── P2.4++ NQA-1 three-party signature endpoints ──────────────────────────────


class _SignRequest(BaseModel):
    role: str
    signer_upn: str
    signer_name: str
    comments: str = ""


class _RevokeRequest(BaseModel):
    reason: str


@app.post("/sessions/{session_id}/nqa1-signatures", tags=["scan-intelligence"])
def post_nqa1_signature(
    session_id: str,
    payload: _SignRequest,
    user: CurrentUser = Depends(get_current_user),
):
    """Sign the session's NQA-1 V&V package as preparer / verifier / approver.

    Ordering enforced: preparer → verifier → approver. Same UPN may not
    hold two roles (separation of duties). Each signature HMAC-chains to
    the previous one and to the SHA-256 of the NQA-1 package JSON.
    """
    if session_id not in orchestrator.sessions:
        raise HTTPException(status_code=404, detail=f"Session not found: {session_id}")

    from agent.models import NQA1Role
    from agent.nqa1_signature import (
        DuplicateRoleError,
        DuplicateSignerError,
        SignatureOrderError,
        compute_package_hash,
    )
    from agent.tools.nqa1_package import build_nqa1_package

    try:
        role = NQA1Role(payload.role)
    except ValueError:
        raise HTTPException(
            status_code=422,
            detail=f"role must be one of preparer|verifier|approver, got {payload.role!r}",
        )

    # Build the package fresh so we sign the *current* content
    _, package_doc = build_nqa1_package(
        session_id=session_id,
        orchestrator=orchestrator,
        audit=audit,
        ncr_service=ncr_service,
        signature_service=nqa1_signatures,
    )
    package_hash = compute_package_hash(package_doc)

    try:
        sig = nqa1_signatures.sign(
            session_id=session_id,
            role=role,
            signer_upn=payload.signer_upn,
            signer_name=payload.signer_name,
            package_hash=package_hash,
            comments=payload.comments,
        )
    except SignatureOrderError as exc:
        raise HTTPException(status_code=409, detail=str(exc))
    except DuplicateRoleError as exc:
        raise HTTPException(status_code=409, detail=str(exc))
    except DuplicateSignerError as exc:
        raise HTTPException(status_code=409, detail=str(exc))

    return sig.model_dump(mode="json")


@app.get("/sessions/{session_id}/nqa1-signatures", tags=["scan-intelligence"])
def get_nqa1_signatures(
    session_id: str,
    user: CurrentUser = Depends(get_current_user),
):
    """List all NQA-1 signatures for a session with chain verification."""
    if session_id not in orchestrator.sessions:
        raise HTTPException(status_code=404, detail=f"Session not found: {session_id}")

    status = nqa1_signatures.status(session_id)
    sigs = nqa1_signatures.list_session(session_id)
    return {
        **status,
        "signatures": [s.model_dump(mode="json") for s in sigs],
    }


@app.patch("/nqa1-signatures/{signature_id}/revoke", tags=["scan-intelligence"])
def patch_nqa1_signature_revoke(
    signature_id: str,
    payload: _RevokeRequest,
    user: CurrentUser = Depends(get_current_user),
):
    """Revoke a signature (append-only — original row preserved).

    After revocation the signer must re-sign with fresh package_hash.
    Downstream signatures (e.g. if approver revoked after verifier)
    remain in the ledger but `fully_signed` becomes False.
    """
    try:
        sig = nqa1_signatures.revoke(
            signature_id=signature_id,
            revoked_by=user.username if hasattr(user, "username") else "api",
            revocation_reason=payload.reason,
        )
    except KeyError:
        raise HTTPException(status_code=404, detail=f"Signature not found: {signature_id}")

    return sig.model_dump(mode="json")


@app.get("/sessions/{session_id}/iso15926", tags=["scan-intelligence"])
def get_iso15926(
    session_id: str,
    format: str = Query(default="json", pattern="^(json|csv)$"),
    user: CurrentUser = Depends(get_current_user),
):
    """ISO 15926-4 class mapping + CFIHOS attribute handover data.

    Every classified element in the session is mapped to an ISO 15926-4
    reference class URI with a CFIHOS (IOGP v1.5) attribute payload
    populated from scan-derived data where possible, marked PENDING
    where operator-supplied data is needed.

    CSV format matches the tag-attribute layout ingested by AVEVA NET,
    Hexagon SDx, and similar EPC handover systems.
    """
    if session_id not in orchestrator.sessions:
        raise HTTPException(status_code=404, detail=f"Session not found: {session_id}")

    instructions = orchestrator._instructions.get(session_id, [])
    if not instructions:
        raise HTTPException(
            status_code=422,
            detail="No classified instructions — run classify_and_prepare first.",
        )

    from agent.tools.iso15926_tools import (
        generate_cfihos_csv,
        generate_iso15926_json,
        map_session,
    )

    mappings = map_session(instructions)

    if format == "csv":
        return Response(
            content=generate_cfihos_csv(mappings),
            media_type="text/csv",
            headers={
                "Content-Disposition": f'attachment; filename="cfihos-{session_id[:8]}.csv"',
            },
        )
    return generate_iso15926_json(session_id, mappings)


@app.get("/sessions/{session_id}/fmea", tags=["scan-intelligence"])
def get_fmea(
    session_id: str,
    format: str = Query(default="html", pattern="^(json|html)$"),
    user: CurrentUser = Depends(get_current_user),
):
    """Failure Mode and Effects Analysis (FMEA) for this session.

    Per NQA-1 Subpart 2.7 § 102 + IEC 61508-3 § 7.4 — every safety-
    related software function has documented failure modes with
    severity / likelihood / detection scoring and a named verifying test.
    The catalog is static (version-pinned in agent/tools/fmea.py) so the
    report is reproducible across runs with the same software identity.
    """
    if session_id not in orchestrator.sessions:
        raise HTTPException(status_code=404, detail=f"Session not found: {session_id}")

    from agent.tools.fmea import build_fmea_document, render_fmea_html

    doc = build_fmea_document(session_id)

    if format == "html":
        return Response(content=render_fmea_html(doc), media_type="text/html")
    return doc


@app.get("/sessions/{session_id}/handover-bundle", tags=["scan-intelligence"])
def get_handover_bundle(
    session_id: str,
    user: CurrentUser = Depends(get_current_user),
):
    """Download a complete handover .zip for this session.

    The archive contains every deliverable artifact (IFC, LOA, deviation
    heatmap, registration QA, NCRs, audit trail, chain integrity, and
    Design Record Package). Sections are skipped gracefully when their
    upstream data isn't yet available — see manifest.json in the zip.
    """
    if session_id not in orchestrator.sessions:
        raise HTTPException(status_code=404, detail=f"Session not found: {session_id}")

    from agent.tools.handover_bundle import build_handover_bundle

    # Auto-sign preparer so the bundled NQA-1 package reflects the agent sig
    _auto_preparer_sign(session_id)

    try:
        zip_bytes, sections = build_handover_bundle(
            session_id=session_id,
            orchestrator=orchestrator,
            audit=audit,
            ncr_service=ncr_service,
            signature_service=nqa1_signatures,
        )
    except KeyError as exc:
        raise HTTPException(status_code=404, detail=str(exc))

    filename = f"scantobim-handover-{session_id[:8]}.zip"
    summary = ",".join(f"{s.name}={'ok' if s.included else 'skip'}" for s in sections)
    return StreamingResponse(
        iter([zip_bytes]),
        media_type="application/zip",
        headers={
            "Content-Disposition": f'attachment; filename="{filename}"',
            "X-ScanToBIM-Sections": summary,
        },
    )


@app.get("/sessions/{session_id}/registration-report", tags=["scan-intelligence"])
def get_registration_report(
    session_id: str,
    format: str = Query(
        default="json",
        pattern="^(json|html)$",
        description="'json' returns structured report, 'html' returns printable",
    ),
    user: CurrentUser = Depends(get_current_user),
):
    """RICS / PAS 128 Registration QA report for the session.

    Classifies every scan station + the session overall against industry
    survey tiers (Cat A ≤3 mm, Cat B ≤10 mm, Cat C ≤50 mm, else FAIL).
    Requires register-scans to have been run first.
    """
    if session_id not in orchestrator.sessions:
        raise HTTPException(status_code=404, detail=f"Session not found: {session_id}")

    reg_result = orchestrator._registration.get(session_id)
    if reg_result is None:
        raise HTTPException(
            status_code=404,
            detail="No registration result available. Run POST /register-scans first.",
        )

    from agent.tools.registration_report import (
        build_qa_report,
        render_registration_html,
    )

    report = build_qa_report(reg_result)

    if format == "html":
        html = render_registration_html(report)
        return Response(content=html, media_type="text/html")

    return report.to_dict()


@app.post("/sessions/{session_id}/deviation-report", tags=["scan-intelligence"])
async def create_deviation_report(
    session_id: str,
    ifc_file: UploadFile = File(..., description="Reference IFC design model (.ifc)"),
    matched_threshold_mm: float = Query(default=25.0, gt=0.0, description="MATCHED threshold mm"),
    shifted_threshold_mm: float = Query(
        default=150.0, gt=0.0, description="SHIFTED/MISSING boundary mm"
    ),
    user: CurrentUser = Depends(get_current_user),
):
    """Compare captured scan segments against a reference IFC and return a deviation report."""
    if session_id not in orchestrator.sessions:
        raise HTTPException(status_code=404, detail=f"Session not found: {session_id}")

    if not orchestrator._segments.get(session_id):
        raise HTTPException(
            status_code=422,
            detail="No scan segments found for this session. Run segmentation first.",
        )

    # Use only the filename base — strip directory components against path traversal.
    raw_name = Path(ifc_file.filename or "upload.ifc").name
    if not raw_name.lower().endswith(".ifc"):
        raise HTTPException(status_code=422, detail="Uploaded file must be an IFC file (.ifc).")

    tmp_dir = Path(tempfile.mkdtemp(prefix=f"stb_ifc_{session_id[:8]}_"))
    try:
        ifc_path = tmp_dir / raw_name
        ifc_path.write_bytes(await ifc_file.read())
        report = orchestrator.generate_deviation_report(
            session_id,
            ifc_path=ifc_path,
            matched_threshold_mm=matched_threshold_mm,
            shifted_threshold_mm=shifted_threshold_mm,
        )
    except ImportError as exc:
        raise HTTPException(
            status_code=501,
            detail=f"IFC parsing requires ifcopenshell: {exc}. Install with: pip install ifcopenshell",
        ) from exc
    finally:
        shutil.rmtree(tmp_dir, ignore_errors=True)

    return report.model_dump(mode="json")


@app.get("/sessions/{session_id}/deviation-report", tags=["scan-intelligence"])
def get_deviation_report(
    session_id: str,
    user: CurrentUser = Depends(get_current_user),
):
    """Retrieve the most recent deviation report for a session."""
    if session_id not in orchestrator.sessions:
        raise HTTPException(status_code=404, detail=f"Session not found: {session_id}")
    report = orchestrator._deviation_reports.get(session_id)
    if report is None:
        raise HTTPException(
            status_code=404,
            detail="No deviation report has been generated for this session yet.",
        )
    return report.model_dump(mode="json")


def _compute_loa_statements_for_session(session_id: str):
    """Shared helper — builds LOAStatements for every segment in the session.

    Returns (statements_list, worst_tier_string, scanner_sigma_mm).
    Raises HTTPException 404/422 as appropriate.
    """
    if session_id not in orchestrator.sessions:
        raise HTTPException(status_code=404, detail=f"Session not found: {session_id}")

    segments = orchestrator._segments.get(session_id, [])
    if not segments:
        raise HTTPException(
            status_code=422,
            detail="No segments available. Run segmentation or ingest-elements first.",
        )

    from agent.tools.loa_tools import (
        DEFAULT_SCANNER_ACCURACY_MM,
        LOATier,
        build_loa_statement,
        worse_tier,
    )

    measured_sigma_mm = DEFAULT_SCANNER_ACCURACY_MM
    statements = []
    for seg in segments:
        r_sigma = seg.tags.get("loa_sigma_mm")
        if r_sigma is None:
            r_sigma = 25.0  # → LOA10 at 2σ=50mm
        statements.append(
            build_loa_statement(
                segment_id=seg.segment_id,
                represented_sigma_mm=float(r_sigma),
                measured_sigma_mm=measured_sigma_mm,
            )
        )

    tier_order = [LOATier.LOA50, LOATier.LOA40, LOATier.LOA30, LOATier.LOA20, LOATier.LOA10]
    worst = tier_order[0]
    for s in statements:
        worst = worse_tier(worst, LOATier(s.effective_tier))

    return statements, worst.value, measured_sigma_mm


@app.get("/sessions/{session_id}/loa-report", tags=["scan-intelligence"])
def get_loa_report(
    session_id: str,
    user: CurrentUser = Depends(get_current_user),
):
    """USIBD LOA v3.1 report for every segment in the session.

    Returns:
      standard_version: "USIBD LOA v3.1"
      scanner_accuracy_mm: assumed scanner+registration σ (default 5 mm)
      total_segments: count
      tier_distribution: {LOA10: n, LOA20: n, ...}  — effective tiers
      statements: list of {segment_id, measured_*, represented_*, effective_tier}
      worst_tier: the least accurate effective tier present (useful for go/no-go gates)
    """
    from agent.tools.loa_tools import summarise_tier_distribution

    statements, worst_tier, measured_sigma_mm = _compute_loa_statements_for_session(session_id)

    return {
        "session_id": session_id,
        "standard_version": "USIBD LOA v3.1",
        "scanner_accuracy_mm": measured_sigma_mm,
        "total_segments": len(statements),
        "tier_distribution": summarise_tier_distribution(statements),
        "worst_tier": worst_tier,
        "statements": [s.to_dict() for s in statements],
    }


@app.get("/sessions/{session_id}/deviation-heatmap", tags=["scan-intelligence"])
def get_deviation_heatmap(
    session_id: str,
    view: str = Query(
        default="plan",
        pattern="^(plan|elevation)$",
        description="'plan' (X-Y) or 'elevation' (Y-Z)",
    ),
    colour_by: str = Query(
        default="deviation",
        pattern="^(deviation|loa)$",
        description="'deviation' (distance mm) or 'loa' (effective tier)",
    ),
    width_px: int = Query(default=1200, gt=200, le=4000),
    height_px: int = Query(default=900, gt=200, le=4000),
    format: str = Query(
        default="svg",
        pattern="^(svg|json)$",
        description="'svg' returns image, 'json' returns summary stats",
    ),
    user: CurrentUser = Depends(get_current_user),
):
    """Render a per-element deviation/LOA heatmap for the session.

    Requires a DeviationReport to be generated first (POST /sessions/{id}/deviation-report).

    Returns SVG image or JSON summary depending on format. When colour_by='loa',
    overlays USIBD LOA tier colouring; otherwise uses deviation magnitude bands.
    """
    if session_id not in orchestrator.sessions:
        raise HTTPException(status_code=404, detail=f"Session not found: {session_id}")

    report = orchestrator._deviation_reports.get(session_id)
    if report is None:
        raise HTTPException(
            status_code=404,
            detail="No deviation report available. Generate one via POST /deviation-report first.",
        )

    from agent.tools.deviation_heatmap import (
        compute_heatmap_summary,
        render_heatmap_svg,
    )

    # Include LOA statements only when requested (cheap enough either way)
    loa_dicts = None
    if colour_by == "loa" or format == "json":
        try:
            statements, _worst, _sigma = _compute_loa_statements_for_session(session_id)
            loa_dicts = [s.to_dict() for s in statements]
        except HTTPException:
            loa_dicts = None  # no segments is fine for deviation-only colouring

    if format == "json":
        summary = compute_heatmap_summary(report, loa_dicts)
        return {"session_id": session_id, "view": view, "colour_by": colour_by, **summary}

    svg = render_heatmap_svg(
        report=report,
        loa_statements=loa_dicts,
        view=view,  # type: ignore[arg-type]
        colour_by=colour_by,  # type: ignore[arg-type]
        width_px=width_px,
        height_px=height_px,
    )
    return Response(content=svg, media_type="image/svg+xml")


@app.get("/sessions/{session_id}/floorplan", tags=["scan-intelligence"])
def export_floorplan(
    session_id: str,
    elevation: float = Query(
        default=1.0, gt=0.0, description="Slice elevation in metres above floor"
    ),
    thickness: float = Query(
        default=0.2, gt=0.0, le=2.0, description="Slice band thickness in metres"
    ),
    format: str = Query(
        default="svg", pattern="^(svg|dxf)$", description="Output format: svg or dxf"
    ),
    user: CurrentUser = Depends(get_current_user),
):
    """Export a 2D floor plan slice at the given elevation as SVG or DXF."""
    if session_id not in orchestrator.sessions:
        raise HTTPException(status_code=404, detail=f"Session not found: {session_id}")

    segments = orchestrator._segments.get(session_id, [])
    merged_pts = orchestrator._merged_clouds.get(session_id)
    if not segments and merged_pts is None:
        raise HTTPException(
            status_code=422,
            detail="No scan data available. Run segmentation or register_scans first.",
        )

    try:
        data = orchestrator.generate_floorplan(
            session_id,
            elevation_m=elevation,
            thickness_m=thickness,
            fmt=format,
        )
    except ImportError as exc:
        raise HTTPException(
            status_code=501,
            detail=f"DXF export requires ezdxf: {exc}. Install with: pip install ezdxf",
        ) from exc
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc

    short_id = session_id[:8]
    if format == "dxf":
        media_type = "application/dxf"
        filename = f"floorplan-{short_id}.dxf"
    else:
        media_type = "image/svg+xml"
        filename = f"floorplan-{short_id}.svg"

    return StreamingResponse(
        iter([data]),
        media_type=media_type,
        headers={"Content-Disposition": f'attachment; filename="{filename}"'},
    )


# ── Run ────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8765)
