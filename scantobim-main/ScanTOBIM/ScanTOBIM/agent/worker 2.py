"""Celery worker — background scan processing for ScanToBIM.

Handles long-running tasks so the API returns immediately with a job_id.
Progress is pushed via the orchestrator's SSE buffer and readable at:
  GET /sessions/{session_id}/progress

Queue names:
  scan_processing  — heavy point cloud work (segmentation, alignment)
  default          — lightweight tasks (IFC export, report generation)

Usage (start worker):
  celery -A agent.worker:celery_app worker --loglevel=info --concurrency=2

Flower monitoring (optional):
  celery -A agent.worker:celery_app flower --port=5555
"""

from __future__ import annotations

import os
from pathlib import Path

import structlog
from celery import Celery
from celery.utils.log import get_task_logger

log = structlog.get_logger(__name__)
task_log = get_task_logger(__name__)

# ── Celery app setup ──────────────────────────────────────────────────────────

REDIS_URL = os.environ.get("REDIS_URL", "redis://localhost:6379/0")

celery_app = Celery(
    "scantobim",
    broker=REDIS_URL,
    backend=REDIS_URL,
    include=["agent.worker"],
)

celery_app.conf.update(
    task_serializer="json",
    accept_content=["json"],
    result_serializer="json",
    timezone="UTC",
    enable_utc=True,
    task_track_started=True,
    task_acks_late=True,  # ack after task completes (safe re-queue on crash)
    worker_prefetch_multiplier=1,  # one task at a time per worker slot
    task_routes={
        "agent.worker.process_scan_file": {"queue": "scan_processing"},
        "agent.worker.export_ifc_task": {"queue": "default"},
        "agent.worker.generate_report": {"queue": "default"},
    },
    result_expires=3600,  # results kept for 1 hour
)

# ── Shared service instances (lazy-initialised per worker process) ────────────

_services: dict = {}


def _get_services():
    """Lazy-init shared services once per worker process."""
    if _services:
        return _services
    from agent.audit import AuditLedger
    from agent.ncr import NCRService
    from agent.orchestrator import ScanToBIMOrchestrator
    from agent.safety_gate import SafetyGateService

    audit = AuditLedger()
    safety_gate = SafetyGateService(audit)
    orchestrator = ScanToBIMOrchestrator(audit=audit, safety_gate=safety_gate)
    ncr_service = NCRService(audit)

    _services.update(
        {
            "audit": audit,
            "safety_gate": safety_gate,
            "orchestrator": orchestrator,
            "ncr_service": ncr_service,
        }
    )
    return _services


# ── Tasks ─────────────────────────────────────────────────────────────────────


@celery_app.task(
    bind=True,
    name="agent.worker.process_scan_file",
    max_retries=2,
    soft_time_limit=600,  # 10-min soft limit → SoftTimeLimitExceeded
    time_limit=660,  # 11-min hard kill
)
def process_scan_file(
    self,
    session_id: str,
    file_path: str,
    zone_id: str = "zone-001",
    voxel_size_m: float = 0.02,
    max_rmse_mm: float = 5.0,
) -> dict:
    """Full pipeline on a real scan file. Runs in Celery worker process.

    Stages (each emits SSE progress events):
      1. Load + validate point cloud
      2. Downsample (voxel grid)
      3. Scan quality gate
      4. Segmentation (RANSAC planes + DBSCAN clusters)
      5. Classification + safety gate check
      6. Clash detection
      7. Execute instructions (Revit bridge / IFC)
    """
    svc = _get_services()
    orch = svc["orchestrator"]
    ncr = svc["ncr_service"]

    task_log.info(f"[{session_id}] process_scan_file starting: {file_path}")

    try:
        # ── Stage 1 & 2: Load file ──────────────────────────────────────────
        orch._push_progress(
            session_id,
            "LOAD",
            {
                "message": f"Loading scan file: {Path(file_path).name}",
                "file": file_path,
            },
        )
        segments = orch.run_segmentation(
            session_id=session_id,
            file_path=Path(file_path),
            zone_id=zone_id,
            use_synthetic=False,
            voxel_size_m=voxel_size_m,
        )
        orch._push_progress(
            session_id,
            "SEGMENTED",
            {
                "message": f"Segmentation complete: {len(segments)} segments found",
                "segment_count": len(segments),
            },
        )

        # ── Stage 3: Classify ────────────────────────────────────────────────
        instructions = orch.classify_and_prepare(session_id, segments)
        orch._push_progress(
            session_id,
            "CLASSIFIED",
            {
                "message": f"Classification complete: {len(instructions)} elements",
                "instruction_count": len(instructions),
            },
        )

        # ── Stage 4: Clash detection ─────────────────────────────────────────
        clashes = orch.run_clash_detection(session_id)
        orch._push_progress(
            session_id,
            "CLASHES",
            {
                "message": f"Clash detection: {len(clashes)} clashes found",
                "clash_count": len(clashes),
            },
        )

        # ── Stage 5: Execute ─────────────────────────────────────────────────
        result = orch.execute_instructions(session_id, ncr_service=ncr)
        orch._push_progress(
            session_id,
            "COMPLETE",
            {
                "message": "Pipeline complete",
                "result": result,
            },
        )

        task_log.info(f"[{session_id}] process_scan_file complete")
        return {
            "session_id": session_id,
            "status": "complete",
            "segments": len(segments),
            "instructions": len(instructions),
            "clashes": len(clashes),
            "result": result,
        }

    except Exception as exc:
        import traceback

        tb = traceback.format_exc()
        task_log.error(f"[{session_id}] process_scan_file failed: {exc}\n{tb}")
        try:
            # Attempt to update session state to FAILED if possible
            session = orch.sessions.get(session_id, None)
            if session is not None:
                session.state = getattr(session, "FAILED", "FAILED")
        except Exception as state_exc:
            task_log.error(f"[{session_id}] failed to update session state: {state_exc}")
        orch._push_progress(
            session_id,
            "ERROR",
            {
                "message": f"Pipeline failed: {exc}",
                "error": str(exc),
                "traceback": tb,
            },
        )
        raise self.retry(exc=exc, countdown=10) from exc


@celery_app.task(
    bind=True,
    name="agent.worker.export_ifc_task",
    max_retries=1,
    soft_time_limit=120,
    time_limit=130,
)
def export_ifc_task(self, session_id: str, site_name: str) -> dict:
    """Background IFC export — returns file path when done."""
    from agent.tools.ifc_export import export_session_to_ifc, export_session_to_ifc_minimal

    svc = _get_services()
    orch = svc["orchestrator"]

    instructions = orch._instructions.get(session_id, [])
    results = orch._results.get(session_id, [])

    try:
        ifc_bytes = export_session_to_ifc(session_id, instructions, results, site_name)
    except ImportError:
        ifc_bytes = export_session_to_ifc_minimal(session_id, instructions)

    # Write to SCAN_FILES_DIR / {session_id}.ifc — cross-platform default
    import os
    import tempfile

    default_dir = Path(tempfile.gettempdir()) / "scantobim" / "scans"
    out_dir = Path(os.environ.get("SCAN_FILES_DIR", str(default_dir)))
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{session_id}.ifc"
    out_path.write_bytes(ifc_bytes)

    task_log.info(f"[{session_id}] IFC exported: {out_path} ({len(ifc_bytes)} bytes)")
    return {
        "session_id": session_id,
        "file_path": str(out_path),
        "size_bytes": len(ifc_bytes),
    }


# ── Job state helpers (used by API) ──────────────────────────────────────────


def get_job_status(task_id: str) -> dict:
    """Query Celery result backend for a task's current status."""
    from celery.result import AsyncResult

    result = AsyncResult(task_id, app=celery_app)
    return {
        "task_id": task_id,
        "status": result.status,  # PENDING | STARTED | SUCCESS | FAILURE | RETRY
        "ready": result.ready(),
        "result": result.result if result.ready() and result.successful() else None,
        "error": str(result.result) if result.failed() else None,
    }
