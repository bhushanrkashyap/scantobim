"""handover_bundle.py — P2.2d consolidated BIM handover deliverable.

Produces a single .zip containing every artifact a client needs to accept
a Scan-to-BIM deliverable:

  README.md                          human-readable manifest
  manifest.json                      machine-readable manifest
  01_ifc/model.ifc                   the classified BIM (if available)
  02_loa/loa_report.json             USIBD LOA v3.1 statement per element
  03_deviation/                      heatmaps + deviation vs design (if report exists)
      deviation_report.json
      heatmap_plan.svg
      heatmap_elevation.svg
      heatmap_summary.json
  04_registration/                   RICS / PAS 128 QA (if registration run)
      registration_qa.json
      registration_qa.html
  05_quality/                        auditability + open issues
      ncrs.json
      audit.json
      chain_integrity.json
  06_drp/design_record_package.html  ISO 19650 design record (if generable)

Every section is optional — skipped if its upstream data isn't present.
The README documents what's present vs missing so a reviewer can see at
a glance whether the bundle is complete.
"""

from __future__ import annotations

import io
import json
import zipfile
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from agent.audit import AuditLedger
    from agent.ncr import NCRService
    from agent.orchestrator import ScanToBIMOrchestrator


# ── Section result ───────────────────────────────────────────────────────────


@dataclass(frozen=True)
class SectionResult:
    name: str  # folder label, e.g. "01_ifc"
    title: str  # human label, e.g. "IFC export"
    included: bool
    reason: str = ""  # when skipped, why
    file_count: int = 0


# ── Main builder ─────────────────────────────────────────────────────────────


def build_handover_bundle(
    session_id: str,
    orchestrator: ScanToBIMOrchestrator,
    audit: AuditLedger,
    ncr_service: NCRService | None = None,
    signature_service: NQA1SignatureService | None = None,
) -> tuple[bytes, list[SectionResult]]:
    """Build the handover zip for a session.

    Returns (zip_bytes, section_results). Caller is responsible for auth and
    session existence checks — this function assumes the session is valid.
    """
    if session_id not in orchestrator.sessions:
        raise KeyError(f"Unknown session: {session_id}")

    session = orchestrator.sessions[session_id]
    sections: list[SectionResult] = []
    buffer = io.BytesIO()

    with zipfile.ZipFile(buffer, "w", zipfile.ZIP_DEFLATED) as zf:
        # 01 — IFC export
        sections.append(_add_ifc(zf, session_id, orchestrator))

        # 02 — USIBD LOA per-element
        sections.append(_add_loa(zf, session_id, orchestrator))

        # 03 — Deviation heatmap + report
        sections.append(_add_deviation(zf, session_id, orchestrator))

        # 04 — Registration QA
        sections.append(_add_registration(zf, session_id, orchestrator))

        # 05 — Quality (NCRs, audit, chain integrity)
        sections.append(_add_quality(zf, session_id, audit, ncr_service))

        # 06 — Design Record Package (DRP)
        sections.append(_add_drp(zf, session_id, orchestrator, audit, ncr_service))

        # 07 — NQA-1 V&V package (with live signatures + chain integrity)
        sections.append(
            _add_nqa1(
                zf,
                session_id,
                orchestrator,
                audit,
                ncr_service,
                signature_service,
            )
        )

        # 08 — FMEA / hazard analysis (static catalog, session-stamped)
        sections.append(_add_fmea(zf, session_id))

        # 09 — ISO 15926 / CFIHOS handover mapping (oil-gas market)
        sections.append(_add_iso15926(zf, session_id, orchestrator))

        # Manifest + README last so they can describe what's in the zip
        _add_manifest(zf, session_id, session, sections)
        _add_readme(zf, session_id, session, sections)

    return buffer.getvalue(), sections


# ── Section writers ──────────────────────────────────────────────────────────


def _add_ifc(zf, session_id, orchestrator) -> SectionResult:
    instructions = orchestrator._instructions.get(session_id, [])
    if not instructions:
        return SectionResult(
            "01_ifc", "IFC export", False, "No classified elements — run classify_and_prepare first"
        )

    results = orchestrator._results.get(session_id, [])
    session = orchestrator.sessions.get(session_id)
    site_name = session.site_name if session else session_id

    try:
        from agent.tools.ifc_export import (
            export_session_to_ifc,
            export_session_to_ifc_minimal,
        )

        try:
            ifc_bytes = export_session_to_ifc(
                session_id=session_id,
                instructions=instructions,
                results=results,
                site_name=site_name,
            )
        except ImportError:
            ifc_bytes = export_session_to_ifc_minimal(session_id, instructions)
    except Exception as exc:
        return SectionResult("01_ifc", "IFC export", False, f"Export failed: {exc}")

    zf.writestr("01_ifc/model.ifc", ifc_bytes)
    return SectionResult("01_ifc", "IFC export", True, file_count=1)


def _add_loa(zf, session_id, orchestrator) -> SectionResult:
    segments = orchestrator._segments.get(session_id, [])
    if not segments:
        return SectionResult(
            "02_loa",
            "USIBD LOA v3.1",
            False,
            "No segments — run segmentation or ingest-elements first",
        )

    from agent.tools.loa_tools import (
        DEFAULT_SCANNER_ACCURACY_MM,
        LOATier,
        build_loa_statement,
        summarise_tier_distribution,
        worse_tier,
    )

    statements = []
    for seg in segments:
        r_sigma = seg.tags.get("loa_sigma_mm") or 25.0  # LOA10 default
        statements.append(
            build_loa_statement(
                segment_id=seg.segment_id,
                represented_sigma_mm=float(r_sigma),
                measured_sigma_mm=DEFAULT_SCANNER_ACCURACY_MM,
            )
        )

    worst = LOATier.LOA50
    for s in statements:
        worst = worse_tier(worst, LOATier(s.effective_tier))

    payload = {
        "session_id": session_id,
        "standard_version": "USIBD LOA v3.1",
        "scanner_accuracy_mm": DEFAULT_SCANNER_ACCURACY_MM,
        "total_segments": len(statements),
        "tier_distribution": summarise_tier_distribution(statements),
        "worst_tier": worst.value,
        "statements": [s.to_dict() for s in statements],
    }
    zf.writestr("02_loa/loa_report.json", json.dumps(payload, indent=2))
    return SectionResult("02_loa", "USIBD LOA v3.1", True, file_count=1)


def _add_deviation(zf, session_id, orchestrator) -> SectionResult:
    report = orchestrator._deviation_reports.get(session_id)
    if report is None:
        return SectionResult(
            "03_deviation",
            "Deviation heatmap",
            False,
            "No deviation report — run /deviation-report first",
        )

    from agent.tools.deviation_heatmap import (
        compute_heatmap_summary,
        render_heatmap_svg,
    )

    # Optionally include LOA for colouring
    loa_dicts = None
    segs = orchestrator._segments.get(session_id, [])
    if segs:
        try:
            from agent.tools.loa_tools import build_loa_statement

            loa_dicts = [
                build_loa_statement(
                    segment_id=s.segment_id,
                    represented_sigma_mm=float(s.tags.get("loa_sigma_mm", 25.0)),
                ).to_dict()
                for s in segs
            ]
        except Exception:
            loa_dicts = None

    zf.writestr(
        "03_deviation/deviation_report.json",
        json.dumps(report.model_dump(mode="json"), indent=2, default=str),
    )

    zf.writestr(
        "03_deviation/heatmap_plan.svg",
        render_heatmap_svg(report, loa_dicts, view="plan", colour_by="deviation"),
    )
    zf.writestr(
        "03_deviation/heatmap_elevation.svg",
        render_heatmap_svg(report, loa_dicts, view="elevation", colour_by="deviation"),
    )

    if loa_dicts is not None:
        zf.writestr(
            "03_deviation/heatmap_plan_loa.svg",
            render_heatmap_svg(report, loa_dicts, view="plan", colour_by="loa"),
        )

    zf.writestr(
        "03_deviation/heatmap_summary.json",
        json.dumps(compute_heatmap_summary(report, loa_dicts), indent=2),
    )

    return SectionResult(
        "03_deviation", "Deviation heatmap", True, file_count=4 if loa_dicts else 3
    )


def _add_registration(zf, session_id, orchestrator) -> SectionResult:
    reg = orchestrator._registration.get(session_id)
    if reg is None:
        return SectionResult(
            "04_registration",
            "Registration QA",
            False,
            "No registration result — run /register-scans first",
        )

    from agent.tools.registration_report import (
        build_qa_report,
        render_registration_html,
    )

    qa = build_qa_report(reg)

    zf.writestr("04_registration/registration_qa.json", json.dumps(qa.to_dict(), indent=2))
    zf.writestr("04_registration/registration_qa.html", render_registration_html(qa))

    return SectionResult("04_registration", "Registration QA", True, file_count=2)


def _add_quality(zf, session_id, audit, ncr_service) -> SectionResult:
    files = 0

    # Audit trail
    try:
        events = audit.get_session_events(session_id)
        audit_payload = {
            "session_id": session_id,
            "event_count": len(events),
            "events": [e.model_dump(mode="json") for e in events],
        }
        zf.writestr("05_quality/audit.json", json.dumps(audit_payload, indent=2, default=str))
        files += 1
    except Exception as exc:
        zf.writestr("05_quality/audit.error.txt", f"Failed to export audit trail: {exc}")
        files += 1

    # Chain integrity
    try:
        chain = audit.verify_chain(session_id)
        zf.writestr("05_quality/chain_integrity.json", json.dumps(chain, indent=2, default=str))
        files += 1
    except Exception as exc:
        zf.writestr("05_quality/chain_integrity.error.txt", f"Failed to verify chain: {exc}")
        files += 1

    # NCRs
    if ncr_service is not None:
        try:
            ncrs = ncr_service.list_session(session_id)
            zf.writestr(
                "05_quality/ncrs.json",
                json.dumps(
                    [n.model_dump(mode="json") for n in ncrs],
                    indent=2,
                    default=str,
                ),
            )
            files += 1
        except Exception as exc:
            zf.writestr("05_quality/ncrs.error.txt", f"Failed to list NCRs: {exc}")
            files += 1

    return SectionResult("05_quality", "Quality / auditability", True, file_count=files)


def _add_nqa1(
    zf,
    session_id,
    orchestrator,
    audit,
    ncr_service,
    signature_service=None,
) -> SectionResult:
    try:
        from agent.tools.nqa1_package import build_nqa1_package

        html, json_doc = build_nqa1_package(
            session_id=session_id,
            orchestrator=orchestrator,
            audit=audit,
            ncr_service=ncr_service,
            signature_service=signature_service,
        )
    except Exception as exc:
        return SectionResult(
            "07_nqa1", "NQA-1 V&V package", False, f"NQA-1 package generation failed: {exc}"
        )

    files = 2
    zf.writestr("07_nqa1/nqa1_package.json", json.dumps(json_doc, indent=2, default=str))
    zf.writestr("07_nqa1/nqa1_package.html", html)

    # Signature detail files — only when the signature service is available
    if signature_service is not None:
        try:
            sigs = signature_service.list_session(session_id)
            status = signature_service.status(session_id)
            chain = signature_service.verify_chain(session_id)
        except Exception as exc:
            zf.writestr(
                "07_nqa1/signatures.error.txt",
                f"Failed to retrieve signatures: {exc}",
            )
            files += 1
        else:
            zf.writestr(
                "07_nqa1/signatures.json",
                json.dumps(
                    [s.model_dump(mode="json") for s in sigs],
                    indent=2,
                    default=str,
                ),
            )
            zf.writestr(
                "07_nqa1/signature_status.json",
                json.dumps(status, indent=2, default=str),
            )
            zf.writestr(
                "07_nqa1/signature_chain_integrity.json",
                json.dumps(chain, indent=2, default=str),
            )
            files += 3

    return SectionResult("07_nqa1", "NQA-1 V&V package", True, file_count=files)


def _add_iso15926(zf, session_id, orchestrator) -> SectionResult:
    instructions = orchestrator._instructions.get(session_id, [])
    if not instructions:
        return SectionResult(
            "09_iso15926",
            "ISO 15926 / CFIHOS handover",
            False,
            "No classified instructions — run classify_and_prepare first",
        )

    try:
        from agent.tools.iso15926_tools import (
            generate_cfihos_csv,
            generate_iso15926_json,
            map_session,
        )

        mappings = map_session(instructions)
        json_doc = generate_iso15926_json(session_id, mappings)
        csv_str = generate_cfihos_csv(mappings)
    except Exception as exc:
        return SectionResult(
            "09_iso15926",
            "ISO 15926 / CFIHOS handover",
            False,
            f"Mapping failed: {exc}",
        )

    zf.writestr("09_iso15926/iso15926_mapping.json", json.dumps(json_doc, indent=2, default=str))
    zf.writestr("09_iso15926/cfihos_attributes.csv", csv_str)
    return SectionResult(
        "09_iso15926",
        "ISO 15926 / CFIHOS handover",
        True,
        file_count=2,
    )


def _add_fmea(zf, session_id) -> SectionResult:
    try:
        from agent.tools.fmea import build_fmea_document, render_fmea_html

        doc = build_fmea_document(session_id)
        html = render_fmea_html(doc)
    except Exception as exc:
        return SectionResult(
            "08_fmea", "FMEA / hazard analysis", False, f"FMEA generation failed: {exc}"
        )

    zf.writestr("08_fmea/fmea_report.json", json.dumps(doc, indent=2, default=str))
    zf.writestr("08_fmea/fmea_report.html", html)
    return SectionResult("08_fmea", "FMEA / hazard analysis", True, file_count=2)


def _add_drp(zf, session_id, orchestrator, audit, ncr_service) -> SectionResult:
    try:
        from agent.drp import generate_drp

        session_summary = orchestrator.get_session_summary(session_id)
        events = [e.model_dump(mode="json") for e in audit.get_session_events(session_id)]
        ncrs = (
            [n.model_dump(mode="json") for n in ncr_service.list_session(session_id)]
            if ncr_service
            else []
        )
        chain = audit.verify_chain(session_id)
        html = generate_drp(
            session_id=session_id,
            session=session_summary,
            audit_events=events,
            ncr_records=ncrs,
            chain_result=chain,
        )
    except Exception as exc:
        return SectionResult(
            "06_drp", "Design Record Package", False, f"DRP generation failed: {exc}"
        )

    zf.writestr("06_drp/design_record_package.html", html)
    return SectionResult("06_drp", "Design Record Package", True, file_count=1)


# ── Manifest + README ────────────────────────────────────────────────────────


def _add_manifest(zf, session_id, session, sections) -> None:
    manifest = {
        "session_id": session_id,
        "site_name": getattr(session, "site_name", None),
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "schema_version": "1.0",
        "sections": [
            {
                "name": s.name,
                "title": s.title,
                "included": s.included,
                "file_count": s.file_count,
                "reason": s.reason if not s.included else None,
            }
            for s in sections
        ],
    }
    zf.writestr("manifest.json", json.dumps(manifest, indent=2, default=str))


def _add_readme(zf, session_id, session, sections) -> None:
    included_lines = []
    missing_lines = []
    for s in sections:
        bullet = f"- **{s.title}** (`{s.name}/`)"
        if s.included:
            included_lines.append(f"{bullet} — {s.file_count} file(s)")
        else:
            missing_lines.append(f"{bullet} — *skipped*: {s.reason}")

    site_name = getattr(session, "site_name", "(unknown site)")
    timestamp = datetime.now(timezone.utc).isoformat()

    included_block = "\n".join(included_lines) if included_lines else "_none_"
    missing_block = "\n".join(missing_lines) if missing_lines else "_none — bundle is complete_"

    readme = f"""# ScanToBIM Handover Bundle

**Site:** {site_name}
**Session:** `{session_id}`
**Generated:** {timestamp}
**Bundle schema:** 1.0

This archive contains every deliverable artifact produced for this
Scan-to-BIM session, organised by topic. Use `manifest.json` for a
machine-readable view.

## Contents — included

{included_block}

## Not present

{missing_block}

## How to review

1. Open `04_registration/registration_qa.html` to confirm the scan
   registration meets your survey accuracy requirement (RICS / PAS 128
   categories A–C).
2. Open `06_drp/design_record_package.html` for the full ISO 19650
   design record, including audit trail and safety-gate decisions.
3. Inspect `02_loa/loa_report.json` for per-element USIBD LOA v3.1
   accuracy declarations.
4. Review `03_deviation/heatmap_plan.svg` and
   `heatmap_elevation.svg` for spatial distribution of deviations
   vs the reference IFC design.
5. Import `01_ifc/model.ifc` into your authoring tool of choice to
   work with the BIM geometry.
6. Cross-reference open items in `05_quality/ncrs.json` with the
   immutable audit log in `05_quality/audit.json`; verify the hash
   chain in `05_quality/chain_integrity.json`.

## Provenance

All files in this archive are produced deterministically from the
session's stored data. The audit ledger is HMAC-chained and
tamper-evident. No file in this archive has been edited by a human
— re-running the handover export will produce an equivalent archive
(timestamps aside).

---
ScanToBIM · Handover Bundle v1.0
"""
    zf.writestr("README.md", readme)
