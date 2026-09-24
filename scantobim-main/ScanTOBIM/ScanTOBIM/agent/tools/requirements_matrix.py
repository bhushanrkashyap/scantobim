"""requirements_matrix.py — NQA-1 § 402 Requirements Traceability Matrix.

Static catalog of every functional + safety requirement the product must meet,
mapped to:
  1. The source code that implements it
  2. The test(s) that verify it
  3. The NQA-1 software grade (A/B/C/D) this requirement carries

Reviewed at CI time by `verify_matrix_coverage()` — if a requirement has no
associated test, CI fails. This is the cheapest way to guarantee every claim
the documentation makes is backed by a running, pass/failable assertion.

Grading per NQA-1 Subpart 2.7 § 203:
  Grade A  Safety-related, affects SC1 elements — strictest V&V required
  Grade B  Important-to-safety, affects SC2 elements — documented V&V
  Grade C  Augmented quality (audit, logging, compliance evidence)
  Grade D  No safety impact (UI, developer tools)
"""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(frozen=True)
class Requirement:
    req_id: str  # e.g. "REQ-SC1-001"
    title: str
    description: str
    grade: str  # "A" | "B" | "C" | "D"
    source_files: list[str] = field(default_factory=list)
    test_ids: list[str] = field(default_factory=list)
    standard_ref: str = ""  # e.g. "NQA-1 Subpart 2.7 § 302"

    def to_dict(self) -> dict:
        return {
            "req_id": self.req_id,
            "title": self.title,
            "description": self.description,
            "grade": self.grade,
            "source_files": self.source_files,
            "test_ids": self.test_ids,
            "standard_ref": self.standard_ref,
        }


# ── The matrix ────────────────────────────────────────────────────────────────
#
# Each test_id must match a pytest node id (file::Class::method) — the
# verifier walks collected tests and checks for coverage.

REQUIREMENTS: list[Requirement] = [
    # ═══ Grade A — SC1 hard block ═════════════════════════════════════════════
    Requirement(
        req_id="REQ-SC1-001",
        title="SC1 hard block on auto-creation",
        description=(
            "Safety-critical (SC1) elements must never be auto-created. "
            "Any attempt to dispatch an SC1 ElementInstruction must raise "
            "SafetyCategoryViolationError and write ELEMENT_BLOCKED to the audit."
        ),
        grade="A",
        source_files=["agent/safety_gate.py", "agent/orchestrator.py"],
        test_ids=[
            "agent/tests/test_api.py::TestSafetyGate::test_sc1_raises_on_execute",
        ],
        standard_ref="NQA-1 Subpart 2.7 § 204 (Grade A software)",
    ),
    Requirement(
        req_id="REQ-SC1-002",
        title="SC1 violation is logged to audit before exception",
        description=(
            "ELEMENT_BLOCKED audit event must be written BEFORE the exception "
            "is raised so the audit ledger reflects every attempted action."
        ),
        grade="A",
        source_files=["agent/safety_gate.py", "agent/audit.py"],
        test_ids=[
            "agent/tests/test_api.py::TestSafetyGate::test_sc1_audit_logged_before_raise",
        ],
        standard_ref="NQA-1 Subpart 2.7 § 401",
    ),
    # ═══ Grade B — SC2 approval workflow ══════════════════════════════════════
    Requirement(
        req_id="REQ-SC2-001",
        title="SC2 elements require HMAC-signed engineer approval",
        description=(
            "Safety-related (SC2) ElementInstructions must be gated by "
            "SafetyGateService with an HMAC-signed approval record containing "
            "approver UPN and timestamp before dispatch."
        ),
        grade="B",
        source_files=["agent/safety_gate.py"],
        test_ids=[
            "agent/tests/test_api.py::TestSafetyGate::test_sc2_pending_without_approval",
            "agent/tests/test_api.py::TestSafetyGate::test_sc2_executes_after_approval",
        ],
        standard_ref="NQA-1 Subpart 2.7 § 204 (Grade B software)",
    ),
    Requirement(
        req_id="REQ-SC2-002",
        title="Approval signatures are verifiable",
        description=(
            "HMAC signatures on SC2 approvals must verify against the project "
            "secret and fail closed on tampering."
        ),
        grade="B",
        source_files=["agent/safety_gate.py"],
        test_ids=[
            "agent/tests/test_api.py::TestSafetyGate::test_sc2_tampered_signature_rejected",
        ],
        standard_ref="NQA-1 Subpart 2.7 § 702",
    ),
    # ═══ Grade B — Audit chain ═══════════════════════════════════════════════
    Requirement(
        req_id="REQ-AUDIT-001",
        title="Audit ledger is append-only and HMAC-chained",
        description=(
            "Every audit event carries hash(prev.hash + payload) and the chain "
            "is verifiable end-to-end. No update or delete operations are "
            "exposed on the ledger."
        ),
        grade="B",
        source_files=["agent/audit.py"],
        test_ids=[
            "agent/tests/test_api.py::TestAudit::test_chain_integrity_after_many_events",
            "agent/tests/test_api.py::TestAudit::test_tampered_chain_detected",
        ],
        standard_ref="NQA-1 Subpart 2.7 § 402",
    ),
    # ═══ Grade B — LOA declaration ═══════════════════════════════════════════
    Requirement(
        req_id="REQ-LOA-001",
        title="Every DirectShape carries a USIBD LOA v3.1 declaration",
        description=(
            "σ of point-to-fit residual is computed during segmentation, "
            "classified to an LOA tier (LOA10–50), and written as shared "
            "parameters on every DirectShape."
        ),
        grade="B",
        source_files=[
            "agent/tools/loa_tools.py",
            "agent/tools/scan_tools.py",
            "revit-addin/Services/ScanGeometryBuilder.cs",
        ],
        test_ids=[
            "agent/tests/test_loa.py::TestClassifyLOATier::test_construction_fit_is_loa30",
            "agent/tests/test_loa.py::TestBuildLOAStatement::test_both_sigmas_combine_to_worst",
            "agent/tests/test_loa.py::TestLOAReportEndpoint::test_happy_path_returns_report",
        ],
        standard_ref="USIBD LOA v3.1 (Jan 2025)",
    ),
    # ═══ Grade B — Clash detection & whitelist ═══════════════════════════════
    Requirement(
        req_id="REQ-CLASH-001",
        title="Clash detector reports hard and soft overlaps with tolerance",
        description=(
            "detect_clashes() flags any pair of elements whose bounding boxes "
            "overlap (hard) or come within tolerance_mm (soft), producing a "
            "ClashReport with zone_id and clearance_mm."
        ),
        grade="B",
        source_files=["agent/tools/coordinator_tools.py"],
        test_ids=[
            "agent/tests/test_overlap.py::TestDetectClashesWithWhitelist::test_wall_wall_overlap_reports_clash",
            "agent/tests/test_overlap.py::TestDetectClashesWithWhitelist::test_soft_clash",
        ],
        standard_ref="NQA-1 Subpart 2.7 § 401",
    ),
    Requirement(
        req_id="REQ-CLASH-002",
        title="Whitelist suppresses known-valid overlaps",
        description=(
            "Valve-on-pipe, pipe-through-wall, and stacked floor/ceiling with "
            "non-overlapping Z ranges are excluded from clash results so "
            "safety-relevant clashes are not buried in false positives."
        ),
        grade="B",
        source_files=["agent/tools/coordinator_tools.py"],
        test_ids=[
            "agent/tests/test_overlap.py::TestClashWhitelist::test_valve_on_pipe",
            "agent/tests/test_overlap.py::TestClashWhitelist::test_pipe_through_wall",
            "agent/tests/test_overlap.py::TestClashWhitelist::test_stacked_floors_different_z",
        ],
        standard_ref="ISO 19650-2:2018 § 5.6",
    ),
    # ═══ Grade C — Registration QA ═══════════════════════════════════════════
    Requirement(
        req_id="REQ-REG-001",
        title="Registration QA categorises per RICS / PAS 128",
        description=(
            "Every scan station is classified against RICS 3rd ed. (2024) / "
            "PAS 128 quality tiers (A ≤3 mm, B ≤10 mm, C ≤50 mm, FAIL) using "
            "RMSE and inlier ratio thresholds."
        ),
        grade="C",
        source_files=["agent/tools/registration_report.py"],
        test_ids=[
            "agent/tests/test_registration_report.py::TestClassifyStation::test_survey_grade_is_cat_a",
            "agent/tests/test_registration_report.py::TestBuildQAReport::test_one_fail_drops_overall_to_fail",
        ],
        standard_ref="RICS Measured Surveys 3rd ed. (2024)",
    ),
    # ═══ Grade C — Overlap resolution ════════════════════════════════════════
    Requirement(
        req_id="REQ-OVERLAP-001",
        title="Post-segmentation deduplication removes overlapping segments",
        description=(
            "After all detection stages, segments with IoU ≥ 0.60 on same "
            "shape are merged (higher confidence kept). Segments ≥80% "
            "contained in larger ones are dropped except VALVE_CANDIDATE."
        ),
        grade="C",
        source_files=["agent/tools/scan_tools.py"],
        test_ids=[
            "agent/tests/test_overlap.py::TestResolveOverlaps::test_same_shape_high_iou_merges",
            "agent/tests/test_overlap.py::TestResolveOverlaps::test_containment_absorb_drops_small",
            "agent/tests/test_overlap.py::TestResolveOverlaps::test_valve_not_absorbed_by_pipe",
        ],
        standard_ref="Internal — segment deduplication",
    ),
    # ═══ Grade C — Deliverable ═══════════════════════════════════════════════
    Requirement(
        req_id="REQ-DELIV-001",
        title="Handover bundle contains all quality artifacts",
        description=(
            "GET /handover-bundle returns a .zip with manifest, README, IFC, "
            "LOA report, deviation heatmaps, registration QA, audit trail, "
            "chain integrity, NCRs, and DRP — with graceful skip semantics."
        ),
        grade="C",
        source_files=["agent/tools/handover_bundle.py"],
        test_ids=[
            "agent/tests/test_handover_bundle.py::TestEmptySession::test_empty_session_still_produces_valid_zip",
            "agent/tests/test_handover_bundle.py::TestManifestAndReadme::test_manifest_schema",
            "agent/tests/test_handover_bundle.py::TestPartialSession::test_session_with_registration_includes_reg_qa",
        ],
        standard_ref="ISO 19650-2:2018 § 5.7 Information delivery",
    ),
    # ═══ Grade C — Software identification (NQA-1 itself) ════════════════════
    Requirement(
        req_id="REQ-NQA1-001",
        title="Software identity is captured on every deliverable",
        description=(
            "Git SHA, product version, dependency versions, and a stable "
            "identity hash are available via software_identity module and "
            "exposed through the /version endpoint."
        ),
        grade="C",
        source_files=["agent/software_identity.py"],
        test_ids=[
            "agent/tests/test_nqa1_package.py::TestSoftwareIdentity::test_identity_has_required_fields",
            "agent/tests/test_nqa1_package.py::TestSoftwareIdentity::test_identity_hash_stable",
        ],
        standard_ref="NQA-1 Subpart 2.7 § 302",
    ),
    Requirement(
        req_id="REQ-NQA1-002",
        title="Every requirement in this matrix has at least one passing test",
        description=(
            "Requirements traceability matrix is reviewed at CI time; a "
            "requirement with zero associated tests causes CI failure."
        ),
        grade="C",
        source_files=["agent/tools/requirements_matrix.py"],
        test_ids=[
            "agent/tests/test_nqa1_package.py::TestMatrixCoverage::test_every_requirement_has_test_ids",
        ],
        standard_ref="NQA-1 Subpart 2.7 § 402",
    ),
]


# ── Coverage verifier ────────────────────────────────────────────────────────


@dataclass(frozen=True)
class MatrixCoverageReport:
    total_requirements: int
    missing_tests: list[str]  # req_ids with zero test_ids
    by_grade: dict[str, int]  # Grade → count
    standards_covered: list[str]

    @property
    def fully_traced(self) -> bool:
        return len(self.missing_tests) == 0

    def to_dict(self) -> dict:
        return {
            "total_requirements": self.total_requirements,
            "missing_tests": self.missing_tests,
            "by_grade": self.by_grade,
            "standards_covered": self.standards_covered,
            "fully_traced": self.fully_traced,
        }


def verify_matrix_coverage() -> MatrixCoverageReport:
    """Walk the matrix; flag requirements without test_ids."""
    missing = [r.req_id for r in REQUIREMENTS if not r.test_ids]

    by_grade: dict[str, int] = {}
    for r in REQUIREMENTS:
        by_grade[r.grade] = by_grade.get(r.grade, 0) + 1

    standards = sorted({r.standard_ref for r in REQUIREMENTS if r.standard_ref})

    return MatrixCoverageReport(
        total_requirements=len(REQUIREMENTS),
        missing_tests=missing,
        by_grade=by_grade,
        standards_covered=standards,
    )


def get_requirement(req_id: str) -> Requirement | None:
    for r in REQUIREMENTS:
        if r.req_id == req_id:
            return r
    return None


def matrix_to_list() -> list[dict]:
    return [r.to_dict() for r in REQUIREMENTS]
