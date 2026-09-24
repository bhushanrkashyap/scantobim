"""Core data models for Scan-to-BIM.

All Pydantic v2 models (requires pydantic >= 2.12.0, Python >= 3.11).

Patterns used:
  • ConfigDict           — per-model config (frozen, validate_assignment, etc.)
  • X | None             — Python 3.10+ union syntax (no typing.Optional)
  • Annotated + Field    — constrained types with metadata
  • @field_validator     — input validation with mode='before'/'after'
  • @model_validator     — cross-field validation
  • frozen=True          — immutable audit/transition records
"""

from __future__ import annotations

import hashlib
import hmac
import uuid
from datetime import datetime, timezone
from enum import Enum
from typing import Annotated
from typing import Optional, Any

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    field_validator,
    model_validator,
)

# ── Re-usable annotated types ─────────────────────────────────────────────────

UUIDStr = Annotated[str, Field(min_length=1, max_length=64)]
NonEmptyStr = Annotated[str, Field(min_length=1, max_length=1024)]
Confidence = Annotated[float, Field(ge=0.0, le=1.0)]
PointCount = Annotated[int, Field(ge=0)]


# ── Enums ─────────────────────────────────────────────────────────────────────


class SafetyCategory(str, Enum):
    """Nuclear/infrastructure safety classification per NQA-1."""

    SC1 = "SC1"  # Safety-critical  — hard block, never auto-create
    SC2 = "SC2"  # Safety-related   — requires human approval
    SC3 = "SC3"  # Safety-relevant  — flagged but auto-proceed
    NS = "NS"  # Non-safety       — auto-proceed


class DSEARZone(str, Enum):
    """DSEAR (Dangerous Substances and Explosive Atmospheres) zone."""

    ZONE_0 = "Zone0"  # Explosive atmosphere continuously present
    ZONE_1 = "Zone1"  # Explosive atmosphere likely in normal operation
    ZONE_2 = "Zone2"  # Explosive atmosphere unlikely, short duration
    NONE = "None"


class ElementType(str, Enum):
    # ── Structural ────────────────────────────────────────────────────────────
    WALL = "wall"
    UNKNOWN = "unknown"  # classifier artifact/unresolved type; filtered before instruction dispatch
    GENERIC_MODEL = "generic_model"  # fallback for unclassified/unknown elements
    FLOOR = "floor"
    CEILING = "ceiling"
    COLUMN = "column"
    BEAM = "beam"
    STAIR = "stair"
    RAMP = "ramp"
    LADDER = "ladder"  # vertical access ladder on vessels/racks
    GRATING = "grating"  # open-steel floor grating over sumps/trenches
    OVERHEAD_CRANE = "overhead_crane"  # bridge/gantry crane beam near ceiling
    CURVED_STRUCTURAL_SURFACE = "curved_structural_surface"  # cylindrical core/tower shell
    # ── Architectural ────────────────────────────────────────────────────────
    DOOR = "door"
    WINDOW = "window"
    RAILING = "railing"
    HATCH = "hatch"  # small access hatch in slab/floor
    # ── Civil ────────────────────────────────────────────────────────────────
    TRENCH = "trench"  # below-grade cable/pipe channel (elongated floor void)
    BUND_WALL = "bund_wall"  # low containment bund around hazardous tanks
    KERB = "kerb"
    SLAB_OPENING = "slab_opening"
    # ── MEP — Piping / Inline ────────────────────────────────────────────────
    PIPE = "pipe"
    CONDUIT = "conduit"
    DUCT = "duct"
    CABLE_TRAY = "cable_tray"
    WIRE = "wire"
    VALVE = "valve"
    SAFETY_RELIEF_VALVE = "safety_relief_valve"  # SRV — SC1 primary circuit
    STRAINER = "strainer"  # inline Y/basket strainer on pipe
    EXPANSION_JOINT = "expansion_joint"  # flexible bellows on pipe
    FIRE_DAMPER = "fire_damper"  # inline damper in duct
    PENETRATION_SEAL = "penetration_seal"  # fire-stop at wall/slab penetration
    PIPE_SUPPORT = "pipe_support"  # bracket/hanger supporting pipe
    # ── MEP — Vessels / Equipment ────────────────────────────────────────────
    TANK = "tank"
    PRESSURE_VESSEL = "pressure_vessel"  # ASME code-stamped vessel
    HEAT_EXCHANGER = "heat_exchanger"  # shell-and-tube HX
    PUMP = "pump"
    COMPRESSOR = "compressor"  # gas compressor unit
    MECHANICAL_EQUIPMENT_RECEIVER = "mechanical_equipment_receiver"  # air/gas receiver tank
    HVAC_EQUIPMENT = "hvac_equipment"
    SPRINKLER = "sprinkler"
    DRAINAGE = "drainage"
    FIRE_HYDRANT = "fire_hydrant"  # standpipe hose connection
    DELUGE_VALVE = "deluge_valve"  # large valve on fire main
    FIRE_EXTINGUISHER = "fire_extinguisher"  # portable wall-mounted extinguisher
    # ── Nuclear / Safety-Critical Vessels ────────────────────────────────────
    PRESSURIZER = "pressurizer"  # PWR primary circuit — SC1
    STEAM_GENERATOR = "steam_generator"  # PWR heat transfer vessel — SC1
    EMERGENCY_DIESEL_GENERATOR = "emergency_diesel_generator"  # EDG — SC1
    SEISMIC_ISOLATOR = "seismic_isolator"  # vibration isolation pad — SC2
    CONTAINMENT_PENETRATION = "containment_penetration"  # reactor containment sleeve — SC1
    RADIATION_MONITOR = "radiation_monitor"  # area ionisation chamber — SC2
    # ── Electrical ───────────────────────────────────────────────────────────
    ELECTRICAL_PANEL = "electrical_panel"
    TRANSFORMER = "transformer"  # step-up/step-down power transformer
    SWITCHGEAR = "switchgear"  # MCC / switchboard row
    UPS_SYSTEM = "ups_system"  # uninterruptible power / battery rack
    JUNCTION_BOX = "junction_box"  # small electrical enclosure
    LIGHTING_FITTING = "lighting_fitting"  # luminaire
    # ── Fire / Life Safety ───────────────────────────────────────────────────
    FIRE_ALARM_PANEL = "fire_alarm_panel"  # wall-mounted FA panel
    SMOKE_DETECTOR = "smoke_detector"  # ceiling-mounted ionisation detector


class Discipline(str, Enum):
    STRUCTURAL = "structural"
    ARCHITECTURAL = "architectural"
    MEP = "mep"
    FIRE_PROTECTION = "fire_protection"
    CIVIL = "civil"
    INFRASTRUCTURE = "infrastructure"


# class Plane(str , Enum):
#     plane_id : int
#     a : int
#     b : int
#     c : int
#     d : int


#     normal_x : float
#     normal_y : float
#     normal_z : float

#     centroid_x : float
#     centroid_y : float
#     centroid_z : float

#     slope_deg  : float
#     area_m2 : float
#     inlier_degree : float
#     classification : Optional[str] = None
#     storey_id : Optional[str] = None
#     step_count : Optional[int]= None
#     point_incides : Optional[int] = None


class SegmentShape(str, Enum):
    PLANE_VERTICAL = "plane_vertical"
    PLANE_HORIZONTAL = "plane_horizontal"
    PLANE_SLOPED = "plane_sloped"  # 0.3 ≤ |normal_z| < 0.85 — stair/ramp
    CYLINDER = "cylinder"  # elongated round cluster — pipe
    BOX = "box"  # elongated rectangular cluster — beam/railing
    VOID = "void"  # rectangular absence in a wall plane — door/window
    VALVE_CANDIDATE = "valve_candidate"  # compact cluster on a pipe axis — valve/fitting
    CURVED_STRUCTURAL_SURFACE = "curved_structural_surface"  # group of chord facets on curved tower/core
    UNKNOWN = "unknown"


class WallClassification(str, Enum):
    ARCHITECTURAL_WALL = "ARCHITECTURAL_WALL"
    WALL_FRAGMENT = "WALL_FRAGMENT"
    CURVED_STRUCTURAL_SURFACE = "CURVED_STRUCTURAL_SURFACE"
    MEP_EQUIPMENT = "MEP_EQUIPMENT"
    STRUCTURAL_NONWALL = "STRUCTURAL_NONWALL"
    FURNITURE_OBJECT = "FURNITURE_OBJECT"
    OTHER = "OTHER"


class AuditEventType(str, Enum):
    SEGMENT_CLASSIFIED = "segment_classified"
    SAFETY_GATE_RAISED = "safety_gate_raised"
    SAFETY_GATE_APPROVED = "safety_gate_approved"
    SAFETY_GATE_REJECTED = "safety_gate_rejected"
    ELEMENT_CREATED = "element_created"
    ELEMENT_BLOCKED = "element_blocked"
    CLASH_DETECTED = "clash_detected"
    NCR_RAISED = "ncr_raised"
    NCR_RESOLVED = "ncr_resolved"
    SESSION_STARTED = "session_started"
    SESSION_COMPLETED = "session_completed"
    # Coordinate alignment + scan quality
    COORDINATE_ALIGNED = "coordinate_aligned"
    SCAN_QUALITY_CHECKED = "scan_quality_checked"
    # IFC export
    IFC_EXPORTED = "ifc_exported"
    # P5 — CDE + compliance
    CDE_STATE_CHANGED = "cde_state_changed"
    DESIGN_RECORD_ISSUED = "design_record_issued"
    CHAIN_VERIFICATION_PASSED = "chain_verification_passed"
    CHAIN_VERIFICATION_FAILED = "chain_verification_failed"
    # Sprint 3.5 — Scan Intelligence
    SCANS_REGISTERED = "scans_registered"
    DEVIATION_REPORT_GENERATED = "deviation_report_generated"
    FLOORPLAN_EXPORTED = "floorplan_exported"


class SessionState(str, Enum):
    PENDING = "pending"
    RUNNING = "running"
    PAUSED = "paused"
    COMPLETED = "completed"
    FAILED = "failed"


class SafetyGateStatus(str, Enum):
    PENDING = "pending"
    APPROVED = "approved"
    REJECTED = "rejected"


# ── P5 enums ───────────────────────────────────────────────────────────────────


class CDEState(str, Enum):
    """ISO 19650 Common Data Environment states."""

    WIP = "WIP"  # Work In Progress — agent creating elements
    SHARED = "SHARED"  # Shared — all elements created, ready for review
    PUBLISHED = "PUBLISHED"  # Published — approved, DRP generated, immutable


class NCRSeverity(str, Enum):
    CRITICAL = "CRITICAL"  # Immediate stop-work
    MAJOR = "MAJOR"  # Must resolve before publish
    MINOR = "MINOR"  # May resolve post-publish with concession


class NCRStatus(str, Enum):
    OPEN = "OPEN"
    RESOLVED = "RESOLVED"
    CLOSED = "CLOSED"


class NQA1Role(str, Enum):
    """Three-party signoff roles for NQA-1 Subpart 2.7 V&V packages.

    Required order: PREPARER → VERIFIER → APPROVER. Same UPN may not
    hold two roles on the same session (separation of duties).
    """

    PREPARER = "preparer"  # author of the V&V evidence (often the agent itself)
    VERIFIER = "verifier"  # independent QA check
    APPROVER = "approver"  # QA manager final sign-off


# ── Geometry Models ────────────────────────────────────────────────────────────


class BoundingBox(BaseModel):
    """Axis-aligned bounding box — all values in millimetres."""

    model_config = ConfigDict(validate_assignment=True)

    min_x: float = Field(description="Min X in mm")
    min_y: float = Field(description="Min Y in mm")
    min_z: float = Field(description="Min Z in mm")
    max_x: float = Field(description="Max X in mm")
    max_y: float = Field(description="Max Y in mm")
    max_z: float = Field(description="Max Z in mm")

    @model_validator(mode="after")
    def max_gte_min(self) -> BoundingBox:
        if self.max_x < self.min_x:
            raise ValueError(f"max_x ({self.max_x}) must be >= min_x ({self.min_x})")
        if self.max_y < self.min_y:
            raise ValueError(f"max_y ({self.max_y}) must be >= min_y ({self.min_y})")
        if self.max_z < self.min_z:
            raise ValueError(f"max_z ({self.max_z}) must be >= min_z ({self.min_z})")
        return self


class Point3D(BaseModel):
    """3-D coordinate in millimetres."""

    model_config = ConfigDict(validate_assignment=True)

    x: float
    y: float
    z: float


class GeometrySegment(BaseModel):
    """A segment extracted from point cloud data by the segmentation pipeline."""

    model_config = ConfigDict(validate_assignment=True, str_strip_whitespace=True)

    segment_id: UUIDStr = Field(default_factory=lambda: str(uuid.uuid4()))
    zone_id: NonEmptyStr = Field(description="Zone this segment belongs to")
    shape: SegmentShape
    normal: Point3D | None = Field(None, description="Surface normal for planes")
    centroid: Point3D = Field(description="Centre point of the segment in mm")
    bounding_box: BoundingBox
    point_count: PointCount
    confidence: Confidence = Field(description="Detection confidence 0–1")
    scan_station_id: str | None = None
    source_file: str | None = None
    tags: dict = Field(default_factory=dict)  # metadata e.g. {"stair_steps": 4}

    # ── Sprint 4: Enhanced Semantic & Color metadata ─────────────────────────
    element_type: ElementType | None = Field(None, description="Assigned BIM category")
    semantic_confidence: Confidence | None = Field(
        None, description="Confidence in category assignment"
    )
    dominant_color: str | None = Field(None, description="Dominant hex color (e.g. #FF0000)")
    color_variance: float | None = Field(None, description="Color homogeneity within segment")
    classification_reason: str | None = Field(None, description="Rationale for the classification")
    safety_category: SafetyCategory | None = Field(
        None, description="Safety category: SC1 / SC2 / SC3 / NS"
    )


# ── Classification & Safety ────────────────────────────────────────────────────


class SafetyClassification(BaseModel):
    """Safety classification result for a geometry segment."""

    model_config = ConfigDict(validate_assignment=True, str_strip_whitespace=True)

    segment_id: UUIDStr
    safety_category: SafetyCategory
    dsear_zone: DSEARZone = DSEARZone.NONE
    classification_reason: NonEmptyStr = Field(description="Why this category was assigned")
    confidence: Confidence
    classified_by: NonEmptyStr = Field(description="'rule_engine' or 'gpt-4o'")


class ElementInstruction(BaseModel):
    """Instruction to create a BIM element in Revit, derived from a classified segment."""

    model_config = ConfigDict(validate_assignment=True, str_strip_whitespace=True)

    instruction_id: UUIDStr = Field(default_factory=lambda: str(uuid.uuid4()))
    segment_id: UUIDStr
    zone_id: NonEmptyStr
    element_type: ElementType
    discipline: Discipline
    safety_category: SafetyCategory
    dsear_zone: DSEARZone = DSEARZone.NONE
    bounding_box: BoundingBox
    centroid: Point3D
    parameters: dict = Field(default_factory=dict)
    # Revit family resolution hints — optional operator overrides.
    # When set, the Revit bridge skips the family_map config and places
    # this exact family + type.  If the family is not loaded in the project
    # the bridge falls back to the config lookup, then category first-match.
    revit_family_hint: str | None = Field(None, description="Exact Revit family name override")
    revit_type_hint: str | None = Field(None, description="Exact Revit type name override")
    # SC2 approval — populated after gate decision
    approval_signature: str | None = Field(None, description="HMAC signature — required for SC2")
    approver_upn: str | None = None
    approved_at_utc: datetime | None = None

    @model_validator(mode="after")
    def sc2_requires_approval(self) -> ElementInstruction:
        # SC2 starts without approval — gets filled after gate approval
        # Validation is enforced by SafetyGateService, not by the model
        return self


# ── Safety Gate ────────────────────────────────────────────────────────────────


class SafetyGateRequest(BaseModel):
    """A raised safety gate — SC2 requires engineer approval before element creation."""

    model_config = ConfigDict(validate_assignment=True)

    gate_id: UUIDStr = Field(default_factory=lambda: str(uuid.uuid4()))
    segment_id: UUIDStr
    element_type: ElementType
    safety_category: SafetyCategory
    classification_reason: str
    session_id: UUIDStr
    zone_id: str
    status: SafetyGateStatus = SafetyGateStatus.PENDING
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))


class SafetyGateDecision(BaseModel):
    """Engineer decision on an SC2 safety gate."""

    model_config = ConfigDict(validate_assignment=True, str_strip_whitespace=True)

    gate_id: UUIDStr
    approved: bool
    approver_upn: NonEmptyStr
    comments: str | None = None
    decided_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))

    @field_validator("approver_upn", mode="after")
    @classmethod
    def upn_must_contain_at(cls, v: str) -> str:
        if "@" not in v:
            raise ValueError(f"approver_upn must be a valid UPN (email format), got {v!r}")
        return v.lower().strip()


# ── Audit ──────────────────────────────────────────────────────────────────────


class AuditEvent(BaseModel):
    """
    Immutable audit event — frozen so no field can be mutated after creation.
    HMAC-SHA256 chain provides tamper detection.
    """

    model_config = ConfigDict(frozen=True)

    event_id: UUIDStr = Field(default_factory=lambda: str(uuid.uuid4()))
    session_id: UUIDStr
    event_type: AuditEventType
    timestamp_utc: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    zone_id: str | None = None
    segment_id: str | None = None
    element_id: str | None = None
    actor: NonEmptyStr = Field(description="Agent name or user UPN")
    detail: dict = Field(default_factory=dict)
    previous_hash: str | None = Field(None, description="Hash of prior event in chain")
    event_hash: str | None = Field(None, description="HMAC-SHA256 of this event")

    def compute_hash(self, secret: str) -> str:
        payload = (
            f"{self.event_id}|{self.session_id}|{self.event_type.value}"
            f"|{self.timestamp_utc.isoformat()}|{self.previous_hash or ''}"
        )
        return hmac.new(secret.encode(), payload.encode(), hashlib.sha256).hexdigest()


# ── Session ────────────────────────────────────────────────────────────────────


class SiteSession(BaseModel):
    """Tracks the lifecycle of one scan-to-BIM agent session."""

    model_config = ConfigDict(validate_assignment=True, str_strip_whitespace=True)

    session_id: UUIDStr = Field(default_factory=lambda: str(uuid.uuid4()))
    site_name: NonEmptyStr
    state: SessionState = SessionState.PENDING
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    zone_ids: list[str] = Field(default_factory=list)
    total_segments: int = Field(default=0, ge=0)
    total_elements_created: int = Field(default=0, ge=0)
    total_clashes: int = Field(default=0, ge=0)
    total_ncrs: int = Field(default=0, ge=0)


# ── Clash & Validation ─────────────────────────────────────────────────────────


class ClashReport(BaseModel):
    """Spatial clash between two BIM elements."""

    model_config = ConfigDict(frozen=True)

    clash_id: UUIDStr = Field(default_factory=lambda: str(uuid.uuid4()))
    element_a_id: str
    element_b_id: str
    clash_type: str = Field(description="'hard' or 'soft'")
    clearance_mm: float
    zone_id: str

    @field_validator("clash_type", mode="after")
    @classmethod
    def valid_clash_type(cls, v: str) -> str:
        if v not in {"hard", "soft"}:
            raise ValueError(f"clash_type must be 'hard' or 'soft', got {v!r}")
        return v


class ValidationResult(BaseModel):
    """Deviation check result — compares element dimensions against source scan segment."""

    model_config = ConfigDict(frozen=True)

    element_id: str
    segment_id: str
    deviation_mm: float = Field(description="Max face deviation from scan points in mm")
    passed: bool
    tolerance_mm: float = Field(default=10.0, gt=0)


class LOAStatementModel(BaseModel):
    """USIBD Level of Accuracy v3.1 declaration for a single BIM element.

    Issued alongside every DirectShape so downstream consumers know how much
    the geometry can be trusted (fabrication? design? schematic?).
    """

    model_config = ConfigDict(frozen=True)

    segment_id: str
    measured_sigma_mm: float = Field(ge=0, description="σ of scan + registration (mm)")
    measured_tier: str = Field(description="LOA10–LOA50")
    represented_sigma_mm: float = Field(ge=0, description="σ of BIM element vs scan points (mm)")
    represented_tier: str = Field(description="LOA10–LOA50")
    effective_tier: str = Field(description="Worse of measured/represented")
    standard_version: str = Field(default="USIBD LOA v3.1")

    @field_validator("measured_tier", "represented_tier", "effective_tier", mode="after")
    @classmethod
    def valid_tier(cls, v: str) -> str:
        valid = {"LOA10", "LOA20", "LOA30", "LOA40", "LOA50"}
        if v not in valid:
            raise ValueError(f"LOA tier must be one of {sorted(valid)}, got {v!r}")
        return v


class ActionResult(BaseModel):
    """Result of a single Revit element creation attempt."""

    model_config = ConfigDict(validate_assignment=True)

    success: bool
    element_id: str | None = None
    instruction_id: str
    error: str | None = None
    duration_ms: float | None = None


# ── P5 models ──────────────────────────────────────────────────────────────────


class NCRRecord(BaseModel):
    """Non-Conformance Report — raised when an element fails validation or compliance."""

    model_config = ConfigDict(validate_assignment=True, str_strip_whitespace=True)

    ncr_id: UUIDStr = Field(default_factory=lambda: str(uuid.uuid4()))
    session_id: UUIDStr
    element_id: str
    description: NonEmptyStr
    severity: NCRSeverity
    status: NCRStatus = NCRStatus.OPEN
    raised_by: NonEmptyStr
    raised_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    resolved_by: str | None = None
    resolved_at: datetime | None = None
    resolution: str | None = None


# ── P2.4++: NQA-1 Subpart 2.7 signature workflow ──────────────────────────────


class NQA1Signature(BaseModel):
    """A PREPARER / VERIFIER / APPROVER signature over an NQA-1 V&V package.

    The `package_hash` is the SHA-256 of the NQA-1 JSON package at the moment
    of signing. If anyone edits the package later, the signature's hash
    validation breaks — tamper-evident.

    Signatures are HMAC-chained (each signature's `previous_signature_hash`
    points at the prior sig's hash) so role ordering is cryptographically
    enforced. Verifier cannot be valid if Preparer is revoked.
    """

    model_config = ConfigDict(frozen=True)

    signature_id: UUIDStr = Field(default_factory=lambda: str(uuid.uuid4()))
    session_id: UUIDStr
    role: NQA1Role
    signer_upn: NonEmptyStr = Field(description="Signer UPN (user@org)")
    signer_name: NonEmptyStr
    timestamp_utc: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    package_hash: str = Field(description="SHA-256 of the NQA-1 package JSON at signing time")
    previous_signature_hash: str | None = None
    signature_hash: str | None = Field(None, description="HMAC-SHA256 of this signature")
    comments: str = ""
    # Revocation (append-only — a revoked signature stays in the ledger)
    revoked_at_utc: datetime | None = None
    revoked_by: str | None = None
    revocation_reason: str | None = None

    @field_validator("signer_upn", mode="after")
    @classmethod
    def upn_format(cls, v: str) -> str:
        if "@" not in v:
            raise ValueError(f"signer_upn must be UPN format (user@org), got {v!r}")
        return v.lower().strip()

    def compute_hash(self, secret: str) -> str:
        """HMAC-SHA256 over canonical signature payload.

        Includes every human-readable field so any DB-level tamper
        (swapping the signer_name, editing comments, etc.) is detectable
        by verify_chain().
        """
        payload = (
            f"{self.signature_id}|{self.session_id}|{self.role.value}|"
            f"{self.signer_upn}|{self.signer_name}|"
            f"{self.timestamp_utc.isoformat()}|"
            f"{self.package_hash}|{self.previous_signature_hash or ''}|"
            f"{self.comments}"
        )
        return hmac.new(secret.encode(), payload.encode(), hashlib.sha256).hexdigest()


class CDETransition(BaseModel):
    """Immutable record of a CDE state transition."""

    model_config = ConfigDict(frozen=True)

    transition_id: UUIDStr = Field(default_factory=lambda: str(uuid.uuid4()))
    session_id: UUIDStr
    from_state: CDEState
    to_state: CDEState
    triggered_by: str
    triggered_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    notes: str | None = None


class CDESessionState(BaseModel):
    """Tracks the current ISO 19650 CDE state of a session."""

    model_config = ConfigDict(validate_assignment=True)

    session_id: UUIDStr
    current_state: CDEState = CDEState.WIP
    transitions: list[CDETransition] = Field(default_factory=list)
    updated_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))


# ── Survey / Coordinate alignment ─────────────────────────────────────────────


class SurveyControlPoint(BaseModel):
    """A pair of corresponding coordinates used to align scan to world/project CRS.

    Both scan and world coordinates are in millimetres.
    At least 3 non-collinear pairs are required for a valid rigid transform.
    """

    model_config = ConfigDict(validate_assignment=True)

    point_id: str | None = None
    scan_x: float
    scan_y: float
    scan_z: float
    world_x: float
    world_y: float
    world_z: float


class ScanQualityResult(BaseModel):
    """Result of the pre-segmentation scan quality gate."""

    model_config = ConfigDict(frozen=True)

    passed: bool
    point_count: int
    area_m2: float
    density_pts_per_m2: float
    min_required_density: float
    noise_ratio: float
    warnings: list[str] = Field(default_factory=list)
    errors: list[str] = Field(default_factory=list)


# ── Sprint 3.5: Multi-scan ICP Registration ───────────────────────────────────


class StationRegistrationDetail(BaseModel):
    """ICP registration result for a single scan station."""

    model_config = ConfigDict(frozen=True)

    station_index: int
    source_file: str
    transform_4x4: list[list[float]]  # 4×4 row-major transform matrix
    rmse_mm: float
    inlier_ratio: float


class StationRegistrationResult(BaseModel):
    """Full multi-scan ICP registration result stored on the session."""

    model_config = ConfigDict(frozen=True)

    session_id: UUIDStr
    station_count: int
    stations: list[StationRegistrationDetail]
    global_rmse_mm: float
    merged_point_count: int
    registered_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))


# ── P2.2c: Registration QA Report (RICS / PAS 128 tiers) ─────────────────────


class StationQAModel(BaseModel):
    """Per-station RICS-categorised QA result."""

    model_config = ConfigDict(frozen=True)

    station_index: int
    source_file: str
    rmse_mm: float = Field(ge=0)
    inlier_ratio: float = Field(ge=0, le=1)
    category: str = Field(description="A / B / C / FAIL")
    pass_fail: str = Field(description="pass | fail")

    @field_validator("category", mode="after")
    @classmethod
    def valid_cat(cls, v: str) -> str:
        if v not in {"A", "B", "C", "FAIL"}:
            raise ValueError(f"category must be A|B|C|FAIL, got {v!r}")
        return v


class RegistrationQAReportModel(BaseModel):
    """RICS/PAS 128 Registration QA report for a session."""

    model_config = ConfigDict(frozen=True)

    session_id: UUIDStr
    generated_at_utc: str
    station_count: int
    stations: list[StationQAModel]
    global_rmse_mm: float = Field(ge=0)
    overall_category: str = Field(description="A / B / C / FAIL")
    overall_pass_fail: str
    merged_point_count: int = Field(ge=0)
    standard_version: str = "RICS Measured Surveys 3rd ed. (2024)"

    @field_validator("overall_category", mode="after")
    @classmethod
    def valid_overall_cat(cls, v: str) -> str:
        if v not in {"A", "B", "C", "FAIL"}:
            raise ValueError(f"overall_category must be A|B|C|FAIL, got {v!r}")
        return v

    @field_validator("overall_pass_fail", mode="after")
    @classmethod
    def valid_pass_fail(cls, v: str) -> str:
        if v not in {"pass", "fail"}:
            raise ValueError(f"overall_pass_fail must be 'pass' or 'fail', got {v!r}")
        return v


# ── Sprint 3.5: Scan vs. Design Deviation ─────────────────────────────────────


class DeviationStatus(str, Enum):
    MATCHED = "MATCHED"  # < 25 mm — within tolerance
    SHIFTED = "SHIFTED"  # 25–150 mm — possible rework
    MISSING = "MISSING"  # no scan segment within 150 mm of IFC element
    EXTRA = "EXTRA"  # scan segment with no corresponding IFC element


class ElementDeviationRecord(BaseModel):
    """Per-element comparison result between IFC design and captured scan."""

    model_config = ConfigDict(frozen=True)

    ifc_global_id: str | None  # None for EXTRA records
    ifc_class: str | None
    ifc_name: str | None
    scan_segment_id: str | None  # None for MISSING records
    distance_mm: float | None  # None when no match found
    status: DeviationStatus
    ifc_centroid_mm: list[float] | None  # [x, y, z]
    scan_centroid_mm: list[float] | None


class DeviationReport(BaseModel):
    """Full scan-vs-design deviation report for a session."""

    model_config = ConfigDict(validate_assignment=True)

    report_id: UUIDStr = Field(default_factory=lambda: str(uuid.uuid4()))
    session_id: UUIDStr
    generated_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    ifc_element_count: int
    scan_segment_count: int
    matched_count: int
    shifted_count: int
    missing_count: int
    extra_count: int
    matched_threshold_mm: float = 25.0
    shifted_threshold_mm: float = 150.0
    records: list[ElementDeviationRecord]
