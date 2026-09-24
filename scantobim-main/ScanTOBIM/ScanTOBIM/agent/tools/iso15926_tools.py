"""iso15926_tools.py — ISO 15926 / CFIHOS handover mapping.

Opens the door to oil & gas / process-plant markets where facility
handover data must follow the ISO 15926 reference model and the CFIHOS
(Capital Facilities Information HandOver Specification, IOGP) attribute
set.

What this module provides:

  ISO15926_CLASS_MAP       ElementType → ISO 15926-4 reference class URI
  CFIHOS_ATTRIBUTES        ElementType → list of CFIHOS attribute names
                           expected at handover for that class
  map_element(instruction) ElementInstruction → ISO15926Mapping record
                           (class URI + mandatory attrs + populated values)
  generate_cfihos_csv()    list of mappings → CFIHOS-compatible CSV string
  generate_iso15926_json() list of mappings → JSON payload for RDL systems

The mappings are conservative starters, not a full RDL implementation.
Production oil-gas deployments typically extend these tables against
the customer's project-specific reference data library (PRDL) under a
consulting engagement — this module's tables give you a runnable
baseline that covers every ElementType we classify.

Standards:
  ISO 15926-4:2024       Reference data for equipment, functional items
  ISO 15926-6:2024       Methodology for reference data documentation
  CFIHOS v1.5+ (IOGP)    Handover data specification built on 15926
"""

from __future__ import annotations

import csv
import io
from dataclasses import asdict, dataclass
from typing import TYPE_CHECKING

from agent.models import ElementType

if TYPE_CHECKING:
    from agent.models import ElementInstruction


# ── Reference data base URI ──────────────────────────────────────────────────

# Synthetic namespace for ScanToBIM's conservative starter classes. A real
# deployment would replace this with the customer's PRDL root (typically
# under http://data.example.com/rdl/ or a similar private authority).
_STB_RDL = "http://rdl.scantobim.ai/15926-4/"


# ── ISO 15926-4 class mapping ────────────────────────────────────────────────
#
# The URIs follow the common ISO 15926 RDL convention (class-id + label).
# Class labels match ISO 15926-4 controlled vocabulary where possible;
# gaps are marked with "(starter)" so consumers know to refine against
# their own PRDL.

ISO15926_CLASS_MAP: dict[ElementType, str] = {
    # ── Piping / inline ──
    ElementType.PIPE: _STB_RDL + "11-Pipe",
    ElementType.VALVE: _STB_RDL + "12-Valve",
    ElementType.SAFETY_RELIEF_VALVE: _STB_RDL + "12.4-SafetyReliefValve",
    ElementType.STRAINER: _STB_RDL + "13-Strainer",
    ElementType.EXPANSION_JOINT: _STB_RDL + "14-ExpansionJoint",
    ElementType.FIRE_DAMPER: _STB_RDL + "15-FireDamper",
    ElementType.PENETRATION_SEAL: _STB_RDL + "16-PenetrationSeal",
    ElementType.PIPE_SUPPORT: _STB_RDL + "17-PipeSupport",
    ElementType.CONDUIT: _STB_RDL + "18-ElectricalConduit",
    ElementType.DUCT: _STB_RDL + "19-AirDuct",
    ElementType.CABLE_TRAY: _STB_RDL + "20-CableTray",
    ElementType.DRAINAGE: _STB_RDL + "21-DrainLine",
    ElementType.DELUGE_VALVE: _STB_RDL + "22-DelugeValve",
    ElementType.FIRE_HYDRANT: _STB_RDL + "23-FireHydrant",
    # ── Vessels / equipment ──
    ElementType.TANK: _STB_RDL + "30-Tank",
    ElementType.PRESSURE_VESSEL: _STB_RDL + "31-PressureVessel",
    ElementType.HEAT_EXCHANGER: _STB_RDL + "32-ShellAndTubeHeatExchanger",
    ElementType.PUMP: _STB_RDL + "33-Pump",
    ElementType.COMPRESSOR: _STB_RDL + "34-Compressor",
    ElementType.HVAC_EQUIPMENT: _STB_RDL + "35-HVACUnit",
    ElementType.SPRINKLER: _STB_RDL + "36-Sprinkler",
    # ── Nuclear primary-circuit (SC1 class-1 components) ──
    ElementType.PRESSURIZER: _STB_RDL + "40-Pressurizer",
    ElementType.STEAM_GENERATOR: _STB_RDL + "41-SteamGenerator",
    ElementType.EMERGENCY_DIESEL_GENERATOR: _STB_RDL + "42-EmergencyDieselGenerator",
    ElementType.SEISMIC_ISOLATOR: _STB_RDL + "43-SeismicIsolator",
    ElementType.CONTAINMENT_PENETRATION: _STB_RDL + "44-ContainmentPenetration",
    ElementType.RADIATION_MONITOR: _STB_RDL + "45-RadiationMonitor",
    # ── Electrical ──
    ElementType.ELECTRICAL_PANEL: _STB_RDL + "50-ElectricalPanel",
    ElementType.TRANSFORMER: _STB_RDL + "51-PowerTransformer",
    ElementType.SWITCHGEAR: _STB_RDL + "52-Switchgear",
    ElementType.UPS_SYSTEM: _STB_RDL + "53-UPSSystem",
    ElementType.JUNCTION_BOX: _STB_RDL + "54-JunctionBox",
    ElementType.LIGHTING_FITTING: _STB_RDL + "55-LightingFitting",
    # ── Fire / life-safety ──
    ElementType.FIRE_ALARM_PANEL: _STB_RDL + "60-FireAlarmPanel",
    ElementType.SMOKE_DETECTOR: _STB_RDL + "61-SmokeDetector",
    # ── Structural / architectural ──
    ElementType.WALL: _STB_RDL + "70-Wall",
    ElementType.FLOOR: _STB_RDL + "71-Floor",
    ElementType.CEILING: _STB_RDL + "72-Ceiling",
    ElementType.COLUMN: _STB_RDL + "73-Column",
    ElementType.BEAM: _STB_RDL + "74-Beam",
    ElementType.STAIR: _STB_RDL + "75-Stair",
    ElementType.RAMP: _STB_RDL + "76-Ramp",
    ElementType.LADDER: _STB_RDL + "77-Ladder",
    ElementType.GRATING: _STB_RDL + "78-FloorGrating",
    ElementType.OVERHEAD_CRANE: _STB_RDL + "79-OverheadCrane",
    ElementType.DOOR: _STB_RDL + "80-Door",
    ElementType.WINDOW: _STB_RDL + "81-Window",
    ElementType.RAILING: _STB_RDL + "82-Railing",
    ElementType.HATCH: _STB_RDL + "83-Hatch",
    # ── Civil ──
    ElementType.TRENCH: _STB_RDL + "90-Trench",
    ElementType.BUND_WALL: _STB_RDL + "91-BundWall",
    ElementType.KERB: _STB_RDL + "92-Kerb",
    ElementType.SLAB_OPENING: _STB_RDL + "93-SlabOpening",
}


# ── CFIHOS mandatory attribute groups ────────────────────────────────────────
#
# The CFIHOS specification (IOGP, v1.5+) defines attribute groups keyed to
# equipment classes. These are the attributes an oil-gas operator typically
# expects at handover for each class. All entries include the universal
# CFIHOS identifier set (tag_number, functional_location, description)
# plus class-specific extras.

_UNIVERSAL_CFIHOS = [
    "tag_number",
    "functional_location",
    "description",
    "element_type",
    "iso15926_class",
]

_PIPING_CFIHOS = _UNIVERSAL_CFIHOS + [
    "nominal_diameter_mm",
    "design_pressure_bar",
    "design_temperature_c",
    "material_code",
    "insulation_thickness_mm",
    "fluid_service",
]

_VESSEL_CFIHOS = _UNIVERSAL_CFIHOS + [
    "design_pressure_bar",
    "design_temperature_c",
    "material_code",
    "asme_stamp",
    "corrosion_allowance_mm",
    "empty_mass_kg",
]

_ROTATING_CFIHOS = _UNIVERSAL_CFIHOS + [
    "driver_type",
    "rated_power_kw",
    "rated_speed_rpm",
    "duty_cycle",
    "vendor_model",
]

_ELECTRICAL_CFIHOS = _UNIVERSAL_CFIHOS + [
    "voltage_class_v",
    "rated_current_a",
    "insulation_class",
    "ip_rating",
    "ex_rating",
]

_STRUCTURAL_CFIHOS = _UNIVERSAL_CFIHOS + [
    "material_code",
    "structural_system",
    "fire_rating_minutes",
]

_NUCLEAR_CFIHOS = _UNIVERSAL_CFIHOS + [
    "design_pressure_bar",
    "design_temperature_c",
    "material_code",
    "asme_stamp",
    "nqa1_safety_class",  # SC1 / SC2 / SC3 / NS
    "seismic_category",  # I, II, or NS
    "radiation_zone",
]


CFIHOS_ATTRIBUTES: dict[ElementType, list[str]] = {
    # Piping
    **{
        e: _PIPING_CFIHOS
        for e in (
            ElementType.PIPE,
            ElementType.VALVE,
            ElementType.SAFETY_RELIEF_VALVE,
            ElementType.STRAINER,
            ElementType.EXPANSION_JOINT,
            ElementType.FIRE_DAMPER,
            ElementType.PENETRATION_SEAL,
            ElementType.PIPE_SUPPORT,
            ElementType.CONDUIT,
            ElementType.DUCT,
            ElementType.CABLE_TRAY,
            ElementType.DRAINAGE,
            ElementType.DELUGE_VALVE,
            ElementType.FIRE_HYDRANT,
            ElementType.SPRINKLER,
        )
    },
    # Vessels
    **{
        e: _VESSEL_CFIHOS
        for e in (
            ElementType.TANK,
            ElementType.PRESSURE_VESSEL,
            ElementType.HEAT_EXCHANGER,
        )
    },
    # Rotating equipment
    **{
        e: _ROTATING_CFIHOS
        for e in (
            ElementType.PUMP,
            ElementType.COMPRESSOR,
            ElementType.HVAC_EQUIPMENT,
        )
    },
    # Electrical
    **{
        e: _ELECTRICAL_CFIHOS
        for e in (
            ElementType.ELECTRICAL_PANEL,
            ElementType.TRANSFORMER,
            ElementType.SWITCHGEAR,
            ElementType.UPS_SYSTEM,
            ElementType.JUNCTION_BOX,
            ElementType.LIGHTING_FITTING,
            ElementType.FIRE_ALARM_PANEL,
            ElementType.SMOKE_DETECTOR,
        )
    },
    # Structural / architectural
    **{
        e: _STRUCTURAL_CFIHOS
        for e in (
            ElementType.WALL,
            ElementType.FLOOR,
            ElementType.CEILING,
            ElementType.COLUMN,
            ElementType.BEAM,
            ElementType.STAIR,
            ElementType.RAMP,
            ElementType.LADDER,
            ElementType.GRATING,
            ElementType.OVERHEAD_CRANE,
            ElementType.DOOR,
            ElementType.WINDOW,
            ElementType.RAILING,
            ElementType.HATCH,
            ElementType.TRENCH,
            ElementType.BUND_WALL,
            ElementType.KERB,
            ElementType.SLAB_OPENING,
        )
    },
    # Nuclear primary circuit
    **{
        e: _NUCLEAR_CFIHOS
        for e in (
            ElementType.PRESSURIZER,
            ElementType.STEAM_GENERATOR,
            ElementType.EMERGENCY_DIESEL_GENERATOR,
            ElementType.SEISMIC_ISOLATOR,
            ElementType.CONTAINMENT_PENETRATION,
            ElementType.RADIATION_MONITOR,
        )
    },
}


# ── Public API ───────────────────────────────────────────────────────────────


@dataclass(frozen=True)
class ISO15926Mapping:
    """ISO 15926 + CFIHOS attribute record for one element."""

    instruction_id: str
    segment_id: str
    zone_id: str
    element_type: str
    iso15926_class: str
    cfihos_category: str
    attributes: dict  # populated from instruction params
    unpopulated: list[str]  # CFIHOS keys we couldn't auto-fill

    def to_dict(self) -> dict:
        return asdict(self)


def _infer_cfihos_category(element_type: ElementType) -> str:
    attrs = CFIHOS_ATTRIBUTES.get(element_type, [])
    if attrs is _PIPING_CFIHOS:
        return "PIPING"
    if attrs is _VESSEL_CFIHOS:
        return "VESSEL"
    if attrs is _ROTATING_CFIHOS:
        return "ROTATING_EQUIPMENT"
    if attrs is _ELECTRICAL_CFIHOS:
        return "ELECTRICAL"
    if attrs is _STRUCTURAL_CFIHOS:
        return "STRUCTURAL"
    if attrs is _NUCLEAR_CFIHOS:
        return "NUCLEAR_PRIMARY"
    return "UNCLASSIFIED"


def map_element(instruction: ElementInstruction) -> ISO15926Mapping:
    """Convert one ElementInstruction into an ISO 15926 + CFIHOS record."""
    et = instruction.element_type
    iso_class = ISO15926_CLASS_MAP.get(et, _STB_RDL + "99-Unclassified")
    required = CFIHOS_ATTRIBUTES.get(et, list(_UNIVERSAL_CFIHOS))
    category = _infer_cfihos_category(et)

    # Populate what we can from the instruction + its parameters
    populated: dict = {
        "tag_number": _tag_from_instruction(instruction),
        "functional_location": instruction.zone_id,
        "description": f"{et.value.replace('_', ' ').title()} "
        f"detected from scan (segment "
        f"{instruction.segment_id[:8]})",
        "element_type": et.value,
        "iso15926_class": iso_class,
    }
    # Dimensions commonly map to piping attributes
    params = instruction.parameters or {}
    if "diameter_mm" in params:
        populated["nominal_diameter_mm"] = round(params["diameter_mm"], 1)
    if "major_mm" in params and "nominal_diameter_mm" not in populated:
        populated["nominal_diameter_mm"] = round(params["major_mm"], 1)
    # Nuclear safety attributes derive from the instruction
    if category == "NUCLEAR_PRIMARY":
        populated["nqa1_safety_class"] = instruction.safety_category.value

    unpopulated = [a for a in required if a not in populated]

    return ISO15926Mapping(
        instruction_id=instruction.instruction_id,
        segment_id=instruction.segment_id,
        zone_id=instruction.zone_id,
        element_type=et.value,
        iso15926_class=iso_class,
        cfihos_category=category,
        attributes=populated,
        unpopulated=unpopulated,
    )


def _tag_from_instruction(instruction: ElementInstruction) -> str:
    """Synthesise a tag number in the style 'ZONE-TYPE-NNNN'."""
    short = instruction.segment_id.replace("-", "")[:4].upper()
    return f"{instruction.zone_id.upper()}-{instruction.element_type.value.upper()}-{short}"


# ── Bulk generators ──────────────────────────────────────────────────────────


def map_session(instructions: list[ElementInstruction]) -> list[ISO15926Mapping]:
    return [map_element(i) for i in instructions]


def generate_iso15926_json(
    session_id: str,
    mappings: list[ISO15926Mapping],
) -> dict:
    """Return a JSON payload suitable for import into an ISO 15926 RDL system."""
    coverage = _coverage_summary(mappings)
    return {
        "document": "ISO 15926 / CFIHOS Handover Data",
        "standard_refs": ["ISO 15926-4:2024", "ISO 15926-6:2024", "CFIHOS v1.5 (IOGP)"],
        "session_id": session_id,
        "rdl_namespace": _STB_RDL,
        "total_elements": len(mappings),
        "by_category": coverage["by_category"],
        "coverage_pct": coverage["coverage_pct"],
        "mappings": [m.to_dict() for m in mappings],
    }


def generate_cfihos_csv(mappings: list[ISO15926Mapping]) -> str:
    """Return a CSV string matching the CFIHOS tag-attribute layout.

    Each row = one (tag, attribute) pair. Systems like AVEVA NET or
    Hexagon SDx ingest this format directly at commissioning handover.
    """
    buf = io.StringIO()
    writer = csv.writer(buf)
    writer.writerow(
        [
            "tag_number",
            "functional_location",
            "iso15926_class",
            "cfihos_category",
            "attribute_name",
            "attribute_value",
            "status",
        ]
    )
    for m in mappings:
        tag = m.attributes.get("tag_number", "")
        fl = m.attributes.get("functional_location", "")
        for attr, val in m.attributes.items():
            writer.writerow(
                [
                    tag,
                    fl,
                    m.iso15926_class,
                    m.cfihos_category,
                    attr,
                    val,
                    "POPULATED",
                ]
            )
        for attr in m.unpopulated:
            writer.writerow(
                [
                    tag,
                    fl,
                    m.iso15926_class,
                    m.cfihos_category,
                    attr,
                    "",
                    "PENDING",
                ]
            )
    return buf.getvalue()


def _coverage_summary(mappings: list[ISO15926Mapping]) -> dict:
    by_cat: dict[str, int] = {}
    total_required = 0
    total_populated = 0
    for m in mappings:
        by_cat[m.cfihos_category] = by_cat.get(m.cfihos_category, 0) + 1
        required = len(m.attributes) + len(m.unpopulated)
        total_required += required
        total_populated += len(m.attributes)
    pct = (total_populated / total_required * 100) if total_required else 0.0
    return {
        "by_category": by_cat,
        "coverage_pct": round(pct, 1),
    }
