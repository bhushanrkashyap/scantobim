"""Canonical ScanTOBIM Taxonomy & Controlled Label Mapping.

Defines the 14 authoritative semantic labels:
  WALL, FLOOR, CEILING, COLUMN, DOOR, WINDOW, BEAM, STAIR,
  PIPE, DUCT, CABLE_TRAY, VALVE, EQUIPMENT, UNKNOWN.

Rules:
- Do not enable a label unless the model or geometric reasoning provides evidence for it.
- Model-specific labels must map through a controlled taxonomy.
- Unsupported labels map strictly to UNSUPPORTED_MODEL_LABEL.
- Never silently guess mappings.
"""

from __future__ import annotations

CANONICAL_LABELS: list[str] = [
    "WALL",
    "FLOOR",
    "CEILING",
    "COLUMN",
    "DOOR",
    "WINDOW",
    "BEAM",
    "STAIR",
    "PIPE",
    "DUCT",
    "CABLE_TRAY",
    "VALVE",
    "EQUIPMENT",
    "UNKNOWN",
]

CANONICAL_LABEL_SET = set(CANONICAL_LABELS)
UNSUPPORTED_MODEL_LABEL: str = "UNSUPPORTED_MODEL_LABEL"

# Controlled dictionary mappings for common research taxonomies
S3DIS_TAXONOMY_MAP: dict[str, str] = {
    "ceiling": "CEILING",
    "floor": "FLOOR",
    "wall": "WALL",
    "beam": "BEAM",
    "column": "COLUMN",
    "window": "WINDOW",
    "door": "DOOR",
    "table": "EQUIPMENT",
    "chair": "EQUIPMENT",
    "sofa": "EQUIPMENT",
    "bookcase": "EQUIPMENT",
    "board": "EQUIPMENT",
    "clutter": "UNKNOWN",
}

SCANNET_TAXONOMY_MAP: dict[str, str] = {
    "wall": "WALL",
    "floor": "FLOOR",
    "cabinet": "EQUIPMENT",
    "bed": "EQUIPMENT",
    "chair": "EQUIPMENT",
    "sofa": "EQUIPMENT",
    "table": "EQUIPMENT",
    "door": "DOOR",
    "window": "WINDOW",
    "bookshelf": "EQUIPMENT",
    "picture": "UNKNOWN",
    "counter": "EQUIPMENT",
    "desk": "EQUIPMENT",
    "curtain": "UNKNOWN",
    "refrigerator": "EQUIPMENT",
    "shower curtain": "UNKNOWN",
    "toilet": "EQUIPMENT",
    "sink": "EQUIPMENT",
    "bathtub": "EQUIPMENT",
    "otherfurniture": "EQUIPMENT",
}

IFC_TAXONOMY_MAP: dict[str, str] = {
    "IfcWall": "WALL",
    "IfcWallStandardCase": "WALL",
    "IfcSlab": "FLOOR",
    "IfcSlabFloor": "FLOOR",
    "IfcSlabRoof": "CEILING",
    "IfcCovering": "CEILING",
    "IfcColumn": "COLUMN",
    "IfcBeam": "BEAM",
    "IfcDoor": "DOOR",
    "IfcWindow": "WINDOW",
    "IfcStair": "STAIR",
    "IfcStairFlight": "STAIR",
    "IfcRailing": "EQUIPMENT",
    "IfcFlowSegment": "PIPE",
    "IfcPipeSegment": "PIPE",
    "IfcDuctSegment": "DUCT",
    "IfcCableCarrierSegment": "CABLE_TRAY",
    "IfcFlowController": "VALVE",
    "IfcValve": "VALVE",
    "IfcFlowFitting": "EQUIPMENT",
    "IfcDistributionElement": "EQUIPMENT",
    "IfcMechanicalEquipment": "EQUIPMENT",
    "IfcEnergyConversionDevice": "EQUIPMENT",
}


def map_to_canonical(source_label: str, source_taxonomy: str = "auto") -> str:
    """Map a model/dataset specific label into the canonical ScanTOBIM taxonomy.

    Args:
        source_label: Raw label string from model or detector.
        source_taxonomy: One of 'auto', 's3dis', 'scannet', 'ifc', 'canonical'.

    Returns:
        Canonical label from CANONICAL_LABELS or UNSUPPORTED_MODEL_LABEL.
    """
    raw = (source_label or "").strip()
    if not raw:
        return "UNKNOWN"

    # Already canonical?
    upper = raw.upper()
    if upper in CANONICAL_LABEL_SET:
        return upper

    tax = source_taxonomy.lower()
    norm = raw.lower()

    if tax == "s3dis" or (tax == "auto" and norm in S3DIS_TAXONOMY_MAP):
        return S3DIS_TAXONOMY_MAP.get(norm, UNSUPPORTED_MODEL_LABEL)

    if tax == "scannet" or (tax == "auto" and norm in SCANNET_TAXONOMY_MAP):
        return SCANNET_TAXONOMY_MAP.get(norm, UNSUPPORTED_MODEL_LABEL)

    if tax == "ifc" or (tax == "auto" and raw in IFC_TAXONOMY_MAP):
        return IFC_TAXONOMY_MAP.get(raw, UNSUPPORTED_MODEL_LABEL)

    # Check case-insensitive IFC
    for ifc_k, ifc_v in IFC_TAXONOMY_MAP.items():
        if ifc_k.lower() == norm:
            return ifc_v

    return UNSUPPORTED_MODEL_LABEL
