"""Model Label Space Definition and Validation — Phase 3A CPU Production.

Defines the authoritative label vocabulary for each registered model and
generates MODEL_LABEL_SPACE.json with per-class mapping status.

Rule:
  A model MUST NOT claim to support a semantic class it was not trained on.
  Unsupported classes → "UNSUPPORTED" in mapping_status.
  The label space is authoritative over the model's stated capabilities.
"""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any

import structlog

logger = structlog.get_logger()

# ── Canonical ScanTOBIM classes ───────────────────────────────────────────────

CANONICAL_CLASSES: list[str] = [
    "WALL", "FLOOR", "CEILING", "COLUMN", "BEAM",
    "DOOR", "WINDOW", "STAIR",
    "PIPE", "DUCT", "CABLE_TRAY", "VALVE", "EQUIPMENT",
    "UNKNOWN",
]


# ── Per-model label space definitions ────────────────────────────────────────

# RandLA-Net trained on SYNTHETIC_ARCHITECTURAL_v1
# Binary classifier: class 0=NON_WALL, class 1=WALL
RANDLANET_ARCHITECTURAL_LABEL_SPACE: list[dict[str, Any]] = [
    {
        "class_id": 0,
        "class_name": "NON_WALL",
        "scan_to_bim_class": None,  # Aggregated; post-processed geometrically
        "mapping_status": "AMBIGUOUS",
        "notes": (
            "Aggregated non-wall class. Sub-classification (FLOOR/CEILING/COLUMN/EQUIPMENT) "
            "requires geometric post-processing. Model does not distinguish sub-classes."
        ),
    },
    {
        "class_id": 1,
        "class_name": "WALL",
        "scan_to_bim_class": "WALL",
        "mapping_status": "SUPPORTED",
        "notes": "Trained on synthetic vertical planar surfaces.",
    },
]

# Unsupported ScanTOBIM classes for RandLA-Net architectural
RANDLANET_UNSUPPORTED: list[str] = [
    "FLOOR",      # Returned as NON_WALL — geometrically derived
    "CEILING",    # Returned as NON_WALL — geometrically derived
    "COLUMN",     # Returned as NON_WALL — geometrically derived
    "BEAM",       # Not in training data
    "DOOR",       # Not in training data
    "WINDOW",     # Not in training data
    "STAIR",      # Not in training data
    "PIPE",       # Not in training data (MEP)
    "DUCT",       # Not in training data (MEP)
    "CABLE_TRAY", # Not in training data (MEP)
    "VALVE",      # Not in training data (MEP)
    "EQUIPMENT",  # Not in training data
]

# PTv2 — S3DIS trained (not available without checkpoint)
PTV2_S3DIS_LABEL_SPACE: list[dict[str, Any]] = [
    {"class_id": 0,  "class_name": "ceiling",  "scan_to_bim_class": "CEILING",   "mapping_status": "SUPPORTED"},
    {"class_id": 1,  "class_name": "floor",    "scan_to_bim_class": "FLOOR",     "mapping_status": "SUPPORTED"},
    {"class_id": 2,  "class_name": "wall",     "scan_to_bim_class": "WALL",      "mapping_status": "SUPPORTED"},
    {"class_id": 3,  "class_name": "beam",     "scan_to_bim_class": "BEAM",      "mapping_status": "SUPPORTED"},
    {"class_id": 4,  "class_name": "column",   "scan_to_bim_class": "COLUMN",    "mapping_status": "SUPPORTED"},
    {"class_id": 5,  "class_name": "window",   "scan_to_bim_class": "WINDOW",    "mapping_status": "SUPPORTED"},
    {"class_id": 6,  "class_name": "door",     "scan_to_bim_class": "DOOR",      "mapping_status": "SUPPORTED"},
    {"class_id": 7,  "class_name": "table",    "scan_to_bim_class": "EQUIPMENT", "mapping_status": "SUPPORTED"},
    {"class_id": 8,  "class_name": "chair",    "scan_to_bim_class": "EQUIPMENT", "mapping_status": "SUPPORTED"},
    {"class_id": 9,  "class_name": "sofa",     "scan_to_bim_class": "EQUIPMENT", "mapping_status": "SUPPORTED"},
    {"class_id": 10, "class_name": "bookcase", "scan_to_bim_class": "EQUIPMENT", "mapping_status": "SUPPORTED"},
    {"class_id": 11, "class_name": "board",    "scan_to_bim_class": "EQUIPMENT", "mapping_status": "SUPPORTED"},
    {"class_id": 12, "class_name": "clutter",  "scan_to_bim_class": "UNKNOWN",   "mapping_status": "AMBIGUOUS"},
    # MEP classes absent from S3DIS training — UNSUPPORTED
    {"class_id": -1, "class_name": "PIPE",       "scan_to_bim_class": None, "mapping_status": "UNSUPPORTED"},
    {"class_id": -1, "class_name": "DUCT",       "scan_to_bim_class": None, "mapping_status": "UNSUPPORTED"},
    {"class_id": -1, "class_name": "CABLE_TRAY", "scan_to_bim_class": None, "mapping_status": "UNSUPPORTED"},
    {"class_id": -1, "class_name": "VALVE",      "scan_to_bim_class": None, "mapping_status": "UNSUPPORTED"},
]


def get_model_label_space(model_key: str) -> dict[str, Any]:
    """Return the label space definition for a registered model.

    Args:
        model_key: One of 'randlanet', 'ptv2', 'kpconv', 'geometric'.

    Returns:
        Dict with model metadata and class list.
    """
    if model_key == "randlanet":
        return {
            "model_key": "randlanet",
            "model_name": "RandLA-Net",
            "training_dataset": "SYNTHETIC_ARCHITECTURAL_v1",
            "classifier_type": "BINARY",
            "num_classes": 2,
            "supported_scan_to_bim_classes": ["WALL"],
            "unsupported_scan_to_bim_classes": RANDLANET_UNSUPPORTED,
            "classes": RANDLANET_ARCHITECTURAL_LABEL_SPACE,
            "notes": (
                "Binary wall classifier. NON_WALL sub-classification (FLOOR, CEILING, "
                "COLUMN, EQUIPMENT) is performed by the downstream geometric adapter."
            ),
        }
    elif model_key == "ptv2":
        return {
            "model_key": "ptv2",
            "model_name": "Point Transformer V2",
            "training_dataset": "S3DIS (when checkpoint available)",
            "classifier_type": "MULTI_CLASS",
            "num_classes": 13,
            "supported_scan_to_bim_classes": [
                "CEILING", "FLOOR", "WALL", "BEAM", "COLUMN",
                "WINDOW", "DOOR", "EQUIPMENT"
            ],
            "unsupported_scan_to_bim_classes": [
                "PIPE", "DUCT", "CABLE_TRAY", "VALVE", "STAIR"
            ],
            "classes": PTV2_S3DIS_LABEL_SPACE,
            "notes": "Multi-class S3DIS classifier. Checkpoint not available — UNAVAILABLE.",
        }
    elif model_key in ("kpconv", "geometric"):
        return {
            "model_key": model_key,
            "model_name": "KPConv" if model_key == "kpconv" else "Geometric Analytical Adapter",
            "training_dataset": "N/A",
            "classifier_type": "GEOMETRIC",
            "num_classes": 0,
            "supported_scan_to_bim_classes": ["WALL", "FLOOR", "CEILING", "COLUMN"],
            "unsupported_scan_to_bim_classes": ["PIPE", "DUCT", "CABLE_TRAY", "VALVE"],
            "classes": [],
            "notes": "Geometric/analytical — no trained weights.",
        }
    else:
        return {"model_key": model_key, "error": "Unknown model key"}


def validate_label_claim(model_key: str, claimed_class: str) -> str:
    """Return 'SUPPORTED', 'UNSUPPORTED', or 'AMBIGUOUS' for a claimed class.

    Args:
        model_key: The model making the claim.
        claimed_class: The ScanTOBIM canonical class being claimed.

    Returns:
        Mapping status string.
    """
    space = get_model_label_space(model_key)
    supported = set(space.get("supported_scan_to_bim_classes", []))
    unsupported = set(space.get("unsupported_scan_to_bim_classes", []))

    if claimed_class in supported:
        return "SUPPORTED"
    elif claimed_class in unsupported:
        return "UNSUPPORTED"
    return "AMBIGUOUS"


def generate_label_space_report(
    output_dir: Path,
    model_keys: list[str] | None = None,
) -> Path:
    """Generate MODEL_LABEL_SPACE.json in output_dir.

    Args:
        output_dir: Directory to write the report.
        model_keys: Models to include. Defaults to all registered.

    Returns:
        Path to the written JSON file.
    """
    if model_keys is None:
        model_keys = ["randlanet", "ptv2", "kpconv", "geometric"]

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    report: dict[str, Any] = {
        "schema_version": "3a.2",
        "report_type": "MODEL_LABEL_SPACE",
        "created_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "canonical_classes": CANONICAL_CLASSES,
        "models": {key: get_model_label_space(key) for key in model_keys},
        "engineering_note": (
            "A model MUST NOT be used to classify a ScanTOBIM class it was not trained on. "
            "UNSUPPORTED classes must be handled by geometric/contextual processing."
        ),
    }

    json_path = output_dir / "MODEL_LABEL_SPACE.json"
    json_path.write_text(json.dumps(report, indent=2), encoding="utf-8")

    logger.info("label_space_report_written", path=str(json_path))
    return json_path
