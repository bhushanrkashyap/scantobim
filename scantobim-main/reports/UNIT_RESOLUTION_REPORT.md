# Unit Resolution Architecture Report

## Overview

In ScanTOBIM Phase 2, unit handling is executed by the formal `UnitResolution` module (`agent.tools.unit_resolution`).
The system never makes unconditional metre assumptions and replaces previous ad-hoc heuristics (such as checking `pt_span > 500.0`) with a multi-layered evaluation pipeline.

---

## 1. Resolution Pipeline

```
           +----------------------------------------+
           | Input Point Cloud & File Path          |
           +----------------------------------------+
                               |
                               v
            [ Step 1: Explicit Environment Override? ]
             - Yes: STB_INPUT_UNIT -> UNIT_VERIFIED
             - No: Proceed to Step 2
                               |
                               v
            [ Step 2: File Format Metadata Inspection ]
             - LAS/LAZ: Header scales (e.g. 0.001 -> mm), VLR CRS EPSG
             - E57: cartesianBounds, scale fields
             - PLY/PCD: Header comments and format tags
                               |
                               v
            [ Step 3: Geometric Plausibility Analysis ]
             - Bounding diagonal & coordinate range
             - Architectural envelope bounds (1m - 2000m vs 1000mm - 2000000mm)
             - Typical storey height and room dimension checks
                               |
                               v
            [ Step 4: Confidence & Consensus Scoring ]
             - High confidence -> UNIT_VERIFIED or UNIT_INFERRED
             - Disagreement or weak evidence -> UNIT_AMBIGUOUS (scale = 1.0)
```

---

## 2. Status Contracts

1. **`UNIT_VERIFIED`**: Explicit file metadata (LAS scale/CRS, E57 header) or user environment override definitively establishes the unit.
2. **`UNIT_INFERRED`**: Geometric plausibility and coordinate distributions indicate a standard architectural building envelope in a specific unit.
3. **`UNIT_AMBIGUOUS`**: Conflicting metadata, out-of-range coordinates, or featureless point clusters prevent unambiguous identification. **No silent scaling is applied** ($s = 1.0$).
4. **`UNIT_INVALID`**: Corrupt metadata or unparseable coordinate coordinates detected.

---

## 3. Strict Non-Destructive Invariance

The source point cloud file is **never modified in-place**.
The unit conversion factor is registered in `AuthoritativeTransform.scale_source_to_canonical` and applied reversibly:
$$\mathbf{p}_{\text{canonical\_m}} = s \cdot (\mathbf{p}_{\text{source}} - \mathbf{t}_{\text{offset}})$$
Downstream modules query `GLOBAL_TRANSFORM.source_units` and `GLOBAL_TRANSFORM.unit_confidence`.
