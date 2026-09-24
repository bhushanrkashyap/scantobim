# Phase 2 — Spatial Architecture & Generality Audit

## Executive Summary

Phase 2 establishes a single authoritative spatial foundation for ScanTOBIM.
Prior to this phase, spatial processing relied on fragmented coordinate representations, Z-only rotation assumptions, ad-hoc threshold-based unit conversions (`ptp > 500`), unconditional Z-up vertical orientation assumptions, and ICP-only registration lacking global alignment or degradation guards.

With the completion of Phase 2, the ScanTOBIM system operates under a strictly defined, reversible mathematical contract that decouples all downstream BIM and geometric feature recognition from the source coordinate system, position, orientation, scale, and scan station topology.

---

## 1. Spatial Contract & Coordinate Transformation Matrix

| Capability | Phase 1 Baseline | Phase 2 Foundation | Status |
|---|---|---|---|
| **Authoritative Contract** | `AuthoritativeTransform` (Z-rotation only) | Full rigid $SE(3)$ transform ($4 \times 4$) with $SO(3)$ rotation matrix ($3 \times 3$) and origin translation | **REMEDIATED** |
| **Reversibility Guarantee** | None verified programmatically | Numerical round-trip verified: $\|T^{-1}(T(p)) - p\| < 0.01\text{ mm}$ | **VERIFIED** |
| **Source Coordinate Safety** | Source coordinates sometimes modified in-place | Source coordinates are sacred and never modified; all transformations are derived reversibly | **VERIFIED** |
| **Survey / UTM Coordinates** | Centroid offset computed in `processor_cli` but not unified with `AuthoritativeTransform` | `AuthoritativeTransform` unifies survey offset, bounding box tracking, and forward/inverse transforms | **UNIFIED** |
| **Revit Feet Conversion** | Split between C# and Python with multiple conversion factor constants | Centralized in `coordinate_system.py`, re-exported, and matched exactly with C# `MmToFt()` ($1/304.8$) | **CENTRALIZED** |

---

## 2. Unit Resolution Foundation

| Capability | Phase 1 Baseline | Phase 2 Foundation | Status |
|---|---|---|---|
| **Unit Resolution Mechanism** | Ad-hoc `diag > 500` or `pt_span > 500` in loader | Formal `resolve_units()` component combining metadata inspection and geometric plausibility | **REMEDIATED** |
| **Metadata Readers** | None | Specialized readers for LAS (header scale, VLR CRS), E57 (bounds & tags), PLY, and PCD | **IMPLEMENTED** |
| **Plausibility Envelope** | Hardcoded range checks | Explicit building envelope distributions: Metres ($1\text{m} - 2000\text{m}$), Millimetres ($1000\text{mm} - 2\times 10^6\text{mm}$) | **VERIFIED** |
| **Ambiguity Policy** | Silently defaulted or assumed metres | Returns `UNIT_AMBIGUOUS` with scaling factor $1.0$ (never silently scales or corrupts source data) | **VERIFIED** |
| **Provenance Logging** | Missing | Explicit provenance trail (`evidence` list, `confidence_score`, `detected_unit`, `status`) | **LOGGED** |

---

## 3. Orientation & Vertical Axis Estimation

| Capability | Phase 1 Baseline | Phase 2 Foundation | Status |
|---|---|---|---|
| **Up-Axis Assumption** | Default Z-up unconditional | Data-derived estimation via surface normal histograms, dominant horizontal planes, and PCA | **REMEDIATED** |
| **Consensus Mechanism** | None | Multi-method consensus (cosine similarity threshold $> 0.95$) | **IMPLEMENTED** |
| **Ambiguity Handling** | Fallback to Z-up silently | Emits `ORIENTATION_AMBIGUOUS` when features lack strong vertical or planar evidence | **VERIFIED** |
| **Horizontal Frame** | Assumed aligned with X/Y axes | Estimated via circular histogram of wall surface normal projections to find dominant wall axes | **IMPLEMENTED** |
| **Canonical Frame** | Undefined | `build_canonical_frame()` constructs right-handed orthonormal $SO(3)$ rotation matrix to canonical frame | **IMPLEMENTED** |

---

## 4. Multi-Scan Registration Engine

| Capability | Phase 1 Baseline | Phase 2 Foundation | Status |
|---|---|---|---|
| **Single Scan Handling** | Undefined / would attempt registration if called | `resolve_registration_mode()` returns `REGISTRATION_NOT_REQUIRED` without fabricating synthetic targets | **VERIFIED** |
| **Global Alignment** | Identity matrix assumed as initial guess | FPFH feature descriptor extraction and RANSAC global alignment | **IMPLEMENTED** |
| **Local Refinement** | Point-to-plane ICP | Point-to-plane ICP initialized with RANSAC global alignment | **UPGRADED** |
| **Degradation Detection** | None (ICP always accepted) | Degradation check detects if ICP worsens fitness ($< 0.9 \times$) or RMSE ($> 1.5 \times$) and retains RANSAC transform | **VERIFIED** |
| **Validation Gates** | None | `RegistrationQuality` validates fitness ($\ge 0.3$), RMSE ($\le 0.2\text{m}$), $\det(R) \approx 1.0$, and orthogonality | **VERIFIED** |

---

## 5. Verification Summary

* **Phase 2 Targeted Tests**: 15/15 passed (`agent/tests/test_phase2_spatial_foundation.py`)
* **Phase 1 Regressions**: 10/10 passed (`agent/tests/test_generality_phase1.py`)
* **Registration Suite**: 14/14 passed (`agent/tests/test_registration.py`)
* **Processor CLI Suite**: 24/24 passed (`agent/tests/test_processor_cli.py`)
* **Stage 2 Reconstruction Suite**: 16/16 passed (`agent/tests/test_stage2_core.py`, `test_stage2_pipeline.py`)
