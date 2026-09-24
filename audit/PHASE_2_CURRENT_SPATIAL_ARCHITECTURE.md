# Phase 2 — Current Spatial Architecture Audit

## 1. Executive Summary

Before implementing Phase 2 spatial foundation changes, every existing coordinate transformation, unit conversion, origin offset, orientation assumption, and registration path in the ScanTOBIM repository has been cataloged. This audit identifies the gaps that Phase 2 must fill and the existing implementations that must be preserved, unified, or replaced.

---

## 2. Existing Transform Implementations

### 2.1 AuthoritativeTransform (`agent/tools/coordinate_system.py`)

The **single declared** authoritative transform. Current capabilities:

| Field | Type | Current State |
|---|---|---|
| `transform_id` | str | `"TRANSFORM_E57_TO_BIM_REVIT_2025_V1"` |
| `source_coordinate_system` | str | `"E57_LOCAL_METRIC"` |
| `target_coordinate_system` | str | `"REVIT_INTERNAL_FEET"` |
| `origin_offset_m` | ndarray(3,) | Translation offset only |
| `rotation_deg_z` | float | Single-axis Z rotation only |
| `up_axis` | ndarray(3,) | Default `[0, 0, 1]` — **not estimated** |

**GAP**: No full 4×4 rigid transform (SO(3) rotation matrix). Only Z-rotation. No scale field. No forward/inverse matrix pair. No unit confidence. No orientation confidence.

**Methods**:
- `e57_to_bim_mm()`: subtracts origin, applies Z-rotation, scales ×1000
- `bim_mm_to_e57_m()`: scales ×0.001, applies inverse Z-rotation, adds origin
- `bim_mm_to_revit_feet()`: scales ×(1/304.8)
- `e57_to_revit_feet()`: chains e57→mm→feet
- `stamp_segment()`: writes provenance tags
- `to_dict()`: serializes, builds a 4×4 translation-only matrix

**Singleton**: `GLOBAL_TRANSFORM = AuthoritativeTransform()` — used by 6 modules.

### 2.2 RigidTransform (`agent/tools/survey_tools.py`)

A **parallel** transform class from the survey alignment subsystem:

| Field | Type | Current State |
|---|---|---|
| `R` | ndarray(3,3) | Full SO(3) rotation via Kabsch/SVD |
| `t` | ndarray(3,) | Translation in mm |
| `rmse_mm` | float | Fit residual |

**Used by**: `compute_rigid_transform()` (Kabsch from ≥3 control points), `apply_transform()` (segments), `validate_transform_quality()`.

**GAP**: This is a **competing** transform class that is not connected to `AuthoritativeTransform`. It operates in mm and does full 3D rotation, while `AuthoritativeTransform` operates in meters with Z-only rotation.

### 2.3 compute_survey_transform (`agent/tools/geometry_tools.py`, L1213)

A centering-only helper:

```python
origin_offset = (min_pt + max_pt) * 0.5
centered_points = points_xyz - origin_offset
T_survey_to_local = np.eye(4); T_survey_to_local[0:3, 3] = -origin_offset
```

Called by `processor_cli.py` L714. Result assigned to `GLOBAL_TRANSFORM.origin_offset_m`.

**GAP**: Pure translation; no rotation estimation. The returned 4×4 matrix is translation-only. Does not build an inverse.

### 2.4 ICP Registration (`agent/tools/registration_tools.py`)

Multi-scan point-to-plane ICP:

- `load_scan_as_o3d()`: loads scan files, has **heuristic mm→m scaling** (`diag > 500`)
- `register_pair_icp()`: Open3D point-to-plane ICP, returns (T_4x4, rmse, inlier_ratio)
- `register_all_scans()`: sequential pairwise chaining, requires ≥2 files

**GAP**: No FPFH feature extraction. No RANSAC global alignment before ICP. No validation gates (orthogonality, determinant, finite check). No registration graph for N>2 scans. No single-scan bypass mode. Unit heuristic `diag > 500` is fragile.

### 2.5 Registration Report (`agent/tools/registration_report.py`)

RICS/PAS 128 QA classification (Cat A/B/C/FAIL). Well-structured. **No gap** — this is already a proper quality report generator.

---

## 3. Unit Handling

### 3.1 Current Unit Detection Paths

| Location | Method | Status |
|---|---|---|
| `scan_tools.py::_load_las()` L497 | `STB_INPUT_UNIT` env var override | Explicit but opt-in |
| `scan_tools.py::_load_las()` L505 | `ptp > 500 && median > 2000` heuristic | Fragile magnitude check |
| `registration_tools.py::load_scan_as_o3d()` L97 | `diag > 500` → divide by 1000 | Fragile magnitude check |
| `scan_tools.py` L3438 | `STB_COORD_NORMALIZE_THRESHOLD_M` | Coordinate normalization threshold |

**GAP**: No formal `UnitResolution` result object. No `VERIFIED`/`INFERRED`/`AMBIGUOUS`/`INVALID` status. No E57 metadata inspection. No LAS header unit reading (LAS 1.4 defines coordinate system VLRs). No PLY/PCD metadata inspection. Two separate and inconsistent magnitude heuristics (`ptp > 500` vs `diag > 500`).

### 3.2 Unit Conversion Scatter

Inline `* 1000.0` and `/ 1000.0` conversions scattered across ~30 locations in `scan_tools.py`. These are mostly m→mm conversions for segment tags, which is correct behavior within the established pipeline (scan_tools operates in meters internally, outputs mm). The conversion constant is correct but undocumented in most call sites.

`geometry_tools.py` defines centralized conversion functions (`meters_to_mm`, `mm_to_meters`, `meters_to_revit_feet`, etc.) and also defines the same constants as `coordinate_system.py` (`MM_PER_METER`, `FEET_PER_METER`, etc.) — **duplicate** constant definitions.

### 3.3 C# Unit Conversions

`ScanGeometryBuilder.cs` L39: `MmToFeet = 1.0 / 304.8` — centralized constant.
`RevitElementFactory.cs` L2603: `MmToFt(double mm) => mm / 304.8` — utility method.
`AgentModels.cs`: Three separate `MmToFt()` definitions in different classes.

**GAP**: Multiple duplicated `MmToFt` definitions in C#. The mm→ft conversion is at least consistent (all use 304.8), but could be unified.

---

## 4. Vertical Axis Handling

### 4.1 Current State

| Location | Implementation |
|---|---|
| `coordinate_system.py` L40 | `up_axis` field defaults to `[0,0,1]` |
| `scan_tools.py::detect_planes()` L843 | Reads `STB_UP_AXIS` env var (x/y/z), defaults to z |
| `stage2_pipeline.py::extract_vertical_surfaces()` L100 | Reads `GLOBAL_TRANSFORM.up_axis`, falls back to `[0,0,1]` |
| `geometry_tools.py::classify_plane_orientation()` L1156 | Accepts optional `up_axis` param, defaults to `[0,0,1]` |

**GAP**: The up-axis is **never estimated from data**. It is always either the default `[0,0,1]` or an explicit env-var override. Phase 2 must add actual estimation from surface normals, floor/slab candidates, and PCA.

### 4.2 Horizontal Frame

`infer_dominant_orientations()` in `stage2_pipeline.py` L138 performs histogram-based grid direction inference from wall normal angles. This is good and data-derived. But it only works in the horizontal plane and assumes Z is up.

**GAP**: No explicit canonical horizontal basis construction. The existing orientation inference is useful but downstream of vertical-axis estimation.

---

## 5. Coordinate Flow (Python → JSON → C#)

```
E57/LAS scan file
    ↓
scan_tools.py (loads, operates in METRES)
    ↓
detect_planes() → segments with centroids/bounding_boxes in METRES
    ↓
segment_to_model() → converts to MM, writes to segment dict
    ↓
processor_cli.py → serializes to results.json (coordinates in MM)
    ↓
SidecarModels.cs → deserializes SidecarSegment (coordinates in MM)
    ↓
ScanGeometryBuilder.cs → converts MM → Revit feet (÷ 304.8)
    ↓
RevitElementFactory.cs → creates Revit elements (internal feet)
```

### Metadata passed in sidecar JSON:
- `coord_origin_x_mm`, `coord_origin_y_mm`, `coord_origin_z_mm` (from `processor_cli.py`)
- `scan_z_min_mm`, `scan_z_max_mm`
- Per-segment `tags.source_coordinate_system`, `tags.target_coordinate_system`, `tags.transform_id`

**GAP**: The sidecar JSON does not carry the full transform (4×4 matrix, rotation, scale). C# does not consume `AuthoritativeTransform.to_dict()`. The provenance tags are stamped but not consumed by Revit.

---

## 6. Large Survey Coordinates

`scan_tools.py` L3438: Conditionally subtracts `pts_min` when any coordinate exceeds `STB_COORD_NORMALIZE_THRESHOLD_M` (default 100m). The offset is stored in `coord_origin_m` and propagated to `GLOBAL_TRANSFORM.origin_offset_m`.

**GAP**: The normalization uses `pts_min` (not centroid), which shifts the origin to a corner rather than center. `compute_survey_transform()` in `geometry_tools.py` uses centroid `(min + max) / 2` — inconsistency. The inverse transform (canonical → source) is not explicitly stored or validated.

---

## 7. Transform Duplication Summary

| Transform | Location | Rotation | Translation | Scale | Unit |
|---|---|---|---|---|---|
| `AuthoritativeTransform` | `coordinate_system.py` | Z-only | 3D offset | ×1000 (hardcoded) | metres→mm |
| `RigidTransform` | `survey_tools.py` | Full 3D (Kabsch) | 3D | None | mm |
| `compute_survey_transform` | `geometry_tools.py` | None | Centroid centering | None | metres |
| `register_pair_icp` | `registration_tools.py` | Full 3D (ICP) | Full 3D | None | metres |
| C# MmToFt | Multiple C# files | None | None | ÷304.8 | mm→ft |

**5 independent transform implementations** across the codebase. Phase 2 must unify these under one authoritative spatial contract.

---

## 8. Identified Phase 2 Work Items

1. **Upgrade `AuthoritativeTransform`** to support full 4×4 rigid transform with SO(3) rotation, scale, forward/inverse, unit status, orientation status
2. **Create `UnitResolution` component** with formal `VERIFIED`/`INFERRED`/`AMBIGUOUS`/`INVALID` status from file metadata + geometric plausibility
3. **Create vertical-axis estimator** from surface normals, floor candidates, PCA — with `ORIENTATION_VERIFIED`/`INFERRED`/`AMBIGUOUS` confidence
4. **Create horizontal frame estimator** downstream of vertical axis
5. **Unify registration pipeline** with FPFH + RANSAC + ICP path and validation gates; Mode A (single scan) vs Mode B (multi-scan)
6. **Eliminate duplicate unit constants** between `coordinate_system.py` and `geometry_tools.py`
7. **Reconcile centering strategies** (`pts_min` vs centroid)
8. **Extend sidecar JSON schema** to carry full transform metadata for C# consumption
9. **Create Phase 2 test suite** with rotation/translation/scale invariance tests
10. **Document canonical unit choice** (metres for computation, mm for segment output)
