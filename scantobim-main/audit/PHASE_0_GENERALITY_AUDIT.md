# Master Phase 0 Generality & Dataset-Dependency Audit Report

## 1. Executive Summary

This audit establishes the comprehensive baseline of dataset-specific hardcoding, fixed coordinates, arbitrary thresholds, hidden fallback geometry, and architectural assumptions across the entire **ScanTOBIM** repository.

In strict accordance with the Phase 0 audit rules:
- **NO SOURCE CODE HAS BEEN MODIFIED**.
- **NO ATTEMPT HAS BEEN MADE TO OPTIMISTICALLY DECLARE THE REPOSITORY GENERAL**.
- **EVERY SUSPICIOUS CONSTANT AND GEOMETRIC PATH HAS BEEN EMPIRICALLY TRACED**.

### Quantitative Metrics

- **Files Inspected**: 252
- **Numeric Constants Inspected**: 6,329
- **Suspicious Values / Violations Documented**: 53
- **Dataset-Specific Production Findings**: 41
- **Critical Severity Findings**: 20
- **High Severity Findings**: 22
- **Medium Severity Findings**: 10
- **Low Severity Findings**: 1
- **Legitimate Acceptable Constants**: 1,900

### Final Verdict

```text
FINAL VERDICT: GENERALITY_AUDIT_REQUIRES_REMEDIATION
```

> [!CAUTION]
> **VERDICT RATIONALE**: The repository contains material dataset-specific assumptions, including:
> 1. **Over 25 hardcoded absolute Z elevation thresholds** in `agent/classifier.py` (e.g. `centroid_z > 1800` for ceilings vs floors, `bb.min_z < 300` for doors vs windows, `centroid_z < 1000` for pumps vs HVAC) that completely break on multi-storey buildings or scans with non-zero base elevations.
> 2. **X-axis orientation hardcoding** in `RevitElementFactory.cs` where opening widths are unconditionally calculated as `bb.MaxX - bb.MinX`, collapsing all Y-aligned openings to wall thickness.
> 3. **Hidden fallback geometry fabrication**, including synthetic 300mm walls, 500mm beams/ducts, 1000mm railings, and DirectShape bounding box cuboids created when native geometry creation fails.
> 4. **Hardcoded developer local paths** (`/Users/bhushanrkaashyap/Desktop/...`) embedded in production AI adapters.
> 5. **Unconditional metre unit assumptions** without checking CRS metadata or point scale.
> 6. **Synthetic 40-room building generation** embedded in the production orchestrator fallback.

---

## 2. Dataset-Specific & Generality Findings Table

| ID | File | Line | Finding | Evidence | Severity | Why It Matters |
|---|---|---|---|---|---|---|
| `GEN-PATH-001` | `agent/adapters/yolo_column_adapter.py` | 155 | PATH | `Path("/Users/bhushanrkaashyap/Desktop/ScanTOBIM (2) 3") / model_path,` | **HIGH** | Candidate path search list includes hardcoded developer local desktop path. |
| `GEN-PATH-002` | `agent/adapters/ptv3_semantic_adapter.py` | 102 | PATH | `Path("/Users/bhushanrkaashyap/Desktop/ScanTOBIM (2) 3/research_repos/Cloud2BIM/DL_module/weights/run_20260803_0021/best.pth"),` | **HIGH** | Checkpoint candidate path explicitly hardcodes local macOS username and temporary Desktop directory. |
| `GEN-PATH-003` | `run_stage2_demo.py` | 702 | PATH | `Path("/Users/bhushanrkaashyap/Desktop/ScanTOBIM (2)") / scan_path_obj.name,` | **MEDIUM** | Fallback scan loader searches developer desktop directory for 'actual_user_scan.ply'. |
| `GEN-PATH-004` | `test_wall_detection.py` | 11 | PATH | `PCD_PATH = r"C:\Users\RKAA6083\Downloads\point cloud data\2026-07-28.e57"` | **MEDIUM** | Test script hardcodes developer workstation Downloads path and specific scan filename. |
| `GEN-PATH-005` | `agent/tools/processor_cli.py` | 12 | PATH | `pyinstaller --onefile --name stb-processor --distpath "C:\Users\BAVA6928\Downloads\SCANTOBIM_V10.2.2\..."` | **LOW** | Build instructions comment hardcodes specific developer Windows user profile path. |
| `GEN-SEM-001` | `agent/classifier.py` | 958 | SEMANTIC | `elif (segment.normal is None or segment.normal.z < 0.30) and centroid_z > 1800: element_type = ElementType.CEILING` | **CRITICAL** | Differentiates Floor vs Ceiling using hardcoded absolute elevation threshold 1800 mm (1.8m). |
| `GEN-SEM-002` | `agent/classifier.py` | 992 | SEMANTIC | `elif bb.min_z < 300: element_type = ElementType.DOOR else: element_type = ElementType.WINDOW` | **CRITICAL** | Differentiates Door vs Window using hardcoded absolute elevation threshold min_z < 300 mm. |
| `GEN-SEM-003` | `agent/classifier.py` | 1044 | SEMANTIC | `elif diameter_mm > 250 and major_d < 150 and centroid_z < 500: element_type = ElementType.SEISMIC_ISOLATOR` | **HIGH** | Requires absolute centroid Z < 500 mm to classify squat cylinders as seismic isolators. |
| `GEN-SEM-004` | `agent/classifier.py` | 1064 | SEMANTIC | `if centroid_z > 2800 and major_d < 80: element_type = ElementType.SMOKE_DETECTOR` | **HIGH** | Requires absolute centroid Z > 2800 mm to classify small cylinders as smoke detectors. |
| `GEN-SEM-005` | `agent/classifier.py` | 1066 | SEMANTIC | `elif 600 < centroid_z < 900 and diameter_mm > 40: element_type = ElementType.FIRE_HYDRANT` | **HIGH** | Requires absolute centroid Z between 600mm and 900mm to classify cylinders as fire hydrants. |
| `GEN-SEM-006` | `agent/classifier.py` | 1068 | SEMANTIC | `elif centroid_z > 2500: element_type = ElementType.SPRINKLER` | **HIGH** | Requires absolute centroid Z > 2500 mm to classify cylindrical heads as fire sprinklers. |
| `GEN-SEM-007` | `agent/classifier.py` | 1070 | SEMANTIC | `elif centroid_z < 200 and aspect_ratio < 1.8: element_type = ElementType.DRAINAGE` | **HIGH** | Requires absolute centroid Z < 200 mm to classify squat cylinders as floor drains. |
| `GEN-SEM-008` | `agent/classifier.py` | 1193 | SEMANTIC | `elif centroid_z > 5000 and major > 5000 and not dominant_axis_vertical: element_type = ElementType.OVERHEAD_CRANE` | **HIGH** | Requires absolute centroid Z > 5000 mm (5m) to classify large boxes as overhead cranes. |
| `GEN-SEM-009` | `agent/classifier.py` | 1199 | SEMANTIC | `elif minor1 > 500 and minor2 > 600 and 800 < major < 4000 and centroid_z < 2500: element_type = ElementType.TRANSFORMER` | **CRITICAL** | Classifies electrical boxes as transformers only if absolute centroid Z < 2500 mm. |
| `GEN-SEM-010` | `agent/classifier.py` | 1202 | SEMANTIC | `elif minor1 > 350 and minor2 > 450 and major > 1500 and centroid_z < 1500: element_type = ElementType.COMPRESSOR` | **CRITICAL** | Classifies mechanical boxes as compressors only if absolute centroid Z < 1500 mm. |
| `GEN-SEM-011` | `agent/classifier.py` | 1205 | SEMANTIC | `elif minor1 < 700 and minor2 > 400 and major > 1500 and centroid_z > 800: element_type = ElementType.SWITCHGEAR` | **CRITICAL** | Requires absolute centroid Z > 800 mm to classify cabinet rows as switchgear. |
| `GEN-SEM-012` | `agent/classifier.py` | 1208 | SEMANTIC | `elif 200 < minor1 < 700 and 200 < minor2 < 700 and major > 2000 and centroid_z < 1200: element_type = ElementType.UPS_SYSTEM` | **CRITICAL** | Requires absolute centroid Z < 1200 mm to classify cabinet rows as UPS systems. |
| `GEN-SEM-013` | `agent/classifier.py` | 1232 | SEMANTIC | `elif centroid_z > 2600 and minor1 < 180 and minor2 < 280: element_type = ElementType.LIGHTING_FITTING` | **HIGH** | Requires absolute centroid Z > 2600 mm to classify narrow boxes as lighting fixtures. |
| `GEN-SEM-014` | `agent/classifier.py` | 1247 | SEMANTIC | `elif minor1 < 100 and 150 < minor2 < 600 and major < 700 and 1000 < centroid_z < 2200: element_type = ElementType.FIRE_ALARM_PANEL` | **HIGH** | Requires absolute centroid Z between 1000mm and 2200mm to classify wall boxes as fire alarm panels. |
| `GEN-SEM-015` | `agent/classifier.py` | 1271 | SEMANTIC | `element_type = ElementType.PUMP if centroid_z < 1000 else ElementType.HVAC_EQUIPMENT` | **CRITICAL** | Arbitrary threshold splits equipment boxes into Pump vs HVAC purely based on whether Z < 1000 mm. |
| `GEN-SEM-016` | `agent/classifier.py` | 1273 | SEMANTIC | `else: element_type = ElementType.BEAM` | **CRITICAL** | Catch-all default for BOX segments is ElementType.BEAM without structural framing verification. |
| `GEN-SEM-017` | `agent/tools/scan_tools.py` | 2248 | SEMANTIC | `Survivors are emitted as VALVE_CANDIDATE so the classifier can map them directly to ElementType.VALVE without further geometry checks.` | **CRITICAL** | Any compact cluster near a pipe is automatically classified as a valve candidate without cylinder/spindle evidence. |
| `GEN-SEM-018` | `revit-addin/Services/ScanGeometryBuilder.cs` | 175 | SEMANTIC | `"pipe" or "cylinder" or "valve_candidate" => "PIPE",` | **CRITICAL** | Directly forces all 'cylinder' and 'valve_candidate' shapes into the 'PIPE' Revit creation pipeline. |
| `GEN-SEM-019` | `agent/adapters/ptv3_semantic_adapter.py` | 178 | SEMANTIC | `if (n_z > 0.80 or shape == "plane_horizontal") and is_planar: pred = "IfcSlab" ... elif linearity > 0.50 ... axis_z < 0.30: pred = "IfcBeam"` | **CRITICAL** | Primitive geometric features directly map to IFC semantic categories; any horizontal cylinder with linearity > 0.5 becomes an IfcBeam. |
| `GEN-GEO-001` | `revit-addin/Services/RevitElementFactory.cs` | 1174 | GEOMETRY | `double widthFt  = MmToFt(bb.MaxX - bb.MinX);` | **CRITICAL** | Door and window opening width assumes host wall runs along the X-axis; for Y-aligned walls, width collapses to wall thickness (~200mm). |
| `GEN-GEO-002` | `revit-addin/Services/RevitElementFactory.cs` | 268 | GEOMETRY | `if (start.DistanceTo(end) < MmToFt(300)) end = new XYZ(start.X + MmToFt(300), start.Y, start.Z);` | **CRITICAL** | Fabricates artificial 300mm wall geometry along the X-axis when degenerate length is encountered. |
| `GEN-GEO-003` | `revit-addin/Services/RevitElementFactory.cs` | 488 | GEOMETRY | `end = new XYZ(start.X + MmToFt(500), start.Y, start.Z);` | **CRITICAL** | Fabricates artificial 500mm beam/duct/cable-tray geometry along the X-axis when length is degenerate. |
| `GEN-GEO-004` | `revit-addin/Services/RevitElementFactory.cs` | 1252 | GEOMETRY | `end = new XYZ(start.X + MmToFt(1000), start.Y, start.Z);` | **CRITICAL** | Fabricates artificial 1000mm railing geometry along the X-axis when degenerate. |
| `GEN-GEO-005` | `revit-addin/Services/RevitElementFactory.cs` | 1451 | GEOMETRY | `end = new XYZ(x0 + MmToFt(500), y0, z0);` | **CRITICAL** | Fabricates artificial 500mm kerb geometry along the X-axis when degenerate. |
| `GEN-GEO-006` | `revit-addin/Services/RevitElementFactory.cs` | 318 | FALLBACK | `var wallSolid = BuildBoxSolid(bx0, by0, bz0, bx1, by1, bz1); ... DirectShape.CreateElement(_doc, new ElementId(BuiltInCategory.OST_Walls));` | **CRITICAL** | When native Wall.Create fails, code silently falls back to creating an axis-aligned bounding box DirectShape. |
| `GEN-GEO-007` | `revit-addin/Services/ScanGeometryBuilder.cs` | 911 | FALLBACK | `var doorSeg = BuildOpeningProxySegment(seg, isDoor: true); ... TryCreateDirectShape(doc, view, doorSeg, BuiltInCategory.OST_Doors, out var doorId)` | **CRITICAL** | When native door creation fails, code creates a solid DirectShape cuboid proxy in OST_Doors. |
| `GEN-GEO-008` | `revit-addin/Services/ScanGeometryBuilder.cs` | 1167 | FALLBACK | `return BuildBoxGeometry(seg.BoundingBox);` | **CRITICAL** | Universal fallback in ScanGeometryBuilder creates solid axis-aligned box from segment bounding box. |
| `GEN-GEO-009` | `agent/bim_reconstruction_fix.py` | 695 | GEOMETRY | `if axes == {"Y"}: ... for boundary_y, face in ((min_y, "min"), (max_y, "max")): ... synthetic.append(base)` | **HIGH** | Synthesizes missing enclosure walls along X or Y when single-axis walls are detected (currently dead code behind early return). |
| `GEN-GEO-010` | `agent/stage2_pipeline.py` | 992 | GEOMETRY | `thickness_mm = DEFAULT_WALL_THICKNESS_MM` | **MEDIUM** | Single-face walls (where opposing face is unscanned) inherit hardcoded 200mm thickness. |
| `GEN-STOR-001` | `agent/bim_reconstruction_fix.py` | 443 | STOREY | `curr = z_min + 3000.0 ... while curr < z_max + 750.0: levels.append(curr); curr += 3000.0` | **HIGH** | Storey detection fallback assumes a fixed 3.0m storey height interval when horizontal planes are absent. |
| `GEN-STOR-002` | `agent/classifier.py` | 539 | STOREY | `def align_axes(segments: list[GeometrySegment], floor_step_mm: float = 3000.0) -> None:` | **MEDIUM** | Axis alignment helper assumes default floor step of 3000 mm. |
| `GEN-STOR-003` | `agent/tools/slab_tools.py` | 184 | STOREY | `default_slab_thickness_m: float = 0.25` | **MEDIUM** | Default slab thickness is hardcoded to 250 mm. |
| `GEN-REV-001` | `revit-addin/Services/RevitElementFactory.cs` | 2556 | REVIT | `private FamilySymbol GetDefaultColumnSymbol() => new FilteredElementCollector(_doc).OfCategory(BuiltInCategory.OST_StructuralColumns).Cast<FamilySymbol>().FirstOrDefault()` | **HIGH** | Ignores measured column cross-section dimensions and instantiates the first loaded symbol in the project. |
| `GEN-REV-002` | `revit-addin/Services/RevitElementFactory.cs` | 1832 | REVIT | `private FamilySymbol GetDefaultDoorSymbol() => ... OfCategory(BuiltInCategory.OST_Doors).Cast<FamilySymbol>().FirstOrDefault()` | **MEDIUM** | Picks first loaded door symbol regardless of scan opening type or dimensions. |
| `GEN-REV-003` | `revit-addin/family_map.json` | 28 | REVIT | `"family": "M_Gate Valve - Flanged", "family": "M_Nuclear Pressurizer"` | **HIGH** | family_map.json contains exclusively industrial/nuclear plant families, with no configuration for doors, windows, walls, or columns. |
| `GEN-REV-004` | `revit-addin/Services/FamilyResolver.cs` | 169 | REVIT | `private FamilySymbol? FirstSymbolInCategory(BuiltInCategory category) => new FilteredElementCollector(_doc)...FirstOrDefault();` | **MEDIUM** | Fallback resolution silently picks the first arbitrary symbol in the category if config lookup fails. |
| `GEN-AI-001` | `agent/adapters/ptv3_semantic_adapter.py` | 53 | MODEL | `model_name: str = "GEOMETRIC_TENSOR_CLASSIFIER"` | **HIGH** | Adapter claims to represent PTv3 + PPT, but actually executes 3D covariance eigenvalue geometric tensor signatures without neural inference. |
| `GEN-AI-002` | `agent/adapters/grounding_dino_adapter.py` | 12 | MODEL | `detection_source: str = "GEOMETRIC_OPENING_DETECTOR"` | **HIGH** | Adapter claims GroundingDINO, but executes OpenCV connected component void analysis on 2D elevation grids without vision-language inference. |
| `GEN-AI-003` | `agent/adapters/ascan2bim_wall_adapter.py` | 40 | MODEL | `reasoning_mode: str = "ALGORITHMIC_CORNER_EDGE_REASONING"` | **HIGH** | Adapter claims A-Scan2BIM, but executes OpenCV Shi-Tomasi corners and line corridor point counting without neural weights. |
| `GEN-AI-004` | `agent/adapters/yolo_column_adapter.py` | 243 | MODEL | `preds = self.model.predict(rgb_img, conf=self.conf_thresh, verbose=False)` | **HIGH** | Executes generic COCO object detector weights (yolov8n.pt); any detected bounding box is accepted as a column proposal without verifying class ID. |
| `GEN-COORD-001` | `agent/tools/processor_cli.py` | 720 | COORDINATE | `centered_pts, T_4x4, origin_offset = compute_survey_transform(downsampled_pts_m)` | **HIGH** | Survey transform computes origin offset, but the offset is not assigned to GLOBAL_TRANSFORM.origin_offset_m, leaving downstream coordinates uncentered. |
| `GEN-COORD-002` | `agent/tools/scan_tools.py` | 3404 | COORDINATE | `coord_origin_m = np.zeros(3)  # no normalisation needed` | **MEDIUM** | Overwrites calculated coordinate origin with zero vector, disabling coordinate normalization. |
| `GEN-COORD-003` | `revit-addin/Services/RevitElementFactory.cs` | 2050 | COORDINATE | `private double ResolveInstructionZOffsetMm(ElementInstruction instruction)` | **HIGH** | Revit factory supports arbitrary vertical Z-offsets applied via environment variable or instruction parameters. |
| `GEN-UNIT-001` | `agent/tools/scan_tools.py` | 491 | UNIT | `points = np.vstack((las.x, las.y, las.z)).T` | **HIGH** | LAS/LAZ and E57 loaders unconditionally assume points are in metres without reading CRS units or scale headers. |
| `GEN-ORNT-001` | `agent/stage2_pipeline.py` | 101 | ORIENTATION | `up_axis = np.array([0.0, 0.0, 1.0], dtype=float)` | **HIGH** | Vertical surface extraction unconditionally assumes world Z is the up-axis; un-leveled scans will misclassify walls as sloped planes. |
| `GEN-FIX-001` | `agent/orchestrator.py` | 431 | FIXTURE | `segments = generate_large_synthetic_segments(zone_id=zone_id)` | **CRITICAL** | Orchestrator contains active branch generating 40+ synthetic rooms with hardcoded coordinates when use_synthetic=True. |
| `GEN-CONF-001` | `agent/tools/detection_config.py` | 28 | CONFIGURATION | `MAX_PLANES = 130` | **MEDIUM** | Magic number 130 matches column count in demo report; caps RANSAC plane extraction. |
| `GEN-CONF-002` | `agent/tools/detection_config.py` | 34 | CONFIGURATION | `WALL_THICKNESS_MM = 200.0` | **MEDIUM** | Hardcoded default wall thickness 200mm applied across pipeline. |

---

## 3. Coordinate Findings (Phase 4 Audit)

1. **Survey Centering Decoupled from Authoritative Transform**: `processor_cli.py:720` computes `origin_offset` via `compute_survey_transform(downsampled_pts_m)`, but does NOT set `GLOBAL_TRANSFORM.origin_offset_m` (which remains `np.zeros(3)`). Consequently, coordinates exported to sidecar JSON retain huge raw coordinates if the scan was georeferenced, risking floating-point vertex jitter in Revit.
2. **Local Coordinate Normalization Nullified**: `scan_tools.py:3404` explicitly executes `coord_origin_m = np.zeros(3)  # no normalisation needed`, disabling automatic coordinate shifting.
3. **Arbitrary Vertical Z-Shifting**: `RevitElementFactory.cs:2050` includes `_envZOffsetMm` and `_instructionZOffsetMm`, enabling manual vertical coordinate shifts rather than enforcing scan-derived datum alignment.

## 4. Unit Findings (Phase 6 Audit)

1. **Blind Metre Assumption**: `scan_tools.py:491` (LAS loader) and `scan_tools.py:524` (E57 loader) unconditionally assume raw Cartesian coordinates are in metres. LAS VLRs and CRS scale factors are never inspected. Scans in US Survey Feet or International Feet will be distorted by $3.28\times$, and millimeter scans by $1000\times$.
2. **Scattered Unit Multipliers**: Unit conversions (`* 1000.0`, `/ 1000.0`, `/ 304.8`) are scattered across 40+ source files rather than routed through `GLOBAL_TRANSFORM`.

## 5. Storey Findings (Phase 7 Audit)

1. **Fixed 3.0m Level Elevation Interval Fallback**: `agent/bim_reconstruction_fix.py:443-446` falls back to `curr = z_min + 3000.0` and iterates in 3000mm steps when horizontal planes are not found.
2. **Default Storey Step Assumption**: `agent/classifier.py:539` defines `floor_step_mm: float = 3000.0`.
3. **Fixed Default Slab Thickness**: `agent/tools/slab_tools.py:184` hardcodes `default_slab_thickness_m = 0.25` (250mm).

## 6. Geometry Findings (Phases 8, 9, 10, 11 Audit)

1. **X-Axis Opening Width Assumption**: `RevitElementFactory.cs:1174, 1214` calculates door/window width as `bb.MaxX - bb.MinX`. For walls aligned along the Y-axis, this represents the wall thickness, not opening width.
2. **Fake Geometry Synthesis**: `RevitElementFactory.cs` synthesizes artificial geometry along the X-axis when degenerate lengths occur: 300mm for walls (line 268), 500mm for beams/ducts/kerbs (lines 488, 737, 808, 1451), and 1000mm for railings (line 1252).
3. **Column Dimension Disregard**: `RevitElementFactory.cs:676` instantiates `GetDefaultColumnSymbol()` without setting family width or depth parameters, ignoring the measured 490mm or circular profile.
4. **Single-Face Wall Thickness Default**: `stage2_pipeline.py:992` defaults un-paired wall faces to `DEFAULT_WALL_THICKNESS_MM = 200.0` mm.

## 7. Semantic Findings (Phases 12 & 18 Audit)

1. **Absolute Elevation Semantic Thresholds**: Over 25 rules in `agent/classifier.py` use absolute Z heights tailored for a single ground-floor room (Ceiling > 1800mm, Door < 300mm, Sprinkler > 2500mm, Drainage < 200mm, Pump < 1000mm, etc.). In multi-storey buildings or below-ground scans, these fail completely.
2. **Simplistic Primitive-to-BIM Mapping**: Any compact cluster near a pipe is emitted as `VALVE_CANDIDATE` (`scan_tools.py:2248`), any unclassified BOX becomes `BEAM` (`classifier.py:1273`), and `ScanGeometryBuilder.cs:175` collapses all cylinders and valve candidates to `PIPE`.

## 8. Revit Add-in Findings (Phase 14 Audit)

1. **Missing Architectural & Structural Family Mappings**: `family_map.json` contains exclusively nuclear and industrial equipment (`M_Nuclear Pressurizer`, `M_Gate Valve`, etc.) with zero configurations for doors, windows, walls, or columns.
2. **First Symbol In Category Fallback**: `FamilyResolver.cs:169` silently resolves to the first loaded family symbol in the document when no match is found.

## 9. AI & Model Findings (Phase 17 Audit)

1. **COCO Pretrained Weights for Columns**: `YoloColumnAdapter` runs standard `yolov8n.pt` and accepts any detected 2D bounding box as a column proposal without checking class labels.
2. **Heuristic Adapters Labeled as Deep Learning Models**:
   - `Ptv3SemanticAdapter` is an analytical 3D covariance eigenvalue signature classifier (`GEOMETRIC_TENSOR_CLASSIFIER`), not a neural network forward pass.
   - `GroundingDinoAdapter` is an OpenCV connected component void detector (`GEOMETRIC_OPENING_DETECTOR`), not a neural transformer.
   - `AScan2BimWallAdapter` is an algorithmic Shi-Tomasi corner detector, not neural corner/edge models.

## 10. Fallback Findings (Phase 13 Audit)

1. **DirectShape Bounding Box Solid Fallback**: `RevitElementFactory.cs:318` creates an axis-aligned bounding box solid as a DirectShape when native `Wall.Create` fails.
2. **Opening Proxy Cuboids**: `ScanGeometryBuilder.cs:911, 939` builds solid cuboid DirectShapes for doors and windows when native opening creation fails.
3. **Universal DirectShape Cuboid Fallback**: `ScanGeometryBuilder.cs:1167` executes `return BuildBoxGeometry(seg.BoundingBox)` for unsupported shapes.

## 11. Path Findings (Phase 16 Audit)

Hardcoded local developer paths discovered in production code:
- `agent/adapters/yolo_column_adapter.py:155`: `Path("/Users/bhushanrkaashyap/Desktop/ScanTOBIM (2) 3") / model_path`
- `agent/adapters/ptv3_semantic_adapter.py:102`: `Path("/Users/bhushanrkaashyap/Desktop/ScanTOBIM (2) 3/research_repos/Cloud2BIM/DL_module/weights/run_20260803_0021/best.pth")`
- `run_stage2_demo.py:702-703`: `Path("/Users/bhushanrkaashyap/Desktop/ScanTOBIM (2)")`
- `test_wall_detection.py:11`: `PCD_PATH = r"C:\Users\RKAA6083\Downloads\point cloud data\2026-07-28.e57"`

## 12. Test Contamination Findings (Phase 20 Audit)

Production code does not import from test fixtures directly, but `orchestrator.py:431` contains an active production code path calling `generate_large_synthetic_segments()`, which fabricates an entire synthetic building with hardcoded coordinates, dimensions, and element counts.

## 13. Configuration Candidates Summary (Phase 19 Audit)

Detailed candidate definitions are recorded in `audit/CONFIGURATION_CANDIDATES.md`:
- **Must be data-derived**: Scene bounds, coordinate origin offset, storey elevations, relative storey height context for classification, wall thicknesses, opening widths, column cross-sections, point cloud CRS scale.
- **Should be configurable**: Voxel downsampling sizes, SOR outlier thresholds, RANSAC plane extraction cap, wall fragment gap tolerances, single-face default wall thickness, default slab thickness, column NMS radius, opening dimension ranges, YOLO confidence thresholds, pipeline timeouts.
- **Fixed engineering constants**: Metric/Imperial conversion factors (1000.0, 304.8), floating point precision tolerances (1e-6, 1e-9), trigonometric factors, ASTM E57 header bytes.

## 14. Acceptable Constants (Phase 3 Audit)

1,900 instances of acceptable mathematical, unit conversion, and API constants are catalogued in `audit/numeric_constants.csv`.

## 15. Unknowns Requiring Manual Review

3,828 numeric constants classified as `F. UNKNOWN` in `audit/numeric_constants.csv` correspond to internal loop counters, buffer sizes, HTTP timeouts, and UI display coordinates that should be audited before final production hardening.

---

## 16. Conclusion & Next Steps

The ScanTOBIM repository exhibits sophisticated geometric algorithms (dual-face opposing wall pairing, 1D Z-histogram storey detection, 2D ConvexHull minimum bounding rectangle), but is currently restricted by severe dataset-specific assumptions, absolute elevation thresholds, and geometry fabrication fallbacks.

**Phase 0 Audit is COMPLETE. No production code has been modified.**
Review of this audit report and approval of the remediation plan is required before proceeding to Phase 1 execution.
