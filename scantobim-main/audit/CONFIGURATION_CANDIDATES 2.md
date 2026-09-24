# Configuration Candidates & Parameter Classification — ScanTOBIM

## Executive Summary

A core objective of Phase 0 is distinguishing between:
1. **Values that must be derived from point cloud data** (scene-dependent geometry, boundaries, elevations, dimensions).
2. **Parameters that should be configurable** (algorithm tolerances, performance limits, project defaults).
3. **Fixed engineering constants** (mathematical identities, unit conversion constants, API constraints).

Hardcoding data-dependent properties or burying configurable algorithmic thresholds as magic numbers in source files severely compromises repository generality across diverse scans, building types, and coordinate origins.

---

## 1. Parameters That MUST Be Data-Derived

These values are inherently variable across buildings and point cloud datasets. Hardcoding them or falling back to fixed assumptions creates false geometry, breaks multi-storey buildings, and fails on uncentered scans.

| Parameter | Current File & Line Location | Current Hardcoded / Fallback Value | Why It MUST Be Data-Derived |
|---|---|---|---|
| **Scene Spatial Bounds & Extents** | `RevitElementFactory.cs:2736-2737`<br>`fusion_engine.py:181` | `minX = -2000.0`, `maxX = 2000.0`<br>`min_x_mm = -10000.0`, `max_x_mm = 10000.0` | Building extents vary from small mechanical rooms (5m) to large infrastructure (>100m). Arbitrary fallback bounds truncate large scans or bloat small scenes. |
| **Coordinate Origin Offset & Centering** | `geometry_tools.py:172`<br>`coordinate_system.py:38, 120` | `GLOBAL_TRANSFORM.origin_offset_m = np.zeros(3)` | Scans captured in UTM, state plane, or site coordinates have large coordinates (>100,000 m). Origin offset must be computed from scan centroid/base and propagated authoritatively to Revit. |
| **Building Storey Elevations & Level Intervals** | `bim_reconstruction_fix.py:413, 443-446, 491`<br>`classifier.py:539` | `[0.0, 3000.0]`<br>`curr += 3000.0`<br>`floor_step_mm = 3000.0` | Commercial, residential, and industrial facilities have storey heights ranging from 2.4m to over 8.0m. Storey levels must be detected from 1D Z-histogram peak prominence. |
| **Relative Storey Context for Semantic Categorization** | `classifier.py:813, 815, 842, 958, 992, 1044, 1064, 1066, 1068, 1070, 1193, 1199, 1202, 1205, 1208, 1232, 1247, 1253, 1256, 1259, 1261, 1264, 1266, 1268, 1271, 1291` | Absolute Z thresholds:<br>`centroid_z > 1800` (Ceiling)<br>`centroid_z < 1800` (Floor)<br>`bb.min_z < 300` (Door)<br>`centroid_z < 1000` (Pump vs HVAC)<br>`centroid_z > 2500` (Sprinkler) | **CRITICAL FLAW**: In any building where storeys are at elevations > 1.8m (Levels 1, 2, 3...) or negative elevations (e.g. the demo scan Z = -14.75m), absolute Z checks completely fail. Elevations must be normalized relative to local storey floor datum ($z_{\text{local}} = z - z_{\text{storey\_floor}}$). |
| **Wall Thickness** | `stage2_pipeline.py:992`<br>`RevitElementFactory.cs:241`<br>`detection_config.py:34-35` | `DEFAULT_WALL_THICKNESS_MM = 200.0` (forced on single-face walls) | Real walls range from thin drywall (100mm) to heavy structural concrete (>400mm). While un-paired faces cannot measure both faces, the thickness must be flagged as unmeasured and configurable per project rather than assumed as 200mm ground truth. |
| **Wall Length & Centerline Geometry** | `RevitElementFactory.cs:268-269` | `end = new XYZ(start.X + MmToFt(300), start.Y, start.Z)` | Wall endpoints must be derived from inlier cluster line projections. Synthesizing artificial 300mm walls along the X-axis when degenerate violates geometry integrity. |
| **Door & Window Opening Dimensions** | `RevitElementFactory.cs:1174, 1214` | `double widthFt = MmToFt(bb.MaxX - bb.MinX);` | Opening width along a wall running along the Y-axis is $\Delta Y$, not $\Delta X$. Hardcoding $\Delta X$ as width collapses all openings on Y-walls to the wall thickness (~200mm). |
| **Structural Column Cross-Section & Dimensions** | `RevitElementFactory.cs:676, 2556-2564` | `GetDefaultColumnSymbol()` (First loaded symbol) | Column cross-section dimensions (e.g. 450x450mm, 600mm dia) must set family symbol parameters or match appropriate family types, rather than blindly instantiating the first symbol in the project. |
| **Linear MEP / Cable Tray 3D Trajectory** | `discipline_tools.py:168`<br>`RevitElementFactory.cs:737, 808, 872` | `end = new XYZ(start.X + MmToFt(500), start.Y, start.Z)` | Cable tray runs must follow true 3D eigenvector trajectories. Fabricating 500mm X-aligned segments destroys MEP connectivity. |
| **Point Cloud CRS Units & Coordinate Scale** | `scan_tools.py:491, 524`<br>`multi_res_pcd.py:85` | Unconditionally assumes source coordinates are in **metres** | Scans in international feet or survey feet will be scaled incorrectly by $3.28\times$, and millimeter scans by $1000\times$. Scale must be verified against metadata or bounding box plausibility. |

---

## 2. Parameters That SHOULD Be Configurable

These parameters represent algorithmic tuning knobs, performance constraints, and project-specific defaults. They should be externalized in JSON/YAML configuration files, CLI arguments, or environment variables.

| Parameter | Current File & Line Location | Current Hardcoded Value | Category | Recommended Configuration Mechanism |
|---|---|---|---|---|
| **Voxel Downsampling Size** | `detection_config.py:10`<br>`multi_res_pcd.py:45` | `VOXEL_SIZE_M = 0.02`<br>`base_voxel_size_m = 0.05` | Algorithm Parameter | Configuration profile (`detection_profile.json`) or CLI `--voxel-size` |
| **SOR Neighbors & Ratio** | `detection_config.py:14-17` | `SOR_NB_NEIGHBORS = 20`<br>`SOR_STD_RATIO = 2.0` | Algorithm Parameter | `detection_profile.json` or CLI `--sor-std-ratio` |
| **Normal Estimation Radius** | `detection_config.py:21-22` | `NORMAL_RADIUS_VOXEL_MULT = 5.0`<br>`NORMAL_MAX_NN = 30` | Algorithm Parameter | `detection_profile.json` |
| **RANSAC Max Planes Cap** | `detection_config.py:28` | `MAX_PLANES = 130` | Performance Ceiling | Dynamic convergence stopping or CLI `--max-planes` |
| **Wall Fragment Gap Tolerance** | `detection_config.py:59` | `WALL_FRAGMENT_GAP_M = 1.20` | Algorithm Parameter | `detection_profile.json` (tune for doorway bridging) |
| **Wall Coplanar Lateral Offset** | `detection_config.py:58` | `WALL_COPLANAR_OFFSET_M = 0.08` | Algorithm Parameter | `detection_profile.json` |
| **Wall Thickness Acceptance Range** | `detection_config.py:36-37` | `MIN_WALL_THICKNESS_MM = 100.0`<br>`MAX_WALL_THICKNESS_MM = 600.0` | Architectural Filter | Project settings / Building code configuration |
| **Default Wall Thickness** | `detection_config.py:34-35` | `DEFAULT_WALL_THICKNESS_MM = 200.0` | Project Fallback | Project template configuration (`project_defaults.json`) |
| **Default Slab Thickness** | `slab_tools.py:184` | `default_slab_thickness_m = 0.25` | Project Fallback | Project template configuration (`project_defaults.json`) |
| **Storey Detection Min Height** | `slab_tools.py:183` | `min_storey_height_m = 2.2` | Algorithm Parameter | `detection_profile.json` |
| **Storey Peak Prominence Ratio** | `slab_tools.py:231` | `prominence = 0.10 * max_density` | Algorithm Parameter | `detection_profile.json` |
| **Column Height Ratio Threshold** | `yolo_column_adapter.py:187` | `min_column_height_ratio = 0.65` | Geometric Filter | `detection_profile.json` |
| **Column Spatial NMS Radius** | `yolo_column_adapter.py:189` | `nms_radius_m = 0.90` | Algorithm Parameter | `detection_profile.json` |
| **Column Cross-Section Limits** | `yolo_column_adapter.py:399` | `width < 0.15 or depth > 1.20` | Structural Filter | Project structural profile |
| **Opening Grid Resolution** | `grounding_dino_adapter.py:115` | `grid_res_m = 0.10` | Algorithm Parameter | `detection_profile.json` |
| **Opening Dimension Bounds** | `grounding_dino_adapter.py:116-119` | `min_width = 0.70`, `max_width = 3.50`<br>`min_height = 0.90`, `max_height = 3.00` | Architectural Filter | Project architectural standards profile |
| **Door vs Window Sill Threshold** | `grounding_dino_adapter.py:241` | `sill_z_m - base_z <= 0.35` (350mm) | Architectural Filter | Project architectural standards profile |
| **YOLO Model Checkpoint Path** | `yolo_column_adapter.py:142` | `model_path = "yolov8n.pt"` | Neural Model Config | Environment variable `STB_YOLO_MODEL_PATH` or config |
| **YOLO Confidence Threshold** | `yolo_column_adapter.py:142` | `conf_thresh = 0.20` | Neural Model Config | CLI `--yolo-conf` or config |
| **Pipeline Processing Timeout** | `worker.py:221` | `time_limit = 130` | System Operation | Environment variable `STB_PIPELINE_TIMEOUT` |
| **Revit Family & Type Mappings** | `family_map.json` | Nuclear / Plant family defaults | Revit Integration | User project family map (`%APPDATA%\ScanToBIM\family_map.json`) |

---

## 3. Fixed Engineering Constants

These values are legitimate, mathematically immutable, or externally imposed by standards and APIs. They belong directly in code as constants.

| Constant | Location | Value | Engineering Justification |
|---|---|---|---|
| **Millimetres per Metre** | `coordinate_system.py:23`<br>`geometry_tools.py` | `1000.0` | Standard SI unit definition ($1\text{ m} = 1000\text{ mm}$). |
| **Metres per Millimetre** | `coordinate_system.py:24` | `0.001` | Standard SI unit definition. |
| **Feet per Metre** | `coordinate_system.py:25` | `3.280839895013123` | International foot definition ($1\text{ ft} = 0.3048\text{ m}$). |
| **Millimetres per Foot** | `coordinate_system.py:27` | `304.8` | International foot definition ($1\text{ ft} = 304.8\text{ mm}$). |
| **Degrees to Radians Factor** | `coordinate_system.py:48`<br>`stage2_pipeline.py` | `pi / 180.0` | Fundamental trigonometric identity. |
| **Collinear / Parallel Dot Product Tolerance** | `stage2_pipeline.py:873`<br>`classifier.py:282` | `0.95` ($\approx 18.2^\circ$) | Standard directional tolerance for parallel surface pairing in noisy point clouds. |
| **Numerical Zero Precision** | `GeometryUtils.cs:11`<br>`geometry_tools.py` | `1e-6`, `1e-9` | Standard floating-point precision floor to prevent division by zero in vector normalization. |
| **ASTM-E57 Magic Bytes Header** | `scan_tools.py:243`<br>`test_reconstruction_pipeline.py:60` | `b"ASTM-E57"` | ASTM E57.04 standard file signature required by format specification. |
| **Revit Wall Join End Indices** | `RevitElementFactory.cs:291-292` | `0` (Start), `1` (End) | Autodesk Revit API `WallUtils.DisallowWallJoinAtEnd` parameter contract. |
| **Color Depth Scaling** | `scan_tools.py:499` | `255.0` (8-bit), `65535.0` (16-bit) | Standard binary digital image color channel depth specifications. |
| **OpenCV Morphology Kernel Shape** | `grounding_dino_adapter.py:203` | `cv2.MORPH_RECT, (3, 3)` | Standard 8-connectivity spatial neighborhood kernel. |
