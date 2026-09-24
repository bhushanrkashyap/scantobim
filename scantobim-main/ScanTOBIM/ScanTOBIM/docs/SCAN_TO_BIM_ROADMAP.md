# Scan-to-BIM Implementation Roadmap

> **Status:** Approved Baseline Audit — Ready for Phase 2 Execution  
> **Target Architecture:** Point Cloud (PLY/E57/LAS) ➔ Python Detection & Reconstruction Core ➔ Standardized Detection Contract ➔ Dual-Head Output (Standalone 3D Visualizer + C# Revit 2025 Add-in)

---

## Executive Summary & Engineering Strategy

This roadmap details the concrete, phased engineering tasks required to stabilize, repair, and modernize the Scan-to-BIM system. The technical audit revealed that while core geometric algorithms (RANSAC, DBSCAN, normal estimation) and Revit add-in plumbing exist, critical flaws in **wall detection, merging, coordinate orientation, and data pipeline divergence** currently prevent accurate BIM generation on real-world scans.

### Critical Principles:
1. **Single Source of Truth:** A single detection data contract must feed both the **Standalone 3D Visualizer** and the **Revit Add-in**. No divergent detection logic.
2. **Arbitrary Orientation Support:** Zero assumption of world X/Y alignment. Buildings at arbitrary angles (e.g., 45° angled walls as verified in `actual_user_scan.ply`) must be detected and reconstructed as clean oriented bounding boxes and centerline paths.
3. **No Fake / Placeholder Code:** Every task includes rigorous automated regression tests and real-scan validation criteria.

---

## Phased Task Matrix

| Task ID | Component / Milestone | Priority | Risk | Dependencies |
|---------|------------------------|----------|------|--------------|
| **TASK-01** | Establish Reproducible Baseline & Test Fixtures | P0 | Low | None |
| **TASK-02** | Clean Repository Structure & Dependency Management | P0 | Low | TASK-01 |
| **TASK-03** | Unified Detection Data Model Contract | P0 | Med | TASK-02 |
| **TASK-04** | Point Cloud Preprocessing & Orientation Normalization | P1 | Med | TASK-03 |
| **TASK-05** | Overhaul Wall Detection & Merge Pipeline | P1 | High | TASK-04 |
| **TASK-06** | Overhaul Floor, Ceiling & Slab Opening Detection | P1 | Med | TASK-04 |
| **TASK-07** | Repair Doors, Windows & Secondary Architectural Elements | P1 | Med | TASK-05, TASK-06 |
| **TASK-08** | Standalone 3D Detection Visualizer (Interactive) | P1 | Med | TASK-03, TASK-05 |
| **TASK-09** | Automated Visual Validation & Diagnostics Pipeline | P2 | Low | TASK-08 |
| **TASK-10** | Harden FastAPI HTTP Service & Sidecar CLI Sync | P2 | Med | TASK-03, TASK-05 |
| **TASK-11** | Stabilize C# Revit Add-in & Sidecar Communication | P2 | Med | TASK-10 |
| **TASK-12** | Native Revit Geometry Generation Overhaul | P2 | High | TASK-05, TASK-11 |
| **TASK-13** | End-to-End Real-Scan Integration Validation | P3 | Med | TASK-12 |
| **TASK-14** | Pipeline Performance Optimization & Memory Capping | P3 | Med | TASK-13 |
| **TASK-15** | Production Documentation, Runbooks & Code Cleanup | P3 | Low | TASK-14 |

---

## Detailed Task Specifications

### [TASK-01] Establish Reproducible Baseline & Test Fixtures
- **Objective:** Lock down the test suite, ensure clean execution across all environments, and create standardized real-world scan fixtures.
- **Current State:** 628 Python tests pass, but test runs take >3 minutes due to full-scale point cloud processing. `actual_user_scan.ply` (477k points) exists in `ScanTOBIM/ScanTOBIM/artifacts/` but is missing from test fixtures.
- **Files/Modules:**
  - `ScanTOBIM/ScanTOBIM/agent/tests/conftest.py`
  - `ScanTOBIM/ScanTOBIM/agent/tests/`
  - `ScanTOBIM/ScanTOBIM/artifacts/actual_user_scan.ply`
- **Dependencies:** None
- **Implementation Approach:**
  - Standardize pytest execution to support fast unit tests (<5s) and integration fixtures.
  - Create downsampled test scans (10k and 50k points of real scan data) in `tests/fixtures/`.
  - Add pytest markers (`@pytest.mark.slow`, `@pytest.mark.integration`).
- **Acceptance Criteria:** `pytest -m "not slow"` runs in under 10 seconds; full suite passes reproducibly on macOS and Linux.
- **Tests Required:** Benchmark test runner script verifying execution time and deterministic outputs.
- **Expected Output:** Fast CI test loop with locked fixture data.
- **Risk Level:** Low

---

### [TASK-02] Clean Repository Structure & Dependency Management
- **Objective:** Eliminate directory nesting anomalies (`ScanTOBIM (2) 2/ScanTOBIM/ScanTOBIM`), resolve packaging configurations, and formalize runtime vs. dev dependencies.
- **Current State:** Repository contains nested directories, duplicate runner scripts in root and subdirectories (`run_stage2_demo.py`), legacy Windows build artifacts, and an unfrozen dependency environment where `open3d`, `numpy`, `scipy`, and `torch` have conflicting version requirements.
- **Files/Modules:**
  - `pyproject.toml`, `Makefile`, `make.ps1`
  - `agent/requirements.txt`, `wall-detection-runtime.txt`, `wall-detection-dev.txt`
- **Dependencies:** TASK-01
- **Implementation Approach:**
  - Normalize project paths so scripts can be executed seamlessly from repository root or subpackages.
  - Fix missing optional dependency handling (e.g. graceful warnings if `pye57` is missing on macOS ARM64).
  - Clean out untracked compilation caches (`.cpython-311.pyc`), stale local databases (`demo_audit.db`), and redundant spec files.
- **Acceptance Criteria:** `make test`, `make lint`, and `python -m agent.tools.processor_cli --help` execute without import errors or path hacks.
- **Tests Required:** Clean install and CLI invocation test.
- **Expected Output:** Single standardized workspace layout with clear virtualenv instructions.
- **Risk Level:** Low

---

### [TASK-03] Unified Detection Data Model Contract
- **Objective:** Standardize the detection data schema (`results.json` / `SidecarResults` / `GeometrySegment`) to eliminate divergence between the Revit add-in, the HTTP API, and the visualizer.
- **Current State:** Three divergent data models exist:
  1. `agent/models.py::GeometrySegment` (Pydantic, Python internal)
  2. `agent/tools/processor_cli.py::_build_metadata()` and `results.json` v1.0
  3. `revit-addin/Models/SidecarModels.cs::SidecarResults` (C# DTO)
  In addition, critical geometric details (oriented 2D polygon boundaries, wall pairing links, inlier point indices) are missing from the JSON contract.
- **Files/Modules:**
  - `agent/models.py`
  - `agent/tools/processor_cli.py`
  - `revit-addin/Models/SidecarModels.cs`
  - `revit-addin/Models/AgentModels.cs`
- **Dependencies:** TASK-02
- **Implementation Approach:**
  - Define Schema v2.0 for `results.json`:
    - Explicit oriented centerline endpoints (`start_xyz_mm`, `end_xyz_mm`).
    - Explicit wall thickness (`thickness_mm`), height (`height_mm`), and orientation angle (`angle_deg`).
    - Explicit polygon boundary vertices for floors/ceilings (`boundary_polygon_mm`: `[[x,y,z], ...]`).
    - Element relationship linkages (`host_id`, `paired_face_segment_id`).
    - Segment point indices or voxel point subsample for visualizer rendering.
  - Update Pydantic models in Python and C# classes in `SidecarModels.cs` to mirror the schema with zero serialization mismatch.
- **Acceptance Criteria:** Bidirectional serialization round-trip tests between Python and C# pass with zero loss of geometric precision.
- **Tests Required:** `test_schema_v2_roundtrip.py` validating full JSON serialization against C# deserializer expectations.
- **Expected Output:** Versioned, documented schema specification (`docs/DETECTION_CONTRACT_V2.md`).
- **Risk Level:** Medium

---

### [TASK-04] Point Cloud Preprocessing & Orientation Normalization
- **Objective:** Make point cloud preprocessing robust against scale confusion (mm vs. meters), survey coordinates (millions of mm offset), and building rotation relative to coordinate axes.
- **Current State:** 
  - `actual_user_scan.ply` points are in meters (-14m to +7m Z), but certain tools assume mm.
  - Survey coordinate normalization subtracts min bounds, altering the scanner origin.
  - Up-axis is hardcoded to Z, which is correct for plumb scanners, but horizontal rotation is ignored, forcing angled walls into false orthogonal bins.
- **Files/Modules:**
  - `agent/tools/scan_tools.py` (`load_point_cloud`, `downsample`, `remove_statistical_outliers`)
  - `agent/tools/detection_config.py`
- **Dependencies:** TASK-03
- **Implementation Approach:**
  - Implement automatic unit detection: scan extent heuristics (>1000 span = mm, <500 span = meters).
  - Calculate principal horizontal axes via 2D PCA on horizontal plane centroids or weighted normal orientation histogram (`infer_dominant_orientations`).
  - Store transformation matrix (`T_survey_to_local`) in metadata so all geometry can be mapped back to absolute site coordinates.
- **Acceptance Criteria:** Preprocessor accurately normalizes unit scale, preserves spatial continuity, and detects building grid rotation without altering original survey coordinates.
- **Tests Required:** Unit tests with synthetic scans in mm, meters, and offset coordinates.
- **Expected Output:** Preprocessed Open3D point cloud with explicit coordinate metadata.
- **Risk Level:** Medium

---

### [TASK-05] Overhaul Wall Detection & Merge Pipeline
- **Objective:** Fix the root causes of wall detection failure: duplicate planes, fragmentation, false-positive 4-wall envelope forcing, and failed opposing face pairing on rotated buildings.
- **Current State:**
  - In `actual_user_scan.ply`, walls are angled at ~46° and ~135°.
  - `scan_tools.py::merge_walls_hybrid()` uses rigid 5° and 1.2m storey bins that fail to merge overlapping fragments, outputting **91 duplicate walls**.
  - `orchestrator.py::_make_wall_envelope_candidates()` forces all walls into a 4-wall axis-aligned box ("west", "east", "south", "north") if >4 walls exist, destroying true geometry!
  - `agent/bim_reconstruction_fix.py::_wall_axis()` forces every wall to strictly "X" or "Y" (`return "X" if abs(ax) >= abs(ay) else "Y"`).
  - `stage2_pipeline.py` has superior Union-Find merging and curvature rejection, but is **disconnected** from `scan_tools.py` and `processor_cli.py`.
- **Files/Modules:**
  - `agent/tools/scan_tools.py` (`detect_planes`, `merge_walls`)
  - `agent/stage2_pipeline.py` (integrate as core engine)
  - `agent/orchestrator.py` (remove destructive `_make_wall_envelope_candidates`)
  - `agent/bim_reconstruction_fix.py` (remove hardcoded X/Y axis snapping)
- **Dependencies:** TASK-04
- **Implementation Approach:**
  1. Integrate the `stage2_pipeline.py` geometric feature extractor and Union-Find fragment grouping into `scan_tools.py::run_segmentation()` and `processor_cli.py`.
  2. Implement continuous angular clustering (clustering walls by normal angle modulo 180° with 8° tolerance) rather than discrete histogram bins.
  3. Pair parallel opposing wall faces dynamically with adaptive thickness search (100mm to 800mm) along the wall normal direction. If paired, create a single solid wall with true thickness; if single-faced (e.g. interior scan), assign standard estimated thickness (200mm) without creating overlapping duplicates.
  4. Delete or disable the destructive 4-wall envelope reducer in `orchestrator.py`.
  5. Allow walls to retain arbitrary 2D direction vectors `(axis_x, axis_y)` and true 3D start/end points.
- **Acceptance Criteria:** Running detection on `actual_user_scan.ply` outputs clean, non-duplicate physical walls (10-15 walls instead of 91), accurately reflecting the 46° and 136° room layout, with zero overlapping duplicate planes.
- **Tests Required:**
  - `test_angled_walls_detection.py` (verifies walls at 30°, 45°, 60° retain orientation).
  - `test_real_scan_wall_count.py` (verifies duplicate count drops on real scan).
- **Expected Output:** Accurate wall geometry dictionary with true lengths, heights, thicknesses, and orientation vectors.
- **Risk Level:** High (core algorithm change)

---

### [TASK-06] Overhaul Floor, Ceiling & Slab Opening Detection
- **Objective:** Enable accurate multi-level floor and ceiling detection with polygon boundaries, eliminating single-Z slab collapses.
- **Current State:** Floors and ceilings are detected as horizontal planes, but currently exported only as axis-aligned or 2D OBB bounding boxes (`floor_length_mm`, `floor_width_mm`). Complex floor outlines (L-shaped rooms, hallways) lose their shape.
- **Files/Modules:**
  - `agent/tools/scan_tools.py` (`detect_planes`, `merge_coplanar_horizontal`, `_detect_slab_openings`)
  - `agent/tools/floorplan_tools.py`
  - `agent/bim_reconstruction_fix.py` (`detect_levels`)
- **Dependencies:** TASK-04
- **Implementation Approach:**
  - Extract the alpha-shape / 2D concave hull of floor inliers in the horizontal plane to obtain the true boundary polygon.
  - Simplify the polygon using Douglas-Peucker (10cm tolerance) and snap edges to dominant wall axes.
  - Detect floor elevations via Z-histogram clustering to automatically discover true building levels (e.g., Basement, Ground Floor, Storey 1).
- **Acceptance Criteria:** Floor slabs accurately match scanned room perimeters; levels are clustered correctly.
- **Tests Required:** Polygon boundary validation test on synthetic L-shaped floor slab.
- **Expected Output:** Polygon boundary vertices exported in `results.json`.
- **Risk Level:** Medium

---

### [TASK-07] Repair Doors, Windows & Secondary Architectural Elements
- **Objective:** Stabilize door/window opening detection in walls and properly associate them with host wall segments.
- **Current State:** `detect_wall_openings` finds empty 2D grid runs on vertical planes, creating `SegmentShape.VOID`, but lacks host wall assignment (`host_wall_id`). In Revit, doors/windows require host elements.
- **Files/Modules:**
  - `agent/tools/scan_tools.py` (`detect_openings`, `_detect_wall_openings`)
  - `agent/classifier.py`
  - `agent/bim_reconstruction_fix.py` (`attach_doors_to_walls`)
- **Dependencies:** TASK-05, TASK-06
- **Implementation Approach:**
  - When a void is detected on a wall plane, explicitly record the wall's `segment_id` as `host_segment_id`.
  - Calculate opening insertion point and dimensions (`width`, `height`, `sill_elevation`) relative to the host wall's start coordinate.
  - Validate that doors reach the bottom level of the host wall and windows have positive sill heights.
- **Acceptance Criteria:** Openings reference their valid host wall; dimensions reflect physical voids.
- **Tests Required:** Synthetic wall with door opening test verifying host linkage.
- **Expected Output:** Opening segments with explicit host references in detection contract.
- **Risk Level:** Medium

---

### [TASK-08] Standalone 3D Detection Visualizer (Interactive)
- **Objective:** Implement a dedicated 3D visualization and inspection tool that loads the original point cloud alongside detected geometry WITHOUT requiring Revit.
- **Current State:** Only static 2D/3D PNGs are generated via matplotlib in `run_stage2_demo.py`. No interactive 3D inspection tool exists that allows layer toggling or element inspection.
- **Files/Modules:**
  - `tools/viewer3d/` (new standalone visualizer package)
  - `tools/viewer3d/viewer.py`
  - `agent/tools/processor_cli.py` (CLI flag `--visualize`)
- **Dependencies:** TASK-03, TASK-05
- **Implementation Approach:**
  - Dual implementation strategy:
    1. **Native Open3D Interactive Window:** Uses `open3d.visualization.O3DVisualizer` (already installed). Supports toggling layers (Raw Cloud, Downsampled, Wall Solids, Floor Slabs, Voids, MEP), colored by confidence or element type, with element picking/inspection.
    2. **Web-Based Three.js Viewer:** Embedded in `frontend/viewer.html` served by FastAPI. Reads `results.json` + downsampled PLY, allowing inspection directly in any web browser on macOS, Windows, and Linux.
  - **CRITICAL:** Both viewers MUST load the EXACT `results.json` generated by `processor_cli.py` (the identical file that Revit consumes).
- **Acceptance Criteria:**
  - Visualizer renders `actual_user_scan.ply` with colored overlay of detected walls, floors, and openings.
  - User can toggle visibility of point cloud vs. detected BIM solids.
  - Clicking on a wall displays its ID, length, height, thickness, orientation, and confidence.
- **Tests Required:** Headless render verification and viewer invocation smoke tests.
- **Expected Output:** `python run_viewer.py --input scan.ply --results results.json` displays the full interactive 3D model.
- **Risk Level:** Medium

---

### [TASK-09] Automated Visual Validation & Diagnostics Pipeline
- **Objective:** Generate automated, high-resolution diagnostic report images (Plan view overlay, 3D perspective, Elevation, Section 16 table) during every pipeline run.
- **Current State:** `run_stage2_demo.py` generates debug PNGs, but this is not integrated into `processor_cli.py` or the FastAPI backend.
- **Files/Modules:**
  - `agent/tools/diagnostic_visuals.py` (refactored from `run_stage2_demo.py`)
  - `agent/tools/processor_cli.py`
- **Dependencies:** TASK-08
- **Implementation Approach:**
  - Extract the 2D plan and elevation rendering code into a zero-display matplotlib module.
  - Add `--render-diagnostics <dir>` to `processor_cli.py`.
  - Automatically produce plan view overlays showing wall centerlines atop the point cloud density slice.
- **Acceptance Criteria:** Every CLI and HTTP processing run optionally saves diagnostic images showing exactly how detected walls align with raw points.
- **Tests Required:** Test verifying generation of valid PNGs from synthetic and real scan data.
- **Expected Output:** Diagnostic PNG bundle (`01_plan_overlay.png`, `02_3d_overlay.png`, etc.).
- **Risk Level:** Low

---

### [TASK-10] Harden FastAPI HTTP Service & Sidecar CLI Sync
- **Objective:** Ensure the HTTP API (`agent/main.py`) and the CLI sidecar (`processor_cli.py`) execute the exact same detection pipeline with identical parameters and outputs.
- **Current State:** `main.py::_run_pipeline_background` calls `orchestrator.run_segmentation()` which previously branched differently from `processor_cli.py`.
- **Files/Modules:**
  - `agent/main.py`
  - `agent/orchestrator.py`
  - `agent/tools/processor_cli.py`
- **Dependencies:** TASK-03, TASK-05
- **Implementation Approach:**
  - Refactor `orchestrator.run_segmentation` to directly invoke the updated detection pipeline from TASK-05.
  - Ensure progress reporting (`PROGRESS:N:message`) and error handling are synchronized.
  - Verify endpoints: `POST /sessions/{id}/upload`, `POST /sessions/{id}/ingest-elements`, `GET /jobs/{id}`.
- **Acceptance Criteria:** Running detection via HTTP upload produces byte-for-byte compatible segments with the CLI tool.
- **Tests Required:** `test_api.py` and `test_processor_cli.py` integration tests.
- **Expected Output:** Unified, robust processing engine accessible via CLI and HTTP.
- **Risk Level:** Medium

---

### [TASK-11] Stabilize C# Revit Add-in & Sidecar Communication
- **Objective:** Fix sidecar discovery, process execution, and results parsing in `ScanToBIMAgent` C# codebase.
- **Current State:**
  - `ProcessScanCommand.cs` has hardcoded paths to local directories (`C:\Users\RKAA6083\...`).
  - Sidecar path discovery has 5 priority levels, but error handling on crash/timeout needs improvement.
- **Files/Modules:**
  - `revit-addin/Commands/ProcessScanCommand.cs`
  - `revit-addin/Services/ScanResultStore.cs`
  - `revit-addin/Models/SidecarModels.cs`
- **Dependencies:** TASK-10
- **Implementation Approach:**
  - Replace all hardcoded author paths with standard environment variable overrides and `Path.Combine(Environment.GetFolderPath(SpecialFolder.LocalApplicationData), "ScanToBIM")`.
  - Add timeout management (kill sidecar if hung >15 minutes on massive scans).
  - Validate JSON schema version during deserialization in C#; display friendly error dialog if schema is mismatched.
- **Acceptance Criteria:** C# addin runs sidecar reliably across different user profiles without path errors.
- **Tests Required:** C# xUnit tests for `SidecarModels` deserialization and command initialization.
- **Expected Output:** Clean, robust C# communication service.
- **Risk Level:** Medium

---

### [TASK-12] Native Revit Geometry Generation Overhaul
- **Objective:** Fix the "Can't keep elements joined" and "Highlighted walls overlap" errors in Revit by generating clean, joined native walls and direct shapes.
- **Current State:** `Project1_Error Report.html` documents dozens of Revit errors because `BuildGeometryCommand` / `ScanGeometryBuilder` attempts to create walls from overlapping or slightly off-axis lines, with a default 750mm concrete wall type (`Wall-Fnd_750Con_Footing`).
- **Files/Modules:**
  - `revit-addin/Services/ScanGeometryBuilder.cs`
  - `revit-addin/Services/RevitElementFactory.cs`
  - `revit-addin/Commands/BuildGeometryCommand.cs`
  - `revit-addin/Services/GeometryUtils.cs`
- **Dependencies:** TASK-05, TASK-11
- **Implementation Approach:**
  1. Use cleaned wall centerlines from Schema v2 (TASK-05), where duplicate and overlapping parallel walls have already been merged.
  2. In `RevitElementFactory.cs::CreateWall`:
     - Select a standard wall type matching the detected `wall_thickness_mm` (e.g. Generic 200mm rather than 750mm foundation footing).
     - Trim or extend intersecting wall endpoints at corners so lines meet cleanly at a single vertex.
     - Enforce minimum wall length threshold (>= 300mm) to prevent micro-segment creation.
     - Temporarily disable automatic join (`Wall.Create(..., flip=false, structural=false)`) or use `WallUtils.DisallowWallJoinAtEnd` if walls are nearly collinear to prevent Revit join transaction crashes.
  3. For floors: use `boundary_polygon_mm` to build a clean `CurveLoop` for `Floor.Create`.
- **Acceptance Criteria:** Building geometry in Revit succeeds with zero fatal errors, minimal non-fatal warnings, and clean 3D wall joins.
- **Tests Required:** Manual Revit test protocol (`REVIT_TEST_GUIDE.md`) + xUnit mock geometry builder tests.
- **Expected Output:** Production-quality Revit model generated from scan data.
- **Risk Level:** High (Revit API nuances)

---

### [TASK-13] End-to-End Real-Scan Integration Validation
- **Objective:** Execute the entire pipeline from `actual_user_scan.ply` through Python detection, standalone 3D visualization, and Revit geometry generation.
- **Current State:** Pipeline has only been run piecewise or on synthetic benchmarks.
- **Files/Modules:** Entire system.
- **Dependencies:** TASK-08, TASK-12
- **Implementation Approach:**
  - Run `processor_cli.py` on `actual_user_scan.ply`.
  - Validate in Standalone 3D Visualizer: inspect detected walls against the point cloud.
  - Ingest `results.json` into Revit and trigger "Build Geometry".
  - Inspect generated Revit walls, floors, and levels for geometric fidelity and zero element joining errors.
- **Acceptance Criteria:** Real scan produces accurate 3D BIM model in Revit and identical visualization in standalone viewer.
- **Tests Required:** End-to-end integration test script.
- **Expected Output:** Validated BIM project and signed off QA report.
- **Risk Level:** Medium

---

### [TASK-14] Pipeline Performance Optimization & Memory Capping
- **Objective:** Optimize runtime and memory usage for 10M–50M point industrial scans.
- **Current State:** Memory usage scales linearly with raw points; Statistical Outlier Removal (SOR) on full-resolution cloud is O(N log N).
- **Files/Modules:**
  - `agent/tools/scan_tools.py`
  - `agent/tools/detection_config.py`
- **Dependencies:** TASK-13
- **Implementation Approach:**
  - Swap order when scan size > 5M points: voxel downsample first to 30mm, then run SOR on downsampled points.
  - Parallelize RANSAC plane searches using Open3D multi-threaded kernels where applicable.
  - Implement streaming memory buffers for large E57 / LAZ files.
- **Acceptance Criteria:** 10M point scan processes in < 60 seconds on standard workstation.
- **Tests Required:** Benchmark performance tests.
- **Expected Output:** High-performance processing profile.
- **Risk Level:** Medium

---

### [TASK-15] Production Documentation, Runbooks & Code Cleanup
- **Objective:** Complete comprehensive technical documentation, deployment guides, and code cleanup.
- **Current State:** Documentation exists in `ScanTOBIM/ScanTOBIM/docs`, but contains outdated references and legacy architectural notes.
- **Files/Modules:**
  - `docs/`
  - `README.md`, `CHANGELOG.md`
- **Dependencies:** TASK-14
- **Implementation Approach:**
  - Update user manuals, operator guides, and API documentation.
  - Document the Standalone 3D Visualizer usage.
  - Remove all dead code, debug print statements, and deprecated experiment scripts.
- **Acceptance Criteria:** Clean, documented codebase with updated CHANGELOG and user guides.
- **Tests Required:** Documentation link checking and linters.
- **Expected Output:** Production-ready Scan-to-BIM release package.
- **Risk Level:** Low
