# Repository Inventory — ScanTOBIM

## 1. Executive Summary

- **Total Ingested Files**: 252
- **Scope of Audit**: Complete repository tree across Python backend, C# Revit Add-in, web frontend, configuration, build tooling, test suites, and generated artifacts.

### Classification by Lifecycle Status

| Status | File Count | Percentage | Description |
|---|---|---|---|
| **PRODUCTION** | 106 | 42.1% | Active source code executed during pipeline inference and Revit building |
| **GENERATED** | 81 | 32.1% | Compiled binaries (.dll, .pdb), intermediate compiler cache, build logs, and prior execution reports |
| **TEST** | 43 | 17.1% | Automated unit, regression, integration, and manual testing suites |
| **DOCUMENTATION** | 21 | 8.3% | Architectural PRDs, roadmaps, implementation guides, HTML documentation |
| **FIXTURE** | 1 | 0.4% | Local binary weights (.pt) and graphic plan overlay assets |

### Classification by Functional Architecture Role

| Functional Role | File Count | Primary Technologies |
|---|---|---|
| **Unit / Integration Test Suite** | 41 |
| **Compiled Revit Add-in Binary / MSBuild Intermediate Artifact** | 32 |
| **Generated Audit, Benchmark & Validation Report** | 25 |
| **Unknown** | 23 |
| **Technical Documentation & Architectural Specification** | 15 |
| **Enterprise Quality, NQA-1 Provenance & Safety Gate** | 14 |
| **PyInstaller Build Log / Intermediate Output** | 14 |
| **Build System / Tooling / Environment Configuration** | 12 |
| **Standalone Pipeline Script / Demo Runner** | 8 |
| **Web Dashboard / Frontend UI** | 8 |
| **Revit Add-in Infrastructure & Bootstrap** | 8 |
| **AI / Neural / Algorithmic Model Adapter** | 7 |
| **Core Agent Pipeline Orchestrator & CLI** | 5 |
| **Revit WPF Dockable Panel UI** | 5 |
| **Semantic Classifier & Typology Inference** | 4 |
| **Geometric Feature Detection & Shape Fitting** | 4 |
| **Revit Shared Parameter Definitions** | 4 |
| **Revit Ribbon UI & External Command Handler** | 4 |
| **Point-Cloud Loader & Multi-Resolution Preprocessing** | 3 |
| **Stage 2 Wall Grouping & Thickness Inference** | 2 |
| **Coordinate Transformation / Unit Mapping** | 2 |
| **Revit Family & Category Mapping Configuration** | 2 |
| **Revit Add-in Data Transfer Models** | 2 |
| **Revit Family & Type Resolution Service** | 2 |
| **Pretrained Neural Model Weights (YOLOv8n)** | 1 |
| **BIM Reconstruction & Level Elevation Inference** | 1 |
| **Central Detection Thresholds & Parameters** | 1 |
| **Revit Safety Gate Service** | 1 |
| **Revit Native Element Factory (Wall/Floor/Column/Opening)** | 1 |
| **Revit DirectShape & Custom Geometry Builder** | 1 |

## 2. Granular Repository Inventory Table

| # | Relative Path | File Type | Size | Status | Functional Role |
|---|---|---|---|---|---|
| 1 | `.gitignore` | `(no ext)` | 515 B | `PRODUCTION` | Build System / Tooling / Environment Configuration |
| 2 | `ScanToBIMAgent.csproj` | `.csproj` | 1,430 B | `PRODUCTION` | Build System / Tooling / Environment Configuration |
| 3 | `run_stage2_demo.py` | `.py` | 349 B | `PRODUCTION` | Standalone Pipeline Script / Demo Runner |
| 4 | `test_plan_overlay.png` | `.png` | 249,957 B | `TEST` | Unit / Integration Test Suite |
| 5 | `yolov8n.pt` | `.pt` | 6,549,796 B | `FIXTURE` | Pretrained Neural Model Weights (YOLOv8n) |
| 6 | `artifacts/FINAL_DETECTION_AUDIT.json` | `.json` | 1,506,243 B | `GENERATED` | Generated Audit, Benchmark & Validation Report |
| 7 | `artifacts/FINAL_DETECTION_AUDIT.md` | `.md` | 3,151 B | `GENERATED` | Generated Audit, Benchmark & Validation Report |
| 8 | `artifacts/POINT_CLOUD_RESOLUTION_AUDIT.json` | `.json` | 1,770 B | `GENERATED` | Generated Audit, Benchmark & Validation Report |
| 9 | `artifacts/POINT_CLOUD_RESOLUTION_AUDIT.md` | `.md` | 1,840 B | `GENERATED` | Generated Audit, Benchmark & Validation Report |
| 10 | `artifacts/results_real_e57_fusion.json` | `.json` | 263,439 B | `GENERATED` | Generated Audit, Benchmark & Validation Report |
| 11 | `artifacts/scan-2026-07-28-2-20260922-115720.json` | `.json` | 1,501,558 B | `GENERATED` | Generated Audit, Benchmark & Validation Report |
| 12 | `docs/tasks/PRD.md` | `.md` | 9,899 B | `DOCUMENTATION` | Technical Documentation & Architectural Specification |
| 13 | `docs/tasks/progress.txt` | `.txt` | 0 B | `DOCUMENTATION` | Technical Documentation & Architectural Specification |
| 14 | `docs/tasks/prompt.md` | `.md` | 14,206 B | `DOCUMENTATION` | Technical Documentation & Architectural Specification |
| 15 | `ScanTOBIM/ScanTOBIM/CHANGELOG.md` | `.md` | 10,825 B | `DOCUMENTATION` | Technical Documentation & Architectural Specification |
| 16 | `ScanTOBIM/ScanTOBIM/Caddyfile` | `(no ext)` | 3,351 B | `PRODUCTION` | Build System / Tooling / Environment Configuration |
| 17 | `ScanTOBIM/ScanTOBIM/Makefile` | `(no ext)` | 7,794 B | `PRODUCTION` | Build System / Tooling / Environment Configuration |
| 18 | `ScanTOBIM/ScanTOBIM/SCANTOBIM_V10.2.2.sln` | `.sln` | 2,400 B | `PRODUCTION` | Build System / Tooling / Environment Configuration |
| 19 | `ScanTOBIM/ScanTOBIM/check_wall_detection.py` | `.py` | 1,360 B | `PRODUCTION` | Standalone Pipeline Script / Demo Runner |
| 20 | `ScanTOBIM/ScanTOBIM/demo.py` | `.py` | 7,474 B | `PRODUCTION` | Standalone Pipeline Script / Demo Runner |
| 21 | `ScanTOBIM/ScanTOBIM/demo_runner.py` | `.py` | 13,072 B | `PRODUCTION` | Standalone Pipeline Script / Demo Runner |
| 22 | `ScanTOBIM/ScanTOBIM/make.ps1` | `.ps1` | 10,333 B | `PRODUCTION` | Build System / Tooling / Environment Configuration |
| 23 | `ScanTOBIM/ScanTOBIM/multi_pcd_wall_pipeline.py` | `.py` | 18,107 B | `PRODUCTION` | Standalone Pipeline Script / Demo Runner |
| 24 | `ScanTOBIM/ScanTOBIM/run_stage2_demo.py` | `.py` | 29,670 B | `PRODUCTION` | Standalone Pipeline Script / Demo Runner |
| 25 | `ScanTOBIM/ScanTOBIM/stb-processor.spec` | `.spec` | 1,846 B | `PRODUCTION` | Build System / Tooling / Environment Configuration |
| 26 | `ScanTOBIM/ScanTOBIM/test_wall_detection.py` | `.py` | 4,638 B | `TEST` | Unit / Integration Test Suite |
| 27 | `ScanTOBIM/ScanTOBIM/wall-detection-runtime.txt` | `.txt` | 393 B | `PRODUCTION` | Unknown |
| 28 | `ScanTOBIM/ScanTOBIM/frontend/audit.html` | `.html` | 13,777 B | `DOCUMENTATION` | Web Dashboard / Frontend UI |
| 29 | `ScanTOBIM/ScanTOBIM/frontend/dashboard.html` | `.html` | 14,674 B | `DOCUMENTATION` | Web Dashboard / Frontend UI |
| 30 | `ScanTOBIM/ScanTOBIM/frontend/gates.html` | `.html` | 18,237 B | `DOCUMENTATION` | Web Dashboard / Frontend UI |
| 31 | `ScanTOBIM/ScanTOBIM/frontend/login.html` | `.html` | 4,084 B | `DOCUMENTATION` | Web Dashboard / Frontend UI |
| 32 | `ScanTOBIM/ScanTOBIM/frontend/ncrs.html` | `.html` | 17,837 B | `DOCUMENTATION` | Web Dashboard / Frontend UI |
| 33 | `ScanTOBIM/ScanTOBIM/frontend/session.html` | `.html` | 36,267 B | `DOCUMENTATION` | Web Dashboard / Frontend UI |
| 34 | `ScanTOBIM/ScanTOBIM/frontend/sessions.html` | `.html` | 15,285 B | `DOCUMENTATION` | Web Dashboard / Frontend UI |
| 35 | `ScanTOBIM/ScanTOBIM/frontend/js/api.js` | `.js` | 15,521 B | `PRODUCTION` | Web Dashboard / Frontend UI |
| 36 | `ScanTOBIM/ScanTOBIM/agent/Dockerfile` | `(no ext)` | 4,090 B | `PRODUCTION` | Build System / Tooling / Environment Configuration |
| 37 | `ScanTOBIM/ScanTOBIM/agent/__init__.py` | `.py` | 0 B | `PRODUCTION` | Unknown |
| 38 | `ScanTOBIM/ScanTOBIM/agent/audit.py` | `.py` | 7,048 B | `PRODUCTION` | Enterprise Quality, NQA-1 Provenance & Safety Gate |
| 39 | `ScanTOBIM/ScanTOBIM/agent/auth.py` | `.py` | 8,668 B | `PRODUCTION` | Enterprise Quality, NQA-1 Provenance & Safety Gate |
| 40 | `ScanTOBIM/ScanTOBIM/agent/bim_reconstruction_fix.py` | `.py` | 74,027 B | `PRODUCTION` | BIM Reconstruction & Level Elevation Inference |
| 41 | `ScanTOBIM/ScanTOBIM/agent/cde_state.py` | `.py` | 8,418 B | `PRODUCTION` | Enterprise Quality, NQA-1 Provenance & Safety Gate |
| 42 | `ScanTOBIM/ScanTOBIM/agent/classifier.py` | `.py` | 78,078 B | `PRODUCTION` | Semantic Classifier & Typology Inference |
| 43 | `ScanTOBIM/ScanTOBIM/agent/determinism.py` | `.py` | 6,895 B | `PRODUCTION` | Enterprise Quality, NQA-1 Provenance & Safety Gate |
| 44 | `ScanTOBIM/ScanTOBIM/agent/deviation_check.py` | `.py` | 6,171 B | `PRODUCTION` | Enterprise Quality, NQA-1 Provenance & Safety Gate |
| 45 | `ScanTOBIM/ScanTOBIM/agent/drp.py` | `.py` | 15,437 B | `PRODUCTION` | Enterprise Quality, NQA-1 Provenance & Safety Gate |
| 46 | `ScanTOBIM/ScanTOBIM/agent/element_type_rules.json` | `.json` | 2,574 B | `PRODUCTION` | Semantic Classifier & Typology Inference |
| 47 | `ScanTOBIM/ScanTOBIM/agent/licensing.py` | `.py` | 4,875 B | `PRODUCTION` | Enterprise Quality, NQA-1 Provenance & Safety Gate |
| 48 | `ScanTOBIM/ScanTOBIM/agent/main.py` | `.py` | 79,068 B | `PRODUCTION` | Core Agent Pipeline Orchestrator & CLI |
| 49 | `ScanTOBIM/ScanTOBIM/agent/models.py` | `.py` | 31,728 B | `PRODUCTION` | Enterprise Quality, NQA-1 Provenance & Safety Gate |
| 50 | `ScanTOBIM/ScanTOBIM/agent/ncr.py` | `.py` | 6,743 B | `PRODUCTION` | Enterprise Quality, NQA-1 Provenance & Safety Gate |
| 51 | `ScanTOBIM/ScanTOBIM/agent/nqa1_signature.py` | `.py` | 13,225 B | `PRODUCTION` | Enterprise Quality, NQA-1 Provenance & Safety Gate |
| 52 | `ScanTOBIM/ScanTOBIM/agent/orchestrator.py` | `.py` | 47,634 B | `PRODUCTION` | Core Agent Pipeline Orchestrator & CLI |
| 53 | `ScanTOBIM/ScanTOBIM/agent/pyproject.toml` | `.toml` | 993 B | `PRODUCTION` | Build System / Tooling / Environment Configuration |
| 54 | `ScanTOBIM/ScanTOBIM/agent/report.py` | `.py` | 15,077 B | `PRODUCTION` | Enterprise Quality, NQA-1 Provenance & Safety Gate |
| 55 | `ScanTOBIM/ScanTOBIM/agent/requirements.txt` | `.txt` | 566 B | `PRODUCTION` | Build System / Tooling / Environment Configuration |
| 56 | `ScanTOBIM/ScanTOBIM/agent/revit_bridge.py` | `.py` | 11,426 B | `PRODUCTION` | Core Agent Pipeline Orchestrator & CLI |
| 57 | `ScanTOBIM/ScanTOBIM/agent/safety_gate.py` | `.py` | 10,325 B | `PRODUCTION` | Enterprise Quality, NQA-1 Provenance & Safety Gate |
| 58 | `ScanTOBIM/ScanTOBIM/agent/segment_merger.py` | `.py` | 6,954 B | `PRODUCTION` | Enterprise Quality, NQA-1 Provenance & Safety Gate |
| 59 | `ScanTOBIM/ScanTOBIM/agent/software_identity.py` | `.py` | 5,009 B | `PRODUCTION` | Enterprise Quality, NQA-1 Provenance & Safety Gate |
| 60 | `ScanTOBIM/ScanTOBIM/agent/stage2_pipeline.py` | `.py` | 53,766 B | `PRODUCTION` | Stage 2 Wall Grouping & Thickness Inference |
| 61 | `ScanTOBIM/ScanTOBIM/agent/stb-processor.spec` | `.spec` | 2,045 B | `PRODUCTION` | Build System / Tooling / Environment Configuration |
| 62 | `ScanTOBIM/ScanTOBIM/agent/worker.py` | `.py` | 9,318 B | `PRODUCTION` | Core Agent Pipeline Orchestrator & CLI |
| 63 | `ScanTOBIM/ScanTOBIM/agent/tools/__init__.py` | `.py` | 0 B | `PRODUCTION` | Unknown |
| 64 | `ScanTOBIM/ScanTOBIM/agent/tools/build_semantic_checkpoint.py` | `.py` | 4,320 B | `PRODUCTION` | Standalone Pipeline Script / Demo Runner |
| 65 | `ScanTOBIM/ScanTOBIM/agent/tools/checker_tools.py` | `.py` | 9,310 B | `PRODUCTION` | Standalone Pipeline Script / Demo Runner |
| 66 | `ScanTOBIM/ScanTOBIM/agent/tools/column_tools.py` | `.py` | 8,055 B | `PRODUCTION` | Geometric Feature Detection & Shape Fitting |
| 67 | `ScanTOBIM/ScanTOBIM/agent/tools/coordinate_system.py` | `.py` | 4,681 B | `PRODUCTION` | Coordinate Transformation / Unit Mapping |
| 68 | `ScanTOBIM/ScanTOBIM/agent/tools/coordinator_tools.py` | `.py` | 13,497 B | `PRODUCTION` | Unknown |
| 69 | `ScanTOBIM/ScanTOBIM/agent/tools/detection_config.py` | `.py` | 3,424 B | `PRODUCTION` | Central Detection Thresholds & Parameters |
| 70 | `ScanTOBIM/ScanTOBIM/agent/tools/deviation_heatmap.py` | `.py` | 10,897 B | `PRODUCTION` | Unknown |
| 71 | `ScanTOBIM/ScanTOBIM/agent/tools/deviation_tools.py` | `.py` | 12,940 B | `PRODUCTION` | Unknown |
| 72 | `ScanTOBIM/ScanTOBIM/agent/tools/discipline_tools.py` | `.py` | 20,188 B | `PRODUCTION` | Unknown |
| 73 | `ScanTOBIM/ScanTOBIM/agent/tools/export_semantic_points.py` | `.py` | 2,805 B | `PRODUCTION` | Unknown |
| 74 | `ScanTOBIM/ScanTOBIM/agent/tools/floorplan_tools.py` | `.py` | 14,758 B | `PRODUCTION` | Unknown |
| 75 | `ScanTOBIM/ScanTOBIM/agent/tools/fmea.py` | `.py` | 27,915 B | `PRODUCTION` | Unknown |
| 76 | `ScanTOBIM/ScanTOBIM/agent/tools/geometry_tools.py` | `.py` | 42,681 B | `PRODUCTION` | Geometric Feature Detection & Shape Fitting |
| 77 | `ScanTOBIM/ScanTOBIM/agent/tools/handover_bundle.py` | `.py` | 19,275 B | `PRODUCTION` | Unknown |
| 78 | `ScanTOBIM/ScanTOBIM/agent/tools/ifc_export.py` | `.py` | 12,900 B | `PRODUCTION` | Unknown |
| 79 | `ScanTOBIM/ScanTOBIM/agent/tools/iso15926_tools.py` | `.py` | 16,099 B | `PRODUCTION` | Unknown |
| 80 | `ScanTOBIM/ScanTOBIM/agent/tools/loa_tools.py` | `.py` | 7,872 B | `PRODUCTION` | Unknown |
| 81 | `ScanTOBIM/ScanTOBIM/agent/tools/multi_res_pcd.py` | `.py` | 20,769 B | `PRODUCTION` | Point-Cloud Loader & Multi-Resolution Preprocessing |
| 82 | `ScanTOBIM/ScanTOBIM/agent/tools/nqa1_package.py` | `.py` | 23,964 B | `PRODUCTION` | Unknown |
| 83 | `ScanTOBIM/ScanTOBIM/agent/tools/open3d_viewer.py` | `.py` | 24,523 B | `PRODUCTION` | Point-Cloud Loader & Multi-Resolution Preprocessing |
| 84 | `ScanTOBIM/ScanTOBIM/agent/tools/opening_tools.py` | `.py` | 6,014 B | `PRODUCTION` | Geometric Feature Detection & Shape Fitting |
| 85 | `ScanTOBIM/ScanTOBIM/agent/tools/processor_cli.py` | `.py` | 61,919 B | `PRODUCTION` | Core Agent Pipeline Orchestrator & CLI |
| 86 | `ScanTOBIM/ScanTOBIM/agent/tools/randla_net.py` | `.py` | 17,288 B | `PRODUCTION` | Unknown |
| 87 | `ScanTOBIM/ScanTOBIM/agent/tools/registration_report.py` | `.py` | 10,747 B | `PRODUCTION` | Unknown |
| 88 | `ScanTOBIM/ScanTOBIM/agent/tools/registration_tools.py` | `.py` | 11,546 B | `PRODUCTION` | Unknown |
| 89 | `ScanTOBIM/ScanTOBIM/agent/tools/requirements_matrix.py` | `.py` | 14,424 B | `PRODUCTION` | Unknown |
| 90 | `ScanTOBIM/ScanTOBIM/agent/tools/scan_tools.py` | `.py` | 202,741 B | `PRODUCTION` | Point-Cloud Loader & Multi-Resolution Preprocessing |
| 91 | `ScanTOBIM/ScanTOBIM/agent/tools/semantic_models.py` | `.py` | 8,259 B | `PRODUCTION` | Semantic Classifier & Typology Inference |
| 92 | `ScanTOBIM/ScanTOBIM/agent/tools/slab_tools.py` | `.py` | 10,243 B | `PRODUCTION` | Geometric Feature Detection & Shape Fitting |
| 93 | `ScanTOBIM/ScanTOBIM/agent/tools/survey_tools.py` | `.py` | 9,230 B | `PRODUCTION` | Unknown |
| 94 | `ScanTOBIM/ScanTOBIM/agent/tools/validate_run.py` | `.py` | 3,761 B | `PRODUCTION` | Unknown |
| 95 | `ScanTOBIM/ScanTOBIM/agent/tools/wall_classifier.py` | `.py` | 11,254 B | `PRODUCTION` | Semantic Classifier & Typology Inference |
| 96 | `ScanTOBIM/ScanTOBIM/agent/tools/__pycache__/__init__.cpython-310.pyc` | `.pyc` | 198 B | `GENERATED` | Unknown |
| 97 | `ScanTOBIM/ScanTOBIM/agent/tools/__pycache__/processor_cli.cpython-310.pyc` | `.pyc` | 35,751 B | `GENERATED` | Unknown |
| 98 | `ScanTOBIM/ScanTOBIM/agent/tests/__init__.py` | `.py` | 0 B | `TEST` | Unit / Integration Test Suite |
| 99 | `ScanTOBIM/ScanTOBIM/agent/tests/audit_verifier.py` | `.py` | 3,595 B | `TEST` | Unit / Integration Test Suite |
| 100 | `ScanTOBIM/ScanTOBIM/agent/tests/conftest.py` | `.py` | 6,539 B | `TEST` | Unit / Integration Test Suite |
| 101 | `ScanTOBIM/ScanTOBIM/agent/tests/test_api.py` | `.py` | 13,097 B | `TEST` | Unit / Integration Test Suite |
| 102 | `ScanTOBIM/ScanTOBIM/agent/tests/test_bim_reconstruction_stage.py` | `.py` | 14,424 B | `TEST` | Unit / Integration Test Suite |
| 103 | `ScanTOBIM/ScanTOBIM/agent/tests/test_cde_state.py` | `.py` | 7,261 B | `TEST` | Unit / Integration Test Suite |
| 104 | `ScanTOBIM/ScanTOBIM/agent/tests/test_classifier_internal.py` | `.py` | 4,542 B | `TEST` | Unit / Integration Test Suite |
| 105 | `ScanTOBIM/ScanTOBIM/agent/tests/test_determinism.py` | `.py` | 16,083 B | `TEST` | Unit / Integration Test Suite |
| 106 | `ScanTOBIM/ScanTOBIM/agent/tests/test_deviation.py` | `.py` | 13,362 B | `TEST` | Unit / Integration Test Suite |
| 107 | `ScanTOBIM/ScanTOBIM/agent/tests/test_deviation_heatmap.py` | `.py` | 14,393 B | `TEST` | Unit / Integration Test Suite |
| 108 | `ScanTOBIM/ScanTOBIM/agent/tests/test_floorplan.py` | `.py` | 15,642 B | `TEST` | Unit / Integration Test Suite |
| 109 | `ScanTOBIM/ScanTOBIM/agent/tests/test_fmea.py` | `.py` | 11,701 B | `TEST` | Unit / Integration Test Suite |
| 110 | `ScanTOBIM/ScanTOBIM/agent/tests/test_handover_bundle.py` | `.py` | 19,413 B | `TEST` | Unit / Integration Test Suite |
| 111 | `ScanTOBIM/ScanTOBIM/agent/tests/test_hybrid_fusion.py` | `.py` | 7,845 B | `TEST` | Unit / Integration Test Suite |
| 112 | `ScanTOBIM/ScanTOBIM/agent/tests/test_ingest.py` | `.py` | 18,079 B | `TEST` | Unit / Integration Test Suite |
| 113 | `ScanTOBIM/ScanTOBIM/agent/tests/test_iso15926.py` | `.py` | 13,477 B | `TEST` | Unit / Integration Test Suite |
| 114 | `ScanTOBIM/ScanTOBIM/agent/tests/test_loa.py` | `.py` | 16,109 B | `TEST` | Unit / Integration Test Suite |
| 115 | `ScanTOBIM/ScanTOBIM/agent/tests/test_ncr.py` | `.py` | 5,227 B | `TEST` | Unit / Integration Test Suite |
| 116 | `ScanTOBIM/ScanTOBIM/agent/tests/test_nqa1_package.py` | `.py` | 15,633 B | `TEST` | Unit / Integration Test Suite |
| 117 | `ScanTOBIM/ScanTOBIM/agent/tests/test_nqa1_signature.py` | `.py` | 24,241 B | `TEST` | Unit / Integration Test Suite |
| 118 | `ScanTOBIM/ScanTOBIM/agent/tests/test_orchestrator_priority.py` | `.py` | 5,976 B | `TEST` | Unit / Integration Test Suite |
| 119 | `ScanTOBIM/ScanTOBIM/agent/tests/test_overlap.py` | `.py` | 19,485 B | `TEST` | Unit / Integration Test Suite |
| 120 | `ScanTOBIM/ScanTOBIM/agent/tests/test_processor_cli.py` | `.py` | 16,977 B | `TEST` | Unit / Integration Test Suite |
| 121 | `ScanTOBIM/ScanTOBIM/agent/tests/test_reconstruction_pipeline.py` | `.py` | 11,298 B | `TEST` | Unit / Integration Test Suite |
| 122 | `ScanTOBIM/ScanTOBIM/agent/tests/test_registration.py` | `.py` | 10,036 B | `TEST` | Unit / Integration Test Suite |
| 123 | `ScanTOBIM/ScanTOBIM/agent/tests/test_registration_report.py` | `.py` | 14,563 B | `TEST` | Unit / Integration Test Suite |
| 124 | `ScanTOBIM/ScanTOBIM/agent/tests/test_revit_bridge.py` | `.py` | 30,875 B | `TEST` | Unit / Integration Test Suite |
| 125 | `ScanTOBIM/ScanTOBIM/agent/tests/test_revit_wall_contract.py` | `.py` | 4,282 B | `TEST` | Unit / Integration Test Suite |
| 126 | `ScanTOBIM/ScanTOBIM/agent/tests/test_semantic_models.py` | `.py` | 1,534 B | `TEST` | Unit / Integration Test Suite |
| 127 | `ScanTOBIM/ScanTOBIM/agent/tests/test_sprint3.py` | `.py` | 16,304 B | `TEST` | Unit / Integration Test Suite |
| 128 | `ScanTOBIM/ScanTOBIM/agent/tests/test_stage2_adversarial_failures.py` | `.py` | 14,155 B | `TEST` | Unit / Integration Test Suite |
| 129 | `ScanTOBIM/ScanTOBIM/agent/tests/test_stage2_core.py` | `.py` | 9,640 B | `TEST` | Unit / Integration Test Suite |
| 130 | `ScanTOBIM/ScanTOBIM/agent/tests/test_stage2_pipeline.py` | `.py` | 4,224 B | `TEST` | Stage 2 Wall Grouping & Thickness Inference |
| 131 | `ScanTOBIM/ScanTOBIM/agent/tests/test_synthetic_profile.py` | `.py` | 3,146 B | `TEST` | Unit / Integration Test Suite |
| 132 | `ScanTOBIM/ScanTOBIM/agent/tests/test_upload.py` | `.py` | 12,125 B | `TEST` | Unit / Integration Test Suite |
| 133 | `ScanTOBIM/ScanTOBIM/agent/tests/test_utf8_pipeline.py` | `.py` | 2,548 B | `TEST` | Unit / Integration Test Suite |
| 134 | `ScanTOBIM/ScanTOBIM/agent/tests/test_wall_axis.py` | `.py` | 11,794 B | `TEST` | Unit / Integration Test Suite |
| 135 | `ScanTOBIM/ScanTOBIM/agent/tests/integration/conftest.py` | `.py` | 600 B | `TEST` | Unit / Integration Test Suite |
| 136 | `ScanTOBIM/ScanTOBIM/agent/tests/integration/test_stage2_ransac.py` | `.py` | 2,491 B | `TEST` | Unit / Integration Test Suite |
| 137 | `ScanTOBIM/ScanTOBIM/agent/tests/integration/test_wall_axis_real.py` | `.py` | 10,293 B | `TEST` | Unit / Integration Test Suite |
| 138 | `ScanTOBIM/ScanTOBIM/agent/adapters/__init__.py` | `.py` | 355 B | `PRODUCTION` | AI / Neural / Algorithmic Model Adapter |
| 139 | `ScanTOBIM/ScanTOBIM/agent/adapters/ascan2bim_wall_adapter.py` | `.py` | 7,715 B | `PRODUCTION` | AI / Neural / Algorithmic Model Adapter |
| 140 | `ScanTOBIM/ScanTOBIM/agent/adapters/fusion_engine.py` | `.py` | 25,770 B | `PRODUCTION` | AI / Neural / Algorithmic Model Adapter |
| 141 | `ScanTOBIM/ScanTOBIM/agent/adapters/grounding_dino_adapter.py` | `.py` | 9,914 B | `PRODUCTION` | AI / Neural / Algorithmic Model Adapter |
| 142 | `ScanTOBIM/ScanTOBIM/agent/adapters/ptv3_semantic_adapter.py` | `.py` | 8,686 B | `PRODUCTION` | AI / Neural / Algorithmic Model Adapter |
| 143 | `ScanTOBIM/ScanTOBIM/agent/adapters/yolo_column_adapter.py` | `.py` | 20,896 B | `PRODUCTION` | AI / Neural / Algorithmic Model Adapter |
| 144 | `ScanTOBIM/ScanTOBIM/agent/adapters/__pycache__/__init__.cpython-310.pyc` | `.pyc` | 579 B | `GENERATED` | AI / Neural / Algorithmic Model Adapter |
| 145 | `ScanTOBIM/ScanTOBIM/docs/SCAN_TO_BIM_ROADMAP.md` | `.md` | 24,948 B | `DOCUMENTATION` | Technical Documentation & Architectural Specification |
| 146 | `ScanTOBIM/ScanTOBIM/docs/algorithm-guide.html` | `.html` | 82,480 B | `DOCUMENTATION` | Technical Documentation & Architectural Specification |
| 147 | `ScanTOBIM/ScanTOBIM/docs/component-detection-plan.html` | `.html` | 79,923 B | `DOCUMENTATION` | Technical Documentation & Architectural Specification |
| 148 | `ScanTOBIM/ScanTOBIM/docs/deployment.html` | `.html` | 84,591 B | `DOCUMENTATION` | Technical Documentation & Architectural Specification |
| 149 | `ScanTOBIM/ScanTOBIM/docs/full-implementation-plan.html` | `.html` | 70,534 B | `DOCUMENTATION` | Technical Documentation & Architectural Specification |
| 150 | `ScanTOBIM/ScanTOBIM/docs/index.html` | `.html` | 155,136 B | `DOCUMENTATION` | Technical Documentation & Architectural Specification |
| 151 | `ScanTOBIM/ScanTOBIM/docs/randlanet-stb-workflow.md` | `.md` | 2,091 B | `DOCUMENTATION` | Technical Documentation & Architectural Specification |
| 152 | `ScanTOBIM/ScanTOBIM/docs/scan-to-bim-guide.html` | `.html` | 47,435 B | `DOCUMENTATION` | Technical Documentation & Architectural Specification |
| 153 | `ScanTOBIM/ScanTOBIM/docs/uml.html` | `.html` | 87,419 B | `DOCUMENTATION` | Technical Documentation & Architectural Specification |
| 154 | `ScanTOBIM/ScanTOBIM/build/stb-processor.spec` | `.spec` | 1,983 B | `GENERATED` | Build System / Tooling / Environment Configuration |
| 155 | `ScanTOBIM/ScanTOBIM/build/dist/stb-processor.exe` | `.exe` | 182,179,679 B | `GENERATED` | PyInstaller Build Log / Intermediate Output |
| 156 | `ScanTOBIM/ScanTOBIM/build/stb-processor/Analysis-00.toc` | `.toc` | 5,135,846 B | `GENERATED` | PyInstaller Build Log / Intermediate Output |
| 157 | `ScanTOBIM/ScanTOBIM/build/stb-processor/EXE-00.toc` | `.toc` | 2,753,844 B | `GENERATED` | PyInstaller Build Log / Intermediate Output |
| 158 | `ScanTOBIM/ScanTOBIM/build/stb-processor/PKG-00.toc` | `.toc` | 2,751,934 B | `GENERATED` | PyInstaller Build Log / Intermediate Output |
| 159 | `ScanTOBIM/ScanTOBIM/build/stb-processor/PYZ-00.pyz` | `.pyz` | 26,773,452 B | `GENERATED` | PyInstaller Build Log / Intermediate Output |
| 160 | `ScanTOBIM/ScanTOBIM/build/stb-processor/PYZ-00.toc` | `.toc` | 1,310,390 B | `GENERATED` | PyInstaller Build Log / Intermediate Output |
| 161 | `ScanTOBIM/ScanTOBIM/build/stb-processor/base_library.zip` | `.zip` | 880,569 B | `GENERATED` | PyInstaller Build Log / Intermediate Output |
| 162 | `ScanTOBIM/ScanTOBIM/build/stb-processor/stb-processor.pkg` | `.pkg` | 215,573,698 B | `GENERATED` | PyInstaller Build Log / Intermediate Output |
| 163 | `ScanTOBIM/ScanTOBIM/build/stb-processor/warn-stb-processor.txt` | `.txt` | 148,542 B | `GENERATED` | PyInstaller Build Log / Intermediate Output |
| 164 | `ScanTOBIM/ScanTOBIM/build/stb-processor/xref-stb-processor.html` | `.html` | 9,791,009 B | `GENERATED` | Technical Documentation & Architectural Specification |
| 165 | `ScanTOBIM/ScanTOBIM/build/stb-processor/localpycs/pyimod01_archive.pyc` | `.pyc` | 3,141 B | `GENERATED` | PyInstaller Build Log / Intermediate Output |
| 166 | `ScanTOBIM/ScanTOBIM/build/stb-processor/localpycs/pyimod02_importers.pyc` | `.pyc` | 22,899 B | `GENERATED` | PyInstaller Build Log / Intermediate Output |
| 167 | `ScanTOBIM/ScanTOBIM/build/stb-processor/localpycs/pyimod03_ctypes.pyc` | `.pyc` | 3,636 B | `GENERATED` | PyInstaller Build Log / Intermediate Output |
| 168 | `ScanTOBIM/ScanTOBIM/build/stb-processor/localpycs/pyimod04_pywin32.pyc` | `.pyc` | 1,069 B | `GENERATED` | PyInstaller Build Log / Intermediate Output |
| 169 | `ScanTOBIM/ScanTOBIM/build/stb-processor/localpycs/struct.pyc` | `.pyc` | 287 B | `GENERATED` | PyInstaller Build Log / Intermediate Output |
| 170 | `ScanTOBIM/ScanTOBIM/revit-addin/App.cs` | `.cs` | 7,344 B | `PRODUCTION` | Revit Add-in Infrastructure & Bootstrap |
| 171 | `ScanTOBIM/ScanTOBIM/revit-addin/ScanToBIMAgent.addin` | `.addin` | 552 B | `PRODUCTION` | Revit Add-in Infrastructure & Bootstrap |
| 172 | `ScanTOBIM/ScanTOBIM/revit-addin/ScanToBIMAgent.csproj` | `.csproj` | 1,430 B | `PRODUCTION` | Revit Add-in Infrastructure & Bootstrap |
| 173 | `ScanTOBIM/ScanTOBIM/revit-addin/family_map.json` | `.json` | 8,263 B | `PRODUCTION` | Revit Family & Category Mapping Configuration |
| 174 | `ScanTOBIM/ScanTOBIM/revit-addin/UI/AgentPanelPage.xaml` | `.xaml` | 10,651 B | `PRODUCTION` | Revit WPF Dockable Panel UI |
| 175 | `ScanTOBIM/ScanTOBIM/revit-addin/UI/AgentPanelPage.xaml.cs` | `.cs` | 2,525 B | `PRODUCTION` | Revit WPF Dockable Panel UI |
| 176 | `ScanTOBIM/ScanTOBIM/revit-addin/UI/AgentPanelViewModel.cs` | `.cs` | 7,275 B | `PRODUCTION` | Revit WPF Dockable Panel UI |
| 177 | `ScanTOBIM/ScanTOBIM/revit-addin/obj/ScanToBIMAgent.csproj.nuget.dgspec.json` | `.json` | 2,743 B | `GENERATED` | Compiled Revit Add-in Binary / MSBuild Intermediate Artifact |
| 178 | `ScanTOBIM/ScanTOBIM/revit-addin/obj/ScanToBIMAgent.csproj.nuget.g.props` | `.props` | 1,141 B | `GENERATED` | Compiled Revit Add-in Binary / MSBuild Intermediate Artifact |
| 179 | `ScanTOBIM/ScanTOBIM/revit-addin/obj/ScanToBIMAgent.csproj.nuget.g.targets` | `.targets` | 150 B | `GENERATED` | Compiled Revit Add-in Binary / MSBuild Intermediate Artifact |
| 180 | `ScanTOBIM/ScanTOBIM/revit-addin/obj/project.assets.json` | `.json` | 2,531 B | `GENERATED` | Compiled Revit Add-in Binary / MSBuild Intermediate Artifact |
| 181 | `ScanTOBIM/ScanTOBIM/revit-addin/obj/project.nuget.cache` | `.cache` | 269 B | `GENERATED` | Compiled Revit Add-in Binary / MSBuild Intermediate Artifact |
| 182 | `ScanTOBIM/ScanTOBIM/revit-addin/obj/Debug/net8.0-windows/.NETCoreApp,Version=v8.0.AssemblyAttributes.cs` | `.cs` | 198 B | `GENERATED` | Compiled Revit Add-in Binary / MSBuild Intermediate Artifact |
| 183 | `ScanTOBIM/ScanTOBIM/revit-addin/obj/Debug/net8.0-windows/ScanToBIMAgent.AssemblyInfo.cs` | `.cs` | 1,134 B | `GENERATED` | Compiled Revit Add-in Binary / MSBuild Intermediate Artifact |
| 184 | `ScanTOBIM/ScanTOBIM/revit-addin/obj/Debug/net8.0-windows/ScanToBIMAgent.AssemblyInfoInputs.cache` | `.cache` | 66 B | `GENERATED` | Compiled Revit Add-in Binary / MSBuild Intermediate Artifact |
| 185 | `ScanTOBIM/ScanTOBIM/revit-addin/obj/Debug/net8.0-windows/ScanToBIMAgent.GeneratedMSBuildEditorConfig.editorconfig` | `.editorconfig` | 944 B | `GENERATED` | Compiled Revit Add-in Binary / MSBuild Intermediate Artifact |
| 186 | `ScanTOBIM/ScanTOBIM/revit-addin/obj/Debug/net8.0-windows/ScanToBIMAgent.GlobalUsings.g.cs` | `.cs` | 183 B | `GENERATED` | Compiled Revit Add-in Binary / MSBuild Intermediate Artifact |
| 187 | `ScanTOBIM/ScanTOBIM/revit-addin/obj/Debug/net8.0-windows/ScanToBIMAgent.assets.cache` | `.cache` | 155 B | `GENERATED` | Compiled Revit Add-in Binary / MSBuild Intermediate Artifact |
| 188 | `ScanTOBIM/ScanTOBIM/revit-addin/obj/Debug/net8.0-windows/ScanToBIMAgent.csproj.AssemblyReference.cache` | `.cache` | 855,592 B | `GENERATED` | Compiled Revit Add-in Binary / MSBuild Intermediate Artifact |
| 189 | `ScanTOBIM/ScanTOBIM/revit-addin/obj/Debug/net8.0-windows/ScanToBIMAgent.csproj.CoreCompileInputs.cache` | `.cache` | 66 B | `GENERATED` | Compiled Revit Add-in Binary / MSBuild Intermediate Artifact |
| 190 | `ScanTOBIM/ScanTOBIM/revit-addin/obj/Debug/net8.0-windows/ScanToBIMAgent.csproj.FileListAbsolute.txt` | `.txt` | 2,700 B | `GENERATED` | Compiled Revit Add-in Binary / MSBuild Intermediate Artifact |
| 191 | `ScanTOBIM/ScanTOBIM/revit-addin/obj/Debug/net8.0-windows/ScanToBIMAgent.dll` | `.dll` | 241,664 B | `GENERATED` | Compiled Revit Add-in Binary / MSBuild Intermediate Artifact |
| 192 | `ScanTOBIM/ScanTOBIM/revit-addin/obj/Debug/net8.0-windows/ScanToBIMAgent.g.resources` | `.resources` | 7,761 B | `GENERATED` | Compiled Revit Add-in Binary / MSBuild Intermediate Artifact |
| 193 | `ScanTOBIM/ScanTOBIM/revit-addin/obj/Debug/net8.0-windows/ScanToBIMAgent.pdb` | `.pdb` | 96,992 B | `GENERATED` | Compiled Revit Add-in Binary / MSBuild Intermediate Artifact |
| 194 | `ScanTOBIM/ScanTOBIM/revit-addin/obj/Debug/net8.0-windows/ScanToBIMAgent_MarkupCompile.cache` | `.cache` | 601 B | `GENERATED` | Compiled Revit Add-in Binary / MSBuild Intermediate Artifact |
| 195 | `ScanTOBIM/ScanTOBIM/revit-addin/obj/Debug/net8.0-windows/UI/AgentPanelPage.baml` | `.baml` | 7,519 B | `GENERATED` | Revit WPF Dockable Panel UI |
| 196 | `ScanTOBIM/ScanTOBIM/revit-addin/obj/Debug/net8.0-windows/UI/AgentPanelPage.g.cs` | `.cs` | 3,062 B | `GENERATED` | Revit WPF Dockable Panel UI |
| 197 | `ScanTOBIM/ScanTOBIM/revit-addin/obj/Debug/net8.0-windows/ref/ScanToBIMAgent.dll` | `.dll` | 38,400 B | `GENERATED` | Compiled Revit Add-in Binary / MSBuild Intermediate Artifact |
| 198 | `ScanTOBIM/ScanTOBIM/revit-addin/obj/Debug/net8.0-windows/refint/ScanToBIMAgent.dll` | `.dll` | 38,400 B | `GENERATED` | Compiled Revit Add-in Binary / MSBuild Intermediate Artifact |
| 199 | `ScanTOBIM/ScanTOBIM/revit-addin/bin/x64/Debug/net8.0-windows/ScanToBIMAgent.addin` | `.addin` | 549 B | `GENERATED` | Compiled Revit Add-in Binary / MSBuild Intermediate Artifact |
| 200 | `ScanTOBIM/ScanTOBIM/revit-addin/bin/x64/Debug/net8.0-windows/ScanToBIMAgent.deps.json` | `.json` | 434 B | `GENERATED` | Compiled Revit Add-in Binary / MSBuild Intermediate Artifact |
| 201 | `ScanTOBIM/ScanTOBIM/revit-addin/bin/x64/Debug/net8.0-windows/ScanToBIMAgent.dll` | `.dll` | 150,528 B | `GENERATED` | Compiled Revit Add-in Binary / MSBuild Intermediate Artifact |
| 202 | `ScanTOBIM/ScanTOBIM/revit-addin/bin/x64/Debug/net8.0-windows/ScanToBIMAgent.pdb` | `.pdb` | 54,612 B | `GENERATED` | Compiled Revit Add-in Binary / MSBuild Intermediate Artifact |
| 203 | `ScanTOBIM/ScanTOBIM/revit-addin/bin/x64/Debug/net8.0-windows/Resources/ScanToBIM.SharedParameters.txt` | `.txt` | 1,864 B | `GENERATED` | Revit Shared Parameter Definitions |
| 204 | `ScanTOBIM/ScanTOBIM/revit-addin/bin/TempBuild/ScanToBIMAgent.addin` | `.addin` | 549 B | `GENERATED` | Compiled Revit Add-in Binary / MSBuild Intermediate Artifact |
| 205 | `ScanTOBIM/ScanTOBIM/revit-addin/bin/TempBuild/ScanToBIMAgent.deps.json` | `.json` | 434 B | `GENERATED` | Compiled Revit Add-in Binary / MSBuild Intermediate Artifact |
| 206 | `ScanTOBIM/ScanTOBIM/revit-addin/bin/TempBuild/ScanToBIMAgent.dll` | `.dll` | 220,160 B | `GENERATED` | Compiled Revit Add-in Binary / MSBuild Intermediate Artifact |
| 207 | `ScanTOBIM/ScanTOBIM/revit-addin/bin/TempBuild/ScanToBIMAgent.pdb` | `.pdb` | 86,452 B | `GENERATED` | Compiled Revit Add-in Binary / MSBuild Intermediate Artifact |
| 208 | `ScanTOBIM/ScanTOBIM/revit-addin/bin/TempBuild/Resources/ScanToBIM.SharedParameters.txt` | `.txt` | 1,864 B | `GENERATED` | Revit Shared Parameter Definitions |
| 209 | `ScanTOBIM/ScanTOBIM/revit-addin/bin/Debug/net8.0-windows/ScanToBIMAgent.addin` | `.addin` | 552 B | `GENERATED` | Compiled Revit Add-in Binary / MSBuild Intermediate Artifact |
| 210 | `ScanTOBIM/ScanTOBIM/revit-addin/bin/Debug/net8.0-windows/ScanToBIMAgent.deps.json` | `.json` | 434 B | `GENERATED` | Compiled Revit Add-in Binary / MSBuild Intermediate Artifact |
| 211 | `ScanTOBIM/ScanTOBIM/revit-addin/bin/Debug/net8.0-windows/ScanToBIMAgent.dll` | `.dll` | 241,664 B | `GENERATED` | Compiled Revit Add-in Binary / MSBuild Intermediate Artifact |
| 212 | `ScanTOBIM/ScanTOBIM/revit-addin/bin/Debug/net8.0-windows/ScanToBIMAgent.pdb` | `.pdb` | 96,992 B | `GENERATED` | Compiled Revit Add-in Binary / MSBuild Intermediate Artifact |
| 213 | `ScanTOBIM/ScanTOBIM/revit-addin/bin/Debug/net8.0-windows/Resources/ScanToBIM.SharedParameters.txt` | `.txt` | 1,864 B | `GENERATED` | Revit Shared Parameter Definitions |
| 214 | `ScanTOBIM/ScanTOBIM/revit-addin/Resources/ScanToBIM.SharedParameters.txt` | `.txt` | 1,864 B | `PRODUCTION` | Revit Shared Parameter Definitions |
| 215 | `ScanTOBIM/ScanTOBIM/revit-addin/Resources/family_map.json` | `.json` | 13,664 B | `PRODUCTION` | Revit Family & Category Mapping Configuration |
| 216 | `ScanTOBIM/ScanTOBIM/revit-addin/Models/AgentModels.cs` | `.cs` | 17,938 B | `PRODUCTION` | Revit Add-in Data Transfer Models |
| 217 | `ScanTOBIM/ScanTOBIM/revit-addin/Models/SidecarModels.cs` | `.cs` | 14,305 B | `PRODUCTION` | Revit Add-in Data Transfer Models |
| 218 | `ScanTOBIM/ScanTOBIM/revit-addin/SafetyGate/SafetyGateService.cs` | `.cs` | 12,552 B | `PRODUCTION` | Revit Safety Gate Service |
| 219 | `ScanTOBIM/ScanTOBIM/revit-addin/ManualTests/REVIT_TEST_GUIDE.md` | `.md` | 5,610 B | `TEST` | Revit Add-in Infrastructure & Bootstrap |
| 220 | `ScanTOBIM/ScanTOBIM/revit-addin/Properties/AssemblyInfo.cs` | `.cs` | 122 B | `PRODUCTION` | Revit Add-in Infrastructure & Bootstrap |
| 221 | `ScanTOBIM/ScanTOBIM/revit-addin/Commands/BuildGeometryCommand.cs` | `.cs` | 9,680 B | `PRODUCTION` | Revit Ribbon UI & External Command Handler |
| 222 | `ScanTOBIM/ScanTOBIM/revit-addin/Commands/ProcessScanCommand.cs` | `.cs` | 26,151 B | `PRODUCTION` | Revit Ribbon UI & External Command Handler |
| 223 | `ScanTOBIM/ScanTOBIM/revit-addin/Commands/StartAgentCommand.cs` | `.cs` | 1,511 B | `PRODUCTION` | Revit Ribbon UI & External Command Handler |
| 224 | `ScanTOBIM/ScanTOBIM/revit-addin/Commands/SyncToBackendCommand.cs` | `.cs` | 13,120 B | `PRODUCTION` | Revit Ribbon UI & External Command Handler |
| 225 | `ScanTOBIM/ScanTOBIM/revit-addin/Services/AgentBridgeService.cs` | `.cs` | 18,521 B | `PRODUCTION` | Revit Add-in Infrastructure & Bootstrap |
| 226 | `ScanTOBIM/ScanTOBIM/revit-addin/Services/ElementTypeConfigService.cs` | `.cs` | 8,937 B | `PRODUCTION` | Revit Family & Type Resolution Service |
| 227 | `ScanTOBIM/ScanTOBIM/revit-addin/Services/FamilyResolver.cs` | `.cs` | 11,893 B | `PRODUCTION` | Revit Family & Type Resolution Service |
| 228 | `ScanTOBIM/ScanTOBIM/revit-addin/Services/GeometryUtils.cs` | `.cs` | 2,588 B | `PRODUCTION` | Coordinate Transformation / Unit Mapping |
| 229 | `ScanTOBIM/ScanTOBIM/revit-addin/Services/RevitElementFactory.cs` | `.cs` | 114,201 B | `PRODUCTION` | Revit Native Element Factory (Wall/Floor/Column/Opening) |
| 230 | `ScanTOBIM/ScanTOBIM/revit-addin/Services/ScanGeometryBuilder.cs` | `.cs` | 144,907 B | `PRODUCTION` | Revit DirectShape & Custom Geometry Builder |
| 231 | `ScanTOBIM/ScanTOBIM/revit-addin/Services/ScanResultStore.cs` | `.cs` | 1,323 B | `PRODUCTION` | Revit Add-in Infrastructure & Bootstrap |
| 232 | `ScanTOBIM/ScanTOBIM/revit-addin/Services/SharedParameterLoader.cs` | `.cs` | 6,147 B | `PRODUCTION` | Revit Add-in Infrastructure & Bootstrap |
| 233 | `ScanTOBIM/ScanTOBIM/reports/AI_GEOMETRIC_FUSION_REPORT.md` | `.md` | 4,760 B | `GENERATED` | Generated Audit, Benchmark & Validation Report |
| 234 | `ScanTOBIM/ScanTOBIM/reports/DL_RESEARCH_INTEGRATION.md` | `.md` | 11,014 B | `GENERATED` | Generated Audit, Benchmark & Validation Report |
| 235 | `ScanTOBIM/ScanTOBIM/reports/FINAL_AI_GEOMETRIC_VALIDATION.json` | `.json` | 263,439 B | `GENERATED` | Generated Audit, Benchmark & Validation Report |
| 236 | `ScanTOBIM/ScanTOBIM/reports/FINAL_AI_GEOMETRIC_VALIDATION.md` | `.md` | 20,055 B | `GENERATED` | Generated Audit, Benchmark & Validation Report |
| 237 | `ScanTOBIM/ScanTOBIM/reports/FINAL_DETECTION_AUDIT.json` | `.json` | 1,706,055 B | `GENERATED` | Generated Audit, Benchmark & Validation Report |
| 238 | `ScanTOBIM/ScanTOBIM/reports/FINAL_DETECTION_AUDIT.md` | `.md` | 3,120 B | `GENERATED` | Generated Audit, Benchmark & Validation Report |
| 239 | `ScanTOBIM/ScanTOBIM/reports/FINAL_REAL_E57_VALIDATION.md` | `.md` | 6,870 B | `GENERATED` | Generated Audit, Benchmark & Validation Report |
| 240 | `ScanTOBIM/ScanTOBIM/reports/FINAL_RESEARCH_PROVENANCE.md` | `.md` | 9,014 B | `GENERATED` | Generated Audit, Benchmark & Validation Report |
| 241 | `ScanTOBIM/ScanTOBIM/reports/FINAL_VERIFICATION_REPORT.md` | `.md` | 21,611 B | `GENERATED` | Generated Audit, Benchmark & Validation Report |
| 242 | `ScanTOBIM/ScanTOBIM/reports/POINT_CLOUD_RESOLUTION_AUDIT.json` | `.json` | 1,707 B | `GENERATED` | Generated Audit, Benchmark & Validation Report |
| 243 | `ScanTOBIM/ScanTOBIM/reports/POINT_CLOUD_RESOLUTION_AUDIT.md` | `.md` | 1,801 B | `GENERATED` | Generated Audit, Benchmark & Validation Report |
| 244 | `ScanTOBIM/ScanTOBIM/reports/RESEARCH_PROVENANCE.md` | `.md` | 7,840 B | `GENERATED` | Generated Audit, Benchmark & Validation Report |
| 245 | `audit/REPOSITORY_INVENTORY.md` | `.md` | 17,665 B | `DOCUMENTATION` | Technical Documentation & Architectural Specification |
| 246 | `reports/AI_GEOMETRIC_FUSION_REPORT.md` | `.md` | 4,760 B | `GENERATED` | Generated Audit, Benchmark & Validation Report |
| 247 | `reports/DL_RESEARCH_INTEGRATION.md` | `.md` | 11,014 B | `GENERATED` | Generated Audit, Benchmark & Validation Report |
| 248 | `reports/FINAL_AI_GEOMETRIC_VALIDATION.json` | `.json` | 263,439 B | `GENERATED` | Generated Audit, Benchmark & Validation Report |
| 249 | `reports/FINAL_AI_GEOMETRIC_VALIDATION.md` | `.md` | 20,055 B | `GENERATED` | Generated Audit, Benchmark & Validation Report |
| 250 | `reports/FINAL_REAL_E57_VALIDATION.md` | `.md` | 6,870 B | `GENERATED` | Generated Audit, Benchmark & Validation Report |
| 251 | `reports/FINAL_RESEARCH_PROVENANCE.md` | `.md` | 9,014 B | `GENERATED` | Generated Audit, Benchmark & Validation Report |
| 252 | `reports/FINAL_VERIFICATION_REPORT.md` | `.md` | 21,611 B | `GENERATED` | Generated Audit, Benchmark & Validation Report |
