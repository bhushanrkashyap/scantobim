# SCANTO BIM — DUPLICATE IMPLEMENTATION INVENTORY

**Audit Date:** 2026-09-24  
**Repository Root:** `scantobim-main`  
**Total Duplicate Groups Identified:** 72  

## Executive Summary

A full automated scan across the entire repository was performed across all directories:
- `agent/tools/` (Production Python Tools)
- `agent/tests/` (Test Suites and Pytest Fixtures)
- Root and CLI runners (`demo.py`, `check_wall_detection.py`, `demo_runner.py`, `worker.py`)
- `revit-addin/` (C# source files, `.addin` manifests, `.csproj` project files, build caches)
- `reports/` and `audit/` (Verification reports, generality audits, remediation reports)
- Binary models (`yolov8n.pt`)

| Category | Total Groups | Byte-Identical | Differentiating Logic |
| :--- | :--- | :--- | :--- |
| Python Production Tools | 18 | 15 | 3 (`coordinate_system`, `registration_tools`, `scan_tools`) |
| Python Test Fixtures | 8 | 8 | 0 |
| Python CLI & Worker Scripts | 5 | 5 | 0 |
| Audit & Verification Reports | 24 | 23 | 1 (`FINAL_DETECTION_AUDIT.json` timestamp) |
| C# & Revit Add-In Artifacts | 19 | 8 | 0 |
| ML Weights & Assets | 1 | 1 | 0 |
| Cross-Directory Duplicate Files | 2 | 1 | 1 (`run_stage2_demo.py` forwarder) |
| Empty macOS Duplicate Directories | 2 | 2 | 0 |
| **Total** | **72** | **63** | **5** |

## Detailed Duplicate Inventory

| ID | Category | Canonical Path | Duplicate Path | Identical? | Notes / Analysis |
| :--- | :--- | :--- | :--- | :--- | :--- |
| DUP-001 | PRODUCTION_TOOL | `fmea.py` | `fmea 2.py` | YES | Byte-for-byte identical copy. |
| DUP-002 | PRODUCTION_TOOL | `opening_tools.py` | `opening_tools 2.py` | YES | Byte-for-byte identical copy. |
| DUP-003 | PRODUCTION_TOOL | `wall_classifier.py` | `wall_classifier 2.py` | YES | Byte-for-byte identical copy. |
| DUP-004 | PRODUCTION_TOOL | `coordinate_system.py` | `coordinate_system 2.py` | NO (Differs) | Canonical AST +7.  |
| DUP-005 | PRODUCTION_TOOL | `column_tools.py` | `column_tools 2.py` | YES | Byte-for-byte identical copy. |
| DUP-006 | PRODUCTION_TOOL | `loa_tools.py` | `loa_tools 2.py` | YES | Byte-for-byte identical copy. |
| DUP-007 | PRODUCTION_TOOL | `registration_tools.py` | `registration_tools 2.py` | NO (Differs) | Canonical AST +8. Duplicate AST +1 (['downsample_for_icp']).  |
| DUP-008 | PRODUCTION_TOOL | `deviation_heatmap.py` | `deviation_heatmap 2.py` | YES | Byte-for-byte identical copy. |
| DUP-009 | PRODUCTION_TOOL | `build_semantic_checkpoint.py` | `build_semantic_checkpoint 2.py` | YES | Byte-for-byte identical copy. |
| DUP-010 | PRODUCTION_TOOL | `scan_tools.py` | `scan_tools 2.py` | NO (Differs) | Byte-for-byte identical copy. |
| DUP-011 | PRODUCTION_TOOL | `semantic_models.py` | `semantic_models 2.py` | YES | Byte-for-byte identical copy. |
| DUP-012 | PRODUCTION_TOOL | `detection_config.py` | `detection_config 2.py` | YES | Byte-for-byte identical copy. |
| DUP-013 | PRODUCTION_TOOL | `handover_bundle.py` | `handover_bundle 2.py` | YES | Byte-for-byte identical copy. |
| DUP-014 | PRODUCTION_TOOL | `registration_report.py` | `registration_report 2.py` | YES | Byte-for-byte identical copy. |
| DUP-015 | PRODUCTION_TOOL | `coordinator_tools.py` | `coordinator_tools 2.py` | YES | Byte-for-byte identical copy. |
| DUP-016 | PRODUCTION_TOOL | `open3d_viewer.py` | `open3d_viewer 2.py` | YES | Byte-for-byte identical copy. |
| DUP-017 | PRODUCTION_TOOL | `nqa1_package.py` | `nqa1_package 2.py` | YES | Byte-for-byte identical copy. |
| DUP-018 | PRODUCTION_TOOL | `validate_run.py` | `validate_run 2.py` | YES | Byte-for-byte identical copy. |
| DUP-019 | TEST_FIXTURE | `test_nqa1_package.py` | `test_nqa1_package 2.py` | YES | Byte-for-byte identical copy. |
| DUP-020 | TEST_FIXTURE | `test_overlap.py` | `test_overlap 2.py` | YES | Byte-for-byte identical copy. |
| DUP-021 | TEST_FIXTURE | `test_ncr.py` | `test_ncr 2.py` | YES | Byte-for-byte identical copy. |
| DUP-022 | TEST_FIXTURE | `test_utf8_pipeline.py` | `test_utf8_pipeline 2.py` | YES | Byte-for-byte identical copy. |
| DUP-023 | TEST_FIXTURE | `test_iso15926.py` | `test_iso15926 2.py` | YES | Byte-for-byte identical copy. |
| DUP-024 | TEST_FIXTURE | `test_loa.py` | `test_loa 2.py` | YES | Byte-for-byte identical copy. |
| DUP-025 | TEST_FIXTURE | `test_reconstruction_pipeline.py` | `test_reconstruction_pipeline 2.py` | YES | Byte-for-byte identical copy. |
| DUP-026 | TEST_FIXTURE | `test_stage2_core.py` | `test_stage2_core 2.py` | YES | Byte-for-byte identical copy. |
| DUP-027 | SCRIPT_OR_ROOT | `demo.py` | `demo 2.py` | YES | Byte-for-byte identical copy. |
| DUP-028 | SCRIPT_OR_ROOT | `check_wall_detection.py` | `check_wall_detection 2.py` | YES | Byte-for-byte identical copy. |
| DUP-029 | SCRIPT_OR_ROOT | `test_wall_detection.py` | `test_wall_detection 2.py` | YES | Byte-for-byte identical copy. |
| DUP-030 | SCRIPT_OR_ROOT | `demo_runner.py` | `demo_runner 2.py` | YES | Byte-for-byte identical copy. |
| DUP-031 | SCRIPT_OR_ROOT | `worker.py` | `worker 2.py` | YES | Byte-for-byte identical copy. |
| DUP-032 | AUDIT_OR_REPORT | `FINAL_VERIFICATION_REPORT.md` | `FINAL_VERIFICATION_REPORT 2.md` | YES | Byte-for-byte identical copy. |
| DUP-033 | AUDIT_OR_REPORT | `FINAL_DETECTION_AUDIT.md` | `FINAL_DETECTION_AUDIT 2.md` | YES | Byte-for-byte identical copy. |
| DUP-034 | AUDIT_OR_REPORT | `FINAL_RESEARCH_PROVENANCE.md` | `FINAL_RESEARCH_PROVENANCE 2.md` | YES | Byte-for-byte identical copy. |
| DUP-035 | AUDIT_OR_REPORT | `FINAL_REAL_E57_VALIDATION.md` | `FINAL_REAL_E57_VALIDATION 2.md` | YES | Byte-for-byte identical copy. |
| DUP-036 | AUDIT_OR_REPORT | `AI_GEOMETRIC_FUSION_REPORT.md` | `AI_GEOMETRIC_FUSION_REPORT 2.md` | YES | Byte-for-byte identical copy. |
| DUP-037 | AUDIT_OR_REPORT | `FINAL_DETECTION_AUDIT.json` | `FINAL_DETECTION_AUDIT 2.json` | NO (Differs) | Byte-for-byte identical copy. |
| DUP-038 | AUDIT_OR_REPORT | `POINT_CLOUD_RESOLUTION_AUDIT.md` | `POINT_CLOUD_RESOLUTION_AUDIT 2.md` | YES | Byte-for-byte identical copy. |
| DUP-039 | AUDIT_OR_REPORT | `FINAL_AI_GEOMETRIC_VALIDATION.json` | `FINAL_AI_GEOMETRIC_VALIDATION 2.json` | YES | Byte-for-byte identical copy. |
| DUP-040 | AUDIT_OR_REPORT | `FINAL_AI_GEOMETRIC_VALIDATION.md` | `FINAL_AI_GEOMETRIC_VALIDATION 2.md` | YES | Byte-for-byte identical copy. |
| DUP-041 | AUDIT_OR_REPORT | `POINT_CLOUD_RESOLUTION_AUDIT.json` | `POINT_CLOUD_RESOLUTION_AUDIT 2.json` | YES | Byte-for-byte identical copy. |
| DUP-042 | AUDIT_OR_REPORT | `numeric_constants.csv` | `numeric_constants 2.csv` | YES | Byte-for-byte identical copy. |
| DUP-043 | AUDIT_OR_REPORT | `PHASE_1_GENERALITY_REAUDIT.json` | `PHASE_1_GENERALITY_REAUDIT 2.json` | YES | Byte-for-byte identical copy. |
| DUP-044 | AUDIT_OR_REPORT | `PHASE_1_REMEDIATION_REPORT.json` | `PHASE_1_REMEDIATION_REPORT 2.json` | YES | Byte-for-byte identical copy. |
| DUP-045 | AUDIT_OR_REPORT | `REPOSITORY_INVENTORY.md` | `REPOSITORY_INVENTORY 2.md` | YES | Byte-for-byte identical copy. |
| DUP-046 | AUDIT_OR_REPORT | `PHASE_1_GENERALITY_REAUDIT.md` | `PHASE_1_GENERALITY_REAUDIT 2.md` | YES | Byte-for-byte identical copy. |
| DUP-047 | AUDIT_OR_REPORT | `PHASE_1_REMEDIATION_REPORT.md` | `PHASE_1_REMEDIATION_REPORT 2.md` | YES | Byte-for-byte identical copy. |
| DUP-048 | AUDIT_OR_REPORT | `CONFIGURATION_CANDIDATES.md` | `CONFIGURATION_CANDIDATES 2.md` | YES | Byte-for-byte identical copy. |
| DUP-049 | AUDIT_OR_REPORT | `PHASE_0_GENERALITY_AUDIT.md` | `PHASE_0_GENERALITY_AUDIT 2.md` | YES | Byte-for-byte identical copy. |
| DUP-050 | AUDIT_OR_REPORT | `PHASE_0_GENERALITY_AUDIT.json` | `PHASE_0_GENERALITY_AUDIT 2.json` | YES | Byte-for-byte identical copy. |
| DUP-051 | AUDIT_OR_REPORT | `FINAL_VERIFICATION_REPORT.md` | `FINAL_VERIFICATION_REPORT 2.md` | YES | Byte-for-byte identical copy. |
| DUP-052 | AUDIT_OR_REPORT | `DL_RESEARCH_INTEGRATION.md` | `DL_RESEARCH_INTEGRATION 2.md` | YES | Byte-for-byte identical copy. |
| DUP-053 | AUDIT_OR_REPORT | `FINAL_RESEARCH_PROVENANCE.md` | `FINAL_RESEARCH_PROVENANCE 2.md` | YES | Byte-for-byte identical copy. |
| DUP-054 | AUDIT_OR_REPORT | `AI_GEOMETRIC_FUSION_REPORT.md` | `AI_GEOMETRIC_FUSION_REPORT 2.md` | YES | Byte-for-byte identical copy. |
| DUP-055 | AUDIT_OR_REPORT | `FINAL_AI_GEOMETRIC_VALIDATION.json` | `FINAL_AI_GEOMETRIC_VALIDATION 2.json` | YES | Byte-for-byte identical copy. |
| DUP-056 | REVIT_ADDIN_OR_BUILD | `ScanToBIMAgent.csproj` | `ScanToBIMAgent 2.csproj` | YES | Byte-for-byte identical copy. |
| DUP-057 | REVIT_ADDIN_OR_BUILD | `ScanToBIMAgent.addin` | `ScanToBIMAgent 2.addin` | YES | Byte-for-byte identical copy. |
| DUP-058 | REVIT_ADDIN_OR_BUILD | `ScanToBIMAgent.csproj.nuget.g.props` | `ScanToBIMAgent.csproj.nuget.g 2.props` | YES | Byte-for-byte identical copy. |
| DUP-059 | REVIT_ADDIN_OR_BUILD | `project.nuget.cache` | `project.nuget 2.cache` | YES | Byte-for-byte identical copy. |
| DUP-060 | REVIT_ADDIN_OR_BUILD | `ScanToBIMAgent.addin` | `ScanToBIMAgent 2.addin` | YES | Byte-for-byte identical copy. |
| DUP-061 | REVIT_ADDIN_OR_BUILD | `ScanToBIMAgent.dll` | `ScanToBIMAgent 2.dll` | YES | Byte-for-byte identical copy. |
| DUP-062 | REVIT_ADDIN_OR_BUILD | `ScanToBIMAgent.pdb` | `ScanToBIMAgent 2.pdb` | YES | Byte-for-byte identical copy. |
| DUP-063 | REVIT_ADDIN_OR_BUILD | `SafetyGateService.cs` | `SafetyGateService 2.cs` | YES | Byte-for-byte identical copy. |
| DUP-064 | MODEL_WEIGHTS | `yolov8n.pt` | `yolov8n 2.pt` | YES | Byte-for-byte identical copy. |
| DUP-065 | REVIT_ADDIN_OR_BUILD | `family_map.json` | `family_map 2.json` | YES | Byte-for-byte identical copy. |
| DUP-066 | REVIT_ADDIN_OR_BUILD | `project.assets.json` | `project.assets 2.json` | YES | Byte-for-byte identical copy. |
| DUP-067 | REVIT_ADDIN_OR_BUILD | `ScanToBIMAgent.csproj.nuget.dgspec.json` | `ScanToBIMAgent.csproj.nuget.dgspec 2.json` | YES | Byte-for-byte identical copy. |
| DUP-068 | REVIT_ADDIN_OR_BUILD | `ScanToBIMAgent.deps.json` | `ScanToBIMAgent.deps 2.json` | YES | Byte-for-byte identical copy. |
| DUP-069 | CROSS_DIRECTORY_SCRIPT | `run_stage2_demo.py` | `run_stage2_demo.py` | NO (Differs) | Outer file is a 349-byte forwarder wrapper script pointing to inner full demo script. |
| DUP-070 | CROSS_DIRECTORY_CSPROJ | `ScanToBIMAgent.csproj` | `ScanToBIMAgent.csproj` | YES | Outer file is an unreferenced duplicate copy of Revit Addin project file. |
| DUP-071 | EMPTY_DIRECTORY_DUPLICATE | `tasks` | `tasks 2` | YES | Empty macOS Finder duplicate directory artifact. |
| DUP-072 | EMPTY_DIRECTORY_DUPLICATE | `Resources` | `Resources 2` | YES | Empty macOS Finder duplicate directory artifact. |
