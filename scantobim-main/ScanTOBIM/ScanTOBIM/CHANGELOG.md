# Changelog

All notable changes to ScanToBIM are documented here.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [1.2.1] — 2026-04-20

### Changed — Cross-platform support (Ubuntu + Windows)

- **Makefile**: removed a legacy hardcoded interpreter path — now
  auto-detects a usable Python 3.11+ via
  `command -v python3.11 / python3 / python`, with a clear error
  message on failure. Works on Ubuntu 22.04+, Debian 12+, RHEL 9+,
  and any POSIX Linux with Python 3.11 installed.
- **`make.ps1`** (new): drop-in PowerShell runner for Windows 10/11
  with the same target names (`install`, `server`, `test`, `demo`,
  `docker-up`, `build-sidecar`, etc.). No `make.exe` required.
- **`build/build-sidecar.ps1`** (new): native Windows PyInstaller
  wrapper — produces `build\dist\stb-processor.exe` with the same
  hidden-import list as the bash script. Uses Windows path separators
  for `--add-data` (semicolon vs colon).
- **`build/build-sidecar.sh`**: comments + `uname` detection updated
  for Ubuntu / Linux primary; Git-Bash / Cygwin / MSYS remain
  supported but de-emphasised in favour of the PowerShell path.
- **Scan upload dir default**: `/tmp/scans` → `tempfile.gettempdir() /
  "scantobim" / "scans"` in both `agent/main.py` and `agent/worker.py`.
  On Ubuntu this resolves to `/tmp/scantobim/scans`; on Windows to
  `%TEMP%\scantobim\scans`. Override with the existing
  `SCAN_FILES_DIR` env var.
- **`docs/deployment.html`**: added a PowerShell — Windows 10 / 11
  code block next to the existing Ubuntu / Linux one. Supported-OS
  row in the requirements table now reads "Ubuntu 22.04+ or Windows
  10/11 (PowerShell 5.1+)".
- **Revit addin comment**: sidecar-discovery Priority 3 comment
  updated to "Linux — no .exe extension (dev-machine build)".
- **Release archive**: stale desktop metadata files removed; tar
  exclusion list unchanged (already excluded).

### Notes for first-time setup

Ubuntu / Debian:
    sudo apt install python3.11 python3.11-venv
    cd ScanToBIM
    make install
    make env
    make server

Windows 10 / 11 (PowerShell):
    # Install Python 3.11 via python.org or Microsoft Store
    cd ScanToBIM
    .\make.ps1 install
    .\make.ps1 env
    .\make.ps1 server

Docker continues to work identically on both platforms via
`make docker-up` / `.\make.ps1 docker-up`.

All 480 tests pass. No new Python dependencies.

---

## [1.2.0] — 2026-04-20

### Added — Quality deliverables (P2.2a–d)

**P2.2a — USIBD LOA v3.1 per-element accuracy**
- `agent/tools/loa_tools.py` — tier classification, σ computation, `LOAStatement` builder
- Segmentation computes σ per segment; tags carry `loa_sigma_mm` + `loa_tier`
- `GET /sessions/{id}/loa-report` with worst-tier gate + tier distribution
- 3 new Revit shared parameters: `ScanToBIM_LOATier`, `ScanToBIM_LOASigmaMm`, `ScanToBIM_LOAStandard`
- `ScanGeometryBuilder` writes LOA params to every DirectShape
- 37 new tests

**P2.2b — Deviation heatmap**
- `agent/tools/deviation_heatmap.py` — zero-dep SVG renderer
- `GET /sessions/{id}/deviation-heatmap?view=plan|elevation&colour_by=deviation|loa&format=svg|json`
- Inline tooltips, XML escaping, 1 m scale bar, legend, per-element dots
- 19 new tests

**P2.2c — Registration QA report (RICS / PAS 128)**
- `agent/tools/registration_report.py` — Cat A/B/C/FAIL classification
- `RegistrationQAReportModel` + `StationQAModel` Pydantic types
- `GET /sessions/{id}/registration-report?format=json|html`
- Printable HTML deliverable with XSS-safe escaping
- 32 new tests

**P2.2d — Handover bundle**
- `agent/tools/handover_bundle.py` — single .zip with 9 sections:
  `01_ifc/`, `02_loa/`, `03_deviation/`, `04_registration/`, `05_quality/`,
  `06_drp/`, `07_nqa1/`, `08_fmea/`, `09_iso15926/`
- Graceful skip with reason logged when upstream data missing
- `GET /sessions/{id}/handover-bundle` returns zipped stream with `X-ScanToBIM-Sections` header
- 17 new tests

### Added — NQA-1 compliance layer (P2.4)

- `agent/determinism.py` — `set_global_seed`, `deterministic_segment_id`,
  input/output hash, `ProvenanceRecord`
- `agent/nqa1_signature.py` — three-party signature workflow (Preparer / Verifier / Approver)
  with HMAC-chained storage, ordering enforcement, separation-of-duties
- `agent/tools/nqa1_package.py` — V&V package with identity, RTM, session evidence
- `agent/tools/fmea.py` — 17-entry failure-mode catalog with RPN + test-ID traceability
- `GET /sessions/{id}/provenance` — seeds, hashes, identity
- `GET /sessions/{id}/fmea?format=json|html`
- Sign / list / revoke endpoints for signatures
- NQA-1 HTML renders signatures live with chain-valid banner
- 28 determinism tests + 35 signature tests + 24 FMEA tests

### Added — ISO 15926 / CFIHOS handover (P2.3)

- `agent/tools/iso15926_tools.py` — ElementType → ISO 15926-4 class URI for 45 types
- CFIHOS attribute groups: PIPING, VESSEL, ROTATING_EQUIPMENT, ELECTRICAL, STRUCTURAL, NUCLEAR_PRIMARY
- `GET /sessions/{id}/iso15926?format=json|csv` — CSV ready for AVEVA NET / Hexagon SDx
- 23 new tests

### Added — Frontend integration (P2.5)

- Quality Deliverables panel on session page — LOA, Registration QA, Heatmap cards
- "Download Handover .zip" button with section-summary toast
- NQA-1 Signatures card with role-gated sign buttons
- 5 new `api.js` methods for the above

### Changed

- Removed all references to specific consultancy brands + cloud providers from source
  files, docs, and shared parameters — deliverables are now vendor-neutral
- Email placeholders in tests, demos, and docs switched to `@example.com`
- Revit add-in manifest `VendorId` / `VendorDescription` made vendor-neutral
- Developer-workflow memory file excluded from release archive

### Test & Quality

- **480 tests pass** (was 239 at v1.1.0 — added 241 tests this release)
- No new Python dependencies; zero regressions from scrub
- CI continues to gate at 70% line coverage

---

## [1.1.0] — 2026-04-18

### Added — Revit Plugin Sprints (P1–P5)

**P1 — Sidecar CLI (`stb-processor.exe`)**
- `agent/tools/processor_cli.py` — standalone CLI wrapping `run_segmentation()`
- Handles large scan files (5–50 GB) locally; emits `PROGRESS:N:msg` tokens to stdout
- Writes `results.json` schema v1.0 with segments, metadata, warnings
- `build/build-sidecar.sh` — PyInstaller `--onefile` build script with hidden imports for open3d, laspy, scipy, numpy, pydantic
- 19 tests in `agent/tests/test_processor_cli.py`

**P2 — Revit `ProcessScan` command**
- `revit-addin/Commands/ProcessScanCommand.cs` — OpenFileDialog → ZoneInputWindow → sidecar discovery → background progress window
- 4-level sidecar discovery: env var → DLL dir → no-ext → repo `build/dist`
- Deserialises results into `ScanResultStore.Latest` for downstream ribbon commands
- `revit-addin/Models/SidecarModels.cs` — `SidecarResults`, `SidecarSegment`, shape helpers (`IsCylinder`, `IsValveCandidate`)

**P3 — DirectShape geometry builder**
- `revit-addin/Services/ScanGeometryBuilder.cs` — box tessellation (6 quads) + cylinder prism (24 sides)
- Colour overrides by confidence: green ≥0.85, amber ≥0.60, red <0.60
- `ScanToBIM_SegmentId` shared parameter written per element
- `BuildGeometryCommand.cs` — transaction wrapper with rollback on failure

**P4 — Backend ingest endpoint + sync command**
- `POST /sessions/{id}/ingest-elements` — accepts pre-classified segments (~KB JSON) instead of raw scan files
- Runs `classify_and_prepare` + safety gate evaluation + clash detection
- `SyncToBackendCommand.cs` — WPF dialog → JWT auth → session create → ingest POST
- `_ingest_sources` module-level dict tracks `{source, metadata}` per session
- 18 tests in `agent/tests/test_ingest.py`

**P5 — Frontend UI wiring**
- Violet plugin badge in sessions table for `scan_source === 'revit_plugin'`
- Web Upload / Revit Plugin two-tab toggle in create-session modal
- Plugin metadata panel on session detail: source file, size, raw point count, processing time

### Added — Overlap resolution

- `agent/tools/scan_tools.py::_resolve_overlaps()` — post-segmentation deduplication
  - Same-shape IoU merge (≥60% → keep higher confidence, absorb point count)
  - Containment absorb (≥80% enclosed → drop smaller, exempting `VALVE_CANDIDATE`)
- `agent/tools/coordinator_tools.py::_is_whitelisted_pair()` — clash detection whitelist
  - Valve/fitting on pipe
  - Conduit through wall/floor/ceiling
  - Stacked horizontal planes at different Z elevations
- `revit-addin/Services/ScanGeometryBuilder.cs` — 5 mm inset on all BB faces (prevents z-fighting)
- Prefer RANSAC-fitted radius from `fitted_radius_mm` tag over AABB heuristic
- 29 tests in `agent/tests/test_overlap.py`

### Added — Sprint 3.5 Scan Intelligence

- `agent/tools/registration_tools.py` — multi-station cloud-to-cloud registration
- `agent/tools/deviation_tools.py` — element-vs-scan deviation reporting
- `agent/tools/floorplan_tools.py` — 2D floor-plan export from plane segments

### Changed

- `agent/main.py` — added `Field` import, `IngestElementsRequest` model, `ingest-elements` endpoint
- `list_sessions` response includes `scan_source`; `GET /sessions/{id}` includes `scan_source` and `scan_metadata`
- `agent/pyproject.toml` — added `python-multipart>=0.0.12` dependency
- `revit-addin/App.cs` — ribbon now has 5 buttons: Start Agent, Load Params, Process Scan, Build Geometry, Sync to Backend

### Fixed

- Overlapping segments across detection stages (RANSAC → DBSCAN → region growing → euclidean) no longer produce duplicate DirectShapes
- Clash detector no longer flags valve-on-pipe, pipe-through-wall, or stacked floor/ceiling as false positives
- Revit DirectShape z-fighting at shared boundaries resolved via 5 mm face inset
- Cylinder radius overestimate (AABB `min(width, depth)/2`) replaced with RANSAC-fitted radius when available

### Chore

- Removed third-party vendor name references from `deploy.sh` and `docs/full-implementation-plan.html`

### Test & Quality

- **239 tests pass**, 4 skipped
- **71.56 % line coverage** (gate: 70 %)
- CI workflow picks up all new test modules via pytest auto-discovery

---

## [1.0.0] — Earlier

Initial release: Sprints 0–3 foundation, Revit bridge, web frontend, algorithm suite.

- Agentic pipeline: point cloud → Open3D segmentation → safety classification → Revit element creation
- SQLite audit ledger with HMAC chain for tamper detection
- FastAPI agent API on port 8765
- SC1 hard block, SC2 approval workflow
- 30 nuclear/industrial element types
- IFC export, clash detection, NCR auto-generation
- Interactive HTML product documentation
