# SCANTO BIM — FULL DETECTION RECOVERY + REAL E57 + AI/GEOMETRIC FUSION

## MISSION

The current implementation is NOT producing sufficiently correct detection.

Do NOT patch the screenshot.
Do NOT add fake geometry.
Do NOT hardcode tower geometry.
Do NOT simply increase/decrease one threshold.

Perform a full engineering recovery of the MAIN ScanTOBIM repository.

The final target is:

REAL 53.27M-POINT E57
→ complete source preservation
→ adaptive multi-resolution processing
→ REAL AI inference where checkpoints/modules exist
→ A-Scan2BIM / CVPR / Cloud2BIM research integration
→ candidate fusion
→ full-resolution local geometric verification
→ semantic BIM objects
→ native Revit 2025 elements
→ correct reconstructed building.

The provided Revit screenshot is ONLY a visual acceptance reference.
Never copy its coordinates, dimensions, positions, object counts, levels or geometry.

---

# 1. FIRST: STOP AND AUDIT

Before modifying code:

1. Create a Git checkpoint.
2. Inspect the ENTIRE MAIN ScanTOBIM repository.
3. Trace the complete execution path:
   E57/PLY
   → loader
   → downsampling
   → preprocessing
   → AI adapters
   → geometric detectors
   → candidate fusion
   → JSON/sidecar
   → C# Revit builder
   → Revit API.
4. Identify every location where points are removed.
5. Identify every location where geometry is synthesized.
6. Identify every fallback.
7. Identify every hardcoded dimension/coordinate/elevation.
8. Identify every DirectShape path.
9. Identify every semantic classifier.
10. Identify every model checkpoint and whether it actually exists.

Do NOT implement until this audit is complete.

---

# 2. POINT CLOUD DATA LOSS — FIX THIS FIRST

The source of truth is the original approximately 53,275,505-point E57.

The current runs show different point counts, including approximately 45M raw points and much smaller detection representations.

Do NOT assume the reduced representation is the source of truth.

Implement an explicit multi-resolution architecture:

SOURCE:
~53.27M E57 points

LEVEL 0:
full-resolution indexed source

LEVEL 1:
adaptive detection representation

LEVEL 2:
local full-resolution neighborhoods

Rules:

* Never permanently discard the original source.
* Never silently replace the source with a downsampled PLY.
* Record source point count.
* Record points loaded.
* Record points after each downsampling stage.
* Record points passed to each detector.
* Record final local full-resolution points used for geometry.
* Every detected BIM object must retain a mapping to its source/local point indices or reproducible spatial query.

Use adaptive voxelization rather than a blind global 0.05m reduction.

For global semantic inference, use a manageable representation.

For final geometry:
retrieve the original/full-resolution points inside each candidate region and recompute dimensions/normals/planes from those points.

This is mandatory.

Generate:

reports/POINT_CLOUD_RESOLUTION_AUDIT.json
reports/POINT_CLOUD_RESOLUTION_AUDIT.md

---

# 3. FIX THE WINDOWS UTF-8 FAILURE

Current execution reports:

Stage2 wall grouping failed because the Windows 'charmap' codec cannot encode Unicode character U+2192.

Fix the entire pipeline to use UTF-8.

Check:

* Python open()
* subprocess stdout
* subprocess stderr
* logging handlers
* JSON writing
* CSV writing
* sidecar writing
* C# process launching
* console output

Use explicit UTF-8 encoding.

Do not merely remove the arrow character.

After fixing, Stage2 wall grouping must actually execute successfully.

Add a regression test containing Unicode such as:

→
✓
μ
°

---

# 4. REAL AI INFERENCE — NO FAKE AI CLAIMS

The current log shows:

PTV3/PPT checkpoint=None
YOLO initialization fallback / No module

Therefore the current system must NOT claim that these models are actively running.

Inspect the actual research repositories and integrate the real implementations where technically possible.

Use the official repositories:

CVPR-2024 Scan-to-BIM:
Saiga1105/Scan-to-BIM-CVPR-2024

A-Scan2BIM:
weiliansong/A-Scan2BIM

Cloud2BIM:
VaclavNezerka/Cloud2BIM

You have permission to inspect the repositories and their source code as research references.
Clone/inspect the complete repositories rather than relying on summaries.

Preserve licenses and attribution.

---

# 5. PTv3 + PPT

Implement a real PTv3/PPT adapter.

Do not call tensor statistics or geometric signatures "PTv3 inference".

The adapter must expose:

* model name
* config
* checkpoint
* checkpoint existence
* checkpoint hash
* device
* input shape
* inference runtime
* output shape
* semantic class mapping
* per-class point counts

Actual execution must contain a model forward/inference operation.

If the checkpoint is unavailable:

* report it clearly
* do not fake inference
* keep the adapter disabled
* continue geometric processing
* provide the exact checkpoint required.

Use chunked/windowed inference for the large scan.
Never force the complete 53M points into one GPU tensor.

Fuse overlapping chunk predictions back into source/local point space.

---

# 6. YOLOv8

Implement REAL YOLOv8 inference.

Required flow:

point cloud
→ appropriate 2D projection / CVPR representation
→ YOLOv8
→ 2D candidate boxes
→ map candidates back to 3D
→ retrieve full-resolution 3D points
→ DBSCAN
→ RANSAC
→ ConvexHull / minimum bounding rectangle
→ verticality/density validation
→ confidence
→ structural column candidate.

Do not accept the existing geometric 646-column result blindly.

Produce:

geometry-only candidates
YOLO-only candidates
intersection
rejected YOLO candidates
rejected geometric candidates
final validated columns.

Every final column must have source-point evidence.

---

# 7. GroundingDINO

Implement actual GroundingDINO inference, not geometric void detection mislabeled as GroundingDINO.

Required flow:

validated wall
→ wall-side orthographic projection
→ GroundingDINO
→ prompts such as door/window
→ 2D candidate boxes
→ 3D back-projection
→ full-resolution point extraction
→ wall intersection check
→ dimensions
→ opening validation
→ Door/Window candidate.

Use the CVPR-2024 approach as the reference.

If the actual checkpoint is unavailable, report it honestly and retain the geometric opening detector as a separate fallback named:

GEOMETRIC_OPENING_DETECTOR

Never call that GroundingDINO.

---

# 8. A-SCAN2BIM

Inspect the actual A-Scan2BIM source.

Determine separately whether these are really executing:

* HEAT corner detector
* edge classifier
* candidate wall enumeration
* next-wall prediction
* metric-learning model.

If pretrained weights are unavailable, do NOT simulate them.

Use the repository's useful wall reasoning architecture where executable.

The wall pipeline should become:

PTv3 wall semantics
+
A-Scan2BIM wall candidates/reasoning
+
Cloud2BIM occupancy/contours
+
MAIN geometric wall detector
→ candidate fusion
→ PCA/RANSAC
→ full-resolution thickness estimation
→ wall centerline
→ storey association
→ native Revit Wall.Create.

---

# 9. CLOUD2BIM

Inspect the complete Cloud2BIM implementation.

Reuse the strongest validated CPU-compatible algorithms:

* Z histogram slab/storey detection
* occupancy maps
* morphological closing
* contour extraction
* PCA wall fitting
* wall grouping
* opening detection
* space generation
* IFC geometry concepts.

Do not blindly merge the repository.

Adapt components behind clean interfaces.

---

# 10. CANDIDATE FUSION

Create a canonical candidate model.

Every candidate must contain:

id
semantic_type
source_algorithm
source_repository
source_points
full_resolution_points
bbox
oriented_geometry
dimensions
storey
confidence
validation_results
revit_element_type
revit_ready

Fusion must:

* spatially deduplicate
* merge compatible candidates
* reject contradictory candidates
* preserve provenance
* preserve confidence
* never turn every candidate into geometry.

---

# 11. FIX THE CURRENT 1,128 BOX PROBLEM

The current Revit build reports approximately:

1,128 box
1,228 valve_candidate
277 cylinder
35 plane_vertical
11 plane_horizontal

This is NOT acceptable as the final semantic BIM pipeline.

Find exactly why so many generic candidates are classified as box/valve_candidate.

Do NOT fix it by deleting all boxes.

Determine:

* what detector generated them
* why they were classified
* source point count
* dimensions
* density
* location
* semantic confidence
* whether they correspond to actual scan objects.

Only genuine unsupported geometry may become DirectShape.

Structural elements must use native Revit objects whenever possible.

---

# 12. FIX VALVE_CANDIDATE FAILURE

The current log contains hundreds of:

BuildGeometry returned null or empty
shape: valve_candidate

Do not create a fallback cuboid.

For every valve candidate:

candidate
→ semantic validation
→ geometry validation
→ MEP-specific geometry

If it cannot be confidently identified:
REJECT / RECLASSIFY

Do not send it to the Revit geometry builder.

---

# 13. FIX DirectShape VISIBILITY OVERRIDE FAILURE

Current errors show:

View.SetElementOverrides()
"The view type does not support Visibility/Graphics Overrides."

This must NEVER cause geometry creation to fail.

Refactor:

CreateGeometry()
and
ApplyVisualStyle()

into separate operations.

Correct behavior:

geometry creation succeeds
→ element is committed

then:

try ApplyColourOverride()
catch:
log warning
keep element

The visualization layer must never invalidate geometry.

---

# 14. REMOVE UNSAFE GENERIC BOX FALLBACKS

Search all C# geometry code for:

BuildBoxGeometry
BoundingBox
AABB
DirectShape fallback
box fallback

Classify every usage.

Allowed:

* genuine unsupported geometry
* explicitly marked fallback with provenance and confidence

Forbidden:

* failed wall → room-sized box
* failed column → arbitrary box
* failed cable tray → floor-spanning box
* failed opening → fake box
* failed semantic classification → generic box.

---

# 15. COORDINATE SYSTEM — ONE AUTHORITATIVE TRANSFORM

Implement:

E57 coordinates
→ local normalized coordinates
→ BIM coordinates
→ Revit internal feet

with one transform object.

Never scatter:
*1000
/1000
meters-to-feet
centering offsets
manual Z offsets

throughout the code.

Every BIM object must retain:

source_coordinate_system
target_coordinate_system
transform_id

Verify all level elevations using the same transform.

---

# 16. NATIVE REVIT FIRST

Required mapping:

Level
→ Level

Wall
→ Wall.Create

Floor/Slab
→ Floor.Create

Structural Column
→ FamilyInstance

Door
→ native Door FamilyInstance if available

Window
→ native Window FamilyInstance if available

Opening
→ doc.Create.NewOpening / appropriate Revit API

Cable Tray
→ CableTray.Create

Pipe
→ Pipe.Create where supported

Unsupported custom geometry
→ DirectShape ONLY when necessary.

Record native vs DirectShape counts.

---

# 17. FULL-RESOLUTION LOCAL REFINEMENT

This is critical.

AI and global geometry may operate on a reduced representation.

But before creating a BIM object:

candidate
→ spatial query against ORIGINAL E57
→ retrieve all local source points
→ recompute:
centroid
normals
PCA
RANSAC
dimensions
thickness
orientation
density
verticality
topology

Only then create the BIM element.

This gives us speed AND accuracy.

---

# 18. VALIDATION TARGETS

For every final BIM object:

* source point count > 0
* source region known
* geometry valid
* dimensions physically plausible
* no NaN/Inf
* correct storey
* correct coordinate transform
* confidence recorded
* provenance recorded
* Revit element ID recorded after creation.

Generate:

reports/FINAL_DETECTION_AUDIT.json
reports/FINAL_DETECTION_AUDIT.md

---

# 19. REAL E57 ACCEPTANCE TEST

Run against the REAL source E57.

Report:

source point count
loaded point count
detection point count
full-resolution refinement point count
voxel size
SOR parameters
storeys
walls
slabs
columns
doors
windows
openings
cable trays
pipes
other MEP
rejected candidates
duplicate candidates
DirectShapes
native Revit elements
failed elements
runtime per stage.

Never report only "total elements".

---

# 20. RESEARCH PROVENANCE

Create:

reports/RESEARCH_PROVENANCE.md

For every adapted implementation:

repository
source file
source function/class
MAIN file
MAIN function/class
algorithm
license
adaptation
execution status
checkpoint requirement.

Explicitly distinguish:

REAL NEURAL INFERENCE
vs
GEOMETRIC ADAPTATION
vs
RESEARCH-INSPIRED IMPLEMENTATION.

---

# 21. DO NOT STOP AT DOCUMENTATION

Actually modify and execute the code.

Do not claim success from imports.

A model is considered integrated only when:

checkpoint exists
→ model loads
→ inference executes
→ predictions are produced
→ predictions enter fusion
→ predictions affect validated BIM candidates.

---

# 22. FINAL ACCEPTANCE

Do not declare success until:

1. Stage2 UTF-8 failure is fixed.
2. Original E57 source is preserved.
3. Point-count flow is auditable.
4. Adaptive detection resolution works.
5. Full-resolution local refinement works.
6. Real AI inference is proven or explicitly marked unavailable.
7. 1,128-box explosion is resolved.
8. valve_candidate explosion is resolved.
9. DirectShape visualization failures no longer destroy geometry.
10. Native Revit element ratio is maximized.
11. Walls are reconstructed from actual scan geometry.
12. Columns are independently validated.
13. Slabs/levels come from the scan.
14. Doors/windows come from evidence.
15. Cable trays follow actual detected 3D paths.
16. Coordinate transformation is consistent.
17. Every BIM element is traceable to point-cloud evidence.
18. The final Revit model is generated from the real scan.

Do not add another repository.

Do not add cosmetic changes.

Do not hardcode the screenshot.

Fix the detection and reconstruction pipeline itself.

At the end provide the exact command used for the final real-E57 run and the exact breakdown of:

SOURCE POINTS
→ DETECTION POINTS
→ CANDIDATES
→ VALIDATED CANDIDATES
→ NATIVE REVIT ELEMENTS
→ DIRECTSHAPES
→ FAILED ELEMENTS.
