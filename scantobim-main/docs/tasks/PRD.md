# SCANTO BIM — DETECTION RECOVERY & NATIVE BIM PRD

## Status

ACTIVE — Ralph Loop execution

## Mission

Recover and complete the ScanTOBIM pipeline so that real scan data produces reliable, traceable BIM objects.

The original scan contains approximately 53 million points.

The current pipeline is reducing the working representation to approximately 4.3 million points and producing poor detection results.

The system must therefore distinguish between:

* source data
* detection representation
* refined representation
* final BIM geometry.

The source scan must remain authoritative.

---

# 1. BUSINESS / ENGINEERING OUTCOME

The final system must support:

REAL E57
→ preprocessing
→ adaptive-resolution detection
→ AI/geometric detection
→ candidate fusion
→ full-resolution validation
→ semantic BIM
→ Revit.

The result must represent the actual scanned building.

The system must not generate an artificial tower based on hardcoded geometry.

---

# 2. CURRENT KNOWN PROBLEMS

## P0 — Point-cloud information loss

Original source:

~53M points.

Current working representation:

~4.3M points in one observed processing path.

Problem:

Detection quality is not sufficient.

Required solution:

multi-resolution architecture with source preservation and local full-resolution refinement.

---

## P0 — AI adapters are not proven active

Observed initialization includes states equivalent to:

`PTv3/PPT checkpoint=None`

and:

`YOLO ... No module`

These must be investigated.

An adapter initialization message does not prove inference.

Required:

prove checkpoint loading + model construction + forward pass + output.

If unavailable:

explicitly mark unavailable.

---

## P0 — Stage2 wall grouping failure

Known failure:

Windows `charmap` encoding cannot encode Unicode arrow character.

Required:

fix the actual encoding path and verify Stage2 executes successfully.

---

## P0 — Generic box explosion

Current geometry build has produced a large number of generic boxes.

Trace the origin.

Determine whether they represent:

* actual detected objects;
* fallback objects;
* misclassified structures;
* duplicated candidates;
* projection artifacts.

Do not blindly delete.

---

## P0 — Valve candidate explosion

Current geometry build has produced a large number of `valve_candidate` objects and many geometry-generation failures.

Required:

trace the classifier and candidate generator.

Require physical/geometric evidence before classification.

Do not use a cuboid as a fake valve.

---

## P0 — DirectShape overuse

DirectShape is currently heavily used.

Required:

maximize native Revit elements where possible.

DirectShape remains a fallback for validated unsupported geometry.

---

## P0 — Coordinate consistency

Logs indicate multiple coordinate/level values that require verification.

Required:

one authoritative coordinate transformation.

Document units at every boundary.

---

# 3. POINT-CLOUD ARCHITECTURE

Implement:

```text
SOURCE E57
    |
    +--> FULL RESOLUTION SOURCE INDEX
    |
    +--> COARSE REPRESENTATION
    |
    +--> MEDIUM DETECTION REPRESENTATION
    |
    +--> FINE REPRESENTATION
    |
    +--> LOCAL FULL-RESOLUTION QUERY
             |
             v
       GEOMETRIC REFINEMENT
```

Every candidate must retain enough information to locate its corresponding source region.

---

# 4. RESOLUTION AUDIT

Create:

`POINT_CLOUD_RESOLUTION_AUDIT.json`

The report must contain, for every major processing stage:

* stage
* input point count
* output point count
* reduction ratio
* sampling method
* voxel size
* dilution factor
* bounding box
* coordinate system
* units
* purpose
* whether source mapping is retained.

Acceptance:

No stage may silently discard the source without documentation.

---

# 5. DETECTION OBJECT MODEL

Every candidate should contain:

```text
candidate_id
semantic_type
source_algorithms
source_point_count
full_resolution_point_count
bbox
centroid
orientation
dimensions
thickness
storey
confidence
confidence_evidence
geometric_validation
ai_validation
provenance
status
revit_ready
native_revit_type
```

---

# 6. DETECTOR REQUIREMENTS

## 6.1 Storeys

Derive storeys from scan evidence.

Do not hardcode:

* number of floors
* elevations
* spacing.

Use horizontal density/planar evidence.

---

## 6.2 Slabs / Floors

Use scan-derived horizontal planar evidence.

Possible techniques:

* Z density histogram
* plane fitting
* continuity
* thickness estimation
* storey association.

---

## 6.3 Walls

Use:

* vertical planar evidence
* RANSAC
* PCA
* occupancy
* line fitting
* thickness estimation
* continuity
* full-resolution refinement.

Stage2 wall grouping must execute successfully.

Record:

```text
raw walls
duplicate-filtered walls
merged walls
rejected walls
final walls
```

---

## 6.4 Columns

Use:

* verticality
* cross-section
* planar faces
* point density
* storey continuity
* local full-resolution evidence.

If YOLO is available, compare genuine YOLO output against geometric candidates.

---

## 6.5 Openings

Detect openings from actual gaps in wall geometry and/or genuine AI detection.

Use full-resolution validation.

---

## 6.6 MEP

Only classify:

* pipes
* cable trays
* mechanical equipment
* valves

when geometric evidence supports the classification.

A rectangular scan region must not automatically become a valve.

---

# 7. AI INTEGRATION

## CVPR-2024

Inspect and reuse appropriate components:

* PTV3/PPT
* YOLO
* GroundingDINO
* geometric reconstruction.

Verify checkpoints.

Verify inference.

Record actual runtime and output counts.

---

## A-SCAN2BIM

Inspect:

* wall enumeration
* corner detection
* edge classification
* thickness prediction
* wall post-processing
* next-wall prediction.

Separate neural functionality from CPU-compatible geometric functionality.

---

## CLOUD2BIM

Inspect:

* slab detection
* occupancy maps
* wall segmentation
* contours
* PCA
* openings
* spaces
* IFC concepts.

Use adapters.

Do not blindly merge repositories.

---

# 8. CANDIDATE FUSION

Different detectors may identify the same physical object.

Implement candidate fusion based on:

* spatial overlap
* distance
* orientation
* dimensions
* storey
* semantic compatibility
* point support.

Do not simply concatenate detector results.

---

# 9. CONFIDENCE

Confidence must be explainable.

Possible terms:

```text
geometric_fit
point_density
dimension_validity
orientation_consistency
storey_consistency
cross_detector_agreement
ai_confidence
full_resolution_support
```

Store individual terms.

Do not invent confidence values.

---

# 10. FULL-RESOLUTION VALIDATION

For every accepted major BIM object:

1. locate source region;
2. retrieve local high-resolution/full-resolution points;
3. refit geometry;
4. calculate residual;
5. calculate dimensions;
6. verify semantic plausibility;
7. calculate confidence;
8. accept/reject.

---

# 11. REVIT

Prefer:

```text
Level
Wall.Create
Floor.Create
FamilyInstance
NewOpening
CableTray.Create
Pipe.Create
```

Use DirectShape only where necessary.

Every created object must retain candidate provenance.

---

# 12. GEOMETRY FAILURE HANDLING

Geometry generation must reject:

* null geometry
* empty geometry
* zero dimensions
* NaN coordinates
* infinite coordinates
* invalid transformations.

It must not silently create a fake fallback box.

---

# 13. VISUAL STYLING

Visual styling is independent from geometry creation.

A failure in:

`View.SetElementOverrides`

must produce a warning only.

It must never cause an otherwise valid geometry candidate to fail.

---

# 14. OUTPUT REPORTS

Generate:

```text
POINT_CLOUD_RESOLUTION_AUDIT.json
DETECTION_VALIDATION_REPORT.json
RESEARCH_PROVENANCE_REPORT.md
```

Optionally:

```text
DETECTION_VALIDATION_REPORT.md
```

---

# 15. REQUIRED TESTS

## Test A — Point-cloud audit

Verify source and derived counts.

## Test B — Stage2

Verify wall grouping completes.

## Test C — AI availability

Verify every claimed AI model.

## Test D — Candidate validation

Verify invalid candidates are rejected.

## Test E — Real scan

Run against the actual point cloud.

## Test F — Revit

Build actual BIM geometry.

## Test G — Visual inspection

Verify objects correspond to scan geometry.

---

# 16. ACCEPTANCE CRITERIA

### A. Data integrity

* [ ] Original scan remains authoritative.
* [ ] Point reduction is explicitly reported.
* [ ] Full-resolution local lookup works.
* [ ] No silent permanent data loss.

### B. Detection

* [ ] Storeys scan-derived.
* [ ] Slabs scan-derived.
* [ ] Walls scan-derived.
* [ ] Columns scan-derived.
* [ ] Openings scan-derived.
* [ ] MEP candidates evidence-based.

### C. AI

* [ ] PTv3/PPT status verified.
* [ ] YOLO status verified.
* [ ] GroundingDINO status verified.
* [ ] A-Scan2BIM neural status verified.
* [ ] No fake AI inference.

### D. Pipeline

* [ ] Stage2 works.
* [ ] Candidate fusion works.
* [ ] Confidence is explainable.
* [ ] Full-resolution refinement works.
* [ ] Coordinate transformation is authoritative.

### E. Revit

* [ ] Native elements used where possible.
* [ ] DirectShape only used when justified.
* [ ] Generic box explosion resolved.
* [ ] Valve candidate explosion resolved.
* [ ] Geometry failures are validated.
* [ ] Styling errors cannot invalidate geometry.

### F. Evidence

* [ ] Resolution audit exists.
* [ ] Detection validation report exists.
* [ ] Research provenance exists.
* [ ] Real scan execution evidence exists.
* [ ] Revit build evidence exists.

---

# 17. DEFINITION OF DONE

The task is NOT complete when:

* code compiles;
* tests pass only on synthetic data;
* an adapter prints "initialized";
* a UI reports "success";
* geometry is merely visible.

The task is complete when real scan data produces validated, traceable BIM objects and the complete pipeline has execution evidence.
