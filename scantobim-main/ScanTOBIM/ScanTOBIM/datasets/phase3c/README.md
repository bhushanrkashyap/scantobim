# Phase 3C Ground-Truth Dataset: Authentic ASTM E57 Point Cloud Benchmark

## Dataset Architecture & Verification Standard

This directory contains the ground-truth benchmark and training dataset for **Phase 3C: Real Object Detection + Multi-Class Semantic Validation**.

### Key Rules & Invariants
1. **Zero Data Leakage**:
   - The authentic Leica ASTM E57 scan (53,275,505 points) is divided into **spatially disjoint volumetric sectors**.
   - Neighboring points from the same physical object NEVER cross partition boundaries.
   - `TRAIN`, `VALIDATION`, and `TEST` regions are separated by distinct spatial coordinates.
2. **First-Class Unknowns**:
   - Unclassified, ambiguous, or unsupported geometries are explicitly annotated as `UNKNOWN` rather than forced into synthetic or incorrect semantic classes.
3. **Point-Level Source Provenance**:
   - Every annotated point stores its authentic `point_id` mapping 1-to-1 back to the raw E57 stream.
4. **Instance Ground Truth**:
   - Every object instance possesses a unique `object_id`, `instance_id`, `semantic_class`, `superclass`, 3D bounding box, and exact point ID membership list.

### Partition Directory Structure
- `annotations/`: NPZ archives containing point coordinates, source point IDs, class IDs, and instance IDs.
- `splits/`: JSON partition manifest files specifying the spatial bounding boxes and instance rosters for `TRAIN`, `VALIDATION`, `TEST`, and `REAL_DEPLOYMENT`.
- `metadata/`: Regional metadata files with class distributions, point densities, and scan coordinate origins.
