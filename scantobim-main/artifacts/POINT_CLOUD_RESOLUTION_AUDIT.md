# Point Cloud Multi-Resolution & Resolution Audit Report

## 1. Executive Summary

- **Source File**: `2026-07-28 (2).e57`
- **Source Point Count (ASTM E57)**: `53,275,505`
- **Loaded Point Count (Level 0)**: `53,275,505`
- **Adaptive Representation (Level 1)**: `1,207,133`
- **Level 0 Data Loss**: `0 points (0.00% reduction — 100% preserved)`

### Bounding Box Extents (Raw E57 Coordinates in Metres)
| Axis | Min (m) | Max (m) | Total Span (m) |
|---|---|---|---|
| **X** | -7.120 | 13.817 | 20.937 |
| **Y** | -8.453 | 14.271 | 22.724 |
| **Z** | -14.903 | 11.408 | 26.311 |

---

## 2. Multi-Resolution Architecture Stages

| Stage | Input Points | Output Points | Ratio | Sampling Method | Voxel Size (m) | Purpose |
|---|---|---|---|---|---|---|
| **SOURCE_E57_PRESERVATION** | 53,275,505 | 53,275,505 | 1.0000 | raw_source_indexing | None | Source of truth point preservation (Level 0) |
| **LEVEL1_ADAPTIVE_DETECTION_REPRESENTATION** | 53,275,505 | 1,207,133 | 0.0227 | adaptive_voxel_stratification | 0.050 | Detection representation for global detectors (storeys, RANSAC, YOLO) |

---

## 3. Level 2 Local Full-Resolution Refinement Protocol

Every detected candidate (wall, column, floor slab, opening) executes a Level 2 spatial query
against the original 53.27M-point Level 0 index:
1. **Spatial Bounding Query**: Extracts authentic raw scan points within the candidate boundary + padding.
2. **Centroid Recomputation**: Center of gravity computed directly from un-decimated points.
3. **3D Covariance Eigen-Decomposition**: Recomputes exact surface normal and dimensional variances.
4. **Planar Residual Verification**: Measures root-mean-square orthogonal distance to the fitted plane.
5. **Physical Plausibility Clamp**: Verifies wall thickness (100–400 mm) and column dimensions.
