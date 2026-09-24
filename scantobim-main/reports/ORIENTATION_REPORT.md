# Vertical Axis & Orientation Estimation Report

## Overview

ScanTOBIM Phase 2 introduces data-derived vertical axis estimation and horizontal frame construction via `agent.tools.orientation_tools`.
The system eliminates unconditional world-Z assumptions and determines the physical vertical orientation from geometric evidence.

---

## 1. Vertical Axis Estimation Methods

Three complementary methods are executed in prioritized consensus:

1. **Surface Normal Distribution Analysis**:
   Surface normals are sampled across the downsampled cloud. In building environments, structural elements are predominantly horizontal (slabs, floors, ceilings) or vertical (walls, columns). Histograms of normal vector projections identify the axis with the highest concentration of orthogonal and parallel normals.

2. **Dominant Horizontal Plane (Floor/Slab) Analysis**:
   RANSAC plane segmentation detects the largest planar segments in the point cloud. The normal vectors of the most extensive horizontal-like surfaces directly define the gravity/vertical axis candidates.

3. **Principal Component Analysis (PCA)**:
   For buildings spanning primarily in the horizontal plane (large footprint relative to height), the principal component corresponding to the minimum eigenvalue approximates the vertical axis.

---

## 2. Multi-Method Consensus & Confidence

* **Agreement Criterion**: Independent estimates must achieve a cosine similarity exceeding $0.95$:
  $$|\mathbf{u}_1 \cdot \mathbf{u}_2| > 0.95$$
* **`ORIENTATION_VERIFIED`**: Agreement between surface normal distribution and dominant plane detection.
* **`ORIENTATION_INFERRED`**: Strong evidence from at least one robust method (e.g. clearly defined floor plane).
* **`ORIENTATION_AMBIGUOUS`**: Disagreement among methods or featureless geometry (e.g. isotropic spherical noise). When ambiguous, the system records `ORIENTATION_AMBIGUOUS` and maintains the default frame without false certainty.

---

## 3. Horizontal Frame & Canonical Transformation

To establish a canonical frame:
1. All vertical surface normals are projected onto the horizontal plane perpendicular to the estimated vertical axis.
2. A circular histogram ($0^\circ - 180^\circ$) identifies the dominant orthogonal building wall directions ($\mathbf{h}_1, \mathbf{h}_2$).
3. `build_canonical_frame()` constructs the rotation matrix:
   $$\mathbf{R} = \begin{bmatrix} \mathbf{h}_1^T \\ \mathbf{h}_2^T \\ \mathbf{u}^T \end{bmatrix}$$
   The resulting matrix is guaranteed to be in $SO(3)$ ($\det(\mathbf{R}) = +1$, $\mathbf{R}\mathbf{R}^T = \mathbf{I}$).
