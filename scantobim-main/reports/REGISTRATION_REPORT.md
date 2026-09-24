# Multi-Scan Registration & Alignment Engine Report

## Overview

ScanTOBIM Phase 2 establishes a dual-mode registration framework within `agent.tools.registration_tools`.
The system strictly enforces the principle that single-scan inputs must never undergo fake registration against synthetic targets, while multi-scan inputs must follow a rigorous global-to-local registration pipeline with quality validation gates and degradation detection.

---

## 1. Input Modes & Dispatch

### Mode A: Standalone Point Cloud
* **Condition**: Exactly one point cloud file supplied.
* **Resolution**: `resolve_registration_mode()` emits:
  $$\text{status} = \text{REGISTRATION\_NOT\_REQUIRED}$$
* **Invariant**: No synthetic target, reference cube, or identity transform is fabricated. Downstream pipeline proceeds directly to spatial statistics and canonical normalization.

### Mode B: Multi-Scan Network
* **Condition**: $\ge 2$ scan files supplied.
* **Resolution**: Station 0 establishes the world reference frame. Each subsequent scan station undergoes global alignment followed by local refinement against the accumulated registered cloud.

---

## 2. Multi-Scan Alignment Pipeline

```
           +-----------------------------------------------+
           | Reference Scan 0 (Station 0) + Moving Scan i  |
           +-----------------------------------------------+
                                   |
                                   v
           [ Preprocessing: Voxel Grid + Normal Estimation ]
                                   |
                                   v
           [ Feature Extraction: 33-dimensional FPFH ]
             - Radius: Derived from resolution (voxel * 5)
                                   |
                                   v
           [ Global Alignment: RANSAC Feature Matching ]
             - Computes initial SE(3) transformation T_ransac
             - Validation Gate: Fitness >= 0.3, RMSE <= 0.2m
             - If invalid: Registration rejected immediately
                                   |
                                   v
           [ Local Refinement: Point-to-Plane ICP ]
             - Initialized with T_ransac (never blind identity)
             - Convergence criteria: max 50 iterations
                                   |
                                   v
           [ Degradation Check & Quality Evaluation ]
             - Is fitness(ICP) < 0.9 * fitness(RANSAC)?
             - Is RMSE(ICP) > 1.5 * RMSE(RANSAC)?
             - Yes: Retain T_ransac ("FPFH_RANSAC (ICP degraded)")
             - No:  Retain T_icp ("FPFH_RANSAC_ICP")
```

---

## 3. Registration Quality Validation Gate

Every pairwise registration is quantitatively validated by `_validate_transform()` against strict acceptance thresholds:
* **Fitness (Inlier Ratio)**: $\ge 0.30$
* **RMSE**: $\le 0.20\text{ m}$ ($200\text{ mm}$)
* **Rotation Determinant**: $|\det(\mathbf{R}) - 1.0| < 0.01$
* **Orthogonality Error**: $\|\mathbf{R}^T \mathbf{R} - \mathbf{I}\|_\infty < 0.01$

If any threshold is violated, the status is set to `REGISTRATION_FAILED` with detailed `failure_reasons` recorded in the audit trail.
