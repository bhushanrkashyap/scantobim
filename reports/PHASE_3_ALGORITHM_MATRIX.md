# Phase 3 Algorithm Execution Matrix

| Algorithm | Available | Checkpoint | Loaded | Forward/Run | Actual status | Fallback |
|---|:---:|---|:---:|:---:|---|---|
| PTv2 | No | Missing | No | No | `UNAVAILABLE` | Geometric Tensor Classifier |
| RandLA-Net | No | Missing | No | No | `UNAVAILABLE` | Geometric Tensor Classifier |
| KPConv | No | Missing | No | No | `UNAVAILABLE` | Geometric Tensor Classifier |
| PointGroup | No | Missing | No | No | `UNAVAILABLE` | Density-Adaptive KDTree |
| SoftGroup | No | Missing | No | No | `UNAVAILABLE` | Density-Adaptive KDTree |
| Mask3D | No | Missing | No | No | `UNAVAILABLE` | Density-Adaptive KDTree |
| Geometric Tensor Classifier | Yes | N/A | Yes | Yes | `GEOMETRIC_ADAPTER` | None (Active) |
| Plane fitting (SVD / RANSAC) | Yes | N/A | Yes | Yes | `ALGORITHMIC_ADAPTER` | None (Active) |
| Cylinder fitting (RANSAC) | Yes | N/A | Yes | Yes | `ALGORITHMIC_ADAPTER` | None (Active) |
| HDBSCAN / KDTree Clustering | Yes | N/A | Yes | Yes | `ALGORITHMIC_ADAPTER` | None (Active) |
| PCA / Oriented Bounding Box | Yes | N/A | Yes | Yes | `ALGORITHMIC_ADAPTER` | None (Active) |
