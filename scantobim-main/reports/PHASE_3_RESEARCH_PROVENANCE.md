# Phase 3 Research Provenance Report

## Engineering Transparency & Provenance Ledger

| Algorithm | Paper / Origin | Checkpoint | Status | Actual Execution | Adaptation / Role |
|---|---|---|---|---|---|
| **PTv2 (Point Transformer V2)** | Point Transformer V2: Grouped Vector Attention and Partition-based Pooling (NeurIPS 2022) | `ptv2_scannet.pth` | `UNAVAILABLE` | Attempted load; verified missing weights | Adapter interface with health_check, infer, and taxonomy mapping |
| **RandLA-Net** | RandLA-Net: Efficient Semantic Segmentation on Large-Scale Point Clouds (CVPR 2020) | `randlanet_semantickitti.pth` | `UNAVAILABLE` | Multi-resolution spatial chunking adapter created; weights verified missing | Spatial block indexing with overlap reconciliation and source index preservation |
| **KPConv (Kernel Point Convolution)** | KPConv: Flexible and Deformable Convolution for Point Clouds (ICCV 2019) | `kpconv_s3dis.pth` | `UNAVAILABLE` | Adapter verified missing weights | Subsampling and radius neighbor kernel convolution adapter |
| **PointGroup / SoftGroup / Mask3D** | PointGroup (CVPR 2020) / SoftGroup (CVPR 2022) / Mask3D (ICLR 2023) | `None` | `UNAVAILABLE` | Attempted load; weights missing | Instance segmentation adapter contract |
| **Geometric Tensor Classifier** | Differential Geometry and Tensor Invariants for 3D Urban Point Clouds (Wechsler et al.) | `N/A (Analytical / Tensor Math)` | `GEOMETRIC_ADAPTER` | Executed on all points | kNN covariance surface normal estimation, planarity, verticality, horizontality classification |
| **Density-Adaptive KDTree Region Growing** | Euclidean Cluster Extraction and Normal Consistency Clustering (Rusu 2009) | `N/A (Algorithmic)` | `ALGORITHMIC_ADAPTER` | Executed on semantic partitions | Density-adaptive distance query with normal angular consistency (cos < 40 deg) |
| **Cloud2BIM Z-Histogram Storey & Slab Extraction** | Cloud2BIM: Automated Building Information Model Reconstruction (Macher et al.) | `N/A (Algorithmic)` | `ALGORITHMIC_ADAPTER` | Executed on scan Z-profile | Zero-padded boundary peak prominence detection and 2D convex hull / polygon simplification |
| **Orthogonal Wall Plane & Opposing Face Thickness** | Automated 3D Reconstruction of Wall Geometry from Point Clouds (Previtali et al.) | `N/A (Algorithmic)` | `ALGORITHMIC_ADAPTER` | Executed on wall instances | Vertical plane projection, bimodal thickness analysis, 5-vector local frame, collinear merge, L/T junction topology |
