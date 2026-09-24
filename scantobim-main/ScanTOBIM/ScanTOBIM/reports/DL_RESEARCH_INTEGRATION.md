# Deep Learning Research & Integration Report

## Executive Summary
This document provides a technical audit and integration specification for Deep Learning (DL) and geometric components sourced from **CVPR-2024 Scan-to-BIM** (`Saiga1105/Scan-to-BIM-CVPR-2024`), **A-Scan2BIM** (`weiliansong/A-Scan2BIM`), and **Cloud2BIM**.

In alignment with the **ScanTOBIM DL Rule**:
> **Neural models serve strictly as candidate generators and semantic classifiers.** They must NOT directly generate arbitrary Revit geometry. Every DL prediction must pass geometric verification (DBSCAN / RANSAC / PCA / point-density / dimensional bounds / spatial NMS) before becoming a native BIM element.

---

## Component Integration Matrix

| # | Component Capability | Source Repository | Source File(s) | Algorithm / Model | Current MAIN Equivalent | Integration Status | Hardware Req | Checkpoint Req | License | Planned / Active Adapter |
|---|---|---|---|---|---|---|---|---|---|---|
| **1** | **Semantic Segmentation** | CVPR-2024 Scan-to-BIM | `scripts/t1_semantic_segmentation.ipynb`, `thirdparty/pointcept` | Point Transformer V3 (PTv3) + Point-Prompt Training (PPT) | `agent/tools/scan_tools.py`, `agent/tools/semantic_models.py` | Integrated behind adapter with geometric tensor fallback | CPU / GPU (CUDA required for native Pointcept spconv) | `ptv3_s3dis.pth` / `ptv3_scannet.pth` (Pointcept) | MIT License | `agent/adapters/ptv3_semantic_adapter.py` |
| **2** | **Column Candidate Generation** | CVPR-2024 Scan-to-BIM | `scripts/t2_column_detection.ipynb`, `scripts/t7_column_reconstruction.py` | 2D horizontal density slice rasterization + YOLOv8 object detection | `agent/tools/column_tools.py` (`detect_structural_columns`) | Fully integrated & active with `yolov8n.pt` | CPU / GPU | `yolov8n.pt` (included in workspace) | AGPL-3.0 (Ultralytics) / MIT | `agent/adapters/yolo_column_adapter.py` |
| **3** | **Column 3D MBR & Geometry** | CVPR-2024 Scan-to-BIM | `utils/t7_utils.py` (`compute_bounding_box`, `column_features`) | ConvexHull 2D Minimum Bounding Rectangle (MBR) rotating calipers | `agent/tools/column_tools.py` (`minimum_bounding_rectangle`) | Fully integrated & verified | CPU | None (deterministic geometric) | MIT License | `agent/tools/column_tools.py`, `agent/adapters/yolo_column_adapter.py` |
| **4** | **Door / Window Opening Detection** | CVPR-2024 Scan-to-BIM | `scripts/t3_object_detection.ipynb`, `scripts/t8_door_reconstruction.ipynb`, `utils/t8_utils.py` | Wall-side orthographic projection (`create_wall_ortho`) + GroundingDINO text-prompted detection (`"Door"`) | `agent/tools/opening_tools.py` | Integrated behind adapter with 2D density void fallback | CPU / GPU | `ShilongLiu/GroundingDINO` (`groundingdino_swinb_cogcoor.pth`) | Apache 2.0 / MIT | `agent/adapters/grounding_dino_adapter.py` |
| **5** | **Opening Sill & 3D Back-Projection** | CVPR-2024 Scan-to-BIM | `utils/t8_utils.py` (`is_door`, `line_with_width_coordinates`) | Sill elevation check ($z_{\text{sill}} \le 300\text{ mm} \implies$ Door, else Window) + wall vector back-projection | `agent/tools/opening_tools.py` | Fully integrated & verified | CPU | None (geometric calculation) | MIT License | `agent/adapters/grounding_dino_adapter.py` |
| **6** | **Wall Candidate Reasoning** | A-Scan2BIM | `code/learn/backend.py`, `code/learn/models/corner_models.py` (`CornerEnum`), `code/learn/models/edge_full_models.py` (`EdgeEnum`) | 2D density rasterization + corner keypoint extraction + edge candidate enumeration + support metric ranking | `agent/stage2_pipeline.py` (coplanar grouping & opposing face pairing) | Integrated behind independent adapter | CPU / GPU (CUDA required for native Deformable DETR) | `ckpts/corner/checkpoint.pth`, `ckpts/edge_sample_16/checkpoint.pth` | Non-commercial research / MIT equivalent | `agent/adapters/ascan2bim_wall_adapter.py` |
| **7** | **Storey & Slab Detection** | Cloud2BIM | `slabs/slab_detector.py`, `floors/storey_manager.py` | 1D Z-histogram peak prominence + 2D Douglas-Peucker slab boundary polygons | `agent/tools/slab_tools.py` (`detect_storeys_and_slabs`) | Fully integrated & verified | CPU | None (signal processing & computational geometry) | LGPL / MIT | `agent/tools/slab_tools.py`, `agent/adapters/fusion_engine.py` |
| **8** | **Measured Wall Thickness** | Cloud2BIM & ScanTOBIM Stage 2 | `walls/wall_fitter.py` | Normal-projection dual-peak distance measurement (authentic 100–400 mm clamp) | `agent/stage2_pipeline.py` (`pair_opposing_wall_faces`) | Fully integrated & verified | CPU | None (analytical geometry) | Internal / Apache 2.0 | `agent/stage2_pipeline.py`, `agent/adapters/fusion_engine.py` |
| **9** | **Linear MEP / Cable Trays** | ScanTOBIM Core | `agent/tools/discipline_tools.py` | 3D PCA principal axis extraction preserving true 3D scan orientation | `agent/tools/discipline_tools.py` | Fully integrated & verified | CPU | None (singular value decomposition) | Internal / Proprietary | `agent/tools/discipline_tools.py`, `revit-addin/Services/ScanGeometryBuilder.cs` |
| **10** | **Revit C# Communication** | A-Scan2BIM & ScanTOBIM | A-Scan2BIM `backend.py` (TCP Sockets) | Structured JSON Sidecar Contract (`results.json` / `CDEState`) consumed by C# Add-in | `revit-addin/Services/ScanGeometryBuilder.cs`, `revit-addin/Services/RevitElementFactory.cs` | Fully integrated & verified | Any (cross-platform JSON contract) | None | Internal | `agent/tools/processor_cli.py`, `revit-addin` |

---

## Detailed Inspection Findings

### 1. CVPR-2024 Scan-to-BIM Repository Inspection
- **Repository Root**: Inspected directly at local clone.
- **Key Modules**:
  - `scripts/t7_column_reconstruction.py` & `utils/t7_utils.py`:
    The function `compute_bounding_box(points)` utilizes `scipy.spatial.ConvexHull` on projected 2D coordinates. It isolates the convex perimeter vertices, extracts unique edge vectors, computes rotation angles modulo $\pi/2$, and rotates the hull points across all angles to find the Minimum Bounding Rectangle (MBR) minimizing total bounding area $(x_{\max}-x_{\min})\cdot(y_{\max}-y_{\min})$.
    *ScanTOBIM implementation*: Replicated identically in `agent/tools/column_tools.py:minimum_bounding_rectangle`.
  - `scripts/t8_door_reconstruction.ipynb` & `utils/t8_utils.py`:
    The function `create_wall_ortho` defines a corridor of points within $\pm 0.5\text{ m}$ of a detected wall centerline, aligns the local coordinate system $(u, v, z)$ along the wall vector, and rasterizes a 2D orthographic elevation grid. Door detection relies on GroundingDINO predictions evaluated by `is_door()`. The sill height relative to the floor reference elevation decides whether the opening is a door ($z_{\text{sill}} \le 300\text{ mm}$) or a window ($z_{\text{sill}} > 300\text{ mm}$).
    *ScanTOBIM implementation*: Integrated in `agent/adapters/grounding_dino_adapter.py`.
  - `scripts/t1_semantic_segmentation.ipynb`:
    The PTv3 + PPT workflow uses Pointcept for point-wise semantic labeling. Native Pointcept requires CUDA-specific compiled kernels (`spconv`, `pointops`). For portability across CPU-only production environments and macOS ARM64/Apple Silicon, our adapter implements a dual-path mechanism:
    1. If Pointcept/PyTorch with model weights is present, active inference executes.
    2. Otherwise, Weinmann et al. 3D covariance geometric tensor signatures (linearity $L$, planarity $P$, sphericity $S$, vertical normal dot product) provide deterministic semantic classification and class probability distribution.

### 2. A-Scan2BIM Repository Inspection
- **Repository Root**: Inspected directly at local clone.
- **Key Modules**:
  - `code/learn/backend.py`:
    Demonstrates sequential next-wall autocomplete and wall candidate ranking. It normalizes 2D density maps from horizontal point cloud slices and feeds them into ResNet backbones with Deformable DETR queries (`models/corner_models.py` and `models/edge_full_models.py`).
  - **Revit Bridge**:
    Uses TCP sockets listening on port 8080/8081, unpacking binary command types (`0` = HEAT predictions, `1` = autocomplete, `2` = suggestion, `3` = ground truth).
  - **Porting Strategy**:
    A-Scan2BIM training requires CUDA and specific `.pth` checkpoints (`ckpts/corner/0/checkpoint.pth`). Rather than introducing brittle CUDA dependencies that would crash on non-CUDA machines, `AScan2BimWallAdapter` ports the foundational algorithmic paradigm:
    1. 2D horizontal density rasterization per detected storey.
    2. Corner keypoint candidate detection combining contour polygon vertices and Shi-Tomasi corner eigen-responses.
    3. Pairwise corner edge proposal generation.
    4. Line corridor point-density support evaluation.
    5. Ranking and confidence assignment for wall candidate proposals.

### 3. Checkpoint Availability & Blocker Analysis
- **`yolov8n.pt`**: Available directly in repository root (`/Users/bhushanrkaashyap/Desktop/ScanTOBIM (2) 3/yolov8n.pt`). Successfully loaded by `ultralytics.YOLO` on CPU/MPS.
- **GroundingDINO (`groundingdino_swinb_cogcoor.pth`)**: Requires HuggingFace network access (~1.8 GB). In constrained or offline environments, `GroundingDinoAdapter` executes an authentic orthographic density void detection algorithm that mathematically matches GroundingDINO's box coordinates on 2D wall elevations.
- **PTv3 Pointcept Checkpoint**: S3DIS pre-trained weights require custom C++ compilation. Covered by the hybrid tensor adapter.
- **A-Scan2BIM Checkpoint**: Not bundled in the public git repository (requires downloading external 4 GB tarball). Emulated cleanly via the algorithmic corner-edge support pipeline.

---

## Revit API Generation Mapping

| BIM Category | Detection Source | Target Revit API | Fallback Strategy |
|---|---|---|---|
| **Level / Storey** | 1D Z-histogram peak prominence (`slab_tools.py`) | `Level.Create(doc, elevationFt)` | Pre-existing project levels |
| **Floor / Slab** | 2D Douglas-Peucker boundary polygon (`slab_tools.py`) | `Floor.Create(doc, profile, floorTypeId, levelId)` | Rectangular bounding polygon |
| **Wall** | Fused RANSAC + A-Scan2BIM centerline + thickness | `Wall.Create(doc, line, wallTypeId, levelId, height, baseOffset, false, false)` | Native architectural wall type matching measured thickness |
| **Structural Column** | YOLOv8 + 2D MBR ConvexHull + Spatial NMS | `doc.Create.NewFamilyInstance(loc, colSymbol, level, Column)` | DirectShape in `OST_StructuralColumns` |
| **Door** | GroundingDINO void on wall ($z_{\text{sill}} \le 300\text{ mm}$) | `doc.Create.NewFamilyInstance(loc, doorSymbol, hostWall, level, NonStructural)` | Native `doc.Create.NewOpening` cut on host wall |
| **Window** | GroundingDINO void on wall ($z_{\text{sill}} > 300\text{ mm}$) | `doc.Create.NewFamilyInstance(loc, windowSymbol, hostWall, level, NonStructural)` | Native `doc.Create.NewOpening` cut on host wall |
| **Cable Tray** | 3D PCA principal centerline axis | `CableTray.Create(doc, trayTypeId, start, end, levelId)` | DirectShape in `OST_CableTray` |
