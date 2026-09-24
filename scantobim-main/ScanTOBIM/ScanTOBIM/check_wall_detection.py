"""Standalone proof that a curved (tower) wall is detected as vertical planes.

Runs the REAL detect_planes on a synthetic vertical cylinder — no Revit, no IFC.
Answers one question: does the shaft come out as PLANE_VERTICAL wall candidates?

    python3 check_wall_detection.py
"""

import numpy as np
import open3d as o3d

from agent.tools.scan_tools import detect_planes

# Synthetic vertical cylinder (a tower shaft): axis = Z, radius 3 m, height 10 m.
# Open3D coords are metres, matching the pipeline.
R, H, N_THETA, N_Z = 3.0, 10.0, 400, 200
theta = np.linspace(0, 2 * np.pi, N_THETA, endpoint=False)
z = np.linspace(0, H, N_Z)
TH, Z = np.meshgrid(theta, z)
pts = np.column_stack([(R * np.cos(TH)).ravel(), (R * np.sin(TH)).ravel(), Z.ravel()])

pcd = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(pts))
planes, _residual = detect_planes(pcd, min_inliers=200)

vertical = [p for p in planes if "PLANE_VERTICAL" in str(p.get("shape"))]
print(f"points={len(pts)}  planes={len(planes)}  vertical(wall candidates)={len(vertical)}")

# A 3 m-radius shaft needs ~25 facets within the 25 mm plane tolerance; the old
# cap of 20 would truncate. Expect the shaft to peel into many vertical planes.
assert len(vertical) >= 10, "curved shaft did NOT produce vertical wall planes — detection broken"
print("OK — curved shaft detected as vertical walls")
