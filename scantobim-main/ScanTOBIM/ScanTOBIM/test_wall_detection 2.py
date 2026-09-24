import os
import sys
import numpy as np
import open3d as o3d

from agent.tools.scan_tools import detect_planes


def run_wall_detection(pcd_path: str | None = None, visualize: bool = False):
    """Run interactive plane and wall detection on an E57 point cloud file."""
    import pye57

    if not pcd_path:
        pcd_path = os.environ.get("PCD_PATH", "")
    if not pcd_path or not os.path.exists(pcd_path):
        print("Usage: python test_wall_detection.py <path_to_e57>")
        return

    print("=" * 60)
    print("Loading Point Cloud:", pcd_path)
    print("=" * 60)

    e57 = pye57.E57(pcd_path)

    if e57.scan_count == 0:
        raise RuntimeError("The E57 file contains no scans.")

    scan = e57.read_scan_raw(0, ignore_unsupported_fields=True)

    required_fields = {"cartesianX", "cartesianY", "cartesianZ"}
    missing_fields = required_fields.difference(scan)
    if missing_fields:
        raise RuntimeError(
            f"The first E57 scan is missing required fields: {sorted(missing_fields)}"
        )

    xyz = np.column_stack([
        scan["cartesianX"],
        scan["cartesianY"],
        scan["cartesianZ"]
    ])

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(xyz)

    print("Loaded points:", len(pcd.points))

    if len(pcd.points) == 0:
        raise RuntimeError("The first E57 scan contains no Cartesian points.")

    print("=" * 60)
    print("Detecting Planes")
    print("=" * 60)

    planes, residual = detect_planes(pcd, min_inliers=200)

    print("Plane Count:", len(planes))
    if len(planes) == 0:
        raise RuntimeError("No planes detected")

    print("\nFIRST PLANE")
    print(planes[0])

    wall_visuals = []
    for idx, plane in enumerate(planes):
        shape = str(plane.get("shape", "")).lower()
        if "vertical" not in shape:
            continue
        wall = plane.get("inlier_cloud")
        if wall is not None:
            wall.paint_uniform_color([1, 0, 0])
            wall_visuals.append(wall)
            try:
                obb = wall.get_oriented_bounding_box()
                obb.color = (1, 1, 0)
                wall_visuals.append(obb)
            except Exception:
                pass

    print(f"\nVisual Wall Objects: {len(wall_visuals)}")

    if visualize and len(wall_visuals) > 0:
        o3d.visualization.draw_geometries(wall_visuals, window_name="Detected Walls")


if __name__ == "__main__":
    target_path = sys.argv[1] if len(sys.argv) > 1 else None
    run_wall_detection(target_path, visualize=True)