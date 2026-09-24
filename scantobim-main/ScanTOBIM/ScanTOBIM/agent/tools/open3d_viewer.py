"""Open3D Diagnostic & Demonstration Viewer for Scan-to-BIM Detection Engine.

Reads the exact same `results.json` produced by the Python detection pipeline
and consumed by Revit, visualizing the detected architectural, structural, and
MEP elements in 3D before involving Revit.

Features:
- Visualizes the exact detection representation (no separate calculations).
- Layer/category toggling:
    [1] Toggle Raw Scan Points
    [2] Toggle Walls
    [3] Toggle Floors & Ceilings
    [4] Toggle Pipes, Cylinders & MEP Valves
    [5] Toggle Doors, Windows & Openings
    [6] Toggle Stairs & Ramps
    [P] Print detailed property table (ID, type, angle, thickness, dimensions)
    [S] Save high-resolution screenshot
    [H] Show help & key controls
- Headless rendering mode (--save-views <dir>) to generate manager-demo images.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
from pathlib import Path

import numpy as np
import open3d as o3d


# ── Color Palette (Curated harmonious aesthetic) ──────────────────────────────
COLOR_PALETTE = {
    "wall": [0.95, 0.65, 0.15],        # Warm Amber/Gold
    "bund_wall": [0.85, 0.55, 0.10],   # Deep Amber
    "floor": [0.45, 0.50, 0.55],       # Slate Grey
    "ceiling": [0.65, 0.70, 0.85],     # Muted Lavender
    "grating": [0.35, 0.60, 0.60],     # Steel Cyan
    "kerb": [0.70, 0.60, 0.40],        # Sandstone
    "stair": [0.20, 0.75, 0.35],       # Vibrant Emerald Green
    "ramp": [0.30, 0.85, 0.50],        # Mint Green
    "pipe": [0.88, 0.22, 0.18],        # Crimson Red
    "cylinder": [0.88, 0.22, 0.18],    # Crimson Red
    "valve": [0.75, 0.20, 0.80],       # Magenta/Purple
    "deluge_valve": [0.60, 0.15, 0.85],# Deep Purple
    "strainer": [0.85, 0.40, 0.80],    # Pinkish Purple
    "door": [0.15, 0.65, 0.85],        # Sky Blue
    "window": [0.40, 0.80, 0.95],      # Light Cyan
    "opening": [1.00, 0.90, 0.20],     # Yellow
    "void": [1.00, 0.90, 0.20],        # Yellow
    "beam": [0.55, 0.40, 0.30],        # Timber Brown
    "column": [0.60, 0.35, 0.25],      # Dark Brown
    "raw_scan": [0.75, 0.75, 0.75],    # Muted Light Grey
    "other": [0.50, 0.55, 0.60],       # Neutral Slate
}


def _create_oriented_box_mesh(
    centroid_m: np.ndarray,
    axis_2d: np.ndarray,
    length_m: float,
    thickness_m: float,
    height_m: float,
    color: list[float],
) -> o3d.geometry.TriangleMesh:
    """Create a 3D box mesh aligned along an arbitrary 2D wall axis."""
    axis = axis_2d / max(np.linalg.norm(axis_2d), 1e-6)
    normal = np.array([-axis[1], axis[0]], dtype=float)

    hl = length_m / 2.0
    ht = thickness_m / 2.0
    hh = height_m / 2.0

    # 4 base corners in local coordinates, then rotated
    c_xy = centroid_m[:2]
    z_mid = centroid_m[2]

    # Corner offsets: (±length, ±thickness)
    c1 = c_xy - axis * hl - normal * ht
    c2 = c_xy + axis * hl - normal * ht
    c3 = c_xy + axis * hl + normal * ht
    c4 = c_xy - axis * hl + normal * ht

    z_low = z_mid - hh
    z_high = z_mid + hh

    vertices = np.array([
        [c1[0], c1[1], z_low],
        [c2[0], c2[1], z_low],
        [c3[0], c3[1], z_low],
        [c4[0], c4[1], z_low],
        [c1[0], c1[1], z_high],
        [c2[0], c2[1], z_high],
        [c3[0], c3[1], z_high],
        [c4[0], c4[1], z_high],
    ], dtype=float)

    triangles = np.array([
        # Bottom
        [0, 2, 1], [0, 3, 2],
        # Top
        [4, 5, 6], [4, 6, 7],
        # Front (-normal)
        [0, 1, 5], [0, 5, 4],
        # Back (+normal)
        [2, 3, 7], [2, 7, 6],
        # Left (-axis)
        [0, 4, 7], [0, 7, 3],
        # Right (+axis)
        [1, 2, 6], [1, 6, 5],
    ], dtype=int)

    mesh = o3d.geometry.TriangleMesh()
    mesh.vertices = o3d.utility.Vector3dVector(vertices)
    mesh.triangles = o3d.utility.Vector3iVector(triangles)
    mesh.compute_vertex_normals()
    mesh.paint_uniform_color(color)
    return mesh


def _create_aabb_mesh(
    mins_m: np.ndarray,
    maxs_m: np.ndarray,
    color: list[float],
) -> o3d.geometry.TriangleMesh:
    """Create an axis-aligned bounding box mesh."""
    box = o3d.geometry.AxisAlignedBoundingBox(min_bound=mins_m, max_bound=maxs_m)
    mesh = o3d.geometry.TriangleMesh.create_box(
        width=max(maxs_m[0] - mins_m[0], 0.05),
        height=max(maxs_m[1] - mins_m[1], 0.05),
        depth=max(maxs_m[2] - mins_m[2], 0.05),
    )
    mesh.translate(mins_m)
    mesh.compute_vertex_normals()
    mesh.paint_uniform_color(color)
    return mesh


def _create_cylinder_mesh(
    centroid_m: np.ndarray,
    radius_m: float,
    height_m: float,
    color: list[float],
    axis_3d: np.ndarray | None = None,
) -> o3d.geometry.TriangleMesh:
    """Create a 3D cylinder mesh representing a pipe or circular column."""
    rad = max(radius_m, 0.02)
    h = max(height_m, 0.10)
    cyl = o3d.geometry.TriangleMesh.create_cylinder(radius=rad, height=h, resolution=20)
    cyl.translate(np.array([0, 0, -h / 2.0]))  # Centre cylinder at origin

    if axis_3d is not None and np.linalg.norm(axis_3d) > 1e-4:
        target_axis = axis_3d / np.linalg.norm(axis_3d)
        default_axis = np.array([0.0, 0.0, 1.0])
        dot = float(np.dot(default_axis, target_axis))
        if abs(dot) < 0.999:
            rot_axis = np.cross(default_axis, target_axis)
            rot_axis /= np.linalg.norm(rot_axis)
            angle = math.acos(np.clip(dot, -1.0, 1.0))
            # Rodrigues formula
            R = o3d.geometry.get_rotation_matrix_from_axis_angle(rot_axis * angle)
            cyl.rotate(R, center=np.zeros(3))
        elif dot < 0:
            cyl.rotate(o3d.geometry.get_rotation_matrix_from_axis_angle(np.array([np.pi, 0, 0])), center=np.zeros(3))

    cyl.translate(centroid_m)
    cyl.compute_vertex_normals()
    cyl.paint_uniform_color(color)
    return cyl


def _create_wireframe_box(
    mins_m: np.ndarray,
    maxs_m: np.ndarray,
    color: list[float],
) -> o3d.geometry.LineSet:
    """Create a wireframe box for openings/voids."""
    aabb = o3d.geometry.AxisAlignedBoundingBox(min_bound=mins_m, max_bound=maxs_m)
    line_set = o3d.geometry.LineSet.create_from_axis_aligned_bounding_box(aabb)
    line_set.paint_uniform_color(color)
    return line_set


class DiagnosticViewer:
    """Structured Open3D Demonstration & Validation Viewer."""

    def __init__(self, results_data: dict, scan_cloud: o3d.geometry.PointCloud | None = None):
        self.results = results_data
        self.segments = results_data.get("segments", [])
        self.metadata = results_data.get("metadata", {})
        self.raw_cloud = scan_cloud

        # Category layers: name -> list of Open3D geometries
        self.layers: dict[str, list[o3d.geometry.Geometry]] = {
            "walls": [],
            "floors": [],
            "ceilings": [],
            "pipes_mep": [],
            "stairs_ramps": [],
            "openings": [],
            "other": [],
            "raw_scan": [],
        }

        # Visibility flags
        self.visible: dict[str, bool] = {
            "walls": True,
            "floors": True,
            "ceilings": True,
            "pipes_mep": True,
            "stairs_ramps": True,
            "openings": True,
            "other": True,
            "raw_scan": False if scan_cloud else False,
        }

        self.element_properties: list[dict] = []
        self._build_scene()

    def _build_scene(self) -> None:
        """Parse structured results.json segments and build corresponding 3D geometries."""
        if self.raw_cloud is not None:
            c = o3d.geometry.PointCloud(self.raw_cloud)
            if not c.has_colors():
                c.paint_uniform_color(COLOR_PALETTE["raw_scan"])
            self.layers["raw_scan"].append(c)

        for seg in self.segments:
            elem_type = str(seg.get("element_type", "unknown")).lower()
            shape = str(seg.get("shape", "")).lower()
            tags = seg.get("tags", {}) or {}
            confidence = float(seg.get("confidence", 1.0))
            point_count = int(seg.get("point_count", 0))

            bb = seg.get("bounding_box", {}) or {}
            c_dict = seg.get("centroid", {}) or {}
            centroid_m = np.array([
                float(c_dict.get("x", 0.0)) / 1000.0,
                float(c_dict.get("y", 0.0)) / 1000.0,
                float(c_dict.get("z", 0.0)) / 1000.0,
            ])
            mins_m = np.array([
                float(bb.get("min_x", 0.0)) / 1000.0,
                float(bb.get("min_y", 0.0)) / 1000.0,
                float(bb.get("min_z", 0.0)) / 1000.0,
            ])
            maxs_m = np.array([
                float(bb.get("max_x", 0.0)) / 1000.0,
                float(bb.get("max_y", 0.0)) / 1000.0,
                float(bb.get("max_z", 0.0)) / 1000.0,
            ])

            dims_m = maxs_m - mins_m

            # Record property data for manager inspection
            prop = {
                "id": seg.get("segment_id", "")[:8],
                "type": elem_type,
                "shape": shape,
                "confidence": confidence,
                "points": point_count,
                "centroid_m": centroid_m.tolist(),
                "dims_m": [round(float(d), 2) for d in dims_m],
                "wall_angle_deg": tags.get("wall_angle_deg"),
                "wall_thickness_mm": tags.get("wall_thickness_mm"),
                "pipe_radius_mm": tags.get("radius_mm"),
            }
            self.element_properties.append(prop)

            # 1. Walls & Bund Walls
            if elem_type in ("wall", "bund_wall") or shape == "plane_vertical":
                ax = float(tags.get("wall_axis_x", 1.0))
                ay = float(tags.get("wall_axis_y", 0.0))
                axis_2d = np.array([ax, ay], dtype=float)

                wall_len_m = float(tags.get("wall_length_mm", max(dims_m[0], dims_m[1]) * 1000.0)) / 1000.0
                wall_thick_m = float(tags.get("wall_thickness_mm", 200.0)) / 1000.0
                wall_height_m = max(float(dims_m[2]), 0.50)

                col = COLOR_PALETTE.get(elem_type, COLOR_PALETTE["wall"])
                mesh = _create_oriented_box_mesh(
                    centroid_m=centroid_m,
                    axis_2d=axis_2d,
                    length_m=wall_len_m,
                    thickness_m=wall_thick_m,
                    height_m=wall_height_m,
                    color=col,
                )
                self.layers["walls"].append(mesh)

            # 2. Floors
            elif elem_type in ("floor", "grating") or (shape == "plane_horizontal" and centroid_m[2] < 1.8):
                col = COLOR_PALETTE.get(elem_type, COLOR_PALETTE["floor"])
                if "floor_axis_x" in tags and "floor_length_mm" in tags:
                    f_ax = float(tags.get("floor_axis_x", 1.0))
                    f_ay = float(tags.get("floor_axis_y", 0.0))
                    f_axis = np.array([f_ax, f_ay], dtype=float)
                    f_len_m = float(tags.get("floor_length_mm", dims_m[0] * 1000.0)) / 1000.0
                    f_wid_m = float(tags.get("floor_width_mm", dims_m[1] * 1000.0)) / 1000.0
                    f_thick_m = max(float(dims_m[2]), 0.15)
                    c_x = float(tags.get("floor_center_x_mm", centroid_m[0] * 1000.0)) / 1000.0
                    c_y = float(tags.get("floor_center_y_mm", centroid_m[1] * 1000.0)) / 1000.0
                    f_c = np.array([c_x, c_y, centroid_m[2]], dtype=float)
                    mesh = _create_oriented_box_mesh(
                        centroid_m=f_c,
                        axis_2d=f_axis,
                        length_m=f_len_m,
                        thickness_m=f_wid_m,
                        height_m=f_thick_m,
                        color=col,
                    )
                else:
                    mesh = _create_aabb_mesh(mins_m, maxs_m, color=col)
                self.layers["floors"].append(mesh)

            # 3. Ceilings
            elif elem_type == "ceiling" or (shape == "plane_horizontal" and centroid_m[2] >= 1.8):
                col = COLOR_PALETTE["ceiling"]
                mesh = _create_aabb_mesh(mins_m, maxs_m, color=col)
                self.layers["ceilings"].append(mesh)

            # 4. Pipes, Cylinders & Valves (MEP)
            elif elem_type in ("pipe", "cylinder", "valve", "deluge_valve", "strainer", "junction_box") or shape in ("cylinder", "valve_candidate"):
                col = COLOR_PALETTE.get(elem_type, COLOR_PALETTE["pipe"])
                if shape == "cylinder" or "radius_mm" in tags:
                    rad_m = float(tags.get("radius_mm", 50.0)) / 1000.0
                    h_m = float(max(dims_m))
                    pipe_mesh = _create_cylinder_mesh(centroid_m, rad_m, h_m, color=col)
                    self.layers["pipes_mep"].append(pipe_mesh)
                elif shape == "valve_candidate" or "valve" in elem_type:
                    # Small compact valve sphere/box
                    sphere = o3d.geometry.TriangleMesh.create_sphere(radius=max(min(dims_m) * 0.7, 0.10))
                    sphere.translate(centroid_m)
                    sphere.compute_vertex_normals()
                    sphere.paint_uniform_color(col)
                    self.layers["pipes_mep"].append(sphere)
                else:
                    mesh = _create_aabb_mesh(mins_m, maxs_m, color=col)
                    self.layers["pipes_mep"].append(mesh)

            # 5. Stairs & Ramps
            elif elem_type in ("stair", "ramp") or shape == "plane_sloped":
                col = COLOR_PALETTE.get(elem_type, COLOR_PALETTE["stair"])
                mesh = _create_aabb_mesh(mins_m, maxs_m, color=col)
                self.layers["stairs_ramps"].append(mesh)

            # 6. Doors, Windows & Openings / Voids
            elif elem_type in ("door", "window", "slab_opening", "opening", "void") or shape == "void":
                col = COLOR_PALETTE.get(elem_type, COLOR_PALETTE["opening"])
                if "wall_axis_x" in tags and "wall_axis_y" in tags:
                    w_ax = float(tags.get("wall_axis_x", 1.0))
                    w_ay = float(tags.get("wall_axis_y", 0.0))
                    w_axis = np.array([w_ax, w_ay], dtype=float)
                    op_len_m = float(tags.get("opening_width_mm", max(dims_m[0], dims_m[1]) * 1000.0)) / 1000.0
                    op_thick_m = float(tags.get("wall_thickness_mm", 220.0)) / 1000.0
                    op_height_m = max(float(dims_m[2]), 1.0)
                    wf_box = _create_oriented_box_mesh(
                        centroid_m=centroid_m,
                        axis_2d=w_axis,
                        length_m=op_len_m,
                        thickness_m=op_thick_m,
                        height_m=op_height_m,
                        color=col,
                    )
                    wf_lines = o3d.geometry.LineSet.create_from_triangle_mesh(wf_box)
                    wf_lines.paint_uniform_color(col)
                    self.layers["openings"].append(wf_lines)
                else:
                    wf = _create_wireframe_box(mins_m, maxs_m, color=col)
                    self.layers["openings"].append(wf)

            # 7. Other elements (beams, columns, etc.)
            else:
                col = COLOR_PALETTE.get(elem_type, COLOR_PALETTE["other"])
                mesh = _create_aabb_mesh(mins_m, maxs_m, color=col)
                self.layers["other"].append(mesh)

    def print_summary(self) -> None:
        """Print high-level detection summary for the manager."""
        print("\n" + "=" * 70)
        print("SCAN-TO-BIM STAGE 2 DETECTION SUMMARY")
        print("=" * 70)
        print(f"Source Scan:       {self.metadata.get('input_file', 'unknown')}")
        print(f"File Size:         {self.metadata.get('file_size_bytes', 0) / 1e6:.1f} MB")
        print(f"Total Segments:    {len(self.segments)}")
        print(f"Processing Time:   {self.metadata.get('processing_time_s', 0.0):.2f} s")
        print("-" * 70)
        print("Layer Breakdown:")
        for layer, geoms in self.layers.items():
            print(f"  • {layer.upper():<14}: {len(geoms):>3} visual objects")
        print("=" * 70 + "\n")

    def print_property_table(self) -> None:
        """Print itemized properties for manager inspection."""
        print("\n" + "=" * 95)
        print("ITEMIZED DETECTION PROPERTIES (SAMPLE)")
        print("=" * 95)
        fmt = "{:<10} {:<15} {:<18} {:<10} {:<14} {:<14} {:<12}"
        print(fmt.format("ID", "ELEMENT TYPE", "SHAPE", "CONF", "WALL ANGLE", "THICKNESS/RAD", "POINTS"))
        print("-" * 95)
        for p in self.element_properties[:35]:  # Show representative slice
            ang_str = f"{p['wall_angle_deg']:.1f}°" if p['wall_angle_deg'] is not None else "-"
            thick_str = (
                f"{p['wall_thickness_mm']:.0f}mm"
                if p['wall_thickness_mm'] is not None
                else (f"R={p['pipe_radius_mm']:.0f}mm" if p['pipe_radius_mm'] is not None else "-")
            )
            print(fmt.format(
                p["id"],
                p["type"][:14],
                p["shape"][:17],
                f"{p['confidence']:.2f}",
                ang_str,
                thick_str,
                str(p["points"]),
            ))
        if len(self.element_properties) > 35:
            print(f"... and {len(self.element_properties) - 35} more elements.")
        print("=" * 95 + "\n")

    def save_views(self, output_dir: Path) -> list[Path]:
        """Save standard 3D orthographic and perspective views for reports/demo."""
        output_dir.mkdir(parents=True, exist_ok=True)
        saved_paths = []

        all_geoms = []
        for name, geoms in self.layers.items():
            if name != "raw_scan":  # Keep clean for CAD-like rendering
                all_geoms.extend(geoms)

        if not all_geoms:
            print("No geometries available to render.")
            return []

        vis = o3d.visualization.Visualizer()
        vis.create_window(window_name="ScanToBIM Stage 2 Visualizer", width=1280, height=800, visible=False)

        for g in all_geoms:
            vis.add_geometry(g)

        opt = vis.get_render_option()
        opt.background_color = np.array([0.12, 0.13, 0.15])  # Modern dark mode canvas
        opt.light_on = True

        # View 1: 3D Axonometric Overview
        vis.poll_events()
        vis.update_renderer()
        view_ctl = vis.get_view_control()
        view_ctl.set_front([-0.6, -0.6, 0.5])
        view_ctl.set_up([0, 0, 1])
        vis.poll_events()
        vis.update_renderer()
        p1 = output_dir / "01_stage2_3d_axonometric.png"
        vis.capture_screen_image(str(p1), do_render=True)
        saved_paths.append(p1)

        # View 2: Top Plan View (Floorplan orientation)
        view_ctl.set_front([0, 0, 1])
        view_ctl.set_up([0, 1, 0])
        vis.poll_events()
        vis.update_renderer()
        p2 = output_dir / "02_stage2_top_plan.png"
        vis.capture_screen_image(str(p2), do_render=True)
        saved_paths.append(p2)

        # View 3: Side Elevation View
        view_ctl.set_front([1, 0, 0])
        view_ctl.set_up([0, 0, 1])
        vis.poll_events()
        vis.update_renderer()
        p3 = output_dir / "03_stage2_elevation.png"
        vis.capture_screen_image(str(p3), do_render=True)
        saved_paths.append(p3)

        vis.destroy_window()
        print(f"Generated {len(saved_paths)} high-resolution Open3D diagnostic views in {output_dir}:")
        for p in saved_paths:
            print(f"  • {p.name}")
        return saved_paths

    def run_interactive(self) -> None:
        """Launch the interactive Open3D window with keyboard shortcuts."""
        vis = o3d.visualization.VisualizerWithKeyCallback()
        vis.create_window(window_name="ScanToBIM Stage 2 — Manager Demonstration", width=1440, height=900)

        # Add initial visible geometries
        active_geoms: list[o3d.geometry.Geometry] = []
        for name, is_vis in self.visible.items():
            if is_vis:
                for g in self.layers[name]:
                    vis.add_geometry(g, reset_bounding_box=(len(active_geoms) == 0))
                    active_geoms.append(g)

        opt = vis.get_render_option()
        opt.background_color = np.array([0.10, 0.11, 0.13])
        opt.light_on = True

        def toggle_layer(layer_name: str, key_char: str):
            def callback(visualizer):
                self.visible[layer_name] = not self.visible[layer_name]
                state_str = "VISIBLE" if self.visible[layer_name] else "HIDDEN"
                print(f"[Key {key_char}] Layer '{layer_name.upper()}' is now {state_str} ({len(self.layers[layer_name])} items)")

                # Rebuild visible geometries
                visualizer.clear_geometries()
                for l_name, vis_flag in self.visible.items():
                    if vis_flag:
                        for geom in self.layers[l_name]:
                            visualizer.add_geometry(geom, reset_bounding_box=False)
                visualizer.poll_events()
                visualizer.update_renderer()
                return False
            return callback

        # Register key shortcuts
        vis.register_key_callback(ord("1"), toggle_layer("raw_scan", "1"))
        vis.register_key_callback(ord("2"), toggle_layer("walls", "2"))
        vis.register_key_callback(ord("3"), toggle_layer("floors", "3"))
        vis.register_key_callback(ord("4"), toggle_layer("pipes_mep", "4"))
        vis.register_key_callback(ord("5"), toggle_layer("openings", "5"))
        vis.register_key_callback(ord("6"), toggle_layer("stairs_ramps", "6"))

        def print_props(visualizer):
            self.print_property_table()
            return False

        vis.register_key_callback(ord("P"), print_props)
        vis.register_key_callback(ord("p"), print_props)

        print("\n" + "=" * 70)
        print("OPEN3D DEMO INTERACTIVE CONTROLS:")
        print("  [1] Toggle Raw Scan Points")
        print("  [2] Toggle Walls (45.8° / 135.8° preserved)")
        print("  [3] Toggle Floors & Ceilings")
        print("  [4] Toggle Pipes & MEP Equipment")
        print("  [5] Toggle Doors & Openings")
        print("  [6] Toggle Stairs & Ramps")
        print("  [P] Print Itemized Properties Table")
        print("  [Q] Exit Viewer")
        print("=" * 70 + "\n")

        vis.run()
        vis.destroy_window()


def main():
    parser = argparse.ArgumentParser(description="Open3D Stage 2 Diagnostic & Demo Viewer")
    parser.add_argument("--results", required=True, help="Path to results.json (detection output)")
    parser.add_argument("--scan", default=None, help="Optional raw scan file (.ply or .e57) for overlay")
    parser.add_argument("--save-views", default=None, help="Directory to save PNG view renders into")
    parser.add_argument("--no-interactive", action="store_true", help="Do not open GUI window")
    args = parser.parse_args()

    results_path = Path(args.results)
    if not results_path.exists():
        print(f"Error: results file '{results_path}' not found.", file=sys.stderr)
        sys.exit(1)

    with open(results_path) as f:
        results_data = json.load(f)

    scan_cloud = None
    if args.scan:
        scan_path = Path(args.scan)
        if scan_path.exists():
            print(f"Loading reference scan '{scan_path}' for overlay...")
            if scan_path.suffix.lower() == ".ply":
                scan_cloud = o3d.io.read_point_cloud(str(scan_path))
                if len(scan_cloud.points) > 100000:
                    scan_cloud = scan_cloud.voxel_down_sample(voxel_size=0.08)
            elif scan_path.suffix.lower() == ".e57":
                try:
                    from agent.tools.scan_tools import _load_e57
                    scan_cloud = _load_e57(scan_path)
                    if len(scan_cloud.points) > 100000:
                        scan_cloud = scan_cloud.voxel_down_sample(voxel_size=0.08)
                except Exception as ex:
                    print(f"Warning: Could not load scan overlay: {ex}")

    viewer = DiagnosticViewer(results_data, scan_cloud=scan_cloud)
    viewer.print_summary()

    if args.save_views:
        viewer.save_views(Path(args.save_views))

    if not args.no_interactive and "DISPLAY" in os.environ:
        viewer.run_interactive()


if __name__ == "__main__":
    main()
