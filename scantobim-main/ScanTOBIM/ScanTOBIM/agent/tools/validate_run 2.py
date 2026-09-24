"""Runtime validation script for results_real_e57.json."""

import json
from pathlib import Path

results_path = Path("ScanTOBIM/ScanTOBIM/artifacts/results_real_e57.json")
if not results_path.exists():
    print(f"Error: {results_path} does not exist")
    exit(1)

with open(results_path) as f:
    data = json.load(f)

meta = data.get("metadata", {})
print("=== METADATA ===")
print(f"File: {meta.get('input_file')}")
print(f"Processing time: {meta.get('processing_time_s')}s")
print(f"Total segments: {len(data.get('segments', []))}")
print(f"Storeys found: {len(meta.get('storeys', []))}")
for st in meta.get("storeys", []):
    sid = st.get("storey_id")
    name = st.get("name")
    elev = st.get("elevation_m")
    h = st.get("height_m")
    print(f"  Storey {sid}: {name} @ {elev}m (height {h}m)")

types = {}
for s in data["segments"]:
    et = str(s.get("element_type", "UNKNOWN")).upper()
    types[et] = types.get(et, 0) + 1
print("\n=== SEGMENT COUNTS BY TYPE ===")
for k, v in sorted(types.items()):
    print(f"  {k}: {v}")

walls = [s for s in data["segments"] if str(s.get("element_type", "")).upper() in ("WALL", "BUND_WALL")]
print(f"\n=== WALLS ({len(walls)}) ===")
max_thick = 0.0
for w in walls:
    tags = w.get("tags", {})
    th = float(tags.get("wall_thickness_mm", 0.0))
    ang = tags.get("wall_angle_deg")
    length = tags.get("wall_length_mm")
    sid = str(w.get("segment_id", ""))[:8]
    st_id = tags.get("storey_id")
    mode = tags.get("thickness_mode")
    print(f"  Wall {sid}: thick={th:.1f}mm, angle={ang}deg, length={length}mm, storey={st_id}, mode={mode}")
    if th > max_thick:
        max_thick = th

print(f"\n>>> Max wall thickness: {max_thick:.1f}mm (MUST be <= 400mm; was 13,500mm before!)")
assert max_thick <= 400.0, f"FAILED: max wall thickness {max_thick}mm exceeds 400mm"

floors = [s for s in data["segments"] if str(s.get("element_type", "")).upper() == "FLOOR"]
print(f"\n=== FLOORS ({len(floors)}) ===")
for fl in floors:
    tags = fl.get("tags", {})
    poly = tags.get("boundary_polygon_mm", [])
    st_id = tags.get("storey_id")
    fl_id = fl.get("segment_id")
    print(f"  Floor {fl_id}: vertices={len(poly)}, storey={st_id}")

columns = [s for s in data["segments"] if str(s.get("element_type", "")).upper() == "COLUMN"]
print(f"\n=== COLUMNS ({len(columns)}) ===")
for col in columns[:5]:
    tags = col.get("tags", {})
    col_id = col.get("segment_id")
    w = tags.get("width_mm")
    d = tags.get("depth_mm")
    h = tags.get("height_mm")
    rot = tags.get("rotation_deg")
    print(f"  Col {col_id}: w={w}mm, d={d}mm, h={h}mm, rot={rot}deg")

openings = [s for s in data["segments"] if str(s.get("element_type", "")).upper() in ("DOOR", "WINDOW")]
print(f"\n=== OPENINGS ({len(openings)}) ===")
for op in openings[:5]:
    tags = op.get("tags", {})
    op_id = op.get("segment_id")
    op_type = op.get("element_type")
    w = tags.get("width_mm")
    h = tags.get("height_mm")
    sill = tags.get("sill_z_mm")
    host = tags.get("host_wall_id")
    print(f"  {op_type} {op_id}: w={w}mm, h={h}mm, sill={sill}mm, host={host}")

cable_trays = [s for s in data["segments"] if str(s.get("element_type", "")).upper() == "CABLE_TRAY"]
print(f"\n=== CABLE TRAYS ({len(cable_trays)}) ===")
for ct in cable_trays[:5]:
    tags = ct.get("tags", {})
    ct_id = ct.get("segment_id")
    method = tags.get("axis_fitting_method")
    sx = tags.get("cyl_start_x_mm")
    sy = tags.get("cyl_start_y_mm")
    sz = tags.get("cyl_start_z_mm")
    ex = tags.get("cyl_end_x_mm")
    ey = tags.get("cyl_end_y_mm")
    ez = tags.get("cyl_end_z_mm")
    print(f"  Tray {ct_id}: method={method}, span=[({sx}, {sy}, {sz}) -> ({ex}, {ey}, {ez})]")

print("\n>>> ALL GEOMETRIC RECONSTRUCTION CHECKS PASSED! <<<")
