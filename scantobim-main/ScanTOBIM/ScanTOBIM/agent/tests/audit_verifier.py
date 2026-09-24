import json
from pathlib import Path

data = json.load(open("reports/FINAL_DETECTION_AUDIT.json"))
segments = data["segments"]

print("=== SEGMENTS SUMMARY ===")
print("Total count:", len(segments))

cats = {}
for s in segments:
    et = s.get("element_type", "UNKNOWN").upper()
    cats[et] = cats.get(et, 0) + 1

for c, cnt in sorted(cats.items()):
    pts = sum(s.get("point_count", 0) for s in segments if s.get("element_type", "").upper() == c)
    confs = [s.get("confidence", 0) for s in segments if s.get("element_type", "").upper() == c]
    avg_conf = sum(confs) / len(confs) if confs else 0
    print(f"{c:25} Count: {cnt:4d} | Pts: {pts:10,d} | Avg Conf: {avg_conf:.3f}")

print("\n=== WALLS DETAIL (8 Walls) ===")
walls = [s for s in segments if s.get("element_type", "").upper() in ("WALL", "BUND_WALL")]
print(f"Found {len(walls)} walls:")
for i, w in enumerate(walls):
    bb = w.get("bounding_box", {})
    dx = bb.get("max_x", 0) - bb.get("min_x", 0)
    dy = bb.get("max_y", 0) - bb.get("min_y", 0)
    dz = bb.get("max_z", 0) - bb.get("min_z", 0)
    tags = w.get("tags", {})
    t = tags.get("wall_thickness_mm", "N/A")
    length = tags.get("wall_length_mm", max(dx, dy))
    sid = w.get("segment_id")
    pts = w.get("point_count")
    st = tags.get("storey_id")
    conf = w.get("confidence")
    print(f"Wall {i+1}: ID={sid} | Pts={pts:,} | Thickness={t}mm | Len={length:.1f}mm | H={dz:.1f}mm | Storey={st} | Conf={conf}")

print("\n=== FLOORS DETAIL (15 Floors) ===")
floors = [s for s in segments if s.get("element_type", "").upper() == "FLOOR"]
print(f"Found {len(floors)} floors:")
for i, f in enumerate(floors):
    bb = f.get("bounding_box", {})
    z_min = bb.get("min_z", 0)
    z_max = bb.get("max_z", 0)
    tags = f.get("tags", {})
    sid = f.get("segment_id")
    pts = f.get("point_count")
    st = tags.get("storey_id")
    print(f"Floor {i+1}: ID={sid} | Pts={pts:,} | Z_min={z_min:.1f}mm | Z_max={z_max:.1f}mm | Storey={st}")

print("\n=== OPENINGS DETAIL ===")
windows = [s for s in segments if s.get("element_type", "").upper() == "WINDOW"]
doors = [s for s in segments if s.get("element_type", "").upper() == "DOOR"]
print(f"Windows: {len(windows)}, Doors: {len(doors)}")
for i, w in enumerate(windows[:5]):
    sid = w.get("segment_id")
    pts = w.get("point_count")
    c = w.get("centroid")
    t = w.get("tags", {})
    print(f"Sample Window {i+1}: ID={sid} | Pts={pts:,} | Centroid={c} | Tags={t}")
for i, d in enumerate(doors):
    sid = d.get("segment_id")
    pts = d.get("point_count")
    c = d.get("centroid")
    t = d.get("tags", {})
    print(f"Door {i+1}: ID={sid} | Pts={pts:,} | Centroid={c} | Tags={t}")

print("\n=== MEP DETAIL ===")
pipes = [s for s in segments if s.get("element_type", "").upper() == "PIPE"]
valves = [s for s in segments if s.get("element_type", "").upper() == "VALVE"]
ducts = [s for s in segments if s.get("element_type", "").upper() == "DUCT"]
print(f"Pipes: {len(pipes)}, Valves: {len(valves)}, Ducts: {len(ducts)}")
for i, v in enumerate(valves):
    sid = v.get("segment_id")
    pts = v.get("point_count")
    c = v.get("centroid")
    t = v.get("tags", {})
    print(f"Valve {i+1}: ID={sid} | Pts={pts} | Centroid={c} | Tags={t}")

print("\n=== REVIT TARGET ELEMENT TYPE BREAKDOWN ===")
revit_targets = {}
for s in segments:
    tags = s.get("tags") or {}
    rt = tags.get("revit_api_target") or "DirectShape"
    revit_targets[rt] = revit_targets.get(rt, 0) + 1

for rt, cnt in sorted(revit_targets.items()):
    print(f"{rt:35}: {cnt:4d}")
print("Total Revit Targets:", sum(revit_targets.values()))
