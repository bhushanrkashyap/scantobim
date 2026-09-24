# ScanToBIM Add-in — Manual Test Guide

Pre-requisites: Revit 2025, .NET 8, `dotnet build` passes, Python agent running on port 8765.

---

## 0. Build the add-in

```
cd revit-addin
dotnet build -c Debug
```

Copy output DLL to Revit add-ins folder:
```
# Windows
xcopy /Y bin\Debug\net8.0-windows\ScanToBIMAgent.dll "%APPDATA%\Autodesk\Revit\Addins\2025\"
xcopy /Y ScanToBIMAgent.addin                         "%APPDATA%\Autodesk\Revit\Addins\2025\"
```

---

## Test 1 — Add-in loads without error

1. Start Revit 2025. Open any project (Architectural or Structural template).
2. **Expected**: "ScanToBIM" tab appears in the ribbon.
3. **Expected**: No startup error dialog.

---

## Test 2 — Load shared parameters

1. Click **ScanToBIM → Load Params**
2. **Expected**: Dialog shows 4 lines:
   ```
   ✓  ScanToBIM_SafetyCategory  — bound to 6 categories
   ✓  ScanToBIM_ScanStationId   — bound to 6 categories
   ✓  ScanToBIM_SegmentId       — bound to 6 categories
   ✓  ScanToBIM_AgentSession    — bound to 6 categories
   ```
3. Run again. **Expected**: All 4 lines show `·  already bound (skipped)` — idempotent.

---

## Test 3 — Start Agent panel

1. Click **ScanToBIM → Start Agent**
2. **Expected**: Dockable "ScanToBIM Agent" panel opens on the right.
3. Panel shows: `○  OFFLINE` and "Start Bridge" button.
4. Click **Start Bridge**.
5. **Expected**: Status changes to `● CONNECTED  port:8766`
6. In a terminal: `curl http://localhost:8766/health`
   **Expected**:
   ```json
   {"Connected":true,"QueueDepth":0,"ElementsCreated":0,"ElementsBlocked":0}
   ```

---

## Test 4 — Create a wall from Python agent

With the bridge running and Python agent running (`uvicorn agent.main:app --port 8765`):

```bash
# Create session (uses synthetic data)
curl -s -X POST http://localhost:8765/sessions \
  -H "Content-Type: application/json" \
  -d '{"site_name":"Revit Manual Test","use_synthetic":true}'
```

**Expected in Revit**:
- Walls, floors, and columns appear in the model.
- Panel log shows lines like:
  ```
  [Factory] ✓  wall      →  revit-1234  (45 ms)  [NS]
  [Factory] ✓  floor     →  revit-5678  (38 ms)  [NS]
  ```
- Stats counters increment: Elements Created > 0.

---

## Test 5 — SC1 hard block

Send a manually crafted SC1 instruction directly to the bridge:

```bash
curl -s -X POST http://localhost:8766/revit-actions \
  -H "Content-Type: application/json" \
  -d '{
    "instruction_id": "test-sc1-001",
    "segment_id":     "seg-sc1-test",
    "zone_id":        "zone-001",
    "element_type":   "wall",
    "discipline":     "structural",
    "safety_category":"SC1",
    "bounding_box":   {"min_x":0,"min_y":0,"min_z":0,"max_x":5000,"max_y":200,"max_z":3000},
    "centroid":       {"x":2500,"y":100,"z":1500}
  }'
```

**Expected in Revit**:
- NO wall is created in the model.
- Panel log shows: `🔴 BLOCKED  wall      [test-sc1] SC1_HARD_BLOCKED: ...`
- Stats: Elements Blocked increments.

Poll result:
```bash
curl http://localhost:8766/revit-results/test-sc1-001
```
**Expected**: `{"Success":false,"Error":"SC1_HARD_BLOCKED: ..."}`

---

## Test 6 — SC2 pending approval (no signature)

```bash
curl -s -X POST http://localhost:8766/revit-actions \
  -H "Content-Type: application/json" \
  -d '{
    "instruction_id":    "test-sc2-001",
    "segment_id":        "seg-sc2-test",
    "zone_id":           "zone-001",
    "element_type":      "column",
    "discipline":        "structural",
    "safety_category":   "SC2",
    "approval_signature": "",
    "bounding_box":      {"min_x":0,"min_y":0,"min_z":0,"max_x":600,"max_y":600,"max_z":3000},
    "centroid":          {"x":300,"y":300,"z":1500}
  }'
```

**Expected**: Panel shows `🟡 SC2 Pending  column`.

Now send again **with a signature**:
```bash
curl -s -X POST http://localhost:8766/revit-actions \
  -H "Content-Type: application/json" \
  -d '{
    "instruction_id":    "test-sc2-002",
    "segment_id":        "seg-sc2-test",
    "zone_id":           "zone-001",
    "element_type":      "column",
    "discipline":        "structural",
    "safety_category":   "SC2",
    "approval_signature": "approved-by:engineer@example.com",
    "approver_upn":      "engineer@example.com",
    "bounding_box":      {"min_x":0,"min_y":0,"min_z":0,"max_x":600,"max_y":600,"max_z":3000},
    "centroid":          {"x":300,"y":300,"z":1500}
  }'
```

**Expected**: Column created in Revit. Panel shows `✓  Created`.

---

## Test 7 — Shared parameters on element

After Test 4 or 6, select any created wall in Revit:
- Open **Properties panel → Identity Data**
- **Expected**: Four parameters visible:
  - `ScanToBIM_SafetyCategory`  → value = `NS` or `SC2`
  - `ScanToBIM_ScanStationId`   → value = segment UUID
  - `ScanToBIM_SegmentId`       → same UUID
  - `ScanToBIM_AgentSession`    → instruction UUID

---

## Test 8 — Shutdown

1. Click **Stop Bridge** in the panel.
2. **Expected**: Log shows `[Bridge] Stopped.`
3. `curl http://localhost:8766/health` → connection refused.
4. Close Revit — no crash or hang.

---

## Pass criteria summary

| Test | Check | Pass |
|------|-------|------|
| 1 | Ribbon tab loads | no error dialog |
| 2 | Load Params | 4 params bound, idempotent |
| 3 | Bridge starts | health endpoint responds |
| 4 | Walls/floors/columns created | model populated, log shows ✓ |
| 5 | SC1 hard block | zero elements created, log shows 🔴 |
| 6 | SC2 without sig blocked, with sig created | correct behaviour both calls |
| 7 | Shared params on element | 4 params populated |
| 8 | Clean shutdown | no crash |
