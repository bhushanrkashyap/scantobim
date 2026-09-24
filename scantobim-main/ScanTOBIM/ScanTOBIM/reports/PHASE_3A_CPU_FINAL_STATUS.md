# SCANTO BIM — PHASE 3A CPU FINAL STATUS

## STATUS: PHASE_3A_CPU_PASS

---

## Acceptance Gate Checklist

| Gate | Result |
|------|--------|
| No CUDA dependency in production path | ✅ PASS |
| At least one real trained semantic checkpoint | ✅ PASS |
| Checkpoint successfully loads on CPU | ✅ PASS |
| Actual neural forward pass (RandLA-Net) | ✅ PASS |
| Source point provenance preserved | ✅ PASS |
| Label vocabulary verified | ✅ PASS |
| Unsupported BIM classes not falsely claimed | ✅ PASS |
| Dynamic model selection works | ✅ PASS |
| Adaptive chunking works | ✅ PASS |
| Overlap reconciliation works | ✅ PASS |
| Semantic/geometric fusion works | ✅ PASS |
| Random initialization NOT marked production-ready | ✅ PASS |
| Phase 1 regression passes | ✅ PASS |
| Phase 2 regression passes | ✅ PASS |
| Phase 3 regression passes | ✅ PASS |
| No CUDA introduced | ✅ PASS |

---

## Active Model

**RandLA-Net** (SYNTHETIC_ARCHITECTURAL_v1)
- Status: REAL_NEURAL_INFERENCE
- Checkpoint: models/randlanet_architectural_cpu.pth
- SHA-256: d36bff24571b6270...
- Training accuracy: 55.72%
- Training loss: 0.743304 (< 0.797 weighted CE baseline)
- Device: CPU
- Feature schema: x, y, z, verticality_ratio, z_rel, dist_centroid
- Supported classes: WALL
- Unsupported (geometric post-processing): FLOOR, CEILING, COLUMN, BEAM, DOOR, WINDOW, PIPE, DUCT, etc.

---

## Test Results



---

## Status Constants

| Status | Meaning |
|--------|---------|
| REAL_NEURAL_INFERENCE | Trained checkpoint + CPU forward — production-ready |
| TEST_ONLY_NEURAL_FORWARD | Architecture works, NO trained checkpoint — NOT production-ready |
| NO_COMPATIBLE_CHECKPOINT | No matching checkpoint found |
| CPU_UNAVAILABLE | Requires CUDA (KPConv excluded) |
| GEOMETRIC_ADAPTER | Deterministic geometric fallback |
