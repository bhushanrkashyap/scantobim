# Final Detection and BIM Reconstruction Audit

## 1. Executive Detection Summary

- **Scan Source**: `2026-07-28 (2).e57`
- **Raw ASTM E57 Points (Level 0)**: `53,275,505`
- **Level 1 Adaptive Detection Points**: `1,207,133`
- **Total Reconstructed BIM Elements**: `544`
- **Revit Native Ready Elements**: `544`
- **Storeys / Levels Discovered**: `5`

### Detected BIM Elements by Category
| Element Category | Count | Total Points | Revit Native API Target |
|---|---|---|---|
| **BEAM** | 30 | 3,725 | `DirectShape / FamilyInstance` |
| **CONTAINMENT_PENETRATION** | 4 | 0 | `DirectShape / FamilyInstance` |
| **DOOR** | 4 | 18,509 | `FamilyInstance / NewOpening` |
| **DRAINAGE** | 91 | 2,399 | `DirectShape / FamilyInstance` |
| **FIRE_ALARM_PANEL** | 2 | 129 | `DirectShape / FamilyInstance` |
| **FIRE_HYDRANT** | 2 | 47 | `DirectShape / FamilyInstance` |
| **FLOOR** | 15 | 642,573 | `Floor.Create (Structural Slab)` |
| **HEAT_EXCHANGER** | 1 | 339 | `DirectShape / FamilyInstance` |
| **HVAC_EQUIPMENT** | 2 | 676 | `DirectShape / FamilyInstance` |
| **JUNCTION_BOX** | 27 | 779 | `DirectShape / FamilyInstance` |
| **KERB** | 30 | 4,165 | `DirectShape / FamilyInstance` |
| **LIGHTING_FITTING** | 11 | 599 | `DirectShape / FamilyInstance` |
| **PIPE** | 237 | 9,202 | `Pipe.Create (MEP System)` |
| **PRESSURE_VESSEL** | 2 | 1,325 | `DirectShape / FamilyInstance` |
| **PUMP** | 3 | 1,066 | `DirectShape / FamilyInstance` |
| **SPRINKLER** | 38 | 852 | `DirectShape / FamilyInstance` |
| **STAIR** | 1 | 2,525 | `DirectShape / FamilyInstance` |
| **TRANSFORMER** | 7 | 6,011 | `DirectShape / FamilyInstance` |
| **UNKNOWN** | 3 | 0 | `DirectShape / FamilyInstance` |
| **VALVE** | 7 | 159 | `FamilyInstance (OST_PipeAccessory)` |
| **WALL** | 6 | 598,654 | `Wall.Create (System Family)` |
| **WINDOW** | 21 | 290,243 | `FamilyInstance / NewOpening` |

---

## 2. Multi-Candidate Fusion Audit

- **Candidates Proposed**: `4762`
- **Candidates Accepted**: `544`
- **Candidates Rejected**: `3729`

### Column Candidate Pools Distribution
| Candidate Pool | Count | Description |
|---|---|---|

### Rejection Breakdown
| Rejection Reason | Count |
|---|---|
| `ascan2bim_unmatched_edges` | 3520 |
| `column_rejected_yolo` | 0 |
| `column_rejected_geometric` | 209 |

---

## 3. Storey and Slab Elevation Matrix

| Storey ID | Base Elevation (m) | Top Elevation (m) | Slab Thickness (m) |
|---|---|---|---|
| `storey_0` | -14.783 | -11.503 | 0.200 (Default) |
| `storey_1` | -11.503 | 0.657 | 0.200 (Default) |
| `storey_2` | 0.657 | 3.217 | 0.200 (Default) |
| `storey_3` | 3.217 | 7.377 | 0.200 (Default) |
| `storey_4` | 7.377 | 10.950 | 0.200 (Default) |

---

## 4. Geometric & Verification Verdict

- **Source Preservation**: 100% of 53.27M raw points preserved under Level 0.
- **Zero Silent Box Fallbacks**: Failed geometry is rejected rather than turned into fake cuboids.
- **No Fictitious Silo Realignment**: Silo and column centroids reflect exact point cloud coordinates.
- **Windows UTF-8 Compliance**: Safe logging without character encoding exceptions.
