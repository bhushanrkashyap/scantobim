"""Central detection parameters — single source of truth for algorithm thresholds.

No inline magic numbers in the detection pipeline (CLAUDE.md "Generalization over
hardcoding"). Values trace to the algorithm reference cards A1–A3 (pre-processing).
"""

# A1 · Voxel downsample.
# Operational default kept at 20 mm for cost/backward-compat. Reference A1 canonical
# is 0.01 (10 mm) — pass voxel_size_m=0.01 for the fine structural pass.
VOXEL_SIZE_M = 0.02

# A2 · Statistical Outlier Removal. nb_neighbors adapts down for small clouds so SOR
# does not wipe them out; SOR_MIN_NEIGHBORS keeps it meaningful.
SOR_NB_NEIGHBORS = 20
SOR_STD_RATIO = 2.0
SOR_MIN_NEIGHBORS = 6
SOR_NEIGHBOR_DIVISOR = 50

# A3 · Surface normal estimation. Radius derived at runtime as MULT × voxel so it
# tracks point spacing across scan densities (0.10 @20 mm, 0.05 @10 mm — both 5×).
NORMAL_RADIUS_VOXEL_MULT = 5.0
NORMAL_MAX_NN = 30

import os

# B1 · Iterative RANSAC plane cap. The loop already self-terminates when a plane
# drops below min_inliers, so this is only a worst-case cost ceiling — NOT a
# detection threshold. Configurable via STB_MAX_PLANES (defaults to 500 for large scenes).
MAX_PLANES = int(os.environ.get("STB_MAX_PLANES", "500"))

# ── BIM export geometry defaults ────────────────────────────────────────────────
# A single scanned surface has no measurable thickness, so planar elements (walls,
# slabs) need an assumed thickness to render as a solid in Revit. Physical defaults.
# LIMITATION: fixed, not derived from the scan — tune per project / building code.
WALL_THICKNESS_MM = 200.0  # typical wall thickness (applied to the thin horizontal axis)
DEFAULT_WALL_THICKNESS_MM = WALL_THICKNESS_MM  # Inferred thickness for single-face wall scans
MIN_WALL_THICKNESS_MM = 100.0  # Lower bound to prevent razor-thin walls
MAX_WALL_THICKNESS_MM = 600.0  # Upper bound to prevent massive/pathological walls
SLAB_THICKNESS_MM = 200.0  # floor/ceiling slab thickness (applied to the thin Z axis)
MIN_ELEMENT_DIM_MM = 10.0  # floor on every axis to prevent degenerate (zero-dim) IFC solids

# ── Optional semantic wall masking (DL front-end) ───────────────────────────
# Runtime toggle: when enabled, a semantic model can pre-filter wall points
# before geometric RANSAC. Geometry-only mode remains the default.
SEMANTIC_ENABLE = False
SEMANTIC_MODEL = "pointnext"  # pointnext|pointmetabase|ptv1|ptv3|swin3d|randlanet
SEMANTIC_WALL_THRESHOLD = 0.50
SEMANTIC_WALL_CLASS_ID = 1
SEMANTIC_MIN_WALL_POINTS = 200
SEMANTIC_FALLBACK_TO_GEOMETRY = True

# ── Stage 2 Architectural Wall & Curvature Detection Thresholds ─────────────
# Baseline geometric thresholds per Scan-to-BIM Stage 2 specification
WALL_MIN_HEIGHT_M = 0.40
WALL_MIN_LENGTH_M = 0.30
WALL_MIN_POINT_COUNT = 100
WALL_MIN_DENSITY = 5.0  # pts/m²
WALL_ANGLE_TOL_DEG = 5.0
WALL_COPLANAR_OFFSET_M = 0.08  # 80mm lateral offset tolerance (protects separate parallel faces)
WALL_FRAGMENT_GAP_M = 1.20  # 1200mm fragment bridging tolerance (bridges doorways/windows)
WALL_Z_OVERLAP = 0.30  # 30% vertical overlap for multi-storey continuity

# Curvature / Cylindrical facet clustering thresholds
CURVATURE_NORMAL_STEP_MAX_DEG = 35.0
CURVATURE_RADIAL_TOL_M = 0.45
CURVATURE_MIN_FACETS = 3

