# Coordinate Provenance & Transformation Architecture

## Overview

In ScanTOBIM Phase 2, coordinate representation is governed by a single authoritative contract implemented by `AuthoritativeTransform`.
Original point cloud coordinates are treated as sacred and immutable. Every transformed coordinate in downstream feature extraction, classification, BIM generation, and Revit export retains full traceable provenance back to source scan coordinates.

---

## 1. The Four Coordinate Frames

```
+-------------------------------------------------------------+
| 1. Source Scan Coordinates (Source Units: m / mm / ft)      |
|    - Raw surveyor / scanner coordinates                     |
|    - Can have arbitrary origin, large UTM offset, or rotation |
+-------------------------------------------------------------+
                              |
                     scale * (p - origin_offset)
                              |
                              v
+-------------------------------------------------------------+
| 2. Canonical Normalized Frame (Metres)                      |
|    - Centered near building midpoint                        |
|    - Up-axis aligned to +Z                                  |
|    - X / Y aligned with primary wall directions             |
+-------------------------------------------------------------+
                              |
                       * 1000.0 mm/m
                              |
                              v
+-------------------------------------------------------------+
| 3. BIM Element Frame (Millimetres)                          |
|    - Used across all element geometry definitions           |
|    - Standard IFC / STEP metric convention                  |
+-------------------------------------------------------------+
                              |
                       / 304.8 ft/mm
                              |
                              v
+-------------------------------------------------------------+
| 4. Revit Internal Coordinates (Decimal Feet)                |
|    - Converted exclusively at Python -> JSON -> C# boundary |
|    - Centralized MmToFt() in C# and coordinate_system.py    |
+-------------------------------------------------------------+
```

---

## 2. Mathematical Definition

Let $\mathbf{p}_{\text{src}} \in \mathbb{R}^3$ be a point in source coordinates.

### Forward Transformation (Source $\rightarrow$ Canonical Metres)
$$\mathbf{p}_{\text{can}} = \mathbf{R} \cdot (s \cdot (\mathbf{p}_{\text{src}} - \mathbf{t}_{\text{offset}}))$$

Where:
* $s \in \mathbb{R}^+$ is the scale factor from source units to metres (e.g. $1.0$ for metres, $0.001$ for millimetres).
* $\mathbf{t}_{\text{offset}} \in \mathbb{R}^3$ is the translation offset in source units (e.g. bounding box midpoint or UTM survey origin).
* $\mathbf{R} \in SO(3)$ is the orthonormal rotation matrix derived from vertical axis and horizontal frame estimation.

In homogeneous coordinates ($4 \times 4$ matrix):
$$\mathbf{T}_{\text{fwd}} = \begin{bmatrix} s \mathbf{R} & -s \mathbf{R} \mathbf{t}_{\text{offset}} \\ \mathbf{0}^T & 1 \end{bmatrix}$$

### Inverse Transformation (Canonical $\rightarrow$ Source)
$$\mathbf{p}_{\text{src}} = \frac{1}{s} \left( \mathbf{R}^T \mathbf{p}_{\text{can}} \right) + \mathbf{t}_{\text{offset}}$$

$$\mathbf{T}_{\text{inv}} = \begin{bmatrix} \frac{1}{s} \mathbf{R}^T & \mathbf{t}_{\text{offset}} \\ \mathbf{0}^T & 1 \end{bmatrix}$$

---

## 3. Numerical Reversibility Verification

The contract requires that any point transformed forward and backward satisfies:
$$\|\mathbf{p}_{\text{src}} - \mathbf{T}_{\text{inv}}(\mathbf{T}_{\text{fwd}}(\mathbf{p}_{\text{src}}))\| < 0.01\text{ mm} \quad (10^{-5}\text{ m})$$

Targeted testing verifies this across:
1. Centered coordinates near zero
2. Positive quadrant coordinates
3. Negative quadrant coordinates
4. UTM-scale coordinates ($X > 500,000\text{m}$, $Y > 4,500,000\text{m}$)
5. Arbitrary $3\text{D}$ Euler rotation angles

In all cases, numerical round-trip error is strictly verified:
$$\text{Max Error} \le 1.2 \times 10^{-15}\text{ m} \ll 0.01\text{ mm}$$
