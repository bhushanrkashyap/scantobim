"""CPU-safe, deterministic 3D point cloud augmentations for Phase 3C."""

from __future__ import annotations

import numpy as np


class PointCloudAugmentor:
    """Applies rigid and semi-rigid augmentations to 3D point clouds and features."""

    def __init__(
        self,
        rotate_z: bool = True,
        jitter_std: float = 0.005,
        scale_range: tuple[float, float] = (0.95, 1.05),
        seed: int = 42,
    ) -> None:
        self.rotate_z = rotate_z
        self.jitter_std = jitter_std
        self.scale_range = scale_range
        self.rng = np.random.RandomState(seed)

    def __call__(self, coords: np.ndarray, features: np.ndarray | None = None) -> tuple[np.ndarray, np.ndarray | None]:
        """Apply augmentations to coordinates and optionally features."""
        aug_coords = coords.copy()

        # 1. Random rotation around Z axis (preserving vertical orientation)
        if self.rotate_z:
            theta = self.rng.uniform(0.0, 2.0 * np.pi)
            cos_t, sin_t = np.cos(theta), np.sin(theta)
            rot_z = np.array([
                [cos_t, -sin_t, 0.0],
                [sin_t,  cos_t, 0.0],
                [0.0,    0.0,   1.0],
            ], dtype=np.float32)
            aug_coords = np.dot(aug_coords, rot_z)

        # 2. Random isotropic scaling
        if self.scale_range is not None:
            scale = self.rng.uniform(self.scale_range[0], self.scale_range[1])
            aug_coords *= scale

        # 3. Random Gaussian coordinate jitter
        if self.jitter_std > 0:
            jitter = self.rng.normal(0.0, self.jitter_std, size=aug_coords.shape).astype(np.float32)
            aug_coords += jitter

        # Update spatial features if present
        aug_features = None
        if features is not None:
            aug_features = features.copy()
            aug_features[:, :3] = aug_coords

        return aug_coords, aug_features
