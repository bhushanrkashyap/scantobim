from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from agent.tools.detection_config import (
    NORMAL_MAX_NN,
    NORMAL_RADIUS_VOXEL_MULT,
    SOR_MIN_NEIGHBORS,
    SOR_NB_NEIGHBORS,
    SOR_NEIGHBOR_DIVISOR,
    SOR_STD_RATIO,
)
from agent.tools.scan_tools import downsample, load_point_cloud, remove_statistical_outliers
from agent.tools.geometry_tools import estimate_normals


def export_semantic_points(input_path: Path, output_path: Path, voxel_size_m: float) -> Path:
    pcd = load_point_cloud(input_path)
    if len(pcd.points) == 0:
        raise ValueError("Input point cloud is empty.")

    sor_neighbors = min(
        SOR_NB_NEIGHBORS,
        max(SOR_MIN_NEIGHBORS, len(pcd.points) // SOR_NEIGHBOR_DIVISOR),
    )
    pcd = remove_statistical_outliers(
        pcd,
        nb_neighbors=sor_neighbors,
        std_ratio=SOR_STD_RATIO,
    )

    if len(pcd.points) == 0:
        raise ValueError("No points left after SOR filtering.")

    pcd = downsample(pcd, voxel_size_m)
    if len(pcd.points) == 0:
        raise ValueError("No points left after voxel downsampling.")

    normal_result = estimate_normals(
        pcd,
        radius=NORMAL_RADIUS_VOXEL_MULT * voxel_size_m,
        max_nn=NORMAL_MAX_NN,
    )
    if not normal_result.get("success"):
        raise RuntimeError(f"Normal estimation failed: {normal_result.get('error', 'unknown')}")

    pcd = normal_result["pcd"]
    pcd.translate(normal_result["origin"])

    points_m = np.asarray(pcd.points, dtype=np.float64)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    np.save(output_path, points_m)
    return output_path


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="export-semantic-points",
        description="Export STB semantic-stage points after preprocessing.",
    )
    parser.add_argument("--input", "-i", required=True, type=Path, help="Input scan (.e57/.las/.laz/.ply/.pcd/.xyz)")
    parser.add_argument("--output", "-o", required=True, type=Path, help="Output .npy path for semantic-stage points")
    parser.add_argument(
        "--voxel-size",
        type=float,
        default=0.01,
        metavar="M",
        help="Voxel size in metres (must match processor run)",
    )
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    out = export_semantic_points(args.input, args.output, args.voxel_size)
    arr = np.load(out, allow_pickle=False)
    print(
        f"Exported semantic-stage points to {out} | count={arr.shape[0]} | dims={arr.shape[1] if arr.ndim > 1 else 1}",
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
