from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np


def _load_predictions(path: Path) -> np.ndarray:
    suffix = path.suffix.lower()
    if suffix == ".npy":
        arr = np.load(path, allow_pickle=False)
    elif suffix == ".npz":
        z = np.load(path, allow_pickle=False)
        if "wall_mask" in z:
            arr = z["wall_mask"]
        elif "scores" in z:
            arr = z["scores"]
        elif "class_ids" in z:
            arr = z["class_ids"]
        else:
            keys = list(z.keys())
            if not keys:
                raise ValueError("Empty npz file")
            arr = z[keys[0]]
    else:
        raise ValueError("Prediction file must be .npy or .npz")

    arr = np.asarray(arr)
    if arr.ndim != 1:
        arr = arr.reshape(-1)
    return arr


def _to_wall_mask(preds: np.ndarray, mode: str, wall_class_id: int, threshold: float) -> np.ndarray:
    if mode == "bool":
        return preds.astype(bool)

    if mode == "class":
        return preds.astype(np.int64) == int(wall_class_id)

    if mode == "prob":
        return preds.astype(np.float64) >= float(threshold)

    # auto
    if preds.dtype == bool:
        return preds.astype(bool)
    if np.issubdtype(preds.dtype, np.integer):
        return preds.astype(np.int64) == int(wall_class_id)
    if np.issubdtype(preds.dtype, np.floating):
        return preds.astype(np.float64) >= float(threshold)

    try:
        return preds.astype(np.float64) >= float(threshold)
    except Exception as exc:  # noqa: BLE001
        raise ValueError(f"Unsupported prediction dtype: {preds.dtype}") from exc


def build_checkpoint(
    points_path: Path,
    predictions_path: Path,
    output_path: Path,
    mode: str,
    wall_class_id: int,
    threshold: float,
) -> Path:
    points = np.load(points_path, allow_pickle=False)
    if points.ndim != 2 or points.shape[1] != 3:
        raise ValueError("Points file must be an (N,3) array in metres")

    preds = _load_predictions(predictions_path)
    if preds.shape[0] != points.shape[0]:
        raise ValueError(
            f"Prediction length mismatch: preds={preds.shape[0]} vs points={points.shape[0]}. "
            "Ensure RandLA-Net used the exact exported semantic-stage point set."
        )

    wall_mask = _to_wall_mask(preds, mode, wall_class_id, threshold)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    if output_path.suffix.lower() == ".npz":
        np.savez_compressed(output_path, wall_mask=wall_mask)
    else:
        np.save(output_path, wall_mask)

    return output_path


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="build-semantic-checkpoint",
        description="Build an STB-compatible wall mask checkpoint from model predictions.",
    )
    parser.add_argument("--points", required=True, type=Path, help="Semantic-stage points .npy exported from STB")
    parser.add_argument("--predictions", required=True, type=Path, help="RandLA-Net predictions file (.npy/.npz)")
    parser.add_argument("--output", required=True, type=Path, help="Output wall mask file (.npy or .npz)")
    parser.add_argument(
        "--mode",
        choices=["auto", "bool", "class", "prob"],
        default="auto",
        help="Interpretation mode for predictions",
    )
    parser.add_argument("--wall-class-id", type=int, default=1, help="Wall class id when mode is class/auto-integer")
    parser.add_argument("--threshold", type=float, default=0.50, help="Threshold when mode is prob/auto-float")
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    out = build_checkpoint(
        points_path=args.points,
        predictions_path=args.predictions,
        output_path=args.output,
        mode=args.mode,
        wall_class_id=args.wall_class_id,
        threshold=args.threshold,
    )
    arr = np.load(out, allow_pickle=False)
    arr = np.asarray(arr)
    kept = int(arr.astype(bool).sum())
    print(
        f"Checkpoint written: {out} | total={arr.shape[0]} | wall_points={kept}",
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
