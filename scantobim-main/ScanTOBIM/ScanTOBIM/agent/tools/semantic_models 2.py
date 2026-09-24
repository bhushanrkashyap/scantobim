from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

_SUPPORTED_MODELS = {
    "pointnext",
    "pointmetabase",
    "ptv1",
    "ptv3",
    "swin3d",
    "randlanet",
    "noop",
    "geometry",
}
_LAST_REPORT: dict[str, Any] = {
    "enabled": False,
    "model": None,
    "success": False,
    "provider": "none",
    "message": "semantic stage not run",
    "wall_points": 0,
    "total_points": 0,
}


@dataclass
class SemanticMaskResult:
    success: bool
    model: str
    provider: str
    message: str
    wall_mask: np.ndarray | None


def get_last_semantic_report() -> dict[str, Any]:
    return dict(_LAST_REPORT)


def set_last_semantic_report(report: dict[str, Any]) -> None:
    _LAST_REPORT.clear()
    _LAST_REPORT.update(report)


def predict_wall_mask(
    points_m: np.ndarray,
    model: str,
    wall_threshold: float,
    wall_class_id: int,
    checkpoint_path: Path | None = None,
) -> SemanticMaskResult:
    model_key = (model or "").strip().lower()
    total_points = int(points_m.shape[0])

    if model_key not in _SUPPORTED_MODELS:
        msg = f"Unsupported semantic model '{model}'. Supported: {sorted(_SUPPORTED_MODELS)}"
        set_last_semantic_report(
            {
                "enabled": True,
                "model": model,
                "success": False,
                "provider": "invalid",
                "message": msg,
                "wall_points": 0,
                "total_points": total_points,
            }
        )
        return SemanticMaskResult(False, model_key, "invalid", msg, None)

    if model_key in {"noop", "geometry"}:
        mask = np.ones(total_points, dtype=bool)
        msg = "geometry passthrough selected (no semantic filtering)"
        set_last_semantic_report(
            {
                "enabled": True,
                "model": model_key,
                "success": True,
                "provider": "passthrough",
                "message": msg,
                "wall_points": int(mask.sum()),
                "total_points": total_points,
            }
        )
        return SemanticMaskResult(True, model_key, "passthrough", msg, mask)

    if model_key == "randlanet" and (checkpoint_path is None or not checkpoint_path.exists()):
        try:
            from agent.tools.randla_net import predict_randla

            mask, probs = predict_randla(
                points_m,
                wall_threshold=wall_threshold,
                wall_class_id=wall_class_id,
            )
            msg = f"Generated wall mask via RandLA algorithm (points={total_points}, walls={int(mask.sum())})"
            set_last_semantic_report(
                {
                    "enabled": True,
                    "model": model_key,
                    "success": True,
                    "provider": "randla_net",
                    "message": msg,
                    "wall_points": int(mask.sum()),
                    "total_points": total_points,
                }
            )
            return SemanticMaskResult(True, model_key, "randla_net", msg, mask)
        except Exception as exc:  # noqa: BLE001
            msg = f"RandLA prediction error: {exc}"
            set_last_semantic_report(
                {
                    "enabled": True,
                    "model": model_key,
                    "success": False,
                    "provider": "randla_net_error",
                    "message": msg,
                    "wall_points": 0,
                    "total_points": total_points,
                }
            )
            return SemanticMaskResult(False, model_key, "randla_net_error", msg, None)

    if checkpoint_path is None:
        msg = (
            f"Model '{model_key}' selected but no checkpoint/prediction file provided. "
            "Provide --semantic-checkpoint with a .npy/.npz wall prediction array."
        )
        set_last_semantic_report(
            {
                "enabled": True,
                "model": model_key,
                "success": False,
                "provider": "stub",
                "message": msg,
                "wall_points": 0,
                "total_points": total_points,
            }
        )
        return SemanticMaskResult(False, model_key, "stub", msg, None)

    if not checkpoint_path.exists():
        msg = f"Semantic checkpoint not found: {checkpoint_path}"
        set_last_semantic_report(
            {
                "enabled": True,
                "model": model_key,
                "success": False,
                "provider": "stub",
                "message": msg,
                "wall_points": 0,
                "total_points": total_points,
            }
        )
        return SemanticMaskResult(False, model_key, "stub", msg, None)

    try:
        preds = _load_predictions(checkpoint_path)
    except Exception as exc:  # noqa: BLE001
        msg = f"Failed to load semantic predictions: {exc}"
        set_last_semantic_report(
            {
                "enabled": True,
                "model": model_key,
                "success": False,
                "provider": "stub",
                "message": msg,
                "wall_points": 0,
                "total_points": total_points,
            }
        )
        return SemanticMaskResult(False, model_key, "stub", msg, None)

    if preds.shape[0] != total_points:
        msg = (
            f"Prediction length mismatch for model '{model_key}': "
            f"preds={preds.shape[0]} points={total_points}"
        )
        set_last_semantic_report(
            {
                "enabled": True,
                "model": model_key,
                "success": False,
                "provider": "stub",
                "message": msg,
                "wall_points": 0,
                "total_points": total_points,
            }
        )
        return SemanticMaskResult(False, model_key, "stub", msg, None)

    mask = _to_wall_mask(preds, wall_threshold=wall_threshold, wall_class_id=wall_class_id)
    msg = f"Loaded wall mask from {checkpoint_path.name}"
    set_last_semantic_report(
        {
            "enabled": True,
            "model": model_key,
            "success": True,
            "provider": "file",
            "message": msg,
            "wall_points": int(mask.sum()),
            "total_points": total_points,
            "checkpoint": str(checkpoint_path),
        }
    )
    return SemanticMaskResult(True, model_key, "file", msg, mask)


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
            # fallback to first array in file
            keys = list(z.keys())
            if not keys:
                raise ValueError("Empty npz file")
            arr = z[keys[0]]
    else:
        raise ValueError("Only .npy/.npz prediction files are currently supported")

    arr = np.asarray(arr)
    if arr.ndim != 1:
        arr = arr.reshape(-1)
    return arr


def _to_wall_mask(preds: np.ndarray, wall_threshold: float, wall_class_id: int) -> np.ndarray:
    if preds.dtype == bool:
        return preds.astype(bool)

    if np.issubdtype(preds.dtype, np.integer):
        return preds.astype(np.int64) == int(wall_class_id)

    if np.issubdtype(preds.dtype, np.floating):
        return preds.astype(np.float64) >= float(wall_threshold)

    # Last-resort conversion for unsupported dtypes
    try:
        as_float = preds.astype(np.float64)
        return as_float >= float(wall_threshold)
    except Exception as exc:  # noqa: BLE001
        raise ValueError(f"Unsupported prediction dtype: {preds.dtype}") from exc
