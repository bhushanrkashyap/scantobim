from __future__ import annotations

import numpy as np

from agent.tools.semantic_models import get_last_semantic_report, predict_wall_mask


def test_predict_wall_mask_geometry_passthrough():
    pts = np.zeros((10, 3), dtype=np.float64)
    result = predict_wall_mask(
        points_m=pts,
        model="geometry",
        wall_threshold=0.5,
        wall_class_id=1,
    )
    assert result.success is True
    assert result.wall_mask is not None
    assert int(result.wall_mask.sum()) == 10


def test_predict_wall_mask_from_npy_probabilities(tmp_path):
    pts = np.zeros((6, 3), dtype=np.float64)
    pred_path = tmp_path / "scores.npy"
    np.save(pred_path, np.array([0.1, 0.8, 0.9, 0.2, 0.51, 0.49], dtype=np.float64))

    result = predict_wall_mask(
        points_m=pts,
        model="ptv3",
        wall_threshold=0.5,
        wall_class_id=1,
        checkpoint_path=pred_path,
    )
    assert result.success is True
    assert result.wall_mask is not None
    assert result.wall_mask.tolist() == [False, True, True, False, True, False]


def test_predict_wall_mask_reports_failure_without_checkpoint():
    pts = np.zeros((5, 3), dtype=np.float64)
    result = predict_wall_mask(
        points_m=pts,
        model="swin3d",
        wall_threshold=0.5,
        wall_class_id=1,
        checkpoint_path=None,
    )
    assert result.success is False
    report = get_last_semantic_report()
    assert report["enabled"] is True
    assert report["success"] is False
