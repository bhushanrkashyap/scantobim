"""Phase 3C Multi-Class Neural Training and Validation Package."""

from agent.phase3.training.augment import PointCloudAugmentor
from agent.phase3.training.checkpoints import (
    compute_file_sha256,
    save_multiclass_checkpoint,
)
from agent.phase3.training.dataset import Phase3CDataset
from agent.phase3.training.metrics import (
    ClassMetric,
    SemanticEvaluationMetrics,
    compute_semantic_metrics,
)
from agent.phase3.training.train import train_multiclass_randlanet
from agent.phase3.training.validate import evaluate_model_on_point_cloud

__all__ = [
    "ClassMetric",
    "Phase3CDataset",
    "PointCloudAugmentor",
    "SemanticEvaluationMetrics",
    "compute_file_sha256",
    "compute_semantic_metrics",
    "evaluate_model_on_point_cloud",
    "save_multiclass_checkpoint",
    "train_multiclass_randlanet",
]
