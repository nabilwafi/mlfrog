"""Per-window and aggregated ablation metrics."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass(frozen=True, slots=True)
class WindowMetrics:
    experiment_id: str
    window: str
    valid_year: int
    status: str
    n_features: int
    roc_auc: float
    pr_auc: float
    log_loss: float
    precision: float
    recall: float
    f1: float
    calibration_error: float
    best_iteration: int
    train_seconds: float
    infer_seconds: float
    train_rows: int
    valid_rows: int

    def to_dict(self) -> dict[str, Any]:
        return {
            "experiment_id": self.experiment_id,
            "window": self.window,
            "valid_year": self.valid_year,
            "status": self.status,
            "n_features": self.n_features,
            "roc_auc": self.roc_auc,
            "pr_auc": self.pr_auc,
            "log_loss": self.log_loss,
            "precision": self.precision,
            "recall": self.recall,
            "f1": self.f1,
            "calibration_error": self.calibration_error,
            "best_iteration": self.best_iteration,
            "train_seconds": self.train_seconds,
            "infer_seconds": self.infer_seconds,
            "train_rows": self.train_rows,
            "valid_rows": self.valid_rows,
        }


@dataclass
class ExperimentResult:
    experiment_id: str
    name: str
    description: str
    n_features: int
    feature_names: tuple[str, ...]
    feature_categories: dict[str, int]
    windows: list[WindowMetrics] = field(default_factory=list)
    mean_roc_auc: float = float("nan")
    std_roc_auc: float = float("nan")
    mean_pr_auc: float = float("nan")
    mean_log_loss: float = float("nan")
    mean_precision: float = float("nan")
    mean_recall: float = float("nan")
    mean_f1: float = float("nan")
    mean_calibration_error: float = float("nan")
    mean_best_iteration: float = float("nan")
    total_train_seconds: float = 0.0
    total_infer_seconds: float = 0.0

    def aggregate(self) -> None:
        ok = [w for w in self.windows if w.status == "ok"]
        if not ok:
            return
        import numpy as np

        self.mean_roc_auc = float(np.mean([w.roc_auc for w in ok]))
        self.std_roc_auc = float(np.std([w.roc_auc for w in ok], ddof=1)) if len(ok) > 1 else 0.0
        self.mean_pr_auc = float(np.mean([w.pr_auc for w in ok]))
        self.mean_log_loss = float(np.mean([w.log_loss for w in ok]))
        self.mean_precision = float(np.mean([w.precision for w in ok]))
        self.mean_recall = float(np.mean([w.recall for w in ok]))
        self.mean_f1 = float(np.mean([w.f1 for w in ok]))
        self.mean_calibration_error = float(np.mean([w.calibration_error for w in ok]))
        self.mean_best_iteration = float(np.mean([w.best_iteration for w in ok]))
        self.total_train_seconds = float(sum(w.train_seconds for w in ok))
        self.total_infer_seconds = float(sum(w.infer_seconds for w in ok))

    def summary_row(self) -> dict[str, Any]:
        return {
            "experiment_id": self.experiment_id,
            "name": self.name,
            "description": self.description,
            "n_features": self.n_features,
            "categories": ",".join(f"{k}:{v}" for k, v in sorted(self.feature_categories.items())),
            "mean_roc_auc": self.mean_roc_auc,
            "std_roc_auc": self.std_roc_auc,
            "mean_pr_auc": self.mean_pr_auc,
            "mean_log_loss": self.mean_log_loss,
            "mean_precision": self.mean_precision,
            "mean_recall": self.mean_recall,
            "mean_f1": self.mean_f1,
            "mean_calibration_error": self.mean_calibration_error,
            "mean_best_iteration": self.mean_best_iteration,
            "total_train_seconds": self.total_train_seconds,
            "total_infer_seconds": self.total_infer_seconds,
            "n_windows_ok": sum(1 for w in self.windows if w.status == "ok"),
            "features": "|".join(self.feature_names),
        }
