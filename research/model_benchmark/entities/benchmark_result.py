"""Benchmark result entities."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass(frozen=True, slots=True)
class BenchmarkWindowResult:
    algorithm: str
    window: str
    valid_year: int
    status: str
    roc_auc: float
    pr_auc: float
    precision: float
    recall: float
    f1: float
    log_loss: float
    calibration_error: float
    best_threshold: float
    best_iteration: int
    train_seconds: float
    infer_seconds: float
    train_rows: int
    valid_rows: int
    proba_mean: float
    proba_std: float
    proba_min: float
    proba_max: float
    tn: int
    fp: int
    fn: int
    tp: int

    def to_dict(self) -> dict[str, Any]:
        return {
            "algorithm": self.algorithm,
            "window": self.window,
            "valid_year": self.valid_year,
            "status": self.status,
            "roc_auc": self.roc_auc,
            "pr_auc": self.pr_auc,
            "precision": self.precision,
            "recall": self.recall,
            "f1": self.f1,
            "log_loss": self.log_loss,
            "calibration_error": self.calibration_error,
            "best_threshold": self.best_threshold,
            "best_iteration": self.best_iteration,
            "train_seconds": self.train_seconds,
            "infer_seconds": self.infer_seconds,
            "train_rows": self.train_rows,
            "valid_rows": self.valid_rows,
            "proba_mean": self.proba_mean,
            "proba_std": self.proba_std,
            "proba_min": self.proba_min,
            "proba_max": self.proba_max,
            "tn": self.tn,
            "fp": self.fp,
            "fn": self.fn,
            "tp": self.tp,
        }


@dataclass
class BenchmarkModelResult:
    algorithm: str
    n_features: int
    windows: list[BenchmarkWindowResult] = field(default_factory=list)
    mean_roc_auc: float = float("nan")
    std_roc_auc: float = float("nan")
    mean_pr_auc: float = float("nan")
    mean_precision: float = float("nan")
    mean_recall: float = float("nan")
    mean_f1: float = float("nan")
    mean_log_loss: float = float("nan")
    mean_calibration_error: float = float("nan")
    mean_best_threshold: float = float("nan")
    total_train_seconds: float = 0.0
    total_infer_seconds: float = 0.0
    pooled_proba: list[float] = field(default_factory=list)
    pooled_y: list[int] = field(default_factory=list)

    def aggregate(self) -> None:
        import numpy as np

        ok = [w for w in self.windows if w.status == "ok"]
        if not ok:
            return
        self.mean_roc_auc = float(np.mean([w.roc_auc for w in ok]))
        self.std_roc_auc = float(np.std([w.roc_auc for w in ok], ddof=1)) if len(ok) > 1 else 0.0
        self.mean_pr_auc = float(np.mean([w.pr_auc for w in ok]))
        self.mean_precision = float(np.mean([w.precision for w in ok]))
        self.mean_recall = float(np.mean([w.recall for w in ok]))
        self.mean_f1 = float(np.mean([w.f1 for w in ok]))
        self.mean_log_loss = float(np.mean([w.log_loss for w in ok]))
        self.mean_calibration_error = float(np.mean([w.calibration_error for w in ok]))
        self.mean_best_threshold = float(np.mean([w.best_threshold for w in ok]))
        self.total_train_seconds = float(sum(w.train_seconds for w in ok))
        self.total_infer_seconds = float(sum(w.infer_seconds for w in ok))

    def summary_row(self) -> dict[str, Any]:
        import numpy as np

        proba = np.asarray(self.pooled_proba, dtype=float)
        return {
            "algorithm": self.algorithm,
            "n_features": self.n_features,
            "mean_roc_auc": self.mean_roc_auc,
            "std_roc_auc": self.std_roc_auc,
            "mean_pr_auc": self.mean_pr_auc,
            "mean_precision": self.mean_precision,
            "mean_recall": self.mean_recall,
            "mean_f1": self.mean_f1,
            "mean_log_loss": self.mean_log_loss,
            "mean_calibration_error": self.mean_calibration_error,
            "mean_best_threshold": self.mean_best_threshold,
            "total_train_seconds": self.total_train_seconds,
            "total_infer_seconds": self.total_infer_seconds,
            "n_windows_ok": sum(1 for w in self.windows if w.status == "ok"),
            "proba_mean": float(proba.mean()) if len(proba) else float("nan"),
            "proba_std": float(proba.std(ddof=1)) if len(proba) > 1 else 0.0,
        }
