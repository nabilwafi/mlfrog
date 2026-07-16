"""Classification evaluation metrics (no trading metrics)."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    brier_score_loss,
    confusion_matrix,
    f1_score,
    log_loss,
    precision_score,
    recall_score,
    roc_auc_score,
)


@dataclass(frozen=True, slots=True)
class EvaluationResult:
    metrics: dict[str, Any]
    feature_importance: dict[str, float]

    def to_dict(self) -> dict[str, Any]:
        return {
            "metrics": self.metrics,
            "feature_importance": self.feature_importance,
        }


class Evaluator:
    """Compute binary classification metrics from y_true and P(y=1)."""

    def evaluate(
        self,
        y_true: np.ndarray,
        y_proba: np.ndarray,
        *,
        feature_importance: dict[str, float] | None = None,
        threshold: float = 0.5,
    ) -> EvaluationResult:
        y_true = np.asarray(y_true, dtype=int).ravel()
        y_proba = np.asarray(y_proba, dtype=float).ravel()
        if len(y_true) == 0:
            raise ValueError("empty evaluation arrays")
        if len(y_true) != len(y_proba):
            raise ValueError(
                f"length mismatch y_true={len(y_true)} y_proba={len(y_proba)}"
            )

        y_pred = (y_proba >= threshold).astype(int)
        metrics: dict[str, Any] = {
            "n_samples": int(len(y_true)),
            "positive_rate": float(y_true.mean()),
            "threshold": float(threshold),
            "accuracy": float(accuracy_score(y_true, y_pred)),
            "precision": float(precision_score(y_true, y_pred, zero_division=0)),
            "recall": float(recall_score(y_true, y_pred, zero_division=0)),
            "f1": float(f1_score(y_true, y_pred, zero_division=0)),
            "log_loss": float(log_loss(y_true, y_proba, labels=[0, 1])),
            "brier_score": float(brier_score_loss(y_true, y_proba)),
        }

        # ROC/PR need both classes present
        if len(np.unique(y_true)) >= 2:
            metrics["roc_auc"] = float(roc_auc_score(y_true, y_proba))
            metrics["pr_auc"] = float(average_precision_score(y_true, y_proba))
        else:
            metrics["roc_auc"] = float("nan")
            metrics["pr_auc"] = float("nan")

        cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
        metrics["confusion_matrix"] = {
            "tn": int(cm[0, 0]),
            "fp": int(cm[0, 1]),
            "fn": int(cm[1, 0]),
            "tp": int(cm[1, 1]),
        }

        return EvaluationResult(
            metrics=metrics,
            feature_importance=dict(feature_importance or {}),
        )

    def importance_frame(self, importance: dict[str, float]) -> pd.DataFrame:
        rows = [{"feature": k, "importance": float(v)} for k, v in importance.items()]
        frame = pd.DataFrame(rows)
        if frame.empty:
            return pd.DataFrame(columns=["feature", "importance"])
        return frame.sort_values("importance", ascending=False).reset_index(drop=True)
