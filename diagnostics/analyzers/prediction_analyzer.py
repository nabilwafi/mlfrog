"""Validation prediction quality diagnostics."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    confusion_matrix,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)


class PredictionAnalyzer:
    def analyze(
        self,
        y_true: np.ndarray,
        y_proba: np.ndarray,
        *,
        timestamps: pd.Series | None = None,
        threshold: float = 0.5,
    ) -> tuple[dict[str, Any], pd.DataFrame]:
        y_true = np.asarray(y_true, dtype=int).ravel()
        y_proba = np.asarray(y_proba, dtype=float).ravel()
        y_pred = (y_proba >= threshold).astype(int)

        metrics: dict[str, Any] = {
            "n_samples": int(len(y_true)),
            "threshold": float(threshold),
            "predicted_positives": int((y_pred == 1).sum()),
            "predicted_negatives": int((y_pred == 0).sum()),
            "accuracy": float(accuracy_score(y_true, y_pred)),
            "precision": float(precision_score(y_true, y_pred, zero_division=0)),
            "recall": float(recall_score(y_true, y_pred, zero_division=0)),
            "f1": float(f1_score(y_true, y_pred, zero_division=0)),
        }
        if len(np.unique(y_true)) >= 2:
            metrics["roc_auc"] = float(roc_auc_score(y_true, y_proba))
            metrics["pr_auc"] = float(average_precision_score(y_true, y_proba))
            fpr, tpr, _ = roc_curve(y_true, y_proba)
            prec, rec, _ = precision_recall_curve(y_true, y_proba)
            metrics["roc_curve"] = {
                "fpr": [float(x) for x in fpr.tolist()],
                "tpr": [float(x) for x in tpr.tolist()],
            }
            metrics["pr_curve"] = {
                "precision": [float(x) for x in prec.tolist()],
                "recall": [float(x) for x in rec.tolist()],
            }
        else:
            metrics["roc_auc"] = float("nan")
            metrics["pr_auc"] = float("nan")
            metrics["roc_curve"] = {"fpr": [], "tpr": []}
            metrics["pr_curve"] = {"precision": [], "recall": []}

        cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
        metrics["confusion_matrix"] = {
            "tn": int(cm[0, 0]),
            "fp": int(cm[0, 1]),
            "fn": int(cm[1, 0]),
            "tp": int(cm[1, 1]),
        }

        dist = pd.DataFrame(
            {
                "timestamp": timestamps.values if timestamps is not None else np.arange(len(y_true)),
                "y_true": y_true,
                "y_proba": y_proba,
                "y_pred": y_pred,
            }
        )
        return metrics, dist
