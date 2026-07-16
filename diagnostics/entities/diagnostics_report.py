"""Diagnostics report entity — structured findings + warnings."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any


@dataclass(frozen=True, slots=True)
class DiagnosticWarning:
    code: str
    message: str
    explanation: str
    severity: str = "warning"  # warning | critical

    def to_dict(self) -> dict[str, str]:
        return {
            "code": self.code,
            "message": self.message,
            "explanation": self.explanation,
            "severity": self.severity,
        }


@dataclass
class DiagnosticsReport:
    symbol: str
    timeframe: str
    side: str
    algorithm: str
    created_at: datetime
    dataset: dict[str, Any] = field(default_factory=dict)
    features: dict[str, Any] = field(default_factory=dict)
    labels: dict[str, Any] = field(default_factory=dict)
    model: dict[str, Any] = field(default_factory=dict)
    predictions: dict[str, Any] = field(default_factory=dict)
    probabilities: dict[str, Any] = field(default_factory=dict)
    thresholds: dict[str, Any] = field(default_factory=dict)
    drift: dict[str, Any] = field(default_factory=dict)
    warnings: list[DiagnosticWarning] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.symbol = self.symbol.strip().upper()
        self.timeframe = self.timeframe.strip().upper()
        self.side = self.side.strip().lower()
        if self.created_at.tzinfo is None:
            self.created_at = self.created_at.replace(tzinfo=timezone.utc)

    def add_warning(
        self,
        code: str,
        message: str,
        explanation: str,
        *,
        severity: str = "warning",
    ) -> None:
        self.warnings.append(
            DiagnosticWarning(
                code=code,
                message=message,
                explanation=explanation,
                severity=severity,
            )
        )

    def to_metadata(self) -> dict[str, Any]:
        return {
            "symbol": self.symbol,
            "timeframe": self.timeframe,
            "side": self.side,
            "algorithm": self.algorithm,
            "created_at": self.created_at.isoformat(),
            "warning_count": len(self.warnings),
            "warnings": [w.to_dict() for w in self.warnings],
            "dataset": self.dataset,
            "features_summary": {
                k: self.features[k]
                for k in (
                    "n_features",
                    "constant_features",
                    "near_constant_features",
                    "high_correlation_pairs",
                )
                if k in self.features
            },
            "labels_summary": {
                k: self.labels[k]
                for k in (
                    "tp_pct",
                    "sl_pct",
                    "timeout_pct",
                    "positive_rate",
                    "negative_rate",
                )
                if k in self.labels
            },
            "model_summary": {
                k: self.model[k]
                for k in (
                    "algorithm",
                    "best_iteration",
                    "tree_count",
                    "unused_features",
                    "model_size_bytes",
                )
                if k in self.model
            },
            "predictions_summary": {
                k: self.predictions[k]
                for k in (
                    "roc_auc",
                    "pr_auc",
                    "f1",
                    "precision",
                    "recall",
                    "accuracy",
                    "predicted_positives",
                    "predicted_negatives",
                )
                if k in self.predictions
            },
            "probabilities_summary": {
                k: self.probabilities[k]
                for k in (
                    "min",
                    "max",
                    "mean",
                    "median",
                    "std",
                    "flags",
                )
                if k in self.probabilities
            },
            "drift_summary": {
                k: self.drift[k]
                for k in ("max_psi", "max_ks", "target_drift", "probability_drift")
                if k in self.drift
            },
            "metadata": self.metadata,
        }
