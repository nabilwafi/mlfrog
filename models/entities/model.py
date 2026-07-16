"""Trained model artifact metadata + handle."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Mapping


@dataclass
class Model:
    symbol: str
    timeframe: str
    side: str
    algorithm: str
    feature_version: str
    label_version: str
    dataset_version: str
    trained_at: datetime
    hyperparameters: dict[str, Any]
    feature_names: tuple[str, ...]
    train_rows: int
    validation_rows: int
    metrics: dict[str, Any] = field(default_factory=dict)
    feature_importance: dict[str, float] = field(default_factory=dict)
    metadata: dict[str, Any] = field(default_factory=dict)
    estimator: Any = field(default=None, repr=False)

    def __post_init__(self) -> None:
        self.symbol = self.symbol.strip().upper()
        self.timeframe = self.timeframe.strip().upper()
        self.side = self.side.strip().lower()
        self.algorithm = self.algorithm.strip().lower()
        if self.side not in {"long", "short"}:
            raise ValueError(f"invalid side: {self.side!r}")
        if self.estimator is None:
            raise ValueError("Model.estimator is required")
        if not self.feature_names:
            raise ValueError("feature_names must be non-empty")

    def to_metadata(self) -> dict[str, Any]:
        return {
            "symbol": self.symbol,
            "timeframe": self.timeframe,
            "side": self.side,
            "algorithm": self.algorithm,
            "dataset_version": self.dataset_version,
            "feature_version": self.feature_version,
            "label_version": self.label_version,
            "training_timestamp": self.trained_at.isoformat(),
            "hyperparameters": self.hyperparameters,
            "training_rows": self.train_rows,
            "validation_rows": self.validation_rows,
            "feature_names": list(self.feature_names),
            "metrics": self.metrics,
            "metadata": self.metadata,
        }
