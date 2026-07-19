"""Prediction batch from a trained model."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Mapping


@dataclass(frozen=True, slots=True)
class Prediction:
    symbol: str
    timeframe: str
    side: str
    algorithm: str
    timestamps: tuple[datetime, ...]
    probabilities: tuple[float, ...]
    labels_pred: tuple[int, ...]
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        n = len(self.timestamps)
        if len(self.probabilities) != n or len(self.labels_pred) != n:
            raise ValueError("Prediction arrays length mismatch")
        object.__setattr__(self, "metadata", dict(self.metadata))
