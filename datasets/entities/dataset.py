"""ML-ready dataset (one split, one side)."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from typing import Any

import pandas as pd


@dataclass
class Dataset:
    symbol: str
    timeframe: str
    side: str
    strategy: str
    feature_version: str
    label_version: str
    split: str
    created_at: datetime
    frame: pd.DataFrame = field(repr=False)
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.symbol = self.symbol.strip().upper()
        self.timeframe = self.timeframe.strip().upper()
        self.side = self.side.strip().lower()
        self.split = self.split.strip().lower()
        if self.side not in {"long", "short"}:
            raise ValueError(f"invalid side: {self.side!r}")
        if self.frame is None or self.frame.empty:
            raise ValueError("Dataset.frame must be non-empty")
        required = {
            "timestamp",
            "symbol",
            "timeframe",
            "feature_version",
            "label_version",
            "split",
            "label",
        }
        missing = required - set(self.frame.columns)
        if missing:
            raise ValueError(f"Dataset.frame missing columns: {sorted(missing)}")

    @property
    def size(self) -> int:
        return len(self.frame)

    @property
    def feature_names(self) -> list[str]:
        skip = {
            "timestamp",
            "symbol",
            "timeframe",
            "feature_version",
            "label_version",
            "split",
            "side",
            "label",
            "strategy",
        }
        return [c for c in self.frame.columns if c not in skip]
