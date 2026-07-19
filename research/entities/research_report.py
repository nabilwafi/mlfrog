"""Research report entity — hypothesis evaluation outputs."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any


@dataclass
class ResearchReport:
    symbol: str
    timeframe: str
    side: str
    algorithm: str
    created_at: datetime
    walk_forward: dict[str, Any] = field(default_factory=dict)
    feature_stability: dict[str, Any] = field(default_factory=dict)
    feature_drift: dict[str, Any] = field(default_factory=dict)
    regimes: dict[str, Any] = field(default_factory=dict)
    label_stability: dict[str, Any] = field(default_factory=dict)
    conclusions: dict[str, Any] = field(default_factory=dict)
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.symbol = self.symbol.strip().upper()
        self.timeframe = self.timeframe.strip().upper()
        self.side = self.side.strip().lower()
        if self.created_at.tzinfo is None:
            self.created_at = self.created_at.replace(tzinfo=timezone.utc)

    def to_metadata(self) -> dict[str, Any]:
        return {
            "symbol": self.symbol,
            "timeframe": self.timeframe,
            "side": self.side,
            "algorithm": self.algorithm,
            "created_at": self.created_at.isoformat(),
            "walk_forward": self.walk_forward,
            "feature_stability": self.feature_stability,
            "feature_drift": self.feature_drift,
            "regimes": self.regimes,
            "label_stability": self.label_stability,
            "conclusions": self.conclusions,
            "metadata": self.metadata,
        }
