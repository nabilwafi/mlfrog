"""Feature diagnostics report entity."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any


@dataclass
class FeatureDiagnosticsReport:
    symbol: str
    timeframe: str
    side: str
    created_at: datetime
    keep: list[str] = field(default_factory=list)
    remove: list[str] = field(default_factory=list)
    correlated_groups: list[list[str]] = field(default_factory=list)
    drift_sensitive: list[str] = field(default_factory=list)
    stable: list[str] = field(default_factory=list)
    candidate_library_v3: list[str] = field(default_factory=list)
    category_power: dict[str, float] = field(default_factory=dict)
    summaries: dict[str, Any] = field(default_factory=dict)
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.symbol = self.symbol.strip().upper()
        self.timeframe = self.timeframe.strip().upper()
        self.side = self.side.strip().lower()
        if self.created_at.tzinfo is None:
            self.created_at = self.created_at.replace(tzinfo=timezone.utc)

    def to_dict(self) -> dict[str, Any]:
        return {
            "symbol": self.symbol,
            "timeframe": self.timeframe,
            "side": self.side,
            "created_at": self.created_at.isoformat(),
            "keep": self.keep,
            "remove": self.remove,
            "correlated_groups": self.correlated_groups,
            "drift_sensitive": self.drift_sensitive,
            "stable": self.stable,
            "candidate_library_v3": self.candidate_library_v3,
            "category_power": self.category_power,
            "summaries": self.summaries,
            "metadata": self.metadata,
        }
