"""Single barrier / event label."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Literal, Mapping

ExitReason = Literal["TP", "SL", "TIMEOUT"]
Side = Literal["long", "short"]


@dataclass(frozen=True, slots=True)
class Label:
    timestamp: datetime
    entry_price: float
    tp_price: float
    sl_price: float
    expire_timestamp: datetime
    holding_bars: int
    realized_return: float
    exit_reason: ExitReason
    side: Side
    label: int
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.timestamp.tzinfo is None:
            raise ValueError("Label.timestamp must be timezone-aware")
        if self.expire_timestamp.tzinfo is None:
            raise ValueError("Label.expire_timestamp must be timezone-aware")
        if self.expire_timestamp < self.timestamp:
            raise ValueError("expire_timestamp must be >= timestamp")
        if self.holding_bars < 0:
            raise ValueError("holding_bars must be >= 0")
        if self.entry_price <= 0:
            raise ValueError("entry_price must be > 0")
        if self.exit_reason not in {"TP", "SL", "TIMEOUT"}:
            raise ValueError(f"invalid exit_reason: {self.exit_reason!r}")
        if self.side not in {"long", "short"}:
            raise ValueError(f"invalid side: {self.side!r}")
        object.__setattr__(self, "metadata", dict(self.metadata))
