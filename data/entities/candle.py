"""Immutable OHLCV candle."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime


@dataclass(frozen=True, slots=True)
class Candle:
    timestamp: datetime
    open: float
    high: float
    low: float
    close: float
    tick_volume: float
    spread: float
    real_volume: float

    def __post_init__(self) -> None:
        if self.timestamp.tzinfo is None:
            raise ValueError("Candle.timestamp must be timezone-aware")
        for name in ("open", "high", "low", "close"):
            value = getattr(self, name)
            if value < 0:
                raise ValueError(f"Candle.{name} cannot be negative (got {value})")
        if self.high < self.open:
            raise ValueError(f"High ({self.high}) must be >= Open ({self.open})")
        if self.high < self.close:
            raise ValueError(f"High ({self.high}) must be >= Close ({self.close})")
        if self.low > self.open:
            raise ValueError(f"Low ({self.low}) must be <= Open ({self.open})")
        if self.low > self.close:
            raise ValueError(f"Low ({self.low}) must be <= Close ({self.close})")
        if self.high < self.low:
            raise ValueError(f"High ({self.high}) must be >= Low ({self.low})")
        if self.tick_volume < 0:
            raise ValueError(f"tick_volume cannot be negative (got {self.tick_volume})")
        if self.real_volume < 0:
            raise ValueError(f"real_volume cannot be negative (got {self.real_volume})")
        if self.spread < 0:
            raise ValueError(f"spread cannot be negative (got {self.spread})")
