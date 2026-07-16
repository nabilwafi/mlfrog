"""Immutable market-data snapshot."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime

from data.entities.candle import Candle


@dataclass(frozen=True, slots=True)
class MarketData:
    symbol: str
    timeframe: str
    timezone: str
    candles: tuple[Candle, ...]

    def __post_init__(self) -> None:
        if not self.symbol.strip():
            raise ValueError("symbol must be non-empty")
        if not self.timeframe.strip():
            raise ValueError("timeframe must be non-empty")
        if not self.timezone.strip():
            raise ValueError("timezone must be non-empty")
        object.__setattr__(self, "symbol", self.symbol.strip().upper())
        object.__setattr__(self, "timeframe", self.timeframe.strip().upper())
        if not isinstance(self.candles, tuple):
            object.__setattr__(self, "candles", tuple(self.candles))

    @property
    def start_time(self) -> datetime | None:
        if not self.candles:
            return None
        return self.candles[0].timestamp

    @property
    def end_time(self) -> datetime | None:
        if not self.candles:
            return None
        return self.candles[-1].timestamp

    @property
    def total_candles(self) -> int:
        return len(self.candles)
