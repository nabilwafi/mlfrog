"""Market data provider ABC."""

from __future__ import annotations

from abc import ABC, abstractmethod
from datetime import datetime

from data.entities.market_data import MarketData


class BaseMarketProvider(ABC):
    """Fetches market data and always returns domain MarketData (never raw frames)."""

    @abstractmethod
    def fetch(
        self,
        symbol: str,
        timeframe: str,
        start: datetime,
        end: datetime,
        *,
        timezone: str = "UTC",
    ) -> MarketData:
        """Fetch candles in [start, end]."""

    @abstractmethod
    def health_check(self) -> bool:
        """Return True when the provider is usable."""

    @abstractmethod
    def close(self) -> None:
        """Release resources."""

    def __enter__(self) -> BaseMarketProvider:
        return self

    def __exit__(self, *_: object) -> None:
        self.close()
