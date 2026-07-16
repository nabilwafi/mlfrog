"""Execution / market-data ports — Decision never imports broker impl."""

from __future__ import annotations

from typing import Protocol

import pandas as pd


class MarketDataPort(Protocol):
    """Read-only rates. Implementations must not place orders."""

    def closed_bars(self, symbol: str, timeframe: str, lookback: int) -> pd.DataFrame: ...


class ExecutionPort(Protocol):
    """Future live broker. Paper stack must NOT implement order_send via this yet."""

    def open_long(self, *args, **kwargs): ...  # pragma: no cover

    def close(self, *args, **kwargs): ...  # pragma: no cover
