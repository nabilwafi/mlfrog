"""Shared helpers to build MarketData from tabular rows (provider-internal only)."""

from __future__ import annotations

from datetime import datetime
from zoneinfo import ZoneInfo

import pandas as pd

from data.entities.candle import Candle
from data.entities.market_data import MarketData

_COLUMN_ALIASES: dict[str, tuple[str, ...]] = {
    "timestamp": ("timestamp", "Date", "date", "time", "datetime"),
    "open": ("open", "Open"),
    "high": ("high", "High"),
    "low": ("low", "Low"),
    "close": ("close", "Close"),
    "tick_volume": ("tick_volume", "Tick Volume", "tick volume", "TickVolume"),
    "spread": ("spread", "Spread"),
    "real_volume": ("real_volume", "Volume", "volume", "Real Volume"),
}


def _resolve(df: pd.DataFrame, logical: str, *, required: bool = True) -> str | None:
    for candidate in _COLUMN_ALIASES[logical]:
        if candidate in df.columns:
            return candidate
    if required:
        raise KeyError(f"missing column for {logical!r}; have {list(df.columns)}")
    return None


def dataframe_to_market_data(
    df: pd.DataFrame,
    *,
    symbol: str,
    timeframe: str,
    timezone: str,
    start: datetime | None = None,
    end: datetime | None = None,
) -> MarketData:
    """Convert a provider-local DataFrame into MarketData. Not part of public API."""
    if df is None or df.empty:
        return MarketData(symbol=symbol, timeframe=timeframe, timezone=timezone, candles=())

    work = df.copy()
    ts_col = _resolve(work, "timestamp")
    open_col = _resolve(work, "open")
    high_col = _resolve(work, "high")
    low_col = _resolve(work, "low")
    close_col = _resolve(work, "close")
    tick_col = _resolve(work, "tick_volume")
    spread_col = _resolve(work, "spread", required=False)
    real_col = _resolve(work, "real_volume", required=False)

    assert ts_col and open_col and high_col and low_col and close_col and tick_col

    tz = ZoneInfo(timezone)
    ts = pd.to_datetime(work[ts_col], utc=True)
    work = work.assign(_ts=ts)
    if start is not None:
        start_utc = pd.Timestamp(start if start.tzinfo else start.replace(tzinfo=tz)).tz_convert(
            "UTC"
        )
        work = work[work["_ts"] >= start_utc]
    if end is not None:
        end_utc = pd.Timestamp(end if end.tzinfo else end.replace(tzinfo=tz)).tz_convert("UTC")
        work = work[work["_ts"] <= end_utc]
    work = work.sort_values("_ts").reset_index(drop=True)

    candles: list[Candle] = []
    for i in range(len(work)):
        ts_dt = work.at[i, "_ts"].to_pydatetime().astimezone(tz)
        spread = float(work.at[i, spread_col]) if spread_col else 0.0
        real_volume = float(work.at[i, real_col]) if real_col else 0.0
        candles.append(
            Candle(
                timestamp=ts_dt,
                open=float(work.at[i, open_col]),
                high=float(work.at[i, high_col]),
                low=float(work.at[i, low_col]),
                close=float(work.at[i, close_col]),
                tick_volume=float(work.at[i, tick_col]),
                spread=spread,
                real_volume=real_volume,
            )
        )
    return MarketData(
        symbol=symbol,
        timeframe=timeframe,
        timezone=timezone,
        candles=tuple(candles),
    )
