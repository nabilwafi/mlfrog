"""Bar duration helpers for causal context availability."""

from __future__ import annotations

import pandas as pd

# Open-time bars: feature from bar at T is known only after the bar closes.
TIMEFRAME_DURATION: dict[str, pd.Timedelta] = {
    "M1": pd.Timedelta(minutes=1),
    "M5": pd.Timedelta(minutes=5),
    "M15": pd.Timedelta(minutes=15),
    "M30": pd.Timedelta(minutes=30),
    "H1": pd.Timedelta(hours=1),
    "H4": pd.Timedelta(hours=4),
    "D1": pd.Timedelta(days=1),
    "W1": pd.Timedelta(weeks=1),
}


def bar_duration(timeframe: str) -> pd.Timedelta:
    key = timeframe.upper()
    if key not in TIMEFRAME_DURATION:
        raise ValueError(f"unsupported timeframe for context: {timeframe!r}")
    return TIMEFRAME_DURATION[key]


def available_at(timestamps: pd.Series, timeframe: str) -> pd.Series:
    """First instant when an open-time bar's OHLC (and derived features) are known."""
    ts = pd.to_datetime(timestamps, utc=True)
    return ts + bar_duration(timeframe)
