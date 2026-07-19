"""Shared transforms for stationary / normalized features."""

from __future__ import annotations

import numpy as np
import pandas as pd


def ema(series: pd.Series, period: int) -> pd.Series:
    return series.astype(float).ewm(span=period, adjust=False).mean()


def true_range(high: pd.Series, low: pd.Series, close: pd.Series) -> pd.Series:
    prev = close.shift(1)
    ranges = pd.concat(
        [
            (high - low).abs(),
            (high - prev).abs(),
            (low - prev).abs(),
        ],
        axis=1,
    )
    return ranges.max(axis=1)


def atr(high: pd.Series, low: pd.Series, close: pd.Series, period: int = 14) -> pd.Series:
    return true_range(high, low, close).ewm(alpha=1 / period, adjust=False).mean()


def rsi(close: pd.Series, period: int = 14) -> pd.Series:
    delta = close.astype(float).diff()
    gain = delta.clip(lower=0.0)
    loss = (-delta).clip(lower=0.0)
    avg_gain = gain.ewm(alpha=1 / period, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1 / period, adjust=False).mean()
    rs = avg_gain / avg_loss.replace(0.0, np.nan)
    return 100.0 - (100.0 / (1.0 + rs))


def macd(
    close: pd.Series, *, fast: int = 12, slow: int = 26, signal: int = 9
) -> tuple[pd.Series, pd.Series, pd.Series]:
    line = ema(close, fast) - ema(close, slow)
    sig = ema(line, signal)
    hist = line - sig
    return line, sig, hist


def rolling_zscore(series: pd.Series, window: int) -> pd.Series:
    mean = series.rolling(window, min_periods=max(5, window // 5)).mean()
    std = series.rolling(window, min_periods=max(5, window // 5)).std(ddof=0)
    return (series - mean) / std.replace(0.0, np.nan)


def rolling_percentile(series: pd.Series, window: int) -> pd.Series:
    def _pct(arr: np.ndarray) -> float:
        x = arr[-1]
        if not np.isfinite(x):
            return np.nan
        finite = arr[np.isfinite(arr)]
        if len(finite) == 0:
            return np.nan
        return float((finite <= x).mean())

    return series.rolling(window, min_periods=max(5, window // 5)).apply(_pct, raw=True)


def rolling_rank(series: pd.Series, window: int) -> pd.Series:
    def _rank(arr: np.ndarray) -> float:
        x = arr[-1]
        if not np.isfinite(x):
            return np.nan
        finite = arr[np.isfinite(arr)]
        if len(finite) == 0:
            return np.nan
        # average rank of last value among window, scaled to [0, 1]
        order = finite.argsort().argsort()
        # find last finite index corresponding to x — use count <= x
        return float((finite < x).sum() + 0.5 * (finite == x).sum()) / len(finite)

    return series.rolling(window, min_periods=max(5, window // 5)).apply(_rank, raw=True)


def safe_div(num: pd.Series, den: pd.Series) -> pd.Series:
    return num.astype(float) / den.astype(float).replace(0.0, np.nan)


def slope(series: pd.Series, lookback: int = 3) -> pd.Series:
    """Normalized slope: (x_t - x_{t-n}) / n / |x|."""
    lag = series.shift(lookback)
    raw = (series - lag) / float(lookback)
    return safe_div(raw, series.abs())


def streak_duration(condition: pd.Series) -> pd.Series:
    """Bars since condition flipped (inclusive length of current True/False run)."""
    flags = condition.astype(bool).to_numpy()
    out = np.zeros(len(flags), dtype=float)
    run = 0
    prev = None
    for i, flag in enumerate(flags):
        if prev is None or flag != prev:
            run = 1
        else:
            run += 1
        out[i] = float(run)
        prev = flag
    return pd.Series(out, index=condition.index)
