"""H1 path windows for fixed production entries."""

from __future__ import annotations

import json
from typing import Any

import numpy as np
import pandas as pd

from research.position_mgmt import MAX_BARS
from settings.paths import RAW


def load_h1(symbol: str = "XAUUSD") -> pd.DataFrame:
    path = RAW / symbol.upper() / "H1" / "data.parquet"
    h1 = pd.read_parquet(path)
    h1 = h1.sort_values("timestamp").reset_index(drop=True)
    h1["timestamp"] = pd.to_datetime(h1["timestamp"], utc=True)
    return h1


def load_production_trades(heat_log: pd.DataFrame) -> pd.DataFrame:
    """Exact same entries as production heat (accepted only)."""
    t = heat_log.loc[~heat_log["skipped"].astype(bool)].copy()
    t["timestamp"] = pd.to_datetime(t["timestamp"], utc=True)
    return t.sort_values("timestamp").reset_index(drop=True)


def wilder_atr(high: np.ndarray, low: np.ndarray, close: np.ndarray, period: int = 14) -> np.ndarray:
    n = len(close)
    tr = np.empty(n, dtype=float)
    tr[0] = high[0] - low[0]
    for i in range(1, n):
        tr[i] = max(high[i] - low[i], abs(high[i] - close[i - 1]), abs(low[i] - close[i - 1]))
    atr = np.full(n, np.nan, dtype=float)
    if n < period:
        return atr
    atr[period - 1] = float(np.mean(tr[:period]))
    alpha = 1.0 / period
    for i in range(period, n):
        atr[i] = atr[i - 1] * (1.0 - alpha) + tr[i] * alpha
    return atr


def adx_series(high: np.ndarray, low: np.ndarray, close: np.ndarray, period: int = 14) -> np.ndarray:
    """Classic Wilder ADX. ponytail: no DI smoothing edge-cases beyond period."""
    n = len(close)
    plus_dm = np.zeros(n)
    minus_dm = np.zeros(n)
    tr = np.zeros(n)
    for i in range(1, n):
        up = high[i] - high[i - 1]
        dn = low[i - 1] - low[i]
        plus_dm[i] = up if up > dn and up > 0 else 0.0
        minus_dm[i] = dn if dn > up and dn > 0 else 0.0
        tr[i] = max(high[i] - low[i], abs(high[i] - close[i - 1]), abs(low[i] - close[i - 1]))
    atr = wilder_atr(high, low, close, period)
    adx = np.full(n, np.nan)
    if n < 2 * period:
        return adx
    # smoothed DM
    sp = np.zeros(n)
    sm = np.zeros(n)
    sp[period] = plus_dm[1 : period + 1].sum()
    sm[period] = minus_dm[1 : period + 1].sum()
    satr = atr.copy()
    dx = np.full(n, np.nan)
    for i in range(period + 1, n):
        sp[i] = sp[i - 1] - sp[i - 1] / period + plus_dm[i]
        sm[i] = sm[i - 1] - sm[i - 1] / period + minus_dm[i]
        if satr[i] and satr[i] > 0:
            pdi = 100.0 * sp[i] / satr[i]
            mdi = 100.0 * sm[i] / satr[i]
            denom = pdi + mdi
            dx[i] = 0.0 if denom <= 0 else 100.0 * abs(pdi - mdi) / denom
    # ADX = Wilder smooth of DX
    start = 2 * period
    if start >= n or not np.isfinite(dx[period + 1 : start + 1]).all():
        # fallback: rolling mean of dx
        for i in range(period, n):
            w = dx[max(period, i - period + 1) : i + 1]
            w = w[np.isfinite(w)]
            adx[i] = float(np.mean(w)) if len(w) else np.nan
        return adx
    adx[start] = float(np.nanmean(dx[period + 1 : start + 1]))
    for i in range(start + 1, n):
        if np.isfinite(dx[i]) and np.isfinite(adx[i - 1]):
            adx[i] = (adx[i - 1] * (period - 1) + dx[i]) / period
    return adx


def prepare_market(h1: pd.DataFrame) -> dict[str, Any]:
    high = h1["high"].to_numpy(dtype=float)
    low = h1["low"].to_numpy(dtype=float)
    close = h1["close"].to_numpy(dtype=float)
    vol = h1["tick_volume"].to_numpy(dtype=float) if "tick_volume" in h1.columns else np.ones(len(h1))
    atr = wilder_atr(high, low, close, 14)
    adx = adx_series(high, low, close, 14)
    ts = h1["timestamp"].to_numpy()
    return {"high": high, "low": low, "close": close, "vol": vol, "atr": atr, "adx": adx, "ts": ts}


def entry_indices(trades: pd.DataFrame, h1_ts: np.ndarray) -> np.ndarray:
    """Map each trade timestamp to H1 row index (exact match preferred)."""
    h1_ts = pd.to_datetime(h1_ts, utc=True)
    idx_map = {t: i for i, t in enumerate(h1_ts)}
    out = np.full(len(trades), -1, dtype=int)
    # numeric searchsorted fallback
    h1_i8 = h1_ts.view("i8") if hasattr(h1_ts, "view") else np.asarray(h1_ts.asi8)
    for i, t in enumerate(pd.to_datetime(trades["timestamp"], utc=True)):
        if t in idx_map:
            out[i] = idx_map[t]
        else:
            pos = int(np.searchsorted(h1_i8, t.value, side="right") - 1)
            out[i] = pos if pos >= 0 else -1
    return out


def window_end(entry_i: int, n: int, max_bars: int = MAX_BARS) -> int:
    return min(entry_i + max_bars, n - 1)
