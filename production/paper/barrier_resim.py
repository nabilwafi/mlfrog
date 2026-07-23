"""Resimulate triple-barrier exits on H1 OHLC (SL-first same bar)."""

from __future__ import annotations

import numpy as np
import pandas as pd


def load_h1(path: str | None = None) -> pd.DataFrame:
    p = path or "artifacts/raw/XAUUSD/H1/data.parquet"
    h = pd.read_parquet(p)
    h = h.sort_values("timestamp").reset_index(drop=True)
    h["timestamp"] = pd.to_datetime(h["timestamp"], utc=True)
    return h


def resim_barrier_row(
    *,
    side: str,
    entry_ts: pd.Timestamp,
    entry_price: float,
    atr: float,
    high: np.ndarray,
    low: np.ndarray,
    close: np.ndarray,
    ts_index: pd.DatetimeIndex,
    tp_mult: float,
    sl_mult: float,
    horizon: int,
) -> tuple[float, int, str]:
    """
    Returns (net_return_approx as realized_return, holding_bars, exit_reason).
    Costs not re-applied (same convention as label realized_return).
    """
    side_l = side.lower()
    # entry bar = timestamp match; path starts next bar
    loc = ts_index.searchsorted(entry_ts, side="left")
    if loc >= len(ts_index) or ts_index[loc] != entry_ts:
        # nearest <= entry
        loc = int(ts_index.searchsorted(entry_ts, side="right") - 1)
    if loc < 0 or loc >= len(ts_index) - 1:
        return 0.0, 0, "NO_PATH"

    entry = float(entry_price)
    a = float(atr)
    if side_l == "long":
        tp = entry + tp_mult * a
        sl = entry - sl_mult * a
    else:
        tp = entry - tp_mult * a
        sl = entry + sl_mult * a

    end_j = min(loc + horizon, len(ts_index) - 1)
    for j in range(loc + 1, end_j + 1):
        if side_l == "long":
            hit_sl = low[j] <= sl
            hit_tp = high[j] >= tp
            if hit_sl and hit_tp:
                return (sl - entry) / entry, j - loc, "SL"
            if hit_sl:
                return (sl - entry) / entry, j - loc, "SL"
            if hit_tp:
                return (tp - entry) / entry, j - loc, "TP"
        else:
            hit_sl = high[j] >= sl
            hit_tp = low[j] <= tp
            if hit_sl and hit_tp:
                return (entry - sl) / entry, j - loc, "SL"
            if hit_sl:
                return (entry - sl) / entry, j - loc, "SL"
            if hit_tp:
                return (entry - tp) / entry, j - loc, "TP"

    # timeout
    px = float(close[end_j])
    holding = end_j - loc
    if side_l == "long":
        return (px - entry) / entry, holding, "TIMEOUT"
    return (entry - px) / entry, holding, "TIMEOUT"


def attach_resim_returns(
    panel: pd.DataFrame,
    h1: pd.DataFrame,
    *,
    tp_mult: float,
    sl_mult: float,
    horizon: int,
) -> pd.DataFrame:
    high = h1["high"].to_numpy(dtype=float)
    low = h1["low"].to_numpy(dtype=float)
    close = h1["close"].to_numpy(dtype=float)
    ts_index = pd.DatetimeIndex(pd.to_datetime(h1["timestamp"], utc=True))

    rets = []
    holds = []
    reasons = []
    for _, row in panel.iterrows():
        atr = float(row["atr_entry"] if "atr_entry" in row.index else row.get("atr_price", row.get("atr", 0)))
        r, h, reason = resim_barrier_row(
            side=str(row["side"]),
            entry_ts=pd.Timestamp(row["timestamp"]),
            entry_price=float(row["entry_price"]),
            atr=atr,
            high=high,
            low=low,
            close=close,
            ts_index=ts_index,
            tp_mult=tp_mult,
            sl_mult=sl_mult,
            horizon=horizon,
        )
        rets.append(r)
        holds.append(h)
        reasons.append(reason)

    out = panel.copy()
    out["net_return"] = rets  # overwrite for portfolio settle
    out["holding_bars"] = holds
    out["exit_reason_resim"] = reasons
    return out
