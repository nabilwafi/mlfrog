"""Causal H1 path diagnostics for ATRE (no future leak)."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from research.atre import COST, HORIZON, SL_ATR, TP_ATR
from research.position_mgmt.services.paths import entry_indices, prepare_market, window_end


def _adverse_atr(side: str, entry: float, low: float, high: float, atr: float) -> float:
    if atr <= 0:
        return 0.0
    if side == "long":
        return max(0.0, (entry - low) / atr)
    return max(0.0, (high - entry) / atr)


def _favor_atr(side: str, entry: float, low: float, high: float, atr: float) -> float:
    if atr <= 0:
        return 0.0
    if side == "long":
        return max(0.0, (high - entry) / atr)
    return max(0.0, (entry - low) / atr)


def _pnl_frac(side: str, entry: float, px: float) -> float:
    return (px - entry) / entry if side == "long" else (entry - px) / entry


def walk_trade_path(
    *,
    side: str,
    entry: float,
    atr_entry: float,
    ei: int,
    mkt: dict[str, Any],
    horizon: int = HORIZON,
) -> dict[str, Any]:
    """
    Walk original triple-barrier path. Returns bar-level arrays + final outcome.
    Same-bar: SL before TP (conservative).
    """
    high, low, close = mkt["high"], mkt["low"], mkt["close"]
    atr_s, adx_s, vol = mkt["atr"], mkt["adx"], mkt["vol"]
    n = len(close)
    if ei < 0 or ei >= n - 1 or atr_entry <= 0:
        return {
            "ok": False,
            "net_return": -COST,
            "exit_reason": "bad",
            "hit_tp": False,
            "bars": [],
        }

    is_long = side == "long"
    sl = entry - SL_ATR * atr_entry if is_long else entry + SL_ATR * atr_entry
    tp = entry + TP_ATR * atr_entry if is_long else entry - TP_ATR * atr_entry
    end = min(ei + horizon, n - 1)

    adx0 = float(adx_s[ei]) if np.isfinite(adx_s[ei]) else np.nan
    vol0 = float(np.nanmean(vol[max(0, ei - 10) : ei + 1]))
    atr0 = float(atr_entry)

    max_adverse = 0.0
    max_favor = 0.0
    underwater = 0
    bars: list[dict[str, Any]] = []
    exit_reason = "TIMEOUT"
    exit_px = float(close[end])
    j_exit = end

    # structure: prior swing proxy = min/max of prior 5 bars before entry
    prior_lo = float(np.min(low[max(0, ei - 5) : ei + 1]))
    prior_hi = float(np.max(high[max(0, ei - 5) : ei + 1]))

    for j in range(ei + 1, end + 1):
        h, l, c = float(high[j]), float(low[j]), float(close[j])
        adv = _adverse_atr(side, entry, l, h, atr0)
        fav = _favor_atr(side, entry, l, h, atr0)
        max_adverse = max(max_adverse, adv)
        max_favor = max(max_favor, fav)
        mid_pnl = _pnl_frac(side, entry, c)
        if mid_pnl < 0:
            underwater += 1

        atr_j = float(atr_s[j]) if np.isfinite(atr_s[j]) else atr0
        adx_j = float(adx_s[j]) if np.isfinite(adx_s[j]) else adx0
        vol_j = float(np.nanmean(vol[max(0, j - 5) : j + 1]))

        atr_ratio = atr_j / atr0 if atr0 > 0 else 1.0
        adx_ratio = (adx_j / adx0) if (adx0 == adx0 and adx0 and adx0 > 0) else 1.0
        vol_ratio = (vol_j / vol0) if vol0 > 0 else 1.0

        struct_fail = (c < prior_lo) if is_long else (c > prior_hi)
        mom_collapse = (atr_ratio < 0.7) or (adx_ratio < 0.7) or (vol_ratio < 0.5)

        dist_sl = abs(c - sl) / atr0 if atr0 > 0 else 0.0
        dist_tp = abs(tp - c) / atr0 if atr0 > 0 else 0.0

        bars.append(
            {
                "bar": j - ei,
                "close": c,
                "adverse_atr": float(max_adverse),
                "favor_atr": float(max_favor),
                "underwater_bars": int(underwater),
                "floating_pnl": float(mid_pnl),
                "atr_ratio": float(atr_ratio),
                "adx_ratio": float(adx_ratio if adx_ratio == adx_ratio else 1.0),
                "vol_ratio": float(vol_ratio),
                "struct_fail": bool(struct_fail),
                "mom_collapse": bool(mom_collapse),
                "dist_sl_atr": float(dist_sl),
                "dist_tp_atr": float(dist_tp),
            }
        )

        hit_sl = (l <= sl) if is_long else (h >= sl)
        hit_tp = (h >= tp) if is_long else (l <= tp)
        if hit_sl and hit_tp:
            exit_reason, exit_px, j_exit = "SL", sl, j
            break
        if hit_sl:
            exit_reason, exit_px, j_exit = "SL", sl, j
            break
        if hit_tp:
            exit_reason, exit_px, j_exit = "TP", tp, j
            break
        if j >= end:
            exit_reason, exit_px, j_exit = "TIMEOUT", c, j
            break

    net = _pnl_frac(side, entry, exit_px) - COST
    return {
        "ok": True,
        "net_return": float(net),
        "exit_reason": exit_reason,
        "hit_tp": exit_reason == "TP",
        "hit_sl": exit_reason == "SL",
        "holding_bars": int(j_exit - ei),
        "max_adverse_atr": float(max_adverse),
        "max_favor_atr": float(max_favor),
        "underwater_bars": int(underwater),
        "bars": bars,
        "entry": entry,
        "atr_entry": atr_entry,
        "sl": sl,
        "tp": tp,
        "side": side,
        "ei": ei,
    }


def build_path_library(trades: pd.DataFrame, h1: pd.DataFrame) -> list[dict[str, Any]]:
    mkt = prepare_market(h1)
    eis = entry_indices(trades, mkt["ts"])
    lib = []
    for i in range(len(trades)):
        row = trades.iloc[i]
        atr = float(row["atr_entry"]) if pd.notna(row.get("atr_entry")) else float(row.get("atr_price", np.nan))
        if not np.isfinite(atr) or atr <= 0:
            atr = float(row["entry_price"]) * float(row.get("distance_to_sl", 0.001)) / SL_ATR
        path = walk_trade_path(
            side=str(row["side"]),
            entry=float(row["entry_price"]),
            atr_entry=atr,
            ei=int(eis[i]),
            mkt=mkt,
        )
        path["trade_i"] = i
        path["valid_year"] = int(row["valid_year"]) if "valid_year" in row.index else int(pd.Timestamp(row["timestamp"]).year)
        path["confidence"] = float(row.get("confidence", 50) or 50)
        path["d1_regime"] = str(row.get("d1_regime", "") or "")
        path["session_asia"] = float(row.get("session_asia", 0) or 0)
        path["session_london"] = float(row.get("session_london", 0) or 0)
        path["session_newyork"] = float(row.get("session_newyork", 0) or 0)
        path["session_overlap"] = float(row.get("session_london_ny_overlap", 0) or 0)
        path["vol_rank"] = float(row.get("volatility_rank", row.get("atr_percentile_252", 0.5)) or 0.5)
        path["m5_q"] = float(row.get("m5_entry_quality", 50) or 50) / 100.0
        path["h4_swing"] = float(row.get("ctx_h4_swing_quality", 0.5) or 0.5)
        path["stored_net"] = float(row["net_return"])
        lib.append(path)
    return lib
