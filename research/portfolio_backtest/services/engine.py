"""Chronological portfolio engine with fixed-lot / risk sizing."""

from __future__ import annotations

from typing import Any, Literal

import numpy as np
import pandas as pd

from risk.position_sizing.fixed_fractional import size_long
from settings.strategy import CONTRACT_SIZE, SL_ATR_MULT, VOLUME_MIN
from research.portfolio_backtest import META_THRESHOLD


SystemKind = Literal["primary", "primary_plus_meta"]


def prepare_trade_universe(
    oof: pd.DataFrame, *, system: SystemKind, meta_thr: float = META_THRESHOLD
) -> pd.DataFrame:
    """Chronological candidate set. Primary = all; Meta = meta_proba >= thr."""
    frame = oof.copy()
    if "timestamp" not in frame.columns:
        raise ValueError("oof missing timestamp")
    frame = frame.sort_values("timestamp").reset_index(drop=True)
    if system == "primary_plus_meta":
        frame = frame.loc[frame["meta_proba"].astype(float) >= float(meta_thr)].copy()
    frame = frame.reset_index(drop=True)
    if "atr_entry" in frame.columns:
        atr = frame["atr_entry"].astype(float).to_numpy()
    else:
        atr = (
            frame["entry_price"].astype(float).to_numpy()
            * frame["distance_to_sl"].astype(float).to_numpy()
            / SL_ATR_MULT
        )
    frame["atr_price"] = atr
    frame["sl_dist"] = atr * SL_ATR_MULT
    return frame


def _lots_from_equity(
    equity: float,
    atr: float,
    *,
    mode: str,
    fixed_lots: float | None,
    risk_pct: float | None,
    enforce_volume_min: bool,
) -> float:
    if equity <= 0:
        return 0.0
    if mode == "fixed":
        return float(fixed_lots or 0.0)
    if enforce_volume_min:
        return float(
            size_long(
                equity=equity,
                atr=float(atr),
                risk_pct=float(risk_pct or 0.0),
                sl_atr_mult=SL_ATR_MULT,
                contract_size=CONTRACT_SIZE,
                enable_confidence_sizing=False,
            ).lots
        )
    sl_dist = SL_ATR_MULT * float(atr)
    if sl_dist <= 0:
        return 0.0
    return max((float(equity) * float(risk_pct or 0.0)) / (sl_dist * CONTRACT_SIZE), 0.0)


def run_portfolio(
    trades: pd.DataFrame,
    *,
    starting_equity: float,
    mode: str,
    fixed_lots: float | None = None,
    risk_pct: float | None = None,
    enforce_volume_min: bool = True,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Sequential compounding: size at entry from current equity, then settle PnL.
    Returns (trade_log, equity_curve).
    """
    n = len(trades)
    if n == 0:
        curve = pd.DataFrame(
            [{"timestamp": None, "equity": float(starting_equity), "drawdown": 0.0, "trade_i": -1}]
        )
        return pd.DataFrame(), curve

    ts = trades["timestamp"].to_numpy()
    years = trades["valid_year"].to_numpy(dtype=int)
    sides = trades["side"].astype(str).to_numpy()
    entry = trades["entry_price"].to_numpy(dtype=float)
    net = trades["net_return"].to_numpy(dtype=float)
    atr = trades["atr_price"].to_numpy(dtype=float)
    meta_p = (
        trades["meta_proba"].to_numpy(dtype=float)
        if "meta_proba" in trades.columns
        else np.full(n, np.nan)
    )

    eq = float(starting_equity)
    peak = eq
    lots_a = np.zeros(n)
    pnl_a = np.zeros(n)
    eq_before = np.zeros(n)
    eq_after = np.zeros(n)
    skipped = np.zeros(n, dtype=bool)
    curve_eq = [eq]
    curve_dd = [0.0]
    curve_ts: list[Any] = [None]
    curve_i = [-1]

    for i in range(n):
        lots = _lots_from_equity(
            eq,
            atr[i],
            mode=mode,
            fixed_lots=fixed_lots,
            risk_pct=risk_pct,
            enforce_volume_min=enforce_volume_min,
        )
        min_ok = lots >= VOLUME_MIN if enforce_volume_min else lots > 0
        eq_before[i] = eq
        if (not min_ok) or eq <= 0:
            skipped[i] = True
            eq_after[i] = eq
            continue
        pnl = lots * CONTRACT_SIZE * entry[i] * net[i]
        lots_a[i] = lots
        pnl_a[i] = pnl
        eq = eq + pnl
        eq_after[i] = eq
        peak = max(peak, eq)
        dd = (peak - eq) / peak if peak > 0 else 0.0
        curve_eq.append(eq)
        curve_dd.append(dd)
        curve_ts.append(ts[i])
        curve_i.append(int(np.sum(~skipped[: i + 1]) - 1))
        if eq <= 0:
            # mark remaining as skipped
            skipped[i + 1 :] = True
            eq_after[i + 1 :] = eq
            eq_before[i + 1 :] = eq
            break

    log = pd.DataFrame(
        {
            "trade_i": np.arange(n),
            "timestamp": ts,
            "valid_year": years,
            "side": sides,
            "entry_price": entry,
            "net_return": net,
            "meta_proba": meta_p,
            "lots": lots_a,
            "pnl": pnl_a,
            "equity_before": eq_before,
            "equity_after": eq_after,
            "skipped": skipped,
            "skip_reason": np.where(skipped, "lots_below_min_or_blown", ""),
        }
    )
    curve = pd.DataFrame(
        {
            "timestamp": curve_ts,
            "equity": curve_eq,
            "drawdown": curve_dd,
            "trade_i": curve_i,
        }
    )
    return log, curve
