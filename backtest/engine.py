"""
Event-driven LONG backtest for LightGBM v1 @ threshold 0.52.

Cost / PnL uses Method A (grounded economics):
    PnL_USD = (exit_fill - entry_fill) * CONTRACT_SIZE * lots

Verified Finex XAUUSD symbol_info:
    digits=2, point=0.01, trade_tick_size=0.01, trade_contract_size=100,
    trade_tick_value=10 (INCONSISTENT with contract*point=$1/point).
We intentionally ignore trade_tick_value and use Method A.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal

import numpy as np
import pandas as pd

# --- Verified symbol specs (Finex XAUUSD via mt5.symbol_info) ---
POINT = 0.01
CONTRACT_SIZE = 100.0  # oz per 1.0 lot

# --- Strategy / label-consistent barriers ---
SELECTED_THR = 0.52
SL_ATR_MULT = 1.5
TP_ATR_MULT = 2.0
HORIZON_BARS = 8

# --- Cost model ---
# Per-bar Spread from raw MT5 (points). Fallback if missing/invalid.
FALLBACK_SPREAD_POINTS = 19
ASSUMED_SLIPPAGE_POINTS = 5

# --- Risk ---
STARTING_EQUITY = 10_000.0
RISK_PER_TRADE_PCT = 0.01
BLOWN_DRAWDOWN_PCT = 0.30  # stop if equity < start * (1 - this)
VOLUME_STEP = 0.01
VOLUME_MIN = 0.01

ExitReason = Literal["tp", "sl", "timeout"]


@dataclass
class BacktestConfig:
    apply_costs: bool = True
    selected_thr: float = SELECTED_THR
    sl_atr_mult: float = SL_ATR_MULT
    tp_atr_mult: float = TP_ATR_MULT
    horizon_bars: int = HORIZON_BARS
    slippage_points: float = ASSUMED_SLIPPAGE_POINTS
    fallback_spread_points: float = FALLBACK_SPREAD_POINTS
    starting_equity: float = STARTING_EQUITY
    risk_per_trade_pct: float = RISK_PER_TRADE_PCT
    blown_drawdown_pct: float = BLOWN_DRAWDOWN_PCT
    point: float = POINT
    contract_size: float = CONTRACT_SIZE
    # Volatility regime guard (entry-only). Requires df columns atr_zscore + regime_tier.
    enable_regime_guard: bool = False
    regime_size_reduction_elevated: float = 0.5
    # Confidence sizing (multiplies with regime mult). Requires df column confidence_tier.
    enable_confidence_sizing: bool = False
    confidence_size_mult: dict[str, float] = field(
        default_factory=lambda: {"medium": 0.75, "high": 1.25}
    )


@dataclass
class Trade:
    entry_time: pd.Timestamp
    exit_time: pd.Timestamp
    entry_idx: int
    exit_idx: int
    bars_held: int
    atr: float
    spread_points_entry: float
    spread_points_exit: float
    entry_ref_close: float
    entry_fill: float
    exit_raw: float
    exit_fill: float
    sl_level: float
    tp_level: float
    lots: float
    pnl_usd: float
    reason: ExitReason
    equity_after: float
    skipped_signals_while_open: int = 0
    regime_tier: str = "normal"
    size_mult: float = 1.0
    atr_zscore: float = float("nan")
    confidence_tier: str = "n/a"
    confidence_size_mult: float = 1.0
    regime_size_mult: float = 1.0


@dataclass
class BacktestResult:
    trades: list[Trade] = field(default_factory=list)
    equity_curve: pd.DataFrame = field(default_factory=pd.DataFrame)
    n_signals_raw: int = 0
    n_signals_skipped_stacking: int = 0
    n_fallback_spread_uses: int = 0
    n_signals_blocked_extreme: int = 0
    n_trades_size_reduced_elevated: int = 0
    blown_account: bool = False
    blown_time: pd.Timestamp | None = None
    config: BacktestConfig = field(default_factory=BacktestConfig)


def load_backtest_frame(
    *,
    preds_path: Path,
    features_path: Path,
    h1_raw_path: Path,
) -> pd.DataFrame:
    """Build chronological test frame: Date, OHLC, Spread, atr_h1, y_prob, y_true."""
    preds = pd.read_parquet(preds_path)
    preds["Date"] = pd.to_datetime(preds["Date"], utc=True)

    feat = pd.read_parquet(features_path, columns=["Date", "atr_h1"])
    feat["Date"] = pd.to_datetime(feat["Date"], utc=True)

    h1 = pd.read_csv(h1_raw_path, parse_dates=["Date"])
    if getattr(h1["Date"].dtype, "tz", None) is None:
        h1["Date"] = pd.to_datetime(h1["Date"], utc=True)
    # Align to close-time index used by features/preds (open + 1h).
    h1 = h1.rename(columns={"Date": "open_time"})
    h1["Date"] = h1["open_time"] + pd.Timedelta(hours=1)

    df = (
        preds.merge(feat, on="Date", how="left")
        .merge(
            h1[["Date", "Open", "High", "Low", "Close", "Spread"]],
            on="Date",
            how="left",
        )
        .sort_values("Date")
        .reset_index(drop=True)
    )
    missing = df[["atr_h1", "High", "Low", "Close"]].isna().any(axis=1).sum()
    if missing:
        raise ValueError(f"Backtest frame has {missing} rows missing OHLC/ATR after join")
    return df


def _spread_points(row: pd.Series, cfg: BacktestConfig) -> tuple[float, bool]:
    """Return (spread_points, used_fallback)."""
    sp = row.get("Spread", np.nan)
    if pd.isna(sp) or float(sp) < 0:
        return float(cfg.fallback_spread_points), True
    # Historical zeros exist outside test; treat non-positive as fallback.
    if float(sp) <= 0:
        return float(cfg.fallback_spread_points), True
    return float(sp), False


def _cost_price(spread_points: float, cfg: BacktestConfig) -> float:
    if not cfg.apply_costs:
        return 0.0
    return (spread_points + cfg.slippage_points) * cfg.point


def _round_lots(lots: float) -> float:
    if lots < VOLUME_MIN:
        return 0.0
    steps = np.floor(lots / VOLUME_STEP + 1e-12)
    return float(max(VOLUME_MIN, steps * VOLUME_STEP))


def run_backtest(df: pd.DataFrame, cfg: BacktestConfig) -> BacktestResult:
    """Event-driven LONG-only backtest, one position at a time (no stacking)."""
    n = len(df)
    high = df["High"].to_numpy(dtype=float)
    low = df["Low"].to_numpy(dtype=float)
    close = df["Close"].to_numpy(dtype=float)
    atr = df["atr_h1"].to_numpy(dtype=float)
    prob = df["y_prob"].to_numpy(dtype=float)
    dates = df["Date"].to_numpy()

    equity = float(cfg.starting_equity)
    peak = equity
    floor = cfg.starting_equity * (1.0 - cfg.blown_drawdown_pct)

    trades: list[Trade] = []
    equity_rows: list[dict] = []
    n_signals_raw = 0
    n_skipped = 0
    n_fallback = 0
    n_blocked_extreme = 0
    n_size_reduced = 0
    blown = False
    blown_time = None

    if cfg.enable_regime_guard:
        if "regime_tier" not in df.columns or "atr_zscore" not in df.columns:
            raise ValueError("regime guard requires columns atr_zscore + regime_tier")
        regime = df["regime_tier"].astype(str).to_numpy()
        atr_z = df["atr_zscore"].to_numpy(dtype=float)
    else:
        regime = np.array(["normal"] * n, dtype=object)
        atr_z = np.full(n, np.nan)

    if cfg.enable_confidence_sizing:
        if "confidence_tier" not in df.columns:
            raise ValueError("confidence sizing requires column confidence_tier")
        conf_tiers = df["confidence_tier"].astype(str).to_numpy()
    else:
        conf_tiers = np.array(["n/a"] * n, dtype=object)

    pos_open = False
    entry_i = -1
    entry_time = None
    entry_ref = 0.0
    entry_fill = 0.0
    sl_level = 0.0
    tp_level = 0.0
    entry_atr = 0.0
    entry_spread_pts = 0.0
    lots = 0.0
    skipped_while_open = 0
    entry_regime = "normal"
    entry_size_mult = 1.0
    entry_regime_mult = 1.0
    entry_conf_mult = 1.0
    entry_conf_tier = "n/a"
    entry_atr_z = float("nan")

    def mark_equity(ts, eq):
        nonlocal peak
        peak = max(peak, eq)
        dd = 0.0 if peak <= 0 else (peak - eq) / peak
        equity_rows.append(
            {
                "timestamp": pd.Timestamp(ts),
                "equity": eq,
                "drawdown_pct": dd * 100.0,
            }
        )

    mark_equity(dates[0], equity)

    i = 0
    while i < n:
        # Manage open position on this bar (bar i is a post-entry bar).
        if pos_open:
            # Conservative: SL before TP if both touch same candle.
            hit_sl = low[i] <= sl_level
            hit_tp = high[i] >= tp_level
            bars_held = i - entry_i
            reason: ExitReason | None = None
            exit_raw = 0.0

            if hit_sl:
                reason = "sl"
                exit_raw = sl_level
            elif hit_tp:
                reason = "tp"
                exit_raw = tp_level
            elif bars_held >= cfg.horizon_bars:
                reason = "timeout"
                exit_raw = close[i]

            if reason is not None:
                sp_exit, used_fb = _spread_points(df.iloc[i], cfg)
                if used_fb:
                    n_fallback += 1
                exit_cost = _cost_price(sp_exit, cfg)
                exit_fill = exit_raw - exit_cost  # LONG exit: sell at worse price
                pnl = (exit_fill - entry_fill) * cfg.contract_size * lots
                equity += pnl

                trades.append(
                    Trade(
                        entry_time=pd.Timestamp(entry_time),
                        exit_time=pd.Timestamp(dates[i]),
                        entry_idx=entry_i,
                        exit_idx=i,
                        bars_held=bars_held,
                        atr=entry_atr,
                        spread_points_entry=entry_spread_pts,
                        spread_points_exit=sp_exit,
                        entry_ref_close=entry_ref,
                        entry_fill=entry_fill,
                        exit_raw=exit_raw,
                        exit_fill=exit_fill,
                        sl_level=sl_level,
                        tp_level=tp_level,
                        lots=lots,
                        pnl_usd=pnl,
                        reason=reason,
                        equity_after=equity,
                        skipped_signals_while_open=skipped_while_open,
                        regime_tier=entry_regime,
                        size_mult=entry_size_mult,
                        atr_zscore=entry_atr_z,
                        confidence_tier=entry_conf_tier,
                        confidence_size_mult=entry_conf_mult,
                        regime_size_mult=entry_regime_mult,
                    )
                )
                mark_equity(dates[i], equity)
                pos_open = False
                skipped_while_open = 0

                if equity < floor:
                    blown = True
                    blown_time = pd.Timestamp(dates[i])
                    break

                # After close, allow same bar to open a new signal (optional).
                # Baseline: evaluate entry on this bar after close.
            else:
                # Still open: skip new signals on this bar.
                if prob[i] >= cfg.selected_thr:
                    n_signals_raw += 1
                    n_skipped += 1
                    skipped_while_open += 1
                i += 1
                continue

        # Flat: maybe open
        if (not pos_open) and (prob[i] >= cfg.selected_thr):
            n_signals_raw += 1
            if i + 1 >= n:
                # Need at least one future bar to manage; skip orphan signal.
                i += 1
                continue

            # Entry-only regime guard (do not force-close open trades).
            tier = str(regime[i])
            regime_mult = 1.0
            if cfg.enable_regime_guard:
                if tier == "extreme":
                    n_blocked_extreme += 1
                    i += 1
                    continue
                if tier == "elevated":
                    regime_mult = float(cfg.regime_size_reduction_elevated)
                elif not np.isfinite(atr_z[i]):
                    # Warm-up / undefined z: treat as normal (allow trade).
                    tier = "normal"

            conf_tier = str(conf_tiers[i])
            conf_mult = 1.0
            if cfg.enable_confidence_sizing:
                conf_mult = float(cfg.confidence_size_mult.get(conf_tier, 1.0))

            # final = regime × confidence (product, never replace)
            size_mult = regime_mult * conf_mult

            sp_entry, used_fb = _spread_points(df.iloc[i], cfg)
            if used_fb:
                n_fallback += 1
            entry_cost = _cost_price(sp_entry, cfg)
            entry_ref = close[i]
            entry_atr = atr[i]
            if not np.isfinite(entry_atr) or entry_atr <= 0:
                i += 1
                continue

            sl_dist = cfg.sl_atr_mult * entry_atr
            risk_amount = equity * cfg.risk_per_trade_pct * size_mult
            raw_lots = risk_amount / (sl_dist * cfg.contract_size)
            lots = _round_lots(raw_lots)
            if lots <= 0:
                i += 1
                continue

            if size_mult < 1.0 - 1e-12:
                n_size_reduced += 1

            entry_fill = entry_ref + entry_cost
            sl_level = entry_ref - cfg.sl_atr_mult * entry_atr
            tp_level = entry_ref + cfg.tp_atr_mult * entry_atr
            # Barriers from label-consistent close reference (not fill), matching training labels.

            pos_open = True
            entry_i = i
            entry_time = dates[i]
            entry_spread_pts = sp_entry
            skipped_while_open = 0
            entry_regime = tier
            entry_size_mult = size_mult
            entry_regime_mult = regime_mult
            entry_conf_mult = conf_mult
            entry_conf_tier = conf_tier
            entry_atr_z = float(atr_z[i]) if np.isfinite(atr_z[i]) else float("nan")
            # Position starts being checked from next bar.
            i += 1
            continue

        i += 1

    # If still open at end of data, force timeout at last bar.
    if pos_open and not blown:
        i = n - 1
        sp_exit, used_fb = _spread_points(df.iloc[i], cfg)
        if used_fb:
            n_fallback += 1
        exit_cost = _cost_price(sp_exit, cfg)
        exit_raw = close[i]
        exit_fill = exit_raw - exit_cost
        pnl = (exit_fill - entry_fill) * cfg.contract_size * lots
        equity += pnl
        trades.append(
            Trade(
                entry_time=pd.Timestamp(entry_time),
                exit_time=pd.Timestamp(dates[i]),
                entry_idx=entry_i,
                exit_idx=i,
                bars_held=i - entry_i,
                atr=entry_atr,
                spread_points_entry=entry_spread_pts,
                spread_points_exit=sp_exit,
                entry_ref_close=entry_ref,
                entry_fill=entry_fill,
                exit_raw=exit_raw,
                exit_fill=exit_fill,
                sl_level=sl_level,
                tp_level=tp_level,
                lots=lots,
                pnl_usd=pnl,
                reason="timeout",
                equity_after=equity,
                skipped_signals_while_open=skipped_while_open,
                regime_tier=entry_regime,
                size_mult=entry_size_mult,
                atr_zscore=entry_atr_z,
                confidence_tier=entry_conf_tier,
                confidence_size_mult=entry_conf_mult,
                regime_size_mult=entry_regime_mult,
            )
        )
        mark_equity(dates[i], equity)

    eq_df = pd.DataFrame(equity_rows)
    if not eq_df.empty:
        eq_df = eq_df.drop_duplicates(subset=["timestamp"], keep="last").reset_index(drop=True)

    return BacktestResult(
        trades=trades,
        equity_curve=eq_df,
        n_signals_raw=n_signals_raw,
        n_signals_skipped_stacking=n_skipped,
        n_fallback_spread_uses=n_fallback,
        n_signals_blocked_extreme=n_blocked_extreme,
        n_trades_size_reduced_elevated=n_size_reduced,
        blown_account=blown,
        blown_time=blown_time,
        config=cfg,
    )


def compute_metrics(result: BacktestResult) -> dict:
    cfg = result.config
    trades = result.trades
    eq = result.equity_curve

    n = len(trades)
    wins = [t for t in trades if t.reason == "tp"]
    losses = [t for t in trades if t.reason == "sl"]
    timeouts = [t for t in trades if t.reason == "timeout"]

    # Realized win-rate among resolved TP/SL only (matches model precision definition).
    n_resolved = len(wins) + len(losses)
    winrate = len(wins) / n_resolved if n_resolved else float("nan")

    pnls = np.array([t.pnl_usd for t in trades], dtype=float) if trades else np.array([])
    gross_profit = float(pnls[pnls > 0].sum()) if len(pnls) else 0.0
    gross_loss = float(-pnls[pnls < 0].sum()) if len(pnls) else 0.0
    profit_factor = gross_profit / gross_loss if gross_loss > 0 else float("inf")

    start_eq = cfg.starting_equity
    end_eq = float(eq["equity"].iloc[-1]) if len(eq) else start_eq
    total_return = end_eq / start_eq - 1.0

    # CAGR from first to last equity timestamp.
    if len(eq) >= 2:
        t0 = pd.Timestamp(eq["timestamp"].iloc[0])
        t1 = pd.Timestamp(eq["timestamp"].iloc[-1])
        years = max((t1 - t0).total_seconds() / (365.25 * 24 * 3600), 1e-9)
        cagr = (end_eq / start_eq) ** (1.0 / years) - 1.0
    else:
        years = 0.0
        cagr = 0.0

    max_dd = float(eq["drawdown_pct"].max()) if len(eq) else 0.0

    # Max drawdown duration: longest consecutive period with drawdown_pct > 0.
    max_dd_bars = 0
    cur = 0
    if len(eq):
        for dd in eq["drawdown_pct"].to_numpy():
            if dd > 0:
                cur += 1
                max_dd_bars = max(max_dd_bars, cur)
            else:
                cur = 0

    # Sharpe on trade returns (pnl/equity_before), annualized loosely by trades/year.
    # Simpler: daily equity returns if we have equity marks; use trade PnL / start as proxy.
    # Use equity curve step returns.
    sharpe = float("nan")
    if len(eq) >= 3:
        rets = eq["equity"].pct_change().dropna().to_numpy()
        if rets.std(ddof=1) > 0:
            # Equity marks are per trade event, not daily. Annualize by events/year.
            events_per_year = len(rets) / years if years > 0 else len(rets)
            sharpe = float(rets.mean() / rets.std(ddof=1) * np.sqrt(events_per_year))

    avg_bars = float(np.mean([t.bars_held for t in trades])) if trades else float("nan")

    return {
        "apply_costs": cfg.apply_costs,
        "starting_equity": start_eq,
        "ending_equity": end_eq,
        "total_return_pct": total_return * 100.0,
        "cagr_pct": cagr * 100.0,
        "years": years,
        "max_drawdown_pct": max_dd,
        "max_drawdown_duration_events": max_dd_bars,
        "sharpe_annualized_approx": sharpe,
        "risk_free_rate_assumed": 0.0,
        "profit_factor": profit_factor,
        "gross_profit": gross_profit,
        "gross_loss": gross_loss,
        "n_trades": n,
        "n_tp": len(wins),
        "n_sl": len(losses),
        "n_timeout": len(timeouts),
        "winrate_tp_sl": winrate,
        "avg_bars_held": avg_bars,
        "n_signals_raw": result.n_signals_raw,
        "n_signals_skipped_stacking": result.n_signals_skipped_stacking,
        "n_fallback_spread_uses": result.n_fallback_spread_uses,
        "n_signals_blocked_extreme": result.n_signals_blocked_extreme,
        "n_trades_size_reduced_elevated": result.n_trades_size_reduced_elevated,
        "blown_account": result.blown_account,
        "blown_time": result.blown_time,
    }
