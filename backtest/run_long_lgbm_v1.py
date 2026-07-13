"""
Run LONG LightGBM v1 backtest (with-cost and no-cost), save equity + report.

Usage:
    python -m backtest.run_long_lgbm_v1
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from backtest.engine import (
    ASSUMED_SLIPPAGE_POINTS,
    CONTRACT_SIZE,
    FALLBACK_SPREAD_POINTS,
    POINT,
    SELECTED_THR,
    BacktestConfig,
    compute_metrics,
    load_backtest_frame,
    run_backtest,
)

# Model-eval sanity targets (from modeling/robustness reports)
EVAL_N_SIGNALS = 878
EVAL_PRECISION = 0.5228


def _trades_to_df(result) -> pd.DataFrame:
    rows = []
    for t in result.trades:
        rows.append(
            {
                "entry_time": t.entry_time,
                "exit_time": t.exit_time,
                "bars_held": t.bars_held,
                "reason": t.reason,
                "lots": t.lots,
                "atr": t.atr,
                "spread_points_entry": t.spread_points_entry,
                "spread_points_exit": t.spread_points_exit,
                "entry_ref_close": t.entry_ref_close,
                "entry_fill": t.entry_fill,
                "exit_raw": t.exit_raw,
                "exit_fill": t.exit_fill,
                "sl_level": t.sl_level,
                "tp_level": t.tp_level,
                "pnl_usd": t.pnl_usd,
                "equity_after": t.equity_after,
                "skipped_signals_while_open": t.skipped_signals_while_open,
            }
        )
    return pd.DataFrame(rows)


def _plot_equity(eq: pd.DataFrame, title: str, out_path: Path) -> None:
    fig, axes = plt.subplots(2, 1, figsize=(11, 7), sharex=True, gridspec_kw={"height_ratios": [2, 1]})
    ax0, ax1 = axes
    ax0.plot(eq["timestamp"], eq["equity"], color="#1f77b4", lw=1.5)
    ax0.set_ylabel("Equity (USD)")
    ax0.set_title(title)
    ax0.grid(True, alpha=0.3)

    ax1.fill_between(eq["timestamp"], eq["drawdown_pct"], 0.0, color="#d62728", alpha=0.35)
    ax1.plot(eq["timestamp"], eq["drawdown_pct"], color="#d62728", lw=1.0)
    ax1.set_ylabel("Drawdown %")
    ax1.set_xlabel("Time (UTC)")
    ax1.grid(True, alpha=0.3)
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=140)
    plt.close(fig)


def _fmt_metrics(m: dict) -> str:
    lines = [
        f"- apply_costs: {m['apply_costs']}",
        f"- starting_equity: {m['starting_equity']:.2f}",
        f"- ending_equity: {m['ending_equity']:.2f}",
        f"- total_return_pct: {m['total_return_pct']:.2f}%",
        f"- CAGR_pct: {m['cagr_pct']:.2f}% (years={m['years']:.3f})",
        f"- max_drawdown_pct: {m['max_drawdown_pct']:.2f}%",
        f"- max_drawdown_duration_events: {m['max_drawdown_duration_events']}",
        f"- sharpe_annualized_approx: {m['sharpe_annualized_approx']:.4f} (rf=0 assumed)",
        f"- profit_factor: {m['profit_factor']:.4f}",
        f"- gross_profit: {m['gross_profit']:.2f}",
        f"- gross_loss: {m['gross_loss']:.2f}",
        f"- n_trades: {m['n_trades']}",
        f"- n_tp / n_sl / n_timeout: {m['n_tp']} / {m['n_sl']} / {m['n_timeout']}",
        f"- winrate_tp_sl: {m['winrate_tp_sl']:.4f}",
        f"- avg_bars_held: {m['avg_bars_held']:.3f}",
        f"- n_signals_raw (incl. skipped): {m['n_signals_raw']}",
        f"- n_signals_skipped_stacking: {m['n_signals_skipped_stacking']}",
        f"- n_fallback_spread_uses: {m['n_fallback_spread_uses']}",
        f"- blown_account: {m['blown_account']} ({m['blown_time']})",
    ]
    return "\n".join(lines)


def build_report(m_cost: dict, m_nocost: dict) -> str:
    # Sanity vs model eval
    wr = m_cost["winrate_tp_sl"]
    n_tr = m_cost["n_trades"]
    n_skip = m_cost["n_signals_skipped_stacking"]
    n_raw = m_cost["n_signals_raw"]

    lines = [
        "# Backtest Report — LONG LightGBM v1 @ thr=0.52",
        "",
        "## Unit / cost assumptions (verified)",
        "",
        "From live `mt5.symbol_info('XAUUSD')` (Finex):",
        f"- digits=2, point={POINT}, trade_tick_size=0.01, trade_contract_size={CONTRACT_SIZE}",
        "- trade_tick_value=10.0 (INCONSISTENT with contract*point = 1.0 USD/point/lot)",
        "- Formula check: `tick_value/tick_size*point = 10/0.01*0.01 = 10` != Method A `1.0`",
        "  Hypothesis that tick_size=0.1 is **FALSE** (actual tick_size=0.01). Genuine Finex quirk.",
        "- **PnL Method A used:** `(exit_fill - entry_fill) * contract_size * lots`",
        f"- Example: 19 points spread = {19 * POINT:.2f} USD/oz = **{19 * POINT * CONTRACT_SIZE:.2f} USD per 1.0 lot**",
        f"- Slippage {ASSUMED_SLIPPAGE_POINTS} points = **{ASSUMED_SLIPPAGE_POINTS * POINT * CONTRACT_SIZE:.2f} USD per 1.0 lot per side**",
        f"- Per-bar Spread from raw H1; fallback flat {FALLBACK_SPREAD_POINTS} points if missing/invalid",
        f"- ASSUMED_SLIPPAGE_POINTS = {ASSUMED_SLIPPAGE_POINTS}",
        "",
        "## Simulation rules",
        f"- Threshold: {SELECTED_THR} (frozen from validation)",
        "- Barriers: SL=1.5*ATR, TP=2.0*ATR from entry close (label-consistent); horizon=8 H1 bars",
        "- Ambiguous same-candle TP+SL -> SL first (conservative, same as labeling)",
        "- Timeout -> close at close of horizon bar (realized PnL, not forced loss)",
        "- No stacking: skip new signals while a position is open",
        "- Risk: 1% equity per trade; stop if equity < 70% of starting equity",
        "",
        "## Sanity check vs model evaluation",
        f"- Model-eval thresholded signals: n={EVAL_N_SIGNALS}, precision={EVAL_PRECISION}",
        f"- Backtest (with cost): n_trades={n_tr}, winrate_tp_sl={wr:.4f}",
        f"- Signals seen raw={n_raw}, skipped_by_no_stacking={n_skip}",
        "",
        "### Explanation of differences (expected, not automatic bug)",
        "- Model precision counts **all** test rows with y_prob>=0.52 (n=878).",
        "- Backtest **skips** signals while a position is open (no stacking), so n_trades < 878.",
        "- Win-rate compares TP vs SL exits only; timeouts are separate.",
        "- Costs change PnL/equity but do **not** change TP/SL touch outcomes.",
        "",
        "## WITH costs (per-bar spread + slippage)",
        _fmt_metrics(m_cost),
        "",
        "## WITHOUT costs (spread=0, slippage=0)",
        _fmt_metrics(m_nocost),
        "",
        "## Cost impact",
        (
            f"- Ending equity with cost: {m_cost['ending_equity']:.2f} | "
            f"without cost: {m_nocost['ending_equity']:.2f}"
        ),
        (
            f"- CAGR with cost: {m_cost['cagr_pct']:.2f}% | "
            f"without cost: {m_nocost['cagr_pct']:.2f}%"
        ),
        (
            f"- Max DD with cost: {m_cost['max_drawdown_pct']:.2f}% | "
            f"without cost: {m_nocost['max_drawdown_pct']:.2f}%"
        ),
        (
            f"- Equity eaten by costs (end): "
            f"{m_nocost['ending_equity'] - m_cost['ending_equity']:.2f} USD"
        ),
        "",
        "## Artifacts",
        "- `data/backtest/equity_with_cost.parquet`",
        "- `data/backtest/equity_no_cost.parquet`",
        "- `data/backtest/trades_with_cost.parquet`",
        "- `data/backtest/trades_no_cost.parquet`",
        "- `data/backtest/equity_with_cost.png`",
        "- `data/backtest/equity_no_cost.png`",
        "",
    ]
    return "\n".join(lines)


def main() -> None:
    base = Path("data")
    preds_path = base / "models/lightgbm_long_test_preds.parquet"
    features_path = base / "features/xauusd_h1_h4_d1_features.parquet"
    h1_path = base / "raw/XAUUSD_H1.csv"
    out_dir = base / "backtest"
    report_path = base / "reports/backtest_report.md"
    out_dir.mkdir(parents=True, exist_ok=True)

    print("Loading backtest frame...")
    df = load_backtest_frame(
        preds_path=preds_path,
        features_path=features_path,
        h1_raw_path=h1_path,
    )
    print(
        f"rows={len(df)} | {df['Date'].min()} -> {df['Date'].max()} | "
        f"signals={(df['y_prob'] >= SELECTED_THR).sum()}"
    )

    cfg_cost = BacktestConfig(apply_costs=True)
    cfg_flat = BacktestConfig(apply_costs=False)

    print("Running WITH costs...")
    res_cost = run_backtest(df, cfg_cost)
    print("Running WITHOUT costs...")
    res_flat = run_backtest(df, cfg_flat)

    m_cost = compute_metrics(res_cost)
    m_flat = compute_metrics(res_flat)

    # Save artifacts
    res_cost.equity_curve.to_parquet(out_dir / "equity_with_cost.parquet", index=False)
    res_flat.equity_curve.to_parquet(out_dir / "equity_no_cost.parquet", index=False)
    _trades_to_df(res_cost).to_parquet(out_dir / "trades_with_cost.parquet", index=False)
    _trades_to_df(res_flat).to_parquet(out_dir / "trades_no_cost.parquet", index=False)

    _plot_equity(
        res_cost.equity_curve,
        "LONG LGBM v1 @0.52 — WITH costs (per-bar spread + slip 5)",
        out_dir / "equity_with_cost.png",
    )
    _plot_equity(
        res_flat.equity_curve,
        "LONG LGBM v1 @0.52 — NO costs",
        out_dir / "equity_no_cost.png",
    )

    report = build_report(m_cost, m_flat)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(report, encoding="utf-8")

    print(report)
    print(f"\nSaved report -> {report_path}")


if __name__ == "__main__":
    main()
