"""
Run LONG LightGBM v2 backtest using the SAME engine as v1.

Does not overwrite v1 backtest artifacts — writes to data/backtest_v2/.

Usage:
    python -m backtest.run_long_lgbm_v2
"""

from __future__ import annotations

import pickle
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

from backtest.engine import (
    ASSUMED_SLIPPAGE_POINTS,
    CONTRACT_SIZE,
    FALLBACK_SPREAD_POINTS,
    POINT,
    BacktestConfig,
    compute_metrics,
    load_backtest_frame,
    run_backtest,
)
from backtest.run_long_lgbm_v1 import _fmt_metrics, _plot_equity, _trades_to_df


def build_report(m_cost: dict, m_nocost: dict, thr: float, n_eval: int, prec_eval: float | None) -> str:
    wr = m_cost["winrate_tp_sl"]
    lines = [
        f"# Backtest Report — LONG LightGBM v2 @ thr={thr:.2f}",
        "",
        "## Unit / cost assumptions (same engine as v1)",
        f"- Method A: `(exit_fill - entry_fill) * {CONTRACT_SIZE} * lots`",
        f"- point={POINT}, contract={CONTRACT_SIZE}",
        f"- Slippage {ASSUMED_SLIPPAGE_POINTS} pts; fallback spread {FALLBACK_SPREAD_POINTS} pts",
        "",
        "## Simulation rules (unchanged)",
        f"- Threshold: {thr:.2f} (majority walk-forward)",
        "- Barriers: SL=1.5*ATR, TP=2.0*ATR; horizon=8; SL-first on ambiguity",
        "- No stacking; 1% risk; stop if equity < 70% start",
        "- Sealed test window: preds from walk-forward v2 (typically 2026+)",
        "",
        "## Sanity vs model eval",
        f"- Model-eval thresholded signals: n={n_eval}, precision={prec_eval}",
        f"- Backtest (with cost): n_trades={m_cost['n_trades']}, winrate_tp_sl={wr:.4f}",
        f"- Signals raw={m_cost['n_signals_raw']}, skipped_stacking={m_cost['n_signals_skipped_stacking']}",
        "",
        "## WITH costs",
        _fmt_metrics(m_cost),
        "",
        "## WITHOUT costs",
        _fmt_metrics(m_nocost),
        "",
        "## Cost impact",
        (
            f"- End equity with/without: {m_cost['ending_equity']:.2f} / "
            f"{m_nocost['ending_equity']:.2f}"
        ),
        (
            f"- CAGR with/without: {m_cost['cagr_pct']:.2f}% / {m_nocost['cagr_pct']:.2f}%"
        ),
        (
            f"- MaxDD with/without: {m_cost['max_drawdown_pct']:.2f}% / "
            f"{m_nocost['max_drawdown_pct']:.2f}%"
        ),
        "",
        "## Artifacts",
        "- `data/backtest_v2/` (equity/trades parquet + png)",
        "- `data/reports/backtest_report_v2.md`",
        "",
    ]
    return "\n".join(lines)


def main() -> None:
    base = Path("data")
    model_path = base / "models_v2/long/lightgbm_long_v2.pkl"
    preds_path = base / "models_v2/long/lightgbm_long_v2_test_preds.parquet"
    features_path = base / "features/xauusd_h1_h4_d1_features_v2.parquet"
    h1_path = base / "raw/XAUUSD_H1.csv"
    out_dir = base / "backtest_v2"
    report_path = base / "reports/backtest_report_v2.md"

    with open(model_path, "rb") as f:
        bundle = pickle.load(f)
    thr = bundle.get("selected_thr")
    if thr is None:
        raise SystemExit("LONG v2 has no valid majority threshold — skip backtest")

    print("Loading backtest frame (v2)...")
    df = load_backtest_frame(
        preds_path=preds_path,
        features_path=features_path,
        h1_raw_path=h1_path,
    )
    n_eval = int((df["y_prob"] >= thr).sum())
    prec_eval = None
    if "y_true" in df.columns and n_eval > 0:
        prec_eval = float(df.loc[df["y_prob"] >= thr, "y_true"].mean())
    print(
        f"rows={len(df)} | {df['Date'].min()} -> {df['Date'].max()} | "
        f"thr={thr} signals={n_eval}"
    )

    cfg_cost = BacktestConfig(apply_costs=True, selected_thr=float(thr))
    cfg_flat = BacktestConfig(apply_costs=False, selected_thr=float(thr))

    print("Running WITH costs...")
    res_cost = run_backtest(df, cfg_cost)
    print("Running WITHOUT costs...")
    res_flat = run_backtest(df, cfg_flat)

    m_cost = compute_metrics(res_cost)
    m_flat = compute_metrics(res_flat)

    out_dir.mkdir(parents=True, exist_ok=True)
    res_cost.equity_curve.to_parquet(out_dir / "equity_with_cost.parquet", index=False)
    res_flat.equity_curve.to_parquet(out_dir / "equity_no_cost.parquet", index=False)
    _trades_to_df(res_cost).to_parquet(out_dir / "trades_with_cost.parquet", index=False)
    _trades_to_df(res_flat).to_parquet(out_dir / "trades_no_cost.parquet", index=False)

    _plot_equity(
        res_cost.equity_curve,
        f"LONG LGBM v2 @{thr:.2f} — WITH costs",
        out_dir / "equity_with_cost.png",
    )
    _plot_equity(
        res_flat.equity_curve,
        f"LONG LGBM v2 @{thr:.2f} — NO costs",
        out_dir / "equity_no_cost.png",
    )

    report = build_report(m_cost, m_flat, float(thr), n_eval, prec_eval)
    report_path.write_text(report, encoding="utf-8")
    print(report)
    print(f"\nSaved report -> {report_path}")


if __name__ == "__main__":
    main()
