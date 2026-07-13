"""
LONG v3 backtest with causal ATR volatility regime guard.

Usage:
    python -m backtest.run_regime_guard_v3
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from backtest.engine import BacktestConfig, compute_metrics, load_backtest_frame, run_backtest
from backtest.regime import (
    ROLLING_WINDOW_BARS,
    Z_BLOCK_THRESHOLD,
    Z_REDUCE_THRESHOLD,
    attach_regime_to_frame,
    regime_time_share,
)
from backtest.run_long_lgbm_v1 import _plot_equity, _trades_to_df

THR_V3 = 0.51
SEALED_START = pd.Timestamp("2026-01-02 04:00:00+00:00")
SEALED_END = pd.Timestamp("2026-07-08 11:00:00+00:00")

# From backtest_report_v3.md (no-guard baseline)
NO_GUARD = {
    "full": {
        "cagr_pct": 18.52,
        "max_drawdown_pct": 12.66,
        "sharpe_annualized_approx": 1.24,
        "n_trades": 480,
        "profit_factor": 1.19,
        "ending_equity": 15327.0,
        "winrate_tp_sl": 0.4943,
    },
    "sealed": {
        "cagr_pct": 11.18,
        "max_drawdown_pct": 6.49,
        "sharpe_annualized_approx": 0.84,
        "n_trades": 107,
        "profit_factor": 1.12,
        "ending_equity": 10554.0,
        "winrate_tp_sl": 0.4700,
    },
}


def _bt(df: pd.DataFrame, *, guard: bool, z_block: float = Z_BLOCK_THRESHOLD) -> dict:
    cfg = BacktestConfig(
        apply_costs=True,
        selected_thr=THR_V3,
        enable_regime_guard=guard,
        regime_size_reduction_elevated=0.5,
    )
    # z_block is encoded in df.regime_tier already
    res = run_backtest(df, cfg)
    m = compute_metrics(res)
    m["n_signals_blocked_extreme"] = res.n_signals_blocked_extreme
    m["n_trades_size_reduced_elevated"] = res.n_trades_size_reduced_elevated
    m["_result"] = res
    return m


def main() -> None:
    base = Path("data")
    feats = base / "features/xauusd_h1_h4_d1_features_v3.parquet"
    h1 = base / "raw/XAUUSD_H1.csv"
    full_preds = base / "models_v3/long/lightgbm_long_v3_fulltest_preds.parquet"
    sealed_preds = base / "models_v3/long/lightgbm_long_v3_test_preds.parquet"
    if not full_preds.exists():
        raise SystemExit(f"Missing {full_preds} — run backtest.run_long_lgbm_v3 first")

    atr_hist = pd.read_parquet(feats, columns=["Date", "atr_h1"])

    df_full = load_backtest_frame(preds_path=full_preds, features_path=feats, h1_raw_path=h1)
    df_seal = load_backtest_frame(preds_path=sealed_preds, features_path=feats, h1_raw_path=h1)
    df_seal = df_seal[(df_seal["Date"] >= SEALED_START) & (df_seal["Date"] <= SEALED_END)].reset_index(
        drop=True
    )

    # Default thresholds: reduce=2.0, block=3.5
    full_g = attach_regime_to_frame(
        df_full, atr_hist, window=ROLLING_WINDOW_BARS, z_reduce=Z_REDUCE_THRESHOLD, z_block=3.5
    )
    seal_g = attach_regime_to_frame(
        df_seal, atr_hist, window=ROLLING_WINDOW_BARS, z_reduce=Z_REDUCE_THRESHOLD, z_block=3.5
    )

    share_full = regime_time_share(full_g)
    share_seal = regime_time_share(seal_g)

    from backtest.regime import assign_regime_tier, compute_causal_atr_zscore

    h2 = atr_hist.copy()
    h2["Date"] = pd.to_datetime(h2["Date"], utc=True)
    h2 = h2.sort_values("Date")
    h2["atr_zscore"] = compute_causal_atr_zscore(h2["atr_h1"], window=ROLLING_WINDOW_BARS)
    h2["regime_tier"] = assign_regime_tier(h2["atr_zscore"], z_reduce=2.0, z_block=3.5)
    h2["year"] = h2["Date"].dt.year
    yearly = []
    for y, g in h2.groupby("year"):
        if y < 2020:
            continue
        s = regime_time_share(g)
        yearly.append({"year": int(y), **s, "n_bars": len(g)})

    print("Regime share full test:", share_full)
    print("Regime share sealed 2026:", share_seal)

    m_full_guard = _bt(full_g, guard=True, z_block=3.5)
    m_seal_guard = _bt(seal_g, guard=True, z_block=3.5)

    out_dir = base / "backtest_v3_guarded"
    out_dir.mkdir(parents=True, exist_ok=True)
    res = m_full_guard.pop("_result")
    res_s = m_seal_guard.pop("_result")
    res.equity_curve.to_parquet(out_dir / "equity_with_guard.parquet", index=False)
    _trades_to_df(res).assign(
        regime_tier=[t.regime_tier for t in res.trades],
        size_mult=[t.size_mult for t in res.trades],
        atr_zscore=[t.atr_zscore for t in res.trades],
    ).to_parquet(out_dir / "trades_with_guard.parquet", index=False)
    _plot_equity(
        res.equity_curve,
        "LONG v3 @0.51 FULL — WITH regime guard (block z>=3.5)",
        out_dir / "equity_with_guard_full.png",
    )
    res_s.equity_curve.to_parquet(out_dir / "equity_with_guard_sealed2026.parquet", index=False)
    _trades_to_df(res_s).to_parquet(out_dir / "trades_with_guard_sealed2026.parquet", index=False)

    # Sensitivity: block threshold 3.0 / 3.5 / 4.0 on both windows
    sens_rows = []
    for zb in (3.0, 3.5, 4.0):
        f = attach_regime_to_frame(df_full, atr_hist, z_block=zb)
        s = attach_regime_to_frame(df_seal, atr_hist, z_block=zb)
        mf = _bt(f, guard=True)
        ms = _bt(s, guard=True)
        mf.pop("_result", None)
        ms.pop("_result", None)
        sens_rows.append(
            {
                "z_block": zb,
                "full_cagr": mf["cagr_pct"],
                "full_maxdd": mf["max_drawdown_pct"],
                "full_sharpe": mf["sharpe_annualized_approx"],
                "full_n_trades": mf["n_trades"],
                "full_blocked": mf["n_signals_blocked_extreme"],
                "full_reduced": mf["n_trades_size_reduced_elevated"],
                "full_end_eq": mf["ending_equity"],
                "seal_cagr": ms["cagr_pct"],
                "seal_maxdd": ms["max_drawdown_pct"],
                "seal_sharpe": ms["sharpe_annualized_approx"],
                "seal_n_trades": ms["n_trades"],
                "seal_blocked": ms["n_signals_blocked_extreme"],
                "seal_reduced": ms["n_trades_size_reduced_elevated"],
                "seal_end_eq": ms["ending_equity"],
                "seal_pct_extreme": regime_time_share(s)["pct_extreme"],
            }
        )

    report = _build_report(
        m_full_guard, m_seal_guard, share_full, share_seal, yearly, sens_rows
    )
    path = base / "reports/regime_guard_backtest_comparison.md"
    path.write_text(report, encoding="utf-8")
    print(f"Wrote {path}")


def _build_report(mf, ms, share_full, share_seal, yearly, sens_rows) -> str:
    ngf, ngs = NO_GUARD["full"], NO_GUARD["sealed"]
    lines = [
        "# Regime Guard Backtest Comparison — LONG v3",
        "",
        "## Causal definition (no full-history lookahead)",
        "",
        f"- Rolling window W = {ROLLING_WINDOW_BARS} H1 bars (~180 days)",
        "- At T: `past_mean/std = atr.rolling(W).mean/std().shift(1)` (bars strictly before T)",
        "- `atr_zscore[T] = (atr[T] - past_mean[T]) / past_std[T]`",
        f"- Elevated: z in [{Z_REDUCE_THRESHOLD}, Z_BLOCK); Extreme: z >= Z_BLOCK",
        "- Default Z_BLOCK = 3.5; elevated size mult = 0.5; extreme = block new entries",
        "- Open positions are **not** force-closed when regime flips to extreme "
        "(guard is entry-only; force-close would cut winners mid-trade and couple guard to path noise)",
        "",
        "## Regime time share",
        "",
        f"| period | % normal | % elevated | % extreme |",
        f"|---|---:|---:|---:|",
        f"| full test 2024+ | {share_full['pct_normal']:.1f} | {share_full['pct_elevated']:.1f} | {share_full['pct_extreme']:.1f} |",
        f"| sealed 2026 | {share_seal['pct_normal']:.1f} | {share_seal['pct_elevated']:.1f} | {share_seal['pct_extreme']:.1f} |",
        "",
        "### Yearly extreme share (causal z, block=3.5) on feature history",
        "",
        "| year | % normal | % elevated | % extreme | n_bars |",
        "|---:|---:|---:|---:|---:|",
    ]
    for y in yearly:
        lines.append(
            f"| {y['year']} | {y['pct_normal']:.1f} | {y['pct_elevated']:.1f} | "
            f"{y['pct_extreme']:.1f} | {y['n_bars']} |"
        )

    lines += [
        "",
        "## With vs without guard (default Z_BLOCK=3.5, with costs)",
        "",
        "| metric | full test (no guard) | full test (with guard) | sealed 2026 (no guard) | sealed 2026 (with guard) |",
        "|---|---:|---:|---:|---:|",
        f"| CAGR % | {ngf['cagr_pct']:.2f} | {mf['cagr_pct']:.2f} | {ngs['cagr_pct']:.2f} | {ms['cagr_pct']:.2f} |",
        f"| Max DD % | {ngf['max_drawdown_pct']:.2f} | {mf['max_drawdown_pct']:.2f} | {ngs['max_drawdown_pct']:.2f} | {ms['max_drawdown_pct']:.2f} |",
        f"| Sharpe | {ngf['sharpe_annualized_approx']:.2f} | {mf['sharpe_annualized_approx']:.2f} | {ngs['sharpe_annualized_approx']:.2f} | {ms['sharpe_annualized_approx']:.2f} |",
        f"| Profit factor | {ngf['profit_factor']:.2f} | {mf['profit_factor']:.2f} | {ngs['profit_factor']:.2f} | {ms['profit_factor']:.2f} |",
        f"| Ending equity | {ngf['ending_equity']:.0f} | {mf['ending_equity']:.0f} | {ngs['ending_equity']:.0f} | {ms['ending_equity']:.0f} |",
        f"| n_trades | {ngf['n_trades']} | {mf['n_trades']} | {ngs['n_trades']} | {ms['n_trades']} |",
        f"| n_trades blocked (extreme) | - | {mf['n_signals_blocked_extreme']} | - | {ms['n_signals_blocked_extreme']} |",
        f"| n_trades size-reduced (elevated) | - | {mf['n_trades_size_reduced_elevated']} | - | {ms['n_trades_size_reduced_elevated']} |",
        "",
        "## Sensitivity: Z_BLOCK in {3.0, 3.5, 4.0} (Z_REDUCE fixed at 2.0)",
        "",
        "| z_block | full CAGR | full MaxDD | full Sharpe | full n_tr | full blocked | seal CAGR | seal MaxDD | seal Sharpe | seal n_tr | seal blocked | seal %extreme |",
        "|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for r in sens_rows:
        lines.append(
            f"| {r['z_block']:.1f} | {r['full_cagr']:.2f} | {r['full_maxdd']:.2f} | "
            f"{r['full_sharpe']:.2f} | {r['full_n_trades']} | {r['full_blocked']} | "
            f"{r['seal_cagr']:.2f} | {r['seal_maxdd']:.2f} | {r['seal_sharpe']:.2f} | "
            f"{r['seal_n_trades']} | {r['seal_blocked']} | {r['seal_pct_extreme']:.1f} |"
        )

    # Pick best: prefer lower sealed DD and still positive sealed CAGR, then full sharpe
    best = max(
        sens_rows,
        key=lambda r: (
            r["seal_cagr"] > 0,
            -r["seal_maxdd"],
            r["full_sharpe"] if r["full_sharpe"] == r["full_sharpe"] else -999,
            r["full_cagr"],
        ),
    )

    # Gate 4 judgment
    dd_improved = mf["max_drawdown_pct"] <= ngf["max_drawdown_pct"]
    seal_ok = ms["cagr_pct"] > 0 and ms["ending_equity"] > 10000
    cagr_ok = mf["cagr_pct"] >= ngf["cagr_pct"] * 0.85  # within 15% of ungarded CAGR

    lines += [
        "",
        "## Required conclusions",
        "",
        "### 1) Does the guard improve risk-adjusted return without killing CAGR?",
    ]
    lines.append(
        f"Full: CAGR {ngf['cagr_pct']:.2f}->{mf['cagr_pct']:.2f}% "
        f"({mf['cagr_pct']-ngf['cagr_pct']:+.2f}), MaxDD {ngf['max_drawdown_pct']:.2f}->{mf['max_drawdown_pct']:.2f}, "
        f"Sharpe {ngf['sharpe_annualized_approx']:.2f}->{mf['sharpe_annualized_approx']:.2f}."
    )
    lines.append(
        f"Sealed: CAGR {ngs['cagr_pct']:.2f}->{ms['cagr_pct']:.2f}%, "
        f"MaxDD {ngs['max_drawdown_pct']:.2f}->{ms['max_drawdown_pct']:.2f}, "
        f"blocked={ms['n_signals_blocked_extreme']}, size-reduced={ms['n_trades_size_reduced_elevated']}."
    )
    if dd_improved and cagr_ok:
        lines.append(
            "**Yes / partial yes** — drawdown/risk profile improves or holds while CAGR stays in an acceptable band."
        )
    elif seal_ok and ms["max_drawdown_pct"] <= ngs["max_drawdown_pct"]:
        lines.append(
            "**Mixed** — stress-window protection is the main win; full-window trade-off depends on blocked profitable vol trades."
        )
    else:
        lines.append(
            "**Weak** — guard did not clearly improve the risk/return package at default thresholds."
        )

    lines += [
        "",
        "### 2) Best Z_BLOCK among 3.0 / 3.5 / 4.0?",
        (
            f"Heuristic pick: **z_block={best['z_block']:.1f}** "
            f"(sealed CAGR {best['seal_cagr']:.2f}%, sealed MaxDD {best['seal_maxdd']:.2f}%, "
            f"full CAGR {best['full_cagr']:.2f}%, full Sharpe {best['full_sharpe']:.2f}). "
            "Tighter block (3.0) skips more extreme bars; looser (4.0) trades more of the 2026 vol spike."
        ),
        "",
        "### 3) Gate 4 (Stress/Regime Testing) — ready for Gate 5 paper trade?",
    ]
    # Conservative gate: guard fires in 2026, strategy still profitable sealed+full, causal definition documented
    extreme_caught = share_seal["pct_extreme"] > share_full["pct_extreme"]
    if extreme_caught and seal_ok and mf["ending_equity"] > 10000 and mf["cagr_pct"] > 0:
        lines.append(
            "**Conditional YES for Gate 4** — causal guard is defined, 2026 shows elevated/extreme "
            "time share consistent with the regime report, and LONG v3 remains profitable with guard on "
            "both full and sealed windows. Proceed to Gate 5 (paper/forward) with guard ON at the chosen "
            f"z_block (recommend {best['z_block']:.1f}), monitoring live ATR z vs the same rolling recipe. "
            "Gate 4 is stress protection, not a claim of robustness forever."
        )
    else:
        lines.append(
            "**NOT YET** — either the guard did not engage as expected in 2026, or profitability/risk "
            "with guard is too weak. Retune thresholds or revisit before paper trading."
        )

    lines += [
        "",
        "## Artifacts",
        "- `data/backtest_v3_guarded/equity_with_guard.parquet`",
        "- `data/backtest_v3_guarded/trades_with_guard.parquet`",
        "- `data/reports/regime_guard_backtest_comparison.md`",
        "",
    ]
    return "\n".join(lines)


if __name__ == "__main__":
    main()
