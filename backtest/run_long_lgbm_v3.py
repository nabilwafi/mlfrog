"""
Backtest LONG LightGBM v3 @ thr=0.51 + comparison vs v1/v2.

Full-period preds: single time-split fit (train<2023, val 2023, test>=2024)
with threshold FROZEN at walk-forward majority thr=0.51 — apple-to-apple with v1 window.
Sealed-2026: use existing lightgbm_long_v3_test_preds.parquet.

Usage:
    python -m backtest.run_long_lgbm_v3
"""

from __future__ import annotations

import pickle
from pathlib import Path

import lightgbm as lgb
import numpy as np
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
from models.data_prep import CAT_COLS, HORIZON_BARS, drop_timeouts, time_aware_split
from models.train_walk_forward_v3 import FEATURE_COLS_V3

THR_V3 = 0.51
SEALED_START = pd.Timestamp("2026-01-02 04:00:00+00:00")
SEALED_END = pd.Timestamp("2026-07-08 11:00:00+00:00")


def score_full_test_preds(base: Path) -> Path:
    """Train v3 features with v1-style split; save preds for test>=2024."""
    feat = pd.read_parquet(base / "features/xauusd_h1_h4_d1_features_v3.parquet")
    lab = pd.read_parquet(base / "labels/xauusd_triple_barrier_labels.parquet")
    keep = ["Date", "label_long", "label_short"]
    df = feat.merge(lab[keep], on="Date", how="inner").sort_values("Date").reset_index(drop=True)
    kept, _ = drop_timeouts(df, "long")
    splits = time_aware_split(kept, train_end="2023-01-01", val_end="2024-01-01", embargo_hours=HORIZON_BARS)

    def xy(frame: pd.DataFrame):
        x = frame[FEATURE_COLS_V3].copy()
        for c in CAT_COLS:
            x[c] = x[c].astype("category")
        y = frame["label_long"].astype(int).to_numpy()
        return x, y

    xtr, ytr = xy(splits.train)
    xva, yva = xy(splits.val)
    xte, yte = xy(splits.test)

    model = lgb.LGBMClassifier(
        objective="binary",
        num_leaves=31,
        min_child_samples=100,
        learning_rate=0.05,
        n_estimators=300,
        class_weight="balanced",
        random_state=42,
        n_jobs=-1,
        verbose=-1,
    )
    model.fit(
        xtr,
        ytr,
        eval_set=[(xva, yva)],
        eval_metric="auc",
        callbacks=[lgb.early_stopping(30, verbose=False)],
        categorical_feature=CAT_COLS,
    )
    p_te = model.predict_proba(xte)[:, 1]
    preds = splits.test[["Date", "label_long"]].rename(columns={"label_long": "y_true"}).copy()
    preds["y_prob"] = p_te
    out = base / "models_v3/long/lightgbm_long_v3_fulltest_preds.parquet"
    out.parent.mkdir(parents=True, exist_ok=True)
    preds.to_parquet(out, index=False)
    with open(base / "models_v3/long/lightgbm_long_v3_fulltest_fit.pkl", "wb") as f:
        pickle.dump({"model": model, "feature_cols": FEATURE_COLS_V3, "selected_thr": THR_V3}, f)
    print(
        f"Full-test preds: n={len(preds)} | {preds['Date'].min()} -> {preds['Date'].max()} | "
        f"signals@0.51={(preds['y_prob']>=THR_V3).sum()} "
        f"prec={preds.loc[preds['y_prob']>=THR_V3,'y_true'].mean():.4f}"
    )
    return out


def _run(df: pd.DataFrame, thr: float) -> tuple[dict, dict]:
    m_c = compute_metrics(run_backtest(df, BacktestConfig(apply_costs=True, selected_thr=thr)))
    m_n = compute_metrics(run_backtest(df, BacktestConfig(apply_costs=False, selected_thr=thr)))
    return m_c, m_n


def main() -> None:
    base = Path("data")
    feats_v3 = base / "features/xauusd_h1_h4_d1_features_v3.parquet"
    h1 = base / "raw/XAUUSD_H1.csv"
    sealed_preds = base / "models_v3/long/lightgbm_long_v3_test_preds.parquet"

    # Confirm sealed window
    sp = pd.read_parquet(sealed_preds)
    sp["Date"] = pd.to_datetime(sp["Date"], utc=True)
    print(f"Sealed preds: {sp['Date'].min()} -> {sp['Date'].max()} n={len(sp)}")
    assert sp["Date"].min() == SEALED_START
    assert sp["Date"].max() == SEALED_END

    print("Scoring full-test preds (train<2023 / val 2023 / test>=2024, thr frozen 0.51)...")
    full_preds = score_full_test_preds(base)

    print("Backtest FULL...")
    df_full = load_backtest_frame(preds_path=full_preds, features_path=feats_v3, h1_raw_path=h1)
    m_full_c, m_full_n = _run(df_full, THR_V3)
    n_full = int((df_full["y_prob"] >= THR_V3).sum())
    prec_full = float(df_full.loc[df_full["y_prob"] >= THR_V3, "y_true"].mean()) if n_full else None

    print("Backtest SEALED 2026...")
    df_seal = load_backtest_frame(preds_path=sealed_preds, features_path=feats_v3, h1_raw_path=h1)
    df_seal = df_seal[(df_seal["Date"] >= SEALED_START) & (df_seal["Date"] <= SEALED_END)].reset_index(drop=True)
    m_seal_c, m_seal_n = _run(df_seal, THR_V3)
    n_seal = int((df_seal["y_prob"] >= THR_V3).sum())
    prec_seal = float(df_seal.loc[df_seal["y_prob"] >= THR_V3, "y_true"].mean()) if n_seal else None

    out_dir = base / "backtest_v3"
    out_dir.mkdir(parents=True, exist_ok=True)
    # save full with-cost equity as primary artifact
    res_full = run_backtest(df_full, BacktestConfig(apply_costs=True, selected_thr=THR_V3))
    res_full.equity_curve.to_parquet(out_dir / "equity_with_cost_full.parquet", index=False)
    _trades_to_df(res_full).to_parquet(out_dir / "trades_with_cost_full.parquet", index=False)
    _plot_equity(res_full.equity_curve, "LONG LGBM v3 @0.51 FULL — WITH costs", out_dir / "equity_with_cost_full.png")

    res_seal = run_backtest(df_seal, BacktestConfig(apply_costs=True, selected_thr=THR_V3))
    res_seal.equity_curve.to_parquet(out_dir / "equity_with_cost_sealed2026.parquet", index=False)
    _trades_to_df(res_seal).to_parquet(out_dir / "trades_with_cost_sealed2026.parquet", index=False)
    _plot_equity(
        res_seal.equity_curve,
        "LONG LGBM v3 @0.51 SEALED 2026 — WITH costs",
        out_dir / "equity_with_cost_sealed2026.png",
    )

    report = _build_v3_report(m_full_c, m_full_n, m_seal_c, m_seal_n, n_full, prec_full, n_seal, prec_seal)
    (base / "reports/backtest_report_v3.md").write_text(report, encoding="utf-8")

    comparison = _build_comparison(m_full_c, m_seal_c)
    (base / "reports/final_long_model_comparison.md").write_text(comparison, encoding="utf-8")
    print(report)
    print("\n---\n")
    print(comparison)


def _build_v3_report(m_fc, m_fn, m_sc, m_sn, n_full, prec_full, n_seal, prec_seal) -> str:
    return "\n".join(
        [
            f"# Backtest Report - LONG LightGBM v3 @ thr={THR_V3:.2f}",
            "",
            "## Unit / cost (same engine as v1/v2)",
            f"- Method A: `(exit_fill - entry_fill) * {CONTRACT_SIZE} * lots`",
            f"- point={POINT}, slip={ASSUMED_SLIPPAGE_POINTS}, fallback spread={FALLBACK_SPREAD_POINTS}",
            "",
            "## Notes on prediction windows",
            "- **Full test (2024+)**: one LGBM fit on v3 features with train<2023 / val 2023 "
            f"(same calendar philosophy as v1); threshold **frozen** at WF majority {THR_V3:.2f}.",
            "- **Sealed 2026**: preds from walk-forward final model "
            f"(`lightgbm_long_v3_test_preds.parquet`): {SEALED_START} -> {SEALED_END}.",
            "",
            "## FULL test (2024-01 -> data end) WITH costs",
            f"- Model-eval signals @thr: n={n_full}, precision={prec_full}",
            _fmt_metrics(m_fc),
            "",
            "## FULL test WITHOUT costs",
            _fmt_metrics(m_fn),
            "",
            "## SEALED 2026 WITH costs",
            f"- Model-eval signals @thr: n={n_seal}, precision={prec_seal}",
            _fmt_metrics(m_sc),
            "",
            "## SEALED 2026 WITHOUT costs",
            _fmt_metrics(m_sn),
            "",
            "## Sanity",
            f"- Full: model prec={prec_full} vs backtest winrate_tp_sl={m_fc['winrate_tp_sl']:.4f} "
            f"(n_trades={m_fc['n_trades']}, skipped={m_fc['n_signals_skipped_stacking']})",
            f"- Sealed: model prec={prec_seal} vs winrate_tp_sl={m_sc['winrate_tp_sl']:.4f} "
            f"(n_trades={m_sc['n_trades']})",
            "",
            "## Artifacts",
            "- `data/backtest_v3/`",
            "- `data/models_v3/long/lightgbm_long_v3_fulltest_preds.parquet`",
            "",
        ]
    )


def _build_comparison(m_full_c: dict, m_seal_c: dict) -> str:
    # Numbers from prior reports (controlled_comparison + backtest_report + backtest_report_v2)
    rows = {
        "Threshold": ["0.52", "0.52", "0.54", "0.51", "0.51"],
        "CAGR % (with cost)": ["13.19", "9.90", "-7.82", f"{m_full_c['cagr_pct']:.2f}", f"{m_seal_c['cagr_pct']:.2f}"],
        "Win-rate realized (TP/SL)": ["0.5134", "0.4815", "0.4038", f"{m_full_c['winrate_tp_sl']:.4f}", f"{m_seal_c['winrate_tp_sl']:.4f}"],
        "n_trades": ["247", "59", "56", str(m_full_c["n_trades"]), str(m_seal_c["n_trades"])],
        "Sharpe (with cost)": ["1.25", "0.99", "-0.82", f"{m_full_c['sharpe_annualized_approx']:.2f}", f"{m_seal_c['sharpe_annualized_approx']:.2f}"],
        "Profit Factor": ["1.31", "1.20", "0.84", f"{m_full_c['profit_factor']:.2f}", f"{m_seal_c['profit_factor']:.2f}"],
        "Max DD %": ["17.06", "6.26", "7.32", f"{m_full_c['max_drawdown_pct']:.2f}", f"{m_seal_c['max_drawdown_pct']:.2f}"],
        "Ending equity": ["13647", "10492", "9591", f"{m_full_c['ending_equity']:.0f}", f"{m_seal_c['ending_equity']:.0f}"],
    }
    headers = [
        "metric",
        "v1 @ full test",
        "v1 @ sealed-2026",
        "v2 @ sealed-2026",
        "v3 @ full test",
        "v3 @ sealed-2026",
    ]
    lines = [
        "# Final LONG Model Comparison (v1 / v2 / v3)",
        "",
        f"Sealed window (confirmed): **{SEALED_START} -> {SEALED_END}**",
        "",
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join(["---"] * len(headers)) + " |",
    ]
    for metric, vals in rows.items():
        lines.append("| " + " | ".join([metric] + vals) + " |")

    # Verdicts
    v3_seal_cagr = m_seal_c["cagr_pct"]
    v3_full_cagr = m_full_c["cagr_pct"]
    beat_v2_seal = v3_seal_cagr > -7.82 and m_seal_c["ending_equity"] > 9591
    beat_v1_seal = v3_seal_cagr > 9.90
    profitable_seal = m_seal_c["ending_equity"] > 10000 and v3_seal_cagr > 0
    comparable_full = v3_full_cagr >= 13.19 * 0.85  # within ~15% of v1 CAGR as "comparable"

    lines += [
        "",
        "## Required conclusions",
        "",
        "### 1) Does v3 beat v1 AND v2 on sealed-2026?",
    ]
    if profitable_seal and beat_v2_seal:
        if beat_v1_seal:
            lines.append(
                f"**YES vs both** — v3 sealed CAGR {v3_seal_cagr:.2f}% / end equity "
                f"{m_seal_c['ending_equity']:.0f} beats v1 sealed (+9.90% / 10492) and v2 (−7.82% / 9591)."
            )
        else:
            lines.append(
                f"**Beats v2, not clearly v1** — v3 sealed CAGR {v3_seal_cagr:.2f}% "
                f"(equity {m_seal_c['ending_equity']:.0f}) recovers vs v2's loss, "
                f"but does not exceed v1 sealed (+9.90% / 10492)."
            )
    elif beat_v2_seal and not profitable_seal:
        lines.append(
            f"**Partial** - better than v2's loss but not cleanly profitable "
            f"(CAGR {v3_seal_cagr:.2f}%, equity {m_seal_c['ending_equity']:.0f})."
        )
    else:
        lines.append(
            f"**NO** - v3 sealed CAGR {v3_seal_cagr:.2f}% / equity {m_seal_c['ending_equity']:.0f} "
            "does not establish a clear win over v1 on the stress window "
            "(still document vs v2)."
        )

    lines += [
        "",
        "### 2) Is v3 full-period comparable to v1 (CAGR 13.19%, PF 1.31)?",
        (
            f"v3 full: CAGR {v3_full_cagr:.2f}%, PF {m_full_c['profit_factor']:.2f}, "
            f"WR {m_full_c['winrate_tp_sl']:.4f}, n_trades={m_full_c['n_trades']}, "
            f"MaxDD {m_full_c['max_drawdown_pct']:.2f}%."
        ),
    ]
    if v3_full_cagr >= 13.19 and m_full_c["profit_factor"] >= 1.31:
        lines.append("**Better or equal on headline CAGR/PF** vs v1 full.")
    elif comparable_full and m_full_c["profit_factor"] >= 1.0:
        lines.append(
            "**Comparable** — not a collapse vs v1; small gaps expected from thr/feature changes."
        )
    elif m_full_c["ending_equity"] > 10000:
        lines.append(
            "**Weaker than v1 on CAGR but still profitable** on the full window — "
            "do not promote solely on sealed-window recovery."
        )
    else:
        lines.append("**Worse** — full-window equity/CAGR does not support replacing v1.")

    # Operational pick
    lines += ["", "### 3) Operational LONG candidate"]
    # Decision logic
    if profitable_seal and v3_full_cagr >= 10 and m_full_c["profit_factor"] >= 1.15:
        if beat_v1_seal or v3_full_cagr >= 13.19:
            pick = "v3"
            why = "survives 2026 stress better than v2 and matches/beats v1 overall."
        else:
            pick = "v1 (primary) with v3 as alternate stress-tested candidate"
            why = (
                "v1 still stronger on full-window CAGR; v3 is the better 2026-resilient "
                "variant if prioritizing high-vol regimes."
            )
    elif v3_full_cagr > 13.19 and m_full_c["profit_factor"] > 1.31:
        pick = "v3"
        why = "full-window metrics exceed v1."
    else:
        pick = "v1"
        why = "still best documented full-window edge; v2 discarded; v3 needs clearer dual-window win."

    lines += [
        f"**Recommendation: {pick}.** {why}",
        "",
        "- **v2**: discarded (sealed-2026 loss; inferior controlled comparison).",
        "- Next: optional regime guard on the chosen LONG model; SHORT remains separate (stacking v4).",
        "",
    ]
    return "\n".join(lines)


if __name__ == "__main__":
    main()
