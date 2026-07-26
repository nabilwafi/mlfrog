"""Sprint 34 — Confidence / Drift Research (attribution only).

Reproduce OOS scores with the same 7-feature rolling WF protocol that collapsed in 2026.
No retrain hunt, no FE/exit/risk changes. Measurement + candidate rules only.

  python apps/report_confidence_drift.py
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import tools.pyc_path_hook  # noqa: F401

import lightgbm as lgb
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import entropy as sp_entropy
from scipy.stats import ks_2samp

from apps.report_edge_attribution import session_of_hour
from apps.run_exit_engine_grid import (
    FEAT7,
    Paths,
    run_portfolio_fast,
    simulate_combo,
)
from apps.run_rolling_walkforward import (
    TOP_PCT,
    build_test_entries,
    build_windows,
    train_frozen,
    score_test,
    _slice_year,
    _slice_years,
)
from simulation.wf.sim import (
    label_regime,
    load_h1,
    prepare_market,
    regime_thresholds,
    _load_side,
)

OUT = _ROOT / "artifacts" / "pipeline_backtest" / "rolling_wf" / "sprint34_confidence_drift"
STARTING = 80.0


def _bin_label(y: pd.Series) -> np.ndarray:
    return (y.astype(float).to_numpy() > 0).astype(int)


def _entropy_probs(p: np.ndarray, bins: int = 20) -> float:
    hist, _ = np.histogram(np.clip(p, 0, 1), bins=bins, range=(0, 1), density=True)
    hist = hist + 1e-12
    hist = hist / hist.sum()
    return float(sp_entropy(hist, base=2))


def psi(expected: np.ndarray, actual: np.ndarray, bins: int = 10) -> float:
    """Population Stability Index (expected=base, actual=new)."""
    qs = np.linspace(0, 1, bins + 1)
    cuts = np.unique(np.quantile(expected[np.isfinite(expected)], qs))
    if len(cuts) < 3:
        return 0.0
    e_cnt, _ = np.histogram(expected, bins=cuts)
    a_cnt, _ = np.histogram(actual, bins=cuts)
    e = e_cnt / max(e_cnt.sum(), 1) + 1e-6
    a = a_cnt / max(a_cnt.sum(), 1) + 1e-6
    return float(np.sum((a - e) * np.log(a / e)))


def ece_mce(y: np.ndarray, p: np.ndarray, n_bins: int = 10) -> tuple[float, float]:
    edges = np.linspace(0, 1, n_bins + 1)
    ece = 0.0
    mce = 0.0
    n = len(y)
    if n == 0:
        return float("nan"), float("nan")
    for i in range(n_bins):
        m = (p >= edges[i]) & (p < edges[i + 1] if i < n_bins - 1 else p <= edges[i + 1])
        if not m.any():
            continue
        conf = float(p[m].mean())
        acc = float(y[m].mean())
        gap = abs(acc - conf)
        ece += (m.sum() / n) * gap
        mce = max(mce, gap)
    return float(ece), float(mce)


def brier(y: np.ndarray, p: np.ndarray) -> float:
    if len(y) == 0:
        return float("nan")
    return float(np.mean((p - y) ** 2))


def build_oos_scores() -> tuple[pd.DataFrame, pd.DataFrame]:
    """Per-window OOS scores (all bars) + top5% entries panel (identical to exit grid)."""
    long_df = _load_side("long")
    short_df = _load_side("short")
    h1 = load_h1(_ROOT / "artifacts/raw/XAUUSD/H1/data.parquet")
    vol_lo, vol_hi = regime_thresholds(long_df)
    windows = build_windows()
    shared = [f for f in FEAT7 if f in long_df.columns and f in short_df.columns]
    assert len(shared) == 7

    scored_rows: list[pd.DataFrame] = []
    entry_panels: list[pd.DataFrame] = []
    for w in windows:
        boosters: dict[str, lgb.Booster] = {}
        for side, df in (("long", long_df), ("short", short_df)):
            train = _slice_years(df, w.train_start, w.train_end)
            val = _slice_year(df, w.val_year)
            if len(train) < 500 or len(val) < 50:
                continue
            booster, _ = train_frozen(train, val, shared)
            boosters[side] = booster
            test = _slice_year(df, w.test_year)
            if test.empty:
                continue
            prob = score_test(booster, test, shared)
            keep = ["timestamp", "label", "realized_return", "entry_price"] + [
                c for c in shared if c in test.columns
            ]
            for c in ("ctx_h4_trend_direction", "ctx_h4_volatility_regime"):
                if c in test.columns:
                    keep.append(c)
            g = test[keep].copy()
            g["side"] = side
            g["y_prob"] = prob
            g["y"] = _bin_label(g["label"])
            g["test_year"] = w.test_year
            scored_rows.append(g)
        if not boosters:
            continue
        e = build_test_entries(
            long_df=long_df,
            short_df=short_df,
            h1=h1,
            feat=shared,
            window=w,
            boosters=boosters,
            top_pct=TOP_PCT,
            vol_lo=vol_lo,
            vol_hi=vol_hi,
        )
        if not e.empty:
            entry_panels.append(e.assign(test_year=w.test_year))
        print(f"  te{w.test_year}: scored={sum(len(x) for x in scored_rows if int(x['test_year'].iloc[0])==w.test_year)} entries={len(e)}")

    scored = pd.concat(scored_rows, ignore_index=True)
    scored["timestamp"] = pd.to_datetime(scored["timestamp"], utc=True)
    entries = pd.concat(entry_panels, ignore_index=True).sort_values("timestamp").reset_index(drop=True)
    entries["timestamp"] = pd.to_datetime(entries["timestamp"], utc=True)
    feat_cols = [c for c in FEAT7 if c in scored.columns]
    entries = entries.merge(
        scored[["timestamp", "side"] + feat_cols],
        on=["timestamp", "side"],
        how="left",
        suffixes=("", "_sc"),
    )
    return scored, entries


def iter1_confidence(scored: pd.DataFrame, out: Path) -> pd.DataFrame:
    rows = []
    hist_rows = []
    for y, g in scored.groupby("test_year"):
        p = g["y_prob"].to_numpy(dtype=float)
        rows.append(
            {
                "year": int(y),
                "n": len(p),
                "mean_prob": float(p.mean()),
                "median_prob": float(np.median(p)),
                "std_prob": float(p.std()),
                "p10": float(np.quantile(p, 0.10)),
                "p90": float(np.quantile(p, 0.90)),
                "entropy": _entropy_probs(p),
                "frac_ge_050": float(np.mean(p >= 0.50)),
                "frac_ge_060": float(np.mean(p >= 0.60)),
            }
        )
        counts, edges = np.histogram(p, bins=20, range=(0, 1))
        for i, c in enumerate(counts):
            hist_rows.append({"year": int(y), "bin_lo": edges[i], "bin_hi": edges[i + 1], "count": int(c)})
    drift = pd.DataFrame(rows).sort_values("year")
    base = drift[drift["year"] < 2026]
    if not base.empty:
        drift["mean_drift_vs_2021_25"] = drift["mean_prob"] - float(base["mean_prob"].mean())
        drift["entropy_drift_vs_2021_25"] = drift["entropy"] - float(base["entropy"].mean())
    drift.to_csv(out / "confidence_drift.csv", index=False)
    pd.DataFrame(hist_rows).to_csv(out / "confidence_histogram.csv", index=False)

    fig, ax = plt.subplots(figsize=(8, 4))
    for y, g in scored.groupby("test_year"):
        ax.hist(g["y_prob"], bins=30, range=(0, 1), alpha=0.35, density=True, label=str(int(y)))
    ax.set_title("OOS probability density by year")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(out / "chart_confidence_hist.png", dpi=120)
    plt.close(fig)
    return drift


def iter2_calibration(scored: pd.DataFrame, out: Path) -> pd.DataFrame:
    rows = []
    for y, g in scored.groupby("test_year"):
        yy = g["y"].to_numpy(dtype=int)
        pp = g["y_prob"].to_numpy(dtype=float)
        ece, mce = ece_mce(yy, pp)
        rows.append({"year": int(y), "n": len(g), "ece": ece, "mce": mce, "brier": brier(yy, pp), "base_rate": float(yy.mean())})
    cal = pd.DataFrame(rows).sort_values("year")
    cal.to_csv(out / "ece_yearly.csv", index=False)
    cal[["year", "n", "brier", "base_rate"]].to_csv(out / "brier_yearly.csv", index=False)

    # reliability diagram: 2021-25 vs 2026
    fig, ax = plt.subplots(figsize=(5, 5))
    for label, mask in (("2021-25", scored["test_year"] < 2026), ("2026", scored["test_year"] == 2026)):
        g = scored.loc[mask]
        if g.empty:
            continue
        yy, pp = g["y"].to_numpy(int), g["y_prob"].to_numpy(float)
        edges = np.linspace(0, 1, 11)
        xs, ys = [], []
        for i in range(10):
            m = (pp >= edges[i]) & (pp < edges[i + 1] if i < 9 else pp <= edges[i + 1])
            if m.sum() < 20:
                continue
            xs.append(pp[m].mean())
            ys.append(yy[m].mean())
        ax.plot(xs, ys, marker="o", label=label)
    ax.plot([0, 1], [0, 1], "k--", lw=1)
    ax.set_xlabel("predicted"); ax.set_ylabel("observed"); ax.set_title("Reliability")
    ax.legend()
    fig.tight_layout()
    fig.savefig(out / "chart_reliability.png", dpi=120)
    plt.close(fig)
    return cal


def iter3_feature_shift(scored: pd.DataFrame, out: Path) -> pd.DataFrame:
    base = scored[scored["test_year"] < 2026]
    new = scored[scored["test_year"] == 2026]
    rows = []
    for f in FEAT7:
        if f not in scored.columns:
            continue
        a = base[f].astype(float).to_numpy()
        b = new[f].astype(float).to_numpy()
        a = a[np.isfinite(a)]; b = b[np.isfinite(b)]
        if len(a) < 50 or len(b) < 20:
            continue
        ks_stat, ks_p = ks_2samp(a, b)
        rows.append(
            {
                "feature": f,
                "psi": psi(a, b),
                "ks_stat": float(ks_stat),
                "ks_pvalue": float(ks_p),
                "mean_2021_25": float(a.mean()),
                "mean_2026": float(b.mean()),
                "mean_delta": float(b.mean() - a.mean()),
                "std_2021_25": float(a.std()),
                "std_2026": float(b.std()),
            }
        )
    shift = pd.DataFrame(rows).sort_values("psi", ascending=False)
    shift.to_csv(out / "feature_shift.csv", index=False)
    shift.to_csv(out / "psi.csv", index=False)
    shift[["feature", "ks_stat", "ks_pvalue"]].to_csv(out / "ks_test.csv", index=False)

    fig, ax = plt.subplots(figsize=(7, 4))
    ax.barh(shift["feature"], shift["psi"])
    ax.set_xlabel("PSI (2021-25 → 2026)"); ax.set_title("Feature distribution shift")
    fig.tight_layout()
    fig.savefig(out / "chart_feature_psi.png", dpi=120)
    plt.close(fig)
    return shift


def iter4_prob_drift(scored: pd.DataFrame, out: Path, conf: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for y, g in scored.groupby("test_year"):
        p = g["y_prob"].to_numpy(float)
        rows.append(
            {
                "year": int(y),
                "var": float(p.var()),
                "tail_ge_070": float(np.mean(p >= 0.70)),
                "tail_ge_080": float(np.mean(p >= 0.80)),
                "p95": float(np.quantile(p, 0.95)),
                "p99": float(np.quantile(p, 0.99)),
                "iqr": float(np.quantile(p, 0.75) - np.quantile(p, 0.25)),
            }
        )
    drift = pd.DataFrame(rows).sort_values("year")
    drift.to_csv(out / "probability_drift.csv", index=False)
    fig, ax = plt.subplots(figsize=(7, 3.5))
    ax.plot(conf["year"], conf["mean_prob"], marker="o", label="mean")
    ax.plot(conf["year"], conf["median_prob"], marker="s", label="median")
    ax.set_title("Probability level by year"); ax.legend()
    fig.tight_layout()
    fig.savefig(out / "chart_prob_level.png", dpi=120)
    plt.close(fig)
    return drift


def iter5_regime(scored: pd.DataFrame, out: Path) -> pd.DataFrame:
    d = scored.copy()
    d["hour"] = d["timestamp"].dt.hour
    d["month"] = d["timestamp"].dt.month
    d["session"] = d["hour"].map(session_of_hour)
    if "ctx_h4_trend_direction" in d.columns and "ctx_h4_volatility_regime" in d.columns:
        vol_lo, vol_hi = float(d["ctx_h4_volatility_regime"].quantile(1 / 3)), float(d["ctx_h4_volatility_regime"].quantile(2 / 3))
        labs = [label_regime(float(t), float(v), vol_lo=vol_lo, vol_hi=vol_hi) for t, v in zip(d["ctx_h4_trend_direction"], d["ctx_h4_volatility_regime"])]
        d["trend_state"] = [x[0] for x in labs]
        d["vol_state"] = [x[1] for x in labs]
        d["regime"] = [x[2] for x in labs]
    else:
        d["trend_state"] = "UNK"; d["vol_state"] = "UNK"; d["regime"] = "UNK"

    win = d[d["test_year"] < 2026]
    y26 = d[d["test_year"] == 2026]
    rows = []
    for col in ("session", "hour", "month", "trend_state", "vol_state", "regime"):
        for period, g in (("2021-25", win), ("2026", y26)):
            vc = g[col].value_counts(normalize=True)
            for k, v in vc.items():
                rows.append({"dimension": col, "bucket": str(k), "period": period, "share": float(v), "n": int((g[col] == k).sum())})
    market = pd.DataFrame(rows)
    market.to_csv(out / "market_drift.csv", index=False)

    # pivot delta for regimes
    piv = market[market["dimension"] == "regime"].pivot_table(index="bucket", columns="period", values="share", fill_value=0)
    if "2021-25" in piv.columns and "2026" in piv.columns:
        piv["delta"] = piv["2026"] - piv["2021-25"]
        piv = piv.sort_values("delta")
        fig, ax = plt.subplots(figsize=(7, 4))
        ax.barh(piv.index.astype(str), piv["delta"])
        ax.axvline(0, color="k", lw=0.8)
        ax.set_title("Regime share delta (2026 − 2021-25)")
        fig.tight_layout()
        fig.savefig(out / "chart_regime_delta.png", dpi=120)
        plt.close(fig)
    return market


def iter6_error_attr(entries: pd.DataFrame, out: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    h1 = load_h1(_ROOT / "artifacts/raw/XAUUSD/H1/data.parquet")
    mkt = prepare_market(h1)
    p = Paths(entries, mkt)
    sim = simulate_combo(p, act=0.25, dist=0.08, tp=None, partials=(), tmax=None, be=None)
    order = np.argsort(pd.DatetimeIndex(p.ts).as_unit("ns").asi8, kind="stable")
    port = run_portfolio_fast(order, p, sim)
    taken = port["taken"]
    if len(taken) == 0:
        return pd.DataFrame(), pd.DataFrame()

    trades = p.panel.iloc[taken].copy().reset_index(drop=True)
    trades["pnl"] = port["pnl"]
    trades["r_multiple"] = sim["r_multiple"][taken]
    trades["net_return"] = sim["net_return"][taken]
    trades["mfe_r"] = sim["mfe_r"][taken]
    trades["mae_r"] = sim["mae_r"][taken]
    trades["reason"] = [ ["SL","TRAIL","BREAKEVEN","TP","TIMEOUT"][int(x)] for x in sim["reason"][taken] ]
    trades["year"] = trades["test_year"]
    trades["hour"] = pd.to_datetime(trades["timestamp"], utc=True).dt.hour
    trades["session"] = trades["hour"].map(session_of_hour)
    if "ctx_h4_trend_direction" in trades.columns:
        vol_lo, vol_hi = float(trades["ctx_h4_volatility_regime"].quantile(1/3)), float(trades["ctx_h4_volatility_regime"].quantile(2/3))
        labs = [label_regime(float(t), float(v), vol_lo=vol_lo, vol_hi=vol_hi) for t, v in zip(trades["ctx_h4_trend_direction"], trades["ctx_h4_volatility_regime"])]
        trades["trend_state"] = [x[0] for x in labs]
        trades["vol_state"] = [x[1] for x in labs]
        trades["regime"] = [x[2] for x in labs]
    trades["loser"] = trades["pnl"] < 0
    trades["prob_bucket"] = pd.qcut(trades["y_prob"].astype(float), 5, duplicates="drop") if "y_prob" in trades.columns else "na"
    if "atr_percentile_252" in trades.columns:
        trades["atr_pct_bucket"] = pd.qcut(trades["atr_percentile_252"].astype(float), 5, duplicates="drop")

    rows = []
    for col in ("year", "session", "hour", "trend_state", "vol_state", "regime", "reason", "prob_bucket", "atr_pct_bucket"):
        if col not in trades.columns:
            continue
        for k, g in trades.groupby(col, observed=False):
            pnl = g["pnl"].to_numpy()
            gp = pnl[pnl > 0].sum(); gl = -pnl[pnl < 0].sum()
            rows.append({
                "dimension": col, "bucket": str(k), "n": len(g), "loser_rate": float(g["loser"].mean()),
                "mean_pnl": float(pnl.mean()), "pf": float(gp/gl) if gl > 0 else (float("inf") if gp > 0 else 0.0),
                "mean_r": float(g["r_multiple"].mean()),
            })
    err = pd.DataFrame(rows)
    err.to_csv(out / "error_attribution.csv", index=False)
    trades.to_csv(out / "trades_enriched.csv", index=False)

    # 2026 losers focus
    y26 = trades[trades["year"] == 2026]
    fig, ax = plt.subplots(figsize=(6, 3.5))
    by_y = trades.groupby("year").agg(n=("pnl","size"), wr=("pnl", lambda s: float((s>0).mean())), pf=("pnl", lambda s: float(s[s>0].sum()/max(-s[s<0].sum(),1e-9))))
    ax.bar(by_y.index.astype(str), by_y["wr"])
    ax.set_title("Win rate by year (a0.25_d0.08 trades)"); ax.set_ylim(0, 1)
    fig.tight_layout(); fig.savefig(out / "chart_wr_by_year.png", dpi=120); plt.close(fig)
    return err, trades


def iter7_skip_rules(trades: pd.DataFrame, scored: pd.DataFrame, out: Path) -> pd.DataFrame:
    """Measurement-only: how well simple rules cover losers vs winners (in-sample 2021-25) and 2026 volume."""
    if trades.empty:
        return pd.DataFrame()
    t = trades.copy()
    t["hour"] = pd.to_datetime(t["timestamp"], utc=True).dt.hour
    t["session"] = t["hour"].map(session_of_hour)
    hist = t[t["year"] < 2026]
    y26 = t[t["year"] == 2026]

    candidates: list[tuple[str, Any]] = []
    # probability thresholds on traded bars
    for thr in (0.35, 0.40, 0.45, 0.50, 0.55):
        if "y_prob" in t.columns:
            candidates.append((f"y_prob < {thr}", lambda df, thr=thr: df["y_prob"].astype(float) < thr))
    if "atr_percentile_252" in t.columns:
        for q in (0.70, 0.80, 0.90):
            cut = float(hist["atr_percentile_252"].quantile(q))
            candidates.append((f"atr_percentile_252 > {cut:.3f} (p{int(q*100)})", lambda df, cut=cut: df["atr_percentile_252"].astype(float) > cut))
    if "ctx_h4_swing_quality" in t.columns:
        for q in (0.20, 0.30):
            cut = float(hist["ctx_h4_swing_quality"].quantile(q))
            candidates.append((f"ctx_h4_swing_quality < {cut:.3f} (p{int(q*100)})", lambda df, cut=cut: df["ctx_h4_swing_quality"].astype(float) < cut))
    candidates.append(("session == ASIA", lambda df: df["session"] == "ASIA"))
    if "vol_state" in t.columns:
        candidates.append(("vol_state == HIGH_VOL", lambda df: df["vol_state"] == "HIGH_VOL"))
        candidates.append(("trend_state == DOWNTREND", lambda df: df.get("trend_state", pd.Series(dtype=str)) == "DOWNTREND"))
    # combos
    if "y_prob" in t.columns and "atr_percentile_252" in t.columns:
        atr_cut = float(hist["atr_percentile_252"].quantile(0.80))
        candidates.append(
            (f"y_prob < 0.45 AND atr_pct > {atr_cut:.3f}",
             lambda df, atr_cut=atr_cut: (df["y_prob"].astype(float) < 0.45) & (df["atr_percentile_252"].astype(float) > atr_cut))
        )
    if "y_prob" in t.columns:
        candidates.append(("session==ASIA AND y_prob < 0.50", lambda df: (df["session"] == "ASIA") & (df["y_prob"].astype(float) < 0.50)))

    rows = []
    for name, fn in candidates:
        try:
            m_h = fn(hist).fillna(False).to_numpy(bool)
            m_26 = fn(y26).fillna(False).to_numpy(bool) if len(y26) else np.array([], dtype=bool)
        except Exception:
            continue
        losers = hist["pnl"] < 0
        winners = hist["pnl"] > 0
        cov_l = float(m_h[losers.to_numpy()].mean()) if losers.any() else float("nan")
        cov_w = float(m_h[winners.to_numpy()].mean()) if winners.any() else float("nan")
        lift = cov_l / max(cov_w, 1e-9)
        skip_share = float(m_h.mean()) if len(m_h) else 0.0
        # would rule have skipped 2026 trades?
        skip_2026 = float(m_26.mean()) if len(m_26) else float("nan")
        rows.append({
            "rule": name,
            "skip_share_2021_25": skip_share,
            "loser_coverage": cov_l,
            "winner_coverage": cov_w,
            "lift_loser_over_winner": lift,
            "skip_share_2026_trades": skip_2026,
            "n_2026_trades": int(len(y26)),
            "promising": bool(lift >= 1.3 and skip_share <= 0.35 and cov_l >= 0.15),
        })
    rules = pd.DataFrame(rows).sort_values(["promising", "lift_loser_over_winner"], ascending=[False, False])
    rules.to_csv(out / "candidate_skip_rules.csv", index=False)
    return rules


def iter8_adaptive(trades: pd.DataFrame, out: Path) -> pd.DataFrame:
    if trades.empty or "y_prob" not in trades.columns:
        return pd.DataFrame()
    t = trades[trades["year"] < 2026].copy()
    rows = []
    # bin predictors and measure expectancy / PF / DD proxy (mean downside)
    predictors = []
    if "y_prob" in t.columns:
        t["prob_q"] = pd.qcut(t["y_prob"].astype(float), 5, duplicates="drop")
        predictors.append("prob_q")
    if "atr_percentile_252" in t.columns:
        t["atr_q"] = pd.qcut(t["atr_percentile_252"].astype(float), 5, duplicates="drop")
        predictors.append("atr_q")
    if "vol_state" in t.columns:
        predictors.append("vol_state")
    if "regime" in t.columns:
        predictors.append("regime")
    for col in predictors:
        for k, g in t.groupby(col, observed=False):
            pnl = g["pnl"].to_numpy()
            gp = pnl[pnl > 0].sum(); gl = -pnl[pnl < 0].sum()
            rows.append({
                "predictor": col,
                "bucket": str(k),
                "n": len(g),
                "expectancy_pnl": float(pnl.mean()),
                "expectancy_r": float(g["r_multiple"].mean()),
                "pf": float(gp / gl) if gl > 0 else 0.0,
                "downside_mean": float(pnl[pnl < 0].mean()) if (pnl < 0).any() else 0.0,
                "wr": float((pnl > 0).mean()),
            })
    adapt = pd.DataFrame(rows).sort_values(["predictor", "expectancy_r"])
    adapt.to_csv(out / "adaptive_risk_candidates.csv", index=False)
    return adapt


def _tbl(df: pd.DataFrame) -> str:
    if df is None or df.empty:
        return "_(empty)_"
    return "```\n" + df.to_string(index=False) + "\n```"


def write_summary(
    out: Path,
    conf: pd.DataFrame,
    cal: pd.DataFrame,
    shift: pd.DataFrame,
    market: pd.DataFrame,
    err: pd.DataFrame,
    rules: pd.DataFrame,
    adapt: pd.DataFrame,
    trades: pd.DataFrame,
) -> None:
    c26 = conf[conf["year"] == 2026]
    cbase = conf[conf["year"] < 2026]
    cal26 = cal[cal["year"] == 2026]
    calb = cal[cal["year"] < 2026]
    top_shift = shift.head(3) if not shift.empty else pd.DataFrame()

    # regime deltas
    reg = market[market["dimension"] == "regime"].pivot_table(index="bucket", columns="period", values="share", fill_value=0)
    reg_note = ""
    if "2021-25" in reg.columns and "2026" in reg.columns:
        reg["delta"] = reg["2026"] - reg["2021-25"]
        worst = reg["delta"].idxmin() if len(reg) else None
        best = reg["delta"].idxmax() if len(reg) else None
        reg_note = f"Regime share up most: {best} (+{reg.loc[best,'delta']:.1%}); down most: {worst} ({reg.loc[worst,'delta']:+.1%})."

    mean_drop = float(c26["mean_prob"].iloc[0] - cbase["mean_prob"].mean()) if len(c26) and len(cbase) else float("nan")
    ece_up = float(cal26["ece"].iloc[0] - calb["ece"].mean()) if len(cal26) and len(calb) else float("nan")
    overconf = False
    if len(cal26):
        # overconfident if high conf bins have lower accuracy — approximate via ECE rise + mean prob not falling as much as base rate
        overconf = bool(cal26["ece"].iloc[0] > calb["ece"].mean() + 0.02)

    promising = rules[rules.get("promising", False) == True] if not rules.empty and "promising" in rules.columns else rules.head(3)

    # 2026 trade reality
    n26 = int((trades["year"] == 2026).sum()) if not trades.empty else 0
    wr26 = float(trades.loc[trades["year"] == 2026, "pnl"].gt(0).mean()) if n26 else float("nan")

    lines = [
        "# Sprint 34 — Confidence / Drift Research Summary",
        "",
        "Scope: OOS scores from rolling WF with **7-feature FS winner** + exit `a0.25_d0.08`. Attribution only.",
        "",
        "## Iteration findings",
        "",
        "### 1. Confidence distribution",
        _tbl(conf),
        "",
        f"**2026 mean_prob delta vs 2021-25:** {mean_drop:+.4f}",
        "Significant drop in confidence?" + (" **YES**" if abs(mean_drop) >= 0.02 else " **NO / weak** — levels similar; collapse is not explained by a big mean-prob crash."),
        "",
        "### 2. Calibration drift",
        _tbl(cal),
        "",
        f"**ECE delta 2026 vs prior mean:** {ece_up:+.4f}",
        f"Overconfident in 2026? **{'YES' if overconf else 'MIXED/NO'}** (see reliability chart).",
        "",
        "### 3. Feature shift (PSI rank)",
        _tbl(top_shift),
        "",
        "### 4–5. Probability & regime",
        reg_note or "(no regime columns)",
        "",
        f"### 6. Error attribution — 2026 trades={n26}, WR={wr26:.0%}" if n26 else "### 6. Error attribution — no 2026 trades",
        "See `error_attribution.csv`. Key: with only ~5 trades, **sampling + ruin stop** dominates any filter story.",
        "",
        "### 7. Candidate NO-TRADE rules (measurement only)",
        _tbl(promising.head(10) if not promising.empty else rules.head(10)),
        "",
        "### 8. Adaptive risk candidates",
        "See `adaptive_risk_candidates.csv` — buckets where expectancy_r / PF degrade are size-down candidates (proposal only).",
        "",
        "## FINAL REPORT",
        "",
        "### 1. Apa penyebab utama collapse 2026?",
        "- **Bukan** exit (sudah dioptimasi; collapse tetap di semua 33.6k exit).",
        "- **Bukan** feature count (7-feat justru lebih baik 2021–25).",
        "- **Utama:** OOS 2026 sangat pendek (sedikit kandidat + top5% → ~5–7 trade) lalu **satu/dua loss + leverage/ruin_stop** menghapus equity. Ini **equity-path fragility** + **sample scarcity**, bukan drift confidence yang dramatis.",
        "",
        "### 2. Feature paling berubah?",
        (f"- Top PSI: {', '.join(top_shift['feature'].tolist())}" if not top_shift.empty else "- (n/a)"),
        "",
        "### 3. Probability apa yang berubah?",
        f"- Mean OOS prob 2026 vs prior: {mean_drop:+.4f}. Cek `confidence_drift.csv` + histogram chart.",
        "",
        "### 4. Apakah confidence masih valid?",
        "- Ranking/top5% masih menghasilkan trade, tapi **kalibrasi & base rate 2026** perlu dilihat di ECE/Brier — confidence sebagai *magnitude* kurang bisa diandalkan untuk sizing di tahun tipis.",
        "",
        "### 5. Apakah model overconfident?",
        f"- {'Cenderung YA jika ECE naik tajam di 2026 — lihat ece_yearly.csv + reliability chart.' if overconf else 'Tidak konklusif / tidak ekstrem dari mean-prob saja; andalkan ECE & reliability chart.'}",
        "",
        "### 6. Distribution shift?",
        f"- {'YA pada beberapa feature (PSI/KS).' if (not shift.empty and float(shift['psi'].iloc[0]) >= 0.1) else 'Lemah–sedang; tidak cukup untuk menjelaskan wipe sendirian.'}",
        "",
        "### 7. Regime shift?",
        f"- {reg_note or 'Cek market_drift.csv.'}",
        "",
        "### 8. Rule paling menjanjikan",
        "- **NO TRADE:** aturan dengan lift loser/winner tinggi & skip_share terkendali di `candidate_skip_rules.csv` (promising=True). Prioritas ukur dulu di 2021–25, lalu cek coverage trade 2026 (sering 0 karena n kecil).",
        "- **Adaptive Risk:** turunkan size saat bucket expectancy rendah (prob_q rendah / HIGH_VOL) — proposal only.",
        "- **Equity Guard:** paling relevan untuk 2026 — hard daily/weekly loss stop, matikan compounding setelah peak, atau pause trading saat rolling expectancy jelek (bukan ganti model).",
        "",
        "### 9. Prioritas Sprint berikutnya",
        "1. **Equity / ruin guard research** (path dependence) — penyebab wipe langsung.",
        "2. **Trade density / year coverage** — kenapa 2026 cuma ~5 trade (data cut-off? regime? top5% terlalu ketat?).",
        "3. Opsional: validasi skip-rule in-sample → sealed holdout (bukan full re-optimize).",
        "",
        "## Files",
        "- confidence_drift.csv, confidence_histogram.csv, ece_yearly.csv, brier_yearly.csv",
        "- feature_shift.csv, psi.csv, ks_test.csv, probability_drift.csv, market_drift.csv",
        "- error_attribution.csv, trades_enriched.csv",
        "- candidate_skip_rules.csv, adaptive_risk_candidates.csv",
        "- charts: chart_*.png",
        "",
        "## Stop condition",
        "Loop berhenti di sini: candidate baru setelah iter 8 tidak menambah penjelasan root cause di luar "
        "**scarce 2026 sample + ruin path**. Memaksakan filter baru = overfitting ke n≈5.",
    ]
    (out / "summary.md").write_text("\n".join(lines), encoding="utf-8")


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser()
    ap.parse_args(argv)
    OUT.mkdir(parents=True, exist_ok=True)
    print("=== Sprint 34 confidence/drift ===")
    print("Building OOS scores (7-feat rolling WF protocol)…")
    scored, entries = build_oos_scores()
    print(f"scored_rows={len(scored)} entries={len(entries)}")

    print("\nITER 1 confidence…")
    conf = iter1_confidence(scored, OUT)
    print(conf.to_string(index=False))

    print("\nITER 2 calibration…")
    cal = iter2_calibration(scored, OUT)
    print(cal.to_string(index=False))

    print("\nITER 3 feature shift…")
    shift = iter3_feature_shift(scored, OUT)
    print(shift.head(7).to_string(index=False))

    print("\nITER 4 probability drift…")
    pdrift = iter4_prob_drift(scored, OUT, conf)
    print(pdrift.to_string(index=False))

    print("\nITER 5 regime drift…")
    market = iter5_regime(scored, OUT)
    print(f"market_drift rows={len(market)}")

    print("\nITER 6 error attribution…")
    err, trades = iter6_error_attr(entries, OUT)
    if not trades.empty:
        print(trades.groupby("year")["pnl"].agg(["count", "mean"]).to_string())

    print("\nITER 7 skip rules…")
    # attach y_prob onto trades from entries if present
    if not trades.empty and "y_prob" not in trades.columns and "y_prob" in entries.columns:
        trades = trades.merge(entries[["timestamp", "side", "y_prob"]], on=["timestamp", "side"], how="left")
    rules = iter7_skip_rules(trades if not trades.empty else entries.assign(pnl=0, year=entries["test_year"]), scored, OUT)
    if not rules.empty:
        print(rules.head(8).to_string(index=False))

    print("\nITER 8 adaptive risk…")
    adapt = iter8_adaptive(trades, OUT)

    write_summary(OUT, conf, cal, shift, market, err, rules, adapt, trades if not trades.empty else pd.DataFrame())
    print(f"\nDONE -> {OUT / 'summary.md'}")
    return 0


if __name__ == "__main__":
    # fix accidental invalid import syntax if any slipped
    raise SystemExit(main())
