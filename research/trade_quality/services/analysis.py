"""Trade quality research: buckets, interactions, risk, capital, stability."""

from __future__ import annotations

from typing import Any

import lightgbm as lgb
import numpy as np
import pandas as pd

from research.confidence_layer.services.research_ops import (
    risk_for_confidence,
    run_dynamic_risk_portfolio,
)
from research.portfolio_backtest.services.metrics import profit_factor_pnl, sharpe_from_returns, sortino_from_returns
from research.trade_quality import BUCKETS, RISK_SCHEDULES, STARTING_EQUITY
from research.trade_quality.services.scorers import FEATURE_COLS


def assign_bucket(score: pd.Series) -> pd.Series:
    c = score.astype(float)
    labels = np.array(["0-20"] * len(c), dtype=object)
    for name, lo, hi in BUCKETS:
        labels[(c >= lo) & (c < hi)] = name
    return pd.Series(labels, index=c.index)


def _dd(returns: np.ndarray) -> float:
    eq = np.cumsum(returns)
    peak = np.maximum.accumulate(eq)
    return float((peak - eq).max()) if len(eq) else float("nan")


def bucket_performance(panel: pd.DataFrame, score_col: str = "trade_quality") -> pd.DataFrame:
    rows = []
    for name, lo, hi in BUCKETS:
        sub = panel.loc[(panel[score_col] >= lo) & (panel[score_col] < hi)]
        if sub.empty:
            rows.append({"bucket": name, "trades": 0, "score_col": score_col})
            continue
        net = sub["net_return"].to_numpy(dtype=float)
        trade_ret = net  # fractional
        rows.append(
            {
                "bucket": name,
                "score_col": score_col,
                "trades": int(len(sub)),
                "win_rate": float((net > 0).mean()),
                "expectancy": float(np.mean(net)),
                "profit_factor": profit_factor_pnl(net),
                "annual_return": float(np.sum(net)),
                "drawdown": _dd(net),
                "sharpe": sharpe_from_returns(trade_ret),
                "sortino": sortino_from_returns(trade_ret),
            }
        )
    return pd.DataFrame(rows)


def monotonicity_ok(buckets: pd.DataFrame) -> dict[str, Any]:
    b = buckets.loc[buckets["trades"] > 0].copy()
    if len(b) < 2:
        return {"monotonic": False, "spearman_bucket_expectancy": float("nan")}
    # order buckets by lo
    order = {n: i for i, (n, _, _) in enumerate(BUCKETS)}
    b["ord"] = b["bucket"].map(order)
    b = b.sort_values("ord")
    sp = b["ord"].corr(b["expectancy"], method="spearman")
    diffs = b["expectancy"].diff().dropna()
    return {
        "monotonic": bool((diffs >= -1e-6).mean() >= 0.6) if len(diffs) else False,
        "spearman_bucket_expectancy": float(sp) if sp == sp else float("nan"),
        "frac_nondecreasing_steps": float((diffs >= -1e-6).mean()) if len(diffs) else float("nan"),
    }


def interaction_analysis(panel: pd.DataFrame, score_col: str = "trade_quality") -> pd.DataFrame:
    rows = []
    # High vs low TQ split
    med = float(panel[score_col].median())
    hi = panel.loc[panel[score_col] >= med]
    lo = panel.loc[panel[score_col] < med]

    def edge(sub_hi: pd.DataFrame, sub_lo: pd.DataFrame) -> float:
        if sub_hi.empty or sub_lo.empty:
            return float("nan")
        return float(sub_hi["net_return"].mean() - sub_lo["net_return"].mean())

    # Long vs short
    for side in ("long", "short"):
        rows.append(
            {
                "segment": f"side={side}",
                "tq_edge": edge(hi.loc[hi["side"] == side], lo.loc[lo["side"] == side]),
                "hi_n": int(((panel["side"] == side) & (panel[score_col] >= med)).sum()),
                "hi_expectancy": float(hi.loc[hi["side"] == side, "net_return"].mean())
                if (hi["side"] == side).any()
                else float("nan"),
            }
        )

    if "d1_regime" in panel.columns:
        for reg, g in panel.groupby("d1_regime"):
            g_hi = g.loc[g[score_col] >= med]
            g_lo = g.loc[g[score_col] < med]
            rows.append(
                {
                    "segment": f"regime={reg}",
                    "tq_edge": edge(g_hi, g_lo),
                    "hi_n": int(len(g_hi)),
                    "hi_expectancy": float(g_hi["net_return"].mean()) if len(g_hi) else float("nan"),
                }
            )

    for sess_col, label in (
        ("session_london", "London"),
        ("session_newyork", "NewYork"),
        ("session_asia", "Asia"),
        ("session_london_ny_overlap", "Overlap"),
    ):
        if sess_col not in panel.columns:
            continue
        on = panel.loc[panel[sess_col].astype(float) >= 0.5]
        rows.append(
            {
                "segment": f"session={label}",
                "tq_edge": edge(on.loc[on[score_col] >= med], on.loc[on[score_col] < med]),
                "hi_n": int((on[score_col] >= med).sum()),
                "hi_expectancy": float(on.loc[on[score_col] >= med, "net_return"].mean())
                if (on[score_col] >= med).any()
                else float("nan"),
            }
        )

    if "volatility_rank" in panel.columns or "vol_regime" in panel.columns:
        vol = panel["volatility_rank"] if "volatility_rank" in panel.columns else panel["vol_regime"]
        vmed = float(vol.median())
        for name, mask in (("high_vol", vol >= vmed), ("low_vol", vol < vmed)):
            sub = panel.loc[mask]
            rows.append(
                {
                    "segment": name,
                    "tq_edge": edge(sub.loc[sub[score_col] >= med], sub.loc[sub[score_col] < med]),
                    "hi_n": int((sub[score_col] >= med).sum()),
                    "hi_expectancy": float(sub.loc[sub[score_col] >= med, "net_return"].mean())
                    if (sub[score_col] >= med).any()
                    else float("nan"),
                }
            )

    return pd.DataFrame(rows).sort_values("tq_edge", ascending=False)


def capital_allocation_table(panel: pd.DataFrame, score_col: str = "trade_quality") -> pd.DataFrame:
    rows = []
    for name, lo, hi in BUCKETS:
        sub = panel.loc[(panel[score_col] >= lo) & (panel[score_col] < hi)]
        if sub.empty:
            rows.append({"bucket": name, "trades": 0})
            continue
        net = sub["net_return"].to_numpy(dtype=float)
        downside = net[net < 0]
        upside = net[net > 0]
        rows.append(
            {
                "bucket": name,
                "trades": int(len(sub)),
                "expected_return": float(np.mean(net)),
                "expected_upside": float(np.mean(upside)) if len(upside) else 0.0,
                "expected_downside": float(np.mean(downside)) if len(downside) else 0.0,
                "expected_drawdown": _dd(net),
                "p05_return": float(np.quantile(net, 0.05)),
                "p95_return": float(np.quantile(net, 0.95)),
                "suggested_risk_pct": _suggest_risk(float(np.mean(net)), _dd(net)),
            }
        )
    return pd.DataFrame(rows)


def _suggest_risk(exp: float, dd: float) -> float:
    if exp <= 0:
        return 0.0
    if dd > 0.08:
        return 0.005
    if exp > 0.002:
        return 0.02
    if exp > 0.001:
        return 0.015
    if exp > 0.0005:
        return 0.01
    return 0.005


def _empirical_schedule(panel: pd.DataFrame, score_col: str) -> tuple[tuple[float, float, float], ...]:
    """Auto schedule from bucket expectancy / suggested risk (research only)."""
    cap = capital_allocation_table(panel, score_col=score_col)
    parts: list[tuple[float, float, float]] = []
    for name, lo, hi in BUCKETS:
        row = cap.loc[cap["bucket"] == name]
        risk = float(row["suggested_risk_pct"].iloc[0]) if len(row) and int(row["trades"].iloc[0]) > 0 else 0.0
        parts.append((lo, hi, risk))
    return tuple(parts)


def search_tq_risk(
    panel: pd.DataFrame,
    *,
    score_col: str = "trade_quality",
    starting_equity: float = STARTING_EQUITY,
) -> pd.DataFrame:
    schedules: list[tuple[str, tuple[tuple[float, float, float], ...]]] = list(RISK_SCHEDULES)
    schedules.append(("tq_empirical_buckets", _empirical_schedule(panel, score_col)))

    rows = []
    for name, schedule in schedules:
        drive = score_col
        if name.startswith("conf_"):
            if "confidence" not in panel.columns:
                continue
            drive = "confidence"
        work = panel.copy()
        work["confidence"] = work[drive]  # run_dynamic_risk_portfolio keys off confidence
        _, _, m = run_dynamic_risk_portfolio(work, schedule, starting_equity=starting_equity)
        dd = float(m.get("max_drawdown") or float("nan"))
        cagr = float(m.get("cagr") or float("nan"))
        ret_dd = (cagr / dd) if dd == dd and dd > 0 and cagr == cagr else float("nan")
        rows.append(
            {
                "schedule": name,
                "drive_score": drive,
                "final_equity": m.get("final_equity"),
                "cagr": cagr,
                "max_drawdown": dd,
                "calmar": m.get("calmar"),
                "recovery_factor": m.get("recovery_factor"),
                "profit_factor": m.get("profit_factor"),
                "sharpe": m.get("sharpe"),
                "sortino": m.get("sortino"),
                "return_per_dd": ret_dd,
                "trades": m.get("trades"),
            }
        )
    return pd.DataFrame(rows).sort_values(["calmar", "cagr"], ascending=False)


def yearly_stability(panel: pd.DataFrame, score_col: str = "trade_quality") -> pd.DataFrame:
    rows = []
    for year, g in panel.groupby("valid_year"):
        b = bucket_performance(g, score_col=score_col)
        mono = monotonicity_ok(b)
        # ranking: mean TQ in winners vs losers
        w = g.loc[g["meta_label"] == 1, score_col].mean() if (g["meta_label"] == 1).any() else float("nan")
        l = g.loc[g["meta_label"] == 0, score_col].mean() if (g["meta_label"] == 0).any() else float("nan")
        rows.append(
            {
                "valid_year": int(year),
                "trades": int(len(g)),
                "spearman_tq_net": float(g[score_col].corr(g["net_return"], method="spearman") or float("nan")),
                "mono_spearman": mono.get("spearman_bucket_expectancy"),
                "mean_tq_win": float(w) if w == w else float("nan"),
                "mean_tq_lose": float(l) if l == l else float("nan"),
                "ranking_gap": float(w - l) if w == w and l == l else float("nan"),
                "expectancy": float(g["net_return"].mean()),
            }
        )
    return pd.DataFrame(rows)


def explainability(panel: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """SHAP-style pred_contrib + permutation (Δ Spearman with net after shuffle)."""
    frame = panel.copy()
    feats = [c for c in FEATURE_COLS if c in frame.columns]
    x = frame[feats].astype(float).fillna(0.5)
    y = frame["meta_label"].astype(int).to_numpy()
    if len(np.unique(y)) < 2:
        return pd.DataFrame(), pd.DataFrame()
    dtrain = lgb.Dataset(x, label=y, feature_name=feats)
    booster = lgb.train(
        {"objective": "binary", "verbosity": -1, "seed": 42, "learning_rate": 0.05, "num_leaves": 15},
        dtrain,
        num_boost_round=80,
    )
    contrib = np.asarray(booster.predict(x, pred_contrib=True), dtype=float)
    shap_abs = np.abs(contrib[:, :-1]).mean(axis=0)
    shap_df = pd.DataFrame({"feature": feats, "mean_abs_shap": shap_abs}).sort_values(
        "mean_abs_shap", ascending=False
    )

    # permutation vs net_return spearman of TQ weighted score proxy = model leaf? Use feature alone
    base = frame["trade_quality"].corr(frame["net_return"], method="spearman")
    rng = np.random.default_rng(42)
    perm_rows = []
    for i, f in enumerate(feats):
        xp = x.copy()
        xp[f] = rng.permutation(xp[f].to_numpy())
        # recompute simple quality proxy: mean of feats
        proxy = xp.mean(axis=1)
        sp = proxy.corr(frame["net_return"], method="spearman")
        drop = float(base - sp) if base == base and sp == sp else float("nan")
        perm_rows.append({"feature": f, "permutation_drop": drop})
    perm_df = pd.DataFrame(perm_rows).sort_values("permutation_drop", ascending=False)
    return shap_df, perm_df
