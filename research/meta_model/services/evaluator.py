"""Evaluate Primary vs Primary+Meta across fixed thresholds."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.metrics import average_precision_score, brier_score_loss, roc_auc_score

from research.feature_ablation.services.walk_forward_runner import expected_calibration_error
from research.meta_feature_ablation.services.ablation_runner import (
    _sharpe,
    trading_metrics,
)
from research.meta_model import REFERENCE_THRESHOLD, THRESHOLDS


def _net(frame: pd.DataFrame) -> np.ndarray:
    return frame["net_return"].to_numpy(dtype=float)


def slice_metrics(frame: pd.DataFrame, mask: np.ndarray, *, cost: float) -> dict[str, float]:
    m = trading_metrics(frame, taken_mask=mask, cost=cost)
    n_cand = int(len(frame))
    n_take = int(mask.sum())
    m["n_candidates"] = n_cand
    m["n_skipped"] = int(n_cand - n_take)
    m["trade_reduction_pct"] = float(1.0 - n_take / n_cand) if n_cand else float("nan")
    return m


def classify_at_threshold(y: np.ndarray, p: np.ndarray, *, threshold: float) -> dict[str, float]:
    y = np.asarray(y, dtype=int)
    p = np.clip(np.asarray(p, dtype=float), 1e-6, 1 - 1e-6)
    if len(np.unique(y)) < 2:
        return {"roc_auc": float("nan"), "pr_auc": float("nan"), "brier": float("nan"), "ece": float("nan")}
    return {
        "roc_auc": float(roc_auc_score(y, p)),
        "pr_auc": float(average_precision_score(y, p)),
        "brier": float(brier_score_loss(y, p)),
        "ece": float(expected_calibration_error(y, p)),
    }


def evaluate_thresholds(
    oof: pd.DataFrame,
    *,
    thresholds: tuple[float, ...] = THRESHOLDS,
    cost: float = 0.00015,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Per (threshold, year) + aggregate rows for Primary and Primary+Meta.
    Primary = take all candidates in the year slice.
    """
    years = sorted(int(y) for y in oof["valid_year"].unique())
    rows: list[dict] = []
    for year in years:
        part = oof.loc[oof["valid_year"] == year].copy()
        if part.empty:
            continue
        y = part["meta_label"].to_numpy(dtype=int)
        p = part["meta_proba"].to_numpy(dtype=float)
        clf = classify_at_threshold(y, p, threshold=REFERENCE_THRESHOLD)

        # Primary: take all
        prim = slice_metrics(part, np.ones(len(part), dtype=bool), cost=cost)
        rows.append(
            {
                "system": "primary",
                "threshold": float("nan"),
                "valid_year": year,
                **prim,
                **{f"clf_{k}": v for k, v in clf.items()},
            }
        )
        for thr in thresholds:
            taken = p >= thr
            meta = slice_metrics(part, taken, cost=cost)
            clf_t = classify_at_threshold(y, p, threshold=thr)
            rows.append(
                {
                    "system": "primary_plus_meta",
                    "threshold": thr,
                    "valid_year": year,
                    **meta,
                    **{f"clf_{k}": v for k, v in clf_t.items()},
                }
            )

    by_year = pd.DataFrame(rows)

    # Aggregate across years per system/threshold
    agg_rows = []
    # Primary once
    prim_years = by_year.loc[by_year["system"] == "primary"]
    if not prim_years.empty:
        agg_rows.append(_aggregate(prim_years, system="primary", threshold=float("nan")))
    for thr in thresholds:
        sub = by_year.loc[
            (by_year["system"] == "primary_plus_meta") & (by_year["threshold"] == thr)
        ]
        if not sub.empty:
            agg_rows.append(_aggregate(sub, system="primary_plus_meta", threshold=thr))
    summary = pd.DataFrame(agg_rows)
    return by_year, summary


def _aggregate(frame: pd.DataFrame, *, system: str, threshold: float) -> dict[str, Any]:
    # Reconstruct pooled trade stats carefully: sum trades, weighted expectancy
    n_trades = int(frame["n_trades"].sum())
    n_cand = int(frame["n_candidates"].sum())
    annual_sum = float(frame["annual_return"].sum())
    # Expectancy pooled ≈ total PnL / n_trades
    total_pnl = annual_sum
    exp = total_pnl / n_trades if n_trades else 0.0
    # Win rate: not recoverable exactly without trades; mean of yearly when n>0
    wr = float(frame.loc[frame["n_trades"] > 0, "win_rate"].mean()) if (frame["n_trades"] > 0).any() else float("nan")
    pf = float(frame.loc[frame["n_trades"] > 0, "profit_factor"].replace([np.inf], np.nan).dropna().mean()) if (frame["n_trades"] > 0).any() else float("nan")
    dd = float(frame["max_drawdown"].max())  # worst year DD (conservative)
    sharpe = float(frame.loc[frame["n_trades"] > 1, "sharpe"].mean()) if (frame["n_trades"] > 1).any() else float("nan")
    return {
        "system": system,
        "threshold": threshold,
        "n_trades": n_trades,
        "n_candidates": n_cand,
        "n_skipped": n_cand - n_trades,
        "trade_reduction_pct": float(1.0 - n_trades / n_cand) if n_cand else float("nan"),
        "win_rate": wr,
        "expectancy": exp,
        "avg_return": exp,
        "profit_factor": pf,
        "annual_return": annual_sum,
        "max_drawdown": dd,
        "sharpe": sharpe,
        "roc_auc_mean": float(frame["clf_roc_auc"].mean()),
        "pr_auc_mean": float(frame["clf_pr_auc"].mean()),
        "brier_mean": float(frame["clf_brier"].mean()),
        "ece_mean": float(frame["clf_ece"].mean()),
    }


def equity_curve(
    by_year: pd.DataFrame,
    *,
    threshold: float,
    starting_equity: float = 80.0,
) -> pd.DataFrame:
    """Compound yearly: equity *= (1 + annual_return) for Primary and Meta."""
    years = sorted(int(y) for y in by_year["valid_year"].unique())
    rows = []
    eq_p = float(starting_equity)
    eq_m = float(starting_equity)
    for year in years:
        p_row = by_year.loc[(by_year["system"] == "primary") & (by_year["valid_year"] == year)]
        m_row = by_year.loc[
            (by_year["system"] == "primary_plus_meta")
            & (by_year["threshold"] == threshold)
            & (by_year["valid_year"] == year)
        ]
        r_p = float(p_row["annual_return"].iloc[0]) if not p_row.empty else 0.0
        r_m = float(m_row["annual_return"].iloc[0]) if not m_row.empty else 0.0
        eq_p *= 1.0 + r_p
        eq_m *= 1.0 + r_m
        rows.append(
            {
                "valid_year": year,
                "threshold": threshold,
                "primary_annual_return": r_p,
                "meta_annual_return": r_m,
                "primary_equity": eq_p,
                "meta_equity": eq_m,
            }
        )
    return pd.DataFrame(rows)


def all_equity_curves(
    by_year: pd.DataFrame,
    *,
    thresholds: tuple[float, ...] = THRESHOLDS,
    starting_equity: float = 80.0,
) -> pd.DataFrame:
    return pd.concat(
        [equity_curve(by_year, threshold=t, starting_equity=starting_equity) for t in thresholds],
        ignore_index=True,
    )


def statistical_meaning(
    oof: pd.DataFrame,
    by_year: pd.DataFrame,
    *,
    threshold: float = REFERENCE_THRESHOLD,
    seed: int = 42,
) -> dict[str, Any]:
    """Lightweight checks — not threshold optimization."""
    y = oof["meta_label"].to_numpy(dtype=int)
    p = oof["meta_proba"].to_numpy(dtype=float)
    out: dict[str, Any] = {}
    if len(np.unique(y)) >= 2:
        out["roc_auc_oof"] = float(roc_auc_score(y, p))
        # Mann-Whitney: scores for winners vs losers
        u = stats.mannwhitneyu(p[y == 1], p[y == 0], alternative="greater")
        out["mannwhitney_u"] = float(u.statistic)
        out["mannwhitney_pvalue"] = float(u.pvalue)
    else:
        out["roc_auc_oof"] = float("nan")
        out["mannwhitney_pvalue"] = float("nan")

    # Bootstrap CI on expectancy lift (pooled trades) at threshold
    rng = np.random.default_rng(seed)
    prim_net = _net(oof)
    taken = p >= threshold
    meta_net = prim_net[taken]
    # Sample years with replacement for yearly annual_return lift
    years = sorted(int(y) for y in by_year["valid_year"].unique())
    lifts = []
    for _ in range(2000):
        ys = rng.choice(years, size=len(years), replace=True)
        lp = 0.0
        lm = 0.0
        for yr in ys:
            pr = by_year.loc[(by_year["system"] == "primary") & (by_year["valid_year"] == yr)]
            mr = by_year.loc[
                (by_year["system"] == "primary_plus_meta")
                & (by_year["threshold"] == threshold)
                & (by_year["valid_year"] == yr)
            ]
            lp += float(pr["annual_return"].iloc[0]) if not pr.empty else 0.0
            lm += float(mr["annual_return"].iloc[0]) if not mr.empty else 0.0
        lifts.append(lm - lp)
    lifts_a = np.asarray(lifts, dtype=float)
    out["lift_annual_sum_mean"] = float(np.mean(lifts_a))
    out["lift_annual_sum_ci95_lo"] = float(np.quantile(lifts_a, 0.025))
    out["lift_annual_sum_ci95_hi"] = float(np.quantile(lifts_a, 0.975))
    out["lift_ci_excludes_zero"] = bool(
        out["lift_annual_sum_ci95_lo"] > 0 or out["lift_annual_sum_ci95_hi"] < 0
    )
    out["meta_expectancy"] = float(np.mean(meta_net)) if len(meta_net) else float("nan")
    out["primary_expectancy"] = float(np.mean(prim_net)) if len(prim_net) else float("nan")
    # Years meta AR > primary AR
    wins = 0
    for yr in years:
        pr = by_year.loc[(by_year["system"] == "primary") & (by_year["valid_year"] == yr)]
        mr = by_year.loc[
            (by_year["system"] == "primary_plus_meta")
            & (by_year["threshold"] == threshold)
            & (by_year["valid_year"] == yr)
        ]
        if not pr.empty and not mr.empty and float(mr["annual_return"].iloc[0]) > float(pr["annual_return"].iloc[0]):
            wins += 1
    out["years_meta_beats_primary_ar"] = wins
    out["n_years"] = len(years)
    out["threshold_ref"] = threshold
    # Verdict
    meaningful = (
        out.get("mannwhitney_pvalue", 1.0) < 0.05
        and out.get("roc_auc_oof", 0.5) > 0.5
    )
    out["statistically_meaningful"] = bool(meaningful)
    out["note"] = (
        "meaningful if OOF ROC>0.5 and Mann-Whitney p<0.05 (scores separate winners/losers); "
        "bootstrap CI is for annual-return lift, not a chosen optimal threshold"
    )
    return out


def build_answers(
    summary: pd.DataFrame,
    by_year: pd.DataFrame,
    equity: pd.DataFrame,
    stats_out: dict[str, Any],
    *,
    threshold: float = REFERENCE_THRESHOLD,
) -> dict[str, Any]:
    prim = summary.loc[summary["system"] == "primary"]
    meta = summary.loc[
        (summary["system"] == "primary_plus_meta") & (summary["threshold"] == threshold)
    ]
    p = prim.iloc[0] if not prim.empty else None
    m = meta.iloc[0] if not meta.empty else None

    eq_t = equity.loc[equity["threshold"] == threshold].sort_values("valid_year")
    eq_curve = [
        {
            "year": int(r.valid_year),
            "primary": float(r.primary_equity),
            "meta": float(r.meta_equity),
            "primary_ar": float(r.primary_annual_return),
            "meta_ar": float(r.meta_annual_return),
        }
        for r in eq_t.itertuples()
    ]

    if p is None or m is None:
        return {"error": "missing summary", "threshold_ref": threshold}

    return {
        "threshold_ref": threshold,
        "q1_profitability_improved": bool(float(m["expectancy"]) > float(p["expectancy"]))
        or bool(float(m["annual_return"]) > float(p["annual_return"])),
        "q1_detail": {
            "primary_expectancy": float(p["expectancy"]),
            "meta_expectancy": float(m["expectancy"]),
            "primary_pf": float(p["profit_factor"]),
            "meta_pf": float(m["profit_factor"]),
        },
        "q2_drawdown_reduced": bool(float(m["max_drawdown"]) < float(p["max_drawdown"])),
        "q2_detail": {
            "primary_max_dd": float(p["max_drawdown"]),
            "meta_max_dd": float(m["max_drawdown"]),
        },
        "q3_trade_count_reduced": bool(int(m["n_trades"]) < int(p["n_trades"])),
        "q3_detail": {
            "primary_n": int(p["n_trades"]),
            "meta_n": int(m["n_trades"]),
            "reduction_pct": float(m["trade_reduction_pct"]),
        },
        "q4_annual_return_before": float(p["annual_return"]),
        "q4_annual_return_after": float(m["annual_return"]),
        "q5_equity_curve": eq_curve,
        "q5_start": 80.0,
        "q5_primary_final": float(eq_t["primary_equity"].iloc[-1]) if not eq_t.empty else 80.0,
        "q5_meta_final": float(eq_t["meta_equity"].iloc[-1]) if not eq_t.empty else 80.0,
        "q6_statistically_meaningful": stats_out.get("statistically_meaningful"),
        "q6_detail": stats_out,
    }
