"""Percentile threshold + context filter analysis."""

from __future__ import annotations

import numpy as np
import pandas as pd

from research.probability_calibration.services.metrics import (
    CONTEXT_FILTER_FEATURES,
    PERCENTILES,
    select_top_percentile,
    trade_stats,
)


def analyze_percentile_thresholds(
    calibrated: pd.DataFrame,
    *,
    prob_col: str,
    cost: float,
    percentiles: tuple[float, ...] = PERCENTILES,
) -> pd.DataFrame:
    """
    Per side × percentile: select top-pct within each WF window, pool trades.
    Also report per-window win_rate std as consistency.
    """
    rows: list[dict] = []
    for side in sorted(calibrated["side"].unique()):
        side_df = calibrated.loc[calibrated["side"] == side]
        for pct in percentiles:
            window_stats = []
            trades: list[pd.DataFrame] = []
            for window, g in side_df.groupby("window", sort=True):
                picked = select_top_percentile(g, prob_col, pct)
                if picked.empty:
                    continue
                st = trade_stats(
                    picked["realized_return"].to_numpy(),
                    picked["y_true"].to_numpy(),
                    cost=cost,
                )
                st["window"] = window
                window_stats.append(st)
                trades.append(picked)
            if not trades:
                rows.append(
                    {
                        "side": side,
                        "percentile": pct,
                        "prob_col": prob_col,
                        "n_trades": 0,
                        "win_rate": float("nan"),
                        "avg_return": float("nan"),
                        "expectancy": float("nan"),
                        "profit_factor": float("nan"),
                        "max_drawdown": float("nan"),
                        "wf_windows": 0,
                        "wf_win_rate_mean": float("nan"),
                        "wf_win_rate_std": float("nan"),
                        "wf_positive_expectancy_windows": 0,
                    }
                )
                continue
            all_tr = pd.concat(trades, ignore_index=True)
            pooled = trade_stats(
                all_tr["realized_return"].to_numpy(),
                all_tr["y_true"].to_numpy(),
                cost=cost,
            )
            wr = [float(w["win_rate"]) for w in window_stats if w["n_trades"] > 0]
            exp_pos = sum(
                1 for w in window_stats if w["n_trades"] > 0 and w["expectancy"] == w["expectancy"] and w["expectancy"] > 0
            )
            rows.append(
                {
                    "side": side,
                    "percentile": pct,
                    "prob_col": prob_col,
                    **pooled,
                    "wf_windows": int(len(window_stats)),
                    "wf_win_rate_mean": float(np.mean(wr)) if wr else float("nan"),
                    "wf_win_rate_std": float(np.std(wr, ddof=1)) if len(wr) > 1 else 0.0,
                    "wf_positive_expectancy_windows": int(exp_pos),
                }
            )
    return pd.DataFrame(rows)


def analyze_context_filters(
    calibrated: pd.DataFrame,
    *,
    prob_col: str,
    cost: float,
    primary_percentiles: tuple[float, ...] = (0.05, 0.10, 0.15, 0.20),
    context_features: tuple[str, ...] = CONTEXT_FILTER_FEATURES,
) -> pd.DataFrame:
    """
    Top probability percentile AND structure feature in top half (within-window).
    """
    rows: list[dict] = []
    present = [f for f in context_features if f in calibrated.columns]
    for side in sorted(calibrated["side"].unique()):
        side_df = calibrated.loc[calibrated["side"] == side]
        for pct in primary_percentiles:
            # probability-only baseline row for comparison is in threshold_analysis
            for feat in present:
                trades: list[pd.DataFrame] = []
                window_exp = []
                for window, g in side_df.groupby("window", sort=True):
                    if g[feat].isna().all():
                        continue
                    med = float(g[feat].median())
                    # ponytail: for distance_from_equilibrium, extreme = high |value|
                    if "distance" in feat:
                        rank_col = g[feat].abs()
                        thr = float(rank_col.median())
                        ctx_ok = rank_col >= thr
                    else:
                        ctx_ok = g[feat] >= med
                    pool = g.loc[ctx_ok]
                    picked = select_top_percentile(pool, prob_col, pct)
                    if picked.empty:
                        continue
                    st = trade_stats(
                        picked["realized_return"].to_numpy(),
                        picked["y_true"].to_numpy(),
                        cost=cost,
                    )
                    window_exp.append(st["expectancy"])
                    trades.append(picked)
                if not trades:
                    rows.append(
                        {
                            "side": side,
                            "percentile": pct,
                            "context_feature": feat,
                            "prob_col": prob_col,
                            "n_trades": 0,
                            "win_rate": float("nan"),
                            "avg_return": float("nan"),
                            "expectancy": float("nan"),
                            "profit_factor": float("nan"),
                            "max_drawdown": float("nan"),
                            "wf_windows": 0,
                            "wf_positive_expectancy_windows": 0,
                        }
                    )
                    continue
                all_tr = pd.concat(trades, ignore_index=True)
                pooled = trade_stats(
                    all_tr["realized_return"].to_numpy(),
                    all_tr["y_true"].to_numpy(),
                    cost=cost,
                )
                rows.append(
                    {
                        "side": side,
                        "percentile": pct,
                        "context_feature": feat,
                        "prob_col": prob_col,
                        **pooled,
                        "wf_windows": int(len(trades)),
                        "wf_positive_expectancy_windows": int(
                            sum(1 for e in window_exp if e == e and e > 0)
                        ),
                    }
                )
    return pd.DataFrame(rows)


def pick_optimal_threshold(threshold_df: pd.DataFrame, side: str) -> dict:
    """Prefer positive expectancy, then PF, then WF consistency, then more trades."""
    sub = threshold_df.loc[threshold_df["side"] == side].copy()
    if sub.empty:
        return {"side": side, "percentile": float("nan"), "reason": "no rows"}
    sub = sub.loc[sub["n_trades"] > 0]
    if sub.empty:
        return {"side": side, "percentile": float("nan"), "reason": "no trades"}
    # score: expectancy first among positive; else least-bad
    pos = sub.loc[sub["expectancy"] > 0].copy()
    cand = pos if not pos.empty else sub
    cand = cand.sort_values(
        by=["expectancy", "profit_factor", "wf_positive_expectancy_windows", "n_trades"],
        ascending=[False, False, False, False],
    )
    best = cand.iloc[0]
    return {
        "side": side,
        "percentile": float(best["percentile"]),
        "n_trades": int(best["n_trades"]),
        "win_rate": float(best["win_rate"]),
        "expectancy": float(best["expectancy"]),
        "profit_factor": float(best["profit_factor"]),
        "max_drawdown": float(best["max_drawdown"]),
        "wf_positive_expectancy_windows": int(best["wf_positive_expectancy_windows"]),
        "prob_col": str(best["prob_col"]),
        "reason": "max expectancy among feasible rows",
    }
