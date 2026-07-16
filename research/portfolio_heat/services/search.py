"""Evaluate heat policies, walk-forward, accept/reject."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from research.portfolio_backtest.services.metrics import yearly_report
from research.portfolio_heat import STARTING_EQUITY, HeatPolicy, build_policy_grid
from research.portfolio_heat.services.engine import run_heat_portfolio


def evaluate_policies(
    panel: pd.DataFrame,
    policies: list[HeatPolicy] | None = None,
    *,
    starting_equity: float = STARTING_EQUITY,
) -> tuple[pd.DataFrame, dict[str, tuple[pd.DataFrame, pd.DataFrame, dict[str, Any]]]]:
    policies = policies or build_policy_grid()
    rows = []
    arts: dict[str, tuple[pd.DataFrame, pd.DataFrame, dict[str, Any]]] = {}
    for pol in policies:
        log, curve, m = run_heat_portfolio(panel, pol, starting_equity=starting_equity)
        arts[pol.name] = (log, curve, m)
        rows.append(
            {
                "policy": pol.name,
                "final_equity": m.get("final_equity"),
                "cagr": m.get("cagr"),
                "profit_factor": m.get("profit_factor"),
                "expectancy": m.get("expectancy"),
                "expectancy_frac": m.get("expectancy_frac"),
                "sharpe": m.get("sharpe"),
                "sortino": m.get("sortino"),
                "calmar": m.get("calmar"),
                "recovery_factor": m.get("recovery_factor"),
                "max_drawdown": m.get("max_drawdown"),
                "longest_losing_streak": m.get("longest_losing_streak"),
                "avg_monthly_return": m.get("avg_monthly_return"),
                "return_per_dd": m.get("return_per_dd"),
                "trades": m.get("trades"),
                "n_skipped": m.get("n_skipped"),
                "capital_efficiency": m.get("capital_efficiency"),
                "win_rate": m.get("win_rate"),
            }
        )
    table = pd.DataFrame(rows).sort_values(["calmar", "cagr"], ascending=False)
    return table, arts


def walk_forward_years(
    panel: pd.DataFrame,
    policy: HeatPolicy,
    *,
    starting_equity: float = STARTING_EQUITY,
) -> pd.DataFrame:
    """Per-year standalone runs (fresh equity) + chronological yearly slice from full log."""
    log, _, _ = run_heat_portfolio(panel, policy, starting_equity=starting_equity)
    chron = yearly_report(log, starting_equity=starting_equity)
    rows = []
    for year, g in panel.groupby("valid_year"):
        _, _, m = run_heat_portfolio(g, policy, starting_equity=starting_equity)
        rows.append(
            {
                "valid_year": int(year),
                "cagr": m.get("cagr"),
                "max_drawdown": m.get("max_drawdown"),
                "trades": m.get("trades"),
                "calmar": m.get("calmar"),
                "final_equity": m.get("final_equity"),
                "mode": "standalone_year",
            }
        )
    stand = pd.DataFrame(rows)
    if not chron.empty:
        chron = chron.copy()
        chron["mode"] = "chronological"
    return stand if chron.empty else pd.concat([stand, chron], ignore_index=True)


def acceptance(
    baseline: dict[str, Any],
    candidate: dict[str, Any],
    *,
    severe: float = 0.20,
) -> dict[str, Any]:
    """
    Accept if return improves OR DD decreases, without severe deterioration of the other.
    """
    b_cagr = float(baseline.get("cagr") or float("nan"))
    c_cagr = float(candidate.get("cagr") or float("nan"))
    b_dd = float(baseline.get("max_drawdown") or float("nan"))
    c_dd = float(candidate.get("max_drawdown") or float("nan"))

    ret_up = c_cagr == c_cagr and b_cagr == b_cagr and c_cagr > b_cagr
    dd_down = c_dd == c_dd and b_dd == b_dd and c_dd < b_dd

    cagr_rel = (c_cagr - b_cagr) / abs(b_cagr) if b_cagr == b_cagr and abs(b_cagr) > 1e-12 else float("nan")
    dd_rel = (c_dd - b_dd) / abs(b_dd) if b_dd == b_dd and abs(b_dd) > 1e-12 else float("nan")

    severe_ret = cagr_rel == cagr_rel and cagr_rel < -severe
    severe_dd = dd_rel == dd_rel and dd_rel > severe

    ok = bool((ret_up or dd_down) and not (severe_ret or severe_dd))
    return {
        "accepted": ok,
        "ret_up": bool(ret_up),
        "dd_down": bool(dd_down),
        "cagr_rel": cagr_rel,
        "dd_rel": dd_rel,
        "severe_ret": bool(severe_ret),
        "severe_dd": bool(severe_dd),
    }


def score_policies(
    table: pd.DataFrame,
    baseline_row: pd.Series,
    *,
    min_trades: int = 100,
) -> pd.DataFrame:
    rows = []
    for _, r in table.iterrows():
        acc = acceptance(baseline_row.to_dict(), r.to_dict())
        trades = int(r.get("trades") or 0)
        production_ok = bool(acc["accepted"] and trades >= min_trades)
        rows.append({**r.to_dict(), **acc, "production_ok": production_ok, "trades_ok": trades >= min_trades})
    out = pd.DataFrame(rows)
    # production_ok first, then accepted, then calmar/cagr
    out["_rank"] = np.where(out["production_ok"], 0, np.where(out["accepted"], 1, 2))
    return out.sort_values(["_rank", "calmar", "cagr"], ascending=[True, False, False]).drop(columns=["_rank"])

