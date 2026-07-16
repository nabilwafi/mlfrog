"""Monte Carlo reshuffle + risk of ruin."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from research.portfolio_backtest.services.engine import run_portfolio


def monte_carlo(
    trades: pd.DataFrame,
    *,
    starting_equity: float,
    mode: str,
    fixed_lots: float | None,
    risk_pct: float | None,
    n_sims: int = 1000,
    seed: int = 42,
    enforce_volume_min: bool = True,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """
    Reshuffle trade order; re-run sizing path. Trade net_return/atr kept identical.
    """
    rng = np.random.default_rng(seed)
    n = len(trades)
    if n == 0:
        return pd.DataFrame(), {"n_sims": 0}

    finals = []
    mdds = []
    for _ in range(n_sims):
        idx = rng.permutation(n)
        shuffled = trades.iloc[idx].reset_index(drop=True)
        log, curve = run_portfolio(
            shuffled,
            starting_equity=starting_equity,
            mode=mode,
            fixed_lots=fixed_lots,
            risk_pct=risk_pct,
            enforce_volume_min=enforce_volume_min,
        )
        eq = curve["equity"].to_numpy(dtype=float)
        finals.append(float(eq[-1]))
        peak = np.maximum.accumulate(eq)
        dd = (peak - eq) / np.where(peak > 0, peak, np.nan)
        mdds.append(float(np.nanmax(dd)) if len(dd) else 0.0)

    finals_a = np.asarray(finals, dtype=float)
    mdds_a = np.asarray(mdds, dtype=float)
    q = np.quantile(finals_a, [0.05, 0.25, 0.50, 0.75, 0.95])
    summary = {
        "n_sims": n_sims,
        "median_final_equity": float(np.median(finals_a)),
        "p05_final_equity": float(q[0]),
        "p25_final_equity": float(q[1]),
        "p50_final_equity": float(q[2]),
        "p75_final_equity": float(q[3]),
        "p95_final_equity": float(q[4]),
        "median_max_dd": float(np.median(mdds_a)),
        "p05_max_dd": float(np.quantile(mdds_a, 0.05)),
        "p95_max_dd": float(np.quantile(mdds_a, 0.95)),
        "prob_lose_money": float(np.mean(finals_a < starting_equity)),
        "prob_double": float(np.mean(finals_a >= 2 * starting_equity)),
        "prob_below_50": float(np.mean(finals_a < 50.0)),
        "prob_above_500": float(np.mean(finals_a > 500.0)),
        "prob_ruin_50pct": float(np.mean(finals_a <= 0.5 * starting_equity)),
        "prob_ruin_25pct": float(np.mean(finals_a <= 0.25 * starting_equity)),
        "prob_ruin_10pct": float(np.mean(finals_a <= 0.10 * starting_equity)),
    }
    detail = pd.DataFrame({"final_equity": finals_a, "max_drawdown": mdds_a})
    return detail, summary
