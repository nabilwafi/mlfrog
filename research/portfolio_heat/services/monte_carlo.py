"""Monte Carlo on heat policies (order reshuffle; concurrency approximate)."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from research.portfolio_heat import STARTING_EQUITY, HeatPolicy
from research.portfolio_heat.services.engine import run_heat_portfolio


def monte_carlo_heat(
    panel: pd.DataFrame,
    policy: HeatPolicy,
    *,
    starting_equity: float = STARTING_EQUITY,
    n_sims: int = 500,
    seed: int = 42,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """
    Reshuffle trade order. Heat overlap is approximate under shuffle (ponytail: order MC).
    """
    rng = np.random.default_rng(seed)
    n = len(panel)
    if n == 0:
        return pd.DataFrame(), {"n_sims": 0}

    finals, mdds = [], []
    for _ in range(n_sims):
        idx = rng.permutation(n)
        shuffled = panel.iloc[idx].reset_index(drop=True)
        # keep timestamps monotonic for calendar rules by reassigning sorted times
        # ponytail: preserve calendar path; only permute outcome attribution via net_return/atr swap
        base = panel.sort_values("timestamp").reset_index(drop=True).copy()
        base["net_return"] = shuffled["net_return"].to_numpy()
        base["side"] = shuffled["side"].to_numpy()
        if "holding_bars" in base.columns and "holding_bars" in shuffled.columns:
            base["holding_bars"] = shuffled["holding_bars"].to_numpy()
        _, curve, m = run_heat_portfolio(base, policy, starting_equity=starting_equity)
        finals.append(float(m.get("final_equity", starting_equity)))
        mdds.append(float(m.get("max_drawdown") or 0.0))

    fa = np.asarray(finals, dtype=float)
    da = np.asarray(mdds, dtype=float)
    q = np.quantile(fa, [0.05, 0.50, 0.95])
    summary = {
        "policy": policy.name,
        "n_sims": n_sims,
        "median_equity": float(q[1]),
        "p05_equity": float(q[0]),
        "p95_equity": float(q[2]),
        "ci95_low": float(q[0]),
        "ci95_high": float(q[2]),
        "worst_drawdown": float(np.max(da)) if len(da) else float("nan"),
        "median_drawdown": float(np.median(da)) if len(da) else float("nan"),
        "prob_lose_money": float(np.mean(fa < starting_equity)),
        "prob_double": float(np.mean(fa >= 2 * starting_equity)),
        "risk_of_ruin_50pct": float(np.mean(fa <= 0.5 * starting_equity)),
        "risk_of_ruin_25pct": float(np.mean(fa <= 0.25 * starting_equity)),
    }
    return pd.DataFrame({"final_equity": fa, "max_drawdown": da}), summary
