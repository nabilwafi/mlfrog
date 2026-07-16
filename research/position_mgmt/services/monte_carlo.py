"""Monte Carlo on fixed-entry position management panels."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from research.position_mgmt import BASE_RISK, STARTING_EQUITY
from research.position_mgmt.services.evaluate import run_fixed_portfolio


def monte_carlo_fixed(
    trades: pd.DataFrame,
    *,
    starting_equity: float = STARTING_EQUITY,
    risk_pct: float = BASE_RISK,
    n_sims: int = 500,
    seed: int = 42,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Reshuffle net_return outcomes onto chronological schedule (ponytail MC)."""
    rng = np.random.default_rng(seed)
    n = len(trades)
    if n == 0:
        return pd.DataFrame(), {"n_sims": 0}
    base = trades.sort_values("timestamp").reset_index(drop=True)
    finals, mdds = [], []
    for _ in range(n_sims):
        sh = base.copy()
        idx = rng.permutation(n)
        sh["net_return"] = base["net_return"].to_numpy()[idx]
        _, _, m = run_fixed_portfolio(sh, starting_equity=starting_equity, risk_pct=risk_pct)
        finals.append(float(m.get("final_equity", starting_equity)))
        mdds.append(float(m.get("max_drawdown") or 0.0))
    fa, da = np.asarray(finals), np.asarray(mdds)
    q = np.quantile(fa, [0.05, 0.5, 0.95])
    summary = {
        "n_sims": n_sims,
        "median_equity": float(q[1]),
        "p05_equity": float(q[0]),
        "p95_equity": float(q[2]),
        "ci95_low": float(q[0]),
        "ci95_high": float(q[2]),
        "median_drawdown": float(np.median(da)),
        "worst_drawdown": float(np.max(da)) if len(da) else float("nan"),
        "prob_lose_money": float(np.mean(fa < starting_equity)),
        "prob_double": float(np.mean(fa >= 2 * starting_equity)),
        "risk_of_ruin_50pct": float(np.mean(fa <= 0.5 * starting_equity)),
    }
    return pd.DataFrame({"final_equity": fa, "max_drawdown": da}), summary
