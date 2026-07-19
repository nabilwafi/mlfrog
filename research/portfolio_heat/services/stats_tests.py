"""Bootstrap / Mann-Whitney / permutation tests vs baseline."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
from scipy import stats


def _trade_returns(log: pd.DataFrame) -> np.ndarray:
    taken = log.loc[~log["skipped"].astype(bool)] if "skipped" in log.columns else log
    if taken.empty:
        return np.array([], dtype=float)
    return (taken["pnl"] / taken["equity_before"].replace(0, np.nan)).to_numpy(dtype=float)


def compare_policies(
    baseline_log: pd.DataFrame,
    candidate_log: pd.DataFrame,
    *,
    n_boot: int = 2000,
    n_perm: int = 2000,
    seed: int = 42,
) -> dict[str, Any]:
    a = _trade_returns(baseline_log)
    b = _trade_returns(candidate_log)
    a = a[np.isfinite(a)]
    b = b[np.isfinite(b)]
    out: dict[str, Any] = {
        "n_base": int(len(a)),
        "n_cand": int(len(b)),
        "mean_base": float(np.mean(a)) if len(a) else float("nan"),
        "mean_cand": float(np.mean(b)) if len(b) else float("nan"),
    }
    if len(a) < 5 or len(b) < 5:
        out.update({"mannwhitney_p": float("nan"), "perm_p": float("nan"), "boot_ci_low": float("nan"), "boot_ci_high": float("nan")})
        return out

    # Mann-Whitney: candidate trade returns greater?
    try:
        u = stats.mannwhitneyu(b, a, alternative="greater")
        out["mannwhitney_u"] = float(u.statistic)
        out["mannwhitney_p"] = float(u.pvalue)
    except Exception:
        out["mannwhitney_p"] = float("nan")

    rng = np.random.default_rng(seed)
    delta = float(np.mean(b) - np.mean(a))
    # Bootstrap CI on mean difference (resample each)
    diffs = []
    for _ in range(n_boot):
        sa = rng.choice(a, size=len(a), replace=True)
        sb = rng.choice(b, size=len(b), replace=True)
        diffs.append(float(np.mean(sb) - np.mean(sa)))
    diffs_a = np.asarray(diffs)
    out["boot_mean_diff"] = delta
    out["boot_ci_low"] = float(np.quantile(diffs_a, 0.025))
    out["boot_ci_high"] = float(np.quantile(diffs_a, 0.975))

    # Permutation test on mean difference
    pooled = np.concatenate([a, b])
    n_b = len(b)
    extreme = 0
    for _ in range(n_perm):
        rng.shuffle(pooled)
        d = float(np.mean(pooled[:n_b]) - np.mean(pooled[n_b:]))
        if abs(d) >= abs(delta):
            extreme += 1
    out["perm_p"] = float((extreme + 1) / (n_perm + 1))
    out["mean_diff"] = delta
    return out
