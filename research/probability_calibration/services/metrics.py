"""Trade / calibration metric helpers."""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics import brier_score_loss, log_loss

from research.feature_ablation.services.walk_forward_runner import expected_calibration_error

PERCENTILES: tuple[float, ...] = (0.01, 0.03, 0.05, 0.10, 0.15, 0.20, 0.30)

CONTEXT_FILTER_FEATURES: tuple[str, ...] = (
    "ctx_h4_swing_quality",
    "ctx_h4_distance_from_equilibrium",
    "ctx_h4_rejection_strength",
)


def apply_cost(returns: np.ndarray, cost: float) -> np.ndarray:
    """Haircut each trade by fixed fractional cost (research assumption)."""
    r = np.asarray(returns, dtype=float)
    return r - float(cost)


def profit_factor(returns: np.ndarray) -> float:
    r = np.asarray(returns, dtype=float)
    r = r[np.isfinite(r)]
    if len(r) == 0:
        return float("nan")
    gains = r[r > 0].sum()
    losses = -r[r < 0].sum()
    if losses <= 0:
        return float("inf") if gains > 0 else float("nan")
    return float(gains / losses)


def max_drawdown(returns: np.ndarray) -> float:
    """Max peak-to-trough drawdown of cumulative return (fraction)."""
    r = np.asarray(returns, dtype=float)
    r = r[np.isfinite(r)]
    if len(r) == 0:
        return float("nan")
    equity = np.cumsum(r)
    peak = np.maximum.accumulate(equity)
    dd = peak - equity
    return float(dd.max()) if len(dd) else float("nan")


def trade_stats(returns: np.ndarray, y_true: np.ndarray, *, cost: float) -> dict[str, float]:
    net = apply_cost(returns, cost)
    y = np.asarray(y_true, dtype=int)
    n = int(len(net))
    if n == 0:
        nan = float("nan")
        return {
            "n_trades": 0,
            "win_rate": nan,
            "avg_return": nan,
            "expectancy": nan,
            "profit_factor": nan,
            "max_drawdown": nan,
        }
    return {
        "n_trades": n,
        "win_rate": float(np.mean(y)),
        "avg_return": float(np.mean(net)),
        "expectancy": float(np.mean(net)),
        "profit_factor": profit_factor(net),
        "max_drawdown": max_drawdown(net),
    }


def calibration_scores(y_true: np.ndarray, y_prob: np.ndarray) -> dict[str, float]:
    y = np.asarray(y_true, dtype=int).ravel()
    p = np.clip(np.asarray(y_prob, dtype=float).ravel(), 1e-6, 1.0 - 1e-6)
    if len(y) == 0 or len(np.unique(y)) < 2:
        nan = float("nan")
        return {"brier": nan, "log_loss": nan, "ece": nan}
    return {
        "brier": float(brier_score_loss(y, p)),
        "log_loss": float(log_loss(y, p, labels=[0, 1])),
        "ece": float(expected_calibration_error(y, p)),
    }


def select_top_percentile(frame: pd.DataFrame, prob_col: str, pct: float) -> pd.DataFrame:
    """Keep top pct fraction by probability (pct=0.05 → top 5%)."""
    if frame.empty or pct <= 0:
        return frame.iloc[0:0].copy()
    n = max(1, int(np.ceil(len(frame) * pct)))
    return frame.nlargest(n, prob_col)
