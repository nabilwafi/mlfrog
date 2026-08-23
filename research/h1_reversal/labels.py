"""Candidate reversal labels — training only; never used as live features.

All labels are position-aware. Future path is used ONLY to construct y.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np
import pandas as pd

Side = Literal["long", "short"]


@dataclass(frozen=True)
class BarrierReversalParams:
    """Candidate A — opposite barrier before continuation."""

    horizon_bars: int = 8  # H1 bars forward from decision bar
    adverse_atr: float = 1.0  # opposite move in ATR
    favor_atr: float = 1.0  # continuation barrier


@dataclass(frozen=True)
class ThesisFailureParams:
    """Candidate B — structure / adverse failure without requiring opposite TP."""

    horizon_bars: int = 8
    adverse_atr: float = 1.0
    min_mfe_atr: float = 0.25  # if never made this much MFE, softer thesis


def _atr_wilder(h: np.ndarray, l: np.ndarray, c: np.ndarray, n: int = 14) -> np.ndarray:
    prev = np.roll(c, 1)
    prev[0] = c[0]
    tr = np.maximum(h - l, np.maximum(np.abs(h - prev), np.abs(l - prev)))
    atr = np.full_like(c, np.nan, dtype=float)
    if len(c) < n:
        return atr
    atr[n - 1] = tr[:n].mean()
    alpha = 1.0 / n
    for i in range(n, len(c)):
        atr[i] = atr[i - 1] * (1 - alpha) + tr[i] * alpha
    return atr


def label_opposite_barrier(
    ohlc: pd.DataFrame,
    *,
    side: Side,
    decision_idx: np.ndarray,
    params: BarrierReversalParams | None = None,
) -> np.ndarray:
    """
    For each decision bar i (while hypothetically in `side`):
      y=1 if within horizon, adverse move ≥ adverse_atr * ATR_i
           before favorable move ≥ favor_atr * ATR_i
      y=0 otherwise (continuation / timeout / favor first)
    Causal for features at i; label uses future closes/highs/lows only.
    """
    p = params or BarrierReversalParams()
    h = ohlc["high"].astype(float).to_numpy()
    l = ohlc["low"].astype(float).to_numpy()
    c = ohlc["close"].astype(float).to_numpy()
    atr = _atr_wilder(h, l, c)
    n = len(c)
    y = np.full(len(decision_idx), np.nan)
    for k, i in enumerate(decision_idx):
        i = int(i)
        a = float(atr[i]) if i < n and np.isfinite(atr[i]) else np.nan
        if not np.isfinite(a) or a <= 0:
            continue
        entry = float(c[i])
        end = min(n, i + 1 + int(p.horizon_bars))
        hit_adv = hit_fav = False
        for j in range(i + 1, end):
            if side == "long":
                adv = (entry - float(l[j])) / a
                fav = (float(h[j]) - entry) / a
            else:
                adv = (float(h[j]) - entry) / a
                fav = (entry - float(l[j])) / a
            if (not hit_adv) and adv >= float(p.adverse_atr):
                hit_adv = True
            if (not hit_fav) and fav >= float(p.favor_atr):
                hit_fav = True
            if hit_adv and not hit_fav:
                y[k] = 1.0
                break
            if hit_fav and not hit_adv:
                y[k] = 0.0
                break
        if not np.isfinite(y[k]):
            y[k] = 0.0  # timeout / neither → not reversal
    return y


def label_thesis_failure(
    ohlc: pd.DataFrame,
    *,
    side: Side,
    decision_idx: np.ndarray,
    params: ThesisFailureParams | None = None,
) -> np.ndarray:
    """
    Candidate B: adverse ≥ adverse_atr within horizon AND peak favorable MFE < min_mfe_atr.
    Captures 'never confirmed then failed' vs 'pulled back after progress'.
    """
    p = params or ThesisFailureParams()
    h = ohlc["high"].astype(float).to_numpy()
    l = ohlc["low"].astype(float).to_numpy()
    c = ohlc["close"].astype(float).to_numpy()
    atr = _atr_wilder(h, l, c)
    n = len(c)
    y = np.zeros(len(decision_idx), dtype=float)
    for k, i in enumerate(decision_idx):
        i = int(i)
        a = float(atr[i]) if i < n and np.isfinite(atr[i]) else np.nan
        if not np.isfinite(a) or a <= 0:
            continue
        entry = float(c[i])
        end = min(n, i + 1 + int(p.horizon_bars))
        mfe = 0.0
        mae = 0.0
        for j in range(i + 1, end):
            if side == "long":
                mfe = max(mfe, (float(h[j]) - entry) / a)
                mae = max(mae, (entry - float(l[j])) / a)
            else:
                mfe = max(mfe, (entry - float(l[j])) / a)
                mae = max(mae, (float(h[j]) - entry) / a)
        if mae >= float(p.adverse_atr) and mfe < float(p.min_mfe_atr):
            y[k] = 1.0
    return y


LABEL_CANDIDATES = {
    "A_opposite_barrier": (
        "Opposite ATR barrier before continuation barrier (position-aware).",
        BarrierReversalParams,
    ),
    "B_thesis_failure": (
        "Adverse ATR with never-confirmed MFE (position-aware).",
        ThesisFailureParams,
    ),
    "C_mae_empirical": (
        "MAE exceeds IS quantile of winning-path MAE — params from train folds only.",
        None,
    ),
}
