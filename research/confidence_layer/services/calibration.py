"""Confidence calibration metrics."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
from sklearn.metrics import brier_score_loss


def calibration_report(panel: pd.DataFrame, *, n_bins: int = 10) -> tuple[pd.DataFrame, dict[str, float]]:
    """Reliability of confidence/100 vs meta_label."""
    p = np.clip(panel["confidence"].astype(float).to_numpy() / 100.0, 1e-6, 1 - 1e-6)
    y = panel["meta_label"].astype(int).to_numpy()
    if len(np.unique(y)) < 2:
        return pd.DataFrame(), {"ece": float("nan"), "mce": float("nan"), "brier": float("nan")}

    bins = np.linspace(0, 1, n_bins + 1)
    rows = []
    ece = 0.0
    mce = 0.0
    for i in range(n_bins):
        lo, hi = bins[i], bins[i + 1]
        mask = (p >= lo) & (p < hi) if i < n_bins - 1 else (p >= lo) & (p <= hi)
        if not mask.any():
            continue
        conf = float(p[mask].mean())
        acc = float(y[mask].mean())
        gap = abs(acc - conf)
        ece += gap * (mask.sum() / len(y))
        mce = max(mce, gap)
        rows.append(
            {
                "bin_lo": lo,
                "bin_hi": hi,
                "n": int(mask.sum()),
                "avg_confidence": conf,
                "empirical_win_rate": acc,
                "gap": gap,
            }
        )
    summary = {
        "ece": float(ece),
        "mce": float(mce),
        "brier": float(brier_score_loss(y, p)),
    }
    return pd.DataFrame(rows), summary
