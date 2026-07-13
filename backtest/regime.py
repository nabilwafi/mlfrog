"""
Causal ATR z-score regime tiers (no full-history lookahead).

At time T:
  past_mean/std = rolling stats of atr over the previous W bars (exclude T via shift(1))
  z[T] = (atr[T] - past_mean[T]) / past_std[T]
"""

from __future__ import annotations

import numpy as np
import pandas as pd

ROLLING_WINDOW_BARS = 24 * 180  # ~180 calendar days of H1 bars
Z_REDUCE_THRESHOLD = 2.0
Z_BLOCK_THRESHOLD = 3.5


def compute_causal_atr_zscore(
    atr: pd.Series,
    *,
    window: int = ROLLING_WINDOW_BARS,
) -> pd.Series:
    """Rolling mean/std of *past* ATR only, then z of current ATR vs that baseline."""
    # shift(1): at T, stats use atr[T-window ... T-1], never T or future.
    past_mean = atr.rolling(window=window, min_periods=window).mean().shift(1)
    past_std = atr.rolling(window=window, min_periods=window).std(ddof=0).shift(1)
    z = (atr - past_mean) / past_std.replace(0, np.nan)
    return z


def assign_regime_tier(
    z: pd.Series,
    *,
    z_reduce: float = Z_REDUCE_THRESHOLD,
    z_block: float = Z_BLOCK_THRESHOLD,
) -> pd.Series:
    tier = pd.Series("normal", index=z.index, dtype=object)
    # Undefined z (warm-up) stays normal — allow trading until baseline exists.
    known = z.notna()
    tier[known & (z >= z_reduce) & (z < z_block)] = "elevated"
    tier[known & (z >= z_block)] = "extreme"
    return tier


def attach_regime_to_frame(
    df: pd.DataFrame,
    atr_history: pd.DataFrame,
    *,
    window: int = ROLLING_WINDOW_BARS,
    z_reduce: float = Z_REDUCE_THRESHOLD,
    z_block: float = Z_BLOCK_THRESHOLD,
) -> pd.DataFrame:
    """Compute causal z on full atr_history (Date, atr_h1), merge onto backtest df."""
    hist = atr_history[["Date", "atr_h1"]].copy()
    hist["Date"] = pd.to_datetime(hist["Date"], utc=True)
    hist = hist.sort_values("Date").drop_duplicates("Date").reset_index(drop=True)
    hist["atr_zscore"] = compute_causal_atr_zscore(hist["atr_h1"], window=window)
    hist["regime_tier"] = assign_regime_tier(
        hist["atr_zscore"], z_reduce=z_reduce, z_block=z_block
    )

    out = df.copy()
    out["Date"] = pd.to_datetime(out["Date"], utc=True)
    # Drop if re-attaching
    out = out.drop(columns=[c for c in ("atr_zscore", "regime_tier") if c in out.columns])
    out = out.merge(hist[["Date", "atr_zscore", "regime_tier"]], on="Date", how="left")
    return out


def regime_time_share(df: pd.DataFrame) -> dict[str, float]:
    vc = df["regime_tier"].value_counts(normalize=True)
    return {
        "pct_normal": float(vc.get("normal", 0.0) * 100),
        "pct_elevated": float(vc.get("elevated", 0.0) * 100),
        "pct_extreme": float(vc.get("extreme", 0.0) * 100),
    }
