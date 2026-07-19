"""Causal ATR regime math + attach helpers (logic unchanged from backtest.regime)."""

from __future__ import annotations

import numpy as np
import pandas as pd

from settings.strategy import ROLLING_WINDOW_BARS, Z_BLOCK, Z_REDUCE

# Historical default for assign_regime_tier(z_block=...) when callers omit kwargs.
# Settled operational block is settings.strategy.Z_BLOCK (4.0).
Z_BLOCK_THRESHOLD = 3.5
Z_REDUCE_THRESHOLD = Z_REDUCE


def compute_causal_atr_zscore(
    atr: pd.Series,
    *,
    window: int = ROLLING_WINDOW_BARS,
) -> pd.Series:
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
    hist = atr_history[["Date", "atr_h1"]].copy()
    hist["Date"] = pd.to_datetime(hist["Date"], utc=True)
    hist = hist.sort_values("Date").drop_duplicates("Date").reset_index(drop=True)
    hist["atr_zscore"] = compute_causal_atr_zscore(hist["atr_h1"], window=window)
    hist["regime_tier"] = assign_regime_tier(
        hist["atr_zscore"], z_reduce=z_reduce, z_block=z_block
    )
    out = df.copy()
    out["Date"] = pd.to_datetime(out["Date"], utc=True)
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


# Re-export settled constant for apps that want Gate-4 default explicitly
__all__ = [
    "ROLLING_WINDOW_BARS",
    "Z_REDUCE_THRESHOLD",
    "Z_BLOCK_THRESHOLD",
    "Z_BLOCK",
    "Z_REDUCE",
    "compute_causal_atr_zscore",
    "assign_regime_tier",
    "attach_regime_to_frame",
    "regime_time_share",
]
