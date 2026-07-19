"""Causal D1 regime + M5 entry quality (no lookahead)."""

from __future__ import annotations

import numpy as np
import pandas as pd

from market_context.services.timeframe_utils import available_at


def build_d1_features(d1: pd.DataFrame) -> pd.DataFrame:
    """Features known only after D1 bar close."""
    x = d1.sort_values("timestamp").copy()
    x["timestamp"] = pd.to_datetime(x["timestamp"], utc=True)
    c = x["close"].astype(float)
    h = x["high"].astype(float)
    l = x["low"].astype(float)
    # ATR14
    prev = c.shift(1)
    tr = pd.concat([(h - l), (h - prev).abs(), (l - prev).abs()], axis=1).max(axis=1)
    atr = tr.rolling(14, min_periods=5).mean()
    ema20 = c.ewm(span=20, adjust=False).mean()
    ema50 = c.ewm(span=50, adjust=False).mean()
    slope = ema20.diff(5) / (c.abs() + 1e-9)
    atr_pct = atr / (c.abs() + 1e-9)
    atr_z = (atr_pct - atr_pct.rolling(60, min_periods=20).mean()) / (
        atr_pct.rolling(60, min_periods=20).std() + 1e-9
    )
    # Trend strength: ema20 vs ema50 distance in ATR
    trend_dist = (ema20 - ema50) / (atr + 1e-9)
    range_comp = (h - l) / (atr.rolling(20, min_periods=5).mean() + 1e-9)

    out = pd.DataFrame(
        {
            "timestamp": x["timestamp"],
            "available_at": available_at(x["timestamp"], "D1"),
            "d1_ema_slope": slope,
            "d1_trend_dist": trend_dist,
            "d1_atr_z": atr_z,
            "d1_range_comp": range_comp,
            "d1_close": c,
        }
    )
    out["d1_regime"] = _classify_regime(out)
    return out.dropna(subset=["d1_regime"]).reset_index(drop=True)


def _classify_regime(frame: pd.DataFrame) -> pd.Series:
    slope = frame["d1_ema_slope"].to_numpy(dtype=float)
    dist = frame["d1_trend_dist"].to_numpy(dtype=float)
    atr_z = frame["d1_atr_z"].to_numpy(dtype=float)
    labels = []
    for s, d, z in zip(slope, dist, atr_z):
        if not np.isfinite(s) or not np.isfinite(d):
            labels.append("Sideways")
            continue
        # Compression: low ATR z and flat distance
        if np.isfinite(z) and z < -0.75 and abs(d) < 0.5:
            labels.append("Compression")
            continue
        if d >= 1.0 and s > 0:
            labels.append("Strong Bull")
        elif d >= 0.25 and s > 0:
            labels.append("Weak Bull")
        elif d <= -1.0 and s < 0:
            labels.append("Strong Bear")
        elif d <= -0.25 and s < 0:
            labels.append("Weak Bear")
        else:
            labels.append("Sideways")
    return pd.Series(labels, index=frame.index)


def build_m5_entry_quality(m5: pd.DataFrame) -> pd.DataFrame:
    """M5 execution-quality features; known after M5 bar close."""
    x = m5.sort_values("timestamp").copy()
    x["timestamp"] = pd.to_datetime(x["timestamp"], utc=True)
    o = x["open"].astype(float)
    h = x["high"].astype(float)
    l = x["low"].astype(float)
    c = x["close"].astype(float)
    rng = (h - l).replace(0, np.nan)
    body = (c - o).abs()
    body_ratio = (body / rng).clip(0, 1).fillna(0)
    prev_c = c.shift(1)
    tr = pd.concat([(h - l), (h - prev_c).abs(), (l - prev_c).abs()], axis=1).max(axis=1)
    atr = tr.rolling(14, min_periods=5).mean()
    atr_exp = atr / (atr.rolling(48, min_periods=10).mean() + 1e-9)
    # Local swing distance (lookback 12 bars ~1h)
    roll_high = h.rolling(12, min_periods=3).max()
    roll_low = l.rolling(12, min_periods=3).min()
    dist_swing = pd.concat([(roll_high - c).abs(), (c - roll_low).abs()], axis=1).min(axis=1) / (
        atr + 1e-9
    )
    vol_burst = tr / (tr.rolling(48, min_periods=10).mean() + 1e-9)
    spread = x["spread"].astype(float) if "spread" in x.columns else pd.Series(0.0, index=x.index)

    # Composite 0-100 (deterministic, no learning)
    # Higher body, moderate ATR expansion, not too far from swing, not insane burst
    score = (
        40.0 * body_ratio.fillna(0)
        + 25.0 * ((atr_exp.clip(0.5, 2.0) - 0.5) / 1.5).clip(0, 1).fillna(0)
        + 20.0 * (1.0 - dist_swing.clip(0, 3) / 3.0).fillna(0)
        + 15.0 * (1.0 - (vol_burst.clip(0, 3) - 1.0).abs().clip(0, 2) / 2.0).fillna(0)
    )
    score = score.clip(0, 100)

    return pd.DataFrame(
        {
            "timestamp": x["timestamp"],
            "available_at": available_at(x["timestamp"], "M5"),
            "m5_body_ratio": body_ratio,
            "m5_atr_expansion": atr_exp,
            "m5_dist_swing_atr": dist_swing,
            "m5_vol_burst": vol_burst,
            "m5_spread": spread,
            "m5_entry_quality": score,
        }
    )


def asof_join(
    trades: pd.DataFrame,
    ctx: pd.DataFrame,
    cols: list[str],
) -> pd.DataFrame:
    """Causal asof: trade timestamp must be >= ctx available_at."""
    left = trades.sort_values("timestamp").copy()
    left["timestamp"] = pd.to_datetime(left["timestamp"], utc=True)
    right = ctx.sort_values("available_at").copy()
    right["available_at"] = pd.to_datetime(right["available_at"], utc=True)
    keep = ["available_at"] + cols
    merged = pd.merge_asof(
        left,
        right[keep],
        left_on="timestamp",
        right_on="available_at",
        direction="backward",
    )
    return merged.drop(columns=["available_at"], errors="ignore")
