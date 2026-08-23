"""Deterministic H4 context engine — no ML."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from market_context.builders.h4_context_builder import H4ContextBuilder
from market_context.builders.h4_structure_builder import H4StructureBuilder
from market_context.services.context_join_service import ContextJoinService
from market_context.services.timeframe_utils import available_at
from research.mtf_h4_h1_m5.contracts import (
    H4Direction,
    H4Signal,
    H4Strength,
    H4Structure,
    H4Trend,
    H4Volatility,
    H4_SIGNAL_FEATURE_NAMES,
)

# Reuse existing regime cutpoints from simulation.wf.sim.label_regime
TREND_NEUTRAL_BAND = 0.33
STRENGTH_WEAK = 0.33
STRENGTH_STRONG = 0.66


def _trend_label(v: float) -> H4Trend:
    if v > TREND_NEUTRAL_BAND:
        return "bullish"
    if v < -TREND_NEUTRAL_BAND:
        return "bearish"
    return "neutral"


def _structure_label(v: float) -> H4Structure:
    if v > 0.25:
        return "bullish"
    if v < -0.25:
        return "bearish"
    return "neutral"


def _vol_label(v: float, *, vol_lo: float, vol_hi: float) -> H4Volatility:
    if v <= vol_lo:
        return "low"
    if v >= vol_hi:
        return "high"
    return "normal"


def _strength_label(v: float) -> H4Strength:
    if v < STRENGTH_WEAK:
        return "weak"
    if v > STRENGTH_STRONG:
        return "strong"
    return "normal"


def _direction_from_trend(trend: H4Trend) -> H4Direction:
    if trend == "bullish":
        return "long"
    if trend == "bearish":
        return "short"
    return "neutral"


def build_h4_feature_frame(h4: pd.DataFrame) -> pd.DataFrame:
    """H4-native features only (context + structure builders on H4 OHLC)."""
    h4 = h4.sort_values("timestamp").reset_index(drop=True)
    ctx, _ = H4ContextBuilder().build(h4)
    struct, _ = H4StructureBuilder().build(h4)
    out = ctx.merge(struct, on="timestamp", how="inner")
    out["available_timestamp"] = available_at(out["timestamp"], "H4")
    return out


def vol_terciles_from_train(h4_feats: pd.DataFrame, train_start: int, train_end: int) -> tuple[float, float]:
    """Train-only H4 vol terciles (no test peek)."""
    y = pd.to_datetime(h4_feats["timestamp"], utc=True).dt.year
    train = h4_feats[(y >= train_start) & (y <= train_end)]
    vol = train["ctx_h4_volatility_regime"].astype(float)
    lo, hi = vol.quantile([1 / 3, 2 / 3])
    return float(lo), float(hi)


def h4_bar_to_signal(row: pd.Series, *, vol_lo: float, vol_hi: float) -> H4Signal:
    trend_f = float(row["ctx_h4_trend_direction"])
    struct_f = float(row.get("ctx_h4_swing_direction", 0.0) or 0.0)
    vol_f = float(row["ctx_h4_volatility_regime"])
    str_f = float(row["ctx_h4_trend_strength"])
    trend = _trend_label(trend_f)
    return H4Signal(
        direction=_direction_from_trend(trend),
        trend=trend,
        structure=_structure_label(struct_f),
        volatility=_vol_label(vol_f, vol_lo=vol_lo, vol_hi=vol_hi),
        strength=_strength_label(str_f),
        signal_timestamp=pd.Timestamp(row["timestamp"]).isoformat(),
        available_timestamp=pd.Timestamp(row["available_timestamp"]).isoformat(),
    )


def build_h4_signal_table(
    h4_feats: pd.DataFrame,
    *,
    vol_lo: float,
    vol_hi: float,
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for _, row in h4_feats.iterrows():
        sig = h4_bar_to_signal(row, vol_lo=vol_lo, vol_hi=vol_hi)
        rec = sig.as_dict()
        rec.update(sig.to_features())
        rec["context_bar_timestamp"] = row["timestamp"]
        rows.append(rec)
    out = pd.DataFrame(rows)
    out["available_timestamp"] = pd.to_datetime(out["available_timestamp"], utc=True)
    out["context_bar_timestamp"] = pd.to_datetime(out["context_bar_timestamp"], utc=True)
    return out


def join_h4_signals_to_h1(
    h1_timestamps: pd.Series,
    h4_signals: pd.DataFrame,
) -> pd.DataFrame:
    """Causal join: H1 bar sees last H4 signal whose available_at <= H1 timestamp."""
    join_cols = list(H4_SIGNAL_FEATURE_NAMES) + [
        "direction",
        "trend",
        "structure",
        "volatility",
        "strength",
        "signal_timestamp",
        "context_bar_timestamp",
    ]
    ctx = h4_signals[["available_timestamp", *join_cols]].copy()
    ctx = ctx.rename(columns={"available_timestamp": "available_at"})
    ctx = ctx.sort_values("available_at")
    base = pd.DataFrame({"timestamp": pd.to_datetime(h1_timestamps, utc=True)}).sort_values("timestamp")
    merged = pd.merge_asof(
        base,
        ctx,
        left_on="timestamp",
        right_on="available_at",
        direction="backward",
    )
    return merged


def attach_h4_signals(
    df: pd.DataFrame,
    h4: pd.DataFrame,
    *,
    train_start: int,
    train_end: int,
) -> pd.DataFrame:
    """Add H4 signal columns to H1-side dataframe (train vol terciles only)."""
    h4_feats = build_h4_feature_frame(h4)
    vol_lo, vol_hi = vol_terciles_from_train(h4_feats, train_start, train_end)
    sig = build_h4_signal_table(h4_feats, vol_lo=vol_lo, vol_hi=vol_hi)
    joined = join_h4_signals_to_h1(df["timestamp"], sig)
    out = df.copy()
    out["timestamp"] = pd.to_datetime(out["timestamp"], utc=True)
    for c in H4_SIGNAL_FEATURE_NAMES:
        out[c] = joined[c].to_numpy()
    out["h4_available_timestamp"] = joined["available_at"]
    # Leakage guard: H1 ts must be >= H4 available_at when signal present
    ok = joined["available_at"].isna() | (out["timestamp"] >= joined["available_at"])
    if not ok.all():
        bad = (~ok).sum()
        raise ValueError(f"H4 signal leakage: {bad} H1 rows before available_at")
    return out


def assert_h4_causal_boundary() -> None:
    """Regression: H4 bar 12:00 available at 16:00 — H1@15:00 must not see it."""
    sig = pd.DataFrame(
        {
            "available_timestamp": [pd.Timestamp("2020-01-01T16:00:00Z")],
            "h4_sig_direction": [1.0],
            "h4_sig_trend": [1.0],
            "h4_sig_structure": [0.0],
            "h4_sig_vol": [1.0],
            "h4_sig_strength": [1.0],
            "direction": ["long"],
            "trend": ["bullish"],
            "structure": ["neutral"],
            "volatility": ["normal"],
            "strength": ["normal"],
            "signal_timestamp": ["2020-01-01T12:00:00+00:00"],
            "context_bar_timestamp": [pd.Timestamp("2020-01-01T12:00:00Z")],
        }
    )
    h1_ts = pd.Series(
        [
            pd.Timestamp("2020-01-01T13:00:00Z"),
            pd.Timestamp("2020-01-01T16:00:00Z"),
        ]
    )
    j = join_h4_signals_to_h1(h1_ts, sig)
    assert pd.isna(j.loc[0, "h4_sig_direction"])
    assert float(j.loc[1, "h4_sig_direction"]) == 1.0
