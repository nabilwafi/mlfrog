"""Deterministic D1 context engine — same contract pattern as H4, on daily bars."""

from __future__ import annotations

from typing import Any

import pandas as pd

from market_context.builders.h4_context_builder import H4ContextBuilder
from market_context.builders.h4_structure_builder import H4StructureBuilder
from market_context.services.timeframe_utils import available_at
from research.mtf_h4_h1_m5.contracts import (
    D1Signal,
    D1_SIGNAL_FEATURE_NAMES,
)
from research.mtf_h4_h1_m5.h4_engine import (
    _direction_from_trend,
    _strength_label,
    _structure_label,
    _trend_label,
    _vol_label,
    vol_terciles_from_train,
)


def build_d1_from_h1(h1: pd.DataFrame) -> pd.DataFrame:
    """Resample H1 → D1 OHLC (UTC calendar day, left-labeled open time)."""
    x = h1.sort_values("timestamp").copy()
    x["timestamp"] = pd.to_datetime(x["timestamp"], utc=True)
    x = x.set_index("timestamp")
    d1 = (
        x.resample("1D", label="left", closed="left")
        .agg(open=("open", "first"), high=("high", "max"), low=("low", "min"), close=("close", "last"))
        .dropna()
        .reset_index()
    )
    return d1


def build_d1_feature_frame(d1: pd.DataFrame) -> pd.DataFrame:
    """D1-native features via same deterministic builders as H4 (on D1 OHLC)."""
    d1 = d1.sort_values("timestamp").reset_index(drop=True)
    ctx, _ = H4ContextBuilder().build(d1)
    struct, _ = H4StructureBuilder().build(d1)
    out = ctx.merge(struct, on="timestamp", how="inner")
    out["available_timestamp"] = available_at(out["timestamp"], "D1")
    return out


def d1_bar_to_signal(row: pd.Series, *, vol_lo: float, vol_hi: float) -> D1Signal:
    trend_f = float(row["ctx_h4_trend_direction"])
    struct_f = float(row.get("ctx_h4_swing_direction", 0.0) or 0.0)
    vol_f = float(row["ctx_h4_volatility_regime"])
    str_f = float(row["ctx_h4_trend_strength"])
    trend = _trend_label(trend_f)
    return D1Signal(
        direction=_direction_from_trend(trend),
        trend=trend,
        structure=_structure_label(struct_f),
        volatility=_vol_label(vol_f, vol_lo=vol_lo, vol_hi=vol_hi),
        strength=_strength_label(str_f),
        signal_timestamp=pd.Timestamp(row["timestamp"]).isoformat(),
        available_timestamp=pd.Timestamp(row["available_timestamp"]).isoformat(),
    )


def build_d1_signal_table(
    d1_feats: pd.DataFrame,
    *,
    vol_lo: float,
    vol_hi: float,
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for _, row in d1_feats.iterrows():
        sig = d1_bar_to_signal(row, vol_lo=vol_lo, vol_hi=vol_hi)
        rec = sig.as_dict()
        rec.update(sig.to_features())
        rec["context_bar_timestamp"] = row["timestamp"]
        rows.append(rec)
    out = pd.DataFrame(rows)
    out["available_timestamp"] = pd.to_datetime(out["available_timestamp"], utc=True)
    out["context_bar_timestamp"] = pd.to_datetime(out["context_bar_timestamp"], utc=True)
    return out


def join_d1_signals_to_h1(
    h1_timestamps: pd.Series,
    d1_signals: pd.DataFrame,
) -> pd.DataFrame:
    join_cols = list(D1_SIGNAL_FEATURE_NAMES) + [
        "direction", "trend", "structure", "volatility", "strength",
        "signal_timestamp", "context_bar_timestamp",
    ]
    ctx = d1_signals[["available_timestamp", *join_cols]].copy()
    ctx = ctx.rename(columns={"available_timestamp": "available_at"})
    ctx = ctx.sort_values("available_at")
    base = pd.DataFrame({"timestamp": pd.to_datetime(h1_timestamps, utc=True)}).sort_values("timestamp")
    return pd.merge_asof(
        base, ctx, left_on="timestamp", right_on="available_at", direction="backward",
    )


def attach_d1_signals(
    df: pd.DataFrame,
    d1: pd.DataFrame,
    *,
    train_start: int,
    train_end: int,
) -> pd.DataFrame:
    d1_feats = build_d1_feature_frame(d1)
    vol_lo, vol_hi = vol_terciles_from_train(d1_feats, train_start, train_end)
    sig = build_d1_signal_table(d1_feats, vol_lo=vol_lo, vol_hi=vol_hi)
    joined = join_d1_signals_to_h1(df["timestamp"], sig)
    out = df.copy()
    out["timestamp"] = pd.to_datetime(out["timestamp"], utc=True)
    for c in D1_SIGNAL_FEATURE_NAMES:
        out[c] = joined[c].to_numpy()
    out["d1_available_timestamp"] = joined["available_at"]
    ok = joined["available_at"].isna() | (out["timestamp"] >= joined["available_at"])
    if not ok.all():
        raise ValueError(f"D1 signal leakage: {(~ok).sum()} H1 rows before available_at")
    return out


def assert_d1_causal_boundary() -> None:
    sig = pd.DataFrame(
        {
            "available_timestamp": [pd.Timestamp("2020-01-02T00:00:00Z")],
            "d1_sig_direction": [1.0],
            "d1_sig_trend": [1.0],
            "d1_sig_structure": [0.0],
            "d1_sig_vol": [1.0],
            "d1_sig_strength": [1.0],
            "direction": ["long"],
            "trend": ["bullish"],
            "structure": ["neutral"],
            "volatility": ["normal"],
            "strength": ["normal"],
            "signal_timestamp": ["2020-01-01T00:00:00+00:00"],
            "context_bar_timestamp": [pd.Timestamp("2020-01-01T00:00:00Z")],
        }
    )
    h1_ts = pd.Series([
        pd.Timestamp("2020-01-01T20:00:00Z"),
        pd.Timestamp("2020-01-02T00:00:00Z"),
    ])
    j = join_d1_signals_to_h1(h1_ts, sig)
    assert pd.isna(j.loc[0, "d1_sig_direction"])
    assert float(j.loc[1, "d1_sig_direction"]) == 1.0
