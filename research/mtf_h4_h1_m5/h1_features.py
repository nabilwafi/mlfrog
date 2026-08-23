"""H1 feature sets for hierarchical MTF experiments."""

from __future__ import annotations

# FEAT7 minus ctx_h4_swing_quality — H1-native baseline (Experiment A).
H1_NATIVE: tuple[str, ...] = (
    "hour_cos",
    "atr_percentile_252",
    "ema_trend_duration",
    "rolling_quantile",
    "hour_sin",
    "atr_percent",
)

from research.mtf_h4_h1_m5.contracts import D1_SIGNAL_FEATURE_NAMES, H4_SIGNAL_FEATURE_NAMES

# Experiment B: H1 native + abstract H4 signal (no raw ctx_h4_*).
H1_WITH_H4_SIGNAL: tuple[str, ...] = H1_NATIVE + H4_SIGNAL_FEATURE_NAMES

# Experiment D1: H1 native + abstract D1 signal.
H1_WITH_D1_SIGNAL: tuple[str, ...] = H1_NATIVE + D1_SIGNAL_FEATURE_NAMES

# Production reference (not hierarchical — includes raw ctx merge).
PRODUCTION_FEAT7: tuple[str, ...] = H1_NATIVE + ("ctx_h4_swing_quality",)

# All ctx_h4_* currently stored in dataset v2.
CTX_H4_V2: tuple[str, ...] = (
    "ctx_h4_compression",
    "ctx_h4_expansion",
    "ctx_h4_market_regime",
    "ctx_h4_swing_quality",
    "ctx_h4_trend_direction",
    "ctx_h4_trend_strength",
    "ctx_h4_volatility_regime",
)
