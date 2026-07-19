"""Sprint 20 production Meta Model — Trade vs Skip (primary frozen)."""

FINAL_FEATURES: tuple[str, ...] = (
    "raw_probability",
    "probability_rank",
    "probability_percentile",
    "probability_margin",
    "ctx_h4_rejection_strength",
    "ctx_h4_swing_quality",
    "ctx_h4_swing_strength",
    "ctx_h4_distance_from_equilibrium",
    "ctx_h4_swing_high_distance_atr",
    "ctx_h4_swing_low_distance_atr",
    "rolling_quantile",
    "ema_trend_duration",
    "volatility_rank",
    "atr_percent",
    "hour_of_day",
    "day_of_week",
    "month",
    "rolling_volatility",
)

THRESHOLDS: tuple[float, ...] = (0.40, 0.45, 0.50, 0.55, 0.60)

# Report-reference gate only (not optimized)
REFERENCE_THRESHOLD: float = 0.50
