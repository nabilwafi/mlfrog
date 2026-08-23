"""H1 native feature selection research (38-feature universe).

Excludes ctx_h4_*, d1_*, m5_*, session flags.
"""

from __future__ import annotations

from research.mtf_h4_h1_m5.h1_features import H1_NATIVE

BASELINE_FEATURE_SET: tuple[str, ...] = H1_NATIVE

GROUP_FEATURES: dict[str, tuple[str, ...]] = {
    "trend": (
        "ema20_distance_atr",
        "ema50_distance_atr",
        "ema20_distance_percent",
        "ema50_distance_percent",
        "ema_cross_distance",
        "ema20_slope",
        "ema50_slope",
        "ema_trend_duration",
        "ema_alignment_score",
    ),
    "volatility": (
        "atr_percent",
        "atr_percentile_252",
        "rolling_volatility",
        "volatility_rank",
        "volatility_regime_score",
    ),
    "momentum": (
        "macd_normalized",
        "macd_histogram_zscore",
        "macd_signal_distance",
        "momentum_rank",
        "rsi_percentile",
        "roc_3",
        "roc_12",
        "momentum_acceleration",
    ),
    "candle": (
        "body_percent",
        "upper_wick_percent",
        "lower_wick_percent",
        "close_position",
        "range_percent",
        "body_rank",
    ),
    "session": (
        "hour_sin",
        "hour_cos",
        "day_sin",
        "day_cos",
    ),
    "statistical": (
        "rolling_zscore",
        "rolling_percentile",
        "rolling_rank",
        "rolling_quantile",
        "rolling_std",
        "rolling_mean_distance",
    ),
}

# Candidates = group members not already in CORE baseline.
CANDIDATE_BY_GROUP: dict[str, tuple[str, ...]] = {
    "trend": (
        "ema20_distance_atr",
        "ema50_distance_atr",
        "ema20_distance_percent",
        "ema50_distance_percent",
        "ema_cross_distance",
        "ema20_slope",
        "ema50_slope",
        "ema_alignment_score",
    ),
    "volatility": (
        "rolling_volatility",
        "volatility_rank",
        "volatility_regime_score",
    ),
    "momentum": GROUP_FEATURES["momentum"],
    "candle": GROUP_FEATURES["candle"],
    "session": ("day_sin", "day_cos"),
    "statistical": (
        "rolling_zscore",
        "rolling_percentile",
        "rolling_rank",
        "rolling_std",
        "rolling_mean_distance",
    ),
}

H1_ENGINE_38: tuple[str, ...] = (
    GROUP_FEATURES["trend"]
    + GROUP_FEATURES["volatility"]
    + GROUP_FEATURES["momentum"]
    + GROUP_FEATURES["candle"]
    + GROUP_FEATURES["session"]
    + GROUP_FEATURES["statistical"]
)

FEATURE_GROUP: dict[str, str] = {
    f: g for g, feats in GROUP_FEATURES.items() for f in feats
}

EXCLUDED_PREFIXES: tuple[str, ...] = ("ctx_h4_", "d1_", "m5_", "session_")
EXCLUDED_EXACT: tuple[str, ...] = (
    "session_london",
    "session_newyork",
    "session_london_ny_overlap",
    "session_asia",
    "hour_of_day",
    "day_of_week",
    "month",
)


def feat_union(base: tuple[str, ...] | list[str], extra: tuple[str, ...] | list[str]) -> list[str]:
    out = list(base)
    for f in extra:
        if f not in out:
            out.append(f)
    return out


def feat_without(universe: tuple[str, ...] | list[str], drop: tuple[str, ...] | list[str]) -> list[str]:
    drop_set = set(drop)
    return [f for f in universe if f not in drop_set]


def core_plus_group(group: str) -> list[str]:
    return feat_union(BASELINE_FEATURE_SET, CANDIDATE_BY_GROUP[group])


def all_minus_group(group: str) -> list[str]:
    """Leave-one-group-out on full 38, but keep CORE members of that group."""
    # Remove only the *candidate* extras of the group from ALL 38.
    # For groups that contribute CORE features, CORE stays (task: remove Trend/Vol/... information groups).
    # Spec: start ALL 38, remove Trend / Volatility / ... — remove entire group columns.
    return feat_without(H1_ENGINE_38, GROUP_FEATURES[group])
