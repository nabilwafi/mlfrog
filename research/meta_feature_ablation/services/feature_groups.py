"""Feature group definitions for meta ablation."""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import pandas as pd

# Baseline primary scores (Sprint 18 / brief)
GROUP_A_PRIMARY: tuple[str, ...] = (
    "raw_probability",
    "probability_rank",
    "probability_percentile",
    "probability_margin",
)

# Prefer the Sprint-18 recommended H1 subset + common vols (ponytail: not full 35)
GROUP_B_H1: tuple[str, ...] = (
    "rolling_quantile",
    "rolling_std",
    "rolling_volatility",
    "ema_trend_duration",
    "volatility_rank",
    "ema_alignment_score",
    "rsi_percentile",
    "atr_percent",
    "momentum_rank",
)

GROUP_C_H4: tuple[str, ...] = (
    "ctx_h4_rejection_strength",
    "ctx_h4_swing_quality",
    "ctx_h4_swing_strength",
    "ctx_h4_distance_from_equilibrium",
    "ctx_h4_premium_discount_zone",
    "ctx_h4_h4_range_position",
    "ctx_h4_swing_high_distance_atr",
    "ctx_h4_swing_low_distance_atr",
)

GROUP_E_SESSION: tuple[str, ...] = (
    "hour_of_day",
    "day_of_week",
    "month",
    "session_london",
    "session_newyork",
    "session_london_ny_overlap",
    "session_asia",
)

GROUP_F_ENTRY: tuple[str, ...] = (
    # distance_to_sl/tp are exact linear copies of atr_percent under fixed SL/TP ATR mults
    "atr_percent",
    "atr_entry",
)

# Sequential ablation stages (identical model, growing features)
ABLATION_STAGES: tuple[tuple[str, tuple[str, ...]], ...] = (
    ("A_primary", GROUP_A_PRIMARY),
    ("A_plus_H4", GROUP_A_PRIMARY + GROUP_C_H4),
    ("A_H4_H1", GROUP_A_PRIMARY + GROUP_C_H4 + GROUP_B_H1),
    ("A_H4_H1_Session", GROUP_A_PRIMARY + GROUP_C_H4 + GROUP_B_H1 + GROUP_E_SESSION),
    (
        "A_H4_H1_Session_Entry",
        tuple(dict.fromkeys(GROUP_A_PRIMARY + GROUP_C_H4 + GROUP_B_H1 + GROUP_E_SESSION + GROUP_F_ENTRY)),
    ),
)


def available_features(panel_columns: list[str], wanted: tuple[str, ...]) -> list[str]:
    cols = set(panel_columns)
    return [f for f in wanted if f in cols]


def drop_near_constant(frame: "pd.DataFrame", features: list[str], *, min_std: float = 1e-12) -> list[str]:
    """Exclude constant / degenerate columns from modeling + VIF."""
    keep = []
    for f in features:
        if f not in frame.columns:
            continue
        s = frame[f].astype(float)
        if s.nunique(dropna=True) < 2:
            continue
        if float(s.std(ddof=0)) <= min_std:
            continue
        keep.append(f)
    return keep
