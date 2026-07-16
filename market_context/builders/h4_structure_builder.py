"""H4 market-structure feature builder (Sprint 11).

Causal only: every transform uses bars up to and including the current open-time bar.
Availability for H1 join remains open_time + bar_duration (handled by ContextJoinService).
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from feature_engineering.builders._transforms import atr, safe_div
from market_context.entities.context_feature import ContextFeatureSpec


class H4StructureBuilder:
    """Richer H4 structure signals — swings, BOS, liquidity, price location."""

    PREFIX = "ctx_h4_"

    def __init__(self, params: dict[str, Any] | None = None) -> None:
        self.params = dict(params or {})

    def build(self, frame: pd.DataFrame) -> tuple[pd.DataFrame, list[ContextFeatureSpec]]:
        if frame.empty:
            raise ValueError("empty H4 frame")
        required = {"timestamp", "open", "high", "low", "close"}
        missing = required - set(frame.columns)
        if missing:
            raise ValueError(f"H4 frame missing columns: {sorted(missing)}")

        df = frame.sort_values("timestamp").reset_index(drop=True).copy()
        open_ = df["open"].astype(float)
        high = df["high"].astype(float)
        low = df["low"].astype(float)
        close = df["close"].astype(float)

        atr_n = int(self.params.get("atr_period", 14))
        swing_n = int(self.params.get("swing_window", 5))
        range_n = int(self.params.get("range_window", 20))
        atr14 = atr(high, low, close, atr_n)
        bar_range = (high - low).astype(float)

        # --- Causal swing levels: rolling extrema ending at t (no future bars) ---
        swing_high = high.rolling(swing_n, min_periods=swing_n).max()
        swing_low = low.rolling(swing_n, min_periods=swing_n).min()
        prev_swing_high = swing_high.shift(1)
        prev_swing_low = swing_low.shift(1)
        # Prior swing window for HH/LL comparison
        older_swing_high = swing_high.shift(swing_n)
        older_swing_low = swing_low.shift(swing_n)

        # 1. Swing Structure
        swing_high_distance_atr = safe_div(close - swing_high, atr14)
        swing_low_distance_atr = safe_div(close - swing_low, atr14)
        higher_high_state = (swing_high > older_swing_high).astype(float)
        lower_low_state = (swing_low < older_swing_low).astype(float)
        # Direction: +1 HH without LL, -1 LL without HH, else mixed/neutral
        swing_direction = higher_high_state - lower_low_state
        swing_span = safe_div(swing_high - swing_low, atr14)
        swing_strength = swing_span.clip(upper=5.0) / 5.0

        # Legacy swing_quality (kept for ablation continuity)
        body = (close - open_).abs()
        body_share = safe_div(body, bar_range.replace(0.0, np.nan))
        swing_quality = (
            0.5 * swing_span.clip(upper=3.0) / 3.0 + 0.5 * body_share.clip(0.0, 1.0)
        ).clip(0.0, 1.0)

        # 2. Break Of Structure (vs prior confirmed swing level)
        bullish_structure_break = (close > prev_swing_high).astype(float)
        bearish_structure_break = (close < prev_swing_low).astype(float)
        # Signed distance to nearest broken level (bullish +, bearish -)
        dist_bull = safe_div(close - prev_swing_high, atr14)
        dist_bear = safe_div(prev_swing_low - close, atr14)
        distance_from_structure_break = np.where(
            bullish_structure_break.to_numpy() > 0.5,
            dist_bull.to_numpy(),
            np.where(
                bearish_structure_break.to_numpy() > 0.5,
                -dist_bear.to_numpy(),
                0.0,
            ),
        )
        distance_from_structure_break = pd.Series(
            distance_from_structure_break, index=df.index, dtype=float
        )

        # 3. Liquidity Structure (sweep prior swing then close back inside)
        high_sweep_detection = (
            (high > prev_swing_high) & (close < prev_swing_high)
        ).astype(float)
        low_sweep_detection = (
            (low < prev_swing_low) & (close > prev_swing_low)
        ).astype(float)
        upper_wick = high - np.maximum(open_, close)
        lower_wick = np.minimum(open_, close) - low
        rejection_strength = safe_div(
            np.maximum(upper_wick, lower_wick), bar_range.replace(0.0, np.nan)
        ).clip(0.0, 1.0)
        # Wick reversal: upper wick dominance after high sweep, lower after low sweep
        wick_reversal_score = (
            high_sweep_detection * safe_div(upper_wick, atr14).clip(0.0, 3.0) / 3.0
            + low_sweep_detection * safe_div(lower_wick, atr14).clip(0.0, 3.0) / 3.0
        ).clip(0.0, 1.0)

        # 4. Price Location (rolling H4 range)
        range_high = high.rolling(range_n, min_periods=max(5, range_n // 5)).max()
        range_low = low.rolling(range_n, min_periods=max(5, range_n // 5)).min()
        range_mid = (range_high + range_low) / 2.0
        h4_range_position = safe_div(close - range_low, range_high - range_low).clip(0.0, 1.0)
        # premium (>0.5) / discount (<0.5) as signed continuous score in [-1, 1]
        premium_discount_zone = (h4_range_position - 0.5) * 2.0
        distance_from_equilibrium = safe_div(close - range_mid, atr14)

        p = self.PREFIX
        cols: dict[str, pd.Series] = {
            "timestamp": pd.to_datetime(df["timestamp"], utc=True),
            f"{p}swing_quality": swing_quality.astype(float),
            f"{p}swing_high_distance_atr": swing_high_distance_atr.astype(float),
            f"{p}swing_low_distance_atr": swing_low_distance_atr.astype(float),
            f"{p}higher_high_state": higher_high_state.astype(float),
            f"{p}lower_low_state": lower_low_state.astype(float),
            f"{p}swing_direction": swing_direction.astype(float),
            f"{p}swing_strength": swing_strength.astype(float),
            f"{p}bullish_structure_break": bullish_structure_break.astype(float),
            f"{p}bearish_structure_break": bearish_structure_break.astype(float),
            f"{p}distance_from_structure_break": distance_from_structure_break.astype(float),
            f"{p}high_sweep_detection": high_sweep_detection.astype(float),
            f"{p}low_sweep_detection": low_sweep_detection.astype(float),
            f"{p}rejection_strength": rejection_strength.astype(float),
            f"{p}wick_reversal_score": wick_reversal_score.astype(float),
            f"{p}h4_range_position": h4_range_position.astype(float),
            f"{p}premium_discount_zone": premium_discount_zone.astype(float),
            f"{p}distance_from_equilibrium": distance_from_equilibrium.astype(float),
        }
        out = pd.DataFrame(cols)

        specs = self._specs(p)
        return out, specs

    @staticmethod
    def _specs(prefix: str) -> list[ContextFeatureSpec]:
        p = prefix
        return [
            ContextFeatureSpec(
                name=f"{p}swing_quality",
                category="swing_structure",
                description="Legacy swing clarity (span vs ATR + body share).",
                value_range="[0, 1]",
                depends_on=("high", "low", "atr", "open", "close"),
            ),
            ContextFeatureSpec(
                name=f"{p}swing_high_distance_atr",
                category="swing_structure",
                description="Close minus causal swing-high, in ATR units.",
                value_range="unbounded",
                depends_on=("close", "swing_high", "atr"),
            ),
            ContextFeatureSpec(
                name=f"{p}swing_low_distance_atr",
                category="swing_structure",
                description="Close minus causal swing-low, in ATR units.",
                value_range="unbounded",
                depends_on=("close", "swing_low", "atr"),
            ),
            ContextFeatureSpec(
                name=f"{p}higher_high_state",
                category="swing_structure",
                description="1 if current swing-high > prior swing-high window.",
                value_range="{0, 1}",
                depends_on=("swing_high",),
            ),
            ContextFeatureSpec(
                name=f"{p}lower_low_state",
                category="swing_structure",
                description="1 if current swing-low < prior swing-low window.",
                value_range="{0, 1}",
                depends_on=("swing_low",),
            ),
            ContextFeatureSpec(
                name=f"{p}swing_direction",
                category="swing_structure",
                description="HH state minus LL state (+1 bullish structure, -1 bearish).",
                value_range="[-1, 1]",
                depends_on=("higher_high_state", "lower_low_state"),
            ),
            ContextFeatureSpec(
                name=f"{p}swing_strength",
                category="swing_structure",
                description="Swing span / ATR capped and scaled to [0, 1].",
                value_range="[0, 1]",
                depends_on=("swing_high", "swing_low", "atr"),
            ),
            ContextFeatureSpec(
                name=f"{p}bullish_structure_break",
                category="break_of_structure",
                description="Close above prior swing-high (causal BOS).",
                value_range="{0, 1}",
                depends_on=("close", "prev_swing_high"),
            ),
            ContextFeatureSpec(
                name=f"{p}bearish_structure_break",
                category="break_of_structure",
                description="Close below prior swing-low (causal BOS).",
                value_range="{0, 1}",
                depends_on=("close", "prev_swing_low"),
            ),
            ContextFeatureSpec(
                name=f"{p}distance_from_structure_break",
                category="break_of_structure",
                description="Signed ATR distance past broken swing level (0 if no break).",
                value_range="unbounded",
                depends_on=("close", "prev_swing_high", "prev_swing_low", "atr"),
            ),
            ContextFeatureSpec(
                name=f"{p}high_sweep_detection",
                category="liquidity_structure",
                description="High sweeps prior swing-high then closes back below.",
                value_range="{0, 1}",
                depends_on=("high", "close", "prev_swing_high"),
            ),
            ContextFeatureSpec(
                name=f"{p}low_sweep_detection",
                category="liquidity_structure",
                description="Low sweeps prior swing-low then closes back above.",
                value_range="{0, 1}",
                depends_on=("low", "close", "prev_swing_low"),
            ),
            ContextFeatureSpec(
                name=f"{p}rejection_strength",
                category="liquidity_structure",
                description="Dominant wick as fraction of bar range.",
                value_range="[0, 1]",
                depends_on=("open", "high", "low", "close"),
            ),
            ContextFeatureSpec(
                name=f"{p}wick_reversal_score",
                category="liquidity_structure",
                description="Sweep-conditioned wick rejection strength vs ATR.",
                value_range="[0, 1]",
                depends_on=("high_sweep_detection", "low_sweep_detection", "atr"),
            ),
            ContextFeatureSpec(
                name=f"{p}h4_range_position",
                category="price_location",
                description="Close location in rolling H4 range [0=low, 1=high].",
                value_range="[0, 1]",
                depends_on=("close", "range_high", "range_low"),
            ),
            ContextFeatureSpec(
                name=f"{p}premium_discount_zone",
                category="price_location",
                description="Signed premium/discount score from range mid [-1, 1].",
                value_range="[-1, 1]",
                depends_on=("h4_range_position",),
            ),
            ContextFeatureSpec(
                name=f"{p}distance_from_equilibrium",
                category="price_location",
                description="Close minus range midpoint in ATR units.",
                value_range="unbounded",
                depends_on=("close", "range_mid", "atr"),
            ),
        ]


# Feature sets for ablation (prefixed names).
SWING_QUALITY_ONLY: tuple[str, ...] = ("ctx_h4_swing_quality",)

NEW_STRUCTURE_FEATURES: tuple[str, ...] = (
    "ctx_h4_swing_high_distance_atr",
    "ctx_h4_swing_low_distance_atr",
    "ctx_h4_higher_high_state",
    "ctx_h4_lower_low_state",
    "ctx_h4_swing_direction",
    "ctx_h4_swing_strength",
    "ctx_h4_bullish_structure_break",
    "ctx_h4_bearish_structure_break",
    "ctx_h4_distance_from_structure_break",
    "ctx_h4_high_sweep_detection",
    "ctx_h4_low_sweep_detection",
    "ctx_h4_rejection_strength",
    "ctx_h4_wick_reversal_score",
    "ctx_h4_h4_range_position",
    "ctx_h4_premium_discount_zone",
    "ctx_h4_distance_from_equilibrium",
)

ALL_STRUCTURE_FEATURES: tuple[str, ...] = SWING_QUALITY_ONLY + NEW_STRUCTURE_FEATURES
