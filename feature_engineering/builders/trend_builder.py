"""Trend features — EMA distances / slopes / alignment (no raw EMA levels)."""

from __future__ import annotations

from typing import Any

import pandas as pd

from feature_engineering.builders._transforms import safe_div, slope, streak_duration
from feature_engineering.builders.base_builder import BaseFeatureBuilder
from feature_engineering.entities.feature import Feature
from feature_engineering.entities.feature_metadata import FeatureMetadata
from feature_engineering.registry.feature_registry import FeatureRegistry


@FeatureRegistry.register
class TrendBuilder(BaseFeatureBuilder):
    def name(self) -> str:
        return "trend"

    def build(self, frame: pd.DataFrame, *, context: dict[str, Any]) -> list[Feature]:
        close = frame["close"].astype(float)
        mid = context["intermediates"]
        ema20 = mid["ema_20"]
        ema50 = mid["ema_50"]
        atr14 = mid["atr_14"]

        ema20_distance_atr = safe_div(close - ema20, atr14)
        ema50_distance_atr = safe_div(close - ema50, atr14)
        ema20_distance_percent = safe_div(close - ema20, close) * 100.0
        ema50_distance_percent = safe_div(close - ema50, close) * 100.0
        ema_cross_distance = safe_div(ema20 - ema50, atr14)
        ema20_slope = slope(ema20, int(self.params.get("slope_lookback", 3)))
        ema50_slope = slope(ema50, int(self.params.get("slope_lookback", 3)))
        bullish = ema20 > ema50
        ema_trend_duration = streak_duration(bullish)
        above20 = (close > ema20).astype(float)
        above50 = (close > ema50).astype(float)
        ema20_above50 = (ema20 > ema50).astype(float)
        ema_alignment_score = (above20 + above50 + ema20_above50) / 3.0 * 2.0 - 1.0

        specs = [
            (
                "ema20_distance_atr",
                ema20_distance_atr,
                True,
                True,
                False,
                ("close", "ema_20", "atr_14"),
                "Close minus EMA20 in ATR units (scale-free trend distance).",
            ),
            (
                "ema50_distance_atr",
                ema50_distance_atr,
                True,
                True,
                False,
                ("close", "ema_50", "atr_14"),
                "Close minus EMA50 in ATR units.",
            ),
            (
                "ema20_distance_percent",
                ema20_distance_percent,
                True,
                True,
                False,
                ("close", "ema_20"),
                "Close minus EMA20 as percent of price.",
            ),
            (
                "ema50_distance_percent",
                ema50_distance_percent,
                True,
                True,
                False,
                ("close", "ema_50"),
                "Close minus EMA50 as percent of price.",
            ),
            (
                "ema_cross_distance",
                ema_cross_distance,
                True,
                True,
                False,
                ("ema_20", "ema_50", "atr_14"),
                "EMA20-EMA50 spread in ATR units.",
            ),
            (
                "ema20_slope",
                ema20_slope,
                True,
                True,
                False,
                ("ema_20",),
                "Normalized EMA20 slope over short lookback.",
            ),
            (
                "ema50_slope",
                ema50_slope,
                True,
                True,
                False,
                ("ema_50",),
                "Normalized EMA50 slope over short lookback.",
            ),
            (
                "ema_trend_duration",
                ema_trend_duration,
                False,
                False,
                True,
                ("ema_20", "ema_50"),
                "Length of current EMA20/EMA50 regime (bars).",
            ),
            (
                "ema_alignment_score",
                ema_alignment_score,
                True,
                True,
                False,
                ("close", "ema_20", "ema_50"),
                "Price/EMA stack alignment score in [-1, 1].",
            ),
        ]
        return [
            Feature(
                FeatureMetadata(
                    name=n,
                    category="trend",
                    stationary=st,
                    normalized=nm,
                    drift_sensitive=ds,
                    depends_on=dep,
                    description=desc,
                ),
                values=vals,
            )
            for n, vals, st, nm, ds, dep, desc in specs
        ]
