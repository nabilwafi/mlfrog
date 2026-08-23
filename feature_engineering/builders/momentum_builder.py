"""Momentum features — normalized MACD / RSI percentile (no raw MACD levels)."""

from __future__ import annotations

from typing import Any

import pandas as pd

from feature_engineering.builders._transforms import rolling_percentile, rolling_rank, rolling_zscore, safe_div
from feature_engineering.builders.base_builder import BaseFeatureBuilder
from feature_engineering.entities.feature import Feature
from feature_engineering.entities.feature_metadata import FeatureMetadata
from feature_engineering.registry.feature_registry import FeatureRegistry


@FeatureRegistry.register
class MomentumBuilder(BaseFeatureBuilder):
    def name(self) -> str:
        return "momentum"

    def build(self, frame: pd.DataFrame, *, context: dict[str, Any]) -> list[Feature]:
        mid = context["intermediates"]
        close = frame["close"].astype(float)
        atr14 = mid["atr_14"]
        macd_line = mid["macd"]
        macd_signal = mid["macd_signal"]
        macd_hist = mid["macd_hist"]
        rsi14 = mid["rsi_14"]
        z_window = int(self.params.get("zscore_window", 100))
        rank_window = int(self.params.get("rank_window", 100))

        macd_normalized = safe_div(macd_line, atr14)
        macd_histogram_zscore = rolling_zscore(macd_hist, z_window)
        macd_signal_distance = safe_div(macd_line - macd_signal, atr14)
        momentum = close.pct_change(int(self.params.get("momentum_lookback", 10)))
        momentum_rank = rolling_rank(momentum, rank_window)
        rsi_percentile = rolling_percentile(rsi14, rank_window)
        roc_3 = close.pct_change(3)
        roc_12 = close.pct_change(12)
        mom_base = close.pct_change(6)
        momentum_acceleration = mom_base.diff(3)

        specs: list[tuple[str, pd.Series, tuple[str, ...], str]] = [
            (
                "macd_normalized",
                macd_normalized,
                ("macd", "atr_14"),
                "MACD line divided by ATR (price-scale free).",
            ),
            (
                "macd_histogram_zscore",
                macd_histogram_zscore,
                ("macd_hist",),
                "Rolling z-score of MACD histogram.",
            ),
            (
                "macd_signal_distance",
                macd_signal_distance,
                ("macd", "macd_signal", "atr_14"),
                "MACD minus signal in ATR units.",
            ),
            (
                "momentum_rank",
                momentum_rank,
                ("close",),
                "Rolling rank of short-horizon price momentum.",
            ),
            (
                "rsi_percentile",
                rsi_percentile,
                ("rsi_14",),
                "Rolling percentile of RSI (regime-robust).",
            ),
            (
                "roc_3",
                roc_3,
                ("close",),
                "3-bar rate of change.",
            ),
            (
                "roc_12",
                roc_12,
                ("close",),
                "12-bar rate of change.",
            ),
            (
                "momentum_acceleration",
                momentum_acceleration,
                ("close",),
                "Change in 6-bar momentum over 3 bars.",
            ),
        ]
        return [
            Feature(
                FeatureMetadata(
                    name=name,
                    category="momentum",
                    stationary=True,
                    normalized=True,
                    drift_sensitive=False,
                    depends_on=deps,
                    description=desc,
                ),
                values=vals,
            )
            for name, vals, deps, desc in specs
        ]
