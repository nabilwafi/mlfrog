"""Spread-based liquidity proxies (no order book)."""

from __future__ import annotations

from typing import Any

import pandas as pd

from feature_engineering.builders._transforms import rolling_zscore, safe_div
from feature_engineering.builders.base_builder import BaseFeatureBuilder
from feature_engineering.entities.feature import Feature
from feature_engineering.entities.feature_metadata import FeatureMetadata
from feature_engineering.registry.feature_registry import FeatureRegistry


@FeatureRegistry.register
class LiquidityBuilder(BaseFeatureBuilder):
    def name(self) -> str:
        return "liquidity"

    def build(self, frame: pd.DataFrame, *, context: dict[str, Any]) -> list[Feature]:
        if "spread" not in frame.columns:
            raise ValueError("frame missing spread for LiquidityBuilder")
        spread = frame["spread"].astype(float)
        close = frame["close"].astype(float)
        atr14 = context["intermediates"]["atr_14"]
        window = int(self.params.get("window", 50))
        spread_zscore = rolling_zscore(spread, window)
        spread_ratio = safe_div(spread, atr14)
        spread_pct = safe_div(spread, close) * 10000.0  # approx points per 10k price
        specs = [
            ("spread_zscore", spread_zscore, "Rolling z-score of quoted spread."),
            ("spread_ratio", spread_ratio, "Spread / ATR."),
            ("spread_bps", spread_pct, "Spread as pseudo-bps vs close."),
        ]
        return [
            Feature(
                FeatureMetadata(
                    name=name,
                    category="liquidity",
                    stationary=True,
                    normalized=True,
                    drift_sensitive=False,
                    depends_on=("spread", "close", "atr_14"),
                    description=desc,
                ),
                values=vals,
            )
            for name, vals, desc in specs
        ]
