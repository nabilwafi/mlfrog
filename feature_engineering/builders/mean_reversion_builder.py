"""Mean-reversion location features."""

from __future__ import annotations

from typing import Any

import pandas as pd

from feature_engineering.builders._transforms import rolling_zscore, safe_div
from feature_engineering.builders.base_builder import BaseFeatureBuilder
from feature_engineering.entities.feature import Feature
from feature_engineering.entities.feature_metadata import FeatureMetadata
from feature_engineering.registry.feature_registry import FeatureRegistry


@FeatureRegistry.register
class MeanReversionBuilder(BaseFeatureBuilder):
    def name(self) -> str:
        return "mean_reversion"

    def build(self, frame: pd.DataFrame, *, context: dict[str, Any]) -> list[Feature]:
        close = frame["close"].astype(float)
        window = int(self.params.get("window", 20))
        price_zscore = rolling_zscore(close, window)
        mid = close.rolling(window, min_periods=max(5, window // 5)).mean()
        std = close.rolling(window, min_periods=max(5, window // 5)).std(ddof=0)
        bb_position = safe_div(close - (mid - 2 * std), (4 * std)).clip(0.0, 1.0)
        specs = [
            ("price_zscore", price_zscore, "Rolling z-score of close."),
            ("bb_position", bb_position, "Position inside Bollinger band [0,1]."),
        ]
        return [
            Feature(
                FeatureMetadata(
                    name=name,
                    category="mean_reversion",
                    stationary=True,
                    normalized=True,
                    drift_sensitive=False,
                    depends_on=("close",),
                    description=desc,
                ),
                values=vals,
            )
            for name, vals, desc in specs
        ]
