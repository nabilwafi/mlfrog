"""ATR ratio interaction: atr / price."""

from __future__ import annotations

import pandas as pd

from features.indicators.base_indicator import BaseIndicator
from features.registry.indicator_registry import IndicatorRegistry


@IndicatorRegistry.register
class ATRRatio(BaseIndicator):
    def name(self) -> str:
        return "atr_ratio"

    def required_columns(self) -> list[str]:
        return [
            str(self.params["atr_column"]),
            str(self.params.get("price_column", "close")),
        ]

    def feature_prefix(self) -> str:
        return f"atr_ratio_{self.params['atr_column']}"

    def calculate(self, df: pd.DataFrame) -> pd.Series:
        self.validate_columns(df)
        atr = df[str(self.params["atr_column"])].astype(float)
        price = df[str(self.params.get("price_column", "close"))].astype(float)
        out = atr / price.replace(0.0, pd.NA)
        return out.rename(self.feature_prefix())
