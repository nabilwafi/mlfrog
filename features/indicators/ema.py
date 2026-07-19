"""EMA indicator."""

from __future__ import annotations

import pandas as pd

from features.indicators.base_indicator import BaseIndicator
from features.registry.indicator_registry import IndicatorRegistry


@IndicatorRegistry.register
class EMA(BaseIndicator):
    def name(self) -> str:
        return "ema"

    def required_columns(self) -> list[str]:
        return [str(self.params.get("column", "close"))]

    def feature_prefix(self) -> str:
        period = int(self.params["period"])
        col = str(self.params.get("column", "close"))
        return f"ema_{period}_{col}"

    def calculate(self, df: pd.DataFrame) -> pd.Series:
        self.validate_columns(df)
        period = int(self.params["period"])
        col = str(self.params.get("column", "close"))
        return df[col].astype(float).ewm(span=period, adjust=False).mean().rename(self.feature_prefix())
