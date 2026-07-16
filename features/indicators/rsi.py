"""RSI indicator (Wilder)."""

from __future__ import annotations

import pandas as pd

from features.indicators.base_indicator import BaseIndicator
from features.registry.indicator_registry import IndicatorRegistry


@IndicatorRegistry.register
class RSI(BaseIndicator):
    def name(self) -> str:
        return "rsi"

    def required_columns(self) -> list[str]:
        return [str(self.params.get("column", "close"))]

    def feature_prefix(self) -> str:
        return f"rsi_{int(self.params['period'])}"

    def calculate(self, df: pd.DataFrame) -> pd.Series:
        self.validate_columns(df)
        period = int(self.params["period"])
        col = str(self.params.get("column", "close"))
        delta = df[col].astype(float).diff()
        gain = delta.clip(lower=0.0)
        loss = (-delta).clip(lower=0.0)
        avg_gain = gain.ewm(alpha=1 / period, min_periods=period, adjust=False).mean()
        avg_loss = loss.ewm(alpha=1 / period, min_periods=period, adjust=False).mean()
        rs = avg_gain / avg_loss.replace(0.0, pd.NA)
        rsi = 100.0 - (100.0 / (1.0 + rs))
        return rsi.rename(self.feature_prefix())
