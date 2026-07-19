"""Simple and log returns."""

from __future__ import annotations

import numpy as np
import pandas as pd

from features.indicators.base_indicator import BaseIndicator
from features.registry.indicator_registry import IndicatorRegistry


@IndicatorRegistry.register
class Returns(BaseIndicator):
    def name(self) -> str:
        return "returns"

    def required_columns(self) -> list[str]:
        return [str(self.params.get("column", "close"))]

    def feature_prefix(self) -> str:
        periods = int(self.params.get("periods", 1))
        return f"returns_{periods}"

    def calculate(self, df: pd.DataFrame) -> pd.Series:
        self.validate_columns(df)
        col = str(self.params.get("column", "close"))
        periods = int(self.params.get("periods", 1))
        out = df[col].astype(float).pct_change(periods=periods)
        return out.rename(self.feature_prefix())


@IndicatorRegistry.register
class LogReturns(BaseIndicator):
    def name(self) -> str:
        return "log_returns"

    def required_columns(self) -> list[str]:
        return [str(self.params.get("column", "close"))]

    def feature_prefix(self) -> str:
        periods = int(self.params.get("periods", 1))
        return f"log_returns_{periods}"

    def calculate(self, df: pd.DataFrame) -> pd.Series:
        self.validate_columns(df)
        col = str(self.params.get("column", "close"))
        periods = int(self.params.get("periods", 1))
        price = df[col].astype(float)
        out = np.log(price / price.shift(periods))
        return pd.Series(out, index=df.index, name=self.feature_prefix())
