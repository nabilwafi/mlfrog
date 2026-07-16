"""Rolling volatility of returns."""

from __future__ import annotations

import numpy as np
import pandas as pd

from features.indicators.base_indicator import BaseIndicator
from features.registry.indicator_registry import IndicatorRegistry


@IndicatorRegistry.register
class Volatility(BaseIndicator):
    def name(self) -> str:
        return "volatility"

    def required_columns(self) -> list[str]:
        return [str(self.params.get("column", "close"))]

    def feature_prefix(self) -> str:
        window = int(self.params["window"])
        return f"volatility_{window}"

    def calculate(self, df: pd.DataFrame) -> pd.Series:
        self.validate_columns(df)
        col = str(self.params.get("column", "close"))
        window = int(self.params["window"])
        use_log = bool(self.params.get("use_log", True))
        price = df[col].astype(float)
        rets = np.log(price / price.shift(1)) if use_log else price.pct_change()
        out = rets.rolling(window=window, min_periods=window).std()
        return out.rename(self.feature_prefix())
