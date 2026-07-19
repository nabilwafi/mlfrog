"""MACD indicator."""

from __future__ import annotations

import pandas as pd

from features.indicators.base_indicator import BaseIndicator
from features.registry.indicator_registry import IndicatorRegistry


@IndicatorRegistry.register
class MACD(BaseIndicator):
    def name(self) -> str:
        return "macd"

    def required_columns(self) -> list[str]:
        return [str(self.params.get("column", "close"))]

    def feature_prefix(self) -> str:
        fast = int(self.params["fast"])
        slow = int(self.params["slow"])
        signal = int(self.params["signal"])
        return f"macd_{fast}_{slow}_{signal}"

    def calculate(self, df: pd.DataFrame) -> pd.DataFrame:
        self.validate_columns(df)
        col = str(self.params.get("column", "close"))
        fast = int(self.params["fast"])
        slow = int(self.params["slow"])
        signal = int(self.params["signal"])
        price = df[col].astype(float)
        ema_fast = price.ewm(span=fast, adjust=False).mean()
        ema_slow = price.ewm(span=slow, adjust=False).mean()
        macd_line = ema_fast - ema_slow
        signal_line = macd_line.ewm(span=signal, adjust=False).mean()
        hist = macd_line - signal_line
        prefix = self.feature_prefix()
        return pd.DataFrame(
            {
                f"{prefix}": macd_line,
                f"{prefix}_signal": signal_line,
                f"{prefix}_hist": hist,
            },
            index=df.index,
        )
