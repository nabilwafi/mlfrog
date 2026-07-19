"""ATR indicator (Wilder)."""

from __future__ import annotations

import pandas as pd

from features.indicators.base_indicator import BaseIndicator
from features.registry.indicator_registry import IndicatorRegistry


@IndicatorRegistry.register
class ATR(BaseIndicator):
    def name(self) -> str:
        return "atr"

    def required_columns(self) -> list[str]:
        return ["high", "low", "close"]

    def feature_prefix(self) -> str:
        return f"atr_{int(self.params['period'])}"

    def calculate(self, df: pd.DataFrame) -> pd.Series:
        self.validate_columns(df)
        period = int(self.params["period"])
        high = df["high"].astype(float)
        low = df["low"].astype(float)
        close = df["close"].astype(float)
        prev_close = close.shift(1)
        tr = pd.concat(
            [
                (high - low).abs(),
                (high - prev_close).abs(),
                (low - prev_close).abs(),
            ],
            axis=1,
        ).max(axis=1)
        atr = tr.ewm(alpha=1 / period, min_periods=period, adjust=False).mean()
        return atr.rename(self.feature_prefix())
