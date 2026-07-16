"""Body ratio interaction: body_size / candle range."""

from __future__ import annotations

import pandas as pd

from features.indicators.base_indicator import BaseIndicator
from features.registry.indicator_registry import IndicatorRegistry


@IndicatorRegistry.register
class BodyRatio(BaseIndicator):
    def name(self) -> str:
        return "body_ratio"

    def required_columns(self) -> list[str]:
        return ["open", "high", "low", "close"]

    def feature_prefix(self) -> str:
        return "body_ratio"

    def calculate(self, df: pd.DataFrame) -> pd.Series:
        self.validate_columns(df)
        o = df["open"].astype(float)
        h = df["high"].astype(float)
        l = df["low"].astype(float)
        c = df["close"].astype(float)
        body = (c - o).abs()
        rng = (h - l).replace(0.0, pd.NA)
        return (body / rng).rename(self.feature_prefix())
