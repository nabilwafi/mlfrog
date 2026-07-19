"""Candle geometry features."""

from __future__ import annotations

import pandas as pd

from features.indicators.base_indicator import BaseIndicator
from features.registry.indicator_registry import IndicatorRegistry


@IndicatorRegistry.register
class CandleGeometry(BaseIndicator):
    """Body size, upper wick, lower wick (absolute price units)."""

    def name(self) -> str:
        return "candle"

    def required_columns(self) -> list[str]:
        return ["open", "high", "low", "close"]

    def feature_prefix(self) -> str:
        return "candle"

    def calculate(self, df: pd.DataFrame) -> pd.DataFrame:
        self.validate_columns(df)
        o = df["open"].astype(float)
        h = df["high"].astype(float)
        l = df["low"].astype(float)
        c = df["close"].astype(float)
        body = (c - o).abs()
        upper = h - pd.concat([o, c], axis=1).max(axis=1)
        lower = pd.concat([o, c], axis=1).min(axis=1) - l
        return pd.DataFrame(
            {
                "body_size": body,
                "upper_wick": upper.clip(lower=0.0),
                "lower_wick": lower.clip(lower=0.0),
            },
            index=df.index,
        )
