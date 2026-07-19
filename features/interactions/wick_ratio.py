"""Wick ratio interactions."""

from __future__ import annotations

import pandas as pd

from features.indicators.base_indicator import BaseIndicator
from features.registry.indicator_registry import IndicatorRegistry


@IndicatorRegistry.register
class WickRatio(BaseIndicator):
    def name(self) -> str:
        return "wick_ratio"

    def required_columns(self) -> list[str]:
        return ["open", "high", "low", "close"]

    def feature_prefix(self) -> str:
        return "wick_ratio"

    def calculate(self, df: pd.DataFrame) -> pd.DataFrame:
        self.validate_columns(df)
        o = df["open"].astype(float)
        h = df["high"].astype(float)
        l = df["low"].astype(float)
        c = df["close"].astype(float)
        body_top = pd.concat([o, c], axis=1).max(axis=1)
        body_bot = pd.concat([o, c], axis=1).min(axis=1)
        upper = (h - body_top).clip(lower=0.0)
        lower = (body_bot - l).clip(lower=0.0)
        rng = (h - l).replace(0.0, pd.NA)
        return pd.DataFrame(
            {
                "upper_wick_ratio": upper / rng,
                "lower_wick_ratio": lower / rng,
            },
            index=df.index,
        )
