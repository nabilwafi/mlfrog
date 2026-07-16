"""EMA distance interaction: (price - ema) / atr_or_price."""

from __future__ import annotations

import pandas as pd

from features.indicators.base_indicator import BaseIndicator
from features.registry.indicator_registry import IndicatorRegistry


@IndicatorRegistry.register
class EMADistance(BaseIndicator):
    """
    Requires prior columns on the working frame (produced by indicators).
    params: ema_column, price_column, scale_column (optional ATR), mode=abs|signed
    """

    def name(self) -> str:
        return "ema_distance"

    def required_columns(self) -> list[str]:
        cols = [
            str(self.params.get("price_column", "close")),
            str(self.params["ema_column"]),
        ]
        scale = self.params.get("scale_column")
        if scale:
            cols.append(str(scale))
        return cols

    def feature_prefix(self) -> str:
        ema_col = str(self.params["ema_column"])
        return f"ema_distance_{ema_col}"

    def calculate(self, df: pd.DataFrame) -> pd.Series:
        self.validate_columns(df)
        price = df[str(self.params.get("price_column", "close"))].astype(float)
        ema = df[str(self.params["ema_column"])].astype(float)
        dist = price - ema
        scale_col = self.params.get("scale_column")
        if scale_col:
            scale = df[str(scale_col)].astype(float).replace(0.0, pd.NA)
            out = dist / scale
        else:
            out = dist / price.replace(0.0, pd.NA)
        if bool(self.params.get("absolute", False)):
            out = out.abs()
        return out.rename(self.feature_prefix())
