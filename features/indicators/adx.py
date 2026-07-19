"""ADX indicator (Wilder)."""

from __future__ import annotations

import pandas as pd

from features.indicators.base_indicator import BaseIndicator
from features.registry.indicator_registry import IndicatorRegistry


@IndicatorRegistry.register
class ADX(BaseIndicator):
    def name(self) -> str:
        return "adx"

    def required_columns(self) -> list[str]:
        return ["high", "low", "close"]

    def feature_prefix(self) -> str:
        return f"adx_{int(self.params['period'])}"

    def calculate(self, df: pd.DataFrame) -> pd.DataFrame:
        self.validate_columns(df)
        period = int(self.params["period"])
        high = df["high"].astype(float)
        low = df["low"].astype(float)
        close = df["close"].astype(float)

        up = high.diff()
        down = -low.diff()
        plus_dm = up.where((up > down) & (up > 0.0), 0.0)
        minus_dm = down.where((down > up) & (down > 0.0), 0.0)

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
        plus_di = 100.0 * (
            plus_dm.ewm(alpha=1 / period, min_periods=period, adjust=False).mean() / atr
        )
        minus_di = 100.0 * (
            minus_dm.ewm(alpha=1 / period, min_periods=period, adjust=False).mean() / atr
        )
        dx = (100.0 * (plus_di - minus_di).abs() / (plus_di + minus_di).replace(0.0, pd.NA))
        adx = dx.ewm(alpha=1 / period, min_periods=period, adjust=False).mean()

        prefix = self.feature_prefix()
        return pd.DataFrame(
            {
                f"{prefix}": adx,
                f"plus_di_{period}": plus_di,
                f"minus_di_{period}": minus_di,
            },
            index=df.index,
        )
