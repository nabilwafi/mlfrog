"""Session / calendar features."""

from __future__ import annotations

from zoneinfo import ZoneInfo

import pandas as pd

from features.indicators.base_indicator import BaseIndicator
from features.registry.indicator_registry import IndicatorRegistry


@IndicatorRegistry.register
class Session(BaseIndicator):
    def name(self) -> str:
        return "session"

    def required_columns(self) -> list[str]:
        return ["timestamp"]

    def feature_prefix(self) -> str:
        return "session"

    def calculate(self, df: pd.DataFrame) -> pd.DataFrame:
        self.validate_columns(df)
        tz_name = str(self.params.get("timezone", "UTC"))
        tz = ZoneInfo(tz_name)
        london = self.params.get("london_hours", [7, 16])
        ny = self.params.get("ny_hours", [12, 21])
        asia = self.params.get("asia_hours", [0, 9])

        ts = pd.to_datetime(df["timestamp"], utc=True).dt.tz_convert(tz)
        hour = ts.dt.hour.astype(float)
        dow = ts.dt.dayofweek.astype(float)

        def in_window(h: pd.Series, window: list) -> pd.Series:
            start, end = int(window[0]), int(window[1])
            if start <= end:
                return ((h >= start) & (h < end)).astype(float)
            return ((h >= start) | (h < end)).astype(float)

        return pd.DataFrame(
            {
                "hour": hour,
                "day_of_week": dow,
                "is_london": in_window(hour, list(london)),
                "is_ny": in_window(hour, list(ny)),
                "is_asia": in_window(hour, list(asia)),
            },
            index=df.index,
        )
