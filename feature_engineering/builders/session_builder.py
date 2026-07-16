"""Session / calendar cyclic encodings (no raw hour integer)."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from feature_engineering.builders.base_builder import BaseFeatureBuilder
from feature_engineering.entities.feature import Feature
from feature_engineering.entities.feature_metadata import FeatureMetadata
from feature_engineering.registry.feature_registry import FeatureRegistry


@FeatureRegistry.register
class SessionBuilder(BaseFeatureBuilder):
    def name(self) -> str:
        return "session"

    def build(self, frame: pd.DataFrame, *, context: dict[str, Any]) -> list[Feature]:
        ts = pd.to_datetime(frame["timestamp"], utc=True)
        hour = ts.dt.hour.astype(float) + ts.dt.minute.astype(float) / 60.0
        dow = ts.dt.dayofweek.astype(float)
        hour_sin = np.sin(2 * np.pi * hour / 24.0)
        hour_cos = np.cos(2 * np.pi * hour / 24.0)
        day_sin = np.sin(2 * np.pi * dow / 7.0)
        day_cos = np.cos(2 * np.pi * dow / 7.0)

        specs: list[tuple[str, pd.Series, str]] = [
            ("hour_sin", pd.Series(hour_sin, index=frame.index), "Sine encoding of hour-of-day."),
            ("hour_cos", pd.Series(hour_cos, index=frame.index), "Cosine encoding of hour-of-day."),
            ("day_sin", pd.Series(day_sin, index=frame.index), "Sine encoding of day-of-week."),
            ("day_cos", pd.Series(day_cos, index=frame.index), "Cosine encoding of day-of-week."),
        ]
        return [
            Feature(
                FeatureMetadata(
                    name=name,
                    category="session",
                    stationary=True,
                    normalized=True,
                    drift_sensitive=False,
                    depends_on=("timestamp",),
                    description=desc,
                ),
                values=vals,
            )
            for name, vals, desc in specs
        ]
