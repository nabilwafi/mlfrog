"""Volume activity features from tick_volume."""

from __future__ import annotations

from typing import Any

import pandas as pd

from feature_engineering.builders._transforms import rolling_zscore, safe_div
from feature_engineering.builders.base_builder import BaseFeatureBuilder
from feature_engineering.entities.feature import Feature
from feature_engineering.entities.feature_metadata import FeatureMetadata
from feature_engineering.registry.feature_registry import FeatureRegistry


@FeatureRegistry.register
class VolumeBuilder(BaseFeatureBuilder):
    def name(self) -> str:
        return "volume"

    def build(self, frame: pd.DataFrame, *, context: dict[str, Any]) -> list[Feature]:
        if "tick_volume" not in frame.columns:
            raise ValueError("frame missing tick_volume for VolumeBuilder")
        vol = frame["tick_volume"].astype(float)
        window = int(self.params.get("window", 50))
        vol_ma = vol.rolling(window, min_periods=max(5, window // 5)).mean()
        volume_zscore = rolling_zscore(vol, window)
        volume_ratio = safe_div(vol, vol_ma)
        specs = [
            ("volume_zscore", volume_zscore, "Rolling z-score of tick volume."),
            ("volume_ratio", volume_ratio, "Tick volume / rolling mean."),
        ]
        return [
            Feature(
                FeatureMetadata(
                    name=name,
                    category="volume",
                    stationary=True,
                    normalized=True,
                    drift_sensitive=False,
                    depends_on=("tick_volume",),
                    description=desc,
                ),
                values=vals,
            )
            for name, vals, desc in specs
        ]
