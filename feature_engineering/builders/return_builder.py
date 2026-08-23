"""Explicit multi-horizon returns normalized by ATR."""

from __future__ import annotations

from typing import Any

import pandas as pd

from feature_engineering.builders._transforms import safe_div
from feature_engineering.builders.base_builder import BaseFeatureBuilder
from feature_engineering.entities.feature import Feature
from feature_engineering.entities.feature_metadata import FeatureMetadata
from feature_engineering.registry.feature_registry import FeatureRegistry


@FeatureRegistry.register
class ReturnBuilder(BaseFeatureBuilder):
    def name(self) -> str:
        return "return"

    def build(self, frame: pd.DataFrame, *, context: dict[str, Any]) -> list[Feature]:
        close = frame["close"].astype(float)
        atr14 = context["intermediates"]["atr_14"]
        specs: list[tuple[str, pd.Series, int, str]] = []
        for n in (1, 5, 20):
            ret = close.pct_change(n)
            col = f"return_{n}_atr"
            vals = safe_div(ret * close, atr14)
            specs.append((col, vals, n, f"{n}-bar return in ATR units."))
        return [
            Feature(
                FeatureMetadata(
                    name=name,
                    category="price_return",
                    stationary=True,
                    normalized=True,
                    drift_sensitive=False,
                    depends_on=("close", "atr_14"),
                    description=desc,
                ),
                values=vals,
            )
            for name, vals, _n, desc in specs
        ]
