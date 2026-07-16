"""Volatility features — ATR%/percentiles/ranks (no raw ATR level)."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from feature_engineering.builders._transforms import rolling_percentile, rolling_rank, safe_div
from feature_engineering.builders.base_builder import BaseFeatureBuilder
from feature_engineering.entities.feature import Feature
from feature_engineering.entities.feature_metadata import FeatureMetadata
from feature_engineering.registry.feature_registry import FeatureRegistry


@FeatureRegistry.register
class VolatilityBuilder(BaseFeatureBuilder):
    def name(self) -> str:
        return "volatility"

    def build(self, frame: pd.DataFrame, *, context: dict[str, Any]) -> list[Feature]:
        close = frame["close"].astype(float)
        atr14 = context["intermediates"]["atr_14"]
        vol_window = int(self.params.get("rolling_volatility_window", 20))
        pct_window = int(self.params.get("atr_percentile_window", 252))

        atr_percent = safe_div(atr14, close) * 100.0
        atr_percentile_252 = rolling_percentile(atr14, pct_window)
        log_ret = np.log(safe_div(close, close.shift(1)))
        rolling_volatility = log_ret.rolling(
            vol_window, min_periods=max(5, vol_window // 5)
        ).std(ddof=0)
        volatility_rank = rolling_rank(rolling_volatility, pct_window)
        volatility_regime_score = (atr_percentile_252 - 0.5) * 2.0

        specs: list[tuple[str, pd.Series, tuple[str, ...], str]] = [
            (
                "atr_percent",
                atr_percent,
                ("atr_14", "close"),
                "ATR as percent of price (scale-independent).",
            ),
            (
                "atr_percentile_252",
                atr_percentile_252,
                ("atr_14",),
                "Rolling percentile rank of ATR over ~252 bars.",
            ),
            (
                "rolling_volatility",
                rolling_volatility,
                ("close",),
                "Rolling std of log returns.",
            ),
            (
                "volatility_rank",
                volatility_rank,
                ("rolling_volatility",),
                "Rolling rank of realized volatility.",
            ),
            (
                "volatility_regime_score",
                volatility_regime_score,
                ("atr_percentile_252",),
                "ATR percentile mapped to [-1, 1] regime score.",
            ),
        ]
        return [
            Feature(
                FeatureMetadata(
                    name=name,
                    category="volatility",
                    stationary=True,
                    normalized=True,
                    drift_sensitive=False,
                    depends_on=deps,
                    description=desc,
                ),
                values=vals,
            )
            for name, vals, deps, desc in specs
        ]
