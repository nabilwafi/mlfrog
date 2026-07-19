"""Statistical rolling transforms of returns / close (stationary family)."""

from __future__ import annotations

from typing import Any

import pandas as pd

from feature_engineering.builders._transforms import (
    rolling_percentile,
    rolling_rank,
    rolling_zscore,
    safe_div,
)
from feature_engineering.builders.base_builder import BaseFeatureBuilder
from feature_engineering.entities.feature import Feature
from feature_engineering.entities.feature_metadata import FeatureMetadata
from feature_engineering.registry.feature_registry import FeatureRegistry


@FeatureRegistry.register
class StatisticalBuilder(BaseFeatureBuilder):
    def name(self) -> str:
        return "statistical"

    def build(self, frame: pd.DataFrame, *, context: dict[str, Any]) -> list[Feature]:
        close = frame["close"].astype(float)
        window = int(self.params.get("window", 20))
        ret = close.pct_change()
        mean = ret.rolling(window, min_periods=max(5, window // 5)).mean()
        std = ret.rolling(window, min_periods=max(5, window // 5)).std(ddof=0)
        q = float(self.params.get("quantile", 0.75))

        rolling_z = rolling_zscore(ret, window)
        rolling_pct = rolling_percentile(ret, window)
        rolling_rk = rolling_rank(ret, window)
        rolling_q = ret.rolling(window, min_periods=max(5, window // 5)).quantile(q)
        rolling_std = std
        rolling_mean_distance = safe_div(ret - mean, std)

        specs: list[tuple[str, pd.Series, tuple[str, ...], str]] = [
            (
                "rolling_zscore",
                rolling_z,
                ("close",),
                f"Rolling z-score of returns (window={window}).",
            ),
            (
                "rolling_percentile",
                rolling_pct,
                ("close",),
                f"Rolling percentile of returns (window={window}).",
            ),
            (
                "rolling_rank",
                rolling_rk,
                ("close",),
                f"Rolling rank of returns (window={window}).",
            ),
            (
                "rolling_quantile",
                rolling_q,
                ("close",),
                f"Rolling return quantile q={q} (window={window}).",
            ),
            (
                "rolling_std",
                rolling_std,
                ("close",),
                f"Rolling std of returns (window={window}).",
            ),
            (
                "rolling_mean_distance",
                rolling_mean_distance,
                ("close",),
                "Distance of return from rolling mean in std units.",
            ),
        ]
        return [
            Feature(
                FeatureMetadata(
                    name=name,
                    category="statistical",
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
