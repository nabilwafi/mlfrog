"""Candle geometry as percent-of-range / ranks (scale-free)."""

from __future__ import annotations

from typing import Any

import pandas as pd

from feature_engineering.builders._transforms import rolling_rank, safe_div
from feature_engineering.builders.base_builder import BaseFeatureBuilder
from feature_engineering.entities.feature import Feature
from feature_engineering.entities.feature_metadata import FeatureMetadata
from feature_engineering.registry.feature_registry import FeatureRegistry


@FeatureRegistry.register
class CandleBuilder(BaseFeatureBuilder):
    def name(self) -> str:
        return "candle"

    def build(self, frame: pd.DataFrame, *, context: dict[str, Any]) -> list[Feature]:
        o = frame["open"].astype(float)
        h = frame["high"].astype(float)
        l = frame["low"].astype(float)
        c = frame["close"].astype(float)
        rng = (h - l).replace(0.0, float("nan"))
        body = (c - o).abs()
        upper = h - pd.concat([o, c], axis=1).max(axis=1)
        lower = pd.concat([o, c], axis=1).min(axis=1) - l
        rank_window = int(self.params.get("rank_window", 100))

        body_percent = safe_div(body, rng) * 100.0
        upper_wick_percent = safe_div(upper, rng) * 100.0
        lower_wick_percent = safe_div(lower, rng) * 100.0
        close_position = safe_div(c - l, rng)
        range_percent = safe_div(rng, c) * 100.0
        body_rank = rolling_rank(body_percent, rank_window)

        specs: list[tuple[str, pd.Series, tuple[str, ...], str]] = [
            (
                "body_percent",
                body_percent,
                ("open", "high", "low", "close"),
                "Candle body as percent of high-low range.",
            ),
            (
                "upper_wick_percent",
                upper_wick_percent,
                ("open", "high", "low", "close"),
                "Upper wick as percent of range.",
            ),
            (
                "lower_wick_percent",
                lower_wick_percent,
                ("open", "high", "low", "close"),
                "Lower wick as percent of range.",
            ),
            (
                "close_position",
                close_position,
                ("high", "low", "close"),
                "Close location inside the bar range [0, 1].",
            ),
            (
                "range_percent",
                range_percent,
                ("high", "low", "close"),
                "Bar range as percent of close.",
            ),
            (
                "body_rank",
                body_rank,
                ("body_percent",),
                "Rolling rank of body_percent.",
            ),
        ]
        return [
            Feature(
                FeatureMetadata(
                    name=name,
                    category="candle",
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
