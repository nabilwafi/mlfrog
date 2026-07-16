"""LabelService — load MarketData → per-side pipeline → save LabelSet(s)."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

from data.entities.market_data import MarketData
from data.repositories.market_repository import MarketRepository
from labels.entities.label_set import LabelSet
from labels.pipelines.label_pipeline import LabelPipeline
from labels.repositories.label_repository import LabelRepository
from labels.validators.label_validator import LabelValidator

logger = logging.getLogger(__name__)


class LabelService:
    def __init__(
        self,
        market_repository: MarketRepository,
        label_repository: LabelRepository,
        label_config: dict[str, Any],
        *,
        timezone: str = "UTC",
    ) -> None:
        self._market_repo = market_repository
        self._label_repo = label_repository
        self._label_config = label_config
        self._timezone = timezone
        self._pipeline = LabelPipeline(label_config)

    def load_market(self, symbol: str, timeframe: str) -> MarketData:
        return self._market_repo.load_parquet(symbol, timeframe, timezone=self._timezone)

    def generate(
        self,
        market_data: MarketData,
        *,
        strategy: str | None = None,
        side: str = "long",
    ) -> tuple[LabelSet, dict]:
        return self._pipeline.run(market_data, strategy=strategy, side=side)

    def validate(self, label_set: LabelSet) -> dict:
        horizon = None
        params = self._label_config.get("params") or {}
        if "horizon" in params:
            horizon = int(params["horizon"])
        return LabelValidator(expected_horizon=horizon).validate(label_set)

    def save(self, label_set: LabelSet) -> Path:
        return self._label_repo.save_parquet(label_set)

    def run(
        self,
        symbol: str,
        timeframe: str,
        *,
        strategy: str | None = None,
        sides: list[str] | None = None,
    ) -> list[tuple[LabelSet, Path, dict]]:
        market = self.load_market(symbol, timeframe)
        strat = strategy or str(self._label_config.get("strategy", "triple_barrier"))
        side_list = sides or list(self._label_config.get("sides") or ["long"])
        results: list[tuple[LabelSet, Path, dict]] = []
        for side in side_list:
            label_set, report = self.generate(market, strategy=strat, side=side)
            path = self.save(label_set)
            loaded = self._label_repo.load_parquet(
                symbol,
                timeframe,
                label_set.side,
                label_set.strategy,
                label_set.label_version,
                timezone=self._timezone,
            )
            self.validate(loaded)
            logger.info(
                "LabelService side complete | side=%s n=%s saved=%s class=%s",
                loaded.side,
                loaded.size,
                path,
                loaded.class_counts,
            )
            results.append((loaded, path, report))
        return results
