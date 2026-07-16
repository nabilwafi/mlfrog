"""FeatureService — load MarketData → pipeline → save FeatureSet."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

from data.entities.market_data import MarketData
from data.repositories.market_repository import MarketRepository
from features.entities.feature_set import FeatureSet
from features.pipelines.feature_pipeline import FeaturePipeline
from features.repositories.feature_repository import FeatureRepository
from features.validators.feature_validator import FeatureValidator

logger = logging.getLogger(__name__)


class FeatureService:
    def __init__(
        self,
        market_repository: MarketRepository,
        feature_repository: FeatureRepository,
        feature_config: dict[str, Any],
        *,
        timezone: str = "UTC",
    ) -> None:
        self._market_repo = market_repository
        self._feature_repo = feature_repository
        self._feature_config = feature_config
        self._timezone = timezone
        self._pipeline = FeaturePipeline(feature_config)

    def load_market(self, symbol: str, timeframe: str) -> MarketData:
        logger.info("Loading MarketData | symbol=%s timeframe=%s", symbol, timeframe)
        return self._market_repo.load_parquet(symbol, timeframe, timezone=self._timezone)

    def generate(self, market_data: MarketData) -> tuple[FeatureSet, dict[str, int]]:
        return self._pipeline.run(market_data)

    def validate(self, feature_set: FeatureSet) -> FeatureSet:
        FeatureValidator(allow_nan=False).validate(feature_set)
        return feature_set

    def save(self, feature_set: FeatureSet) -> Path:
        return self._feature_repo.save_parquet(feature_set)

    def run(self, symbol: str, timeframe: str) -> tuple[FeatureSet, Path, dict[str, int]]:
        market = self.load_market(symbol, timeframe)
        feature_set, stats = self.generate(market)
        self.validate(feature_set)
        path = self.save(feature_set)
        loaded = self._feature_repo.load_parquet(
            symbol, timeframe, feature_set.feature_version, timezone=self._timezone
        )
        self.validate(loaded)
        logger.info("FeatureService complete | saved=%s", path)
        return loaded, path, stats
