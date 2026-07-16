"""Orchestrates fetch → validate → save → load."""

from __future__ import annotations

import logging
from datetime import datetime
from pathlib import Path

from data.entities.market_data import MarketData
from data.exceptions import MarketDataError, ValidationError
from data.providers.base_provider import BaseMarketProvider
from data.repositories.market_repository import MarketRepository
from data.validators.candle_validator import CandleValidator

logger = logging.getLogger(__name__)


class MarketDataService:
    def __init__(
        self,
        provider: BaseMarketProvider,
        repository: MarketRepository,
        validator: CandleValidator | None = None,
        *,
        default_format: str = "parquet",
        timezone: str = "UTC",
    ) -> None:
        self._provider = provider
        self._repository = repository
        self._validator = validator or CandleValidator()
        self._default_format = default_format.lower()
        self._timezone = timezone

    def fetch(
        self,
        symbol: str,
        timeframe: str,
        start: datetime,
        end: datetime,
    ) -> MarketData:
        logger.info("Service fetch | symbol=%s timeframe=%s", symbol, timeframe)
        return self._provider.fetch(
            symbol, timeframe, start, end, timezone=self._timezone
        )

    def validate(self, market_data: MarketData) -> MarketData:
        try:
            self._validator.validate(market_data)
        except ValidationError:
            logger.error("Service validation failed")
            raise
        return market_data

    def save(self, market_data: MarketData, *, fmt: str | None = None) -> Path:
        fmt = (fmt or self._default_format).lower()
        if fmt == "csv":
            return self._repository.save_csv(market_data)
        if fmt == "parquet":
            return self._repository.save_parquet(market_data)
        raise MarketDataError(f"unsupported save format: {fmt!r}")

    def load(
        self,
        symbol: str,
        timeframe: str,
        *,
        fmt: str | None = None,
    ) -> MarketData:
        fmt = (fmt or self._default_format).lower()
        if fmt == "csv":
            return self._repository.load_csv(symbol, timeframe, timezone=self._timezone)
        if fmt == "parquet":
            return self._repository.load_parquet(symbol, timeframe, timezone=self._timezone)
        raise MarketDataError(f"unsupported load format: {fmt!r}")

    def fetch_validate_save(
        self,
        symbol: str,
        timeframe: str,
        start: datetime,
        end: datetime,
        *,
        fmt: str | None = None,
    ) -> tuple[MarketData, Path]:
        """Full pipeline: fetch → validate → save → load round-trip check → return saved data."""
        market = self.fetch(symbol, timeframe, start, end)
        self.validate(market)
        path = self.save(market, fmt=fmt)
        loaded = self.load(symbol, timeframe, fmt=fmt)
        self.validate(loaded)
        logger.info(
            "Pipeline complete | fetched=%s saved=%s",
            market.total_candles,
            path,
        )
        return loaded, path

    def close(self) -> None:
        self._provider.close()
