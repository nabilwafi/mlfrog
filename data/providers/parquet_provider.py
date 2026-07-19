"""Parquet file market-data provider."""

from __future__ import annotations

import logging
from datetime import datetime
from pathlib import Path

import pandas as pd

from data.entities.market_data import MarketData
from data.exceptions import ProviderConnectionError
from data.providers._frame_adapter import dataframe_to_market_data
from data.providers.base_provider import BaseMarketProvider

logger = logging.getLogger(__name__)


class ParquetProvider(BaseMarketProvider):
    def __init__(self, source_path: str | Path) -> None:
        self._path = Path(source_path)

    def health_check(self) -> bool:
        ok = self._path.is_file()
        if not ok:
            logger.warning("Parquet health_check failed | path=%s", self._path)
        return ok

    def close(self) -> None:
        logger.debug("Parquet provider close (noop) | path=%s", self._path)

    def fetch(
        self,
        symbol: str,
        timeframe: str,
        start: datetime,
        end: datetime,
        *,
        timezone: str = "UTC",
    ) -> MarketData:
        logger.info(
            "Fetching Parquet | path=%s symbol=%s timeframe=%s",
            self._path,
            symbol,
            timeframe,
        )
        if not self._path.is_file():
            raise ProviderConnectionError(f"Parquet file not found: {self._path}")
        try:
            df = pd.read_parquet(self._path)
        except Exception as exc:
            raise ProviderConnectionError(f"failed reading Parquet {self._path}: {exc}") from exc
        market = dataframe_to_market_data(
            df,
            symbol=symbol,
            timeframe=timeframe,
            timezone=timezone,
            start=start,
            end=end,
        )
        logger.info("Fetched Parquet | candles=%s", market.total_candles)
        return market
