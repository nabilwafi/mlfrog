"""Persist and load MarketData under raw/{symbol}/{timeframe}/."""

from __future__ import annotations

import logging
from pathlib import Path

import pandas as pd

from data.entities.market_data import MarketData
from data.exceptions import RepositoryError
from data.providers._frame_adapter import dataframe_to_market_data

logger = logging.getLogger(__name__)

_FILE_STEM = "data"


class MarketRepository:
    def __init__(self, root: str | Path) -> None:
        self._root = Path(root)

    def _dir(self, symbol: str, timeframe: str) -> Path:
        return self._root / symbol.upper() / timeframe.upper()

    def parquet_path(self, symbol: str, timeframe: str) -> Path:
        return self._dir(symbol, timeframe) / f"{_FILE_STEM}.parquet"

    def csv_path(self, symbol: str, timeframe: str) -> Path:
        return self._dir(symbol, timeframe) / f"{_FILE_STEM}.csv"

    def save_parquet(self, market_data: MarketData) -> Path:
        path = self.parquet_path(market_data.symbol, market_data.timeframe)
        logger.info("Saving parquet | path=%s candles=%s", path, market_data.total_candles)
        try:
            path.parent.mkdir(parents=True, exist_ok=True)
            self._to_frame(market_data).to_parquet(path, index=False)
        except Exception as exc:
            raise RepositoryError(f"save_parquet failed: {exc}") from exc
        logger.info("Saved parquet | path=%s", path)
        return path

    def load_parquet(
        self,
        symbol: str,
        timeframe: str,
        *,
        timezone: str = "UTC",
    ) -> MarketData:
        path = self.parquet_path(symbol, timeframe)
        logger.info("Loading parquet | path=%s", path)
        if not path.is_file():
            raise RepositoryError(f"parquet not found: {path}")
        try:
            df = pd.read_parquet(path)
            market = dataframe_to_market_data(
                df, symbol=symbol, timeframe=timeframe, timezone=timezone
            )
        except RepositoryError:
            raise
        except Exception as exc:
            raise RepositoryError(f"load_parquet failed: {exc}") from exc
        logger.info("Loaded parquet | candles=%s", market.total_candles)
        return market

    def save_csv(self, market_data: MarketData) -> Path:
        path = self.csv_path(market_data.symbol, market_data.timeframe)
        logger.info("Saving CSV | path=%s candles=%s", path, market_data.total_candles)
        try:
            path.parent.mkdir(parents=True, exist_ok=True)
            self._to_frame(market_data).to_csv(path, index=False)
        except Exception as exc:
            raise RepositoryError(f"save_csv failed: {exc}") from exc
        logger.info("Saved CSV | path=%s", path)
        return path

    def load_csv(
        self,
        symbol: str,
        timeframe: str,
        *,
        timezone: str = "UTC",
    ) -> MarketData:
        path = self.csv_path(symbol, timeframe)
        logger.info("Loading CSV | path=%s", path)
        if not path.is_file():
            raise RepositoryError(f"CSV not found: {path}")
        try:
            df = pd.read_csv(path)
            market = dataframe_to_market_data(
                df, symbol=symbol, timeframe=timeframe, timezone=timezone
            )
        except RepositoryError:
            raise
        except Exception as exc:
            raise RepositoryError(f"load_csv failed: {exc}") from exc
        logger.info("Loaded CSV | candles=%s", market.total_candles)
        return market

    def list_symbols(self) -> list[str]:
        if not self._root.exists():
            return []
        return sorted(p.name for p in self._root.iterdir() if p.is_dir())

    def list_timeframes(self, symbol: str) -> list[str]:
        base = self._root / symbol.upper()
        if not base.exists():
            return []
        return sorted(p.name for p in base.iterdir() if p.is_dir())

    @staticmethod
    def _to_frame(market_data: MarketData) -> pd.DataFrame:
        rows = [
            {
                "timestamp": c.timestamp.isoformat(),
                "open": c.open,
                "high": c.high,
                "low": c.low,
                "close": c.close,
                "tick_volume": c.tick_volume,
                "spread": c.spread,
                "real_volume": c.real_volume,
            }
            for c in market_data.candles
        ]
        return pd.DataFrame(rows)
