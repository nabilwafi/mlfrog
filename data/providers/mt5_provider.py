"""MetaTrader 5 market-data provider."""

from __future__ import annotations

import logging
from datetime import datetime
from typing import Any, Iterator

import pandas as pd

from data.connectors.mt5_connector import MT5Connector
from data.connectors.mt5_constants import resolve_timeframe
from data.entities.market_data import MarketData
from data.exceptions import ProviderConnectionError
from data.providers._frame_adapter import dataframe_to_market_data
from data.providers.base_provider import BaseMarketProvider

logger = logging.getLogger(__name__)


def _add_months(dt: datetime, months: int) -> datetime:
    month = dt.month - 1 + months
    year = dt.year + month // 12
    month = month % 12 + 1
    days_in_month = [
        31,
        29 if year % 4 == 0 and (year % 100 != 0 or year % 400 == 0) else 28,
        31,
        30,
        31,
        30,
        31,
        31,
        30,
        31,
        30,
        31,
    ][month - 1]
    day = min(dt.day, days_in_month)
    return dt.replace(year=year, month=month, day=day)


class MT5Provider(BaseMarketProvider):
    def __init__(self, cfg: dict[str, Any]) -> None:
        self._cfg = cfg
        self._chunk = cfg.get("chunk", {"unit": "year", "step": 1})
        self._connector = MT5Connector(cfg, logger)
        self._opened = False

    def _ensure_connected(self) -> None:
        if self._opened:
            return
        try:
            self._connector.connect()
            self._opened = True
            logger.info("MT5 provider connected")
        except Exception as exc:
            raise ProviderConnectionError(f"MT5 connection failed: {exc}") from exc

    def health_check(self) -> bool:
        try:
            self._ensure_connected()
            return True
        except ProviderConnectionError:
            logger.warning("MT5 health_check failed")
            return False

    def close(self) -> None:
        if self._opened:
            self._connector.disconnect()
            self._opened = False
            logger.info("MT5 provider closed")

    def fetch(
        self,
        symbol: str,
        timeframe: str,
        start: datetime,
        end: datetime,
        *,
        timezone: str = "UTC",
    ) -> MarketData:
        self._ensure_connected()
        logger.info(
            "Fetching MT5 | symbol=%s timeframe=%s start=%s end=%s",
            symbol,
            timeframe,
            start,
            end,
        )
        try:
            self._connector.select_symbol(symbol)
            tf = resolve_timeframe(timeframe)
        except Exception as exc:
            raise ProviderConnectionError(f"MT5 prepare failed: {exc}") from exc

        frames = []
        for chunk_start, chunk_end in self._iter_chunks(start, end):
            logger.info("Fetching MT5 chunk | %s -> %s", chunk_start, chunk_end)
            try:
                part = self._connector.copy_rates_range(symbol, tf, chunk_start, chunk_end)
            except Exception as exc:
                raise ProviderConnectionError(f"MT5 fetch failed: {exc}") from exc
            if part is not None and not part.empty:
                frames.append(part)

        if not frames:
            logger.warning("MT5 fetch returned no bars")
            return MarketData(symbol=symbol, timeframe=timeframe, timezone=timezone, candles=())

        df = pd.concat(frames, ignore_index=True)
        df = df.drop_duplicates(subset=["Date"]).sort_values("Date").reset_index(drop=True)
        market = dataframe_to_market_data(
            df,
            symbol=symbol,
            timeframe=timeframe,
            timezone=timezone,
            start=start,
            end=end,
        )
        logger.info("Fetched MT5 | candles=%s", market.total_candles)
        return market

    def _iter_chunks(self, start: datetime, end: datetime) -> Iterator[tuple[datetime, datetime]]:
        unit = str(self._chunk.get("unit", "year")).lower()
        step = int(self._chunk.get("step", 1))
        cursor = start
        while cursor < end:
            if unit == "month":
                nxt = _add_months(cursor, step)
            else:
                nxt = cursor.replace(year=cursor.year + step)
            nxt = min(nxt, end)
            yield cursor, nxt
            cursor = nxt
