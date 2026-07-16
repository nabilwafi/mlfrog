"""Series-level candle validation."""

from __future__ import annotations

import logging
from datetime import timedelta
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError

from data.entities.candle import Candle
from data.entities.market_data import MarketData
from data.exceptions import ValidationError

logger = logging.getLogger(__name__)

_TIMEFRAME_DELTA: dict[str, timedelta] = {
    "M1": timedelta(minutes=1),
    "M5": timedelta(minutes=5),
    "M15": timedelta(minutes=15),
    "M30": timedelta(minutes=30),
    "H1": timedelta(hours=1),
    "H4": timedelta(hours=4),
    "D1": timedelta(days=1),
    "W1": timedelta(days=7),
    "MN1": timedelta(days=28),
}


class CandleValidator:
    """Validates MarketData series integrity."""

    def __init__(self, *, check_continuity: bool = False) -> None:
        self._check_continuity = check_continuity

    def validate(self, market_data: MarketData) -> None:
        logger.info(
            "Validating MarketData | symbol=%s timeframe=%s candles=%s",
            market_data.symbol,
            market_data.timeframe,
            market_data.total_candles,
        )
        try:
            tz = ZoneInfo(market_data.timezone)
        except ZoneInfoNotFoundError as exc:
            logger.error("Validation failed | unknown timezone=%s", market_data.timezone)
            raise ValidationError(f"unknown timezone: {market_data.timezone!r}") from exc

        candles = market_data.candles
        if not candles:
            logger.error("Validation failed | empty candles")
            raise ValidationError("MarketData contains no candles")

        seen: set = set()
        prev_ts = None
        for i, c in enumerate(candles):
            if c.timestamp is None:
                logger.error("Validation failed | missing timestamp index=%s", i)
                raise ValidationError(f"missing timestamp at index {i}")
            if c.timestamp.tzinfo is None:
                logger.error("Validation failed | naive timestamp index=%s", i)
                raise ValidationError(f"timezone-naive timestamp at index {i}")
            # Timezone consistency: candle must match declared zone offset at that instant
            localized = c.timestamp.astimezone(tz)
            if c.timestamp.utcoffset() != localized.utcoffset():
                logger.error("Validation failed | timezone mismatch index=%s", i)
                raise ValidationError(
                    f"timezone inconsistency at index {i}: expected {market_data.timezone}"
                )
            if c.timestamp in seen:
                logger.error("Validation failed | duplicate timestamp=%s", c.timestamp)
                raise ValidationError(f"duplicate timestamp: {c.timestamp.isoformat()}")
            seen.add(c.timestamp)
            if prev_ts is not None and c.timestamp <= prev_ts:
                logger.error("Validation failed | non-ascending at index=%s", i)
                raise ValidationError(
                    f"timestamps not strictly ascending at index {i}: "
                    f"{prev_ts.isoformat()} -> {c.timestamp.isoformat()}"
                )
            prev_ts = c.timestamp
            self._assert_ohlc(c, i)

        if self._check_continuity:
            self._assert_continuity(market_data)

        logger.info(
            "Validation success | symbol=%s timeframe=%s candles=%s",
            market_data.symbol,
            market_data.timeframe,
            market_data.total_candles,
        )

    @staticmethod
    def _assert_ohlc(c: Candle, index: int) -> None:
        if c.open < 0 or c.high < 0 or c.low < 0 or c.close < 0:
            logger.error("Validation failed | negative OHLC index=%s", index)
            raise ValidationError(f"invalid OHLC (negative) at index {index}")
        if c.high < c.open or c.high < c.close or c.low > c.open or c.low > c.close:
            logger.error("Validation failed | OHLC relationship index=%s", index)
            raise ValidationError(f"invalid OHLC relationship at index {index}")
        if c.tick_volume < 0 or c.real_volume < 0:
            logger.error("Validation failed | negative volume index=%s", index)
            raise ValidationError(f"negative volume at index {index}")
        if c.spread < 0:
            logger.error("Validation failed | negative spread index=%s", index)
            raise ValidationError(f"negative spread at index {index}")

    def _assert_continuity(self, market_data: MarketData) -> None:
        delta = _TIMEFRAME_DELTA.get(market_data.timeframe.upper())
        if delta is None:
            raise ValidationError(
                f"cannot check continuity for unknown timeframe {market_data.timeframe!r}"
            )
        candles = market_data.candles
        gaps: list[str] = []
        for i in range(1, len(candles)):
            expected = candles[i - 1].timestamp + delta
            actual = candles[i].timestamp
            if actual > expected + timedelta(seconds=1):
                gaps.append(f"{candles[i - 1].timestamp.isoformat()} -> {actual.isoformat()}")
        if gaps:
            sample = "; ".join(gaps[:5])
            more = f" (+{len(gaps) - 5} more)" if len(gaps) > 5 else ""
            logger.error("Validation failed | continuity gaps=%s", len(gaps))
            raise ValidationError(f"missing timestamps / continuity gaps: {sample}{more}")
