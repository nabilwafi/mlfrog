"""Unit tests for mlfrog data layer (stdlib unittest — no pytest required)."""

from __future__ import annotations

import tempfile
import unittest
from datetime import datetime, timedelta
from pathlib import Path
from zoneinfo import ZoneInfo

from mlfrog.data.entities.candle import Candle
from mlfrog.data.entities.market_data import MarketData
from mlfrog.data.exceptions import ValidationError
from mlfrog.data.providers.csv_provider import CSVProvider
from mlfrog.data.providers.parquet_provider import ParquetProvider
from mlfrog.data.repositories.market_repository import MarketRepository
from mlfrog.data.validators.candle_validator import CandleValidator

UTC = ZoneInfo("UTC")


def _candle(ts: datetime, o: float = 100.0, h: float = 101.0, l: float = 99.0, c: float = 100.5) -> Candle:
    return Candle(
        timestamp=ts,
        open=o,
        high=h,
        low=l,
        close=c,
        tick_volume=10.0,
        spread=1.0,
        real_volume=0.0,
    )


class TestCandle(unittest.TestCase):
    def test_valid(self) -> None:
        c = _candle(datetime(2024, 1, 1, 0, 0, tzinfo=UTC))
        self.assertEqual(c.open, 100.0)

    def test_rejects_negative_open(self) -> None:
        with self.assertRaises(ValueError):
            _candle(datetime(2024, 1, 1, tzinfo=UTC), o=-1)

    def test_rejects_high_below_open(self) -> None:
        with self.assertRaises(ValueError):
            _candle(datetime(2024, 1, 1, tzinfo=UTC), o=100, h=99, l=98, c=98.5)

    def test_rejects_naive_timestamp(self) -> None:
        with self.assertRaises(ValueError):
            Candle(
                timestamp=datetime(2024, 1, 1),
                open=1,
                high=2,
                low=0.5,
                close=1.5,
                tick_volume=0,
                spread=0,
                real_volume=0,
            )

    def test_rejects_negative_spread(self) -> None:
        with self.assertRaises(ValueError):
            Candle(
                timestamp=datetime(2024, 1, 1, tzinfo=UTC),
                open=1,
                high=2,
                low=0.5,
                close=1.5,
                tick_volume=0,
                spread=-1,
                real_volume=0,
            )


class TestMarketData(unittest.TestCase):
    def test_properties(self) -> None:
        t0 = datetime(2024, 1, 1, tzinfo=UTC)
        t1 = t0 + timedelta(hours=1)
        md = MarketData(
            symbol="xauusd",
            timeframe="h1",
            timezone="UTC",
            candles=(_candle(t0), _candle(t1)),
        )
        self.assertEqual(md.symbol, "XAUUSD")
        self.assertEqual(md.timeframe, "H1")
        self.assertEqual(md.total_candles, 2)
        self.assertEqual(md.start_time, t0)
        self.assertEqual(md.end_time, t1)

    def test_empty_times_none(self) -> None:
        md = MarketData(symbol="XAUUSD", timeframe="H1", timezone="UTC", candles=())
        self.assertIsNone(md.start_time)
        self.assertIsNone(md.end_time)
        self.assertEqual(md.total_candles, 0)


class TestCandleValidator(unittest.TestCase):
    def setUp(self) -> None:
        self.v = CandleValidator(check_continuity=False)

    def test_pass(self) -> None:
        t0 = datetime(2024, 1, 1, tzinfo=UTC)
        md = MarketData(
            symbol="XAUUSD",
            timeframe="H1",
            timezone="UTC",
            candles=(_candle(t0), _candle(t0 + timedelta(hours=1))),
        )
        self.v.validate(md)

    def test_duplicate(self) -> None:
        t0 = datetime(2024, 1, 1, tzinfo=UTC)
        md = MarketData(
            symbol="XAUUSD",
            timeframe="H1",
            timezone="UTC",
            candles=(_candle(t0), _candle(t0)),
        )
        with self.assertRaises(ValidationError):
            self.v.validate(md)

    def test_non_ascending(self) -> None:
        t0 = datetime(2024, 1, 1, tzinfo=UTC)
        md = MarketData(
            symbol="XAUUSD",
            timeframe="H1",
            timezone="UTC",
            candles=(_candle(t0 + timedelta(hours=2)), _candle(t0)),
        )
        with self.assertRaises(ValidationError):
            self.v.validate(md)

    def test_empty(self) -> None:
        md = MarketData(symbol="XAUUSD", timeframe="H1", timezone="UTC", candles=())
        with self.assertRaises(ValidationError):
            self.v.validate(md)

    def test_continuity_gap(self) -> None:
        v = CandleValidator(check_continuity=True)
        t0 = datetime(2024, 1, 1, tzinfo=UTC)
        md = MarketData(
            symbol="XAUUSD",
            timeframe="H1",
            timezone="UTC",
            candles=(_candle(t0), _candle(t0 + timedelta(hours=3))),
        )
        with self.assertRaises(ValidationError):
            v.validate(md)


class TestMarketRepository(unittest.TestCase):
    def test_parquet_roundtrip(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            repo = MarketRepository(tmp)
            t0 = datetime(2024, 1, 1, tzinfo=UTC)
            md = MarketData(
                symbol="XAUUSD",
                timeframe="H1",
                timezone="UTC",
                candles=(_candle(t0), _candle(t0 + timedelta(hours=1))),
            )
            path = repo.save_parquet(md)
            self.assertTrue(path.is_file())
            loaded = repo.load_parquet("XAUUSD", "H1", timezone="UTC")
            self.assertEqual(loaded.total_candles, 2)
            self.assertEqual(loaded.candles[0].open, 100.0)
            self.assertEqual(repo.list_symbols(), ["XAUUSD"])
            self.assertEqual(repo.list_timeframes("XAUUSD"), ["H1"])

    def test_csv_roundtrip(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            repo = MarketRepository(tmp)
            t0 = datetime(2024, 1, 1, tzinfo=UTC)
            md = MarketData(
                symbol="XAUUSD",
                timeframe="H1",
                timezone="UTC",
                candles=(_candle(t0),),
            )
            repo.save_csv(md)
            loaded = repo.load_csv("XAUUSD", "H1", timezone="UTC")
            self.assertEqual(loaded.total_candles, 1)


class TestProviders(unittest.TestCase):
    def test_csv_provider(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "sample.csv"
            path.write_text(
                "timestamp,open,high,low,close,tick_volume,spread,real_volume\n"
                "2024-01-01T00:00:00+00:00,100,101,99,100.5,10,1,0\n"
                "2024-01-01T01:00:00+00:00,100.5,102,100,101,11,1,0\n",
                encoding="utf-8",
            )
            provider = CSVProvider(path)
            self.assertTrue(provider.health_check())
            md = provider.fetch(
                "XAUUSD",
                "H1",
                datetime(2024, 1, 1),
                datetime(2024, 1, 2),
                timezone="UTC",
            )
            self.assertEqual(md.total_candles, 2)
            provider.close()

    def test_parquet_provider(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            repo = MarketRepository(root)
            t0 = datetime(2024, 1, 1, tzinfo=UTC)
            md = MarketData(
                symbol="XAUUSD",
                timeframe="H1",
                timezone="UTC",
                candles=(_candle(t0), _candle(t0 + timedelta(hours=1))),
            )
            pq = repo.save_parquet(md)
            provider = ParquetProvider(pq)
            self.assertTrue(provider.health_check())
            loaded = provider.fetch(
                "XAUUSD",
                "H1",
                datetime(2024, 1, 1),
                datetime(2024, 1, 2),
                timezone="UTC",
            )
            self.assertEqual(loaded.total_candles, 2)
            provider.close()


if __name__ == "__main__":
    unittest.main()
