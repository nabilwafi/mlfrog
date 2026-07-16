"""Unit tests for feature engineering layer."""

from __future__ import annotations

import math
import tempfile
import unittest
from datetime import datetime, timedelta
from pathlib import Path
from zoneinfo import ZoneInfo

import pandas as pd

from data.entities.candle import Candle
from data.entities.market_data import MarketData
from features.exceptions import FeatureValidationError
from features.indicators.atr import ATR
from features.indicators.ema import EMA
from features.indicators.rsi import RSI
from features.indicators.returns import LogReturns, Returns
from features.interactions.body_ratio import BodyRatio
from features.pipelines.feature_pipeline import FeaturePipeline, market_data_to_frame
from features.registry.indicator_registry import IndicatorRegistry
from features.repositories.feature_repository import FeatureRepository
from features.validators.feature_validator import FeatureValidator

UTC = ZoneInfo("UTC")


def _synth_market(n: int = 80) -> MarketData:
    candles = []
    t0 = datetime(2024, 1, 1, 0, 0, tzinfo=UTC)
    price = 2000.0
    for i in range(n):
        o = price
        c = price + (1.0 if i % 2 == 0 else -0.5)
        h = max(o, c) + 0.8
        l = min(o, c) - 0.8
        candles.append(
            Candle(
                timestamp=t0 + timedelta(hours=i),
                open=o,
                high=h,
                low=l,
                close=c,
                tick_volume=100.0,
                spread=2.0,
                real_volume=0.0,
            )
        )
        price = c
    return MarketData(symbol="XAUUSD", timeframe="H1", timezone="UTC", candles=tuple(candles))


_MIN_CFG = {
    "feature_version": "v1",
    "pipeline_version": "1.0",
    "drop_na": True,
    "indicators": [
        {"name": "ema", "params": {"period": 10, "column": "close"}},
        {"name": "atr", "params": {"period": 5}},
        {"name": "rsi", "params": {"period": 5}},
        {"name": "returns", "params": {"periods": 1}},
        {"name": "candle", "params": {}},
    ],
    "interactions": [
        {
            "name": "ema_distance",
            "params": {
                "ema_column": "ema_10_close",
                "price_column": "close",
                "scale_column": "atr_5",
            },
        },
        {"name": "body_ratio", "params": {}},
        {"name": "wick_ratio", "params": {}},
    ],
}


class TestIndicators(unittest.TestCase):
    def setUp(self) -> None:
        IndicatorRegistry.discover()
        self.df = market_data_to_frame(_synth_market(60))

    def test_ema(self) -> None:
        s = EMA({"period": 10, "column": "close"}).calculate(self.df)
        self.assertEqual(s.name, "ema_10_close")
        self.assertFalse(s.iloc[20:].isna().any())

    def test_rsi_bounds(self) -> None:
        s = RSI({"period": 14}).calculate(self.df)
        valid = s.dropna()
        self.assertTrue(((valid >= 0) & (valid <= 100)).all())

    def test_atr_positive(self) -> None:
        s = ATR({"period": 14}).calculate(self.df)
        self.assertTrue((s.dropna() > 0).all())

    def test_returns(self) -> None:
        s = Returns({"periods": 1}).calculate(self.df)
        self.assertTrue(math.isnan(s.iloc[0]))

    def test_log_returns(self) -> None:
        s = LogReturns({"periods": 1}).calculate(self.df)
        self.assertEqual(len(s), len(self.df))

    def test_body_ratio(self) -> None:
        s = BodyRatio({}).calculate(self.df)
        self.assertTrue(((s.dropna() >= 0) & (s.dropna() <= 1.0001)).all())

    def test_registry_create(self) -> None:
        ind = IndicatorRegistry.create("ema", {"period": 5, "column": "close"})
        self.assertEqual(ind.name(), "ema")


class TestFeaturePipeline(unittest.TestCase):
    def test_run(self) -> None:
        pipe = FeaturePipeline(_MIN_CFG)
        fs, stats = pipe.run(_synth_market(80))
        self.assertGreater(fs.feature_count, 0)
        self.assertEqual(fs.row_count, stats["row_count"])
        self.assertGreater(stats["nan_removed"], 0)
        self.assertIn("ema_10_close", fs.feature_names)


class TestFeatureValidator(unittest.TestCase):
    def test_rejects_nan(self) -> None:
        pipe = FeaturePipeline({**_MIN_CFG, "drop_na": False})
        # Force build without drop — validator in pipeline uses allow after drop;
        # test validator directly
        from features.entities.feature import Feature
        from features.entities.feature_set import FeatureSet
        from datetime import timezone

        fs = FeatureSet(
            symbol="XAUUSD",
            timeframe="H1",
            feature_version="v1",
            pipeline_version="1.0",
            created_at=datetime.now(tz=timezone.utc),
            features=(
                Feature(
                    name="bad",
                    dtype="float64",
                    values=(1.0, float("nan"), 2.0),
                    metadata={},
                ),
            ),
            timestamps=(
                datetime(2024, 1, 1, tzinfo=UTC),
                datetime(2024, 1, 2, tzinfo=UTC),
                datetime(2024, 1, 3, tzinfo=UTC),
            ),
        )
        with self.assertRaises(FeatureValidationError):
            FeatureValidator(allow_nan=False).validate(fs)

    def test_rejects_duplicate_names(self) -> None:
        from features.entities.feature import Feature
        from features.entities.feature_set import FeatureSet
        from datetime import timezone

        f = Feature(name="x", dtype="float64", values=(1.0,), metadata={})
        # FeatureSet allows constructing then validator catches duplicates
        fs = FeatureSet(
            symbol="XAUUSD",
            timeframe="H1",
            feature_version="v1",
            pipeline_version="1.0",
            created_at=datetime.now(tz=timezone.utc),
            features=(f, Feature(name="x", dtype="float64", values=(2.0,), metadata={})),
            timestamps=(datetime(2024, 1, 1, tzinfo=UTC),),
        )
        with self.assertRaises(FeatureValidationError):
            FeatureValidator().validate(fs)


class TestFeatureRepository(unittest.TestCase):
    def test_roundtrip(self) -> None:
        pipe = FeaturePipeline(_MIN_CFG)
        fs, _ = pipe.run(_synth_market(80))
        with tempfile.TemporaryDirectory() as tmp:
            repo = FeatureRepository(tmp)
            path = repo.save_parquet(fs)
            self.assertTrue(path.is_file())
            loaded = repo.load_parquet("XAUUSD", "H1", "v1", timezone="UTC")
            self.assertEqual(loaded.feature_count, fs.feature_count)
            self.assertEqual(loaded.row_count, fs.row_count)
            self.assertEqual(loaded.feature_names, fs.feature_names)


if __name__ == "__main__":
    unittest.main()
