"""Unit tests for stationary feature engineering."""

from __future__ import annotations

import tempfile
import unittest
from datetime import datetime, timedelta
from pathlib import Path
from zoneinfo import ZoneInfo

from data.entities.candle import Candle
from data.entities.market_data import MarketData
from feature_engineering.registry.feature_registry import FeatureRegistry
from feature_engineering.services.feature_engineering_service import FeatureEngineeringService

UTC = ZoneInfo("UTC")


def _market(n: int = 400) -> MarketData:
    candles = []
    t0 = datetime(2020, 1, 1, tzinfo=UTC)
    price = 1800.0
    for i in range(n):
        o = price
        c = price + (0.3 if i % 3 else -0.2)
        h = max(o, c) + 1.5
        l = min(o, c) - 1.5
        candles.append(
            Candle(
                timestamp=t0 + timedelta(hours=i),
                open=o,
                high=h,
                low=l,
                close=c,
                tick_volume=10.0,
                spread=1.0,
                real_volume=0.0,
            )
        )
        price = c
    return MarketData(symbol="XAUUSD", timeframe="H1", timezone="UTC", candles=tuple(candles))


class FeatureEngineeringTests(unittest.TestCase):
    def setUp(self) -> None:
        FeatureRegistry.clear()
        FeatureRegistry.discover()

    def test_registry(self) -> None:
        for key in ("trend", "volatility", "momentum", "candle", "session", "statistical"):
            self.assertIn(key, FeatureRegistry.keys())

    def test_end_to_end(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            svc = FeatureEngineeringService(
                {
                    "drop_na": True,
                    "builder_params": {
                        "volatility": {
                            "atr_percentile_window": 50,
                            "rolling_volatility_window": 10,
                        },
                        "momentum": {"zscore_window": 30, "rank_window": 30},
                        "candle": {"rank_window": 30},
                        "statistical": {"window": 10},
                    },
                },
                output_root=Path(tmp),
            )
            matrix, metadata, out = svc.run(_market(350))
            names = {m.name for m in metadata}
            for required in (
                "ema20_distance_atr",
                "atr_percent",
                "macd_normalized",
                "body_percent",
                "hour_sin",
                "rolling_zscore",
            ):
                self.assertIn(required, names)
                self.assertIn(required, matrix.columns)
            self.assertNotIn("ema_20_close", matrix.columns)
            self.assertNotIn("atr_14", matrix.columns)
            self.assertNotIn("hour", matrix.columns)
            self.assertTrue((out / "feature_matrix.parquet").is_file())
            self.assertTrue((out / "feature_metadata.json").is_file())
            self.assertTrue((out / "feature_report.md").is_file())
            self.assertTrue(all(m.stationary or m.name == "ema_trend_duration" for m in metadata))


if __name__ == "__main__":
    unittest.main()
