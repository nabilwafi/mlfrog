"""Unit tests for market context layer (Sprint 9)."""

from __future__ import annotations

import tempfile
import unittest
from datetime import datetime, timedelta
from pathlib import Path
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd

from data.entities.candle import Candle
from data.entities.market_data import MarketData
from data.repositories.market_repository import MarketRepository
from market_context.builders.h4_context_builder import H4ContextBuilder
from market_context.repositories.context_repository import ContextRepository
from market_context.services.context_join_service import ContextJoinService
from market_context.services.timeframe_utils import available_at
from market_context.usecases.run_market_context import RunMarketContextUseCase

UTC = ZoneInfo("UTC")


def _candles(start: datetime, n: int, *, hours: int) -> list[Candle]:
    rng = np.random.default_rng(0)
    out: list[Candle] = []
    price = 1800.0
    for i in range(n):
        ts = start + timedelta(hours=hours * i)
        o = price
        c = price + float(rng.normal(0, 2))
        h = max(o, c) + abs(float(rng.normal(1, 0.5)))
        l = min(o, c) - abs(float(rng.normal(1, 0.5)))
        out.append(
            Candle(
                timestamp=ts,
                open=o,
                high=h,
                low=l,
                close=c,
                tick_volume=100.0,
                spread=0.0,
                real_volume=0.0,
            )
        )
        price = c
    return out


class BuilderTests(unittest.TestCase):
    def test_seven_context_features(self) -> None:
        start = datetime(2020, 1, 1, tzinfo=UTC)
        rows = []
        for i, c in enumerate(_candles(start, 80, hours=4)):
            rows.append(
                {
                    "timestamp": c.timestamp,
                    "open": c.open,
                    "high": c.high,
                    "low": c.low,
                    "close": c.close,
                }
            )
        frame = pd.DataFrame(rows)
        out, specs = H4ContextBuilder().build(frame)
        self.assertEqual(len(specs), 7)
        names = [s.name for s in specs]
        self.assertTrue(all(n.startswith("ctx_h4_") for n in names))
        self.assertIn("ctx_h4_trend_direction", names)
        self.assertIn("ctx_h4_swing_quality", names)
        # after warmup some rows finite
        finite = out.dropna()
        self.assertGreater(len(finite), 10)


class JoinLeakageTests(unittest.TestCase):
    def test_incomplete_h4_bar_not_visible(self) -> None:
        # H4 bar opens 00:00, available at 04:00
        h4 = pd.DataFrame(
            {
                "timestamp": [pd.Timestamp("2020-01-01T00:00:00Z")],
                "ctx_h4_trend_direction": [0.9],
            }
        )
        # H1 at 02:00 must NOT see it; H1 at 04:00 may
        base = pd.Series(
            [
                pd.Timestamp("2020-01-01T02:00:00Z"),
                pd.Timestamp("2020-01-01T04:00:00Z"),
            ]
        )
        joined = ContextJoinService().join(
            base, h4, context_timeframe="H4", context_cols=["ctx_h4_trend_direction"]
        )
        self.assertTrue(pd.isna(joined.loc[0, "ctx_h4_trend_direction"]))
        self.assertAlmostEqual(float(joined.loc[1, "ctx_h4_trend_direction"]), 0.9)

    def test_available_at_h4(self) -> None:
        ts = pd.Series([pd.Timestamp("2020-01-01T00:00:00Z")])
        got = available_at(ts, "H4").iloc[0]
        self.assertEqual(got, pd.Timestamp("2020-01-01T04:00:00Z"))


class PipelineE2ETests(unittest.TestCase):
    def test_writes_artifacts(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            raw = root / "raw"
            feats = root / "features"
            ctx_root = root / "context"
            ds_root = root / "datasets"

            start = datetime(2020, 1, 1, tzinfo=UTC)
            h4 = MarketData(
                symbol="XAUUSD",
                timeframe="H4",
                timezone="UTC",
                candles=tuple(_candles(start, 120, hours=4)),
            )
            h1 = MarketData(
                symbol="XAUUSD",
                timeframe="H1",
                timezone="UTC",
                candles=tuple(_candles(start, 480, hours=1)),
            )
            MarketRepository(raw).save_parquet(h4)
            MarketRepository(raw).save_parquet(h1)

            # Minimal H1 feature matrix + labels for dataset v2
            h1_ts = [c.timestamp for c in h1.candles]
            feat_dir = feats / "XAUUSD" / "H1"
            feat_dir.mkdir(parents=True)
            pd.DataFrame(
                {
                    "timestamp": h1_ts,
                    "ema20_distance_atr": np.linspace(-1, 1, len(h1_ts)),
                }
            ).to_parquet(feat_dir / "feature_matrix.parquet", index=False)
            labels = pd.DataFrame(
                {
                    "timestamp": h1_ts,
                    "label": np.where(np.arange(len(h1_ts)) % 3 == 0, 1, -1),
                }
            )

            out = RunMarketContextUseCase(
                market_repo=MarketRepository(raw),
                context_repo=ContextRepository(ctx_root),
                config={
                    "run_feature_engineering": False,
                    "build_dataset_v2": True,
                    "drop_na": True,
                    "feature_version": "v2",
                },
                features_output_root=feats,
                datasets_root=ds_root,
                dataset_config={
                    "splits": {
                        "train": {"start": "2020-01-01", "end": "2020-12-31"},
                    }
                },
            ).execute(
                symbol="XAUUSD",
                base_timeframe="H1",
                context_timeframe="H4",
                labels_by_side={"long": labels},
                h1_feature_matrix_path=feat_dir / "feature_matrix.parquet",
            )

            self.assertTrue((out / "context.parquet").is_file())
            self.assertTrue((out / "context_metadata.json").is_file())
            self.assertTrue((out / "context_report.md").is_file())
            self.assertTrue((out / "context_features.csv").is_file())
            self.assertTrue((ds_root / "XAUUSD" / "H1" / "long" / "v2" / "train.parquet").is_file())
            joined = pd.read_parquet(out / "context.parquet")
            self.assertIn("ctx_h4_trend_direction", joined.columns)
            self.assertGreater(joined["ctx_h4_trend_direction"].notna().sum(), 0)


if __name__ == "__main__":
    unittest.main()
