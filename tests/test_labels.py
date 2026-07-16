"""Unit tests for label (multi-side) and dataset builder layers."""

from __future__ import annotations

import tempfile
import unittest
from datetime import datetime, timedelta, timezone
from pathlib import Path
from zoneinfo import ZoneInfo

from data.entities.candle import Candle
from data.entities.market_data import MarketData
from data.repositories.market_repository import MarketRepository
from datasets.builders.dataset_builder import DatasetBuilder
from datasets.pipelines.dataset_pipeline import DatasetPipeline
from datasets.repositories.dataset_repository import DatasetRepository
from datasets.services.dataset_service import DatasetService
from datasets.validators.dataset_validator import DatasetValidator
from features.entities.feature import Feature
from features.entities.feature_set import FeatureSet
from features.pipelines.feature_pipeline import FeaturePipeline
from features.repositories.feature_repository import FeatureRepository
from labels.entities.label import Label
from labels.entities.label_set import LabelSet
from labels.exceptions import LabelValidationError
from labels.pipelines.label_pipeline import LabelPipeline
from labels.registry.strategy_registry import StrategyRegistry
from labels.repositories.label_repository import LabelRepository
from labels.services.label_service import LabelService
from labels.strategies.triple_barrier import TripleBarrierStrategy
from labels.validators.label_validator import LabelValidator

UTC = ZoneInfo("UTC")


def _synth(n: int = 200, *, trend: float = 0.4) -> MarketData:
    candles = []
    t0 = datetime(2020, 1, 1, tzinfo=UTC)
    price = 2000.0
    for i in range(n):
        o = price
        c = price + trend
        h = max(o, c) + 1.2
        l = min(o, c) - 1.2
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


_TB = {
    "strategy": "triple_barrier",
    "label_version": "v1",
    "sides": ["long", "short"],
    "params": {
        "horizon": 8,
        "tp_atr_mult": 2.0,
        "sl_atr_mult": 1.5,
        "atr_period": 14,
    },
}

_FEAT = {
    "feature_version": "v1",
    "pipeline_version": "1.0",
    "drop_na": True,
    "indicators": [
        {"name": "ema", "params": {"period": 10, "column": "close"}},
        {"name": "atr", "params": {"period": 5}},
        {"name": "returns", "params": {"periods": 1}},
    ],
    "interactions": [],
}

_DS = {
    "feature_version": "v1",
    "label_version": "v1",
    "strategy": "triple_barrier",
    "sides": ["long"],
    "drop_na": True,
    "splits": {
        "train": {"start": "2020-01-01", "end": "2020-01-05"},
        "validation": {"start": "2020-01-06", "end": "2020-01-07"},
        "test": {"start": "2020-01-08", "end": "2020-01-09"},
        "sealed": {"start": "2020-01-10", "end": "2030-01-01"},
    },
}


class TestTripleBarrierMultiSide(unittest.TestCase):
    def setUp(self) -> None:
        StrategyRegistry.discover()

    def test_long_and_short(self) -> None:
        m = _synth(100)
        long_labs = TripleBarrierStrategy({**_TB["params"], "side": "long"}).generate(m)
        short_labs = TripleBarrierStrategy({**_TB["params"], "side": "short"}).generate(m)
        self.assertGreater(len(long_labs), 0)
        self.assertGreater(len(short_labs), 0)
        self.assertTrue(all(x.side == "long" for x in long_labs))
        self.assertTrue(all(x.side == "short" for x in short_labs))
        self.assertLess(long_labs[0].sl_price, long_labs[0].entry_price)
        self.assertGreater(short_labs[0].sl_price, short_labs[0].entry_price)


class TestLabelValidator(unittest.TestCase):
    def test_duplicate_timestamps(self) -> None:
        ts = datetime(2024, 1, 1, tzinfo=UTC)
        lab = Label(
            timestamp=ts,
            entry_price=100.0,
            tp_price=102.0,
            sl_price=99.0,
            expire_timestamp=datetime(2024, 1, 2, tzinfo=UTC),
            holding_bars=1,
            realized_return=0.01,
            exit_reason="TP",
            side="long",
            label=1,
        )
        ls = LabelSet(
            symbol="XAUUSD",
            timeframe="H1",
            strategy="triple_barrier",
            side="long",
            label_version="v1",
            created_at=datetime.now(tz=timezone.utc),
            labels=(lab, lab),
        )
        with self.assertRaises(LabelValidationError):
            LabelValidator().validate(ls)


class TestLabelServiceSides(unittest.TestCase):
    def test_run_both_sides(self) -> None:
        market = _synth(90)
        with tempfile.TemporaryDirectory() as tmp:
            raw = Path(tmp) / "raw"
            lab = Path(tmp) / "labels"
            raw.mkdir()
            MarketRepository(raw).save_parquet(market)
            svc = LabelService(
                MarketRepository(raw), LabelRepository(lab), _TB, timezone="UTC"
            )
            results = svc.run("XAUUSD", "H1", sides=["long", "short"])
            self.assertEqual(len(results), 2)
            sides = {r[0].side for r in results}
            self.assertEqual(sides, {"long", "short"})
            for ls, path, _ in results:
                self.assertTrue(path.is_file())
                self.assertIn(ls.side, str(path))


class TestDatasetBuilder(unittest.TestCase):
    def test_align_and_split(self) -> None:
        market = _synth(250)
        feats, _ = FeaturePipeline(_FEAT).run(market)
        labels, _ = LabelPipeline(_TB).run(market, side="long")
        builder = DatasetBuilder(_DS)
        frame = builder.build_aligned_frame(feats, labels)
        self.assertGreater(len(frame), 0)
        splits = builder.split_time(
            frame,
            symbol="XAUUSD",
            timeframe="H1",
            side="long",
            strategy="triple_barrier",
            feature_version="v1",
            label_version="v1",
        )
        self.assertIn("train", splits)
        DatasetValidator().validate(splits["train"])


class TestDatasetRepository(unittest.TestCase):
    def test_roundtrip(self) -> None:
        market = _synth(250)
        feats, _ = FeaturePipeline(_FEAT).run(market)
        labels, _ = LabelPipeline(_TB).run(market, side="long")
        splits = DatasetPipeline(_DS).run(feats, labels)
        with tempfile.TemporaryDirectory() as tmp:
            repo = DatasetRepository(tmp)
            ds = splits["train"]
            path = repo.save_parquet(ds)
            loaded = repo.load_parquet("XAUUSD", "H1", "long", "train")
            self.assertEqual(loaded.size, ds.size)
            self.assertTrue(path.is_file())


class TestDatasetService(unittest.TestCase):
    def test_end_to_end(self) -> None:
        market = _synth(250)
        with tempfile.TemporaryDirectory() as tmp:
            raw = Path(tmp) / "raw"
            feat = Path(tmp) / "features"
            lab = Path(tmp) / "labels"
            ds_root = Path(tmp) / "datasets"
            raw.mkdir()
            MarketRepository(raw).save_parquet(market)
            fs, _ = FeaturePipeline(_FEAT).run(market)
            FeatureRepository(feat).save_parquet(fs)
            LabelService(
                MarketRepository(raw), LabelRepository(lab), _TB, timezone="UTC"
            ).run("XAUUSD", "H1", sides=["long"])
            results = DatasetService(
                FeatureRepository(feat),
                LabelRepository(lab),
                DatasetRepository(ds_root),
                _DS,
                timezone="UTC",
            ).run("XAUUSD", "H1", sides=["long"])
            self.assertIn("long", results)
            self.assertGreater(len(results["long"]), 0)


if __name__ == "__main__":
    unittest.main()
