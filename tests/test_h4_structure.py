"""Unit tests for H4 structure enhancement (Sprint 11)."""

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
from market_context.builders.h4_structure_builder import (
    ALL_STRUCTURE_FEATURES,
    H4StructureBuilder,
    NEW_STRUCTURE_FEATURES,
)
from market_context.services.context_join_service import ContextJoinService
from research.h4_structure.repositories.h4_structure_repository import H4StructureRepository
from research.h4_structure.services.experiment_catalog import resolve_experiments
from research.h4_structure.usecases.run_h4_structure import RunH4StructureUseCase

UTC = ZoneInfo("UTC")


def _candles(start: datetime, n: int, *, hours: int) -> tuple[Candle, ...]:
    rng = np.random.default_rng(11)
    out: list[Candle] = []
    price = 1900.0
    for i in range(n):
        ts = start + timedelta(hours=hours * i)
        o = price
        c = price + float(rng.normal(0, 3))
        h = max(o, c) + abs(float(rng.normal(1.5, 0.5)))
        l = min(o, c) - abs(float(rng.normal(1.5, 0.5)))
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
    return tuple(out)


class BuilderTests(unittest.TestCase):
    def test_produces_all_structure_features(self) -> None:
        start = datetime(2020, 1, 1, tzinfo=UTC)
        rows = [
            {
                "timestamp": c.timestamp,
                "open": c.open,
                "high": c.high,
                "low": c.low,
                "close": c.close,
            }
            for c in _candles(start, 80, hours=4)
        ]
        out, specs = H4StructureBuilder().build(pd.DataFrame(rows))
        names = {s.name for s in specs}
        for f in ALL_STRUCTURE_FEATURES:
            self.assertIn(f, names)
            self.assertIn(f, out.columns)
        self.assertEqual(len(NEW_STRUCTURE_FEATURES), 16)
        self.assertGreater(len(out.dropna()), 20)

    def test_causal_join_hides_incomplete_bar(self) -> None:
        h4 = pd.DataFrame(
            {
                "timestamp": [pd.Timestamp("2020-01-01T00:00:00Z")],
                "ctx_h4_swing_strength": [0.8],
            }
        )
        base = pd.Series(
            [
                pd.Timestamp("2020-01-01T02:00:00Z"),
                pd.Timestamp("2020-01-01T04:00:00Z"),
            ]
        )
        joined = ContextJoinService().join(
            base, h4, context_timeframe="H4", context_cols=["ctx_h4_swing_strength"]
        )
        self.assertTrue(pd.isna(joined.loc[0, "ctx_h4_swing_strength"]))
        self.assertAlmostEqual(float(joined.loc[1, "ctx_h4_swing_strength"]), 0.8)


class CatalogTests(unittest.TestCase):
    def test_four_experiments(self) -> None:
        exps = resolve_experiments(["a", "b"])
        self.assertEqual(len(exps), 4)
        self.assertEqual(exps[0].experiment_id, "baseline")
        self.assertEqual(exps[-1].experiment_id, "all_structure")


class E2ETests(unittest.TestCase):
    def test_pipeline_writes(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            raw = root / "raw"
            feats = root / "features" / "XAUUSD" / "H1"
            feats.mkdir(parents=True)

            # Multi-year synthetic data for walk-forward window
            h4_rows: list[Candle] = []
            h1_rows: list[Candle] = []
            for year in range(2020, 2025):
                start = datetime(year, 1, 1, tzinfo=UTC)
                h4_rows.extend(_candles(start, 40, hours=4))
                h1_rows.extend(_candles(start, 160, hours=1))

            MarketRepository(raw).save_parquet(
                MarketData(
                    symbol="XAUUSD",
                    timeframe="H4",
                    timezone="UTC",
                    candles=tuple(h4_rows),
                )
            )
            MarketRepository(raw).save_parquet(
                MarketData(
                    symbol="XAUUSD",
                    timeframe="H1",
                    timezone="UTC",
                    candles=tuple(h1_rows),
                )
            )
            ts = [c.timestamp for c in h1_rows]
            rng = np.random.default_rng(2)
            pd.DataFrame(
                {
                    "timestamp": ts,
                    "ema20_distance_atr": rng.normal(0, 1, len(ts)),
                    "rsi_percentile": rng.random(len(ts)),
                }
            ).to_parquet(feats / "feature_matrix.parquet", index=False)
            labels = {
                "long": pd.DataFrame(
                    {
                        "timestamp": ts,
                        "label": rng.choice([-1, 0, 1], size=len(ts), p=[0.4, 0.2, 0.4]),
                    }
                )
            }
            results, out = RunH4StructureUseCase(
                H4StructureRepository(root / "research" / "h4_structure"),
                MarketRepository(raw),
                config={
                    "algorithm": "lightgbm",
                    "trainer_params": {
                        "num_boost_round": 25,
                        "early_stopping_rounds": 8,
                        "verbosity": -1,
                    },
                    "target_mode": "exclude_timeout",
                    "threshold": 0.5,
                    "random_seed": 42,
                    "windows": [
                        {
                            "train_start_year": 2020,
                            "train_end_year": 2022,
                            "valid_year": 2023,
                        }
                    ],
                },
            ).execute(
                symbol="XAUUSD",
                base_timeframe="H1",
                context_timeframe="H4",
                sides=["long"],
                feature_matrix_path=feats / "feature_matrix.parquet",
                labels_by_side=labels,
            )
            self.assertEqual(len(results), 4)
            self.assertTrue((out / "structure_features.parquet").is_file())
            self.assertTrue((out / "ablation_results.csv").is_file())
            self.assertTrue((out / "feature_statistics.csv").is_file())
            self.assertTrue((out / "structure_report.md").is_file())
            self.assertTrue((out / "charts" / "roc_comparison.png").is_file())


if __name__ == "__main__":
    unittest.main()
