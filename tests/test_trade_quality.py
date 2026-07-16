"""Unit tests for Trade Quality Engine."""

from __future__ import annotations

import tempfile
import unittest
from datetime import datetime, timedelta
from pathlib import Path
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd

from research.trade_quality.repositories import TradeQualityRepository
from research.trade_quality.services.analysis import bucket_performance, monotonicity_ok
from research.trade_quality.services.scorers import loo_trade_quality, prepare_features
from research.trade_quality.usecases import RunTradeQualityUseCase

UTC = ZoneInfo("UTC")


def _fake_panel(n_per_year: int = 40) -> pd.DataFrame:
    rng = np.random.default_rng(7)
    t0 = datetime(2023, 1, 1, tzinfo=UTC)
    rows = []
    for year in (2023, 2024, 2025):
        for i in range(n_per_year):
            # meta_proba weakly predictive -> TQ methods can latch on
            edge = 0.15 * (i / n_per_year)
            y = int(rng.random() < (0.45 + edge))
            rows.append(
                {
                    "timestamp": t0.replace(year=year) + timedelta(hours=i * 6),
                    "valid_year": year,
                    "window": f"v{year}",
                    "side": "long" if i % 2 == 0 else "short",
                    "meta_label": y,
                    "net_return": (0.004 + edge * 0.01) if y else (-0.003 - edge * 0.005),
                    "meta_proba": 0.45 + edge + rng.normal(0, 0.03),
                    "raw_probability": 0.4 + 0.1 * y,
                    "entry_price": 2000.0,
                    "atr_entry": 4.0,
                    "distance_to_sl": 0.003,
                    "initial_rr": 1.33,
                    "holding_bars": 4,
                    "confidence": 30 + 50 * edge + rng.normal(0, 5),
                    "d1_regime": "Strong Bull" if i % 3 == 0 else "Sideways",
                    "d1_trend_dist": float(rng.normal()),
                    "m5_entry_quality": 40 + 40 * edge,
                    "ctx_h4_rejection_strength": float(rng.random()),
                    "ctx_h4_swing_quality": float(rng.random()),
                    "ctx_h4_swing_strength": float(rng.random()),
                    "ctx_h4_distance_from_equilibrium": float(rng.normal()),
                    "session_london": float(i % 3 == 0),
                    "session_newyork": float(i % 3 == 1),
                    "session_asia": float(i % 3 == 2),
                    "session_london_ny_overlap": 0.0,
                    "hour_of_day": float(8 + (i % 10)),
                    "atr_percent": 0.2,
                    "atr_percentile_252": float(rng.random()),
                    "volatility_rank": float(rng.random()),
                    "volatility_regime_score": float(rng.random()),
                    "ema_trend_duration": float(i % 20),
                }
            )
    return pd.DataFrame(rows)


class ScorerTests(unittest.TestCase):
    def test_prepare_and_loo(self) -> None:
        panel = _fake_panel()
        feats = prepare_features(panel)
        self.assertIn("h4_context", feats.columns)
        scored, stab = loo_trade_quality(panel)
        self.assertIn("trade_quality", scored.columns)
        self.assertFalse(stab.empty)
        self.assertTrue(scored["trade_quality"].between(0, 100).all())


class BucketTests(unittest.TestCase):
    def test_buckets(self) -> None:
        panel = _fake_panel()
        scored, _ = loo_trade_quality(panel)
        b = bucket_performance(scored)
        self.assertEqual(len(b), 5)
        mono = monotonicity_ok(b)
        self.assertIn("spearman_bucket_expectancy", mono)


class E2ETests(unittest.TestCase):
    def test_run(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            panel = _fake_panel()
            panel_path = root / "confidence_panel.parquet"
            panel.to_parquet(panel_path, index=False)
            out = RunTradeQualityUseCase(
                TradeQualityRepository(root / "out"),
                config={"starting_equity": 80.0},
            ).execute(
                symbol="XAUUSD",
                timeframe="H1",
                confidence_panel_path=panel_path,
            )
            self.assertTrue((out / "trade_quality_report.md").is_file())
            self.assertTrue((out / "trade_quality_panel.parquet").is_file())
            self.assertTrue((out / "answers.json").is_file())
            self.assertTrue((out / "charts" / "tq_distribution.png").is_file())


if __name__ == "__main__":
    unittest.main()
