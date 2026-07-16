"""Unit tests for confidence layer."""

from __future__ import annotations

import tempfile
import unittest
from datetime import datetime, timedelta
from pathlib import Path
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd

from research.confidence_layer.repositories import ConfidenceLayerRepository
from research.confidence_layer.services.confidence import fit_loo_confidence
from research.confidence_layer.services.mtf_context import build_d1_features
from research.confidence_layer.usecases import RunConfidenceLayerUseCase

UTC = ZoneInfo("UTC")


class D1Tests(unittest.TestCase):
    def test_regimes(self) -> None:
        t0 = datetime(2020, 1, 1, tzinfo=UTC)
        rows = []
        px = 1800.0
        for i in range(120):
            px = px * (1.002 if i > 40 else 1.0)
            rows.append(
                {
                    "timestamp": t0 + timedelta(days=i),
                    "open": px,
                    "high": px + 5,
                    "low": px - 5,
                    "close": px + 1,
                    "tick_volume": 100.0,
                    "spread": 0.0,
                    "real_volume": 0.0,
                }
            )
        d1 = build_d1_features(pd.DataFrame(rows))
        self.assertIn("d1_regime", d1.columns)
        self.assertGreater(len(d1), 50)


class E2ETests(unittest.TestCase):
    def test_run(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            rng = np.random.default_rng(0)
            t0 = datetime(2023, 1, 1, tzinfo=UTC)
            rows = []
            for year in (2023, 2024):
                for i in range(60):
                    y = int(rng.random() < 0.5)
                    t = t0.replace(year=year) + timedelta(hours=i * 6)
                    rows.append(
                        {
                            "timestamp": t,
                            "valid_year": year,
                            "window": f"v{year}",
                            "side": "long" if i % 2 == 0 else "short",
                            "meta_label": y,
                            "net_return": 0.004 if y else -0.003,
                            "meta_proba": 0.5 + 0.2 * y + rng.normal(0, 0.05),
                            "raw_probability": 0.4 + 0.1 * y,
                            "entry_price": 2000.0,
                            "atr_entry": 4.0,
                            "distance_to_sl": 0.003,
                            "initial_rr": 1.33,
                            "holding_bars": 4,
                            "ctx_h4_rejection_strength": float(rng.random()),
                            "ctx_h4_swing_quality": float(rng.random()),
                            "ctx_h4_swing_strength": float(rng.random()),
                            "ctx_h4_distance_from_equilibrium": float(rng.normal()),
                            "atr_percent": 0.2,
                            "mae": 0.002,
                            "mfe": 0.005,
                        }
                    )
            oof = pd.DataFrame(rows)
            oof_path = root / "oof.parquet"
            oof.to_parquet(oof_path, index=False)

            # tiny d1/m5
            d1_rows, m5_rows = [], []
            for i in range(400):
                d1_rows.append(
                    {
                        "timestamp": t0 + timedelta(days=i),
                        "open": 1900 + i,
                        "high": 1905 + i,
                        "low": 1895 + i,
                        "close": 1901 + i,
                        "tick_volume": 1.0,
                        "spread": 0.0,
                        "real_volume": 0.0,
                    }
                )
            for i in range(2000):
                m5_rows.append(
                    {
                        "timestamp": t0 + timedelta(minutes=5 * i),
                        "open": 1900.0,
                        "high": 1901.0,
                        "low": 1899.0,
                        "close": 1900.5,
                        "tick_volume": 1.0,
                        "spread": 1.0,
                        "real_volume": 0.0,
                    }
                )
            d1_path = root / "d1.parquet"
            m5_path = root / "m5.parquet"
            pd.DataFrame(d1_rows).to_parquet(d1_path, index=False)
            pd.DataFrame(m5_rows).to_parquet(m5_path, index=False)

            out = RunConfidenceLayerUseCase(
                ConfidenceLayerRepository(root / "out"),
                config={"meta_gate": 0.45, "starting_equity": 80.0},
            ).execute(
                symbol="XAUUSD",
                timeframe="H1",
                oof_path=oof_path,
                candidates_path=None,
                d1_path=d1_path,
                m5_path=m5_path,
            )
            self.assertTrue((out / "confidence_layer_report.md").is_file())
            self.assertTrue((out / "confidence_panel.parquet").is_file())


if __name__ == "__main__":
    unittest.main()
