"""Unit tests for portfolio backtest."""

from __future__ import annotations

import tempfile
import unittest
from datetime import datetime, timedelta
from pathlib import Path
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd

from research.portfolio_backtest.repositories import PortfolioBacktestRepository
from research.portfolio_backtest.services.engine import prepare_trade_universe, run_portfolio
from research.portfolio_backtest.usecases import RunPortfolioBacktestUseCase

UTC = ZoneInfo("UTC")


class EngineTests(unittest.TestCase):
    def test_risk_frac_compounds(self) -> None:
        t0 = datetime(2024, 1, 1, tzinfo=UTC)
        rows = []
        for i in range(20):
            rows.append(
                {
                    "timestamp": t0 + timedelta(hours=i * 8),
                    "valid_year": 2024,
                    "side": "long",
                    "entry_price": 2000.0,
                    "net_return": 0.01 if i % 2 == 0 else -0.005,
                    "meta_proba": 0.5,
                    "atr_entry": 4.0,
                }
            )
        df = pd.DataFrame(rows)
        uni = prepare_trade_universe(df, system="primary_plus_meta", meta_thr=0.45)
        log, curve = run_portfolio(
            uni,
            starting_equity=80.0,
            mode="risk",
            risk_pct=0.01,
            enforce_volume_min=False,
        )
        self.assertFalse(log.empty)
        self.assertGreater(float(curve["equity"].iloc[-1]), 0)


class E2ETests(unittest.TestCase):
    def test_writes(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            rng = np.random.default_rng(0)
            t0 = datetime(2023, 1, 1, tzinfo=UTC)
            rows = []
            for year in (2023, 2024):
                for i in range(40):
                    y = int(rng.random() < 0.5)
                    rows.append(
                        {
                            "timestamp": t0.replace(year=year) + timedelta(hours=i * 6),
                            "valid_year": year,
                            "side": "long" if i % 2 == 0 else "short",
                            "entry_price": 2000.0 + i,
                            "net_return": 0.005 if y else -0.004,
                            "meta_proba": 0.3 + 0.4 * y + rng.normal(0, 0.05),
                            "atr_entry": 3.0 + rng.random(),
                            "meta_label": y,
                        }
                    )
            oof = pd.DataFrame(rows)
            oof_path = root / "oof.parquet"
            oof.to_parquet(oof_path, index=False)
            out = RunPortfolioBacktestUseCase(
                PortfolioBacktestRepository(root / "out"),
                config={
                    "starting_equity": 80.0,
                    "meta_threshold": 0.45,
                    "n_monte_carlo": 50,
                    "random_seed": 1,
                    "preferred_scenario": "C_risk_1pct_frac",
                },
            ).execute(symbol="XAUUSD", timeframe="H1", oof_path=oof_path)
            self.assertTrue((out / "backtest_report.md").is_file())
            self.assertTrue((out / "risk_report.md").is_file())
            self.assertTrue((out / "portfolio_metrics.csv").is_file())
            self.assertTrue((out / "trade_log.csv").is_file())
            self.assertTrue((out / "equity_curve.csv").is_file())


if __name__ == "__main__":
    unittest.main()
