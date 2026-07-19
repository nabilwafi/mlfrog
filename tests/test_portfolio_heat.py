"""Unit tests for portfolio heat."""

from __future__ import annotations

import tempfile
import unittest
from datetime import datetime, timedelta
from pathlib import Path
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd

from research.portfolio_heat import HeatPolicy
from research.portfolio_heat.repositories import PortfolioHeatRepository
from research.portfolio_heat.services.engine import run_heat_portfolio
from research.portfolio_heat.services.search import acceptance
from research.portfolio_heat.usecases import RunPortfolioHeatUseCase

UTC = ZoneInfo("UTC")


def _panel(n: int = 80) -> pd.DataFrame:
    rng = np.random.default_rng(3)
    t0 = datetime(2023, 1, 2, 8, tzinfo=UTC)
    rows = []
    for i in range(n):
        y = int(rng.random() < 0.52)
        year = 2023 + (i // 30)
        if year > 2025:
            year = 2025
        rows.append(
            {
                "timestamp": t0 + timedelta(hours=i * 3),
                "valid_year": year,
                "side": "long" if i % 2 == 0 else "short",
                "meta_label": y,
                "net_return": 0.004 if y else -0.003,
                "meta_proba": 0.5 + 0.1 * y,
                "confidence": 55.0 + rng.normal(0, 8),
                "entry_price": 2000.0,
                "atr_entry": 4.0,
                "distance_to_sl": 0.003,
                "holding_bars": 3 + (i % 4),
                "d1_regime": "Strong Bull" if i % 4 else "Sideways",
                "session_london": float(i % 3 == 0),
                "session_newyork": float(i % 3 == 1),
                "session_asia": float(i % 3 == 2),
                "session_london_ny_overlap": 0.0,
                "atr_percentile_252": float(rng.random()),
                "volatility_rank": float(rng.random()),
            }
        )
    return pd.DataFrame(rows)


class EngineTests(unittest.TestCase):
    def test_baseline_runs(self) -> None:
        log, curve, m = run_heat_portfolio(_panel(), HeatPolicy(name="baseline_skip40_flat1"))
        self.assertGreater(m["trades"], 0)
        self.assertIn("cagr", m)

    def test_max_open_skips(self) -> None:
        _, _, base = run_heat_portfolio(_panel(), HeatPolicy(name="b"))
        _, _, tight = run_heat_portfolio(_panel(), HeatPolicy(name="o1", max_open=1))
        self.assertGreaterEqual(base["trades"], tight["trades"])

    def test_acceptance(self) -> None:
        a = acceptance({"cagr": 0.2, "max_drawdown": 0.2}, {"cagr": 0.22, "max_drawdown": 0.19})
        self.assertTrue(a["accepted"])


class E2ETests(unittest.TestCase):
    def test_run(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            p = root / "panel.parquet"
            _panel().to_parquet(p, index=False)
            out = RunPortfolioHeatUseCase(
                PortfolioHeatRepository(root / "out"),
                config={"starting_equity": 80.0, "n_monte_carlo": 20, "n_bootstrap": 50, "n_permutation": 50},
            ).execute(symbol="XAUUSD", timeframe="H1", confidence_panel_path=p)
            self.assertTrue((out / "portfolio_heat_report.md").is_file())
            self.assertTrue((out / "policy_leaderboard.csv").is_file())


if __name__ == "__main__":
    unittest.main()
