"""Unit tests for position management."""

from __future__ import annotations

import tempfile
import unittest
from datetime import datetime, timedelta
from pathlib import Path
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd

from research.position_mgmt.services.simulator import ManageConfig, simulate_one
from research.position_mgmt.services.paths import prepare_market, wilder_atr

UTC = ZoneInfo("UTC")


def _tiny_mkt(n: int = 40) -> dict:
    rng = np.random.default_rng(0)
    close = 2000 + np.cumsum(rng.normal(0, 1, n))
    high = close + 2
    low = close - 2
    vol = np.full(n, 100.0)
    atr = wilder_atr(high, low, close, 14)
    from research.position_mgmt.services.paths import adx_series

    return {
        "high": high,
        "low": low,
        "close": close,
        "vol": vol,
        "atr": atr,
        "adx": adx_series(high, low, close, 14),
        "ts": pd.date_range("2024-01-01", periods=n, freq="h", tz="UTC").to_numpy(),
    }


class SimTests(unittest.TestCase):
    def test_baseline_runs(self) -> None:
        mkt = _tiny_mkt()
        ei = 20
        res = simulate_one(
            side="long",
            entry=float(mkt["close"][ei]),
            atr_entry=float(mkt["atr"][ei]) if np.isfinite(mkt["atr"][ei]) else 5.0,
            conf=70.0,
            d1_regime="Strong Bull",
            daily_stop_hit=False,
            ei=ei,
            mkt=mkt,
            cfg=ManageConfig(name="baseline", family="baseline"),
        )
        self.assertTrue(np.isfinite(res.net_return))

    def test_be_does_not_crash(self) -> None:
        mkt = _tiny_mkt()
        ei = 20
        res = simulate_one(
            side="long",
            entry=float(mkt["close"][ei]),
            atr_entry=5.0,
            conf=50.0,
            d1_regime="Sideways",
            daily_stop_hit=False,
            ei=ei,
            mkt=mkt,
            cfg=ManageConfig(name="be", family="be", be_trigger_r=0.5),
        )
        self.assertTrue(np.isfinite(res.net_return))


class E2ELight(unittest.TestCase):
    def test_evaluate_smoke(self) -> None:
        # Minimal synthetic without full CLI
        from research.position_mgmt.services.evaluate import run_fixed_portfolio

        t0 = datetime(2023, 1, 1, tzinfo=UTC)
        rows = []
        for i in range(30):
            rows.append(
                {
                    "timestamp": t0 + timedelta(hours=i * 6),
                    "valid_year": 2023,
                    "side": "long",
                    "entry_price": 2000.0,
                    "atr_entry": 4.0,
                    "atr_price": 4.0,
                    "net_return": 0.002 if i % 2 == 0 else -0.0015,
                    "meta_proba": 0.5,
                }
            )
        log, curve, m = run_fixed_portfolio(pd.DataFrame(rows))
        self.assertGreater(m["trades"], 0)


if __name__ == "__main__":
    unittest.main()
