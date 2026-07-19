"""Unit tests for ATRE."""

from __future__ import annotations

import unittest

import numpy as np
import pandas as pd

from research.atre.services.path_diag import walk_trade_path
from research.atre.services.policies import AtrePolicy, apply_policy
from research.position_mgmt.services.paths import adx_series, wilder_atr


def _mkt(n: int = 40) -> dict:
    rng = np.random.default_rng(1)
    close = 2000 + np.cumsum(rng.normal(0, 0.5, n))
    high = close + 3
    low = close - 3
    atr = wilder_atr(high, low, close, 14)
    return {
        "high": high,
        "low": low,
        "close": close,
        "vol": np.full(n, 100.0),
        "atr": atr,
        "adx": adx_series(high, low, close, 14),
        "ts": pd.date_range("2024-01-01", periods=n, freq="h", tz="UTC").to_numpy(),
    }


class PathTests(unittest.TestCase):
    def test_walk(self) -> None:
        mkt = _mkt()
        ei = 20
        p = walk_trade_path(
            side="long",
            entry=float(mkt["close"][ei]),
            atr_entry=float(mkt["atr"][ei]) if np.isfinite(mkt["atr"][ei]) else 5.0,
            ei=ei,
            mkt=mkt,
        )
        self.assertTrue(p["ok"])
        self.assertTrue(len(p["bars"]) > 0)

    def test_early_exit(self) -> None:
        mkt = _mkt()
        ei = 20
        p = walk_trade_path(
            side="long",
            entry=float(mkt["close"][ei]),
            atr_entry=5.0,
            ei=ei,
            mkt=mkt,
        )
        r = apply_policy(p, AtrePolicy(name="e", family="early_exit", mae_atr=0.1))
        self.assertIn("net_return", r)


if __name__ == "__main__":
    unittest.main()
