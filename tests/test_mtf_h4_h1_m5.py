"""Tests for hierarchical H4→H1→M5 temporal alignment."""

from __future__ import annotations

import unittest

import pandas as pd

from research.mtf_h4_h1_m5.h4_engine import assert_h4_causal_boundary, attach_h4_signals
from research.mtf_h4_h1_m5.m5_engine import decide_m5_execution, build_m5_features
from simulation.wf.sim import _load_side


class H4CausalTests(unittest.TestCase):
    def test_h4_boundary(self) -> None:
        assert_h4_causal_boundary()

    def test_attach_no_leakage(self) -> None:
        long_df = _load_side("long")
        h4 = pd.read_parquet("artifacts/raw/XAUUSD/H4/data.parquet")
        h4["timestamp"] = pd.to_datetime(h4["timestamp"], utc=True)
        out = attach_h4_signals(long_df.head(500), h4, train_start=2015, train_end=2019)
        self.assertIn("h4_sig_direction", out.columns)
        mask = out["h4_available_timestamp"].notna()
        if mask.any():
            self.assertTrue((out.loc[mask, "timestamp"] >= out.loc[mask, "h4_available_timestamp"]).all())


class M5ExecutionTests(unittest.TestCase):
    def test_m5_starts_after_h1(self) -> None:
        m5 = pd.read_parquet("artifacts/raw/XAUUSD/M5/data.parquet")
        feat = build_m5_features(m5.head(5000))
        h1_ts = pd.Timestamp("2020-06-01T10:00:00Z")
        res = decide_m5_execution(
            side="long",
            h1_signal_ts=h1_ts,
            h1_ref_price=1700.0,
            one_r=10.0,
            m5=feat,
            strategy="immediate",
        )
        if res.execute_idx is not None:
            self.assertGreaterEqual(
                pd.Timestamp(feat.iloc[res.execute_idx]["timestamp"], tz="UTC"),
                h1_ts,
            )


if __name__ == "__main__":
    unittest.main()
