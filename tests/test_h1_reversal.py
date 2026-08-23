"""Unit tests for H1 reversal decision engine + labels."""

from __future__ import annotations

import unittest

import numpy as np
import pandas as pd

from research.h1_reversal.decision import (
    DecisionAction,
    DecisionConfig,
    MarketState,
    PositionState,
    evaluate,
)
from research.h1_reversal.labels import BarrierReversalParams, label_opposite_barrier


class DecisionEngineTests(unittest.TestCase):
    def test_flat_enters_long_when_gated(self) -> None:
        r = evaluate(
            market=MarketState(price=2000),
            p_long=0.7,
            p_short=0.4,
            reversal_prob=None,
            reversal_stage="normal",
            h4_context="unknown",
            m5_state="unknown",
            position=PositionState(side="flat"),
            entry_gate_long=True,
            entry_gate_short=False,
        )
        self.assertEqual(r.action, DecisionAction.ENTER_LONG)

    def test_long_holds_on_same_direction_signal(self) -> None:
        r = evaluate(
            market=MarketState(price=2000),
            p_long=0.8,
            p_short=0.2,
            reversal_prob=None,
            reversal_stage="normal",
            h4_context="supportive",
            m5_state="recovering",
            position=PositionState(side="long", unrealized_r=0.5, mfe_r=0.6),
        )
        self.assertEqual(r.action, DecisionAction.HOLD_LONG)
        self.assertEqual(r.reason, "thesis_valid")

    def test_long_weakening_waits(self) -> None:
        cfg = DecisionConfig(confirm_bars=2, weaken_opposite_prob=0.55, weaken_own_prob_max=0.45)
        r = evaluate(
            market=MarketState(price=2000),
            p_long=0.30,
            p_short=0.70,
            reversal_prob=None,
            reversal_stage="normal",
            h4_context="neutral",
            m5_state="deteriorating",
            position=PositionState(side="long", weakening_bars=0),
            cfg=cfg,
        )
        self.assertEqual(r.action, DecisionAction.WAIT_FOR_CONFIRMATION)
        self.assertEqual(r.reversal_stage, "weakening")

    def test_long_confirmed_exits(self) -> None:
        cfg = DecisionConfig(confirm_bars=2, confirm_opposite_prob=0.60, weaken_own_prob_max=0.45)
        r = evaluate(
            market=MarketState(price=2000),
            p_long=0.25,
            p_short=0.75,
            reversal_prob=0.82,
            reversal_stage="weakening",
            h4_context="conflicting",
            m5_state="deteriorating",
            position=PositionState(side="long", weakening_bars=2),
            cfg=cfg,
        )
        self.assertEqual(r.action, DecisionAction.EXIT_LONG)
        self.assertEqual(r.reason, "bearish_reversal_confirmed")

    def test_no_direct_reverse_from_flat_cooldown(self) -> None:
        cfg = DecisionConfig(reentry_cooldown_bars=4)
        r = evaluate(
            market=MarketState(price=2000),
            p_long=0.2,
            p_short=0.8,
            reversal_prob=None,
            reversal_stage="normal",
            h4_context="unknown",
            m5_state="unknown",
            position=PositionState(side="flat", last_exit_side="short", bars_since_exit=1),
            entry_gate_short=True,
            cfg=cfg,
        )
        self.assertEqual(r.action, DecisionAction.WAIT)

    def test_hard_exit_r(self) -> None:
        cfg = DecisionConfig(hard_exit_r=-1.0)
        r = evaluate(
            market=MarketState(price=2000),
            p_long=0.6,
            p_short=0.3,
            reversal_prob=None,
            reversal_stage="normal",
            h4_context="supportive",
            m5_state="recovering",
            position=PositionState(side="long", unrealized_r=-1.05),
            cfg=cfg,
        )
        self.assertEqual(r.action, DecisionAction.EXIT_LONG)
        self.assertEqual(r.reason, "hard_exit_r")


class LabelTests(unittest.TestCase):
    def test_opposite_barrier_long(self) -> None:
        # flat then dump 1.2 ATR without upside
        n = 40
        close = np.full(n, 100.0)
        high = close + 0.1
        low = close - 0.1
        # after bar 20, crash
        for i in range(21, 28):
            low[i] = 100.0 - 2.0
            close[i] = 100.0 - 1.5
            high[i] = 100.0 - 1.0
        df = pd.DataFrame({"open": close, "high": high, "low": low, "close": close})
        # ATR will be small early then jump — use many bars of range so ATR ~ known
        for i in range(n):
            high[i] = close[i] + 1.0
            low[i] = close[i] - 1.0
        for i in range(21, 28):
            low[i] = 100.0 - 3.0
            high[i] = 99.0
            close[i] = 98.0
        df = pd.DataFrame({"open": close, "high": high, "low": low, "close": close})
        y = label_opposite_barrier(
            df,
            side="long",
            decision_idx=np.array([20]),
            params=BarrierReversalParams(horizon_bars=8, adverse_atr=1.0, favor_atr=1.0),
        )
        self.assertEqual(float(y[0]), 1.0)


if __name__ == "__main__":
    unittest.main()
