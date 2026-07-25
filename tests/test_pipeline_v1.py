"""Unit + regression tests for Sprint 27 L1–L6 pipeline."""

from __future__ import annotations

import unittest
from datetime import datetime, timezone

from pipeline import EXPECTED_R_REF, META_THRESHOLD, RISK_BASE, RISK_MAX, RISK_MIN, STRUCTURAL_RR
from pipeline.backtest.adapter import golden_metrics_from_synthetic, run_meta_expected_r_backtest
from pipeline.l2_market_state.infer import infer_market_state
from pipeline.l4_meta_edge.edge import decide_meta, expected_r_from_edge
from pipeline.l5_risk.engine import build_risk, lots_from_risk, risk_pct_from_expected_r
from pipeline.l6_portfolio.execution import PortfolioExecution
from pipeline.orchestrator import TradingPipeline
from production.events.bus import EventBus
from production.paper.broker import PaperBroker
from production.paper.state import PortfolioState
import pandas as pd


# Locked from first green run of golden_metrics_from_synthetic()
GOLDEN = {
    "n_trades": 3,
    "final_equity": 90.431494,
    "total_return": 0.130394,
    "max_drawdown": 0.009778,
}


class L2Tests(unittest.TestCase):
    def test_regime_mapping(self) -> None:
        ms = infer_market_state(
            {"d1_regime": "Strong Bull", "session_london": 1.0, "volatility_rank": 0.8, "d1_ema_slope": 1e-3}
        )
        self.assertEqual(ms.trend, "bull")
        self.assertEqual(ms.session, "london")
        self.assertEqual(ms.volatility, "high")
        self.assertEqual(ms.momentum, "strong")


class L4Tests(unittest.TestCase):
    def test_meta_below_threshold(self) -> None:
        out = decide_meta(0.2)
        self.assertFalse(out.trade)
        self.assertIn("below", out.reason)

    def test_meta_ok_and_expected_r(self) -> None:
        out = decide_meta(0.55)
        self.assertTrue(out.trade)
        er = expected_r_from_edge(0.55)
        self.assertAlmostEqual(out.expected_r, er)
        self.assertAlmostEqual(er, 0.55 * (STRUCTURAL_RR + 1) - 1)

    def test_threshold_boundary(self) -> None:
        self.assertFalse(decide_meta(META_THRESHOLD - 1e-9).trade)
        self.assertTrue(decide_meta(META_THRESHOLD).trade)


class L5Tests(unittest.TestCase):
    def test_risk_pct_clamp_low(self) -> None:
        # p=0.45 → small E[R] → RISK_MIN
        er = expected_r_from_edge(0.45)
        self.assertAlmostEqual(risk_pct_from_expected_r(er), RISK_MIN)

    def test_risk_pct_clamp_high(self) -> None:
        er = expected_r_from_edge(0.90)
        self.assertAlmostEqual(risk_pct_from_expected_r(er), RISK_MAX)

    def test_risk_pct_mid(self) -> None:
        er = expected_r_from_edge(0.55)
        expected = RISK_BASE * er / EXPECTED_R_REF
        self.assertAlmostEqual(risk_pct_from_expected_r(er), expected)

    def test_build_risk_lots_and_barriers(self) -> None:
        er = expected_r_from_edge(0.55)
        risk = build_risk(side="long", entry=2000.0, atr=4.0, equity=10_000.0, expected_r=er)
        assert risk is not None
        self.assertAlmostEqual(risk.stop, 1994.0)
        self.assertAlmostEqual(risk.target, 2008.0)
        self.assertAlmostEqual(risk.lot, lots_from_risk(equity=10_000, atr=4.0, risk_pct=risk.risk_pct))

    def test_nonpositive_expected_r_clamps_min(self) -> None:
        self.assertEqual(risk_pct_from_expected_r(-0.1), RISK_MIN)


class L6Tests(unittest.TestCase):
    def test_heat_blocks(self) -> None:
        bus = EventBus()
        state = PortfolioState(equity=10_000, peak_equity=10_000)
        state.roll_day(datetime(2024, 1, 2, 10, tzinfo=timezone.utc))
        state.day_pnl = -100.0  # -1R at 1% of 10k
        pipe = TradingPipeline(portfolio=PortfolioExecution(bus=bus, state=state, broker=PaperBroker()))
        out = pipe.run_scored_signal(
            timestamp=datetime(2024, 1, 2, 11, tzinfo=timezone.utc),
            symbol="XAUUSD",
            side="long",
            probability=0.6,
            meta_probability=0.55,
            entry_price=2000,
            atr=4.0,
            bar_key="heat1",
        )
        self.assertEqual(out["reason"], "heat")

    def test_duplicate(self) -> None:
        bus = EventBus()
        state = PortfolioState(equity=10_000, peak_equity=10_000)
        pipe = TradingPipeline(portfolio=PortfolioExecution(bus=bus, state=state, broker=PaperBroker()))
        kwargs = dict(
            timestamp=datetime(2024, 1, 2, 10, tzinfo=timezone.utc),
            symbol="XAUUSD",
            side="long",
            probability=0.6,
            meta_probability=0.55,
            entry_price=2000,
            atr=4.0,
            bar_key="dup",
        )
        a = pipe.run_scored_signal(**kwargs)
        b = pipe.run_scored_signal(**kwargs)
        self.assertEqual(a["status"], "opened")
        self.assertEqual(b["reason"], "duplicate")


class RegressionTests(unittest.TestCase):
    def test_golden_synthetic(self) -> None:
        got = golden_metrics_from_synthetic()
        self.assertEqual(got["n_trades"], GOLDEN["n_trades"])
        self.assertAlmostEqual(got["final_equity"], GOLDEN["final_equity"], places=5)
        self.assertAlmostEqual(got["total_return"], GOLDEN["total_return"], places=5)
        self.assertAlmostEqual(got["max_drawdown"], GOLDEN["max_drawdown"], places=5)

    def test_confidence_ignored_in_backtest(self) -> None:
        # Low confidence column must not change results
        base = pd.Timestamp("2024-01-02 10:00:00", tz="UTC")
        rows = [
            {
                "timestamp": base,
                "side": "long",
                "entry_price": 2000.0,
                "atr_price": 4.0,
                "meta_proba": 0.55,
                "confidence": 10.0,
                "net_return": 0.01,
                "holding_bars": 4,
            }
        ]
        panel = pd.DataFrame(rows)
        traded, _, m = run_meta_expected_r_backtest(panel, starting_equity=80.0)
        self.assertEqual(int(m["n_trades"]), 1)
        self.assertEqual(len(traded), 1)


class ConfidenceDisabledPaperTests(unittest.TestCase):
    def test_low_confidence_still_opens(self) -> None:
        from production.paper.pipeline import IncomingSignal, ProductionPipeline

        bus = EventBus()
        bus.start(n_workers=1)
        state = PortfolioState(equity=10_000, peak_equity=10_000)
        pipe = ProductionPipeline(bus=bus, state=state, broker=PaperBroker())
        out = pipe.process_signal(
            IncomingSignal(
                timestamp=datetime(2024, 1, 2, 10, tzinfo=timezone.utc),
                symbol="XAUUSD",
                side="long",
                probability=0.6,
                meta_probability=0.55,
                confidence=5.0,  # would have skipped under skip40
                entry_price=2000,
                atr=4.0,
                bar_key="conf_off",
            )
        )
        self.assertEqual(out["status"], "opened")
        pos = state.open_positions[out["trade_id"]]
        # option C: not flat 1%
        self.assertNotAlmostEqual(pos.risk_pct, 0.01, places=5)
        bus.stop()


if __name__ == "__main__":
    unittest.main()
