"""Unit tests for production paper infrastructure."""

from __future__ import annotations

import time
import unittest
from datetime import datetime, timezone

import pandas as pd

from production.events.bus import EventBus, MetricsCollector
from production.events.types import EventType, make_event
from production.paper.broker import PaperBroker
from production.paper.pipeline import IncomingSignal, ProductionPipeline
from production.paper.state import PortfolioState
from production.monitoring.health import HealthReporter
from production.telegram.bot import (
    TelegramNotifier,
    fmt_daily,
    fmt_health,
    fmt_new_trade,
    fmt_skipped,
    fmt_trade_closed,
    split_chat_and_thread,
)
from production.workers.handlers import register_workers
from production.db.writer import PostgresWriter


class SplitChatTests(unittest.TestCase):
    def test_split_colon(self) -> None:
        chat, thread = split_chat_and_thread("-100123:2")
        self.assertEqual(chat, "-100123")
        self.assertEqual(thread, 2)

    def test_split_underscore(self) -> None:
        chat, thread = split_chat_and_thread("-1004343545346_2")
        self.assertEqual(chat, "-1004343545346")
        self.assertEqual(thread, 2)

    def test_plain(self) -> None:
        chat, thread = split_chat_and_thread("-100123")
        self.assertEqual(chat, "-100123")
        self.assertIsNone(thread)


class BusTests(unittest.TestCase):
    def test_publish_async(self) -> None:
        bus = EventBus()
        seen = []
        bus.subscribe(EventType.AUDIT, lambda e: seen.append(e.payload.get("action")))
        bus.start(n_workers=1)
        bus.publish(make_event(EventType.AUDIT, {"component": "t", "action": "ping", "detail": {}}))
        time.sleep(0.2)
        bus.stop()
        self.assertIn("ping", seen)


class PipelineTests(unittest.TestCase):
    def test_low_primary_still_opens_with_fixed_lot(self) -> None:
        """Fixed 0.01 lot: low primary still opens (not a skip gate)."""
        bus = EventBus()
        metrics = MetricsCollector()
        register_workers(bus, db=PostgresWriter(None), telegram=TelegramNotifier(None, None), metrics=metrics)
        bus.start(n_workers=1)
        state = PortfolioState(equity=10_000, peak_equity=10_000)
        pipe = ProductionPipeline(bus=bus, state=state, broker=PaperBroker())
        sig = IncomingSignal(
            timestamp=datetime(2024, 1, 2, 10, tzinfo=timezone.utc),
            symbol="XAUUSD",
            side="long",
            probability=0.2,
            meta_probability=0.9,
            confidence=70,
            entry_price=2000,
            atr=4.0,
            bar_key="t1",
        )
        out = pipe.process_signal(sig)
        self.assertEqual(out["status"], "opened")
        pos = state.open_positions[out["trade_id"]]
        from production import FIXED_LOT

        self.assertAlmostEqual(pos.lot, FIXED_LOT, places=6)
        time.sleep(0.15)
        bus.stop()

    def test_confidence_ignored(self) -> None:
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
                confidence=5.0,
                entry_price=2000,
                atr=4.0,
                bar_key="conf_off",
            )
        )
        self.assertEqual(out["status"], "opened")
        bus.stop()

    def test_open_and_close(self) -> None:
        bus = EventBus()
        register_workers(bus, db=PostgresWriter(None), telegram=TelegramNotifier(None, None), metrics=MetricsCollector())
        bus.start(n_workers=1)
        state = PortfolioState(equity=10_000, peak_equity=10_000)
        pipe = ProductionPipeline(bus=bus, state=state, broker=PaperBroker())
        sig = IncomingSignal(
            timestamp=datetime(2024, 1, 2, 10, tzinfo=timezone.utc),
            symbol="XAUUSD",
            side="long",
            probability=0.6,
            meta_probability=0.55,
            confidence=55,
            entry_price=2000,
            atr=4.0,
            bar_key="t2",
        )
        out = pipe.process_signal(sig)
        self.assertEqual(out["status"], "opened")
        tid = out["trade_id"]
        self.assertIn(tid, state.open_positions)
        pos = state.open_positions[tid]
        from production import FIXED_LOT

        self.assertAlmostEqual(pos.lot, FIXED_LOT, places=6)
        # hit TP
        closed = pipe.on_bar(high=2010, low=1999, close=2005, timestamp=datetime(2024, 1, 2, 12, tzinfo=timezone.utc))
        self.assertTrue(len(closed) >= 1)
        self.assertNotIn(tid, state.open_positions)
        time.sleep(0.15)
        bus.stop()

    def test_duplicate_idempotent(self) -> None:
        bus = EventBus()
        bus.start(n_workers=1)
        state = PortfolioState(equity=10_000, peak_equity=10_000)
        pipe = ProductionPipeline(bus=bus, state=state, broker=PaperBroker())
        kwargs = dict(
            timestamp=datetime(2024, 1, 2, 10, tzinfo=timezone.utc),
            symbol="XAUUSD",
            side="short",
            probability=0.6,
            meta_probability=0.55,
            confidence=55,
            entry_price=2000,
            atr=4.0,
            bar_key="same",
        )
        a = pipe.process_signal(IncomingSignal(**kwargs))
        b = pipe.process_signal(IncomingSignal(**kwargs))
        self.assertEqual(a["status"], "opened")
        self.assertEqual(b["reason"], "duplicate")
        bus.stop()

    def test_blocks_second_side_while_open(self) -> None:
        bus = EventBus()
        bus.start(n_workers=1)
        state = PortfolioState(equity=10_000, peak_equity=10_000)
        pipe = ProductionPipeline(bus=bus, state=state, broker=PaperBroker())
        long = IncomingSignal(
            timestamp=datetime(2024, 1, 2, 10, tzinfo=timezone.utc),
            symbol="XAUUSD",
            side="long",
            probability=0.6,
            meta_probability=0.55,
            confidence=55,
            entry_price=2000,
            atr=4.0,
            bar_key="bar:long",
        )
        short = IncomingSignal(
            timestamp=datetime(2024, 1, 2, 10, tzinfo=timezone.utc),
            symbol="XAUUSD",
            side="short",
            probability=0.6,
            meta_probability=0.60,
            confidence=55,
            entry_price=2000,
            atr=4.0,
            bar_key="bar:short",
        )
        a = pipe.process_signal(long)
        b = pipe.process_signal(short)
        self.assertEqual(a["status"], "opened")
        # max_open=1 trips before opposite check
        self.assertEqual(b["reason"], "max_open")
        self.assertEqual(len(state.open_positions), 1)
        bus.stop()

    def test_multi_entry_same_side(self) -> None:
        """Policy lock: max_open=1 — second same-side entry is blocked."""
        bus = EventBus()
        bus.start(n_workers=1)
        state = PortfolioState(equity=10_000, peak_equity=10_000)
        pipe = ProductionPipeline(bus=bus, state=state, broker=PaperBroker())
        first = pipe.process_signal(
            IncomingSignal(
                timestamp=datetime(2024, 1, 2, 10, tzinfo=timezone.utc),
                symbol="XAUUSD",
                side="long",
                probability=0.6,
                meta_probability=0.55,
                confidence=55,
                entry_price=2000,
                atr=4.0,
                bar_key="stack:0",
            )
        )
        second = pipe.process_signal(
            IncomingSignal(
                timestamp=datetime(2024, 1, 2, 11, tzinfo=timezone.utc),
                symbol="XAUUSD",
                side="long",
                probability=0.6,
                meta_probability=0.55,
                confidence=55,
                entry_price=2000,
                atr=4.0,
                bar_key="stack:1",
            )
        )
        self.assertEqual(first["status"], "opened")
        self.assertEqual(second["reason"], "max_open")
        self.assertEqual(len(state.open_positions), 1)
        bus.stop()

    def test_atr_trail_ratchets_and_exits(self) -> None:
        """After +0.25R, trail SL tightens; pullback hits TRAIL."""
        bus = EventBus()
        bus.start(n_workers=1)
        state = PortfolioState(equity=10_000, peak_equity=10_000)
        pipe = ProductionPipeline(bus=bus, state=state, broker=PaperBroker())
        # atr=4, SL_ATR=1.5 → 1R=6; 0.25R=1.5 → need high >= 2001.5 to arm trail
        out = pipe.process_signal(
            IncomingSignal(
                timestamp=datetime(2024, 1, 2, 10, tzinfo=timezone.utc),
                symbol="XAUUSD",
                side="long",
                probability=0.6,
                meta_probability=0.55,
                confidence=55,
                entry_price=2000,
                atr=4.0,
                bar_key="trail:1",
            )
        )
        self.assertEqual(out["status"], "opened")
        tid = out["trade_id"]
        # arm trail; keep low above trail SL (high - 0.08*atr = 2004 - 0.32 = 2003.68)
        pipe.on_bar(high=2004, low=2003.75, close=2003.8, timestamp=datetime(2024, 1, 2, 11, tzinfo=timezone.utc))
        self.assertIn(tid, state.open_positions)
        self.assertGreater(state.open_positions[tid].stop_loss, 2000 - 1.5 * 4)
        # pullback through trail SL
        closed = pipe.on_bar(
            high=2003.7, low=2003.0, close=2003.2, timestamp=datetime(2024, 1, 2, 12, tzinfo=timezone.utc)
        )
        self.assertTrue(len(closed) >= 1)
        self.assertEqual(closed[0]["exit_reason"], "TRAIL")
        self.assertNotIn(tid, state.open_positions)
        bus.stop()

    def test_atr_risk_scales_fixed_lot(self) -> None:
        """map_conservative: atr_pct 0.85 → 25% of FIXED_LOT."""
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
                confidence=55,
                entry_price=2000,
                atr=4.0,
                atr_percentile=0.85,
                bar_key="atr_risk:1",
            )
        )
        self.assertEqual(out["status"], "opened")
        from production import FIXED_LOT

        self.assertAlmostEqual(state.open_positions[out["trade_id"]].lot, FIXED_LOT * 0.25, places=6)
        bus.stop()

    def test_take_profit_disabled(self) -> None:
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
                confidence=55,
                entry_price=2000,
                atr=4.0,
                bar_key="no_tp:1",
            )
        )
        self.assertEqual(out["status"], "opened")
        self.assertEqual(state.open_positions[out["trade_id"]].take_profit, 0.0)
        # move up but stay below trail activation (0.25R = 1.5) so no TRAIL/TP exit
        closed = pipe.on_bar(
            high=2001.4, low=2000.1, close=2001.0, timestamp=datetime(2024, 1, 2, 11, tzinfo=timezone.utc)
        )
        self.assertEqual(closed, [])
        self.assertIn(out["trade_id"], state.open_positions)
        bus.stop()


class TelegramFmtTests(unittest.TestCase):
    def test_fmt_new_trade(self) -> None:
        text = fmt_new_trade(
            {
                "trade_id": "00000000000000000000000000000400",
                "symbol": "XAUUSD",
                "side": "long",
                "entry_price": 3345.5,
                "stop_loss": 3338.0,
                "take_profit": 3360.0,
                "lot": 0.10,
                "risk_pct": 0.01,
                "probability": 0.82,
                "trail_atr_mult": 0.12,
                "trail_activate_r": 0.5,
                "trend": "bull",
                "volatility": "high",
                "momentum": "strong",
                "structure": "trend",
                "session": "london",
                "entry_time": datetime(2026, 7, 19, 15, 30, tzinfo=timezone.utc),
            }
        )
        self.assertIn("OPEN POSITION", text)
        self.assertIn("TRADE OPEN", text)
        self.assertIn("BUY", text)
        self.assertIn("Entry : 3345.50", text)
        self.assertIn("SL    : 3338.00", text)
        self.assertIn("ATR Trail 0.12", text)
        self.assertIn("Primary: 82.0%", text)
        self.assertIn("TRENDING BULLISH", text)
        self.assertIn("London", text)
        self.assertIn("15:30 UTC", text)
        self.assertNotIn("Strategy", text)
        self.assertNotIn("Confidence", text)

    def test_fmt_trail_update(self) -> None:
        from production.telegram.bot import fmt_trail_update

        text = fmt_trail_update(
            {
                "symbol": "XAUUSD",
                "side": "long",
                "entry_price": 3382.50,
                "mark_price": 3388.20,
                "stop_loss": 3385.00,
                "unrealized_pnl": 57.0,
                "trend": "bull",
                "structure": "trend",
                "volatility": "high",
                "momentum": "strong",
                "session": "london",
                "timestamp": datetime(2026, 7, 19, 16, 0, tzinfo=timezone.utc),
            }
        )
        self.assertIn("UPDATE + TRAILING STOP", text)
        self.assertIn("New SL: 3385.00", text)
        self.assertIn("PROFIT LOCKED", text)
        self.assertNotIn("Strategy", text)
        self.assertNotIn("Reversal", text)

    def test_fmt_trade_closed(self) -> None:
        text = fmt_trade_closed(
            {
                "trade_id": "00000000000000000000000000000400",
                "symbol": "XAUUSD",
                "side": "long",
                "entry_price": 3382.50,
                "exit_price": 3376.50,
                "exit_reason": "SL",
                "pnl": -60,
                "pnl_r": -1.0,
                "duration_seconds": 2 * 3600 + 14 * 60,
                "trend": "bull",
                "structure": "trend",
                "session": "london",
                "exit_time": datetime(2026, 7, 19, 17, 44, tzinfo=timezone.utc),
            }
        )
        self.assertIn("STOP LOSS", text)
        self.assertIn("Entry : 3382.50", text)
        self.assertIn("Exit  : 3376.50", text)
        self.assertIn("17:44 UTC", text)
        self.assertNotIn("Reversal", text)
        self.assertNotIn("Strategy", text)

        trail = fmt_trade_closed(
            {
                "side": "long",
                "entry_price": 2000.0,
                "exit_price": 2004.0,
                "exit_reason": "TRAIL",
                "pnl": 40,
                "pnl_r": 0.8,
                "duration_seconds": 3600,
            }
        )
        self.assertIn("TRAIL STOP", trail)
        self.assertNotIn("Reversal", trail)

    def test_fmt_skipped(self) -> None:
        text = fmt_skipped(
            {
                "entry_price": 3348.25,
                "reason": "confidence",
                "current_value": 68,
                "threshold": 70,
                "timestamp": datetime(2026, 7, 19, 15, 20, tzinfo=timezone.utc),
            }
        )
        self.assertIn("TRADE SKIPPED", text)
        self.assertIn("Price    : 3348.25", text)
        self.assertIn("Reason   : Confidence", text)
        self.assertIn("Value    : 68%", text)
        self.assertIn("Required : ≥70%", text)
        self.assertIn("Time     : 15:20 UTC", text)

    def test_fmt_health(self) -> None:
        text = fmt_health(
            {
                "status": "ok",
                "environment": "paper",
                "mt5": "connected",
                "uptime": "5h 32m",
                "last_ping": "2026-07-19 15:45 UTC",
            }
        )
        self.assertIn("HEALTHCHECK", text)
        self.assertIn("Status: 🟢 ONLINE", text)
        self.assertIn("Environment:\nPAPER", text)
        self.assertIn("✅ Running", text)
        self.assertIn("✅ Connected", text)
        self.assertIn("5h 32m", text)
        self.assertIn("2026-07-19 15:45 UTC", text)

    def test_fmt_health_mt5_down(self) -> None:
        text = fmt_health({"status": "degraded", "mt5": "down", "uptime": "1h 0m", "last_ping": "n/a"})
        self.assertIn("🟡 DEGRADED", text)
        self.assertIn("❌ Disconnected", text)

    def test_fmt_daily(self) -> None:
        text = fmt_daily(
            {
                "date": "2026-07-19",
                "symbol": "XAUUSD",
                "status": "ok",
                "balance": 80.0,
                "equity": 82.35,
                "trades": 5,
                "wins": 3,
                "losses": 2,
                "winrate": 0.6,
                "pnl": 2.35,
                "pnl_pct": 2.94,
                "best_trade_pnl": 1.5,
                "worst_trade_pnl": -0.8,
            }
        )
        self.assertIn("XAUUSD DAILY", text)
        self.assertIn("Balance: $80.00", text)
        self.assertIn("Equity: $82.35", text)
        self.assertIn("PnL: +$2.35", text)
        self.assertIn("Trades: 5", text)
        self.assertIn("W: 3", text)
        self.assertIn("L: 2", text)
        self.assertIn("WR: 60%", text)
        self.assertIn("Best: +$1.50", text)
        self.assertIn("Worst: -$0.80", text)
        self.assertIn("Online", text)


class HealthReporterTests(unittest.TestCase):
    def test_publish_health_event(self) -> None:
        bus = EventBus()
        seen: list[str] = []
        bus.subscribe(EventType.HEALTH, lambda e: seen.append(str(e.payload.get("reason"))))
        bus.start(n_workers=1)
        state = PortfolioState(equity=10_000, peak_equity=10_000)
        hr = HealthReporter(bus, state, interval_seconds=0, probe_throttle_seconds=0)
        hr.publish(reason="start")
        hr.on_http_probe()  # snapshot only — no Telegram event
        time.sleep(0.2)
        bus.stop()
        self.assertIn("start", seen)
        self.assertNotIn("http_probe", seen)
        snap = hr.last_snapshot()
        self.assertIn("uptime", snap)
        self.assertIn("last_ping", snap)

    def test_mt5_down_degrades(self) -> None:
        bus = EventBus()
        payloads: list[dict] = []
        bus.subscribe(EventType.HEALTH, lambda e: payloads.append(dict(e.payload)))
        bus.start(n_workers=1)
        state = PortfolioState(equity=10_000, peak_equity=10_000)
        hr = HealthReporter(
            bus,
            state,
            interval_seconds=0,
            probe_throttle_seconds=0,
            mt5_probe=lambda: {"mt5": "down", "mt5_error": "offline"},
        )
        hr.publish(reason="heartbeat")
        time.sleep(0.2)
        bus.stop()
        self.assertTrue(payloads)
        self.assertEqual(payloads[0]["mt5"], "down")
        self.assertEqual(payloads[0]["status"], "degraded")

    def test_mt5_transition_reason(self) -> None:
        bus = EventBus()
        reasons: list[str] = []
        bus.subscribe(EventType.HEALTH, lambda e: reasons.append(str(e.payload.get("reason"))))
        bus.start(n_workers=1)
        state = PortfolioState(equity=10_000, peak_equity=10_000)
        flag = {"ok": True}

        def probe() -> dict:
            return {"mt5": "connected"} if flag["ok"] else {"mt5": "down", "mt5_error": "x"}

        hr = HealthReporter(bus, state, interval_seconds=0, probe_throttle_seconds=0, mt5_probe=probe)
        hr.publish(reason="heartbeat")
        flag["ok"] = False
        hr.publish(reason="heartbeat")
        time.sleep(0.2)
        bus.stop()
        self.assertTrue(any("mt5_down" in r for r in reasons))


class LiveSourceTests(unittest.TestCase):
    def test_poll_emits_once_per_bar(self) -> None:
        from production.live.signal_source import LiveMT5SignalSource

        ts = pd.Timestamp("2024-06-01 10:00:00", tz="UTC")
        closed = pd.DataFrame(
            [
                {
                    "timestamp": ts,
                    "open": 2300.0,
                    "high": 2305.0,
                    "low": 2298.0,
                    "close": 2302.0,
                    "tick_volume": 100,
                    "spread": 2,
                    "real_volume": 0,
                }
            ]
        )

        class FakeFeed:
            def connect(self) -> None:
                pass

            def disconnect(self) -> None:
                pass

            def latest_closed_bar(self, timeframe: str):
                return closed

            def fetch(self, timeframe: str, *, count: int):
                return closed

        class FakeInference:
            def score_row(self, row, *, side: str, entry_price: float, atr: float):
                from production.live.inference import ScoredSignal

                return ScoredSignal(
                    side=side,
                    timestamp=ts,
                    entry_price=entry_price,
                    atr=atr,
                    probability=0.5,
                    meta_probability=0.55,
                    confidence=55.0,
                    session="london",
                    regime="Sideways",
                    bar_key=ts.isoformat() + ":" + side,
                )

        src = LiveMT5SignalSource({"timezone": "UTC"}, symbol="XAUUSD")
        src._feed = FakeFeed()  # type: ignore[assignment]
        src._features.build_panel = lambda **_: pd.DataFrame(  # type: ignore[method-assign]
            # atr_percent is % of price (VolatilityBuilder): 0.2 => ATR = 0.2% of close
            [{"timestamp": ts, "atr_percent": 0.2, "session_london": 1.0, "d1_regime": "Sideways"}]
        )
        src._inference = FakeInference()  # type: ignore[assignment]

        t1 = src.poll()
        t2 = src.poll()
        self.assertIsNotNone(t1.bar)
        self.assertEqual(t1.bar.symbol, "XAUUSD")
        self.assertEqual(len(t1.signals), 1)  # one side per bar (best meta)
        self.assertAlmostEqual(t1.signals[0].atr, 2302.0 * 0.2 / 100.0, places=6)
        self.assertEqual(len(t2.signals), 0)


class CandleEventTests(unittest.TestCase):
    def test_candle_closed_event(self) -> None:
        bus = EventBus()
        payloads: list[dict] = []
        bus.subscribe(EventType.CANDLE_CLOSED, lambda e: payloads.append(dict(e.payload)))
        bus.start(n_workers=1)
        bus.publish(
            make_event(
                EventType.CANDLE_CLOSED,
                {
                    "symbol": "XAUUSD",
                    "timeframe": "H1",
                    "timestamp": datetime(2024, 6, 1, 10, tzinfo=timezone.utc),
                    "open": 2300.0,
                    "high": 2305.0,
                    "low": 2298.0,
                    "close": 2302.0,
                    "tick_volume": 100.0,
                    "spread": 2.0,
                    "real_volume": 0.0,
                    "source": "mt5_live",
                    "features": {"atr_percent": 0.002},
                },
            )
        )
        time.sleep(0.2)
        bus.stop()
        self.assertEqual(payloads[0]["symbol"], "XAUUSD")
        self.assertEqual(payloads[0]["features"]["atr_percent"], 0.002)


if __name__ == "__main__":
    unittest.main()
