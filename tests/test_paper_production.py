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
    def test_meta_skip(self) -> None:
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
            probability=0.6,
            meta_probability=0.2,
            confidence=70,
            entry_price=2000,
            atr=4.0,
            bar_key="t1",
        )
        out = pipe.process_signal(sig)
        self.assertEqual(out["status"], "skipped")
        self.assertEqual(out["reason"], "meta")
        time.sleep(0.15)
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


class TelegramFmtTests(unittest.TestCase):
    def test_fmt_new_trade(self) -> None:
        text = fmt_new_trade(
            {
                "environment": "live",
                "side": "long",
                "entry_price": 3345.5,
                "stop_loss": 3335.0,
                "take_profit": 3365.0,
                "risk_pct": 0.01,
                "confidence": 78,
                "entry_time": datetime(2026, 7, 19, 15, 30, tzinfo=timezone.utc),
            }
        )
        self.assertIn("NEW TRADE", text)
        self.assertIn("Environment:\nLIVE", text)
        self.assertIn("Side:\nBUY", text)
        self.assertIn("Entry:\n3345.50", text)
        self.assertIn("SL:\n3335.00", text)
        self.assertIn("TP:\n3365.00", text)
        self.assertIn("Risk:\n1%", text)
        self.assertIn("Confidence:\n78%", text)
        self.assertIn("Time:\n15:30 UTC", text)

    def test_fmt_trade_closed(self) -> None:
        text = fmt_trade_closed(
            {
                "environment": "live",
                "exit_reason": "TP",
                "pnl": 120,
                "pnl_r": 2.5,
                "duration_seconds": 3 * 3600 + 20 * 60,
            }
        )
        self.assertIn("TRADE CLOSED", text)
        self.assertIn("Result:\nTP HIT", text)
        self.assertIn("PnL:\n+120$", text)
        self.assertIn("R:\n+2.5R", text)
        self.assertIn("Duration:\n3h 20m", text)

    def test_fmt_skipped(self) -> None:
        text = fmt_skipped(
            {
                "environment": "live",
                "reason": "confidence",
                "current_value": 62,
                "threshold": 70,
            }
        )
        self.assertIn("TRADE SKIPPED", text)
        self.assertIn("Reason:\nConfidence", text)
        self.assertIn("Value:\n62%", text)
        self.assertIn("Threshold:\n70%", text)

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
                "status": "ok",
                "trades": 5,
                "wins": 3,
                "losses": 2,
                "skipped": 7,
                "pnl_pct": 2.35,
                "uptime_seconds": 24 * 3600,
            }
        )
        self.assertIn("DAILY REPORT", text)
        self.assertIn("19 Jul 2026", text)
        self.assertIn("🟢 OK", text)
        self.assertIn("Trades:\n5", text)
        self.assertIn("Win/Loss:\n3/2", text)
        self.assertIn("Skip:\n7", text)
        self.assertIn("+2.35%", text)
        self.assertIn("Uptime:\n24h", text)


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
            [{"timestamp": ts, "atr_percent": 0.002, "session_london": 1.0, "d1_regime": "Sideways"}]
        )
        src._inference = FakeInference()  # type: ignore[assignment]

        t1 = src.poll()
        t2 = src.poll()
        self.assertIsNotNone(t1.bar)
        self.assertEqual(t1.bar.symbol, "XAUUSD")
        self.assertEqual(len(t1.signals), 2)
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
