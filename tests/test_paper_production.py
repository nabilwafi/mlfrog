"""Unit tests for production paper infrastructure."""

from __future__ import annotations

import time
import unittest
from datetime import datetime, timezone

from production.events.bus import EventBus, MetricsCollector
from production.events.types import EventType, make_event
from production.paper.broker import PaperBroker
from production.paper.pipeline import IncomingSignal, ProductionPipeline
from production.paper.state import PortfolioState
from production.monitoring.health import HealthReporter
from production.telegram.bot import TelegramNotifier, fmt_health, fmt_new_trade
from production.workers.handlers import register_workers
from production.db.writer import PostgresWriter


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
    def test_fmt(self) -> None:
        text = fmt_new_trade({"trade_id": "abc", "symbol": "XAUUSD", "side": "long", "entry_price": 1})
        self.assertIn("NEW TRADE", text)

    def test_fmt_health(self) -> None:
        text = fmt_health({"status": "ok", "reason": "start", "equity": 10000, "open_positions": 0, "mt5": "connected"})
        self.assertIn("HEALTH", text)
        self.assertIn("MT5", text)


class HealthReporterTests(unittest.TestCase):
    def test_publish_health_event(self) -> None:
        bus = EventBus()
        seen: list[str] = []
        bus.subscribe(EventType.HEALTH, lambda e: seen.append(str(e.payload.get("reason"))))
        bus.start(n_workers=1)
        state = PortfolioState(equity=10_000, peak_equity=10_000)
        hr = HealthReporter(bus, state, interval_seconds=0, probe_throttle_seconds=0)
        hr.publish(reason="start")
        hr.on_http_probe()
        time.sleep(0.2)
        bus.stop()
        self.assertIn("start", seen)
        self.assertIn("http_probe", seen)

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


if __name__ == "__main__":
    unittest.main()
