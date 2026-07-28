"""Open trades vs history_trades — ticket_id PK + close moves row."""

from __future__ import annotations

import time
import unittest
from datetime import datetime, timezone
from typing import Any

from production.db.writer import PostgresWriter
from production.events.bus import EventBus, MetricsCollector
from production.events.types import EventType, make_event
from production.paper.broker import PaperBroker
from production.telegram.bot import TelegramNotifier
from production.workers.handlers import register_workers


class FakeWriter(PostgresWriter):
    """In-memory stand-in: trades = open, history = closed (keyed by ticket_id)."""

    def __init__(self) -> None:
        super().__init__(None, schema="testing")
        self._enabled = True
        self.open_rows: dict[int, dict[str, Any]] = {}
        self.history: dict[int, dict[str, Any]] = {}

    def insert_open_trade(self, row: dict[str, Any]) -> None:
        p = self._normalize(row)
        tid = int(p["ticket_id"])
        p["ticket_id"] = tid
        self.open_rows[tid] = dict(p)

    def close_trade_to_history(self, row: dict[str, Any]) -> None:
        p = self._normalize(row)
        ticket = int(p["ticket_id"])
        open_row = self.fetch_open_trade(ticket_id=ticket)
        if open_row:
            hist = dict(open_row)
            for key in self._EXIT_FIELDS:
                if p.get(key) is not None:
                    hist[key] = p[key]
            if ticket in self.history:
                existing = self.history[ticket]
                for key in self._EXIT_FIELDS:
                    if existing.get(key) is None and hist.get(key) is not None:
                        existing[key] = hist[key]
            else:
                self.history[ticket] = hist
            self.open_rows.pop(ticket, None)
            return
        if ticket in self.history:
            existing = self.history[ticket]
            for key in self._EXIT_FIELDS:
                if existing.get(key) is None and p.get(key) is not None:
                    existing[key] = p[key]

    def fetch_open_trade(self, *, ticket_id: int | None = None) -> dict[str, Any] | None:
        if ticket_id is None:
            return None
        row = self.open_rows.get(int(ticket_id))
        return dict(row) if row else None

    def summarize_history(self) -> dict[str, Any]:
        pnls = [float(h.get("pnl") or 0) for h in self.history.values()]
        return {
            "total_pnl": sum(pnls),
            "trades": len(pnls),
            "wins": sum(1 for x in pnls if x > 0),
            "losses": sum(1 for x in pnls if x <= 0),
        }


class TestPaperTicketAndHistory(unittest.TestCase):
    def test_paper_broker_assigns_ticket(self) -> None:
        br = PaperBroker()
        a = br.open_order(side="long", entry_price=2000.0, stop_loss=1990.0, take_profit=0.0, lot=0.1)
        b = br.open_order(side="long", entry_price=2010.0, stop_loss=2000.0, take_profit=0.0, lot=0.1)
        self.assertTrue(a.success and b.success)
        self.assertIsNotNone(a.broker_ticket)
        self.assertNotEqual(a.broker_ticket, b.broker_ticket)
        self.assertEqual(a.trade_id, str(a.broker_ticket))
        closed = br.close_order(a.trade_id, 2005.0)
        self.assertEqual(closed.broker_ticket, a.broker_ticket)

    def test_handlers_move_open_to_history(self) -> None:
        bus = EventBus()
        db = FakeWriter()
        register_workers(
            bus,
            db=db,
            telegram=TelegramNotifier(None, None),
            metrics=MetricsCollector(),
        )
        bus.start(n_workers=1)
        entry = {
            "signal_id": "s1",
            "symbol": "XAUUSD",
            "side": "long",
            "entry_time": datetime(2024, 1, 1, tzinfo=timezone.utc),
            "entry_price": 2000.0,
            "stop_loss": 1990.0,
            "take_profit": 0.0,
            "lot": 0.1,
            "risk_pct": 0.5,
            "ticket_id": 42,
            "broker_ticket": 42,
        }
        bus.publish(make_event(EventType.TRADE_OPENED, entry, correlation_id="c1"))
        for _ in range(50):
            if 42 in db.open_rows:
                break
            time.sleep(0.02)
        self.assertIn(42, db.open_rows)
        self.assertEqual(db.open_rows[42]["entry_price"], 2000.0)

        close = {
            "ticket_id": 42,
            "exit_time": datetime(2024, 1, 2, tzinfo=timezone.utc),
            "exit_price": 2010.0,
            "pnl": 100.0,
            "pnl_r": 1.0,
            "duration_seconds": 86400,
            "exit_reason": "tp",
            "entry_price": 9999.0,
        }
        bus.publish(make_event(EventType.TRADE_CLOSED, close, correlation_id="c1"))
        for _ in range(50):
            if 42 in db.history and 42 not in db.open_rows:
                break
            time.sleep(0.02)
        bus.stop()
        self.assertNotIn(42, db.open_rows)
        h = db.history[42]
        self.assertEqual(h["entry_price"], 2000.0)
        self.assertEqual(h["exit_price"], 2010.0)
        self.assertEqual(h["exit_reason"], "tp")

    def test_close_by_ticket_id_only(self) -> None:
        db = FakeWriter()
        db.insert_open_trade(
            {
                "correlation_id": "c2",
                "symbol": "XAUUSD",
                "side": "short",
                "entry_time": datetime(2024, 1, 1, tzinfo=timezone.utc),
                "entry_price": 2000.0,
                "stop_loss": 2010.0,
                "take_profit": 0.0,
                "lot": 0.1,
                "risk_pct": 0.5,
                "ticket_id": 99,
            }
        )
        db.close_trade_to_history(
            {
                "ticket_id": 99,
                "exit_time": datetime(2024, 1, 3, tzinfo=timezone.utc),
                "exit_price": 1990.0,
                "pnl": 50.0,
                "exit_reason": "sl",
            }
        )
        self.assertEqual(db.open_rows, {})
        self.assertEqual(db.history[99]["exit_reason"], "sl")
        self.assertEqual(db.history[99]["entry_price"], 2000.0)

    def test_history_fill_empty_only(self) -> None:
        db = FakeWriter()
        db.history[7] = {
            "ticket_id": 7,
            "entry_price": 2000.0,
            "exit_price": 2010.0,
            "exit_reason": "tp",
            "pnl": None,
        }
        db.close_trade_to_history(
            {
                "ticket_id": 7,
                "exit_price": 9999.0,
                "pnl": 12.0,
                "exit_reason": "sl",
            }
        )
        h = db.history[7]
        self.assertEqual(h["exit_price"], 2010.0)
        self.assertEqual(h["exit_reason"], "tp")
        self.assertEqual(h["pnl"], 12.0)

    def test_summarize_history(self) -> None:
        db = FakeWriter()
        db.history[1] = {"pnl": 10.0}
        db.history[2] = {"pnl": -3.0}
        s = db.summarize_history()
        self.assertEqual(s["total_pnl"], 7.0)
        self.assertEqual(s["trades"], 2)
        self.assertEqual(s["wins"], 1)
        self.assertEqual(s["losses"], 1)


if __name__ == "__main__":
    unittest.main()
