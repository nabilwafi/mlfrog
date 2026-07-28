"""Background workers — consume EventBus; failures isolated.

Lean DB: only TRADE_OPENED / TRADE_CLOSED / CANDLE_CLOSED write Postgres.
"""

from __future__ import annotations

import logging
import time
from typing import Any

from production.db.writer import PostgresWriter
from production.events.bus import EventBus, MetricsCollector
from production.events.types import EventType, ProductionEvent
from production.telegram.bot import (
    TelegramNotifier,
    fmt_daily,
    fmt_error,
    fmt_health,
    fmt_new_trade,
    fmt_skipped,
    fmt_trail_update,
    fmt_trade_closed,
)

logger = logging.getLogger(__name__)


def register_workers(
    bus: EventBus,
    *,
    db: PostgresWriter,
    telegram: TelegramNotifier,
    metrics: MetricsCollector,
    health_telegram: TelegramNotifier | None = None,
    daily_telegram: TelegramNotifier | None = None,
    error_telegram: TelegramNotifier | None = None,
) -> None:
    health_tg = health_telegram if health_telegram is not None else telegram
    daily_tg = daily_telegram if daily_telegram is not None else telegram
    error_tg = error_telegram if error_telegram is not None else telegram

    def on_signal(ev: ProductionEvent) -> None:
        metrics.incr("signals_seen")
        if ev.payload.get("accepted"):
            metrics.incr("signals_accepted")

    def on_opened(ev: ProductionEvent) -> None:
        t0 = time.perf_counter()
        p = dict(ev.payload)
        p["correlation_id"] = ev.correlation_id
        if p.get("ticket_id") is None and p.get("broker_ticket") is not None:
            p["ticket_id"] = p.get("broker_ticket")
        db.insert_open_trade(p)
        metrics.observe_ms("db_write_trade_ms", (time.perf_counter() - t0) * 1000)
        metrics.incr("trades_opened")
        t1 = time.perf_counter()
        telegram.send(fmt_new_trade(p))
        metrics.observe_ms("telegram_ms", (time.perf_counter() - t1) * 1000)

    def on_closed(ev: ProductionEvent) -> None:
        p = dict(ev.payload)
        p["correlation_id"] = ev.correlation_id
        if p.get("ticket_id") is None and p.get("broker_ticket") is not None:
            p["ticket_id"] = p.get("broker_ticket")
        db.close_trade_to_history(p)
        metrics.incr("trades_closed")
        telegram.send(fmt_trade_closed(p))

    def on_trail_update(ev: ProductionEvent) -> None:
        p = dict(ev.payload)
        p["correlation_id"] = ev.correlation_id
        metrics.incr("trail_updates")
        telegram.send(fmt_trail_update(p))

    def on_skipped(ev: ProductionEvent) -> None:
        p = ev.payload
        metrics.incr(f"skip_{p.get('reason', 'unknown')}")
        telegram.send(fmt_skipped({**p, "correlation_id": ev.correlation_id}))

    def on_error(ev: ProductionEvent) -> None:
        p = ev.payload
        metrics.incr("execution_errors")
        t0 = time.perf_counter()
        error_tg.send(fmt_error({**p, "timestamp": ev.timestamp.isoformat()}))
        metrics.observe_ms("telegram_error_ms", (time.perf_counter() - t0) * 1000)

    def on_metric(ev: ProductionEvent) -> None:
        p = ev.payload
        name = str(p.get("name"))
        value = float(p.get("value", 0))
        metrics.incr(name) if p.get("as_counter") else metrics.observe_ms(name, value)

    def on_audit(ev: ProductionEvent) -> None:
        metrics.incr(f"audit_{ev.payload.get('action', 'event')}")

    def on_daily(ev: ProductionEvent) -> None:
        p = dict(ev.payload)
        t0 = time.perf_counter()
        daily_tg.send(fmt_daily(p))
        metrics.observe_ms("telegram_daily_ms", (time.perf_counter() - t0) * 1000)
        metrics.incr("daily_reports")

    def on_heat(ev: ProductionEvent) -> None:
        metrics.incr("heat_triggered")

    def on_health(ev: ProductionEvent) -> None:
        p = dict(ev.payload)
        p.setdefault("timestamp", ev.timestamp.isoformat())
        t0 = time.perf_counter()
        health_tg.send(fmt_health(p))
        metrics.observe_ms("telegram_health_ms", (time.perf_counter() - t0) * 1000)
        metrics.incr("health_notified")

    def on_candle(ev: ProductionEvent) -> None:
        t0 = time.perf_counter()
        db.upsert_candle(dict(ev.payload))
        metrics.observe_ms("db_write_candle_ms", (time.perf_counter() - t0) * 1000)
        metrics.incr("candles_written")

    bus.subscribe(EventType.SIGNAL, on_signal)
    bus.subscribe(EventType.TRADE_OPENED, on_opened)
    bus.subscribe(EventType.TRADE_CLOSED, on_closed)
    bus.subscribe(EventType.TRAIL_UPDATE, on_trail_update)
    bus.subscribe(EventType.TRADE_SKIPPED, on_skipped)
    bus.subscribe(EventType.EXECUTION_ERROR, on_error)
    bus.subscribe(EventType.METRIC, on_metric)
    bus.subscribe(EventType.AUDIT, on_audit)
    bus.subscribe(EventType.DAILY_SUMMARY, on_daily)
    bus.subscribe(EventType.HEAT_TRIGGERED, on_heat)
    bus.subscribe(EventType.HEALTH, on_health)
    bus.subscribe(EventType.CANDLE_CLOSED, on_candle)
