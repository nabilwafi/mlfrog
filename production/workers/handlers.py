"""Background workers — consume EventBus; failures isolated."""

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
    # Health / daily / error can use separate forum topics from trade notifications
    health_tg = health_telegram if health_telegram is not None else telegram
    daily_tg = daily_telegram if daily_telegram is not None else telegram
    error_tg = error_telegram if error_telegram is not None else telegram
    def on_signal(ev: ProductionEvent) -> None:
        t0 = time.perf_counter()
        p = ev.payload
        db.upsert_signal(
            {
                "signal_id": p.get("signal_id"),
                "correlation_id": ev.correlation_id,
                "timestamp": ev.timestamp,
                "symbol": p.get("symbol"),
                "side": p.get("side"),
                "probability": p.get("probability"),
                "meta_probability": p.get("meta_probability"),
                "confidence": p.get("confidence"),
                "threshold_meta": p.get("threshold_meta"),
                "threshold_confidence": p.get("threshold_confidence"),
                "model_version": p.get("model_version"),
                "meta_version": p.get("meta_version"),
                "feature_version": p.get("feature_version"),
                "label_version": p.get("label_version"),
                "pipeline_version": p.get("pipeline_version"),
                "accepted": bool(p.get("accepted")),
                "session": p.get("session"),
                "regime": p.get("regime"),
            }
        )
        metrics.observe_ms("db_write_signal_ms", (time.perf_counter() - t0) * 1000)
        metrics.incr("signals_written")

    def on_opened(ev: ProductionEvent) -> None:
        t0 = time.perf_counter()
        p = dict(ev.payload)
        p["correlation_id"] = ev.correlation_id
        p.setdefault("exit_time", None)
        p.setdefault("exit_price", None)
        p.setdefault("pnl", None)
        p.setdefault("pnl_r", None)
        p.setdefault("duration_seconds", None)
        p.setdefault("mae", None)
        p.setdefault("mfe", None)
        p.setdefault("exit_reason", None)
        p["status"] = "open"
        db.upsert_trade(p)
        db.insert_execution(
            {
                "correlation_id": ev.correlation_id,
                "trade_id": p.get("trade_id"),
                "timestamp": ev.timestamp,
                "latency_ms": p.get("latency_ms"),
                "broker_response": p.get("broker_response", "OK"),
                "spread": p.get("spread"),
                "slippage": p.get("slippage"),
                "retry_count": p.get("retry_count", 0),
                "success": True,
                "error_message": None,
            }
        )
        metrics.observe_ms("db_write_trade_ms", (time.perf_counter() - t0) * 1000)
        metrics.incr("trades_opened")
        t1 = time.perf_counter()
        telegram.send(fmt_new_trade(p))
        metrics.observe_ms("telegram_ms", (time.perf_counter() - t1) * 1000)

    def on_closed(ev: ProductionEvent) -> None:
        p = dict(ev.payload)
        p["correlation_id"] = ev.correlation_id
        p["status"] = "closed"
        db.upsert_trade(p)
        metrics.incr("trades_closed")
        telegram.send(fmt_trade_closed(p))

    def on_skipped(ev: ProductionEvent) -> None:
        p = ev.payload
        db.insert_skip(
            {
                "correlation_id": ev.correlation_id,
                "signal_id": p.get("signal_id"),
                "timestamp": ev.timestamp,
                "symbol": p.get("symbol"),
                "reason": p.get("reason"),
                "threshold": p.get("threshold"),
                "current_value": p.get("current_value"),
                "detail": p.get("detail") or {},
            }
        )
        metrics.incr(f"skip_{p.get('reason', 'unknown')}")
        telegram.send(fmt_skipped({**p, "correlation_id": ev.correlation_id}))

    def on_error(ev: ProductionEvent) -> None:
        p = ev.payload
        db.insert_execution(
            {
                "correlation_id": ev.correlation_id,
                "trade_id": p.get("trade_id"),
                "timestamp": ev.timestamp,
                "latency_ms": p.get("latency_ms"),
                "broker_response": p.get("broker_response", "ERROR"),
                "spread": p.get("spread"),
                "slippage": p.get("slippage"),
                "retry_count": p.get("retry_count", 0),
                "success": False,
                "error_message": p.get("error_message"),
            }
        )
        metrics.incr("execution_errors")
        t0 = time.perf_counter()
        error_tg.send(fmt_error({**p, "timestamp": ev.timestamp.isoformat()}))
        metrics.observe_ms("telegram_error_ms", (time.perf_counter() - t0) * 1000)

    def on_metric(ev: ProductionEvent) -> None:
        p = ev.payload
        name = str(p.get("name"))
        value = float(p.get("value", 0))
        metrics.incr(name) if p.get("as_counter") else metrics.observe_ms(name, value)
        db.insert_metric(name, value, labels=p.get("labels"), correlation_id=ev.correlation_id)

    def on_audit(ev: ProductionEvent) -> None:
        p = ev.payload
        db.insert_audit(
            str(p.get("component", "unknown")),
            str(p.get("action", "event")),
            p.get("detail") or {},
            correlation_id=ev.correlation_id,
        )

    def on_daily(ev: ProductionEvent) -> None:
        p = dict(ev.payload)
        db.upsert_daily(p)
        t0 = time.perf_counter()
        daily_tg.send(fmt_daily(p))
        metrics.observe_ms("telegram_daily_ms", (time.perf_counter() - t0) * 1000)
        metrics.incr("daily_reports")

    def on_heat(ev: ProductionEvent) -> None:
        metrics.incr("heat_triggered")
        db.insert_audit("heat", "triggered", ev.payload, correlation_id=ev.correlation_id)

    def on_health(ev: ProductionEvent) -> None:
        p = dict(ev.payload)
        p.setdefault("timestamp", ev.timestamp.isoformat())
        db.insert_audit("health", str(p.get("reason", "ping")), p, correlation_id=ev.correlation_id)
        t0 = time.perf_counter()
        health_tg.send(fmt_health(p))
        metrics.observe_ms("telegram_health_ms", (time.perf_counter() - t0) * 1000)
        metrics.incr("health_notified")

    def on_candle(ev: ProductionEvent) -> None:
        t0 = time.perf_counter()
        p = dict(ev.payload)
        db.upsert_candle(p)
        metrics.observe_ms("db_write_candle_ms", (time.perf_counter() - t0) * 1000)
        metrics.incr("candles_written")

    bus.subscribe(EventType.SIGNAL, on_signal)
    bus.subscribe(EventType.TRADE_OPENED, on_opened)
    bus.subscribe(EventType.TRADE_CLOSED, on_closed)
    bus.subscribe(EventType.TRADE_SKIPPED, on_skipped)
    bus.subscribe(EventType.EXECUTION_ERROR, on_error)
    bus.subscribe(EventType.METRIC, on_metric)
    bus.subscribe(EventType.AUDIT, on_audit)
    bus.subscribe(EventType.DAILY_SUMMARY, on_daily)
    bus.subscribe(EventType.HEAT_TRIGGERED, on_heat)
    bus.subscribe(EventType.HEALTH, on_health)
    bus.subscribe(EventType.CANDLE_CLOSED, on_candle)
