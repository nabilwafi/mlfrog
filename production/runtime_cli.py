"""Shared CLI bootstrap for run_paper_trading / run_live_trading."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Callable

import yaml

from production.db.writer import PostgresWriter
from production.events.bus import EventBus, MetricsCollector
from production.paper.state import PortfolioState
from production.telegram.bot import TelegramNotifier, split_chat_and_thread
from production.telegram.commands import (
    TelegramCommandListener,
    build_candle_check,
    build_positions_check,
    build_summary_check,
)
from production.workers.handlers import register_workers
from settings.paths import MT5_CONFIG_EXAMPLE, ROOT


def load_config(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(f"config not found: {path} (copy {MT5_CONFIG_EXAMPLE.name})")
    with path.open(encoding="utf-8") as fh:
        data = yaml.safe_load(fh) or {}
    if not isinstance(data, dict):
        raise ValueError("config root must be a mapping")
    return data


def trade_thread_id(tg_cfg: dict[str, Any]) -> Any:
    if tg_cfg.get("trade_thread_id") is not None:
        return tg_cfg.get("trade_thread_id")
    return tg_cfg.get("message_thread_id")


def build_telegram(tg_cfg: dict[str, Any], *, chat_key: str, thread_key: str) -> TelegramNotifier:
    enabled = bool(tg_cfg.get("enabled", False))
    token = tg_cfg.get("bot_token")
    raw_chat = tg_cfg.get(chat_key)
    chat_id, thread_from_chat = split_chat_and_thread(str(raw_chat) if raw_chat is not None else None)
    if thread_key in {"trade_thread_id", "message_thread_id"}:
        thread = trade_thread_id(tg_cfg)
    else:
        thread = tg_cfg.get(thread_key)
    if thread is None:
        thread = thread_from_chat
    if chat_key != "chat_id" and not chat_id:
        chat_id, _ = split_chat_and_thread(str(tg_cfg.get("chat_id") or "") or None)
    return TelegramNotifier(
        token,
        chat_id,
        message_thread_id=thread,
        enabled=enabled and bool(chat_id),
        verify_ssl=bool(tg_cfg.get("verify_ssl", True)),
    )


def wire_topic_fallbacks(tg_cfg: dict[str, Any], *, telegram: TelegramNotifier,
                         health: TelegramNotifier, daily: TelegramNotifier,
                         error: TelegramNotifier) -> tuple[TelegramNotifier, TelegramNotifier, TelegramNotifier]:
    trade_thread = trade_thread_id(tg_cfg)
    if not health.enabled and bool(tg_cfg.get("enabled")) and tg_cfg.get("chat_id"):
        health = TelegramNotifier(
            tg_cfg.get("bot_token"),
            split_chat_and_thread(str(tg_cfg.get("chat_id")))[0],
            message_thread_id=tg_cfg.get("health_message_thread_id") or trade_thread,
            enabled=True,
            verify_ssl=bool(tg_cfg.get("verify_ssl", True)),
        )
    if not daily.enabled and bool(tg_cfg.get("enabled")) and tg_cfg.get("chat_id"):
        daily = TelegramNotifier(
            tg_cfg.get("bot_token"),
            split_chat_and_thread(str(tg_cfg.get("chat_id")))[0],
            message_thread_id=tg_cfg.get("daily_message_thread_id") or trade_thread,
            enabled=True,
            verify_ssl=bool(tg_cfg.get("verify_ssl", True)),
        )
    if not error.enabled and bool(tg_cfg.get("enabled")) and tg_cfg.get("chat_id"):
        error = TelegramNotifier(
            tg_cfg.get("bot_token"),
            split_chat_and_thread(str(tg_cfg.get("chat_id")))[0],
            message_thread_id=tg_cfg.get("error_thread_id") or trade_thread,
            enabled=True,
            verify_ssl=bool(tg_cfg.get("verify_ssl", True)),
        )
    return health, daily, error


def start_stack(
    *,
    dsn: str | None,
    schema: str,
    schema_sql_name: str,
    apply_schema: bool,
    tg_cfg: dict[str, Any],
    bus_workers: int,
    queue_size: int,
) -> tuple[EventBus, MetricsCollector, PostgresWriter, TelegramNotifier, TelegramNotifier, TelegramNotifier, TelegramNotifier]:
    log = logging.getLogger(__name__)
    bus = EventBus(maxsize=int(queue_size))
    metrics = MetricsCollector()
    db = PostgresWriter(str(dsn) if dsn else None, schema=schema)
    if dsn and not db.enabled:
        log.error(
            "postgres_disabled — DSN set but writer off (install: pip install psycopg2-binary). "
            "Candles/trades will NOT be written to DB."
        )
    elif not dsn:
        log.warning("postgres_dsn empty — DB writes no-op")
    else:
        log.info(
            "postgres_enabled schema=%s dsn_host=%s",
            schema,
            str(dsn).split("@")[-1] if "@" in str(dsn) else "set",
        )

    telegram = build_telegram(tg_cfg, chat_key="chat_id", thread_key="trade_thread_id")
    health_telegram = build_telegram(tg_cfg, chat_key="health_chat_id", thread_key="health_message_thread_id")
    daily_telegram = build_telegram(tg_cfg, chat_key="daily_chat_id", thread_key="daily_message_thread_id")
    error_telegram = build_telegram(tg_cfg, chat_key="error_chat_id", thread_key="error_thread_id")
    health_telegram, daily_telegram, error_telegram = wire_topic_fallbacks(
        tg_cfg, telegram=telegram, health=health_telegram, daily=daily_telegram, error=error_telegram
    )

    register_workers(
        bus,
        db=db,
        telegram=telegram,
        metrics=metrics,
        health_telegram=health_telegram,
        daily_telegram=daily_telegram,
        error_telegram=error_telegram,
    )
    bus.start(n_workers=bus_workers)

    if apply_schema and db.enabled:
        sql_path = ROOT / "sql" / schema_sql_name
        db.apply_schema(sql_path.read_text(encoding="utf-8"))
        print(f"schema applied ({schema_sql_name})")
    elif db.enabled:
        try:
            db.ensure_schema()
        except Exception:
            log.exception("ensure_schema_failed")

    return bus, metrics, db, telegram, health_telegram, daily_telegram, error_telegram


def start_telegram_commands(
    tg_cfg: dict[str, Any],
    *,
    cfg: dict[str, Any],
    state: PortfolioState,
    db: PostgresWriter,
    symbol: str,
    timeframe: str,
    environment: str,
    verify_ssl: bool = True,
    use_mt5_account: bool = True,
) -> TelegramCommandListener | None:
    if not bool(tg_cfg.get("enabled")) or not tg_cfg.get("bot_token"):
        return None
    allowed: set[str] = set()
    for key in ("chat_id", "health_chat_id", "daily_chat_id", "error_chat_id"):
        cid, _ = split_chat_and_thread(str(tg_cfg.get(key) or "") or None)
        if cid:
            allowed.add(cid)
    if not allowed:
        return None
    primary = next(iter(sorted(allowed)))
    tg = TelegramNotifier(
        tg_cfg.get("bot_token"),
        primary,
        enabled=True,
        verify_ssl=verify_ssl,
    )
    tf = str(timeframe or "H1").upper()

    def on_candle() -> dict[str, Any]:
        return build_candle_check(cfg=cfg, symbol=symbol, timeframe=tf)

    def on_summary() -> dict[str, Any]:
        return build_summary_check(
            state=state,
            symbol=symbol,
            environment=environment,
            db=db,
            use_mt5_account=use_mt5_account,
        )

    def on_positions() -> dict[str, Any]:
        return build_positions_check(cfg=cfg, state=state, symbol=symbol, timeframe=tf)

    listener = TelegramCommandListener(
        tg,
        allowed_chat_ids=allowed,
        on_check_candle=on_candle,
        on_check_summary=on_summary,
        on_check_positions=on_positions,
    )
    listener.start()
    print(f"telegram commands: /candles /summary /positions (chats={len(allowed)})")
    return listener
