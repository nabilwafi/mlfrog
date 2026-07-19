"""CLI: Production paper trading runtime (Sprint 27).

Frozen research stack. Validates infrastructure under paper fills.

Examples:
  # Dry-run replay from heat trades (no Postgres / Telegram required)
  python apps/run_paper_trading.py --mode replay --max-signals 20

  # Live idle loop with monitoring (needs config)
  python apps/run_paper_trading.py --mode loop
"""

from __future__ import annotations

import argparse
import logging
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import yaml

from production.db.writer import PostgresWriter
from production.events.bus import EventBus, MetricsCollector
from production.logging.structured import setup_json_logging
from production.monitoring.health import HealthReporter
from production.monitoring.mt5_probe import make_mt5_probe
from production.monitoring.mt5_session import disconnect as mt5_disconnect
from production.monitoring.server import start_monitoring_server
from production.live.signal_source import LiveMT5SignalSource
from production.paper.broker import PaperBroker
from production.paper.pipeline import IncomingSignal, ProductionPipeline
from production.paper.runtime import IdleSignalSource, PaperRuntime, ReplaySignalSource
from production.paper.state import PortfolioState
from production.telegram.bot import TelegramNotifier, split_chat_and_thread
from production.workers.handlers import register_workers
from settings.paths import MT5_CONFIG, MT5_CONFIG_EXAMPLE, PAPER, ROOT
from settings.strategy import STARTING_EQUITY


def _trade_thread_id(tg_cfg: dict[str, Any]) -> Any:
    """Prefer trade_thread_id; fall back to legacy message_thread_id."""
    if tg_cfg.get("trade_thread_id") is not None:
        return tg_cfg.get("trade_thread_id")
    return tg_cfg.get("message_thread_id")


def _build_telegram(tg_cfg: dict[str, Any], *, chat_key: str, thread_key: str) -> TelegramNotifier:
    """Build notifier; supports forum topics via trade/health/daily thread ids."""
    enabled = bool(tg_cfg.get("enabled", False))
    token = tg_cfg.get("bot_token")
    raw_chat = tg_cfg.get(chat_key)
    # allow shorthand chat_id:thread in the chat field
    chat_id, thread_from_chat = split_chat_and_thread(str(raw_chat) if raw_chat is not None else None)
    if thread_key in {"trade_thread_id", "message_thread_id"}:
        thread = _trade_thread_id(tg_cfg)
    else:
        thread = tg_cfg.get(thread_key)
    if thread is None:
        thread = thread_from_chat
    # health/daily fall back to main chat_id if dedicated chat omitted
    if chat_key != "chat_id" and not chat_id:
        chat_id, _ = split_chat_and_thread(str(tg_cfg.get("chat_id") or "") or None)
    return TelegramNotifier(
        token,
        chat_id,
        message_thread_id=thread,
        enabled=enabled and bool(chat_id),
        verify_ssl=bool(tg_cfg.get("verify_ssl", True)),
    )


def _load_config(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(f"config not found: {path} (copy {MT5_CONFIG_EXAMPLE.name})")
    with path.open(encoding="utf-8") as fh:
        data = yaml.safe_load(fh) or {}
    if not isinstance(data, dict):
        raise ValueError("config root must be a mapping")
    return data


def _resolve(path: str) -> Path:
    p = Path(path)
    return p if p.is_absolute() else (ROOT / p).resolve()


def _load_replay_signals(heat_path: Path, *, limit: int | None) -> list[IncomingSignal]:
    import pandas as pd

    df = pd.read_parquet(heat_path)
    # Prefer accepted heat trades for infrastructure validation; include some skips via meta/conf
    rows = df.sort_values("timestamp").reset_index(drop=True)
    if limit:
        rows = rows.head(limit)
    out: list[IncomingSignal] = []
    for _, r in rows.iterrows():
        ts = pd.Timestamp(r["timestamp"])
        if ts.tzinfo is None:
            ts = ts.tz_localize("UTC")
        else:
            ts = ts.tz_convert("UTC")
        sess = "overlap"
        if float(r.get("session_london_ny_overlap", 0) or 0) >= 0.5:
            sess = "overlap"
        elif float(r.get("session_london", 0) or 0) >= 0.5:
            sess = "london"
        elif float(r.get("session_newyork", 0) or 0) >= 0.5:
            sess = "newyork"
        elif float(r.get("session_asia", 0) or 0) >= 0.5:
            sess = "asia"
        out.append(
            IncomingSignal(
                timestamp=ts.to_pydatetime(),
                symbol="XAUUSD",
                side=str(r["side"]),
                probability=float(r.get("raw_probability", r.get("probability", 0.5)) or 0.5),
                meta_probability=float(r.get("meta_proba", 0.5) or 0.5),
                confidence=float(r.get("confidence", 50) or 50),
                entry_price=float(r["entry_price"]),
                atr=float(r.get("atr_entry", r.get("atr_price", 1.0)) or 1.0),
                session=sess,
                regime=str(r.get("d1_regime", "unknown") or "unknown"),
                bar_key=ts.isoformat() + ":" + str(r["side"]),
            )
        )
    return out


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Production paper trading (Sprint 27)")
    p.add_argument("--config", type=Path, default=MT5_CONFIG)
    p.add_argument("--mode", choices=("replay", "loop", "live"), default="replay")
    p.add_argument("--max-signals", type=int, default=50)
    p.add_argument("--heat-trades", type=Path, default=None)
    p.add_argument("--apply-schema", action="store_true")
    return p


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    cfg = _load_config(Path(args.config))
    setup_json_logging()

    paper_cfg = dict(cfg.get("paper_trading") or {})
    starting = float(paper_cfg.get("starting_equity", STARTING_EQUITY))
    poll = float(paper_cfg.get("poll_seconds", 5.0))
    mon_host = str(paper_cfg.get("monitoring_host", "127.0.0.1"))
    mon_port = int(paper_cfg.get("monitoring_port", 8787))
    bus_workers = int(paper_cfg.get("event_workers", 2))

    tg_cfg = dict(paper_cfg.get("telegram") or {})
    db_dsn = paper_cfg.get("postgres_dsn")

    bus = EventBus(maxsize=int(paper_cfg.get("event_queue_size", 10_000)))
    metrics = MetricsCollector()
    db = PostgresWriter(str(db_dsn) if db_dsn else None)
    telegram = _build_telegram(tg_cfg, chat_key="chat_id", thread_key="trade_thread_id")
    health_telegram = _build_telegram(
        tg_cfg,
        chat_key="health_chat_id",
        thread_key="health_message_thread_id",
    )
    daily_telegram = _build_telegram(
        tg_cfg,
        chat_key="daily_chat_id",
        thread_key="daily_message_thread_id",
    )
    error_telegram = _build_telegram(
        tg_cfg,
        chat_key="error_chat_id",
        thread_key="error_thread_id",
    )
    trade_thread = _trade_thread_id(tg_cfg)
    # if dedicated chat missing, reuse trade chat + topic thread (or trade thread)
    if not health_telegram.enabled and bool(tg_cfg.get("enabled")) and tg_cfg.get("chat_id"):
        health_telegram = TelegramNotifier(
            tg_cfg.get("bot_token"),
            split_chat_and_thread(str(tg_cfg.get("chat_id")))[0],
            message_thread_id=tg_cfg.get("health_message_thread_id") or trade_thread,
            enabled=True,
            verify_ssl=bool(tg_cfg.get("verify_ssl", True)),
        )
    if not daily_telegram.enabled and bool(tg_cfg.get("enabled")) and tg_cfg.get("chat_id"):
        daily_telegram = TelegramNotifier(
            tg_cfg.get("bot_token"),
            split_chat_and_thread(str(tg_cfg.get("chat_id")))[0],
            message_thread_id=tg_cfg.get("daily_message_thread_id") or trade_thread,
            enabled=True,
            verify_ssl=bool(tg_cfg.get("verify_ssl", True)),
        )
    if not error_telegram.enabled and bool(tg_cfg.get("enabled")) and tg_cfg.get("chat_id"):
        error_telegram = TelegramNotifier(
            tg_cfg.get("bot_token"),
            split_chat_and_thread(str(tg_cfg.get("chat_id")))[0],
            message_thread_id=tg_cfg.get("error_thread_id") or trade_thread,
            enabled=True,
            verify_ssl=bool(tg_cfg.get("verify_ssl", True)),
        )
    log = logging.getLogger(__name__)
    if not telegram.enabled:
        log.warning(
            "telegram_disabled — set paper_trading.telegram chat_id (+ optional trade_thread_id for topics)",
        )
    if not health_telegram.enabled:
        log.warning(
            "health_telegram_disabled — set health_chat_id / health_message_thread_id for HEALTHCHECK topic",
        )
    if not daily_telegram.enabled:
        log.warning(
            "daily_telegram_disabled — set daily_message_thread_id for DAILY REPORT topic",
        )
    if not error_telegram.enabled:
        log.warning(
            "error_telegram_disabled — set error_thread_id for EXECUTION ERROR topic",
        )
    else:
        log.info(
            "telegram_ready trades_chat=%s trade_thread=%s health_thread=%s daily_thread=%s error_thread=%s",
            tg_cfg.get("chat_id"),
            trade_thread,
            tg_cfg.get("health_message_thread_id"),
            tg_cfg.get("daily_message_thread_id"),
            tg_cfg.get("error_thread_id"),
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

    if args.apply_schema and db.enabled:
        schema = (ROOT / "sql" / "production_schema.sql").read_text(encoding="utf-8")
        db.apply_schema(schema)
        print("schema applied")

    state = PortfolioState(equity=starting, peak_equity=starting)
    env_name = str(paper_cfg.get("environment") or ("live" if args.mode == "live" else "paper"))
    pipeline = ProductionPipeline(
        bus=bus,
        state=state,
        broker=PaperBroker(),
        symbol=str(cfg.get("symbol", "XAUUSD")),
        environment=env_name,
    )
    check_mt5 = bool(paper_cfg.get("check_mt5", True))
    health = HealthReporter(
        bus,
        state,
        interval_seconds=float(paper_cfg.get("health_interval_seconds", 300)),
        probe_throttle_seconds=float(paper_cfg.get("health_probe_throttle_seconds", 60)),
        mt5_probe=make_mt5_probe(cfg) if check_mt5 else None,
        environment=env_name,
    )
    server = start_monitoring_server(
        host=mon_host, port=mon_port, metrics=metrics, state=state, bus=bus, health=health
    )
    health.start()

    PAPER.mkdir(parents=True, exist_ok=True)

    try:
        if args.mode == "replay":
            heat_root = _resolve(str((cfg.get("portfolio_heat") or {}).get("output_directory", "artifacts/research/portfolio_heat")))
            heat_path = Path(args.heat_trades) if args.heat_trades else heat_root / "best_policy_trades.parquet"
            if not heat_path.is_file():
                raise FileNotFoundError(f"missing {heat_path}")
            signals = _load_replay_signals(heat_path, limit=args.max_signals)
            source = ReplaySignalSource(signals)
            # drain all signals then stop
            while not source.exhausted:
                for sig in source.next_signals():
                    pipeline.process_signal(sig)
            # close remaining on last known prices (mark flat close)
            now = datetime.now(timezone.utc)
            for tid, pos in list(state.open_positions.items()):
                pipeline.on_bar(high=pos.entry_price, low=pos.entry_price, close=pos.entry_price, timestamp=now)
            pipeline.emit_daily_summary()
            time.sleep(0.5)  # allow workers to drain
            snap = metrics.snapshot()
            (PAPER / "last_replay_metrics.json").write_text(
                __import__("json").dumps({"equity": state.equity, "metrics": snap}, indent=2, default=str),
                encoding="utf-8",
            )
            print(f"replay done equity={state.equity:.2f} opened_keys={state.trades_today} skips={state.skips}")
            print(f"monitoring was on http://{mon_host}:{mon_port}/metrics")
        elif args.mode == "live":
            live_cfg = dict(paper_cfg.get("live") or {})
            source = LiveMT5SignalSource(
                cfg,
                symbol=str(cfg.get("symbol", "XAUUSD")),
                timeframe=str(live_cfg.get("timeframe", cfg.get("timeframe", "H1"))),
                history_bars=int(live_cfg.get("history_bars", 400)),
            )
            runtime = PaperRuntime(
                pipeline=pipeline,
                bus=bus,
                source=source,
                poll_seconds=float(live_cfg.get("poll_seconds", poll)),
            )
            print(f"live paper trading on {cfg.get('symbol', 'XAUUSD')} — MT5 candles + frozen stack + paper fills")
            print(f"monitoring http://{mon_host}:{mon_port}/health")
            runtime.run_forever()
        else:
            runtime = PaperRuntime(
                pipeline=pipeline,
                bus=bus,
                source=IdleSignalSource(),
                poll_seconds=poll,
            )
            print(f"paper loop listening; monitoring http://{mon_host}:{mon_port}/health")
            runtime.run_forever()
    finally:
        health.stop()
        time.sleep(0.3)
        bus.stop()
        server.shutdown()
        mt5_disconnect()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
