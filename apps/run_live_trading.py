"""CLI: Live trading runtime (production schema). Real MT5 order_send with --execute.

Examples:
  python apps/run_live_trading.py              # dry-run orders
  python apps/run_live_trading.py --execute    # REAL order_send
  python apps/run_live_trading.py --apply-schema
"""

from __future__ import annotations

import argparse
import logging
import sys
import time
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import tools.pyc_path_hook  # noqa: F401,E402

from production import EXECUTION_ENABLED, LIVE_SYMBOL, RESEARCH_SYMBOL
from production.live.account import fetch_account, snapshot_dict, sync_equity_into_state
from production.live.broker import LiveBroker
from production.live.signal_source import LiveMT5SignalSource
from production.logging.structured import setup_json_logging
from production.monitoring.health import HealthReporter
from production.monitoring.mt5_probe import make_mt5_probe
from production.monitoring.mt5_session import disconnect as mt5_disconnect
from production.monitoring.server import start_monitoring_server
from production.paper.pipeline import ProductionPipeline
from production.paper.runtime import PaperRuntime
from production.paper.state import PortfolioState
from production.runtime_cli import load_config, start_stack, start_telegram_commands
from settings.paths import MT5_CONFIG, PAPER
from settings.strategy import STARTING_EQUITY


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Live trading (production schema, real MT5 with --execute)")
    p.add_argument("--config", type=Path, default=MT5_CONFIG)
    p.add_argument(
        "--execute",
        action="store_true",
        help="Actually call MT5 order_send (default is dry-run)",
    )
    p.add_argument("--apply-schema", action="store_true")
    return p


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    cfg = load_config(Path(args.config))
    setup_json_logging()
    log = logging.getLogger(__name__)

    # Prefer live_trading config; fall back to paper_trading section for telegram/dsn/poll
    live_section = dict(cfg.get("live_trading") or {})
    paper_cfg = dict(cfg.get("paper_trading") or {})
    rt_cfg = {**paper_cfg, **live_section}

    starting = float(rt_cfg.get("starting_equity", STARTING_EQUITY))
    poll = float(rt_cfg.get("poll_seconds", 5.0))
    mon_host = str(rt_cfg.get("monitoring_host", "127.0.0.1"))
    mon_port = int(rt_cfg.get("monitoring_port", 8788))
    bus_workers = int(rt_cfg.get("event_workers", 2))
    tg_cfg = dict(rt_cfg.get("telegram") or paper_cfg.get("telegram") or {})
    db_dsn = rt_cfg.get("postgres_dsn") or paper_cfg.get("postgres_dsn")

    bus, metrics, db, telegram, *_ = start_stack(
        dsn=str(db_dsn) if db_dsn else None,
        schema="production",
        schema_sql_name="production_schema.sql",
        apply_schema=bool(args.apply_schema),
        tg_cfg=tg_cfg,
        bus_workers=bus_workers,
        queue_size=int(rt_cfg.get("event_queue_size", 10_000)),
    )

    state = PortfolioState(equity=starting, peak_equity=starting)
    env_name = "live"
    trade_symbol = str(rt_cfg.get("symbol") or LIVE_SYMBOL)
    do_execute = bool(args.execute) or bool(EXECUTION_ENABLED)

    broker = LiveBroker(symbol=trade_symbol, execution_enabled=do_execute)
    if do_execute:
        log.warning("LIVE EXECUTION ENABLED — real MT5 order_send ON symbol=%s", trade_symbol)
    else:
        log.warning(
            "LIVE dry-run symbol=%s — orders logged only (pass --execute to send real)",
            trade_symbol,
        )

    pipeline = ProductionPipeline(
        bus=bus,
        state=state,
        broker=broker,
        symbol=trade_symbol,
        environment=env_name,
    )
    check_mt5 = bool(rt_cfg.get("check_mt5", True))
    health = HealthReporter(
        bus,
        state,
        interval_seconds=float(rt_cfg.get("health_interval_seconds", 300)),
        probe_throttle_seconds=float(rt_cfg.get("health_probe_throttle_seconds", 60)),
        mt5_probe=make_mt5_probe(cfg) if check_mt5 else None,
        environment=env_name,
    )
    server = start_monitoring_server(
        host=mon_host, port=mon_port, metrics=metrics, state=state, bus=bus, health=health
    )
    health.start()
    PAPER.mkdir(parents=True, exist_ok=True)

    live_cfg = dict(rt_cfg.get("live") or rt_cfg)
    try:
        source = LiveMT5SignalSource(
            cfg,
            symbol=trade_symbol,
            timeframe=str(live_cfg.get("timeframe", cfg.get("timeframe", "H1"))),
            history_bars=int(live_cfg.get("history_bars", 400)),
            model_symbol=RESEARCH_SYMBOL,
        )
        source.connect()
        try:
            snap = fetch_account()
            sync_equity_into_state(state, snap)
            (PAPER / "last_account_snapshot.json").write_text(
                __import__("json").dumps(snapshot_dict(snap), indent=2),
                encoding="utf-8",
            )
            print(
                f"account login={snap.login} server={snap.server} "
                f"equity=${snap.equity:.2f} balance=${snap.balance:.2f} "
                f"lev=1:{snap.leverage:g} free_margin=${snap.free_margin:.2f}"
            )
        except Exception as exc:
            log.exception("account_sync_failed err=%s — using starting_equity=%.2f", exc, starting)

        n_rec = pipeline.recover_from_mt5(symbol=trade_symbol, db=db)
        if n_rec:
            print(f"recovered {n_rec} open MT5 position(s) — trail/reconcile active")
        elif do_execute:
            log.info("recover_none — no open positions with bot magic on %s", trade_symbol)

        runtime = PaperRuntime(
            pipeline=pipeline,
            bus=bus,
            source=source,
            poll_seconds=float(live_cfg.get("poll_seconds", poll)),
        )
        print(
            f"LIVE on {trade_symbol} (models={RESEARCH_SYMBOL}, schema=production) — "
            f"{'REAL order_send' if do_execute else 'DRY-RUN fills'}"
        )
        print(f"monitoring http://{mon_host}:{mon_port}/health")
        cmd_listener = start_telegram_commands(
            tg_cfg,
            cfg=cfg,
            state=state,
            db=db,
            symbol=trade_symbol,
            timeframe=str(live_cfg.get("timeframe", cfg.get("timeframe", "H1"))),
            environment=env_name,
            verify_ssl=bool(tg_cfg.get("verify_ssl", True)),
            use_mt5_account=True,
        )
        try:
            runtime.run_forever()
        finally:
            if cmd_listener is not None:
                cmd_listener.stop()
    finally:
        health.stop()
        time.sleep(0.3)
        bus.stop()
        server.shutdown()
        mt5_disconnect()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
