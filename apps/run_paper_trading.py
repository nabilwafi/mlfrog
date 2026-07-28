"""CLI: Paper trading runtime (testing schema). Never real MT5 order_send.

Modes:
  replay  — dry replay from heat trades parquet (no MT5)
  loop    — idle loop + monitoring only
  paper   — MT5 candles + frozen stack + paper fills

Examples:
  python apps/run_paper_trading.py --mode replay --max-signals 20
  python apps/run_paper_trading.py --mode paper
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

import tools.pyc_path_hook  # noqa: F401,E402

from production import RESEARCH_SYMBOL, USE_ACCOUNT_EQUITY
from production.live.account import fetch_account, snapshot_dict, sync_equity_into_state
from production.live.signal_source import LiveMT5SignalSource
from production.logging.structured import setup_json_logging
from production.monitoring.health import HealthReporter
from production.monitoring.mt5_probe import make_mt5_probe
from production.monitoring.mt5_session import disconnect as mt5_disconnect
from production.monitoring.server import start_monitoring_server
from production.paper.broker import PaperBroker
from production.paper.pipeline import IncomingSignal, ProductionPipeline
from production.paper.runtime import IdleSignalSource, PaperRuntime, ReplaySignalSource
from production.paper.state import PortfolioState
from production.runtime_cli import load_config, start_stack, start_telegram_commands
from settings.paths import MT5_CONFIG, PAPER, ROOT
from settings.strategy import STARTING_EQUITY


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Paper trading (testing schema, never MT5 execute)")
    p.add_argument("--config", type=Path, default=MT5_CONFIG)
    p.add_argument("--mode", choices=("replay", "loop", "paper"), default="replay")
    p.add_argument("--max-signals", type=int, default=50)
    p.add_argument("--heat-trades", type=Path, default=None)
    p.add_argument("--apply-schema", action="store_true")
    return p


def _resolve(path: str) -> Path:
    p = Path(path)
    return p if p.is_absolute() else (ROOT / p).resolve()


def _load_replay_signals(heat_path: Path, *, limit: int | None) -> list[IncomingSignal]:
    import pandas as pd

    df = pd.read_parquet(heat_path)
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


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    cfg = load_config(Path(args.config))
    setup_json_logging()
    log = logging.getLogger(__name__)

    paper_cfg = dict(cfg.get("paper_trading") or {})
    starting = float(paper_cfg.get("starting_equity", STARTING_EQUITY))
    poll = float(paper_cfg.get("poll_seconds", 5.0))
    mon_host = str(paper_cfg.get("monitoring_host", "127.0.0.1"))
    mon_port = int(paper_cfg.get("monitoring_port", 8787))
    bus_workers = int(paper_cfg.get("event_workers", 2))
    tg_cfg = dict(paper_cfg.get("telegram") or {})
    db_dsn = paper_cfg.get("postgres_dsn")

    bus, metrics, db, telegram, *_ = start_stack(
        dsn=str(db_dsn) if db_dsn else None,
        schema="testing",
        schema_sql_name="testing_schema.sql",
        apply_schema=bool(args.apply_schema),
        tg_cfg=tg_cfg,
        bus_workers=bus_workers,
        queue_size=int(paper_cfg.get("event_queue_size", 10_000)),
    )

    state = PortfolioState(equity=starting, peak_equity=starting)
    env_name = "paper"
    trade_symbol = str(cfg.get("symbol", RESEARCH_SYMBOL))
    broker = PaperBroker()
    pipeline = ProductionPipeline(
        bus=bus,
        state=state,
        broker=broker,
        symbol=trade_symbol,
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
            heat_root = _resolve(
                str((cfg.get("portfolio_heat") or {}).get("output_directory", "artifacts/research/portfolio_heat"))
            )
            heat_path = Path(args.heat_trades) if args.heat_trades else heat_root / "best_policy_trades.parquet"
            if not heat_path.is_file():
                raise FileNotFoundError(f"missing {heat_path}")
            signals = _load_replay_signals(heat_path, limit=args.max_signals)
            source = ReplaySignalSource(signals)
            while not source.exhausted:
                for sig in source.next_signals():
                    pipeline.process_signal(sig)
            now = datetime.now(timezone.utc)
            for tid, pos in list(state.open_positions.items()):
                pipeline.on_bar(high=pos.entry_price, low=pos.entry_price, close=pos.entry_price, timestamp=now)
            pipeline.emit_daily_summary()
            time.sleep(0.5)
            snap = metrics.snapshot()
            (PAPER / "last_replay_metrics.json").write_text(
                __import__("json").dumps({"equity": state.equity, "metrics": snap}, indent=2, default=str),
                encoding="utf-8",
            )
            print(f"replay done equity={state.equity:.2f} opened_keys={state.trades_today} skips={state.skips}")
            print(f"monitoring was on http://{mon_host}:{mon_port}/metrics")
        elif args.mode == "paper":
            live_cfg = dict(paper_cfg.get("live") or paper_cfg.get("paper") or {})
            source = LiveMT5SignalSource(
                cfg,
                symbol=trade_symbol,
                timeframe=str(live_cfg.get("timeframe", cfg.get("timeframe", "H1"))),
                history_bars=int(live_cfg.get("history_bars", 400)),
                model_symbol=trade_symbol,
            )
            source.connect()
            if USE_ACCOUNT_EQUITY:
                try:
                    snap = fetch_account()
                    sync_equity_into_state(state, snap)
                    (PAPER / "last_account_snapshot.json").write_text(
                        __import__("json").dumps(snapshot_dict(snap), indent=2),
                        encoding="utf-8",
                    )
                    print(
                        f"account login={snap.login} server={snap.server} "
                        f"equity=${snap.equity:.2f} balance=${snap.balance:.2f}"
                    )
                except Exception as exc:
                    log.exception("account_sync_failed err=%s — using starting_equity=%.2f", exc, starting)

            runtime = PaperRuntime(
                pipeline=pipeline,
                bus=bus,
                source=source,
                poll_seconds=float(live_cfg.get("poll_seconds", poll)),
            )
            print(f"PAPER on {trade_symbol} — MT5 candles + paper fills (schema=testing)")
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
        else:
            runtime = PaperRuntime(
                pipeline=pipeline,
                bus=bus,
                source=IdleSignalSource(),
                poll_seconds=poll,
            )
            print(f"paper loop listening; monitoring http://{mon_host}:{mon_port}/health")
            cmd_listener = start_telegram_commands(
                tg_cfg,
                cfg=cfg,
                state=state,
                db=db,
                symbol=trade_symbol,
                timeframe=str(cfg.get("timeframe", "H1")),
                environment=env_name,
                verify_ssl=bool(tg_cfg.get("verify_ssl", True)),
                use_mt5_account=False,
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
