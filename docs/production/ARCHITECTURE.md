# Sprint 27 — Production Paper Trading Architecture

**Status:** paper infra + E2E L1–L6 pipeline ([PIPELINE_V1.md](PIPELINE_V1.md)).  
**Goal:** shared production path for paper/backtest — not improve alpha via retrain.

## Verdict on research freeze

| Component | Status |
|-----------|--------|
| Primary Model | FROZEN |
| Meta (≥0.45) | FROZEN (gate ON) |
| Confidence (skip40) | **DISABLED** (`CONFIDENCE_ENABLED=False`) |
| Portfolio Heat (daily_loss −1R vs RISK_BASE) | ON |
| TP/SL (2.0 / 1.5 ATR) | FROZEN |
| Risk sizing | **option C** `expected_r` → `risk_pct` |
| ATRE / Position Mgmt extras | NOT in production |

## Architecture diagram

```
┌─────────────────────────────────────────────────────────────────┐
│                     apps/run_paper_trading.py                    │
│                        (composition root)                        │
└───────────────────────────────┬─────────────────────────────────┘
                                │
        ┌───────────────────────▼───────────────────────┐
        │         TradingPipeline (pipeline/)            │
        │  L2 State → L4 Meta → L5 Risk → L6 Exec        │
        │              → PaperBroker fills                 │
        │              → EventBus.publish (non-blocking) │
        └───────────────────────┬───────────────────────┘
                                │ fire-and-forget
                ┌───────────────▼───────────────┐
                │     EventBus (in-process Q)    │
                └───────────────┬───────────────┘
        ┌───────────┬───────────┼───────────┬───────────┐
        ▼           ▼           ▼           ▼           ▼
   DB Writer   Telegram    Metrics     Audit      Daily Report
   (Postgres)   Sender    Collector   Logger      Generator
```

**Rule:** Only broker execution is synchronous. DB / Telegram / metrics / audit / daily report MUST NOT block fills.

See [PIPELINE_V1.md](PIPELINE_V1.md) for L1–L6 details and option C formulas.

## Folder structure

```
pipeline/                  # L1–L6 + orchestrator + scored-panel backtest
production/
  __init__.py              # re-exports pipeline constants
  events/
    types.py               # EventType, ProductionEvent, correlation IDs
    bus.py                 # async queue + MetricsCollector
  paper/
    state.py               # equity, heat, open positions, idempotency
    broker.py              # paper fills (no MT5 order_send)
    pipeline.py            # thin adapter → TradingPipeline
    runtime.py             # loop / replay / graceful shutdown
  db/
    writer.py              # Postgres upserts (worker-side)
  workers/
    handlers.py            # subscribe handlers for all event types
  telegram/
    bot.py                 # Bot API + message formatters
  monitoring/
    server.py              # /health /metrics /portfolio
  logging/
    structured.py          # JSON logs

sql/production_schema.sql  # Grafana-ready schema
docs/production/
  ARCHITECTURE.md          # this file
  PIPELINE_V1.md
  DEPLOYMENT.md
apps/run_paper_trading.py
tests/test_paper_production.py
tests/test_pipeline_v1.py
```

## Async event queue

- Implementation: `queue.Queue` + daemon worker threads (`EventBus`)
- Publish is `put_nowait` — never blocks the hot path (drops + counts if full)
- Handler exceptions are caught and logged; they never crash execution
- Graceful shutdown: sentinel `None` + join with timeout

## Correlation / idempotency

- Every event carries `correlation_id` (UUID hex)
- `trade_id` = deterministic UUID5 from symbol/side/bar_key (idempotent opens)
- Duplicate bar keys → `skip_reason=duplicate`

## Monitoring endpoints

| Path | Purpose |
|------|---------|
| `GET /health` | liveness |
| `GET /metrics` | counters + latency percentiles |
| `GET /portfolio` | equity, DD, open trades, heat |

Default: `http://127.0.0.1:8787`

## Telegram notifications

| Event | Content |
|-------|---------|
| New trade / closed / skipped | forum topic `trade_thread_id` |
| Execution errors | forum topic `error_thread_id` |
| Daily | forum topic `daily_message_thread_id` — trades, W/L, skips, PnL %, uptime |
| Health | forum topic `health_message_thread_id` — status, env, MT5, uptime |

## Paper vs live

- `apps/run_paper_trading.py` — schema `testing`, `PaperBroker` only (never MT5 `order_send`)
- `apps/run_live_trading.py` — schema `production`, `LiveBroker` (`--execute` for real `order_send`)

MT5 closed H1 bar → feature build → frozen Primary/Meta → fill.

```
MT5 copy_rates (H1/H4/D1/M5)
  → LiveFeatureBuilder (L1)
  → FrozenStackInference (L3/L4 scores)
  → IncomingSignal
  → ProductionPipeline (Meta gate + sizing + broker)
```

First run exports `artifacts/models/frozen/primary_{side}.txt` from historical parquet if missing.

## PostgreSQL schemas

Lean runtime tables (PK = `ticket_id`):

- `sql/testing_schema.sql` → `testing.trades` / `testing.history_trades` / `testing.candles`
- `sql/production_schema.sql` → `production.trades` / `production.history_trades` / `production.candles`
- `sql/research_schema.sql` → `research.wf_*` / `research.fs_*` / `research.exit_grid_*`

`/summary` command: balance & equity from MT5 `account_info`; total PnL / W/L from `history_trades`.

Designed for Grafana: candles, open trades, closed history (PnL).

## Reliability

- Broker retries (paper local)
- Idempotent trade IDs
- Duplicate signal prevention
- Worker isolation
- Graceful SIGINT/SIGTERM shutdown
- Structured JSON logs with correlation IDs
