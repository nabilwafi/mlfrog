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

## Live mode (`--mode live`)

MT5 closed H1 bar → feature build → frozen Primary/Meta → paper fill (no `order_send`). Confidence still scored for logging but **not** used as a gate.

```
MT5 copy_rates (H1/H4/D1/M5)
  → LiveFeatureBuilder (L1)
  → FrozenStackInference (L3/L4 scores)
  → IncomingSignal
  → TradingPipeline (Meta gate + expected_r sizing + PaperBroker)
```

First run exports `artifacts/models/frozen/primary_{side}.txt` from historical parquet if missing.

## PostgreSQL tables

See `sql/production_schema.sql`:

- `trading.signals`
- `trading.trades`
- `trading.skip_logs`
- `trading.execution_logs`
- `trading.daily_statistics`
- `trading.metrics`
- `trading.audit_logs`
- `trading.candles` — live OHLCV + feature snapshot (eval / Grafana)

Designed for Grafana panels: equity, winrate, PnL, latency, heat, skip reasons, spread/slippage.

## Reliability

- Broker retries (paper local)
- Idempotent trade IDs
- Duplicate signal prevention
- Worker isolation
- Graceful SIGINT/SIGTERM shutdown
- Structured JSON logs with correlation IDs
