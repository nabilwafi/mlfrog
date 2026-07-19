# Sprint 27 — Deployment Guide

## Prerequisites

- Python 3.11+
- `pip install -r requirements.txt` (adds `psycopg2-binary` for Postgres)
- Optional: PostgreSQL 14+
- Optional: Telegram bot token + chat id
- Research artifacts present for replay validation:
  - `artifacts/research/portfolio_heat/best_policy_trades.parquet`

## Config

Copy `configs/config.example.yaml` → `configs/config.yaml` and fill:

```yaml
paper_trading:
  starting_equity: 10000
  poll_seconds: 5
  event_workers: 2
  event_queue_size: 10000
  monitoring_host: 0.0.0.0  # LAN-accessible; 127.0.0.1 = local-only
  monitoring_port: 8787
  postgres_dsn: "postgresql://user:pass@localhost:5432/xauusd"
  telegram:
    enabled: false
    bot_token: "YOUR_BOT_TOKEN"
    chat_id: "YOUR_CHAT_ID"
```

Leave `postgres_dsn` empty / null for dry-run (DB writer no-ops).  
Leave telegram `enabled: false` until credentials are set.

## Apply schema

```bash
python apps/run_paper_trading.py --apply-schema --mode replay --max-signals 1
```

Or:

```bash
psql "$POSTGRES_DSN" -f sql/production_schema.sql
```

## Validate infrastructure (replay)

Replays frozen Heat trade signals through the production pipeline (no model retrain):

```bash
python apps/run_paper_trading.py --mode replay --max-signals 50
```

Checks:

1. Meta / Confidence / Heat skips emit events
2. Accepted signals open paper trades
3. Workers drain without blocking fills
4. Metrics written under `artifacts/paper_trading/last_replay_metrics.json`
5. Monitoring endpoints respond during run

## Idle loop (paper process)

```bash
python apps/run_paper_trading.py --mode loop
```

- Serves `/health`, `/metrics`, `/portfolio`
- `/health` also notifies Telegram (throttled; default 60s)
- Heartbeat Telegram every `health_interval_seconds` (default 300)
- Polls signal source (idle by default until a live signal adapter is wired)
- Ctrl+C → daily summary + graceful bus drain

## Grafana

Point Grafana PostgreSQL datasource at the same DSN.

Suggested panels:

| Panel | Query idea |
|-------|------------|
| Equity | `SELECT date, equity FROM trading.daily_statistics ORDER BY date` |
| Winrate | `winrate` from `daily_statistics` |
| Skip reasons | `COUNT(*) GROUP BY reason` on `skip_logs` |
| Latency | `trading.metrics` where `name LIKE '%latency%'` |
| Heat | `heat_triggered` daily |
| Spread / slippage | `execution_logs` |
| Live candles | `SELECT timestamp, close FROM trading.candles WHERE symbol='XAUUSD' ORDER BY timestamp` |
| Feature drift | `features->>'atr_percent'` etc. on `trading.candles` |

## Production checklist

- [ ] Research freeze confirmed (no threshold edits)
- [ ] Schema applied
- [ ] Replay smoke green
- [ ] Telegram test message received
- [ ] `/health` returns ok **and** HEALTH message appears in Telegram
- [ ] Postgres inserts visible
- [ ] Restart process → no duplicate opens for same bar_key
- [ ] Kill -TERM drains workers cleanly

## What this sprint does NOT do

- Retrain Primary / Meta
- Change Meta 0.45 / Conf 40 / Heat −1R / TP-SL
- Place live MT5 `order_send` (paper broker only)
- Add ATRE / trailing / BE / pyramid (research rejected)
