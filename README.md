# XAUUSD ML Trading Platform

Train / rolling walk-forward / paper / live for XAUUSD.

## Design

- [docs/production/ARCHITECTURE.md](docs/production/ARCHITECTURE.md)
- [docs/production/DEPLOYMENT.md](docs/production/DEPLOYMENT.md)

Settled strategy constants: `settings/strategy.py`

## Setup

```bash
pip install -r requirements.txt
```

Copy `configs/config.example.yaml` → `configs/config.yaml` and fill MT5 credentials.
Blobs live under `artifacts/` (gitignored).

## Data pipeline

```bash
python apps/fetch_market_data.py --provider mt5 --symbol XAUUSD --timeframe H1 --start 2015-01-01 --end 2026-12-31
python apps/generate_features.py --symbol XAUUSD --timeframe H1
python apps/generate_labels.py --symbol XAUUSD --timeframe H1 --strategy triple_barrier --sides long short
python apps/run_feature_engineering.py --symbol XAUUSD --timeframe H1
python apps/build_dataset.py --symbol XAUUSD --timeframe H1
python apps/train_model.py --symbol XAUUSD --timeframe H1
```

## Rolling walk-forward

Train 5y / val 1y / test 1y, retrain yearly. Policy: top 5% · ATR trail 0.12 · max_open=1 · lot=0.01 · start=$80.

```bash
python apps/run_rolling_walkforward.py --apply-schema
python apps/run_rolling_walkforward.py
python apps/run_rolling_walkforward.py --debug-files
python apps/audit_rolling_walkforward.py --run-id <run_id>
```

Attribution report from stored DB results (no retrain / no re-backtest):

```bash
python apps/report_performance_attribution.py
python apps/report_edge_attribution.py
python apps/report_feature_library_research.py
```

Schema: `sql/research_schema.sql` → `research.wf_*` / `research.fs_*` / `research.exit_grid_*`  
Debug panels: `artifacts/pipeline_backtest/rolling_wf/`

Sim helpers (trail / ruin-stop portfolio): `simulation/wf/sim.py`

## Paper / live

```bash
# Paper — testing schema, simulated fills (never MT5 order_send)
python apps/run_paper_trading.py --mode replay --max-signals 50
python apps/run_paper_trading.py --mode paper --apply-schema

# Live — production schema; always real MT5 order_send
python apps/run_live_trading.py
python apps/run_live_trading.py --apply-schema
```

Schemas:
- `sql/testing_schema.sql` → rich (`signals`, `skip_logs`, `execution_logs`, `audit_logs`, …) + `trades`/`history_trades`/`candles` (PK `ticket_id`)
- `sql/production_schema.sql` → lean `production.trades` / `history_trades` / `candles` (no `signal_id`/`correlation_id`)

Monitoring: `http://<LAN-IP>:8787/health` · `/metrics` · `/portfolio`

```bash
python -m unittest tests.test_paper_production tests.test_trade_history -v
```
