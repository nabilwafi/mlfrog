-- Sprint 27 production schema (PostgreSQL)
-- Grafana-ready normalized tables. No CSV.

CREATE SCHEMA IF NOT EXISTS trading;

CREATE TABLE IF NOT EXISTS trading.signals (
    signal_id           TEXT PRIMARY KEY,
    correlation_id      TEXT NOT NULL,
    timestamp           TIMESTAMPTZ NOT NULL,
    symbol              TEXT NOT NULL,
    side                TEXT NOT NULL,
    probability         DOUBLE PRECISION,
    meta_probability    DOUBLE PRECISION,
    confidence          DOUBLE PRECISION,
    threshold_meta      DOUBLE PRECISION,
    threshold_confidence DOUBLE PRECISION,
    model_version       TEXT,
    meta_version        TEXT,
    feature_version     TEXT,
    label_version       TEXT,
    pipeline_version    TEXT,
    accepted            BOOLEAN NOT NULL DEFAULT FALSE,
    session             TEXT,
    regime              TEXT,
    created_at          TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_signals_ts ON trading.signals (timestamp);
CREATE INDEX IF NOT EXISTS idx_signals_accepted ON trading.signals (accepted);

CREATE TABLE IF NOT EXISTS trading.trades (
    trade_id            TEXT PRIMARY KEY,
    signal_id           TEXT REFERENCES trading.signals(signal_id),
    correlation_id      TEXT NOT NULL,
    symbol              TEXT NOT NULL,
    side                TEXT NOT NULL,
    entry_time          TIMESTAMPTZ NOT NULL,
    exit_time           TIMESTAMPTZ,
    entry_price         DOUBLE PRECISION NOT NULL,
    exit_price          DOUBLE PRECISION,
    stop_loss           DOUBLE PRECISION NOT NULL,
    take_profit         DOUBLE PRECISION NOT NULL,
    lot                 DOUBLE PRECISION NOT NULL,
    risk_pct            DOUBLE PRECISION NOT NULL,
    pnl                 DOUBLE PRECISION,
    pnl_r               DOUBLE PRECISION,
    duration_seconds    INTEGER,
    mae                 DOUBLE PRECISION,
    mfe                 DOUBLE PRECISION,
    exit_reason         TEXT,
    status              TEXT NOT NULL DEFAULT 'open',
    ticket_id           BIGINT,  -- MT5 position ticket (live); used for restart recovery
    session             TEXT,
    regime              TEXT,
    probability         DOUBLE PRECISION,
    meta_probability    DOUBLE PRECISION,
    confidence          DOUBLE PRECISION,
    created_at          TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    updated_at          TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

-- Idempotent for DBs created before ticket_id existed
ALTER TABLE trading.trades ADD COLUMN IF NOT EXISTS ticket_id BIGINT;

CREATE INDEX IF NOT EXISTS idx_trades_status ON trading.trades (status);
CREATE INDEX IF NOT EXISTS idx_trades_entry ON trading.trades (entry_time);
CREATE INDEX IF NOT EXISTS idx_trades_ticket ON trading.trades (ticket_id) WHERE ticket_id IS NOT NULL;
CREATE INDEX IF NOT EXISTS idx_trades_open_symbol ON trading.trades (symbol) WHERE status = 'open';

CREATE TABLE IF NOT EXISTS trading.skip_logs (
    id                  BIGSERIAL PRIMARY KEY,
    correlation_id      TEXT NOT NULL,
    signal_id           TEXT,
    timestamp           TIMESTAMPTZ NOT NULL,
    symbol              TEXT NOT NULL,
    reason              TEXT NOT NULL,
    threshold           DOUBLE PRECISION,
    current_value       DOUBLE PRECISION,
    detail              JSONB,
    created_at          TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_skip_ts ON trading.skip_logs (timestamp);
CREATE INDEX IF NOT EXISTS idx_skip_reason ON trading.skip_logs (reason);

CREATE TABLE IF NOT EXISTS trading.execution_logs (
    id                  BIGSERIAL PRIMARY KEY,
    correlation_id      TEXT NOT NULL,
    trade_id            TEXT,
    timestamp           TIMESTAMPTZ NOT NULL,
    latency_ms          DOUBLE PRECISION,
    broker_response     TEXT,
    spread              DOUBLE PRECISION,
    slippage            DOUBLE PRECISION,
    retry_count         INTEGER DEFAULT 0,
    success             BOOLEAN NOT NULL DEFAULT TRUE,
    error_message       TEXT,
    created_at          TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_exec_ts ON trading.execution_logs (timestamp);

CREATE TABLE IF NOT EXISTS trading.daily_statistics (
    date                DATE PRIMARY KEY,
    equity              DOUBLE PRECISION NOT NULL,
    daily_r             DOUBLE PRECISION,
    drawdown            DOUBLE PRECISION,
    heat_triggered      INTEGER DEFAULT 0,
    trades              INTEGER DEFAULT 0,
    wins                INTEGER DEFAULT 0,
    winrate             DOUBLE PRECISION,
    pnl                 DOUBLE PRECISION,
    skipped             INTEGER DEFAULT 0,
    meta_rejects        INTEGER DEFAULT 0,
    confidence_rejects  INTEGER DEFAULT 0,
    updated_at          TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

CREATE TABLE IF NOT EXISTS trading.metrics (
    id                  BIGSERIAL PRIMARY KEY,
    timestamp           TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    name                TEXT NOT NULL,
    value               DOUBLE PRECISION NOT NULL,
    labels              JSONB,
    correlation_id      TEXT
);

CREATE INDEX IF NOT EXISTS idx_metrics_name_ts ON trading.metrics (name, timestamp);

CREATE TABLE IF NOT EXISTS trading.audit_logs (
    id                  BIGSERIAL PRIMARY KEY,
    timestamp           TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    correlation_id      TEXT,
    component           TEXT NOT NULL,
    action              TEXT NOT NULL,
    detail              JSONB
);

CREATE INDEX IF NOT EXISTS idx_audit_ts ON trading.audit_logs (timestamp);

-- Live / replay OHLCV for evaluation (Grafana: price series, feature drift)
CREATE TABLE IF NOT EXISTS trading.candles (
    symbol              TEXT NOT NULL,
    timeframe           TEXT NOT NULL,
    timestamp           TIMESTAMPTZ NOT NULL,
    open                DOUBLE PRECISION NOT NULL,
    high                DOUBLE PRECISION NOT NULL,
    low                 DOUBLE PRECISION NOT NULL,
    close               DOUBLE PRECISION NOT NULL,
    tick_volume         DOUBLE PRECISION,
    spread              DOUBLE PRECISION,
    real_volume         DOUBLE PRECISION,
    source              TEXT NOT NULL DEFAULT 'mt5_live',
    features            JSONB,
    created_at          TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    PRIMARY KEY (symbol, timeframe, timestamp)
);

CREATE INDEX IF NOT EXISTS idx_candles_ts ON trading.candles (timestamp);
CREATE INDEX IF NOT EXISTS idx_candles_symbol_tf ON trading.candles (symbol, timeframe);
