-- Live trading schema. Always real MT5 order_send.
-- Lean: candles + open trades + history. PK = ticket_id.
-- No signal_id / correlation_id.

CREATE SCHEMA IF NOT EXISTS production;

CREATE TABLE IF NOT EXISTS production.trades (
    ticket_id           BIGINT PRIMARY KEY,
    symbol              TEXT NOT NULL,
    side                TEXT NOT NULL,
    entry_time          TIMESTAMPTZ NOT NULL,
    entry_price         DOUBLE PRECISION NOT NULL,
    stop_loss           DOUBLE PRECISION NOT NULL,
    take_profit         DOUBLE PRECISION NOT NULL DEFAULT 0,
    lot                 DOUBLE PRECISION NOT NULL,
    risk_pct            DOUBLE PRECISION NOT NULL DEFAULT 0,
    session             TEXT,
    regime              TEXT,
    probability         DOUBLE PRECISION,
    meta_probability    DOUBLE PRECISION,
    confidence          DOUBLE PRECISION,
    created_at          TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    updated_at          TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_production_trades_entry ON production.trades (entry_time);
CREATE INDEX IF NOT EXISTS idx_production_trades_symbol ON production.trades (symbol);

CREATE TABLE IF NOT EXISTS production.history_trades (
    ticket_id           BIGINT PRIMARY KEY,
    symbol              TEXT NOT NULL,
    side                TEXT NOT NULL,
    entry_time          TIMESTAMPTZ NOT NULL,
    exit_time           TIMESTAMPTZ,
    entry_price         DOUBLE PRECISION NOT NULL,
    exit_price          DOUBLE PRECISION,
    stop_loss           DOUBLE PRECISION NOT NULL,
    take_profit         DOUBLE PRECISION NOT NULL DEFAULT 0,
    lot                 DOUBLE PRECISION NOT NULL,
    risk_pct            DOUBLE PRECISION NOT NULL DEFAULT 0,
    pnl                 DOUBLE PRECISION,
    pnl_r               DOUBLE PRECISION,
    duration_seconds    INTEGER,
    mae                 DOUBLE PRECISION,
    mfe                 DOUBLE PRECISION,
    exit_reason         TEXT,
    session             TEXT,
    regime              TEXT,
    probability         DOUBLE PRECISION,
    meta_probability    DOUBLE PRECISION,
    confidence          DOUBLE PRECISION,
    created_at          TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    updated_at          TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_production_history_exit ON production.history_trades (exit_time);
CREATE INDEX IF NOT EXISTS idx_production_history_entry ON production.history_trades (entry_time);
CREATE INDEX IF NOT EXISTS idx_production_history_symbol ON production.history_trades (symbol);

CREATE TABLE IF NOT EXISTS production.candles (
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

CREATE INDEX IF NOT EXISTS idx_production_candles_ts ON production.candles (timestamp);
CREATE INDEX IF NOT EXISTS idx_production_candles_symbol_tf ON production.candles (symbol, timeframe);
