-- Paper trading schema (MT5 candles OK; fills simulated — never real order_send).
-- Same lean shape as production; PK = ticket_id (synthetic paper ticket).

CREATE SCHEMA IF NOT EXISTS testing;

CREATE TABLE IF NOT EXISTS testing.trades (
    ticket_id           BIGINT PRIMARY KEY,
    signal_id           TEXT,
    correlation_id      TEXT,
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

CREATE INDEX IF NOT EXISTS idx_testing_trades_entry ON testing.trades (entry_time);
CREATE INDEX IF NOT EXISTS idx_testing_trades_symbol ON testing.trades (symbol);

CREATE TABLE IF NOT EXISTS testing.history_trades (
    ticket_id           BIGINT PRIMARY KEY,
    signal_id           TEXT,
    correlation_id      TEXT,
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

CREATE INDEX IF NOT EXISTS idx_testing_history_exit ON testing.history_trades (exit_time);
CREATE INDEX IF NOT EXISTS idx_testing_history_entry ON testing.history_trades (entry_time);
CREATE INDEX IF NOT EXISTS idx_testing_history_symbol ON testing.history_trades (symbol);

CREATE TABLE IF NOT EXISTS testing.candles (
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

CREATE INDEX IF NOT EXISTS idx_testing_candles_ts ON testing.candles (timestamp);
CREATE INDEX IF NOT EXISTS idx_testing_candles_symbol_tf ON testing.candles (symbol, timeframe);
