-- Research rolling walk-forward schema (PostgreSQL).
-- Source of truth for training + OOS backtest history. No CSV required.

CREATE SCHEMA IF NOT EXISTS research;

CREATE TABLE IF NOT EXISTS research.wf_runs (
    run_id              TEXT PRIMARY KEY,
    created_at          TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    symbol              TEXT NOT NULL DEFAULT 'XAUUSD',
    timeframe           TEXT NOT NULL DEFAULT 'H1',
    train_years         INTEGER NOT NULL,
    val_years           INTEGER NOT NULL,
    test_years          INTEGER NOT NULL,
    first_test_year     INTEGER NOT NULL,
    last_test_year      INTEGER NOT NULL,
    policy              JSONB NOT NULL,
    feature_version     TEXT,
    label_version       TEXT,
    pipeline_version    TEXT,
    status              TEXT NOT NULL DEFAULT 'running',
    notes               TEXT
);

CREATE TABLE IF NOT EXISTS research.wf_trainings (
    id                  BIGSERIAL PRIMARY KEY,
    run_id              TEXT NOT NULL REFERENCES research.wf_runs(run_id) ON DELETE CASCADE,
    window_id           TEXT NOT NULL,
    side                TEXT NOT NULL,
    train_start_year    INTEGER NOT NULL,
    train_end_year      INTEGER NOT NULL,
    val_year            INTEGER NOT NULL,
    test_year           INTEGER NOT NULL,
    model_version       TEXT NOT NULL,
    feature_version     TEXT,
    label_version       TEXT,
    n_train             INTEGER,
    n_val               INTEGER,
    n_features          INTEGER,
    best_iteration      INTEGER,
    val_logloss         DOUBLE PRECISION,
    feature_names       JSONB,
    model_path          TEXT,
    metrics             JSONB,
    created_at          TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    UNIQUE (run_id, window_id, side)
);

CREATE INDEX IF NOT EXISTS idx_wf_trainings_run ON research.wf_trainings (run_id);
CREATE INDEX IF NOT EXISTS idx_wf_trainings_test ON research.wf_trainings (test_year);

CREATE TABLE IF NOT EXISTS research.wf_backtests (
    id                  BIGSERIAL PRIMARY KEY,
    run_id              TEXT NOT NULL REFERENCES research.wf_runs(run_id) ON DELETE CASCADE,
    window_id           TEXT NOT NULL,
    test_year           INTEGER NOT NULL,
    scope               TEXT NOT NULL DEFAULT 'year',  -- year | combined
    n_candidates        INTEGER,
    n_trades            INTEGER,
    total_return        DOUBLE PRECISION,
    final_equity        DOUBLE PRECISION,
    max_drawdown        DOUBLE PRECISION,
    profit_factor       DOUBLE PRECISION,
    win_rate            DOUBLE PRECISION,
    avg_holding_bars    DOUBLE PRECISION,
    starting_equity     DOUBLE PRECISION,
    metrics             JSONB,
    created_at          TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    UNIQUE (run_id, window_id, scope)
);

CREATE INDEX IF NOT EXISTS idx_wf_backtests_run ON research.wf_backtests (run_id);
CREATE INDEX IF NOT EXISTS idx_wf_backtests_year ON research.wf_backtests (test_year);

CREATE TABLE IF NOT EXISTS research.wf_trades (
    id                  BIGSERIAL PRIMARY KEY,
    run_id              TEXT NOT NULL REFERENCES research.wf_runs(run_id) ON DELETE CASCADE,
    window_id           TEXT NOT NULL,
    test_year           INTEGER NOT NULL,
    trade_seq           INTEGER NOT NULL,
    timestamp           TIMESTAMPTZ NOT NULL,
    side                TEXT NOT NULL,
    entry_price         DOUBLE PRECISION NOT NULL,
    lots                DOUBLE PRECISION NOT NULL,
    y_prob              DOUBLE PRECISION,
    net_return          DOUBLE PRECISION,
    pnl                 DOUBLE PRECISION,
    equity              DOUBLE PRECISION,
    holding_bars        INTEGER,
    exit_reason         TEXT,
    regime              TEXT,
    trend_state         TEXT,
    vol_state           TEXT,
    mfe_pct             DOUBLE PRECISION,
    mae_pct             DOUBLE PRECISION,
    mfe_r               DOUBLE PRECISION,
    mae_r               DOUBLE PRECISION,
    r_multiple          DOUBLE PRECISION,
    created_at          TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

-- Additive columns for older research.wf_trades installs.
ALTER TABLE research.wf_trades ADD COLUMN IF NOT EXISTS regime TEXT;
ALTER TABLE research.wf_trades ADD COLUMN IF NOT EXISTS trend_state TEXT;
ALTER TABLE research.wf_trades ADD COLUMN IF NOT EXISTS vol_state TEXT;
ALTER TABLE research.wf_trades ADD COLUMN IF NOT EXISTS mfe_pct DOUBLE PRECISION;
ALTER TABLE research.wf_trades ADD COLUMN IF NOT EXISTS mae_pct DOUBLE PRECISION;
ALTER TABLE research.wf_trades ADD COLUMN IF NOT EXISTS mfe_r DOUBLE PRECISION;
ALTER TABLE research.wf_trades ADD COLUMN IF NOT EXISTS mae_r DOUBLE PRECISION;
ALTER TABLE research.wf_trades ADD COLUMN IF NOT EXISTS r_multiple DOUBLE PRECISION;

CREATE INDEX IF NOT EXISTS idx_wf_trades_run ON research.wf_trades (run_id);
CREATE INDEX IF NOT EXISTS idx_wf_trades_ts ON research.wf_trades (timestamp);
CREATE INDEX IF NOT EXISTS idx_wf_trades_year ON research.wf_trades (test_year);

CREATE TABLE IF NOT EXISTS research.wf_equity (
    id                  BIGSERIAL PRIMARY KEY,
    run_id              TEXT NOT NULL REFERENCES research.wf_runs(run_id) ON DELETE CASCADE,
    scope               TEXT NOT NULL,  -- year | combined
    test_year           INTEGER,
    timestamp           TIMESTAMPTZ NOT NULL,
    equity              DOUBLE PRECISION NOT NULL,
    pnl                 DOUBLE PRECISION,
    created_at          TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_wf_equity_run ON research.wf_equity (run_id, scope);
CREATE INDEX IF NOT EXISTS idx_wf_equity_ts ON research.wf_equity (timestamp);
