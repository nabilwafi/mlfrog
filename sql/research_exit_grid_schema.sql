-- Exit engine grid search (append-only; no overwrite)
CREATE SCHEMA IF NOT EXISTS research;

CREATE TABLE IF NOT EXISTS research.exit_grid_runs (
    run_id      TEXT PRIMARY KEY,
    created_at  TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    status      TEXT NOT NULL,
    n_combos    INTEGER,
    features    JSONB,
    notes       TEXT
);

CREATE TABLE IF NOT EXISTS research.exit_grid_results (
    run_id          TEXT NOT NULL REFERENCES research.exit_grid_runs(run_id) ON DELETE CASCADE,
    combo_id        TEXT NOT NULL,
    params          JSONB NOT NULL,
    n_trades        INTEGER,
    win_rate        DOUBLE PRECISION,
    profit_factor   DOUBLE PRECISION,
    total_return    DOUBLE PRECISION,
    max_drawdown    DOUBLE PRECISION,
    final_equity    DOUBLE PRECISION,
    avg_win         DOUBLE PRECISION,
    avg_loss        DOUBLE PRECISION,
    avg_r           DOUBLE PRECISION,
    expectancy_r    DOUBLE PRECISION,
    avg_mfe_r       DOUBLE PRECISION,
    avg_mae_r       DOUBLE PRECISION,
    exit_reasons    JSONB,
    yearly          JSONB,
    years_positive  INTEGER,
    ret_2026        DOUBLE PRECISION,
    robustness      DOUBLE PRECISION,
    PRIMARY KEY (run_id, combo_id)
);

CREATE INDEX IF NOT EXISTS idx_exit_grid_results_rank
    ON research.exit_grid_results (run_id, profit_factor DESC, max_drawdown ASC);
