-- Feature selection / ablation experiments (research only). Does not touch production trading tables.

CREATE SCHEMA IF NOT EXISTS research;

CREATE TABLE IF NOT EXISTS research.fs_runs (
    run_id          TEXT PRIMARY KEY,
    created_at      TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    status          TEXT NOT NULL DEFAULT 'running',
    baseline_pf     DOUBLE PRECISION,
    baseline_return DOUBLE PRECISION,
    baseline_dd     DOUBLE PRECISION,
    notes           TEXT
);

CREATE TABLE IF NOT EXISTS research.fs_experiments (
    id              BIGSERIAL PRIMARY KEY,
    run_id          TEXT NOT NULL REFERENCES research.fs_runs(run_id) ON DELETE CASCADE,
    phase           TEXT NOT NULL,  -- ranking|single_ablation|group_ablation|forward|backward|compare
    experiment_id   TEXT NOT NULL,
    features_json   JSONB NOT NULL,
    n_features      INTEGER NOT NULL,
    removed_feature TEXT,
    removed_group   TEXT,
    n_trades        INTEGER,
    win_rate        DOUBLE PRECISION,
    profit_factor   DOUBLE PRECISION,
    total_return    DOUBLE PRECISION,
    max_drawdown    DOUBLE PRECISION,
    final_equity    DOUBLE PRECISION,
    roc_auc         DOUBLE PRECISION,
    avg_pr          DOUBLE PRECISION,
    metrics         JSONB,
    created_at      TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    UNIQUE (run_id, phase, experiment_id)
);

CREATE INDEX IF NOT EXISTS idx_fs_experiments_run ON research.fs_experiments (run_id, phase);
