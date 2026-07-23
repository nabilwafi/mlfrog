-- Reset / seed paper daily_statistics baseline to $80
-- Run once before starting paper live with fixed 0.01 lot.

BEGIN;

-- Optional: wipe old daily rows (comment out if you want to keep history)
-- TRUNCATE trading.daily_statistics;

INSERT INTO trading.daily_statistics (
    date,
    equity,
    daily_r,
    drawdown,
    heat_triggered,
    trades,
    wins,
    winrate,
    pnl,
    skipped,
    meta_rejects,
    confidence_rejects,
    updated_at
) VALUES (
    CURRENT_DATE,
    80.0,
    0.0,
    0.0,
    0,
    0,
    0,
    0.0,
    0.0,
    0,
    0,
    0,
    NOW()
)
ON CONFLICT (date) DO UPDATE SET
    equity = 80.0,
    daily_r = 0.0,
    drawdown = 0.0,
    heat_triggered = 0,
    trades = 0,
    wins = 0,
    winrate = 0.0,
    pnl = 0.0,
    skipped = 0,
    meta_rejects = 0,
    confidence_rejects = 0,
    updated_at = NOW();

COMMIT;

-- Verify
SELECT date, equity, trades, wins, pnl, drawdown, updated_at
FROM trading.daily_statistics
ORDER BY date DESC
LIMIT 5;
