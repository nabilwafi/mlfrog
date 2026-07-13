# Backtest Report - LONG LightGBM v2 @ thr=0.54

## Unit / cost assumptions (same engine as v1)
- Method A: `(exit_fill - entry_fill) * 100.0 * lots`
- point=0.01, contract=100.0
- Slippage 5 pts; fallback spread 19 pts

## Simulation rules (unchanged)
- Threshold: 0.54 (majority walk-forward)
- Barriers: SL=1.5*ATR, TP=2.0*ATR; horizon=8; SL-first on ambiguity
- No stacking; 1% risk; stop if equity < 70% start
- Sealed test window: preds from walk-forward v2 (typically 2026+)

## Sanity vs model eval
- Model-eval thresholded signals: n=175, precision=0.32571428571428573
- Backtest (with cost): n_trades=56, winrate_tp_sl=0.4038
- Signals raw=175, skipped_stacking=114

## WITH costs
- apply_costs: True
- starting_equity: 10000.00
- ending_equity: 9590.79
- total_return_pct: -4.09%
- CAGR_pct: -7.82% (years=0.513)
- max_drawdown_pct: 7.32%
- max_drawdown_duration_events: 43
- sharpe_annualized_approx: -0.8188 (rf=0 assumed)
- profit_factor: 0.8414
- gross_profit: 2170.84
- gross_loss: 2580.05
- n_trades: 56
- n_tp / n_sl / n_timeout: 21 / 31 / 4
- winrate_tp_sl: 0.4038
- avg_bars_held: 4.268
- n_signals_raw (incl. skipped): 175
- n_signals_skipped_stacking: 114
- n_fallback_spread_uses: 0
- blown_account: False (None)

## WITHOUT costs
- apply_costs: False
- starting_equity: 10000.00
- ending_equity: 9646.67
- total_return_pct: -3.53%
- CAGR_pct: -6.77% (years=0.513)
- max_drawdown_pct: 6.96%
- max_drawdown_duration_events: 43
- sharpe_annualized_approx: -0.7007 (rf=0 assumed)
- profit_factor: 0.8612
- gross_profit: 2191.96
- gross_loss: 2545.29
- n_trades: 56
- n_tp / n_sl / n_timeout: 21 / 31 / 4
- winrate_tp_sl: 0.4038
- avg_bars_held: 4.268
- n_signals_raw (incl. skipped): 175
- n_signals_skipped_stacking: 114
- n_fallback_spread_uses: 0
- blown_account: False (None)

## Cost impact
- End equity with/without: 9590.79 / 9646.67
- CAGR with/without: -7.82% / -6.77%
- MaxDD with/without: 7.32% / 6.96%

## Artifacts
- `data/backtest_v2/` (equity/trades parquet + png)
- `data/reports/backtest_report_v2.md`
