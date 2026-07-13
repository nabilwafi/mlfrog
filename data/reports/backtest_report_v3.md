# Backtest Report - LONG LightGBM v3 @ thr=0.51

## Unit / cost (same engine as v1/v2)
- Method A: `(exit_fill - entry_fill) * 100.0 * lots`
- point=0.01, slip=5, fallback spread=19

## Notes on prediction windows
- **Full test (2024+)**: one LGBM fit on v3 features with train<2023 / val 2023 (same calendar philosophy as v1); threshold **frozen** at WF majority 0.51.
- **Sealed 2026**: preds from walk-forward final model (`lightgbm_long_v3_test_preds.parquet`): 2026-01-02 04:00:00+00:00 -> 2026-07-08 11:00:00+00:00.

## FULL test (2024-01 -> data end) WITH costs
- Model-eval signals @thr: n=1495, precision=0.49230769230769234
- apply_costs: True
- starting_equity: 10000.00
- ending_equity: 15327.00
- total_return_pct: 53.27%
- CAGR_pct: 18.52% (years=2.513)
- max_drawdown_pct: 12.66%
- max_drawdown_duration_events: 93
- sharpe_annualized_approx: 1.2409 (rf=0 assumed)
- profit_factor: 1.1893
- gross_profit: 33468.17
- gross_loss: 28141.17
- n_trades: 480
- n_tp / n_sl / n_timeout: 215 / 220 / 45
- winrate_tp_sl: 0.4943
- avg_bars_held: 4.446
- n_signals_raw (incl. skipped): 1495
- n_signals_skipped_stacking: 1015
- n_fallback_spread_uses: 0
- blown_account: False (None)

## FULL test WITHOUT costs
- apply_costs: False
- starting_equity: 10000.00
- ending_equity: 18834.14
- total_return_pct: 88.34%
- CAGR_pct: 28.65% (years=2.513)
- max_drawdown_pct: 11.15%
- max_drawdown_duration_events: 93
- sharpe_annualized_approx: 1.7883 (rf=0 assumed)
- profit_factor: 1.2815
- gross_profit: 40216.01
- gross_loss: 31381.87
- n_trades: 480
- n_tp / n_sl / n_timeout: 215 / 220 / 45
- winrate_tp_sl: 0.4943
- avg_bars_held: 4.446
- n_signals_raw (incl. skipped): 1495
- n_signals_skipped_stacking: 1015
- n_fallback_spread_uses: 0
- blown_account: False (None)

## SEALED 2026 WITH costs
- Model-eval signals @thr: n=368, precision=0.44565217391304346
- apply_costs: True
- starting_equity: 10000.00
- ending_equity: 10554.02
- total_return_pct: 5.54%
- CAGR_pct: 11.18% (years=0.509)
- max_drawdown_pct: 6.49%
- max_drawdown_duration_events: 57
- sharpe_annualized_approx: 0.8385 (rf=0 assumed)
- profit_factor: 1.1169
- gross_profit: 5294.03
- gross_loss: 4740.01
- n_trades: 107
- n_tp / n_sl / n_timeout: 47 / 53 / 7
- winrate_tp_sl: 0.4700
- avg_bars_held: 4.047
- n_signals_raw (incl. skipped): 368
- n_signals_skipped_stacking: 248
- n_fallback_spread_uses: 0
- blown_account: False (None)

## SEALED 2026 WITHOUT costs
- apply_costs: False
- starting_equity: 10000.00
- ending_equity: 10711.93
- total_return_pct: 7.12%
- CAGR_pct: 14.47% (years=0.509)
- max_drawdown_pct: 6.33%
- max_drawdown_duration_events: 54
- sharpe_annualized_approx: 1.0469 (rf=0 assumed)
- profit_factor: 1.1513
- gross_profit: 5417.37
- gross_loss: 4705.44
- n_trades: 107
- n_tp / n_sl / n_timeout: 47 / 53 / 7
- winrate_tp_sl: 0.4700
- avg_bars_held: 4.047
- n_signals_raw (incl. skipped): 368
- n_signals_skipped_stacking: 248
- n_fallback_spread_uses: 0
- blown_account: False (None)

## Sanity
- Full: model prec=0.49230769230769234 vs backtest winrate_tp_sl=0.4943 (n_trades=480, skipped=1015)
- Sealed: model prec=0.44565217391304346 vs winrate_tp_sl=0.4700 (n_trades=107)

## Artifacts
- `data/backtest_v3/`
- `data/models_v3/long/lightgbm_long_v3_fulltest_preds.parquet`
