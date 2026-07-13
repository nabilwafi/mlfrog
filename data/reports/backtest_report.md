# Backtest Report — LONG LightGBM v1 @ thr=0.52

## Unit / cost assumptions (verified)

From live `mt5.symbol_info('XAUUSD')` (Finex):
- digits=2, point=0.01, trade_tick_size=0.01, trade_contract_size=100.0
- trade_tick_value=10.0 (INCONSISTENT with contract*point = 1.0 USD/point/lot)
- Formula check: `tick_value/tick_size*point = 10/0.01*0.01 = 10` != Method A `1.0`
  Hypothesis that tick_size=0.1 is **FALSE** (actual tick_size=0.01). Genuine Finex quirk.
- **PnL Method A used:** `(exit_fill - entry_fill) * contract_size * lots`
- Example: 19 points spread = 0.19 USD/oz = **19.00 USD per 1.0 lot**
- Slippage 5 points = **5.00 USD per 1.0 lot per side**
- Per-bar Spread from raw H1; fallback flat 19 points if missing/invalid
- ASSUMED_SLIPPAGE_POINTS = 5

## Simulation rules
- Threshold: 0.52 (frozen from validation)
- Barriers: SL=1.5*ATR, TP=2.0*ATR from entry close (label-consistent); horizon=8 H1 bars
- Ambiguous same-candle TP+SL -> SL first (conservative, same as labeling)
- Timeout -> close at close of horizon bar (realized PnL, not forced loss)
- No stacking: skip new signals while a position is open
- Risk: 1% equity per trade; stop if equity < 70% of starting equity

## Sanity check vs model evaluation
- Model-eval thresholded signals: n=878, precision=0.5228
- Backtest (with cost): n_trades=247, winrate_tp_sl=0.5134
- Signals seen raw=878, skipped_by_no_stacking=631

### Explanation of differences (expected, not automatic bug)
- Model precision counts **all** test rows with y_prob>=0.52 (n=878).
- Backtest **skips** signals while a position is open (no stacking), so n_trades < 878.
- Win-rate compares TP vs SL exits only; timeouts are separate.
- Costs change PnL/equity but do **not** change TP/SL touch outcomes.

## WITH costs (per-bar spread + slippage)
- apply_costs: True
- starting_equity: 10000.00
- ending_equity: 13647.32
- total_return_pct: 36.47%
- CAGR_pct: 13.19% (years=2.510)
- max_drawdown_pct: 17.06%
- max_drawdown_duration_events: 117
- sharpe_annualized_approx: 1.2465 (rf=0 assumed)
- profit_factor: 1.3110
- gross_profit: 15374.39
- gross_loss: 11727.07
- n_trades: 247
- n_tp / n_sl / n_timeout: 115 / 109 / 23
- winrate_tp_sl: 0.5134
- avg_bars_held: 4.704
- n_signals_raw (incl. skipped): 878
- n_signals_skipped_stacking: 631
- n_fallback_spread_uses: 0
- blown_account: False (None)

## WITHOUT costs (spread=0, slippage=0)
- apply_costs: False
- starting_equity: 10000.00
- ending_equity: 14858.61
- total_return_pct: 48.59%
- CAGR_pct: 17.09% (years=2.510)
- max_drawdown_pct: 13.90%
- max_drawdown_duration_events: 97
- sharpe_annualized_approx: 1.5724 (rf=0 assumed)
- profit_factor: 1.4040
- gross_profit: 16884.75
- gross_loss: 12026.14
- n_trades: 247
- n_tp / n_sl / n_timeout: 115 / 109 / 23
- winrate_tp_sl: 0.5134
- avg_bars_held: 4.704
- n_signals_raw (incl. skipped): 878
- n_signals_skipped_stacking: 631
- n_fallback_spread_uses: 0
- blown_account: False (None)

## Cost impact
- Ending equity with cost: 13647.32 | without cost: 14858.61
- CAGR with cost: 13.19% | without cost: 17.09%
- Max DD with cost: 17.06% | without cost: 13.90%
- Equity eaten by costs (end): 1211.29 USD

## Artifacts
- `data/backtest/equity_with_cost.parquet`
- `data/backtest/equity_no_cost.parquet`
- `data/backtest/trades_with_cost.parquet`
- `data/backtest/trades_no_cost.parquet`
- `data/backtest/equity_with_cost.png`
- `data/backtest/equity_no_cost.png`
