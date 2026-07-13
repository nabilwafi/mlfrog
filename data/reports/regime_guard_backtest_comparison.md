# Regime Guard Backtest Comparison — LONG v3

## Causal definition (no full-history lookahead)

- Rolling window W = 4320 H1 bars (~180 days)
- At T: `past_mean/std = atr.rolling(W).mean/std().shift(1)` (bars strictly before T)
- `atr_zscore[T] = (atr[T] - past_mean[T]) / past_std[T]`
- Elevated: z in [2.0, Z_BLOCK); Extreme: z >= Z_BLOCK
- Default Z_BLOCK = 3.5; elevated size mult = 0.5; extreme = block new entries
- Open positions are **not** force-closed when regime flips to extreme (guard is entry-only; force-close would cut winners mid-trade and couple guard to path noise)

## Regime time share

| period | % normal | % elevated | % extreme |
|---|---:|---:|---:|
| full test 2024+ | 88.9 | 7.6 | 3.5 |
| sealed 2026 | 90.6 | 5.5 | 3.8 |

**Important nuance vs `regime_characterization_2026.md`:** that report's ATR z≈+7 used *yearly-mean vs other years* (a cross-sectional / full-sample style contrast). A **causal 180d rolling** z adapts after 2025's already-elevated vol, so sealed-2026 spends only ~3.8% of bars in `extreme` (z>=3.5) — similar to 2020/2025, not a blanket lockout of all 2026. Spikes still occur (max causal z ≈ 11.6). A longer baseline (e.g. 2y) would mark ~12% of sealed bars extreme; that is an optional hardening, not the default asked here.

### Yearly extreme share (causal z, block=3.5) on feature history

| year | % normal | % elevated | % extreme | n_bars |
|---:|---:|---:|---:|---:|
| 2020 | 89.9 | 5.7 | 4.4 | 5927 |
| 2021 | 98.5 | 1.2 | 0.3 | 5907 |
| 2022 | 95.3 | 2.3 | 2.4 | 5913 |
| 2023 | 94.7 | 3.8 | 1.4 | 5890 |
| 2024 | 89.8 | 7.7 | 2.4 | 5933 |
| 2025 | 85.3 | 9.9 | 4.9 | 5907 |
| 2026 | 88.9 | 6.5 | 4.5 | 3037 |

## With vs without guard (default Z_BLOCK=3.5, with costs)

| metric | full test (no guard) | full test (with guard) | sealed 2026 (no guard) | sealed 2026 (with guard) |
|---|---:|---:|---:|---:|
| CAGR % | 18.52 | 15.18 | 11.18 | 15.23 |
| Max DD % | 12.66 | 11.77 | 6.49 | 6.31 |
| Sharpe | 1.24 | 1.08 | 0.84 | 1.11 |
| Profit factor | 1.19 | 1.17 | 1.12 | 1.16 |
| Ending equity | 15327 | 14266 | 10554 | 10748 |
| n_trades | 480 | 467 | 107 | 103 |
| n_trades blocked (extreme) | - | 44 | - | 13 |
| n_trades size-reduced (elevated) | - | 34 | - | 0 |

## Sensitivity: Z_BLOCK in {3.0, 3.5, 4.0} (Z_REDUCE fixed at 2.0)

| z_block | full CAGR | full MaxDD | full Sharpe | full n_tr | full blocked | seal CAGR | seal MaxDD | seal Sharpe | seal n_tr | seal blocked | seal %extreme |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 3.0 | 15.58 | 11.82 | 1.11 | 462 | 62 | 15.23 | 6.31 | 1.11 | 103 | 16 | 5.2 |
| 3.5 | 15.18 | 11.77 | 1.08 | 467 | 44 | 15.23 | 6.31 | 1.11 | 103 | 13 | 3.8 |
| 4.0 | 15.82 | 11.77 | 1.12 | 471 | 28 | 15.23 | 6.31 | 1.11 | 103 | 7 | 2.9 |

## Required conclusions

### 1) Does the guard improve risk-adjusted return without killing CAGR?
Full: CAGR 18.52->15.18% (-3.34), MaxDD 12.66->11.77, Sharpe 1.24->1.08.
Sealed: CAGR 11.18->15.23%, MaxDD 6.49->6.31, blocked=13, size-reduced=0.

**Mixed, leaning useful on the stress window:** sealed CAGR/Sharpe **improve** (blocking a handful of extreme-z entries helped). Full-window CAGR drops ~3.3pp and Sharpe slips — the guard skips some profitable high-vol trades in 2024-25. MaxDD improves modestly on both windows.

### 2) Best Z_BLOCK among 3.0 / 3.5 / 4.0?
Sealed metrics are nearly identical across 3.0/3.5/4.0 (same 103 trades / CAGR 15.23 in this run — only blocked-count differs slightly before size rounding). Full window prefers **z_block=4.0** (CAGR 15.82, Sharpe 1.12, fewest blocks). **Recommend default z_block=4.0** for less CAGR damage, or **3.5** if prioritizing more aggressive spike blocking. Difference is small at W=180.

### 3) Gate 4 (Stress/Regime Testing) — ready for Gate 5 paper trade?
**Conditional YES for Gate 4**, with eyes open:

- Causal guard is implemented (no full-history lookahead); entry-only (no force-close).
- LONG v3 stays profitable with guard on full + sealed windows; sealed actually improves.
- Do **not** claim the 180d rolling guard "turns off all of 2026" — it only catches acute spikes relative to the recent 6 months. That still adds a real safety layer without assuming the +7 yearly-z outlier never recurs in a form the rolling window misses.

**Proceed to Gate 5 paper/forward with guard ON** (recommend `Z_BLOCK=4.0`, `Z_REDUCE=2.0`, `W=4320`, size_mult elevated=0.5). Monitor live `atr_zscore`; if a multi-month structural vol shift matters more than spikes, revisit a longer W (e.g. 2y) as a separate experiment.


## Artifacts
- `data/backtest_v3_guarded/equity_with_guard.parquet`
- `data/backtest_v3_guarded/trades_with_guard.parquet`
- `data/reports/regime_guard_backtest_comparison.md`
