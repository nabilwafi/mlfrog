# H4 → H1 → M5 Hierarchical Research — Final Report

**Verdict:** `ITERATE`

Research only. H4 & M5 deterministic. H1 ML only. Production unchanged.

Gate: top 21%. Exit: P0 M15 trail. Portfolio: $280.0/year isolated.

## vs Production (FEAT7 + raw ctx_h4_swing_quality)

| track | pf | dd | avg_r | wr | trades |
|---|---|---|---|---|---|
| production_feat7 | 1.33 | 0.538 | 0.059 | 0.802 | 5335 |
| h1_baseline | 1.40 | 0.302 | 0.061 | 0.806 | 5358 |
| h1_baseline_m5_best | 1.56 | 0.301 | 0.108 | 0.772 | 5103 |
| h4_h1 | 1.37 | 0.577 | 0.066 | 0.805 | 5334 |
| h4_h1_m5_best | 1.58 | 0.452 | 0.119 | 0.776 | 5080 |

## Hierarchical ablation (OOS 2022–26)

| track | pf | dd | avg_r | wr | ret | trades |
|---|---|---|---|---|---|---|
| h1_baseline | 1.40 | 0.302 | 0.061 | 0.806 | 3.22 | 5358 |
| h4_h1 | 1.37 | 0.577 | 0.066 | 0.805 | 2.97 | 5334 |
| h4_h1_m5_best | 1.58 | 0.452 | 0.119 | 0.776 | 4.98 | 5080 |
| h1_baseline_m5_best | 1.56 | 0.301 | 0.108 | 0.772 | 4.92 | 5103 |

Best M5 on H4→H1: **pullback_recovery** (retention 97.7%)
Best M5 on H1 baseline: **pullback_recovery** (retention 97.5%)

## Production stack reference

- Pipeline: `prod_v5_feat7_top21_a025_d008_lot01_daily80_m15trail`
- Features: `hour_cos, atr_percentile_252, ema_trend_duration, rolling_quantile, hour_sin, atr_percent, ctx_h4_swing_quality`
- Entry H1 / trail M15 / lot 0.01 / heat 3R

Artifacts: `results/h4_h1_m5/`

