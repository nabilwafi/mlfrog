# FEAT7 vs H1-Native FE — M15 Pullback Stack

**Verdict:** `H1_NATIVE`

Shared stack: WF top 21% → **M15 pullback entry** → P0 **M15 trail** exit.

Production: `prod_v5_feat7_top21_a025_d008_lot01_daily80_m15trail`

## Feature sets

- **FEAT7** (7): `hour_cos, atr_percentile_252, ema_trend_duration, rolling_quantile, hour_sin, atr_percent, ctx_h4_swing_quality`
- **H1 native** (6): `hour_cos, atr_percentile_252, ema_trend_duration, rolling_quantile, hour_sin, atr_percent`
- Diff: FEAT7 adds `ctx_h4_swing_quality` (raw H4 merge)

## OOS @ cost 1.0×

| policy | features | entry | pf | dd | avg_r | trades | retention | mc_prob_ruin |
|---|---|---|---|---|---|---|---|---|
| feat7_immediate | feat7 | h1 close | 1.33 | 0.538 | 0.059 | 5335 | 1.000 | 0.003 |
| feat7_m15_pullback | feat7 | m15 pullback | 1.44 | 0.540 | 0.070 | 4928 | 0.973 | 0.001 |
| h1_native_immediate | h1_native | h1 close | 1.40 | 0.302 | 0.061 | 5358 | 1.000 | 0.002 |
| h1_native_m15_pullback | h1_native | m15 pullback | 1.60 | 0.271 | 0.077 | 4988 | 0.973 | 0.000 |

## Head-to-head (M15 pullback)

| | FEAT7 | H1 native | Δ |
|---|---:|---:|---:|
| PF | 1.44 | 1.60 | +0.16 |
| Max DD | 54.0% | 27.1% | -27.0pp |
| Avg R | 0.070 | 0.077 | +0.007 |
| Trades | 4928 | 4988 | +60 |

## Immediate entry (reference)

- FEAT7 immediate: PF 1.33, DD 53.8%
- H1 native immediate: PF 1.40, DD 30.2%

## Cost stress (M15 pullback)

| cost_mult | pf | dd | trades |
|---|---|---|---|
| 1.00 | 1.44 | 0.540 | 4928 |
| 1.25 | 1.39 | 0.556 | 4928 |
| 1.50 | 1.34 | 0.571 | 4928 |
| 2.00 | 1.24 | 0.832 | 4919 |

| cost_mult | pf | dd | trades |
|---|---|---|---|
| 1.00 | 1.60 | 0.271 | 4988 |
| 1.25 | 1.54 | 0.341 | 4988 |
| 1.50 | 1.49 | 0.430 | 4988 |
| 2.00 | 1.39 | 0.818 | 4981 |

Production unchanged.
