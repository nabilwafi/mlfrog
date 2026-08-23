# M15 vs M5 Pullback Entry

**Winner:** `M15_PULLBACK`

Shared exit: P0 a0.25/d0.08 trail on **M15 close**, from fill bar after H1 close.
Same rule: 0.15 ATR pullback, 0.50 recovery, max wait 4 H1 bars.

Pipeline ref: `prod_v5_feat7_top21_a025_d008_lot01_daily80_m15trail`

## OOS summary

| policy | entry_tf | pf | dd | avg_r | trades | retention | mc_prob_ruin |
|---|---|---|---|---|---|---|---|
| h1_immediate_m15 | H1 | 1.40 | 0.302 | 0.061 | 5358 | 1.000 | 0.002 |
| m15_pullback_m15_exit | M15 | 1.60 | 0.271 | 0.077 | 4988 | 0.973 | 0.000 |
| m5_pullback_m15_exit | M15 | 1.53 | 0.273 | 0.076 | 4992 | 0.975 | 0.000 |

## Head-to-head

| | M15 pullback | M5 pullback | Δ |
|---|---:|---:|---:|
| PF | 1.60 | 1.53 | -0.07 |
| Max DD | 27.1% | 27.3% | +0.2pp |
| Avg R | 0.077 | 0.076 | -0.001 |
| Trades | 4988 | 4992 | +4 |
| Retention | 97.3% | 97.5% | |

## Entry delay diagnostics

- M15 pullback: {'median_delay_m15_bars': 2.0, 'p90_delay_m15_bars': 3.0, 'invalidate_rate': 0.026553837175372047, 'retention': 0.9734461628246279}
- M5 pullback: {'median_delay_m5_bars': 5.0, 'p90_delay_m5_bars': 11.0, 'invalidate_rate': 0.0249489349285089, 'retention': 0.9750510650714911}

Baseline immediate H1: PF 1.40, DD 30.2%

Production unchanged.
