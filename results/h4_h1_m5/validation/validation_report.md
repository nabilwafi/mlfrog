# H1 + M5 Pullback Validation

**Verdict:** `ITERATE`

Compare: Production FEAT7 immediate vs H1 baseline immediate vs H1+M5 pullback.

Production pipeline: `prod_v5_feat7_top21_a025_d008_lot01_daily80_m15trail`

## OOS summary (cost 1.0×)

| policy | pf | dd | avg_r | wr | trades | ret | mc_prob_ruin | mc_median_dd |
|---|---|---|---|---|---|---|---|---|
| production_feat7 | 1.33 | 0.538 | 0.059 | 0.802 | 5335 | 2.71 | 0.003 | 0.280 |
| h1_baseline | 1.40 | 0.302 | 0.061 | 0.806 | 5358 | 3.22 | 0.002 | 0.253 |
| h1_m5_pullback | 1.56 | 0.301 | 0.108 | 0.772 | 5103 | 4.92 | 0.001 | 0.213 |

## Yearly vs production (H1+M5 − prod ΔPF)

| year | oos | prod_pf | h1_m5_pf | delta_pf | prod_trades | h1_m5_trades |
|---|---|---|---|---|---|---|
| 2021 | False | 1.36 | 1.35 | -0.01 | 1157 | 1118 |
| 2022 | True | 1.51 | 1.46 | -0.06 | 1210 | 1159 |
| 2023 | True | 1.25 | 1.66 | 0.41 | 1143 | 1080 |
| 2024 | True | 1.10 | 1.23 | 0.13 | 1205 | 1160 |
| 2025 | True | 1.23 | 1.59 | 0.36 | 1178 | 1143 |
| 2026 | True | 1.50 | 1.74 | 0.24 | 599 | 561 |

## Cost stress (H1+M5 pullback)

| cost_mult | pf | dd | trades | avg_r |
|---|---|---|---|---|
| 1.00 | 1.56 | 0.301 | 5103 | 0.108 |
| 1.25 | 1.51 | 0.323 | 5103 | 0.097 |
| 1.50 | 1.47 | 0.345 | 5103 | 0.085 |
| 2.00 | 1.38 | 0.556 | 5103 | 0.062 |

## LONG / SHORT (OOS pooled)

| side | policy | pf | trades | avg_r |
|---|---|---|---|---|
| long | prod | 1.34 | 2017 | 0.103 |
| long | h1_imm | 1.33 | 2109 | 0.071 |
| long | h1_m5 | 1.54 | 2033 | 0.124 |
| short | prod | 1.35 | 3484 | 0.079 |
| short | h1_imm | 1.44 | 3359 | 0.071 |
| short | h1_m5 | 1.53 | 3223 | 0.114 |

## Pullback parameter grid

| pullback_atr | recovery_frac | pf | dd | retention | trades |
|---|---|---|---|---|---|
| 0.10 | 0.40 | 1.71 | 0.209 | 0.977 | 5135 |
| 0.10 | 0.50 | 1.54 | 0.285 | 0.977 | 5103 |
| 0.15 | 0.50 | 1.56 | 0.301 | 0.975 | 5103 |
| 0.15 | 0.60 | 1.42 | 0.330 | 0.974 | 5092 |
| 0.20 | 0.50 | 1.56 | 0.306 | 0.973 | 5094 |
| 0.25 | 0.50 | 1.57 | 0.297 | 0.970 | 5076 |

## M5 delay diagnostics

{'median_delay_m5': 5.0, 'p90_delay_m5': 11.0, 'invalidate_rate': 0.0249489349285089}

## Gate notes

- OOS years M5 PF > prod: 4/5
- Passes validation gates — not PROMOTE until live paper test

Production unchanged.

