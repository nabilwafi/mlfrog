# D1 vs H4 Context + M15 Pullback Entry

**Verdict:** `H1_ONLY_BEST`

Stack: context (optional) → H1 ML → **M15 pullback entry** → P0 **M15 trail** exit.
D1 bars resampled from H1; same signal contract as H4 (`d1_sig_*`).

Ref: `prod_v5_feat7_top21_a025_d008_lot01_daily80_m15trail`

## OOS

| context | pf | dd | avg_r | trades | retention | mc_prob_ruin |
|---|---|---|---|---|---|---|
| h1_only | 1.60 | 0.271 | 0.077 | 4988 | 0.973 | 0.000 |
| h4_context | 1.53 | 0.513 | 0.079 | 4945 | 0.973 | 0.000 |
| d1_context | 1.41 | 0.296 | 0.062 | 5009 | 0.972 | 0.001 |

## vs H1-only baseline

- H4 context: ΔPF -0.07, ΔDD +24.2pp
- D1 context: ΔPF -0.19, ΔDD +2.6pp

Production unchanged.
