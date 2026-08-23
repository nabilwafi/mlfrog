# H1 + M5 Pullback Entry × Trail Exit (M15 vs M5 clock)

**Verdict:** `ITERATE`

**Target setup (M15 path):**
- Entry: M5 `pullback_recovery` fill (after H1 close)
- Exit: P0 a0.25/d0.08 trail looping each **M15 close**
- Trail starts from first M15 bar at/after M5 fill (causal, after H1 close)
- Horizon: 48 H1 bars from H1 close signal

Production ref: `prod_v5_feat7_top21_a025_d008_lot01_daily80_m15trail`

## OOS comparison

| policy | entry | exit_tf | pf | dd | avg_r | trades | mc_prob_ruin |
|---|---|---|---|---|---|---|---|
| production_feat7_m15 | immediate | M15 | 1.33 | 0.538 | 0.059 | 5335 | 0.003 |
| h1_immediate_m15 | immediate | M15 | 1.40 | 0.302 | 0.061 | 5358 | 0.002 |
| h1_pullback_m15_proper | pullback_recovery | M15 | 1.53 | 0.273 | 0.076 | 4992 | 0.000 |
| h1_pullback_m5 | pullback_recovery | M5 | 1.21 | 0.815 | 0.017 | 4888 | 0.013 |

## Key delta: same pullback entry, exit clock

- Pullback + **M15 close** trail (proper): PF 1.53, DD 27.3%, avgR 0.076
- Pullback + **M5 close** trail: PF 1.21, DD 81.5%, avgR 0.017
- ΔPF (M5−M15 exit clock): -0.32

Production unchanged.
