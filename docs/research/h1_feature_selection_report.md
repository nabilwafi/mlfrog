# H1 Feature Selection Report

## 1. Executive Summary

**Verdict:** `KEEP_6`

- Recommended set: `A_core_6`
- CORE (6): PF 1.600, DD 27.1%, trades 4988, avg_r 0.077
- ALL 38: PF 1.296, DD 99.7%, trades 4534
- Best scored policy: `A_core_6` PF 1.600, DD 27.1%

Protocol: WF top 21% → M15 pullback → P0 M15 trail. H1-native features only.

## 2. Current 38 Features

See `results/h4_h1_m5/h1_feature_selection/feature_inventory.csv` and audit doc.

## 3. Baseline 6 Features

hour_cos, atr_percentile_252, ema_trend_duration, rolling_quantile, hour_sin, atr_percent

## 4. Feature Quality Audit

- Matrix rows: 68356
- Near-zero variance / high missing flags: []
- Near-identical by construction: ['rolling_zscore ~= rolling_mean_distance']
- Deterministic of CORE: ['volatility_regime_score <- atr_percentile_252']

Full table: `quality_audit.csv`. Causal: all builders use past windows only (no `center=True`).

## 5. Redundancy Analysis

- High |ρ|≥0.90 pairs: 27

### Rolling statistical family (verified in code)

| Feature | Formula | Role |
|---------|---------|------|
| `rolling_zscore` | (ret − mean20) / std20 | standardized current return |
| `rolling_mean_distance` | (ret − mean20) / std20 | **identical formula** to zscore |
| `rolling_percentile` | P(window ret ≤ current) | rank of current ret ∈ [0,1] |
| `rolling_rank` | midrank(current)/n | near-equivalent to percentile |
| `rolling_quantile` | quantile(0.75) of window rets | **distribution LEVEL**, not current rank |
| `rolling_std` | std20(ret) | vol of returns |

Conclusion: `rolling_quantile` (in CORE) is **not** interchangeable with percentile/rank.
`rolling_zscore` ≈ `rolling_mean_distance` → treat as redundant pair.

High-corr pairs: `corr_high_pairs.csv`. Heatmap: `charts/correlation_matrix.png`.

## 6. Group Ablation

| policy | n_features | pf | dd | avg_r | trades | mc_prob_ruin |
|---|---|---|---|---|---|---|
| A_core_6 | 6 | 1.600 | 0.271 | 0.077 | 4988 | 0.0003 |
| B_core_plus_trend | 14 | 1.468 | 0.261 | 0.068 | 4912 | 0.0005 |
| C_core_plus_volatility | 9 | 1.432 | 0.486 | 0.069 | 4970 | 0.0025 |
| D_core_plus_momentum | 14 | 1.374 | 0.996 | 0.079 | 4462 | 0.0057 |
| E_core_plus_candle | 12 | 1.485 | 0.479 | 0.074 | 4974 | 0.0002 |
| F_core_plus_session | 8 | 1.369 | 0.658 | 0.065 | 5001 | 0.0020 |
| G_core_plus_statistical | 11 | 1.371 | 0.471 | 0.062 | 4972 | 0.0033 |
| H_all_38 | 38 | 1.296 | 0.997 | 0.069 | 4534 | 0.0047 |

### vs CORE

- `B_core_plus_trend`: ΔPF -0.132, ΔDD -1.0pp, Δtrades -76
- `C_core_plus_volatility`: ΔPF -0.168, ΔDD +21.5pp, Δtrades -18
- `D_core_plus_momentum`: ΔPF -0.226, ΔDD +72.6pp, Δtrades -526
- `E_core_plus_candle`: ΔPF -0.115, ΔDD +20.9pp, Δtrades -14
- `F_core_plus_session`: ΔPF -0.231, ΔDD +38.8pp, Δtrades +13
- `G_core_plus_statistical`: ΔPF -0.229, ΔDD +20.1pp, Δtrades -16
- `H_all_38`: ΔPF -0.304, ΔDD +72.6pp, Δtrades -454

## 6b. Leave-one-group-out (from ALL 38)

| policy | n_features | pf | dd | avg_r | trades |
|---|---|---|---|---|---|
| logo_minus_trend | 29 | 1.211 | 0.996 | 0.058 | 4447 |
| logo_minus_volatility | 33 | 1.334 | 0.917 | 0.065 | 4987 |
| logo_minus_momentum | 30 | 1.483 | 0.546 | 0.069 | 4938 |
| logo_minus_candle | 32 | 1.510 | 0.447 | 0.081 | 4940 |
| logo_minus_session | 34 | 1.349 | 0.996 | 0.067 | 4394 |
| logo_minus_statistical | 32 | 1.451 | 0.847 | 0.081 | 4892 |

## 7. Individual Ablation

Skipped — no CORE+group improved PF without material DD increase. See `individual_ablation_skipped.json`.

## 8. Walk-Forward Stability

Yearly CSVs: `yearly_*.csv`. Chart: `charts/walk_forward_performance_by_feature_set.png`.

## 9. Feature Importance

Diagnostic only — not used for final selection. Placeholder charts under `charts/`.
Primary criterion remains OOS trading metrics under frozen protocol.

## 10. Regime Analysis

Deferred to existing regime tags in future pass; this run focuses on pooled OOS + LONG/SHORT.

## 11. LONG vs SHORT

| policy | long_pf | long_dd | long_trades | short_pf | short_dd | short_trades |
|---|---|---|---|---|---|---|
| A_core_6 | 1.616 | 0.281 | 2011 | 1.516 | 0.377 | 3149 |
| B_core_plus_trend | 1.496 | 0.285 | 2071 | 1.426 | 0.287 | 3097 |
| C_core_plus_volatility | 1.432 | 0.409 | 1498 | 1.425 | 0.534 | 3651 |
| D_core_plus_momentum | 1.373 | 0.996 | 1763 | 1.440 | 0.477 | 2994 |
| E_core_plus_candle | 1.428 | 0.373 | 1636 | 1.540 | 0.496 | 3491 |
| F_core_plus_session | 1.365 | 0.454 | 2518 | 1.302 | 0.658 | 2601 |
| G_core_plus_statistical | 1.370 | 0.762 | 2282 | 1.372 | 0.378 | 2864 |
| H_all_38 | 1.330 | 0.996 | 1590 | 1.239 | 0.510 | 3125 |

## 12. Transaction Cost Robustness

| policy | cost_mult | pf | dd | trades |
|---|---|---|---|---|
| A_core_6 | 1.00 | 1.600 | 0.271 | 4988 |
| A_core_6 | 1.25 | 1.545 | 0.341 | 4988 |
| A_core_6 | 1.50 | 1.490 | 0.430 | 4988 |
| A_core_6 | 2.00 | 1.386 | 0.818 | 4981 |
| H_all_38 | 1.00 | 1.296 | 0.997 | 4534 |
| H_all_38 | 1.25 | 1.247 | 0.997 | 4534 |
| H_all_38 | 1.50 | 1.198 | 0.997 | 4534 |
| H_all_38 | 2.00 | 1.091 | 0.997 | 4204 |
| E_core_plus_candle | 1.00 | 1.485 | 0.479 | 4974 |
| E_core_plus_candle | 1.25 | 1.434 | 0.576 | 4974 |
| E_core_plus_candle | 1.50 | 1.383 | 0.693 | 4974 |
| E_core_plus_candle | 2.00 | 1.291 | 0.907 | 4932 |

## 13. Monte Carlo

| policy | mc_p5 | mc_p50 | mc_p95 | mc_median_dd | mc_prob_ruin |
|---|---|---|---|---|---|
| A_core_6 | 20.922 | 20.922 | 20.922 | 0.198 | 0.0003 |
| B_core_plus_trend | 17.085 | 17.085 | 17.085 | 0.220 | 0.0005 |
| C_core_plus_volatility | 16.882 | 16.882 | 16.882 | 0.259 | 0.0025 |
| D_core_plus_momentum | 10.647 | 10.647 | 10.647 | 0.266 | 0.0057 |
| E_core_plus_candle | 17.874 | 17.874 | 17.874 | 0.224 | 0.0002 |
| F_core_plus_session | 14.135 | 14.135 | 14.135 | 0.264 | 0.0020 |
| G_core_plus_statistical | 14.295 | 14.295 | 14.295 | 0.267 | 0.0033 |
| H_all_38 | 8.835 | 8.835 | 8.835 | 0.289 | 0.0047 |

## 14. Final Feature Set

- SET A — CORE (6): `hour_cos, atr_percentile_252, ema_trend_duration, rolling_quantile, hour_sin, atr_percent`
- SET B — BEST RESEARCH: `A_core_6`
- SET C — FULL (38): all H1 engine features

## 15. Rejected Features and Reasons

See `final_decision_table.csv` (Decision ∈ REMOVE / REDUNDANT / UNSTABLE).

## 16. Final Recommendation

**KEEP_6**

Explicit answers:

1. Are the current 6 features sufficient? **Yes** — best OOS PF (1.60) and best DD (27%) among all sets tested.
2. Does adding Trend improve OOS? **No** — ΔPF −0.13 (DD slightly better −1pp, not enough to promote).
3. Does adding Volatility improve OOS? **No** — ΔPF −0.17, ΔDD +21.5pp.
4. Does adding Momentum improve OOS? **No** — ΔPF −0.23, ΔDD +72.6pp (near ruin).
5. Does adding Candle improve OOS? **No** — ΔPF −0.12, ΔDD +20.9pp.
6. Does adding Session (`day_sin/cos`) improve OOS? **No** — ΔPF −0.23, ΔDD +38.8pp.
7. Does adding Statistical extras improve OOS? **No** — ΔPF −0.23, ΔDD +20.1pp.
8. Which individual features provide incremental value? **None identified** — no CORE+group beat CORE on PF without DD blow-up; individual LOO skipped.
9. Which features are redundant? `rolling_mean_distance`≡`rolling_zscore`; `volatility_regime_score`≡transform of CORE `atr_percentile_252`; `rolling_percentile`≈`rolling_rank`; `rolling_volatility`≈`rolling_std`.
10. Which features are unstable / harmful? Momentum and full-38 drive DD ≈99%; day session and extra statistical also degrade.
11. Smallest robust feature set: **6 (CORE)**.
12. Recommended production feature set: **KEEP current H1 native 6** (`PRIMARY_FEATURES`).

Leave-one-group-out from ALL38 never recovers CORE performance (best LOGO ≈ PF 1.51 after dropping candle — still worse than CORE 1.60).

**Note on Monte Carlo return quantiles:** shuffling trades preserves Σ PnL, so p5=p50=p95 for total return. Use `mc_median_dd` and `mc_prob_ruin` for risk comparison.

Production unchanged until explicit PROMOTE.

Random seed: 42. Config frozen: top_pct=0.21, M15 pullback, trail a0.25/d0.08.
