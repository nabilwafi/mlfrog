# H1 Feature Audit — Hierarchical MTF Sprint

Classification for H4 → H1 signal-contract architecture.

| Feature | Class | Notes |
|---|---|---|
| `atr_percent` | **KEEP** | H1-native; Experiment A baseline |
| `atr_percentile_252` | **KEEP** | H1-native; Experiment A baseline |
| `ema_trend_duration` | **KEEP** | H1-native; Experiment A baseline |
| `hour_cos` | **KEEP** | H1-native; Experiment A baseline |
| `hour_sin` | **KEEP** | H1-native; Experiment A baseline |
| `rolling_quantile` | **KEEP** | H1-native; Experiment A baseline |
| `ctx_h4_swing_quality` | **REPLACE** | Raw H4 merge → use H4 signal contract in Experiment B |
| `ctx_h4_compression` | **EXPERIMENTAL** | In v2 dataset; not in FEAT7; ablation only |
| `ctx_h4_expansion` | **EXPERIMENTAL** | In v2 dataset; not in FEAT7; ablation only |
| `ctx_h4_market_regime` | **EXPERIMENTAL** | In v2 dataset; not in FEAT7; ablation only |
| `ctx_h4_trend_direction` | **EXPERIMENTAL** | In v2 dataset; not in FEAT7; ablation only |
| `ctx_h4_trend_strength` | **EXPERIMENTAL** | In v2 dataset; not in FEAT7; ablation only |
| `ctx_h4_volatility_regime` | **EXPERIMENTAL** | In v2 dataset; not in FEAT7; ablation only |

## H4 signal contract columns (Experiment B)

`h4_sig_direction`, `h4_sig_trend`, `h4_sig_structure`, `h4_sig_vol`, `h4_sig_strength`

These replace raw `ctx_h4_*` in the hierarchical design.

## Production reference

Frozen FEAT7: `hour_cos, atr_percentile_252, ema_trend_duration, rolling_quantile, hour_sin, atr_percent, ctx_h4_swing_quality`

Pipeline: `prod_v5_feat7_top21_a025_d008_lot01_daily80_m15trail`

