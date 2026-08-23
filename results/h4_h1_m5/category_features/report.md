# Feature Category Comparison

Stack: M15 pullback entry + P0 M15 trail. Baseline = H1 native 6.

**New features added:** 13 (return_1_atr, return_5_atr, return_20_atr, volume_zscore, volume_ratio, spread_zscore, spread_ratio, spread_bps, price_zscore, bb_position, roc_3, roc_12, momentum_acceleration)

**Unavailable (no data):** order_flow, cross_asset, funding_derivatives

## OOS results

| policy | n_features | pf | dd | avg_r | trades | mc_prob_ruin |
|---|---|---|---|---|---|---|
| h1_native_6 | 6 | 1.60 | 0.271 | 0.077 | 4988 | 0.000 |
| h1_plus_new_all | 19 | 1.52 | 0.262 | 0.077 | 4946 | 0.000 |
| h1_plus_price_return | 9 | 1.47 | 0.369 | 0.074 | 4935 | 0.001 |
| h1_plus_volume | 8 | 1.42 | 0.423 | 0.068 | 4969 | 0.002 |
| h1_plus_liquidity | 9 | 1.51 | 0.295 | 0.087 | 4969 | 0.006 |
| h1_plus_mean_reversion | 8 | 1.56 | 0.356 | 0.076 | 5047 | 0.001 |
| h1_plus_momentum_new | 9 | 1.56 | 0.350 | 0.081 | 4925 | 0.002 |
| h1_plus_momentum_lib | 11 | 1.45 | 0.364 | 0.080 | 4923 | 0.001 |
| h1_plus_candle_lib | 12 | 1.48 | 0.479 | 0.074 | 4974 | 0.000 |
| engine_v1_35 | 35 | 1.53 | 0.461 | 0.084 | 4888 | 0.003 |
| engine_ext_48 | 48 | 1.42 | 0.666 | 0.070 | 4975 | 0.007 |

## vs baseline (h1_native_6)

- `h1_plus_new_all`: ΔPF -0.08, ΔDD -0.9pp
- `h1_plus_price_return`: ΔPF -0.13, ΔDD +9.9pp
- `h1_plus_volume`: ΔPF -0.18, ΔDD +15.2pp
- `h1_plus_liquidity`: ΔPF -0.09, ΔDD +2.4pp
- `h1_plus_mean_reversion`: ΔPF -0.04, ΔDD +8.5pp
- `h1_plus_momentum_new`: ΔPF -0.04, ΔDD +7.9pp
- `h1_plus_momentum_lib`: ΔPF -0.15, ΔDD +9.4pp
- `h1_plus_candle_lib`: ΔPF -0.12, ΔDD +20.9pp
- `engine_v1_35`: ΔPF -0.07, ΔDD +19.0pp
- `engine_ext_48`: ΔPF -0.18, ΔDD +39.5pp

## Engine comparison

- v1 engine (35): PF 1.53, DD 46.1%
- extended (48): PF 1.42, DD 66.6%
- ΔPF ext−v1: -0.11

Production unchanged.
