# Walk-Forward Validation Report — LightGBM v3 (classic + SMC + reversal)

## Setup
- Features: 39 (= 14 classic + 14 SMC + 11 reversal)
- Embargo: +/-8h; expanding train; val 2021-2025; sealed test >=2026
- Breakeven: 0.4286; min_signals=200

## SHORT (primary focus)
| fold | n_train | n_val | auc_roc | auc_pr | fold_selected_thr | fold_val_precision | fold_val_n |
| --- | --- | --- | --- | --- | --- | --- | --- |
| fold1 | 21556 | 3489 | 0.4736 | 0.4018 | nan | nan | nan |
| fold2 | 25053 | 3554 | 0.5282 | 0.4170 | 0.5100 | 0.4330 | 545.0000 |
| fold3 | 28612 | 3544 | 0.5469 | 0.4193 | nan | nan | nan |
| fold4 | 32156 | 3760 | 0.4925 | 0.3540 | nan | nan | nan |
| fold5 | 35916 | 3888 | 0.5471 | 0.4082 | 0.5400 | 0.4762 | 210.0000 |

- Mean fold AUC-ROC: **0.5177** (std=0.0332)
- v2 mean fold AUC-ROC: 0.5394
- Delta vs v2: -0.0217
- Majority threshold: no threshold above BE with min_signals>=200 in majority of folds (3/5)

### Threshold majority summary (SHORT)
| threshold | n_folds_above_be | majority | mean_precision_when_ok | mean_n_signals |
| --- | --- | --- | --- | --- |
| 0.5000 | 1 | False | 0.4292 | 2251.6000 |
| 0.5100 | 1 | False | 0.4330 | 598.8000 |
| 0.5200 | 0 | False | nan | 226.0000 |
| 0.5300 | 0 | False | nan | 108.8000 |
| 0.5400 | 1 | False | 0.4762 | 42.0000 |
| 0.5500 | 0 | False | nan | 26.8000 |
| 0.5600 | 0 | False | nan | 18.4000 |
| 0.5700 | 0 | False | nan | 8.8000 |
| 0.5800 | 0 | False | nan | 2.8000 |
| 0.5900 | 0 | False | nan | 0.0000 |
| 0.6000 | 0 | False | nan | 0.0000 |

### SHORT validity vs v1/v2
- **NO** — still no majority-valid threshold (same failure mode as v1/v2 SHORT).

### Feature importance top 15 (SHORT)
| feature | importance_gain |
| --- | --- |
| ema_slope_d1 | 139.15 |
| dist_to_bearish_ob_h1 | 128.99 |
| dist_to_last_swing_low_h1 | 102.45 |
| adx_h1 | 101.75 |
| ema_fast_h1 | 67.92 |
| session | 66.85 |
| dist_from_ema_d1_zscore | 65.24 |
| atr_h4 | 52.20 |
| atr_h1 | 46.31 |
| ema_slope_h4 | 46.25 |
| ema_slow_h1 | 46.00 |
| adx_h4 | 35.25 |
| liquidity_sweep_low_h1 | 22.75 |
| dist_to_last_swing_high_h1 | 21.42 |
| rsi_extreme_duration_h1 | 20.97 |
- Reversal features in top 15: ['dist_from_ema_d1_zscore', 'liquidity_sweep_low_h1', 'rsi_extreme_duration_h1']
- SMC features in top 15: ['dist_to_bearish_ob_h1', 'dist_to_last_swing_low_h1', 'dist_to_last_swing_high_h1']

## LONG (sanity check — should not collapse)
| fold | n_train | n_val | auc_roc | auc_pr | fold_selected_thr | fold_val_precision | fold_val_n |
| --- | --- | --- | --- | --- | --- | --- | --- |
| fold1 | 21516 | 3457 | 0.5212 | 0.3872 | 0.5100 | 0.4777 | 291 |
| fold2 | 24982 | 3586 | 0.5455 | 0.4218 | 0.5000 | 0.4322 | 1564 |
| fold3 | 28573 | 3486 | 0.5400 | 0.4120 | 0.5100 | 0.4382 | 744 |
| fold4 | 32059 | 3655 | 0.5524 | 0.4717 | 0.5500 | 0.5349 | 215 |
| fold5 | 35714 | 3799 | 0.5498 | 0.4746 | 0.5400 | 0.5637 | 204 |

- Mean fold AUC-ROC: **0.5418** (std=0.0124)
- v2 mean fold AUC-ROC: 0.5385
- Delta vs v2: +0.0033
- Majority threshold: selected_thr=0.51 (above BE in 4/5 folds, mean_prec=0.4678)

### Threshold majority summary (LONG)
| threshold | n_folds_above_be | majority | mean_precision_when_ok | mean_n_signals |
| --- | --- | --- | --- | --- |
| 0.5000 | 3 | True | 0.4577 | 1522.4000 |
| 0.5100 | 4 | True | 0.4678 | 891.6000 |
| 0.5200 | 2 | False | 0.4837 | 597.0000 |
| 0.5300 | 2 | False | 0.4898 | 412.4000 |
| 0.5400 | 2 | False | 0.5227 | 254.0000 |
| 0.5500 | 1 | False | 0.5349 | 150.0000 |
| 0.5600 | 0 | False | nan | 90.8000 |
| 0.5700 | 0 | False | nan | 40.6000 |
| 0.5800 | 0 | False | nan | 17.6000 |
| 0.5900 | 0 | False | nan | 7.8000 |
| 0.6000 | 0 | False | nan | 3.4000 |

### Sealed test 2026 (LONG)
- n_test=1923; AUC=0.5453
- @thr=0.51: precision=0.4457, n=368, ABOVE breakeven

### Feature importance top 15 (LONG)
| feature | importance_gain |
| --- | --- |
| ema_slope_d1 | 6729.19 |
| atr_h4 | 3662.56 |
| ema_slope_h4 | 3610.92 |
| adx_h4 | 3595.08 |
| ema_slow_h1 | 3587.29 |
| ema_fast_h1 | 3504.29 |
| adx_h1 | 3166.82 |
| atr_h1 | 2556.91 |
| dist_from_ema_d1_zscore | 1682.92 |
| dist_from_ema_h4_zscore | 1599.35 |
| dist_to_last_swing_low_h1 | 1358.21 |
| rsi_h1 | 1343.07 |
| dist_to_nearest_fvg_bullish | 1062.40 |
| dist_to_bearish_ob_h1 | 1058.08 |
| dist_to_bullish_ob_h1 | 973.38 |
- Reversal features in top 15: ['dist_from_ema_d1_zscore', 'dist_from_ema_h4_zscore']

## Artifacts
- `data\models_v3\long\lightgbm_long_v3.pkl`
- `data\models_v3\long\lightgbm_long_v3_test_preds.parquet`
- `data\models_v3\short\lightgbm_short_v3.pkl`
- `data\models_v3\short\lightgbm_short_v3_test_preds.parquet`
